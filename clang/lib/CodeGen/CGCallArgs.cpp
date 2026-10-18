//===--- CGCallArgs.cpp - Evaluate and prepare call arguments ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These classes wrap the information about a call or function
// definition used to handle ABI compliancy.
//
//===----------------------------------------------------------------------===//

#include "CGCall.h"
#include "CGCallLowering.h"
#include "ABIInfo.h"
#include "ABIInfoImpl.h"
#include "CGBlocks.h"
#include "CGCXXABI.h"
#include "CGCleanup.h"
#include "CGDebugInfo.h"
#include "CGRecordLayout.h"
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "CodeGenPGO.h"
#include "QualTypeMapper.h"
#include "TargetInfo.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/RecordLayout.h"
#include "clang/Basic/CodeGenOptions.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CodeGen/CGFunctionInfo.h"
#include "clang/CodeGen/SwiftCallingConv.h"
#include "llvm/ABI/FunctionInfo.h"
#include "llvm/ABI/IRTypeMapper.h"
#include "llvm/ABI/TargetInfo.h"
#include "llvm/ABI/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Assumptions.h"
#include "llvm/IR/AttributeMask.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Type.h"
#include "llvm/Transforms/Utils/Local.h"
#include <optional>
using namespace clang;
using namespace CodeGen;

static bool isInAllocaArgument(CGCXXABI &ABI, QualType type) {
  const CXXRecordDecl *RD = type->getAsCXXRecordDecl();
  return RD && ABI.getRecordArgABI(RD) == CGCXXABI::RAA_DirectInMemory;
}

static AggValueSlot createPlaceholderSlot(CodeGenFunction &CGF, QualType Ty) {
  // FIXME: Generate IR in one pass, rather than going back and fixing up these
  // placeholders.
  llvm::Type *IRTy = CGF.ConvertTypeForMem(Ty);
  llvm::Type *IRPtrTy = llvm::PointerType::getUnqual(CGF.getLLVMContext());
  llvm::Value *Placeholder = llvm::PoisonValue::get(IRPtrTy);

  // FIXME: When we generate this IR in one pass, we shouldn't need
  // this win32-specific alignment hack.
  CharUnits Align = CharUnits::fromQuantity(4);
  Placeholder = CGF.Builder.CreateAlignedLoad(IRPtrTy, Placeholder, Align);

  return AggValueSlot::forAddr(
      Address(Placeholder, IRTy, Align), Ty.getQualifiers(),
      AggValueSlot::IsNotDestructed, AggValueSlot::DoesNotNeedGCBarriers,
      AggValueSlot::IsNotAliased, AggValueSlot::DoesNotOverlap);
}

void CodeGenFunction::EmitDelegateCallArg(CallArgList &args,
                                          const VarDecl *param,
                                          SourceLocation loc) {
  // StartFunction converted the ABI-lowered parameter(s) into a
  // local alloca.  We need to turn that into an r-value suitable
  // for EmitCall.
  Address local = GetAddrOfLocalVar(param);

  QualType type = param->getType();

  // GetAddrOfLocalVar returns a pointer-to-pointer for references,
  // but the argument needs to be the original pointer.
  if (type->isReferenceType()) {
    args.add(RValue::get(Builder.CreateLoad(local)), type);

  // In ARC, move out of consumed arguments so that the release cleanup
  // entered by StartFunction doesn't cause an over-release.  This isn't
  // optimal -O0 code generation, but it should get cleaned up when
  // optimization is enabled.  This also assumes that delegate calls are
  // performed exactly once for a set of arguments, but that should be safe.
  } else if (getLangOpts().ObjCAutoRefCount &&
             param->hasAttr<NSConsumedAttr>() && type->isObjCRetainableType()) {
    llvm::Value *ptr = Builder.CreateLoad(local);
    auto null =
        llvm::ConstantPointerNull::get(cast<llvm::PointerType>(ptr->getType()));
    Builder.CreateStore(null, local);
    args.add(RValue::get(ptr), type);

  // For the most part, we just need to load the alloca, except that
  // aggregate r-values are actually pointers to temporaries.
  } else {
    args.add(convertTempToRValue(local, type, loc), type);
  }

  // Deactivate the cleanup for the callee-destructed param that was pushed.
  if (type->isRecordType() && !CurFuncIsThunk &&
      type->castAsRecordDecl()->isParamDestroyedInCallee() &&
      param->needsDestruction(getContext())) {
    EHScopeStack::stable_iterator cleanup =
        CalleeDestructedParamCleanups.lookup(cast<ParmVarDecl>(param));
    assert(cleanup.isValid() &&
           "cleanup for callee-destructed param not recorded");
    // This unreachable is a temporary marker which will be removed later.
    llvm::Instruction *isActive = Builder.CreateUnreachable();
    args.addArgCleanupDeactivation(cleanup, isActive);
  }
}

static bool isProvablyNull(llvm::Value *addr) {
  return llvm::isa_and_nonnull<llvm::ConstantPointerNull>(addr);
}

static bool isProvablyNonNull(Address Addr, CodeGenFunction &CGF) {
  return llvm::isKnownNonZero(Addr.getBasePointer(), CGF.CGM.getDataLayout());
}

/// Emit the actual writing-back of a writeback.
static void emitWriteback(CodeGenFunction &CGF,
                          const CallArgList::Writeback &writeback) {
  const LValue &srcLV = writeback.Source;
  Address srcAddr = srcLV.getAddress();
  assert(!isProvablyNull(srcAddr.getBasePointer()) &&
         "shouldn't have writeback for provably null argument");

  if (writeback.WritebackExpr) {
    CGF.EmitIgnoredExpr(writeback.WritebackExpr);
    CGF.EmitLifetimeEnd(writeback.Temporary.getBasePointer());
    return;
  }

  llvm::BasicBlock *contBB = nullptr;

  // If the argument wasn't provably non-null, we need to null check
  // before doing the store.
  bool provablyNonNull = isProvablyNonNull(srcAddr, CGF);

  if (!provablyNonNull) {
    llvm::BasicBlock *writebackBB = CGF.createBasicBlock("icr.writeback");
    contBB = CGF.createBasicBlock("icr.done");

    llvm::Value *isNull = CGF.Builder.CreateIsNull(srcAddr, "icr.isnull");
    CGF.Builder.CreateCondBr(isNull, contBB, writebackBB);
    CGF.EmitBlock(writebackBB);
  }

  // Load the value to writeback.
  llvm::Value *value = CGF.Builder.CreateLoad(writeback.Temporary);

  // Cast it back, in case we're writing an id to a Foo* or something.
  value = CGF.Builder.CreateBitCast(value, srcAddr.getElementType(),
                                    "icr.writeback-cast");

  // Perform the writeback.

  // If we have a "to use" value, it's something we need to emit a use
  // of.  This has to be carefully threaded in: if it's done after the
  // release it's potentially undefined behavior (and the optimizer
  // will ignore it), and if it happens before the retain then the
  // optimizer could move the release there.
  if (writeback.ToUse) {
    assert(srcLV.getObjCLifetime() == Qualifiers::OCL_Strong);

    // Retain the new value.  No need to block-copy here:  the block's
    // being passed up the stack.
    value = CGF.EmitARCRetainNonBlock(value);

    // Emit the intrinsic use here.
    CGF.EmitARCIntrinsicUse(writeback.ToUse);

    // Load the old value (primitively).
    llvm::Value *oldValue = CGF.EmitLoadOfScalar(srcLV, SourceLocation());

    // Put the new value in place (primitively).
    CGF.EmitStoreOfScalar(value, srcLV, /*init*/ false);

    // Release the old value.
    CGF.EmitARCRelease(oldValue, srcLV.isARCPreciseLifetime());

  // Otherwise, we can just do a normal lvalue store.
  } else {
    CGF.EmitStoreThroughLValue(RValue::get(value), srcLV);
  }

  // Jump to the continuation block.
  if (!provablyNonNull)
    CGF.EmitBlock(contBB);
}

void CodeGen::deactivateArgCleanupsBeforeCall(CodeGenFunction &CGF,
                                            const CallArgList &CallArgs) {
  ArrayRef<CallArgList::CallArgCleanup> Cleanups =
      CallArgs.getCleanupsToDeactivate();
  // Iterate in reverse to increase the likelihood of popping the cleanup.
  for (const auto &I : llvm::reverse(Cleanups)) {
    CGF.DeactivateCleanupBlock(I.Cleanup, I.IsActiveIP);
    I.IsActiveIP->eraseFromParent();
  }
}

static const Expr *maybeGetUnaryAddrOfOperand(const Expr *E) {
  if (const UnaryOperator *uop = dyn_cast<UnaryOperator>(E->IgnoreParens()))
    if (uop->getOpcode() == UO_AddrOf)
      return uop->getSubExpr();
  return nullptr;
}

/// Emit an argument that's being passed call-by-writeback.  That is,
/// we are passing the address of an __autoreleased temporary; it
/// might be copy-initialized with the current value of the given
/// address, but it will definitely be copied out of after the call.
static void emitWritebackArg(CodeGenFunction &CGF, CallArgList &args,
                             const ObjCIndirectCopyRestoreExpr *CRE) {
  LValue srcLV;

  // Make an optimistic effort to emit the address as an l-value.
  // This can fail if the argument expression is more complicated.
  if (const Expr *lvExpr = maybeGetUnaryAddrOfOperand(CRE->getSubExpr())) {
    srcLV = CGF.EmitLValue(lvExpr);

  // Otherwise, just emit it as a scalar.
  } else {
    Address srcAddr = CGF.EmitPointerWithAlignment(CRE->getSubExpr());

    QualType srcAddrType =
        CRE->getSubExpr()->getType()->castAs<PointerType>()->getPointeeType();
    srcLV = CGF.MakeAddrLValue(srcAddr, srcAddrType);
  }
  Address srcAddr = srcLV.getAddress();

  // The dest and src types don't necessarily match in LLVM terms
  // because of the crazy ObjC compatibility rules.

  llvm::PointerType *destType =
      cast<llvm::PointerType>(CGF.ConvertType(CRE->getType()));
  llvm::Type *destElemType =
      CGF.ConvertTypeForMem(CRE->getType()->getPointeeType());

  // If the address is a constant null, just pass the appropriate null.
  if (isProvablyNull(srcAddr.getBasePointer())) {
    args.add(RValue::get(llvm::ConstantPointerNull::get(destType)),
             CRE->getType());
    return;
  }

  // Create the temporary.
  Address temp =
      CGF.CreateTempAlloca(destElemType, CGF.getPointerAlign(), "icr.temp");
  // Loading an l-value can introduce a cleanup if the l-value is __weak,
  // and that cleanup will be conditional if we can't prove that the l-value
  // isn't null, so we need to register a dominating point so that the cleanups
  // system will make valid IR.
  CodeGenFunction::ConditionalEvaluation condEval(CGF);

  // Zero-initialize it if we're not doing a copy-initialization.
  bool shouldCopy = CRE->shouldCopy();
  if (!shouldCopy) {
    llvm::Value *null =
        llvm::ConstantPointerNull::get(cast<llvm::PointerType>(destElemType));
    CGF.Builder.CreateStore(null, temp);
  }

  llvm::BasicBlock *contBB = nullptr;
  llvm::BasicBlock *originBB = nullptr;

  // If the address is *not* known to be non-null, we need to switch.
  llvm::Value *finalArgument;

  bool provablyNonNull = isProvablyNonNull(srcAddr, CGF);

  if (provablyNonNull) {
    finalArgument = temp.emitRawPointer(CGF);
  } else {
    llvm::Value *isNull = CGF.Builder.CreateIsNull(srcAddr, "icr.isnull");

    finalArgument = CGF.Builder.CreateSelect(
        isNull, llvm::ConstantPointerNull::get(destType),
        temp.emitRawPointer(CGF), "icr.argument");

    // If we need to copy, then the load has to be conditional, which
    // means we need control flow.
    if (shouldCopy) {
      originBB = CGF.Builder.GetInsertBlock();
      contBB = CGF.createBasicBlock("icr.cont");
      llvm::BasicBlock *copyBB = CGF.createBasicBlock("icr.copy");
      CGF.Builder.CreateCondBr(isNull, contBB, copyBB);
      CGF.EmitBlock(copyBB);
      condEval.begin(CGF);
    }
  }

  llvm::Value *valueToUse = nullptr;

  // Perform a copy if necessary.
  if (shouldCopy) {
    RValue srcRV = CGF.EmitLoadOfLValue(srcLV, SourceLocation());
    assert(srcRV.isScalar());

    llvm::Value *src = srcRV.getScalarVal();
    src = CGF.Builder.CreateBitCast(src, destElemType, "icr.cast");

    // Use an ordinary store, not a store-to-lvalue.
    CGF.Builder.CreateStore(src, temp);

    // If optimization is enabled, and the value was held in a
    // __strong variable, we need to tell the optimizer that this
    // value has to stay alive until we're doing the store back.
    // This is because the temporary is effectively unretained,
    // and so otherwise we can violate the high-level semantics.
    if (CGF.CGM.getCodeGenOpts().OptimizationLevel != 0 &&
        srcLV.getObjCLifetime() == Qualifiers::OCL_Strong) {
      valueToUse = src;
    }
  }

  // Finish the control flow if we needed it.
  if (shouldCopy && !provablyNonNull) {
    llvm::BasicBlock *copyBB = CGF.Builder.GetInsertBlock();
    CGF.EmitBlock(contBB);

    // Make a phi for the value to intrinsically use.
    if (valueToUse) {
      llvm::PHINode *phiToUse =
          CGF.Builder.CreatePHI(valueToUse->getType(), 2, "icr.to-use");
      phiToUse->addIncoming(valueToUse, copyBB);
      phiToUse->addIncoming(llvm::PoisonValue::get(valueToUse->getType()),
                            originBB);
      valueToUse = phiToUse;
    }

    condEval.end(CGF);
  }

  args.addWriteback(srcLV, temp, valueToUse);
  args.add(RValue::get(finalArgument), CRE->getType());
}

void CallArgList::allocateArgumentMemory(CodeGenFunction &CGF) {
  assert(!StackBase);

  // Save the stack.
  StackBase = CGF.Builder.CreateStackSave("inalloca.save");
}

void CallArgList::freeArgumentMemory(CodeGenFunction &CGF) const {
  if (StackBase) {
    // Restore the stack after the call.
    CGF.Builder.CreateStackRestore(StackBase);
  }
}

void CodeGenFunction::EmitNonNullArgCheck(RValue RV, QualType ArgType,
                                          SourceLocation ArgLoc,
                                          AbstractCallee AC, unsigned ParmNum) {
  if (!AC.getDecl() || !(SanOpts.has(SanitizerKind::NonnullAttribute) ||
                         SanOpts.has(SanitizerKind::NullabilityArg)))
    return;

  // The param decl may be missing in a variadic function.
  auto PVD = ParmNum < AC.getNumParams() ? AC.getParamDecl(ParmNum) : nullptr;
  unsigned ArgNo = PVD ? PVD->getFunctionScopeIndex() : ParmNum;

  // Prefer the nonnull attribute if it's present.
  const NonNullAttr *NNAttr = nullptr;
  if (SanOpts.has(SanitizerKind::NonnullAttribute))
    NNAttr = getNonNullAttr(AC.getDecl(), PVD, ArgType, ArgNo);

  bool CanCheckNullability = false;
  if (SanOpts.has(SanitizerKind::NullabilityArg) && !NNAttr && PVD &&
      !PVD->getType()->isRecordType()) {
    auto Nullability = PVD->getType()->getNullability();
    CanCheckNullability = Nullability &&
                          *Nullability == NullabilityKind::NonNull &&
                          PVD->getTypeSourceInfo();
  }

  if (!NNAttr && !CanCheckNullability)
    return;

  SourceLocation AttrLoc;
  SanitizerKind::SanitizerOrdinal CheckKind;
  SanitizerHandler Handler;
  if (NNAttr) {
    AttrLoc = NNAttr->getLocation();
    CheckKind = SanitizerKind::SO_NonnullAttribute;
    Handler = SanitizerHandler::NonnullArg;
  } else {
    AttrLoc = PVD->getTypeSourceInfo()->getTypeLoc().findNullabilityLoc();
    CheckKind = SanitizerKind::SO_NullabilityArg;
    Handler = SanitizerHandler::NullabilityArg;
  }

  SanitizerDebugLocation SanScope(this, {CheckKind}, Handler);
  llvm::Value *Cond = EmitNonNullRValueCheck(RV, ArgType);
  llvm::Constant *StaticData[] = {
      EmitCheckSourceLocation(ArgLoc),
      EmitCheckSourceLocation(AttrLoc),
      llvm::ConstantInt::get(Int32Ty, ArgNo + 1),
  };
  EmitCheck(std::make_pair(Cond, CheckKind), Handler, StaticData, {});
}

void CodeGenFunction::EmitNonNullArgCheck(Address Addr, QualType ArgType,
                                          SourceLocation ArgLoc,
                                          AbstractCallee AC, unsigned ParmNum) {
  if (!AC.getDecl() || !(SanOpts.has(SanitizerKind::NonnullAttribute) ||
                         SanOpts.has(SanitizerKind::NullabilityArg)))
    return;

  EmitNonNullArgCheck(RValue::get(Addr, *this), ArgType, ArgLoc, AC, ParmNum);
}

// Check if the call is going to use the inalloca convention. This needs to
// agree with CGFunctionInfo::usesInAlloca. The CGFunctionInfo is arranged
// later, so we can't check it directly.
static bool hasInAllocaArgs(CodeGenModule &CGM, CallingConv ExplicitCC,
                            ArrayRef<QualType> ArgTypes) {
  // The Swift calling conventions don't go through the target-specific
  // argument classification, they never use inalloca.
  // TODO: Consider limiting inalloca use to only calling conventions supported
  // by MSVC.
  if (ExplicitCC == CC_Swift || ExplicitCC == CC_SwiftAsync)
    return false;
  if (!CGM.getTarget().getCXXABI().isMicrosoft())
    return false;
  return llvm::any_of(ArgTypes, [&](QualType Ty) {
    return isInAllocaArgument(CGM.getCXXABI(), Ty);
  });
}

#ifndef NDEBUG
// Determine whether the given argument is an Objective-C method
// that may have type parameters in its signature.
static bool isObjCMethodWithTypeParams(const ObjCMethodDecl *method) {
  const DeclContext *dc = method->getDeclContext();
  if (const ObjCInterfaceDecl *classDecl = dyn_cast<ObjCInterfaceDecl>(dc)) {
    return classDecl->getTypeParamListAsWritten();
  }

  if (const ObjCCategoryDecl *catDecl = dyn_cast<ObjCCategoryDecl>(dc)) {
    return catDecl->getTypeParamList();
  }

  return false;
}
#endif

/// EmitCallArgs - Emit call arguments for a function.
void CodeGenFunction::EmitCallArgs(
    CallArgList &Args, PrototypeWrapper Prototype,
    llvm::iterator_range<CallExpr::const_arg_iterator> ArgRange,
    AbstractCallee AC, unsigned ParamsToSkip, EvaluationOrder Order) {
  SmallVector<QualType, 16> ArgTypes;

  assert((ParamsToSkip == 0 || Prototype.P) &&
         "Can't skip parameters if type info is not provided");

  // This variable only captures *explicitly* written conventions, not those
  // applied by default via command line flags or target defaults, such as
  // thiscall, aapcs, stdcall via -mrtd, etc. Computing that correctly would
  // require knowing if this is a C++ instance method or being able to see
  // unprototyped FunctionTypes.
  CallingConv ExplicitCC = CC_C;

  // First, if a prototype was provided, use those argument types.
  bool IsVariadic = false;
  if (Prototype.P) {
    const auto *MD = dyn_cast<const ObjCMethodDecl *>(Prototype.P);
    if (MD) {
      IsVariadic = MD->isVariadic();
      ExplicitCC = getCallingConventionForDecl(
          MD, CGM.getTarget().getTriple().isOSWindows());
      ArgTypes.assign(MD->param_type_begin() + ParamsToSkip,
                      MD->param_type_end());
    } else {
      const auto *FPT = cast<const FunctionProtoType *>(Prototype.P);
      IsVariadic = FPT->isVariadic();
      ExplicitCC = FPT->getExtInfo().getCC();
      ArgTypes.assign(FPT->param_type_begin() + ParamsToSkip,
                      FPT->param_type_end());
    }

#ifndef NDEBUG
    // Check that the prototyped types match the argument expression types.
    bool isGenericMethod = MD && isObjCMethodWithTypeParams(MD);
    CallExpr::const_arg_iterator Arg = ArgRange.begin();
    for (QualType Ty : ArgTypes) {
      assert(Arg != ArgRange.end() && "Running over edge of argument list!");
      QualType ParamTy = Ty.getNonReferenceType();
      QualType ArgTy = (*Arg)->getType();
      if (const auto *OBT = ParamTy->getAs<OverflowBehaviorType>())
        ParamTy = OBT->getUnderlyingType();
      if (const auto *OBT = ArgTy->getAs<OverflowBehaviorType>())
        ArgTy = OBT->getUnderlyingType();
      assert((isGenericMethod || Ty->isVariablyModifiedType() ||
              ParamTy->isObjCRetainableType() ||
              getContext().getCanonicalType(ParamTy).getTypePtr() ==
                  getContext().getCanonicalType(ArgTy).getTypePtr()) &&
             "type mismatch in call argument!");
      ++Arg;
    }

    // Either we've emitted all the call args, or we have a call to variadic
    // function.
    assert((Arg == ArgRange.end() || IsVariadic) &&
           "Extra arguments in non-variadic function!");
#endif
  }

  // If we still have any arguments, emit them using the type of the argument.
  for (auto *A : llvm::drop_begin(ArgRange, ArgTypes.size()))
    ArgTypes.push_back(IsVariadic ? getVarArgType(A) : A->getType());
  assert((int)ArgTypes.size() == (ArgRange.end() - ArgRange.begin()));

  // We must evaluate arguments from right to left in the MS C++ ABI,
  // because arguments are destroyed left to right in the callee. As a special
  // case, there are certain language constructs that require left-to-right
  // evaluation, and in those cases we consider the evaluation order requirement
  // to trump the "destruction order is reverse construction order" guarantee.
  bool LeftToRight =
      CGM.getTarget().getCXXABI().areArgsDestroyedLeftToRightInCallee()
          ? Order == EvaluationOrder::ForceLeftToRight
          : Order != EvaluationOrder::ForceRightToLeft;

  auto MaybeEmitImplicitObjectSize = [&](unsigned I, const Expr *Arg,
                                         RValue EmittedArg) {
    if (!AC.hasFunctionDecl() || I >= AC.getNumParams())
      return;
    auto *PS = AC.getParamDecl(I)->getAttr<PassObjectSizeAttr>();
    if (PS == nullptr)
      return;

    const auto &Context = getContext();
    auto SizeTy = Context.getSizeType();
    auto T = Builder.getIntNTy(Context.getTypeSize(SizeTy));
    assert(EmittedArg.getScalarVal() && "We emitted nothing for the arg?");
    llvm::Value *V = evaluateOrEmitBuiltinObjectSize(
        Arg, PS->getType(), T, EmittedArg.getScalarVal(), PS->isDynamic());
    Args.add(RValue::get(V), SizeTy);
    // If we're emitting args in reverse, be sure to do so with
    // pass_object_size, as well.
    if (!LeftToRight)
      std::swap(Args.back(), *(&Args.back() - 1));
  };

  // Insert a stack save if we're going to need any inalloca args.
  if (hasInAllocaArgs(CGM, ExplicitCC, ArgTypes)) {
    assert(getTarget().getTriple().getArch() == llvm::Triple::x86 &&
           "inalloca only supported on x86");
    Args.allocateArgumentMemory(*this);
  }

  // Evaluate each argument in the appropriate order.
  size_t CallArgsStart = Args.size();
  for (unsigned I = 0, E = ArgTypes.size(); I != E; ++I) {
    unsigned Idx = LeftToRight ? I : E - I - 1;
    CallExpr::const_arg_iterator Arg = ArgRange.begin() + Idx;
    unsigned InitialArgSize = Args.size();
    // If *Arg is an ObjCIndirectCopyRestoreExpr, check that either the types of
    // the argument and parameter match or the objc method is parameterized.
    assert((!isa<ObjCIndirectCopyRestoreExpr>(*Arg) ||
            getContext().hasSameUnqualifiedType((*Arg)->getType(),
                                                ArgTypes[Idx]) ||
            (isa<ObjCMethodDecl>(AC.getDecl()) &&
             isObjCMethodWithTypeParams(cast<ObjCMethodDecl>(AC.getDecl())))) &&
           "Argument and parameter types don't match");
    EmitCallArg(Args, *Arg, ArgTypes[Idx]);
    // In particular, we depend on it being the last arg in Args, and the
    // objectsize bits depend on there only being one arg if !LeftToRight.
    assert(InitialArgSize + 1 == Args.size() &&
           "The code below depends on only adding one arg per EmitCallArg");
    (void)InitialArgSize;
    // Since pointer argument are never emitted as LValue, it is safe to emit
    // non-null argument check for r-value only.
    if (!Args.back().hasLValue()) {
      RValue RVArg = Args.back().getKnownRValue();
      EmitNonNullArgCheck(RVArg, ArgTypes[Idx], (*Arg)->getExprLoc(), AC,
                          ParamsToSkip + Idx);
      // @llvm.objectsize should never have side-effects and shouldn't need
      // destruction/cleanups, so we can safely "emit" it after its arg,
      // regardless of right-to-leftness
      MaybeEmitImplicitObjectSize(Idx, *Arg, RVArg);
    }
  }

  if (!LeftToRight) {
    // Un-reverse the arguments we just evaluated so they match up with the LLVM
    // IR function.
    std::reverse(Args.begin() + CallArgsStart, Args.end());

    // Reverse the writebacks to match the MSVC ABI.
    Args.reverseWritebacks();
  }
}

namespace {

struct DestroyUnpassedArg final : EHScopeStack::Cleanup {
  DestroyUnpassedArg(Address Addr, QualType Ty) : Addr(Addr), Ty(Ty) {}

  Address Addr;
  QualType Ty;

  void Emit(CodeGenFunction &CGF, Flags flags) override {
    QualType::DestructionKind DtorKind = Ty.isDestructedType();
    if (DtorKind == QualType::DK_cxx_destructor) {
      const CXXDestructorDecl *Dtor = Ty->getAsCXXRecordDecl()->getDestructor();
      assert(!Dtor->isTrivial());
      CGF.EmitCXXDestructorCall(Dtor, Dtor_Complete, /*for vbase*/ false,
                                /*Delegating=*/false, Addr, Ty);
    } else {
      CGF.callCStructDestructor(CGF.MakeAddrLValue(Addr, Ty));
    }
  }
};

} // end anonymous namespace

RValue CallArg::getRValue(CodeGenFunction &CGF) const {
  if (!HasLV)
    return RV;
  LValue Copy = CGF.MakeAddrLValue(CGF.CreateMemTempWithoutCast(Ty), Ty);
  CGF.EmitAggregateCopy(Copy, LV, Ty, AggValueSlot::DoesNotOverlap,
                        LV.isVolatile());
  IsUsed = true;
  return RValue::getAggregate(Copy.getAddress());
}

void CallArg::copyInto(CodeGenFunction &CGF, Address Addr) const {
  LValue Dst = CGF.MakeAddrLValue(Addr, Ty);
  if (!HasLV && RV.isScalar())
    CGF.EmitStoreOfScalar(RV.getScalarVal(), Dst, /*isInit=*/true);
  else if (!HasLV && RV.isComplex())
    CGF.EmitStoreOfComplex(RV.getComplexVal(), Dst, /*init=*/true);
  else {
    auto Addr = HasLV ? LV.getAddress() : RV.getAggregateAddress();
    LValue SrcLV = CGF.MakeAddrLValue(Addr, Ty);
    // We assume that call args are never copied into subobjects.
    CGF.EmitAggregateCopy(Dst, SrcLV, Ty, AggValueSlot::DoesNotOverlap,
                          HasLV ? LV.isVolatileQualified()
                                : RV.isVolatileQualified());
  }
  IsUsed = true;
}

void CodeGenFunction::EmitWritebacks(const CallArgList &args) {
  for (const auto &I : args.writebacks())
    emitWriteback(*this, I);
}

void CodeGenFunction::EmitCallArg(CallArgList &args, const Expr *E,
                                  QualType type) {
  std::optional<DisableDebugLocationUpdates> Dis;
  if (isa<CXXDefaultArgExpr>(E))
    Dis.emplace(*this);
  if (const ObjCIndirectCopyRestoreExpr *CRE =
          dyn_cast<ObjCIndirectCopyRestoreExpr>(E)) {
    assert(getLangOpts().ObjCAutoRefCount);
    return emitWritebackArg(*this, args, CRE);
  }

  // Add writeback for HLSLOutParamExpr.
  // Needs to be before the assert below because HLSLOutArgExpr is an LValue
  // and is not a reference.
  if (const HLSLOutArgExpr *OE = dyn_cast<HLSLOutArgExpr>(E)) {
    EmitHLSLOutArgExpr(OE, args, type);
    return;
  }

  assert(type->isReferenceType() == E->isGLValue() &&
         "reference binding to unmaterialized r-value!");

  if (E->isGLValue()) {
    assert(E->getObjectKind() == OK_Ordinary);
    return args.add(EmitReferenceBindingToExpr(E), type);
  }

  bool HasAggregateEvalKind = hasAggregateEvaluationKind(type);

  // In the Microsoft C++ ABI, aggregate arguments are destructed by the callee.
  // However, we still have to push an EH-only cleanup in case we unwind before
  // we make it to the call.
  if (type->isRecordType() &&
      type->castAsRecordDecl()->isParamDestroyedInCallee()) {
    // If we're using inalloca, use the argument memory.  Otherwise, use a
    // temporary.
    AggValueSlot Slot = args.isUsingInAlloca()
                            ? createPlaceholderSlot(*this, type)
                            : CreateAggTemp(type, "agg.tmp");

    bool DestroyedInCallee = true, NeedsCleanup = true;
    if (const auto *RD = type->getAsCXXRecordDecl())
      DestroyedInCallee = RD->hasNonTrivialDestructor();
    else
      NeedsCleanup = type.isDestructedType();

    if (DestroyedInCallee)
      Slot.setExternallyDestructed();

    EmitAggExpr(E, Slot);
    RValue RV = Slot.asRValue();
    args.add(RV, type);

    if (DestroyedInCallee && NeedsCleanup) {
      // Create a no-op GEP between the placeholder and the cleanup so we can
      // RAUW it successfully.  It also serves as a marker of the first
      // instruction where the cleanup is active.
      pushFullExprCleanup<DestroyUnpassedArg>(NormalAndEHCleanup,
                                              Slot.getAddress(), type);
      // This unreachable is a temporary marker which will be removed later.
      llvm::Instruction *IsActive =
          Builder.CreateFlagLoad(llvm::Constant::getNullValue(Int8PtrTy));
      args.addArgCleanupDeactivation(EHStack.stable_begin(), IsActive);
    }
    return;
  }

  if (HasAggregateEvalKind) {
    auto *ICE = dyn_cast<ImplicitCastExpr>(E);
    if (ICE && ICE->getCastKind() == CK_LValueToRValue &&
        ICE->getSubExpr()->getType().getAddressSpace() !=
            LangAS::hlsl_constant &&
        !type->isArrayParameterType() && !type.isNonTrivialToPrimitiveCopy()) {
      LValue L = EmitLValue(cast<CastExpr>(E)->getSubExpr());
      assert(L.isSimple());
      args.addUncopiedAggregate(L, type);
      return;
    }
  }

  args.add(EmitAnyExprToTemp(E), type);
}

QualType CodeGenFunction::getVarArgType(const Expr *Arg) {
  // System headers on Windows define NULL to 0 instead of 0LL on Win64. MSVC
  // implicitly widens null pointer constants that are arguments to varargs
  // functions to pointer-sized ints.
  if (!getTarget().getTriple().isOSWindows())
    return Arg->getType();

  if (Arg->getType()->isIntegerType() &&
      getContext().getTypeSize(Arg->getType()) <
          getContext().getTargetInfo().getPointerWidth(LangAS::Default) &&
      Arg->isNullPointerConstant(getContext(),
                                 Expr::NPC_ValueDependentIsNotNull)) {
    return getContext().getIntPtrType();
  }

  return Arg->getType();
}

// In ObjC ARC mode with no ObjC ARC exception safety, tell the ARC
// optimizer it can aggressively ignore unwind edges.
