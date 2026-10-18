//===--- CGCall.cpp - Emit calls and runtime call operations ---===//
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
#include "ABIInfo.h"
#include "ABIInfoImpl.h"
#include "CGBlocks.h"
#include "CGCXXABI.h"
#include "CGCallLowering.h"
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
#include "clang/CodeGen/SourceVariadicArguments.h"
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

void CodeGenFunction::AddObjCARCExceptionMetadata(llvm::Instruction *Inst) {
  if (CGM.getCodeGenOpts().OptimizationLevel != 0 &&
      !CGM.getCodeGenOpts().ObjCAutoRefCountExceptions)
    Inst->setMetadata("clang.arc.no_objc_arc_exceptions",
                      CGM.getNoObjCARCExceptionsMetadata());
}

/// Emits a call to the given no-arguments nounwind runtime function.
llvm::CallInst *
CodeGenFunction::EmitNounwindRuntimeCall(llvm::FunctionCallee callee,
                                         const llvm::Twine &name) {
  return EmitNounwindRuntimeCall(callee, ArrayRef<llvm::Value *>(), name);
}

/// Emits a call to the given nounwind runtime function.
llvm::CallInst *
CodeGenFunction::EmitNounwindRuntimeCall(llvm::FunctionCallee callee,
                                         ArrayRef<Address> args,
                                         const llvm::Twine &name) {
  SmallVector<llvm::Value *, 3> values;
  for (auto arg : args)
    values.push_back(arg.emitRawPointer(*this));
  return EmitNounwindRuntimeCall(callee, values, name);
}

llvm::CallInst *
CodeGenFunction::EmitNounwindRuntimeCall(llvm::FunctionCallee callee,
                                         ArrayRef<llvm::Value *> args,
                                         const llvm::Twine &name) {
  llvm::CallInst *call = EmitRuntimeCall(callee, args, name);
  call->setDoesNotThrow();
  return call;
}

/// Emits a simple call (never an invoke) to the given no-arguments
/// runtime function.
llvm::CallInst *CodeGenFunction::EmitRuntimeCall(llvm::FunctionCallee callee,
                                                 const llvm::Twine &name) {
  return EmitRuntimeCall(callee, {}, name);
}

// Calls which may throw must have operand bundles indicating which funclet
// they are nested within.
SmallVector<llvm::OperandBundleDef, 1>
CodeGenFunction::getBundlesForFunclet(llvm::Value *Callee) {
  // There is no need for a funclet operand bundle if we aren't inside a
  // funclet.
  if (!CurrentFuncletPad)
    return (SmallVector<llvm::OperandBundleDef, 1>());

  // Skip intrinsics which cannot throw (as long as they don't lower into
  // regular function calls in the course of IR transformations).
  if (auto *CalleeFn = dyn_cast<llvm::Function>(Callee->stripPointerCasts())) {
    if (CalleeFn->isIntrinsic() && CalleeFn->doesNotThrow()) {
      auto IID = CalleeFn->getIntrinsicID();
      if (!llvm::IntrinsicInst::mayLowerToFunctionCall(IID))
        return (SmallVector<llvm::OperandBundleDef, 1>());
    }
  }

  SmallVector<llvm::OperandBundleDef, 1> BundleList;
  BundleList.emplace_back("funclet", CurrentFuncletPad);
  return BundleList;
}

/// Emits a simple call (never an invoke) to the given runtime function.
llvm::CallInst *CodeGenFunction::EmitRuntimeCall(llvm::FunctionCallee callee,
                                                 ArrayRef<llvm::Value *> args,
                                                 const llvm::Twine &name) {
  llvm::CallInst *call = Builder.CreateCall(
      callee, args, getBundlesForFunclet(callee.getCallee()), name);
  call->setCallingConv(getRuntimeCC());

  if (CGM.shouldEmitConvergenceTokens() && call->isConvergent())
    return cast<llvm::CallInst>(addConvergenceControlToken(call));
  return call;
}

llvm::CallInst *CodeGenFunction::EmitIntrinsicCall(llvm::Intrinsic::ID ID,
                                                   const llvm::Twine &Name) {
  return EmitIntrinsicCall(ID, {}, {}, Name);
}

llvm::CallInst *CodeGenFunction::EmitIntrinsicCall(llvm::Intrinsic::ID ID,
                                                   ArrayRef<llvm::Value *> Args,
                                                   const llvm::Twine &Name) {
  return EmitIntrinsicCall(ID, {}, Args, Name);
}

llvm::CallInst *CodeGenFunction::EmitIntrinsicCall(llvm::Intrinsic::ID ID,
                                                   ArrayRef<llvm::Type *> Types,
                                                   ArrayRef<llvm::Value *> Args,
                                                   const llvm::Twine &Name) {
  llvm::Function *F =
      llvm::Intrinsic::getOrInsertDeclaration(&CGM.getModule(), ID, Types);
  llvm::CallInst *Call =
      Builder.CreateCall(F, Args, getBundlesForFunclet(F), Name);
  if (CGM.shouldEmitConvergenceTokens() && Call->isConvergent())
    return cast<llvm::CallInst>(addConvergenceControlToken(Call));
  return Call;
}

llvm::CallInst *CodeGenFunction::EmitIntrinsicCall(llvm::Intrinsic::ID ID,
                                                   ArrayRef<llvm::Value *> Args,
                                                   llvm::Type *RetTy,
                                                   const llvm::Twine &Name) {
  SmallVector<llvm::Type *> ArgTys;
  ArgTys.reserve(Args.size());
  for (llvm::Value *Arg : Args)
    ArgTys.push_back(Arg->getType());
  llvm::Function *F = llvm::Intrinsic::getOrInsertDeclaration(
      &CGM.getModule(), ID, RetTy, ArgTys);
  llvm::CallInst *Call =
      Builder.CreateCall(F, Args, getBundlesForFunclet(F), Name);
  if (CGM.shouldEmitConvergenceTokens() && Call->isConvergent())
    return cast<llvm::CallInst>(addConvergenceControlToken(Call));
  return Call;
}

/// Emits a call or invoke to the given noreturn runtime function.
void CodeGenFunction::EmitNoreturnRuntimeCallOrInvoke(
    llvm::FunctionCallee callee, ArrayRef<llvm::Value *> args) {
  SmallVector<llvm::OperandBundleDef, 1> BundleList =
      getBundlesForFunclet(callee.getCallee());

  if (getInvokeDest()) {
    llvm::InvokeInst *invoke = Builder.CreateInvoke(
        callee, getUnreachableBlock(), getInvokeDest(), args, BundleList);
    invoke->setDoesNotReturn();
    invoke->setCallingConv(getRuntimeCC());
  } else {
    llvm::CallInst *call = Builder.CreateCall(callee, args, BundleList);
    call->setDoesNotReturn();
    call->setCallingConv(getRuntimeCC());
    Builder.CreateUnreachable();
  }
}

/// Emits a call or invoke instruction to the given nullary runtime function.
llvm::CallBase *
CodeGenFunction::EmitRuntimeCallOrInvoke(llvm::FunctionCallee callee,
                                         const Twine &name) {
  return EmitRuntimeCallOrInvoke(callee, {}, name);
}

/// Emits a call or invoke instruction to the given runtime function.
llvm::CallBase *
CodeGenFunction::EmitRuntimeCallOrInvoke(llvm::FunctionCallee callee,
                                         ArrayRef<llvm::Value *> args,
                                         const Twine &name) {
  llvm::CallBase *call = EmitCallOrInvoke(callee, args, name);
  call->setCallingConv(getRuntimeCC());
  return call;
}

/// Emits a call or invoke instruction to the given function, depending
/// on the current state of the EH stack.
llvm::CallBase *CodeGenFunction::EmitCallOrInvoke(llvm::FunctionCallee Callee,
                                                  ArrayRef<llvm::Value *> Args,
                                                  const Twine &Name) {
  llvm::BasicBlock *InvokeDest = getInvokeDest();
  SmallVector<llvm::OperandBundleDef, 1> BundleList =
      getBundlesForFunclet(Callee.getCallee());

  llvm::CallBase *Inst;
  if (!InvokeDest)
    Inst = Builder.CreateCall(Callee, Args, BundleList, Name);
  else {
    llvm::BasicBlock *ContBB = createBasicBlock("invoke.cont");
    Inst = Builder.CreateInvoke(Callee, ContBB, InvokeDest, Args, BundleList,
                                Name);
    EmitBlock(ContBB);
  }

  // In ObjC ARC mode with no ObjC ARC exception safety, tell the ARC
  // optimizer it can aggressively ignore unwind edges.
  if (CGM.getLangOpts().ObjCAutoRefCount)
    AddObjCARCExceptionMetadata(Inst);

  return Inst;
}

void CodeGenFunction::deferPlaceholderReplacement(llvm::Instruction *Old,
                                                  llvm::Value *New) {
  DeferredReplacements.push_back(
      std::make_pair(llvm::WeakTrackingVH(Old), New));
}

namespace {

/// Specify given \p NewAlign as the alignment of return value attribute. If
/// such attribute already exists, re-set it to the maximal one of two options.
[[nodiscard]] llvm::AttributeList
maybeRaiseRetAlignmentAttribute(llvm::LLVMContext &Ctx,
                                const llvm::AttributeList &Attrs,
                                llvm::Align NewAlign) {
  llvm::Align CurAlign = Attrs.getRetAlignment().valueOrOne();
  if (CurAlign >= NewAlign)
    return Attrs;
  llvm::Attribute AlignAttr = llvm::Attribute::getWithAlignment(Ctx, NewAlign);
  return Attrs.removeRetAttribute(Ctx, llvm::Attribute::AttrKind::Alignment)
      .addRetAttribute(Ctx, AlignAttr);
}

template <typename AlignedAttrTy> class AbstractAssumeAlignedAttrEmitter {
protected:
  CodeGenFunction &CGF;

  /// We do nothing if this is, or becomes, nullptr.
  const AlignedAttrTy *AA = nullptr;

  llvm::Value *Alignment = nullptr;      // May or may not be a constant.
  llvm::ConstantInt *OffsetCI = nullptr; // Constant, hopefully zero.

  AbstractAssumeAlignedAttrEmitter(CodeGenFunction &CGF_, const Decl *FuncDecl)
      : CGF(CGF_) {
    if (!FuncDecl)
      return;
    AA = FuncDecl->getAttr<AlignedAttrTy>();
  }

public:
  /// If we can, materialize the alignment as an attribute on return value.
  [[nodiscard]] llvm::AttributeList
  TryEmitAsCallSiteAttribute(const llvm::AttributeList &Attrs) {
    if (!AA || OffsetCI || CGF.SanOpts.has(SanitizerKind::Alignment))
      return Attrs;
    const auto *AlignmentCI = dyn_cast<llvm::ConstantInt>(Alignment);
    if (!AlignmentCI)
      return Attrs;
    // We may legitimately have non-power-of-2 alignment here.
    // If so, this is UB land, emit it via `@llvm.assume` instead.
    if (!AlignmentCI->getValue().isPowerOf2())
      return Attrs;
    llvm::AttributeList NewAttrs = maybeRaiseRetAlignmentAttribute(
        CGF.getLLVMContext(), Attrs,
        llvm::Align(
            AlignmentCI->getLimitedValue(llvm::Value::MaximumAlignment)));
    AA = nullptr; // We're done. Disallow doing anything else.
    return NewAttrs;
  }

  /// Emit alignment assumption.
  /// This is a general fallback that we take if either there is an offset,
  /// or the alignment is variable or we are sanitizing for alignment.
  void EmitAsAnAssumption(SourceLocation Loc, QualType RetTy, RValue &Ret) {
    if (!AA)
      return;
    CGF.emitAlignmentAssumption(Ret.getScalarVal(), RetTy, Loc,
                                AA->getLocation(), Alignment, OffsetCI);
    AA = nullptr; // We're done. Disallow doing anything else.
  }
};

/// Helper data structure to emit `AssumeAlignedAttr`.
class AssumeAlignedAttrEmitter final
    : public AbstractAssumeAlignedAttrEmitter<AssumeAlignedAttr> {
public:
  AssumeAlignedAttrEmitter(CodeGenFunction &CGF_, const Decl *FuncDecl)
      : AbstractAssumeAlignedAttrEmitter(CGF_, FuncDecl) {
    if (!AA)
      return;
    // It is guaranteed that the alignment/offset are constants.
    Alignment = cast<llvm::ConstantInt>(CGF.EmitScalarExpr(AA->getAlignment()));
    if (Expr *Offset = AA->getOffset()) {
      OffsetCI = cast<llvm::ConstantInt>(CGF.EmitScalarExpr(Offset));
      if (OffsetCI->isNullValue()) // Canonicalize zero offset to no offset.
        OffsetCI = nullptr;
    }
  }
};

/// Helper data structure to emit `AllocAlignAttr`.
class AllocAlignAttrEmitter final
    : public AbstractAssumeAlignedAttrEmitter<AllocAlignAttr> {
public:
  AllocAlignAttrEmitter(CodeGenFunction &CGF_, const Decl *FuncDecl,
                        const CallArgList &CallArgs)
      : AbstractAssumeAlignedAttrEmitter(CGF_, FuncDecl) {
    if (!AA)
      return;
    // Alignment may or may not be a constant, and that is okay.
    Alignment = CallArgs[AA->getParamIndex().getLLVMIndex()]
                    .getRValue(CGF)
                    .getScalarVal();
  }
};

} // namespace

static unsigned getMaxVectorWidth(const llvm::Type *Ty) {
  if (auto *VT = dyn_cast<llvm::VectorType>(Ty))
    return VT->getPrimitiveSizeInBits().getKnownMinValue();
  if (auto *AT = dyn_cast<llvm::ArrayType>(Ty))
    return getMaxVectorWidth(AT->getElementType());

  unsigned MaxVectorWidth = 0;
  if (auto *ST = dyn_cast<llvm::StructType>(Ty))
    for (auto *I : ST->elements())
      MaxVectorWidth = std::max(MaxVectorWidth, getMaxVectorWidth(I));
  return MaxVectorWidth;
}

static void appendSourceVariadicValue(CodeGenFunction &CGF, QualType Ty,
                                      RValue Value,
                                      SmallVectorImpl<llvm::Value *> &Values) {
  using SourceVariadicArguments::Kind;
  Kind ValueKind = Value.isAggregate() ? Kind::Aggregate
                   : Value.isComplex() ? Kind::Complex
                                       : Kind::Scalar;
  if (ValueKind == Kind::Aggregate && !Value.getAggregateAddress().isValid())
    ValueKind = Kind::Ignored;
  Values.push_back(CGF.Builder.getInt8(static_cast<uint8_t>(ValueKind)));
  Values.push_back(CGF.Builder.getInt64(
      CGF.getContext().getTypeSizeInChars(Ty).getQuantity()));
  switch (ValueKind) {
  case Kind::Scalar:
    Values.push_back(Value.getScalarVal());
    break;
  case Kind::Complex:
    Values.push_back(Value.getComplexVal().first);
    Values.push_back(Value.getComplexVal().second);
    break;
  case Kind::Aggregate:
    Values.push_back(Value.getAggregateAddress().emitRawPointer(CGF));
    break;
  case Kind::Ignored:
    break;
  }
}

RValue CodeGenFunction::EmitCall(const CGFunctionInfo &CallInfo,
                                 const CGCallee &Callee,
                                 ReturnValueSlot ReturnValue,
                                 const CallArgList &CallArgs,
                                 llvm::CallBase **callOrInvoke, bool IsMustTail,
                                 SourceLocation Loc,
                                 bool IsVirtualFunctionPointerThunk) {
  // FIXME: We no longer need the types from CallArgs; lift up and simplify.

  assert(Callee.isOrdinary() || Callee.isVirtual());

  SmallVector<llvm::Value *, 8> SourceVariadicValues;
  if (CGM.getCodeGenOpts().PreserveSourceVariadicArguments &&
      CallInfo.isVariadic()) {
    for (const CallArg &Arg :
         llvm::drop_begin(CallArgs, CallInfo.getNumRequiredArgs())) {
      RValue Value =
          Arg.hasLValue()
              ? RValue::getAggregate(Arg.getKnownLValue().getAddress(),
                                     Arg.getKnownLValue().isVolatileQualified())
              : Arg.getKnownRValue();
      appendSourceVariadicValue(*this, Arg.getType(), Value,
                                SourceVariadicValues);
    }
  }

  // Handle struct-return functions by passing a pointer to the
  // location that we would like to return into.
  QualType RetTy = CallInfo.getReturnType();
  const ABIArgInfo &RetAI = CallInfo.getReturnInfo();

  llvm::FunctionType *IRFuncTy = getTypes().GetFunctionType(CallInfo);

  const Decl *TargetDecl = Callee.getAbstractInfo().getCalleeDecl().getDecl();
  if (const FunctionDecl *FD = dyn_cast_or_null<FunctionDecl>(TargetDecl)) {
    // We can only guarantee that a function is called from the correct
    // context/function based on the appropriate target attributes,
    // so only check in the case where we have both always_inline and target
    // since otherwise we could be making a conditional call after a check for
    // the proper cpu features (and it won't cause code generation issues due to
    // function based code generation).
    if ((TargetDecl->hasAttr<AlwaysInlineAttr>() &&
         (TargetDecl->hasAttr<TargetAttr>() ||
          (CurFuncDecl && CurFuncDecl->hasAttr<TargetAttr>()))) ||
        (CurFuncDecl && CurFuncDecl->hasAttr<FlattenAttr>() &&
         (CurFuncDecl->hasAttr<TargetAttr>() ||
          TargetDecl->hasAttr<TargetAttr>())))
      checkTargetFeatures(Loc, FD);
  }

  // Some architectures (such as x86-64) have the ABI changed based on
  // attribute-target/features. Give them a chance to diagnose.
  const FunctionDecl *CallerDecl = dyn_cast_or_null<FunctionDecl>(CurCodeDecl);
  const FunctionDecl *CalleeDecl = dyn_cast_or_null<FunctionDecl>(TargetDecl);
  CGM.getTargetCodeGenInfo().checkFunctionCallABI(CGM, Loc, CallerDecl,
                                                  CalleeDecl, CallArgs, RetTy);

  // 1. Set up the arguments.

  // If we're using inalloca, insert the allocation after the stack save.
  // FIXME: Do this earlier rather than hacking it in here!
  RawAddress ArgMemory = RawAddress::invalid();
  if (llvm::StructType *ArgStruct = CallInfo.getArgStruct()) {
    const llvm::DataLayout &DL = CGM.getDataLayout();
    llvm::Instruction *IP = CallArgs.getStackBase();
    llvm::AllocaInst *AI;
    if (IP) {
      IP = IP->getNextNode();
      AI = new llvm::AllocaInst(ArgStruct, DL.getAllocaAddrSpace(), "argmem",
                                IP->getIterator());
    } else {
      AI = CreateTempAlloca(ArgStruct, "argmem");
    }
    auto Align = CallInfo.getArgStructAlignment();
    AI->setAlignment(Align.getAsAlign());
    AI->setUsedWithInAlloca(true);
    assert(AI->isUsedWithInAlloca() && !AI->isStaticAlloca());
    ArgMemory = RawAddress(AI, ArgStruct, Align);
  }

  ClangToLLVMArgMapping IRFunctionArgs(CGM.getContext(), CallInfo);
  SmallVector<llvm::Value *, 16> IRCallArgs(IRFunctionArgs.totalIRArgs());

  // If the call returns a temporary with struct return, create a temporary
  // alloca to hold the result, unless one is given to us.
  Address SRetPtr = Address::invalid();
  // Original alloca for lifetime markers
  Address SRetAlloca = Address::invalid();
  bool NeedSRetLifetimeEnd = false;
  if (RetAI.isIndirect() || RetAI.isInAlloca() || RetAI.isCoerceAndExpand()) {
    // For virtual function pointer thunks and musttail calls, we must always
    // forward an incoming SRet pointer to the callee, because a local alloca
    // would be de-allocated before the call. These cases both guarantee that
    // there will be an incoming SRet argument of the correct type.
    if ((IsVirtualFunctionPointerThunk || IsMustTail) && RetAI.isIndirect()) {
      SRetPtr = makeNaturalAddressForPointer(CurFn->arg_begin() +
                                                 IRFunctionArgs.getSRetArgNo(),
                                             RetTy, CharUnits::fromQuantity(1));
    } else if (!ReturnValue.isNull()) {
      SRetPtr = ReturnValue.getAddress();
    } else {
      SRetPtr = CreateMemTempWithoutCast(RetTy, "tmp");
      if (HaveInsertPoint() && ReturnValue.isUnused()) {
        NeedSRetLifetimeEnd = EmitLifetimeStart(SRetPtr.getBasePointer());
        if (NeedSRetLifetimeEnd)
          SRetAlloca = SRetPtr;
      }
    }
    if (IRFunctionArgs.hasSRetArg()) {
      // A mismatch between the allocated return value's AS and the target's
      // chosen IndirectAS can happen e.g. when passing the this pointer through
      // a chain involving stores to / loads from the DefaultAS; we address this
      // here, symmetrically with the handling we have for normal pointer args.
      if (SRetPtr.getAddressSpace() != RetAI.getIndirectAddrSpace()) {
        llvm::Value *V = SRetPtr.getBasePointer();
        llvm::Type *Ty = llvm::PointerType::get(getLLVMContext(),
                                                RetAI.getIndirectAddrSpace());

        SRetPtr = SRetPtr.withPointer(performAddrSpaceCast(V, Ty),
                                      SRetPtr.isKnownNonNull());
      }
      IRCallArgs[IRFunctionArgs.getSRetArgNo()] =
          getAsNaturalPointerTo(SRetPtr, RetTy);
    } else if (RetAI.isInAlloca()) {
      Address Addr =
          Builder.CreateStructGEP(ArgMemory, RetAI.getInAllocaFieldIndex());
      Builder.CreateStore(getAsNaturalPointerTo(SRetPtr, RetTy), Addr);
    }
  }

  RawAddress swiftErrorTemp = RawAddress::invalid();
  Address swiftErrorArg = Address::invalid();

  // When passing arguments using temporary allocas, we need to add the
  // appropriate lifetime markers. This vector keeps track of all the lifetime
  // markers that need to be ended right after the call.
  SmallVector<CallLifetimeEnd, 2> CallLifetimeEndAfterCall;

  // Translate all of the arguments as necessary to match the IR lowering.
  assert(CallInfo.arg_size() == CallArgs.size() &&
         "Mismatch between function signature & arguments.");
  unsigned ArgNo = 0;
  CGFunctionInfo::const_arg_iterator info_it = CallInfo.arg_begin();
  for (CallArgList::const_iterator I = CallArgs.begin(), E = CallArgs.end();
       I != E; ++I, ++info_it, ++ArgNo) {
    const ABIArgInfo &ArgInfo = info_it->info;

    // Insert a padding argument to ensure proper alignment.
    if (IRFunctionArgs.hasPaddingArg(ArgNo))
      IRCallArgs[IRFunctionArgs.getPaddingArgNo(ArgNo)] =
          llvm::UndefValue::get(ArgInfo.getPaddingType());

    unsigned FirstIRArg, NumIRArgs;
    std::tie(FirstIRArg, NumIRArgs) = IRFunctionArgs.getIRArgs(ArgNo);

    bool ArgHasMaybeUndefAttr =
        IsArgumentMaybeUndef(TargetDecl, CallInfo.getNumRequiredArgs(), ArgNo);

    switch (ArgInfo.getKind()) {
    case ABIArgInfo::InAlloca: {
      assert(NumIRArgs == 0);
      assert(getTarget().getTriple().getArch() == llvm::Triple::x86);
      if (I->isAggregate()) {
        RawAddress Addr = I->hasLValue()
                              ? I->getKnownLValue().getAddress()
                              : I->getKnownRValue().getAggregateAddress();
        llvm::Instruction *Placeholder =
            cast<llvm::Instruction>(Addr.getPointer());

        if (!ArgInfo.getInAllocaIndirect()) {
          // Replace the placeholder with the appropriate argument slot GEP.
          CGBuilderTy::InsertPoint IP = Builder.saveIP();
          Builder.SetInsertPoint(Placeholder);
          Addr = Builder.CreateStructGEP(ArgMemory,
                                         ArgInfo.getInAllocaFieldIndex());
          Builder.restoreIP(IP);
        } else {
          // For indirect things such as overaligned structs, replace the
          // placeholder with a regular aggregate temporary alloca. Store the
          // address of this alloca into the struct.
          Addr =
              CreateMemTempWithoutCast(info_it->type, "inalloca.indirect.tmp");
          Address ArgSlot = Builder.CreateStructGEP(
              ArgMemory, ArgInfo.getInAllocaFieldIndex());
          Builder.CreateStore(Addr.getPointer(), ArgSlot);
        }
        deferPlaceholderReplacement(Placeholder, Addr.getPointer());
      } else if (ArgInfo.getInAllocaIndirect()) {
        // Make a temporary alloca and store the address of it into the argument
        // struct.
        RawAddress Addr = CreateMemTempWithoutCast(
            I->Ty, getContext().getTypeAlignInChars(I->Ty),
            "indirect-arg-temp");
        I->copyInto(*this, Addr);
        Address ArgSlot =
            Builder.CreateStructGEP(ArgMemory, ArgInfo.getInAllocaFieldIndex());
        Builder.CreateStore(Addr.getPointer(), ArgSlot);
      } else {
        // Store the RValue into the argument struct.
        Address Addr =
            Builder.CreateStructGEP(ArgMemory, ArgInfo.getInAllocaFieldIndex());
        Addr = Addr.withElementType(ConvertTypeForMem(I->Ty));
        I->copyInto(*this, Addr);
      }
      break;
    }

    case ABIArgInfo::Indirect:
    case ABIArgInfo::IndirectAliased: {
      assert(NumIRArgs == 1);
      if (I->isAggregate()) {
        // We want to avoid creating an unnecessary temporary+copy here;
        // however, we need one in three cases:
        // 1. If the argument is not byval, and we are required to copy the
        //    source.  (This case doesn't occur on any common architecture.)
        // 2. If the argument is byval, RV is not sufficiently aligned, and
        //    we cannot force it to be sufficiently aligned.
        // 3. If the argument is byval, but RV is not located in default
        //    or alloca address space.
        Address Addr = I->hasLValue()
                           ? I->getKnownLValue().getAddress()
                           : I->getKnownRValue().getAggregateAddress();
        CharUnits Align = ArgInfo.getIndirectAlign();
        const llvm::DataLayout *TD = &CGM.getDataLayout();

        assert((FirstIRArg >= IRFuncTy->getNumParams() ||
                IRFuncTy->getParamType(FirstIRArg)->getPointerAddressSpace() ==
                    TD->getAllocaAddrSpace()) &&
               "indirect argument must be in alloca address space");

        bool NeedCopy = false;
        if (Addr.getAlignment() < Align &&
            llvm::getOrEnforceKnownAlignment(Addr.emitRawPointer(*this),
                                             Align.getAsAlign(),
                                             *TD) < Align.getAsAlign()) {
          NeedCopy = true;
        } else if (I->hasLValue()) {
          auto LV = I->getKnownLValue();

          bool isByValOrRef =
              ArgInfo.isIndirectAliased() || ArgInfo.getIndirectByVal();

          if (!isByValOrRef ||
              (LV.getAlignment() < getContext().getTypeAlignInChars(I->Ty))) {
            NeedCopy = true;
          }

          if (isByValOrRef && Addr.getType()->getAddressSpace() !=
                                  ArgInfo.getIndirectAddrSpace()) {
            NeedCopy = true;
          }
        }

        if (!NeedCopy) {
          // Skip the extra memcpy call.
          llvm::Value *V = getAsNaturalPointerTo(Addr, I->Ty);
          auto *T = llvm::PointerType::get(CGM.getLLVMContext(),
                                           ArgInfo.getIndirectAddrSpace());

          // FIXME: This should not depend on the language address spaces, and
          // only the contextual values. If the address space mismatches, see if
          // we can look through a cast to a compatible address space value,
          // otherwise emit a copy.
          llvm::Value *Val = performAddrSpaceCast(V, T);
          if (ArgHasMaybeUndefAttr)
            Val = Builder.CreateFreeze(Val);
          IRCallArgs[FirstIRArg] = Val;
          break;
        }
      } else if (I->getType()->isArrayParameterType()) {
        // Don't produce a temporary for ArrayParameterType arguments.
        // ArrayParameterType arguments are only created from
        // HLSL_ArrayRValue casts and HLSLOutArgExpr expressions, both
        // of which create temporaries already. This allows us to just use the
        // scalar for the decayed array pointer as the argument directly.
        IRCallArgs[FirstIRArg] = I->getKnownRValue().getScalarVal();
        break;
      }

      // For non-aggregate args and aggregate args meeting conditions above
      // we need to create an aligned temporary, and copy to it.
      RawAddress AI = CreateMemTempWithoutCast(
          I->Ty, ArgInfo.getIndirectAlign(), "byval-temp");
      llvm::Value *Val = getAsNaturalPointerTo(AI, I->Ty);
      if (ArgHasMaybeUndefAttr)
        Val = Builder.CreateFreeze(Val);
      IRCallArgs[FirstIRArg] = Val;

      // Emit lifetime markers for the temporary alloca and add cleanup code to
      // emit the end lifetime marker after the call.
      if (EmitLifetimeStart(AI.getPointer()))
        CallLifetimeEndAfterCall.emplace_back(AI);

      // Generate the copy.
      I->copyInto(*this, AI);
      break;
    }

    case ABIArgInfo::Ignore:
      assert(NumIRArgs == 0);
      break;

    case ABIArgInfo::Extend:
    case ABIArgInfo::Direct: {
      if (!isa<llvm::StructType>(ArgInfo.getCoerceToType()) &&
          ArgInfo.getCoerceToType() == ConvertType(info_it->type) &&
          ArgInfo.getDirectOffset() == 0) {
        assert(NumIRArgs == 1);
        llvm::Value *V;
        if (!I->isAggregate())
          V = I->getKnownRValue().getScalarVal();
        else
          V = Builder.CreateLoad(
              I->hasLValue() ? I->getKnownLValue().getAddress()
                             : I->getKnownRValue().getAggregateAddress());

        // Implement swifterror by copying into a new swifterror argument.
        // We'll write back in the normal path out of the call.
        if (CallInfo.getExtParameterInfo(ArgNo).getABI() ==
            ParameterABI::SwiftErrorResult) {
          assert(!swiftErrorTemp.isValid() && "multiple swifterror args");

          QualType pointeeTy = I->Ty->getPointeeType();
          swiftErrorArg = makeNaturalAddressForPointer(
              V, pointeeTy, getContext().getTypeAlignInChars(pointeeTy));

          swiftErrorTemp = CreateMemTempWithoutCast(
              pointeeTy, getPointerAlign(), "swifterror.temp");
          V = swiftErrorTemp.getPointer();
          cast<llvm::AllocaInst>(V)->setSwiftError(true);

          llvm::Value *errorValue = Builder.CreateLoad(swiftErrorArg);
          Builder.CreateStore(errorValue, swiftErrorTemp);
        }

        // We might have to widen integers, but we should never truncate.
        if (ArgInfo.getCoerceToType() != V->getType() &&
            V->getType()->isIntegerTy())
          V = Builder.CreateZExt(V, ArgInfo.getCoerceToType());

        // The only plausible mismatch here would be for pointer address spaces.
        // We assume that the target has a reasonable mapping for the DefaultAS
        // (it can be casted to from incoming specific ASes), and insert an AS
        // cast to address the mismatch.
        if (FirstIRArg < IRFuncTy->getNumParams() &&
            V->getType() != IRFuncTy->getParamType(FirstIRArg)) {
          assert(V->getType()->isPointerTy() && "Only pointers can mismatch!");
          V = performAddrSpaceCast(V, IRFuncTy->getParamType(FirstIRArg));
        }

        if (ArgHasMaybeUndefAttr)
          V = Builder.CreateFreeze(V);
        IRCallArgs[FirstIRArg] = V;
        break;
      }

      llvm::StructType *STy =
          dyn_cast<llvm::StructType>(ArgInfo.getCoerceToType());

      // FIXME: Avoid the conversion through memory if possible.
      Address Src = Address::invalid();
      if (!I->isAggregate()) {
        Src = CreateMemTempWithoutCast(I->Ty, "coerce");
        I->copyInto(*this, Src);
      } else {
        Src = I->hasLValue() ? I->getKnownLValue().getAddress()
                             : I->getKnownRValue().getAggregateAddress();
      }

      // If the value is offset in memory, apply the offset now.
      Src = emitAddressAtOffset(*this, Src, ArgInfo);

      // Fast-isel and the optimizer generally like scalar values better than
      // FCAs, so we flatten them if this is safe to do for this argument.
      if (STy && ArgInfo.isDirect() && ArgInfo.getCanBeFlattened()) {
        llvm::Type *SrcTy = Src.getElementType();
        llvm::TypeSize SrcTypeSize =
            CGM.getDataLayout().getTypeAllocSize(SrcTy);
        llvm::TypeSize DstTypeSize = CGM.getDataLayout().getTypeAllocSize(STy);
        if (SrcTypeSize.isScalable()) {
          assert(STy->containsHomogeneousScalableVectorTypes() &&
                 "ABI only supports structure with homogeneous scalable vector "
                 "type");
          assert(SrcTypeSize == DstTypeSize &&
                 "Only allow non-fractional movement of structure with "
                 "homogeneous scalable vector type");
          assert(NumIRArgs == STy->getNumElements());

          llvm::Value *StoredStructValue =
              Builder.CreateLoad(Src, Src.getName() + ".tuple");
          for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i) {
            llvm::Value *Extract = Builder.CreateExtractValue(
                StoredStructValue, i, Src.getName() + ".extract" + Twine(i));
            IRCallArgs[FirstIRArg + i] = Extract;
          }
        } else {
          uint64_t SrcSize = SrcTypeSize.getFixedValue();
          uint64_t DstSize = DstTypeSize.getFixedValue();
          bool HasPFPFields = getContext().hasPFPFields(I->Ty);

          // If the source type is smaller than the destination type of the
          // coerce-to logic, copy the source value into a temp alloca the size
          // of the destination type to allow loading all of it. The bits past
          // the source value are left undef.
          if (HasPFPFields || SrcSize < DstSize) {
            Address TempAlloca = CreateTempAlloca(STy, Src.getAlignment(),
                                                  Src.getName() + ".coerce");
            if (HasPFPFields) {
              // Structures with PFP fields require a coerced load to remove any
              // pointer signatures.
              Builder.CreateStore(
                  CreatePFPCoercedLoad(Src, I->Ty, ArgInfo.getCoerceToType(),
                                       *this),
                  TempAlloca);
            } else
              Builder.CreateMemCpy(TempAlloca, Src, SrcSize);
            Src = TempAlloca;
          } else {
            Src = Src.withElementType(STy);
          }

          assert(NumIRArgs == STy->getNumElements());
          for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i) {
            Address EltPtr = Builder.CreateStructGEP(Src, i);
            llvm::Value *LI = Builder.CreateLoad(EltPtr);
            if (ArgHasMaybeUndefAttr)
              LI = Builder.CreateFreeze(LI);
            IRCallArgs[FirstIRArg + i] = LI;
          }
        }
      } else {
        // In the simple case, just pass the coerced loaded value.
        assert(NumIRArgs == 1);
        llvm::Value *Load =
            CreateCoercedLoad(Src, I->Ty, ArgInfo.getCoerceToType(), *this);

        if (CallInfo.isCmseNSCall()) {
          // For certain parameter types, clear padding bits, as they may reveal
          // sensitive information.
          // Small struct/union types are passed as integer arrays.
          auto *ATy = dyn_cast<llvm::ArrayType>(Load->getType());
          if (ATy != nullptr && isa<RecordType>(I->Ty.getCanonicalType()))
            Load = EmitCMSEClearRecord(Load, ATy, I->Ty);
        }

        if (ArgHasMaybeUndefAttr)
          Load = Builder.CreateFreeze(Load);
        IRCallArgs[FirstIRArg] = Load;
      }

      break;
    }

    case ABIArgInfo::CoerceAndExpand: {
      auto coercionType = ArgInfo.getCoerceAndExpandType();
      auto layout = CGM.getDataLayout().getStructLayout(coercionType);
      auto unpaddedCoercionType = ArgInfo.getUnpaddedCoerceAndExpandType();
      auto *unpaddedStruct = dyn_cast<llvm::StructType>(unpaddedCoercionType);

      Address addr = Address::invalid();
      RawAddress AllocaAddr = RawAddress::invalid();
      bool NeedLifetimeEnd = false;
      if (I->isAggregate()) {
        addr = I->hasLValue() ? I->getKnownLValue().getAddress()
                              : I->getKnownRValue().getAggregateAddress();

      } else {
        RValue RV = I->getKnownRValue();
        assert(RV.isScalar()); // complex should always just be direct

        llvm::Type *scalarType = RV.getScalarVal()->getType();
        auto scalarAlign = CGM.getDataLayout().getPrefTypeAlign(scalarType);

        // Materialize to a temporary.
        addr = CreateTempAlloca(RV.getScalarVal()->getType(),
                                CharUnits::fromQuantity(std::max(
                                    layout->getAlignment(), scalarAlign)),
                                "tmp",
                                /*ArraySize=*/nullptr, &AllocaAddr);
        NeedLifetimeEnd = EmitLifetimeStart(AllocaAddr.getPointer());

        Builder.CreateStore(RV.getScalarVal(), addr);
      }

      addr = addr.withElementType(coercionType);

      unsigned IRArgPos = FirstIRArg;
      unsigned unpaddedIndex = 0;
      for (unsigned i = 0, e = coercionType->getNumElements(); i != e; ++i) {
        llvm::Type *eltType = coercionType->getElementType(i);
        if (ABIArgInfo::isPaddingForCoerceAndExpand(eltType))
          continue;
        Address eltAddr = Builder.CreateStructGEP(addr, i);
        llvm::Value *elt = CreateCoercedLoad(
            eltAddr, I->Ty,
            unpaddedStruct ? unpaddedStruct->getElementType(unpaddedIndex++)
                           : unpaddedCoercionType,
            *this);
        if (ArgHasMaybeUndefAttr)
          elt = Builder.CreateFreeze(elt);
        IRCallArgs[IRArgPos++] = elt;
      }
      assert(IRArgPos == FirstIRArg + NumIRArgs);

      if (NeedLifetimeEnd)
        EmitLifetimeEnd(AllocaAddr.getPointer());
      break;
    }

    case ABIArgInfo::Expand: {
      unsigned IRArgPos = FirstIRArg;
      ExpandTypeToArgs(I->Ty, *I, IRFuncTy, IRCallArgs, IRArgPos);
      assert(IRArgPos == FirstIRArg + NumIRArgs);
      break;
    }

    case ABIArgInfo::TargetSpecific: {
      Address Src = Address::invalid();
      if (!I->isAggregate()) {
        Src = CreateMemTempWithoutCast(I->Ty, "target_coerce");
        I->copyInto(*this, Src);
      } else {
        Src = I->hasLValue() ? I->getKnownLValue().getAddress()
                             : I->getKnownRValue().getAggregateAddress();
      }

      // If the value is offset in memory, apply the offset now.
      Src = emitAddressAtOffset(*this, Src, ArgInfo);
      llvm::Value *Load =
          CGM.getABIInfo().createCoercedLoad(Src, ArgInfo, *this);
      IRCallArgs[FirstIRArg] = Load;
      break;
    }
    }
  }

  const CGCallee &ConcreteCallee = Callee.prepareConcreteCallee(*this);
  llvm::Value *CalleePtr = ConcreteCallee.getFunctionPointer();

  // If we're using inalloca, set up that argument.
  if (ArgMemory.isValid()) {
    llvm::Value *Arg = ArgMemory.getPointer();
    assert(IRFunctionArgs.hasInallocaArg());
    IRCallArgs[IRFunctionArgs.getInallocaArgNo()] = Arg;
  }

  // 2. Prepare the function pointer.

  // If the callee is a bitcast of a non-variadic function to have a
  // variadic function pointer type, check to see if we can remove the
  // bitcast.  This comes up with unprototyped functions.
  //
  // This makes the IR nicer, but more importantly it ensures that we
  // can inline the function at -O0 if it is marked always_inline.
  auto simplifyVariadicCallee = [](llvm::FunctionType *CalleeFT,
                                   llvm::Value *Ptr) -> llvm::Function * {
    if (!CalleeFT->isVarArg())
      return nullptr;

    // Get underlying value if it's a bitcast
    if (llvm::ConstantExpr *CE = dyn_cast<llvm::ConstantExpr>(Ptr)) {
      if (CE->getOpcode() == llvm::Instruction::BitCast)
        Ptr = CE->getOperand(0);
    }

    llvm::Function *OrigFn = dyn_cast<llvm::Function>(Ptr);
    if (!OrigFn)
      return nullptr;

    llvm::FunctionType *OrigFT = OrigFn->getFunctionType();

    // If the original type is variadic, or if any of the component types
    // disagree, we cannot remove the cast.
    if (OrigFT->isVarArg() ||
        OrigFT->getNumParams() != CalleeFT->getNumParams() ||
        OrigFT->getReturnType() != CalleeFT->getReturnType())
      return nullptr;

    for (unsigned i = 0, e = OrigFT->getNumParams(); i != e; ++i)
      if (OrigFT->getParamType(i) != CalleeFT->getParamType(i))
        return nullptr;

    return OrigFn;
  };

  if (llvm::Function *OrigFn = simplifyVariadicCallee(IRFuncTy, CalleePtr)) {
    CalleePtr = OrigFn;
    IRFuncTy = OrigFn->getFunctionType();
  }

  // 3. Perform the actual call.

  // Deactivate any cleanups that we're supposed to do immediately before
  // the call.
  if (!CallArgs.getCleanupsToDeactivate().empty())
    deactivateArgCleanupsBeforeCall(*this, CallArgs);

  // Update the largest vector width if any arguments have vector types.
  for (unsigned i = 0; i < IRCallArgs.size(); ++i)
    LargestVectorWidth = std::max(LargestVectorWidth,
                                  getMaxVectorWidth(IRCallArgs[i]->getType()));

  // Compute the calling convention and attributes.
  unsigned CallingConv;
  llvm::AttributeList Attrs;
  CGM.ConstructAttributeList(CalleePtr->getName(), CallInfo,
                             Callee.getAbstractInfo(), Attrs, CallingConv,
                             /*AttrOnCallSite=*/true,
                             /*IsThunk=*/false);

  if (CallingConv == llvm::CallingConv::X86_VectorCall &&
      getTarget().getTriple().isWindowsArm64EC()) {
    CGM.Error(Loc, "__vectorcall calling convention is not currently "
                   "supported");
  }

  if (const FunctionDecl *FD = dyn_cast_or_null<FunctionDecl>(CurFuncDecl)) {
    if (FD->hasAttr<StrictFPAttr>())
      // All calls within a strictfp function are marked strictfp
      Attrs = Attrs.addFnAttribute(getLLVMContext(), llvm::Attribute::StrictFP);

    // If -ffast-math is enabled and the function is guarded by an
    // '__attribute__((optnone)) adjust the memory attribute so the BE emits the
    // library call instead of the intrinsic.
    if (FD->hasAttr<OptimizeNoneAttr>() && getLangOpts().FastMath)
      CGM.AdjustMemoryAttribute(CalleePtr->getName(), Callee.getAbstractInfo(),
                                Attrs);
  }
  // Add call-site nomerge attribute if exists.
  if (InNoMergeAttributedStmt)
    Attrs = Attrs.addFnAttribute(getLLVMContext(), llvm::Attribute::NoMerge);

  // Add call-site noinline attribute if exists.
  if (InNoInlineAttributedStmt)
    Attrs = Attrs.addFnAttribute(getLLVMContext(), llvm::Attribute::NoInline);

  // Add call-site always_inline attribute if exists.
  // Note: This corresponds to the [[clang::always_inline]] statement attribute.
  if (InAlwaysInlineAttributedStmt &&
      !CGM.getTargetCodeGenInfo().wouldInliningViolateFunctionCallABI(
          CallerDecl, CalleeDecl))
    Attrs =
        Attrs.addFnAttribute(getLLVMContext(), llvm::Attribute::AlwaysInline);

  // Remove call-site convergent attribute if requested.
  if (InNoConvergentAttributedStmt)
    Attrs =
        Attrs.removeFnAttribute(getLLVMContext(), llvm::Attribute::Convergent);

  // Apply some call-site-specific attributes.
  // TODO: work this into building the attribute set.

  // Apply always_inline to all calls within flatten functions.
  // FIXME: should this really take priority over __try, below?
  if (CurCodeDecl && CurCodeDecl->hasAttr<FlattenAttr>() &&
      !InNoInlineAttributedStmt &&
      !(TargetDecl && TargetDecl->hasAttr<NoInlineAttr>()) &&
      !CGM.getTargetCodeGenInfo().wouldInliningViolateFunctionCallABI(
          CallerDecl, CalleeDecl)) {
    Attrs =
        Attrs.addFnAttribute(getLLVMContext(), llvm::Attribute::AlwaysInline);
  }

  // Disable inlining inside SEH __try blocks.
  if (isSEHTryScope()) {
    Attrs = Attrs.addFnAttribute(getLLVMContext(), llvm::Attribute::NoInline);
  }

  // Decide whether to use a call or an invoke.
  bool CannotThrow;
  if (currentFunctionUsesSEHTry()) {
    // SEH cares about asynchronous exceptions, so everything can "throw."
    CannotThrow = false;
  } else if (isCleanupPadScope() &&
             EHPersonality::get(*this).isMSVCXXPersonality()) {
    // The MSVC++ personality will implicitly terminate the program if an
    // exception is thrown during a cleanup outside of a try/catch.
    // We don't need to model anything in IR to get this behavior.
    CannotThrow = true;
  } else {
    // Otherwise, nounwind call sites will never throw.
    CannotThrow = Attrs.hasFnAttr(llvm::Attribute::NoUnwind);

    if (auto *FPtr = dyn_cast<llvm::Function>(CalleePtr))
      if (FPtr->hasFnAttribute(llvm::Attribute::NoUnwind))
        CannotThrow = true;
  }

  // If we made a temporary, be sure to clean up after ourselves. Note that we
  // can't depend on being inside of an ExprWithCleanups, so we need to manually
  // pop this cleanup later on. Being eager about this is OK, since this
  // temporary is 'invisible' outside of the callee.
  // Use the original alloca pointer (before any addrspacecast) for the
  // lifetime end marker, since lifetime intrinsics must reference the alloca
  // address space.
  if (NeedSRetLifetimeEnd)
    pushFullExprCleanup<CallLifetimeEnd>(NormalEHLifetimeMarker, SRetAlloca);

  llvm::BasicBlock *InvokeDest = CannotThrow ? nullptr : getInvokeDest();

  SmallVector<llvm::OperandBundleDef, 1> BundleList =
      getBundlesForFunclet(CalleePtr);

  if (CGM.getCodeGenOpts().PreserveSourceVariadicArguments &&
      CallInfo.isVariadic())
    BundleList.emplace_back(SourceVariadicArguments::CallBundle.str(),
                            SourceVariadicValues);

  if (SanOpts.has(SanitizerKind::KCFI) &&
      !isa_and_nonnull<FunctionDecl>(TargetDecl))
    EmitKCFIOperandBundle(ConcreteCallee, BundleList);

  // Add the pointer-authentication bundle.
  EmitPointerAuthOperandBundle(ConcreteCallee.getPointerAuthInfo(), BundleList);

  if (const FunctionDecl *FD = dyn_cast_or_null<FunctionDecl>(CurFuncDecl))
    if (FD->hasAttr<StrictFPAttr>())
      // All calls within a strictfp function are marked strictfp
      Attrs = Attrs.addFnAttribute(getLLVMContext(), llvm::Attribute::StrictFP);

  AssumeAlignedAttrEmitter AssumeAlignedAttrEmitter(*this, TargetDecl);
  Attrs = AssumeAlignedAttrEmitter.TryEmitAsCallSiteAttribute(Attrs);

  AllocAlignAttrEmitter AllocAlignAttrEmitter(*this, TargetDecl, CallArgs);
  Attrs = AllocAlignAttrEmitter.TryEmitAsCallSiteAttribute(Attrs);

  // Emit the actual call/invoke instruction.
  llvm::CallBase *CI;
  if (!InvokeDest) {
    CI = Builder.CreateCall(IRFuncTy, CalleePtr, IRCallArgs, BundleList);
  } else {
    llvm::BasicBlock *Cont = createBasicBlock("invoke.cont");
    CI = Builder.CreateInvoke(IRFuncTy, CalleePtr, Cont, InvokeDest, IRCallArgs,
                              BundleList);
    EmitBlock(Cont);
  }
  if (CI->getCalledFunction() && CI->getCalledFunction()->hasName() &&
      CI->getCalledFunction()->getName().starts_with("_Z4sqrt")) {
    SetSqrtFPAccuracy(CI);
  }
  if (callOrInvoke) {
    *callOrInvoke = CI;
    if (CGM.getCodeGenOpts().CallGraphSection) {
      QualType CST;
      if (TargetDecl && TargetDecl->getFunctionType())
        CST = QualType(TargetDecl->getFunctionType(), 0);
      else if (const auto *FPT =
                   Callee.getAbstractInfo().getCalleeFunctionProtoType())
        CST = QualType(FPT, 0);
      else if (const auto *FT =
                   Callee.getAbstractInfo().getCalleeFunctionType())
        CST = QualType(FT, 0);
      else
        llvm_unreachable(
            "Cannot find the callee type to generate callee_type metadata.");

      // Set type identifier metadata of indirect calls for call graph section.
      if (!CST.isNull()) {
        if (!CST->isFunctionProtoType()) {
          // Reconstruct a prototype for unprototyped callees from the argument
          // types passed at the call site (after default argument promotion).
          //
          // Basic Rationale & K&R-Style Definitions:
          // The argument types in CallArgs have already undergone C default
          // argument promotion (e.g., char/short -> int, float -> double).
          // Furthermore, for a K&R-style
          // definition (e.g., void foo(x) short x; { ... }), canonical C ABI
          // semantics expect the promoted type (int) at the call boundary
          // and implicitly cast down to the declared type (short) inside the
          // function. Therefore, signature computation at K&R definition
          // sites must also apply default argument promotion (yielding
          // void(int), not void(short)) so definition and call sites match.
          //
          // Signature Strictness & Normalization:
          // Since type identifier matching relies on exact hash equality, any
          // tolerance for C compatibility rules must be done by normalizing
          // types before hashing.
          // - Standard C allows certain exceptions for unprototyped calls (and
          //   variadic va_arg), such as differences in signedness (e.g.,
          //   passing an int to an unsigned int parameter) or
          //   interchangeability of enum types with their underlying integer
          //   types.
          // - Existing CFI normalization (e.g.,
          //   -fsanitize-cfi-icall-experimental-normalize-integers) normalizes
          //   types by bit-width and signedness (e.g., int vs long on LP64,
          //   which C does not treat as compatible), but does not normalize
          //   away signedness or enum mismatches.
          // - In the future, whether to normalize away signedness, enums, or
          //   integer bit-widths depends on whether call graph analysis should
          //   err on the side of inclusion (admitting any C-valid call) or
          //   strictness (like CFI). Any normalization applied here at the call
          //   site must remain strictly matched with definition-site
          //   type signature computation.
          if (const auto *FNPT = CST->getAs<FunctionNoProtoType>()) {
            SmallVector<QualType, 8> ParamTypes;
            // CallArgs already contains default-promoted argument types for
            // unprototyped calls.
            for (const CallArg &Arg : CallArgs)
              ParamTypes.push_back(Arg.getType());
            FunctionProtoType::ExtProtoInfo EPI;
            CST = getContext().getFunctionType(FNPT->getReturnType(),
                                               ParamTypes, EPI);
          }

          llvm::Metadata *MD =
              CGM.CreateMetadataIdentifierForCallGraphType(CST);
          StringRef TypeStr;
          if (auto *MDS = dyn_cast_or_null<llvm::MDString>(MD))
            TypeStr = MDS->getString();

          CGM.getDiags().Report(Loc, diag::warn_cgs_no_proto) << CST << TypeStr;
        }
        CGM.createCalleeTypeMetadataForIcall(CST, *callOrInvoke);
      }
    }
  }

  // If this is within a function that has the guard(nocf) attribute and is an
  // indirect call, add the "guard_nocf" attribute to this call to indicate that
  // Control Flow Guard checks should not be added, even if the call is inlined.
  if (const auto *FD = dyn_cast_or_null<FunctionDecl>(CurFuncDecl)) {
    if (const auto *A = FD->getAttr<CFGuardAttr>()) {
      if (A->getGuard() == CFGuardAttr::GuardArg::nocf &&
          !CI->getCalledFunction())
        Attrs = Attrs.addFnAttribute(getLLVMContext(), "guard_nocf");
    }
  }

  // Apply the attributes and calling convention.
  CI->setAttributes(Attrs);
  CI->setCallingConv(static_cast<llvm::CallingConv::ID>(CallingConv));

  // Apply various metadata.

  if (!CI->getType()->isVoidTy())
    CI->setName("call");

  if (CGM.shouldEmitConvergenceTokens() && CI->isConvergent())
    CI = addConvergenceControlToken(CI);

  // Update largest vector width from the return type.
  LargestVectorWidth =
      std::max(LargestVectorWidth, getMaxVectorWidth(CI->getType()));

  // Insert instrumentation or attach profile metadata at indirect call sites.
  // For more details, see the comment before the definition of
  // IPVK_IndirectCallTarget in InstrProfData.inc.
  if (!CI->getCalledFunction())
    PGO->valueProfile(Builder, llvm::IPVK_IndirectCallTarget, CI, CalleePtr);

  // In ObjC ARC mode with no ObjC ARC exception safety, tell the ARC
  // optimizer it can aggressively ignore unwind edges.
  if (CGM.getLangOpts().ObjCAutoRefCount)
    AddObjCARCExceptionMetadata(CI);

  // Set tail call kind if necessary.
  bool IsPPC = getTarget().getTriple().isPPC();
  bool IsMIPS = getTarget().getTriple().isMIPS();
  bool HasMips16 = false;
  if (IsMIPS) {
    const TargetOptions &TargetOpts = getTarget().getTargetOpts();
    HasMips16 = TargetOpts.FeatureMap.lookup("mips16");
    if (!HasMips16)
      HasMips16 = llvm::is_contained(TargetOpts.Features, "+mips16");
  }
  if (llvm::CallInst *Call = dyn_cast<llvm::CallInst>(CI)) {
    if (TargetDecl && TargetDecl->hasAttr<NotTailCalledAttr>())
      Call->setTailCallKind(llvm::CallInst::TCK_NoTail);
    else if (IsMustTail) {
      if (IsPPC) {
        if (getTarget().getTriple().isOSAIX())
          CGM.getDiags().Report(Loc, diag::err_aix_musttail_unsupported);
        else if (!getTarget().hasFeature("pcrelative-memops")) {
          if (getTarget().hasFeature("longcall"))
            CGM.getDiags().Report(Loc, diag::err_ppc_impossible_musttail) << 0;
          else if (Call->isIndirectCall())
            CGM.getDiags().Report(Loc, diag::err_ppc_impossible_musttail) << 1;
          else if (isa_and_nonnull<FunctionDecl>(TargetDecl)) {
            if (!cast<FunctionDecl>(TargetDecl)->isDefined())
              // The undefined callee may be a forward declaration. Without
              // knowning all symbols in the module, we won't know the symbol is
              // defined or not. Collect all these symbols for later diagnosing.
              CGM.addUndefinedGlobalForTailCall(
                  {cast<FunctionDecl>(TargetDecl), Loc});
            else {
              llvm::GlobalValue::LinkageTypes Linkage = CGM.getFunctionLinkage(
                  GlobalDecl(cast<FunctionDecl>(TargetDecl)));
              if (llvm::GlobalValue::isWeakForLinker(Linkage) ||
                  llvm::GlobalValue::isDiscardableIfUnused(Linkage))
                CGM.getDiags().Report(Loc, diag::err_ppc_impossible_musttail)
                    << 2;
            }
          }
        }
      }
      if (IsMIPS) {
        if (HasMips16)
          CGM.getDiags().Report(Loc, diag::err_mips_impossible_musttail) << 0;
        else if (const auto *FD = dyn_cast_or_null<FunctionDecl>(TargetDecl))
          CGM.addUndefinedGlobalForTailCall({FD, Loc});
      }
      Call->setTailCallKind(llvm::CallInst::TCK_MustTail);
    }
  }

  // Add metadata for calls to MSAllocator functions
  if (getDebugInfo() && TargetDecl && TargetDecl->hasAttr<MSAllocatorAttr>())
    getDebugInfo()->addHeapAllocSiteMetadata(CI, RetTy->getPointeeType(), Loc);

  // Add srcloc metadata for [[gnu::error/warning]] diagnostics. When
  // ShowInliningChain is enabled, also track inline/static calls for the
  // heuristic fallback when debug info is not available. This heuristic is
  // conservative and best-effort since static or inline-annotated functions
  // are still not guaranteed to be inlined.
  if (TargetDecl) {
    bool NeedSrcLoc = TargetDecl->hasAttr<ErrorAttr>();
    if (!NeedSrcLoc && CGM.getCodeGenOpts().ShowInliningChain) {
      if (const auto *FD = dyn_cast<FunctionDecl>(TargetDecl))
        NeedSrcLoc = FD->isInlined() || FD->hasAttr<AlwaysInlineAttr>() ||
                     FD->getStorageClass() == SC_Static ||
                     FD->isInAnonymousNamespace();
    }
    if (NeedSrcLoc) {
      auto *Line = llvm::ConstantInt::get(Int64Ty, Loc.getRawEncoding());
      auto *MD = llvm::ConstantAsMetadata::get(Line);
      CI->setMetadata("srcloc", llvm::MDNode::get(getLLVMContext(), {MD}));
    }
  }

  // 4. Finish the call.

  // If the call doesn't return, finish the basic block and clear the
  // insertion point; this allows the rest of IRGen to discard
  // unreachable code.
  if (CI->doesNotReturn()) {
    if (NeedSRetLifetimeEnd)
      PopCleanupBlock();

    // Strip away the noreturn attribute to better diagnose unreachable UB.
    if (SanOpts.has(SanitizerKind::Unreachable)) {
      // Also remove from function since CallBase::hasFnAttr additionally checks
      // attributes of the called function.
      if (auto *F = CI->getCalledFunction())
        F->removeFnAttr(llvm::Attribute::NoReturn);
      CI->removeFnAttr(llvm::Attribute::NoReturn);

      // Avoid incompatibility with ASan which relies on the `noreturn`
      // attribute to insert handler calls.
      if (SanOpts.hasOneOf(SanitizerKind::Address |
                           SanitizerKind::KernelAddress)) {
        SanitizerScope SanScope(this);
        llvm::IRBuilder<>::InsertPointGuard IPGuard(Builder);
        Builder.SetInsertPoint(CI);
        auto *FnType = llvm::FunctionType::get(CGM.VoidTy, /*isVarArg=*/false);
        llvm::FunctionCallee Fn =
            CGM.CreateRuntimeFunction(FnType, "__asan_handle_no_return");
        EmitNounwindRuntimeCall(Fn);
      }
    }

    EmitUnreachable(Loc);
    Builder.ClearInsertionPoint();

    // FIXME: For now, emit a dummy basic block because expr emitters in
    // generally are not ready to handle emitting expressions at unreachable
    // points.
    EnsureInsertPoint();

    // Return a reasonable RValue.
    return GetUndefRValue(RetTy);
  }

  // If this is a musttail call, return immediately. We do not branch to the
  // epilogue in this case.
  if (IsMustTail) {
    for (auto it = EHStack.find(CurrentCleanupScopeDepth); it != EHStack.end();
         ++it) {
      // A noexcept caller pushes an EHTerminateScope to call std::terminate()
      // if an exception escapes. A musttail call replaces the caller's frame,
      // removing this handler. This is safe if the callee is also nounwind:
      // the callee's own noexcept handler prevents any exception from reaching
      // where the caller's handler would have been.
      if (isa<EHTerminateScope>(&*it)) {
        if (CI->doesNotThrow())
          continue;
        CGM.getDiags().Report(MustTailCall->getBeginLoc(),
                              diag::err_musttail_noexcept_mismatch);
        break;
      }
      EHCleanupScope *Cleanup = dyn_cast<EHCleanupScope>(&*it);
      // Fake uses can be safely emitted immediately prior to the tail call, so
      // we choose to emit them just before the call here.
      if (Cleanup && Cleanup->isFakeUse()) {
        CGBuilderTy::InsertPointGuard IPG(Builder);
        Builder.SetInsertPoint(CI);
        Cleanup->getCleanup()->Emit(*this, EHScopeStack::Cleanup::Flags());
      } else if (!(Cleanup &&
                   Cleanup->getCleanup()->isRedundantBeforeReturn())) {
        CGM.ErrorUnsupported(MustTailCall, "tail call skipping over cleanups");
      }
    }
    if (CI->getType()->isVoidTy())
      Builder.CreateRetVoid();
    else
      Builder.CreateRet(CI);
    Builder.ClearInsertionPoint();
    EnsureInsertPoint();
    return GetUndefRValue(RetTy);
  }

  // Perform the swifterror writeback.
  if (swiftErrorTemp.isValid()) {
    llvm::Value *errorResult = Builder.CreateLoad(swiftErrorTemp);
    Builder.CreateStore(errorResult, swiftErrorArg);
  }

  // Emit any call-associated writebacks immediately.  Arguably this
  // should happen after any return-value munging.
  if (CallArgs.hasWritebacks())
    EmitWritebacks(CallArgs);

  // The stack cleanup for inalloca arguments has to run out of the normal
  // lexical order, so deactivate it and run it manually here.
  CallArgs.freeArgumentMemory(*this);

  // Extract the return value.
  RValue Ret;

  // If the current function is a virtual function pointer thunk, avoid copying
  // the return value of the musttail call to a temporary.
  if (IsVirtualFunctionPointerThunk) {
    Ret = RValue::get(CI);
  } else {
    Ret = [&] {
      switch (RetAI.getKind()) {
      case ABIArgInfo::CoerceAndExpand: {
        auto coercionType = RetAI.getCoerceAndExpandType();

        Address addr = SRetPtr.withElementType(coercionType);

        assert(CI->getType() == RetAI.getUnpaddedCoerceAndExpandType());
        bool requiresExtract = isa<llvm::StructType>(CI->getType());

        unsigned unpaddedIndex = 0;
        for (unsigned i = 0, e = coercionType->getNumElements(); i != e; ++i) {
          llvm::Type *eltType = coercionType->getElementType(i);
          if (ABIArgInfo::isPaddingForCoerceAndExpand(eltType))
            continue;
          Address eltAddr = Builder.CreateStructGEP(addr, i);
          llvm::Value *elt = CI;
          if (requiresExtract)
            elt = Builder.CreateExtractValue(elt, unpaddedIndex++);
          else
            assert(unpaddedIndex == 0);
          Builder.CreateStore(elt, eltAddr);
        }
        [[fallthrough]];
      }

      case ABIArgInfo::InAlloca:
      case ABIArgInfo::Indirect: {
        RValue ret = convertTempToRValue(SRetPtr, RetTy, SourceLocation());
        if (NeedSRetLifetimeEnd)
          PopCleanupBlock();
        return ret;
      }

      case ABIArgInfo::Ignore:
        // If we are ignoring an argument that had a result, make sure to
        // construct the appropriate return value for our caller.
        return GetUndefRValue(RetTy);

      case ABIArgInfo::Extend:
      case ABIArgInfo::Direct: {
        llvm::Type *RetIRTy = ConvertType(RetTy);
        if (RetAI.getCoerceToType() == RetIRTy &&
            RetAI.getDirectOffset() == 0) {
          switch (getEvaluationKind(RetTy)) {
          case TEK_Complex: {
            llvm::Value *Real = Builder.CreateExtractValue(CI, 0);
            llvm::Value *Imag = Builder.CreateExtractValue(CI, 1);
            return RValue::getComplex(std::make_pair(Real, Imag));
          }
          case TEK_Aggregate:
            break;
          case TEK_Scalar: {
            // If the argument doesn't match, perform a bitcast to coerce it.
            // This can happen due to trivial type mismatches.
            llvm::Value *V = CI;
            if (V->getType() != RetIRTy)
              V = Builder.CreateBitCast(V, RetIRTy);
            return RValue::get(V);
          }
          }
        }

        // If coercing a fixed vector from a scalable vector for ABI
        // compatibility, and the types match, use the llvm.vector.extract
        // intrinsic to perform the conversion.
        if (auto *FixedDstTy = dyn_cast<llvm::FixedVectorType>(RetIRTy)) {
          llvm::Value *V = CI;
          if (auto *ScalableSrcTy =
                  dyn_cast<llvm::ScalableVectorType>(V->getType())) {
            if (FixedDstTy->getElementType() ==
                ScalableSrcTy->getElementType()) {
              V = Builder.CreateExtractVector(FixedDstTy, V, uint64_t(0),
                                              "cast.fixed");
              return RValue::get(V);
            }
          }
        }

        Address DestPtr = ReturnValue.getValue();
        bool DestIsVolatile = ReturnValue.isVolatile();
        uint64_t DestSize =
            getContext().getTypeInfoDataSizeInChars(RetTy).Width.getQuantity();

        if (!DestPtr.isValid()) {
          DestPtr = CreateMemTempWithoutCast(RetTy, "coerce");
          DestIsVolatile = false;
          DestSize = getContext().getTypeSizeInChars(RetTy).getQuantity();
        }

        // An empty record can overlap other data (if declared with
        // no_unique_address); omit the store for such types - as there is no
        // actual data to store.
        if (!isEmptyRecord(getContext(), RetTy, true)) {
          // If the value is offset in memory, apply the offset now.
          Address StorePtr = emitAddressAtOffset(*this, DestPtr, RetAI);
          CreateCoercedStore(
              CI, RetTy, StorePtr,
              llvm::TypeSize::getFixed(DestSize - RetAI.getDirectOffset()),
              DestIsVolatile);
        }

        return convertTempToRValue(DestPtr, RetTy, SourceLocation());
      }

      case ABIArgInfo::TargetSpecific: {
        Address DestPtr = ReturnValue.getValue();
        Address StorePtr = emitAddressAtOffset(*this, DestPtr, RetAI);
        bool DestIsVolatile = ReturnValue.isVolatile();
        if (!DestPtr.isValid()) {
          DestPtr = CreateMemTempWithoutCast(RetTy, "target_coerce");
          DestIsVolatile = false;
        }
        CGM.getABIInfo().createCoercedStore(CI, StorePtr, RetAI, DestIsVolatile,
                                            *this);
        return convertTempToRValue(DestPtr, RetTy, SourceLocation());
      }

      case ABIArgInfo::Expand:
      case ABIArgInfo::IndirectAliased:
        llvm_unreachable("Invalid ABI kind for return argument");
      }

      llvm_unreachable("Unhandled ABIArgInfo::Kind");
    }();
  }

  // Emit the assume_aligned check on the return value.
  if (Ret.isScalar() && TargetDecl) {
    AssumeAlignedAttrEmitter.EmitAsAnAssumption(Loc, RetTy, Ret);
    AllocAlignAttrEmitter.EmitAsAnAssumption(Loc, RetTy, Ret);
  }

  // Explicitly call CallLifetimeEnd::Emit just to re-use the code even though
  // we can't use the full cleanup mechanism.
  for (CallLifetimeEnd &LifetimeEnd : CallLifetimeEndAfterCall)
    LifetimeEnd.Emit(*this, /*Flags=*/{});

  if (!ReturnValue.isExternallyDestructed() &&
      RetTy.isDestructedType() == QualType::DK_nontrivial_c_struct)
    pushDestroy(QualType::DK_nontrivial_c_struct, Ret.getAggregateAddress(),
                RetTy);

  // Generate function declaration DISuprogram in order to be used
  // in debug info about call sites.
  if (CGDebugInfo *DI = getDebugInfo()) {
    // Ensure call site info would actually be emitted before collecting
    // further callee info.
    if (CalleeDecl && !CalleeDecl->hasAttr<NoDebugAttr>() &&
        DI->getCallSiteRelatedAttrs() != llvm::DINode::FlagZero) {
      CodeGenFunction CalleeCGF(CGM);
      const GlobalDecl &CalleeGlobalDecl =
          Callee.getAbstractInfo().getCalleeDecl();
      CalleeCGF.CurGD = CalleeGlobalDecl;
      FunctionArgList Args;
      QualType ResTy = CalleeCGF.BuildFunctionArgList(CalleeGlobalDecl, Args);
      DI->EmitFuncDeclForCallSite(
          CI, DI->getFunctionType(CalleeDecl, ResTy, Args), CalleeGlobalDecl);
    }
    // Generate call site target information.
    DI->addCallTargetIfVirtual(CalleeDecl, CI);
  }

  return Ret;
}

CGCallee CGCallee::prepareConcreteCallee(CodeGenFunction &CGF) const {
  if (isVirtual()) {
    const CallExpr *CE = getVirtualCallExpr();
    return CGF.CGM.getCXXABI().getVirtualFunctionPointer(
        CGF, getVirtualMethodDecl(), getThisAddress(), getVirtualFunctionType(),
        CE ? CE->getBeginLoc() : SourceLocation());
  }

  return *this;
}

/* VarArg handling */

static RValue emitNativeVAArg(CodeGenFunction &CGF, VAArgExpr *Expr,
                              Address List, AggValueSlot Slot) {
  const ABIInfo &ABI = CGF.CGM.getABIInfo();
  if (Expr->isMicrosoftABI())
    return ABI.EmitMSVAArg(CGF, List, Expr->getType(), Slot);
  if (Expr->isZOSABI())
    return ABI.EmitZOSVAArg(CGF, List, Expr->getType(), Slot);
  return ABI.EmitVAArg(CGF, List, Expr->getType(), Slot);
}

static RValue emitSourceVAArg(CodeGenFunction &CGF, VAArgExpr *Expr,
                              Address List, AggValueSlot Slot) {
  CodeGenModule &CGM = CGF.CGM;
  ASTContext &Context = CGF.getContext();
  QualType Type = Expr->getType();
  Address Result = !Slot.isIgnored()
                       ? Slot.getAddress()
                       : Address(CGF.CreateMemTemp(Type, "va_arg.result"));

  auto *ListArg =
      ImplicitParamDecl::Create(Context, nullptr, Expr->getExprLoc(), nullptr,
                                Context.VoidPtrTy, ImplicitParamKind::Other);
  auto *ResultArg =
      ImplicitParamDecl::Create(Context, nullptr, Expr->getExprLoc(), nullptr,
                                Context.VoidPtrTy, ImplicitParamKind::Other);
  FunctionArgList Arguments{ListArg, ResultArg};
  const auto &Info = CGM.getTypes().arrangeBuiltinFunctionDeclaration(
      Context.VoidTy, Arguments);
  auto *Native = llvm::Function::Create(
      CGM.getTypes().GetFunctionType(Info), llvm::GlobalValue::InternalLinkage,
      SourceVariadicArguments::NativeReadPrefix, CGM.getModule());
  CGM.SetInternalFunctionAttributes(GlobalDecl(), Native, Info);
  Native->addFnAttr(llvm::Attribute::NoUnwind);
  CodeGenFunction Reader(CGM);
  Reader.StartFunction(GlobalDecl(), Context.VoidTy, Native, Info, Arguments,
                       Expr->getExprLoc(), Expr->getExprLoc());
  Address NativeList(
      Reader.Builder.CreateLoad(Reader.GetAddrOfLocalVar(ListArg)),
      List.getElementType(), List.getAlignment());
  Address NativeResult(
      Reader.Builder.CreateLoad(Reader.GetAddrOfLocalVar(ResultArg)),
      Result.getElementType(), Result.getAlignment());
  AggValueSlot NativeSlot = AggValueSlot::forAddr(
      NativeResult, Type.getQualifiers(), AggValueSlot::IsNotDestructed,
      AggValueSlot::DoesNotNeedGCBarriers, AggValueSlot::IsNotAliased,
      AggValueSlot::DoesNotOverlap);
  RValue Value = emitNativeVAArg(Reader, Expr, NativeList, NativeSlot);
  if (Value.isComplex())
    Reader.EmitStoreOfComplex(Value.getComplexVal(),
                              Reader.MakeAddrLValue(NativeResult, Type), true);
  else if (Value.isScalar())
    Reader.EmitStoreThroughLValue(Value,
                                  Reader.MakeAddrLValue(NativeResult, Type));
  Reader.FinishFunction();

  auto Marker = CGM.getModule().getOrInsertFunction(
      SourceVariadicArguments::ReadMarkerPrefix, Native->getFunctionType());
  cast<llvm::Function>(Marker.getCallee())
      ->addFnAttr(llvm::Attribute::NoUnwind);
  auto Bundles = CGF.getBundlesForFunclet(Marker.getCallee());
  SmallVector<llvm::Value *, 2> Description{
      CGF.Builder.getInt64(Context.getTypeSizeInChars(Type).getQuantity()),
      Native};
  Bundles.emplace_back(SourceVariadicArguments::ReadBundle.str(), Description);
  CGF.Builder.CreateCall(
      Marker, {List.emitRawPointer(CGF), Result.emitRawPointer(CGF)}, Bundles);
  return CGF.convertTempToRValue(Result, Type, Expr->getExprLoc());
}

RValue CodeGenFunction::EmitVAArg(VAArgExpr *VE, Address &VAListAddr,
                                  AggValueSlot Slot) {
  VAListAddr = VE->isMicrosoftABI()
                   ? EmitMSVAListRef(VE->getSubExpr())
                   : (VE->isZOSABI() ? EmitZOSVAListRef(VE->getSubExpr())
                                     : EmitVAListRef(VE->getSubExpr()));
  QualType Ty = VE->getType();
  if (Ty->isVariablyModifiedType())
    EmitVariablyModifiedType(Ty);
  if (CGM.getCodeGenOpts().PreserveSourceVariadicArguments)
    return emitSourceVAArg(*this, VE, VAListAddr, Slot);
  return emitNativeVAArg(*this, VE, VAListAddr, Slot);
}

DisableDebugLocationUpdates::DisableDebugLocationUpdates(CodeGenFunction &CGF)
    : CGF(CGF) {
  CGF.disableDebugInfo();
}

DisableDebugLocationUpdates::~DisableDebugLocationUpdates() {
  CGF.enableDebugInfo();
}
