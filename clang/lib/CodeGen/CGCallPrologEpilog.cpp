//===--- CGCallPrologEpilog.cpp - Emit function ABI entry and exit ---===//
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

static llvm::Value *emitArgumentDemotion(CodeGenFunction &CGF,
                                         const VarDecl *var,
                                         llvm::Value *value) {
  llvm::Type *varType = CGF.ConvertType(var->getType());

  // This can happen with promotions that actually don't change the
  // underlying type, like the enum promotions.
  if (value->getType() == varType)
    return value;

  assert((varType->isIntegerTy() || varType->isFloatingPointTy()) &&
         "unexpected promotion type");

  if (isa<llvm::IntegerType>(varType))
    return CGF.Builder.CreateTrunc(value, varType, "arg.unpromote");

  return CGF.Builder.CreateFPCast(value, varType, "arg.unpromote");
}

/// Returns the attribute (either parameter attribute, or function
/// attribute), which declares argument ArgNo to be non-null.
const NonNullAttr *CodeGen::getNonNullAttr(const Decl *FD, const ParmVarDecl *PVD,
                                         QualType ArgType, unsigned ArgNo) {
  // FIXME: __attribute__((nonnull)) can also be applied to:
  //   - references to pointers, where the pointee is known to be
  //     nonnull (apparently a Clang extension)
  //   - transparent unions containing pointers
  // In the former case, LLVM IR cannot represent the constraint. In
  // the latter case, we have no guarantee that the transparent union
  // is in fact passed as a pointer.
  if (!ArgType->isAnyPointerType() && !ArgType->isBlockPointerType())
    return nullptr;
  // First, check attribute on parameter itself.
  if (PVD) {
    if (auto ParmNNAttr = PVD->getAttr<NonNullAttr>())
      return ParmNNAttr;
  }
  // Check function attributes.
  if (!FD)
    return nullptr;
  for (const auto *NNAttr : FD->specific_attrs<NonNullAttr>()) {
    if (NNAttr->isNonNull(ArgNo))
      return NNAttr;
  }
  return nullptr;
}

namespace {
struct CopyBackSwiftError final : EHScopeStack::Cleanup {
  Address Temp;
  Address Arg;
  CopyBackSwiftError(Address temp, Address arg) : Temp(temp), Arg(arg) {}
  void Emit(CodeGenFunction &CGF, Flags flags) override {
    llvm::Value *errorValue = CGF.Builder.CreateLoad(Temp);
    CGF.Builder.CreateStore(errorValue, Arg);
  }
};
} // namespace

void CodeGenFunction::EmitFunctionProlog(const CGFunctionInfo &FI,
                                         llvm::Function *Fn,
                                         const FunctionArgList &Args) {
  if (CurCodeDecl && CurCodeDecl->hasAttr<NakedAttr>())
    // Naked functions don't have prologues.
    return;

  // If this is an implicit-return-zero function, go ahead and
  // initialize the return value.  TODO: it might be nice to have
  // a more general mechanism for this that didn't require synthesized
  // return statements.
  if (const FunctionDecl *FD = dyn_cast_or_null<FunctionDecl>(CurCodeDecl)) {
    if (FD->hasImplicitReturnZero()) {
      QualType RetTy = FD->getReturnType().getUnqualifiedType();
      llvm::Type *LLVMTy = CGM.getTypes().ConvertType(RetTy);
      llvm::Constant *Zero = llvm::Constant::getNullValue(LLVMTy);
      Builder.CreateStore(Zero, ReturnValue);
    }
  }

  // FIXME: We no longer need the types from FunctionArgList; lift up and
  // simplify.

  ClangToLLVMArgMapping IRFunctionArgs(CGM.getContext(), FI);
  assert(Fn->arg_size() == IRFunctionArgs.totalIRArgs());

  // If we're using inalloca, all the memory arguments are GEPs off of the last
  // parameter, which is a pointer to the complete memory area.
  Address ArgStruct = Address::invalid();
  if (IRFunctionArgs.hasInallocaArg())
    ArgStruct = Address(Fn->getArg(IRFunctionArgs.getInallocaArgNo()),
                        FI.getArgStruct(), FI.getArgStructAlignment());

  // Name the struct return parameter.
  if (IRFunctionArgs.hasSRetArg()) {
    auto AI = Fn->getArg(IRFunctionArgs.getSRetArgNo());
    AI->setName("agg.result");
    AI->addAttr(llvm::Attribute::NoAlias);
  }

  // Track if we received the parameter as a pointer (indirect, byval, or
  // inalloca).  If already have a pointer, EmitParmDecl doesn't need to copy it
  // into a local alloca for us.
  SmallVector<ParamValue, 16> ArgVals;
  ArgVals.reserve(Args.size());

  // Create a pointer value for every parameter declaration.  This usually
  // entails copying one or more LLVM IR arguments into an alloca.  Don't push
  // any cleanups or do anything that might unwind.  We do that separately, so
  // we can push the cleanups in the correct order for the ABI.
  assert(FI.arg_size() == Args.size() &&
         "Mismatch between function signature & arguments.");
  unsigned ArgNo = 0;
  CGFunctionInfo::const_arg_iterator info_it = FI.arg_begin();
  for (FunctionArgList::const_iterator i = Args.begin(), e = Args.end(); i != e;
       ++i, ++info_it, ++ArgNo) {
    const VarDecl *Arg = *i;
    const ABIArgInfo &ArgI = info_it->info;

    bool isPromoted =
        isa<ParmVarDecl>(Arg) && cast<ParmVarDecl>(Arg)->isKNRPromoted();
    // We are converting from ABIArgInfo type to VarDecl type directly, unless
    // the parameter is promoted. In this case we convert to
    // CGFunctionInfo::ArgInfo type with subsequent argument demotion.
    QualType Ty = isPromoted ? info_it->type : Arg->getType();
    assert(hasScalarEvaluationKind(Ty) ==
           hasScalarEvaluationKind(Arg->getType()));

    unsigned FirstIRArg, NumIRArgs;
    std::tie(FirstIRArg, NumIRArgs) = IRFunctionArgs.getIRArgs(ArgNo);

    switch (ArgI.getKind()) {
    case ABIArgInfo::InAlloca: {
      assert(NumIRArgs == 0);
      auto FieldIndex = ArgI.getInAllocaFieldIndex();
      Address V =
          Builder.CreateStructGEP(ArgStruct, FieldIndex, Arg->getName());
      if (ArgI.getInAllocaIndirect())
        V = Address(Builder.CreateLoad(V), ConvertTypeForMem(Ty),
                    getContext().getTypeAlignInChars(Ty));
      ArgVals.push_back(ParamValue::forIndirect(V));
      break;
    }

    case ABIArgInfo::Indirect:
    case ABIArgInfo::IndirectAliased: {
      assert(NumIRArgs == 1);
      Address ParamAddr = makeNaturalAddressForPointer(
          Fn->getArg(FirstIRArg), Ty, ArgI.getIndirectAlign(), false, nullptr,
          nullptr, KnownNonNull);

      if (!hasScalarEvaluationKind(Ty)) {
        // Aggregates and complex variables are accessed by reference. All we
        // need to do is realign the value, if requested. Also, if the address
        // may be aliased, copy it to ensure that the parameter variable is
        // mutable and has a unique adress, as C requires.
        if (ArgI.getIndirectRealign() || ArgI.isIndirectAliased()) {
          RawAddress AlignedTemp = CreateMemTempWithoutCast(Ty, "coerce");

          // Copy from the incoming argument pointer to the temporary with the
          // appropriate alignment.
          //
          // FIXME: We should have a common utility for generating an aggregate
          // copy.
          CharUnits Size = getContext().getTypeSizeInChars(Ty);
          Builder.CreateMemCpy(
              AlignedTemp.getPointer(), AlignedTemp.getAlignment().getAsAlign(),
              ParamAddr.emitRawPointer(*this),
              ParamAddr.getAlignment().getAsAlign(),
              llvm::ConstantInt::get(IntPtrTy, Size.getQuantity()));
          ParamAddr = AlignedTemp;
        }
        ArgVals.push_back(ParamValue::forIndirect(ParamAddr));
      } else {
        // Load scalar value from indirect argument.
        llvm::Value *V =
            EmitLoadOfScalar(ParamAddr, false, Ty, Arg->getBeginLoc());

        if (isPromoted)
          V = emitArgumentDemotion(*this, Arg, V);
        ArgVals.push_back(ParamValue::forDirect(V));
      }
      break;
    }

    case ABIArgInfo::Extend:
    case ABIArgInfo::Direct: {
      auto AI = Fn->getArg(FirstIRArg);
      llvm::Type *LTy = ConvertType(Arg->getType());

      // Prepare parameter attributes. So far, only attributes for pointer
      // parameters are prepared. See
      // http://llvm.org/docs/LangRef.html#paramattrs.
      if (ArgI.getDirectOffset() == 0 && LTy->isPointerTy() &&
          ArgI.getCoerceToType()->isPointerTy()) {
        assert(NumIRArgs == 1);

        if (const ParmVarDecl *PVD = dyn_cast<ParmVarDecl>(Arg)) {
          // Set `nonnull` attribute if any.
          if (getNonNullAttr(CurCodeDecl, PVD, PVD->getType(),
                             PVD->getFunctionScopeIndex()) &&
              !CGM.getCodeGenOpts().NullPointerIsValid)
            AI->addAttr(llvm::Attribute::NonNull);

          QualType OTy = PVD->getOriginalType();
          if (const auto *ArrTy = getContext().getAsConstantArrayType(OTy)) {
            // A C99 array parameter declaration with the static keyword also
            // indicates dereferenceability, and if the size is constant we can
            // use the dereferenceable attribute (which requires the size in
            // bytes).
            if (ArrTy->getSizeModifier() == ArraySizeModifier::Static) {
              QualType ETy = ArrTy->getElementType();
              llvm::Align Alignment =
                  CGM.getNaturalTypeAlignment(ETy).getAsAlign();
              AI->addAttrs(llvm::AttrBuilder(getLLVMContext())
                               .addAlignmentAttr(Alignment));
              uint64_t ArrSize = ArrTy->getZExtSize();
              if (!ETy->isIncompleteType() && ETy->isConstantSizeType() &&
                  ArrSize) {
                llvm::AttrBuilder Attrs(getLLVMContext());
                Attrs.addDereferenceableAttr(
                    getContext().getTypeSizeInChars(ETy).getQuantity() *
                    ArrSize);
                AI->addAttrs(Attrs);
              } else if (getContext().getTargetInfo().getNullPointerValue(
                             ETy.getAddressSpace()) == 0 &&
                         !CGM.getCodeGenOpts().NullPointerIsValid) {
                AI->addAttr(llvm::Attribute::NonNull);
              }
            }
          } else if (const auto *ArrTy =
                         getContext().getAsVariableArrayType(OTy)) {
            // For C99 VLAs with the static keyword, we don't know the size so
            // we can't use the dereferenceable attribute, but in addrspace(0)
            // we know that it must be nonnull.
            if (ArrTy->getSizeModifier() == ArraySizeModifier::Static) {
              QualType ETy = ArrTy->getElementType();
              llvm::Align Alignment =
                  CGM.getNaturalTypeAlignment(ETy).getAsAlign();
              AI->addAttrs(llvm::AttrBuilder(getLLVMContext())
                               .addAlignmentAttr(Alignment));
              if (!getTypes().getTargetAddressSpace(ETy) &&
                  !CGM.getCodeGenOpts().NullPointerIsValid)
                AI->addAttr(llvm::Attribute::NonNull);
            }
          }

          // Set `align` attribute if any.
          const auto *AVAttr = PVD->getAttr<AlignValueAttr>();
          if (!AVAttr)
            if (const auto *TOTy = OTy->getAs<TypedefType>())
              AVAttr = TOTy->getDecl()->getAttr<AlignValueAttr>();
          if (AVAttr && !SanOpts.has(SanitizerKind::Alignment)) {
            // If alignment-assumption sanitizer is enabled, we do *not* add
            // alignment attribute here, but emit normal alignment assumption,
            // so the UBSAN check could function.
            llvm::ConstantInt *AlignmentCI =
                cast<llvm::ConstantInt>(EmitScalarExpr(AVAttr->getAlignment()));
            uint64_t AlignmentInt =
                AlignmentCI->getLimitedValue(llvm::Value::MaximumAlignment);
            if (AI->getParamAlign().valueOrOne() < AlignmentInt) {
              AI->removeAttr(llvm::Attribute::AttrKind::Alignment);
              AI->addAttrs(llvm::AttrBuilder(getLLVMContext())
                               .addAlignmentAttr(llvm::Align(AlignmentInt)));
            }
          }
        }

        // Set 'noalias' if an argument type has the `restrict` qualifier.
        if (Arg->getType().isRestrictQualified())
          AI->addAttr(llvm::Attribute::NoAlias);
      }

      // Prepare the argument value. If we have the trivial case, handle it
      // with no muss and fuss.
      if (!isa<llvm::StructType>(ArgI.getCoerceToType()) &&
          ArgI.getCoerceToType() == ConvertType(Ty) &&
          ArgI.getDirectOffset() == 0) {
        assert(NumIRArgs == 1);

        // LLVM expects swifterror parameters to be used in very restricted
        // ways.  Copy the value into a less-restricted temporary.
        llvm::Value *V = AI;
        if (FI.getExtParameterInfo(ArgNo).getABI() ==
            ParameterABI::SwiftErrorResult) {
          QualType pointeeTy = Ty->getPointeeType();
          assert(pointeeTy->isPointerType());
          RawAddress temp = CreateMemTempWithoutCast(
              pointeeTy, getPointerAlign(), "swifterror.temp");
          Address arg = makeNaturalAddressForPointer(
              V, pointeeTy, getContext().getTypeAlignInChars(pointeeTy));
          llvm::Value *incomingErrorValue = Builder.CreateLoad(arg);
          Builder.CreateStore(incomingErrorValue, temp);
          V = temp.getPointer();

          // Push a cleanup to copy the value back at the end of the function.
          // The convention does not guarantee that the value will be written
          // back if the function exits with an unwind exception.
          EHStack.pushCleanup<CopyBackSwiftError>(NormalCleanup, temp, arg);
        }

        // Ensure the argument is the correct type.
        if (V->getType() != ArgI.getCoerceToType())
          V = Builder.CreateBitCast(V, ArgI.getCoerceToType());

        if (isPromoted)
          V = emitArgumentDemotion(*this, Arg, V);

        // Because of merging of function types from multiple decls it is
        // possible for the type of an argument to not match the corresponding
        // type in the function type. Since we are codegening the callee
        // in here, add a cast to the argument type.
        llvm::Type *LTy = ConvertType(Arg->getType());
        if (V->getType() != LTy)
          V = Builder.CreateBitCast(V, LTy);

        ArgVals.push_back(ParamValue::forDirect(V));
        break;
      }

      // VLST arguments are coerced to VLATs at the function boundary for
      // ABI consistency. If this is a VLST that was coerced to
      // a VLAT at the function boundary and the types match up, use
      // llvm.vector.extract to convert back to the original VLST.
      if (auto *VecTyTo = dyn_cast<llvm::FixedVectorType>(ConvertType(Ty))) {
        llvm::Value *ArgVal = Fn->getArg(FirstIRArg);
        if (auto *VecTyFrom =
                dyn_cast<llvm::ScalableVectorType>(ArgVal->getType())) {
          auto [Coerced, Extracted] = CoerceScalableToFixed(
              *this, VecTyTo, VecTyFrom, ArgVal, Arg->getName());
          if (Extracted) {
            assert(NumIRArgs == 1);
            ArgVals.push_back(ParamValue::forDirect(Coerced));
            break;
          }
        }
      }

      llvm::StructType *STy =
          dyn_cast<llvm::StructType>(ArgI.getCoerceToType());
      Address Alloca = CreateMemTempWithoutCast(
          Ty, getContext().getDeclAlign(Arg), Arg->getName());

      // Pointer to store into.
      Address Ptr = emitAddressAtOffset(*this, Alloca, ArgI);

      // Fast-isel and the optimizer generally like scalar values better than
      // FCAs, so we flatten them if this is safe to do for this argument.
      if (ArgI.isDirect() && ArgI.getCanBeFlattened() && STy &&
          STy->getNumElements() > 1) {
        llvm::TypeSize StructSize = CGM.getDataLayout().getTypeAllocSize(STy);
        llvm::TypeSize PtrElementSize =
            CGM.getDataLayout().getTypeAllocSize(Ptr.getElementType());
        if (StructSize.isScalable()) {
          assert(STy->containsHomogeneousScalableVectorTypes() &&
                 "ABI only supports structure with homogeneous scalable vector "
                 "type");
          assert(StructSize == PtrElementSize &&
                 "Only allow non-fractional movement of structure with"
                 "homogeneous scalable vector type");
          assert(STy->getNumElements() == NumIRArgs);

          llvm::Value *LoadedStructValue = llvm::PoisonValue::get(STy);
          for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i) {
            auto *AI = Fn->getArg(FirstIRArg + i);
            AI->setName(Arg->getName() + ".coerce" + Twine(i));
            LoadedStructValue =
                Builder.CreateInsertValue(LoadedStructValue, AI, i);
          }

          Builder.CreateStore(LoadedStructValue, Ptr);
        } else {
          uint64_t SrcSize = StructSize.getFixedValue();
          uint64_t DstSize = PtrElementSize.getFixedValue();

          Address AddrToStoreInto = Address::invalid();
          if (SrcSize <= DstSize) {
            AddrToStoreInto = Ptr.withElementType(STy);
          } else {
            AddrToStoreInto =
                CreateTempAlloca(STy, Alloca.getAlignment(), "coerce");
          }

          assert(STy->getNumElements() == NumIRArgs);
          for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i) {
            auto AI = Fn->getArg(FirstIRArg + i);
            AI->setName(Arg->getName() + ".coerce" + Twine(i));
            Address EltPtr = Builder.CreateStructGEP(AddrToStoreInto, i);
            Builder.CreateStore(AI, EltPtr);
          }

          if (SrcSize > DstSize) {
            Builder.CreateMemCpy(Ptr, AddrToStoreInto, DstSize);
          }

          // Structures with PFP fields require a coerced store to add any
          // pointer signatures.
          if (getContext().hasPFPFields(Ty)) {
            llvm::Value *Struct = Builder.CreateLoad(Ptr);
            CreatePFPCoercedStore(Struct, Ty, Ptr, *this);
          }
        }
      } else {
        // Simple case, just do a coerced store of the argument into the alloca.
        assert(NumIRArgs == 1);
        auto AI = Fn->getArg(FirstIRArg);
        AI->setName(Arg->getName() + ".coerce");
        CreateCoercedStore(
            AI, Ty, Ptr,
            llvm::TypeSize::getFixed(
                getContext().getTypeSizeInChars(Ty).getQuantity() -
                ArgI.getDirectOffset()),
            /*DstIsVolatile=*/false);
      }

      // Match to what EmitParmDecl is expecting for this type.
      if (CodeGenFunction::hasScalarEvaluationKind(Ty)) {
        llvm::Value *V =
            EmitLoadOfScalar(Alloca, false, Ty, Arg->getBeginLoc());
        if (isPromoted)
          V = emitArgumentDemotion(*this, Arg, V);
        ArgVals.push_back(ParamValue::forDirect(V));
      } else {
        ArgVals.push_back(ParamValue::forIndirect(Alloca));
      }
      break;
    }

    case ABIArgInfo::CoerceAndExpand: {
      // Reconstruct into a temporary.
      Address alloca =
          CreateMemTempWithoutCast(Ty, getContext().getDeclAlign(Arg));
      ArgVals.push_back(ParamValue::forIndirect(alloca));

      auto coercionType = ArgI.getCoerceAndExpandType();
      auto unpaddedCoercionType = ArgI.getUnpaddedCoerceAndExpandType();
      auto *unpaddedStruct = dyn_cast<llvm::StructType>(unpaddedCoercionType);

      alloca = alloca.withElementType(coercionType);

      unsigned argIndex = FirstIRArg;
      unsigned unpaddedIndex = 0;
      for (unsigned i = 0, e = coercionType->getNumElements(); i != e; ++i) {
        llvm::Type *eltType = coercionType->getElementType(i);
        if (ABIArgInfo::isPaddingForCoerceAndExpand(eltType))
          continue;

        auto eltAddr = Builder.CreateStructGEP(alloca, i);
        llvm::Value *elt = Fn->getArg(argIndex++);

        auto paramType = unpaddedStruct
                             ? unpaddedStruct->getElementType(unpaddedIndex++)
                             : unpaddedCoercionType;

        if (auto *VecTyTo = dyn_cast<llvm::FixedVectorType>(eltType)) {
          if (auto *VecTyFrom = dyn_cast<llvm::ScalableVectorType>(paramType)) {
            bool Extracted;
            std::tie(elt, Extracted) = CoerceScalableToFixed(
                *this, VecTyTo, VecTyFrom, elt, elt->getName());
            assert(Extracted && "Unexpected scalable to fixed vector coercion");
          }
        }
        Builder.CreateStore(elt, eltAddr);
      }
      assert(argIndex == FirstIRArg + NumIRArgs);
      break;
    }

    case ABIArgInfo::Expand: {
      // If this structure was expanded into multiple arguments then
      // we need to create a temporary and reconstruct it from the
      // arguments.
      Address Alloca =
          CreateMemTempWithoutCast(Ty, getContext().getDeclAlign(Arg));
      LValue LV = MakeAddrLValue(Alloca, Ty);
      ArgVals.push_back(ParamValue::forIndirect(Alloca));

      auto FnArgIter = Fn->arg_begin() + FirstIRArg;
      ExpandTypeFromArgs(Ty, LV, FnArgIter);
      assert(FnArgIter == Fn->arg_begin() + FirstIRArg + NumIRArgs);
      for (unsigned i = 0, e = NumIRArgs; i != e; ++i) {
        auto AI = Fn->getArg(FirstIRArg + i);
        AI->setName(Arg->getName() + "." + Twine(i));
      }
      break;
    }

    case ABIArgInfo::TargetSpecific: {
      auto *AI = Fn->getArg(FirstIRArg);
      AI->setName(Arg->getName() + ".target_coerce");
      Address Alloca = CreateMemTempWithoutCast(
          Ty, getContext().getDeclAlign(Arg), Arg->getName());
      Address Ptr = emitAddressAtOffset(*this, Alloca, ArgI);
      CGM.getABIInfo().createCoercedStore(AI, Ptr, ArgI, false, *this);
      if (CodeGenFunction::hasScalarEvaluationKind(Ty)) {
        llvm::Value *V =
            EmitLoadOfScalar(Alloca, false, Ty, Arg->getBeginLoc());
        if (isPromoted) {
          V = emitArgumentDemotion(*this, Arg, V);
        }
        ArgVals.push_back(ParamValue::forDirect(V));
      } else {
        ArgVals.push_back(ParamValue::forIndirect(Alloca));
      }
      break;
    }
    case ABIArgInfo::Ignore:
      assert(NumIRArgs == 0);
      // Initialize the local variable appropriately.
      if (!hasScalarEvaluationKind(Ty)) {
        ArgVals.push_back(
            ParamValue::forIndirect(CreateMemTempWithoutCast(Ty)));
      } else {
        llvm::Value *U = llvm::UndefValue::get(ConvertType(Arg->getType()));
        ArgVals.push_back(ParamValue::forDirect(U));
      }
      break;
    }
  }

  if (getTarget().getCXXABI().areArgsDestroyedLeftToRightInCallee()) {
    for (int I = Args.size() - 1; I >= 0; --I)
      EmitParmDecl(*Args[I], ArgVals[I], I + 1);
  } else {
    for (unsigned I = 0, E = Args.size(); I != E; ++I)
      EmitParmDecl(*Args[I], ArgVals[I], I + 1);
  }
}

static void eraseUnusedBitCasts(llvm::Instruction *insn) {
  while (insn->use_empty()) {
    llvm::BitCastInst *bitcast = dyn_cast<llvm::BitCastInst>(insn);
    if (!bitcast)
      return;

    // This is "safe" because we would have used a ConstantExpr otherwise.
    insn = cast<llvm::Instruction>(bitcast->getOperand(0));
    bitcast->eraseFromParent();
  }
}

/// Try to emit a fused autorelease of a return result.
static llvm::Value *tryEmitFusedAutoreleaseOfResult(CodeGenFunction &CGF,
                                                    llvm::Value *result) {
  // We must be immediately followed the cast.
  llvm::BasicBlock *BB = CGF.Builder.GetInsertBlock();
  if (BB->empty())
    return nullptr;
  if (&BB->back() != result)
    return nullptr;

  llvm::Type *resultType = result->getType();

  // result is in a BasicBlock and is therefore an Instruction.
  llvm::Instruction *generator = cast<llvm::Instruction>(result);

  SmallVector<llvm::Instruction *, 4> InstsToKill;

  // Look for:
  //  %generator = bitcast %type1* %generator2 to %type2*
  while (llvm::BitCastInst *bitcast = dyn_cast<llvm::BitCastInst>(generator)) {
    // We would have emitted this as a constant if the operand weren't
    // an Instruction.
    generator = cast<llvm::Instruction>(bitcast->getOperand(0));

    // Require the generator to be immediately followed by the cast.
    if (generator->getNextNode() != bitcast)
      return nullptr;

    InstsToKill.push_back(bitcast);
  }

  // Look for:
  //   %generator = call i8* @objc_retain(i8* %originalResult)
  // or
  //   %generator = call i8* @objc_retainAutoreleasedReturnValue(i8* %originalResult)
  llvm::CallInst *call = dyn_cast<llvm::CallInst>(generator);
  if (!call)
    return nullptr;

  bool doRetainAutorelease;

  if (call->getCalledOperand() == CGF.CGM.getObjCEntrypoints().objc_retain) {
    doRetainAutorelease = true;
  } else if (call->getCalledOperand() ==
             CGF.CGM.getObjCEntrypoints().objc_retainAutoreleasedReturnValue) {
    doRetainAutorelease = false;

    // If we emitted an assembly marker for this call (and the
    // ARCEntrypoints field should have been set if so), go looking
    // for that call.  If we can't find it, we can't do this
    // optimization.  But it should always be the immediately previous
    // instruction, unless we needed bitcasts around the call.
    if (CGF.CGM.getObjCEntrypoints().retainAutoreleasedReturnValueMarker) {
      llvm::Instruction *prev = call->getPrevNode();
      assert(prev);
      if (isa<llvm::BitCastInst>(prev)) {
        prev = prev->getPrevNode();
        assert(prev);
      }
      assert(isa<llvm::CallInst>(prev));
      assert(cast<llvm::CallInst>(prev)->getCalledOperand() ==
             CGF.CGM.getObjCEntrypoints().retainAutoreleasedReturnValueMarker);
      InstsToKill.push_back(prev);
    }
  } else {
    return nullptr;
  }

  result = call->getArgOperand(0);
  InstsToKill.push_back(call);

  // Keep killing bitcasts, for sanity.  Note that we no longer care
  // about precise ordering as long as there's exactly one use.
  while (llvm::BitCastInst *bitcast = dyn_cast<llvm::BitCastInst>(result)) {
    if (!bitcast->hasOneUse())
      break;
    InstsToKill.push_back(bitcast);
    result = bitcast->getOperand(0);
  }

  // Delete all the unnecessary instructions, from latest to earliest.
  for (auto *I : InstsToKill)
    I->eraseFromParent();

  // Do the fused retain/autorelease if we were asked to.
  if (doRetainAutorelease)
    result = CGF.EmitARCRetainAutoreleaseReturnValue(result);

  // Cast back to the result type.
  return CGF.Builder.CreateBitCast(result, resultType);
}

/// If this is a +1 of the value of an immutable 'self', remove it.
static llvm::Value *tryRemoveRetainOfSelf(CodeGenFunction &CGF,
                                          llvm::Value *result) {
  // This is only applicable to a method with an immutable 'self'.
  const ObjCMethodDecl *method =
      dyn_cast_or_null<ObjCMethodDecl>(CGF.CurCodeDecl);
  if (!method)
    return nullptr;
  const VarDecl *self = method->getSelfDecl();
  if (!self->getType().isConstQualified())
    return nullptr;

  // Look for a retain call. Note: stripPointerCasts looks through returned arg
  // functions, which would cause us to miss the retain.
  llvm::CallInst *retainCall = dyn_cast<llvm::CallInst>(result);
  if (!retainCall || retainCall->getCalledOperand() !=
                         CGF.CGM.getObjCEntrypoints().objc_retain)
    return nullptr;

  // Look for an ordinary load of 'self'.
  llvm::Value *retainedValue = retainCall->getArgOperand(0);
  llvm::LoadInst *load =
      dyn_cast<llvm::LoadInst>(retainedValue->stripPointerCasts());
  if (!load || load->isAtomic() || load->isVolatile() ||
      load->getPointerOperand() != CGF.GetAddrOfLocalVar(self).getBasePointer())
    return nullptr;

  // Okay!  Burn it all down.  This relies for correctness on the
  // assumption that the retain is emitted as part of the return and
  // that thereafter everything is used "linearly".
  llvm::Type *resultType = result->getType();
  eraseUnusedBitCasts(cast<llvm::Instruction>(result));
  assert(retainCall->use_empty());
  retainCall->eraseFromParent();
  eraseUnusedBitCasts(cast<llvm::Instruction>(retainedValue));

  return CGF.Builder.CreateBitCast(load, resultType);
}

/// Emit an ARC autorelease of the result of a function.
///
/// \return the value to actually return from the function
static llvm::Value *emitAutoreleaseOfResult(CodeGenFunction &CGF,
                                            llvm::Value *result) {
  // If we're returning 'self', kill the initial retain.  This is a
  // heuristic attempt to "encourage correctness" in the really unfortunate
  // case where we have a return of self during a dealloc and we desperately
  // need to avoid the possible autorelease.
  if (llvm::Value *self = tryRemoveRetainOfSelf(CGF, result))
    return self;

  // At -O0, try to emit a fused retain/autorelease.
  if (CGF.shouldUseFusedARCCalls())
    if (llvm::Value *fused = tryEmitFusedAutoreleaseOfResult(CGF, result))
      return fused;

  return CGF.EmitARCAutoreleaseReturnValue(result);
}

/// Heuristically search for a dominating store to the return-value slot.
static llvm::StoreInst *findDominatingStoreToReturnValue(CodeGenFunction &CGF) {
  llvm::Value *ReturnValuePtr = CGF.ReturnValue.getBasePointer();

  // Check if a User is a store which pointerOperand is the ReturnValue.
  // We are looking for stores to the ReturnValue, not for stores of the
  // ReturnValue to some other location.
  auto GetStoreIfValid = [&CGF,
                          ReturnValuePtr](llvm::User *U) -> llvm::StoreInst * {
    auto *SI = dyn_cast<llvm::StoreInst>(U);
    if (!SI || SI->getPointerOperand() != ReturnValuePtr ||
        SI->getValueOperand()->getType() != CGF.ReturnValue.getElementType())
      return nullptr;
    // These aren't actually possible for non-coerced returns, and we
    // only care about non-coerced returns on this code path.
    // All memory instructions inside __try block are volatile.
    assert(!SI->isAtomic() &&
           (!SI->isVolatile() || CGF.currentFunctionUsesSEHTry()));
    return SI;
  };
  // If there are multiple uses of the return-value slot, just check
  // for something immediately preceding the IP.  Sometimes this can
  // happen with how we generate implicit-returns; it can also happen
  // with noreturn cleanups.
  if (!ReturnValuePtr->hasOneUse()) {
    llvm::BasicBlock *IP = CGF.Builder.GetInsertBlock();
    if (IP->empty())
      return nullptr;

    // Look at directly preceding instruction, skipping bitcasts, lifetime
    // markers, and fake uses and their operands.
    const llvm::Instruction *LoadIntoFakeUse = nullptr;
    for (llvm::Instruction &I : llvm::reverse(*IP)) {
      // Ignore instructions that are just loads for fake uses; the load should
      // immediately precede the fake use, so we only need to remember the
      // operand for the last fake use seen.
      if (LoadIntoFakeUse == &I)
        continue;
      if (isa<llvm::BitCastInst>(&I))
        continue;
      if (auto *II = dyn_cast<llvm::IntrinsicInst>(&I)) {
        if (II->getIntrinsicID() == llvm::Intrinsic::lifetime_end)
          continue;

        if (II->getIntrinsicID() == llvm::Intrinsic::fake_use) {
          LoadIntoFakeUse = dyn_cast<llvm::Instruction>(II->getArgOperand(0));
          continue;
        }
      }
      return GetStoreIfValid(&I);
    }
    return nullptr;
  }

  llvm::StoreInst *store = GetStoreIfValid(ReturnValuePtr->user_back());
  if (!store)
    return nullptr;

  // Now do a first-and-dirty dominance check: just walk up the
  // single-predecessors chain from the current insertion point.
  llvm::BasicBlock *StoreBB = store->getParent();
  llvm::BasicBlock *IP = CGF.Builder.GetInsertBlock();
  llvm::SmallPtrSet<llvm::BasicBlock *, 4> SeenBBs;
  while (IP != StoreBB) {
    if (!SeenBBs.insert(IP).second || !(IP = IP->getSinglePredecessor()))
      return nullptr;
  }

  // Okay, the store's basic block dominates the insertion point; we
  // can do our thing.
  return store;
}

// Helper functions for EmitCMSEClearRecord

// Set the bits corresponding to a field having width `BitWidth` and located at
// offset `BitOffset` (from the least significant bit) within a storage unit of
// `Bits.size()` bytes. Each element of `Bits` corresponds to one target byte.
// Use little-endian layout, i.e.`Bits[0]` is the LSB.
static void setBitRange(SmallVectorImpl<uint64_t> &Bits, int BitOffset,
                        int BitWidth, int CharWidth) {
  assert(CharWidth <= 64);
  assert(static_cast<unsigned>(BitWidth) <= Bits.size() * CharWidth);

  int Pos = 0;
  if (BitOffset >= CharWidth) {
    Pos += BitOffset / CharWidth;
    BitOffset = BitOffset % CharWidth;
  }

  const uint64_t Used = (uint64_t(1) << CharWidth) - 1;
  if (BitOffset + BitWidth >= CharWidth) {
    Bits[Pos++] |= (Used << BitOffset) & Used;
    BitWidth -= CharWidth - BitOffset;
    BitOffset = 0;
  }

  while (BitWidth >= CharWidth) {
    Bits[Pos++] = Used;
    BitWidth -= CharWidth;
  }

  if (BitWidth > 0)
    Bits[Pos++] |= (Used >> (CharWidth - BitWidth)) << BitOffset;
}

// Set the bits corresponding to a field having width `BitWidth` and located at
// offset `BitOffset` (from the least significant bit) within a storage unit of
// `StorageSize` bytes, located at `StorageOffset` in `Bits`. Each element of
// `Bits` corresponds to one target byte. Use target endian layout.
static void setBitRange(SmallVectorImpl<uint64_t> &Bits, int StorageOffset,
                        int StorageSize, int BitOffset, int BitWidth,
                        int CharWidth, bool BigEndian) {

  SmallVector<uint64_t, 8> TmpBits(StorageSize);
  setBitRange(TmpBits, BitOffset, BitWidth, CharWidth);

  if (BigEndian)
    std::reverse(TmpBits.begin(), TmpBits.end());

  for (uint64_t V : TmpBits)
    Bits[StorageOffset++] |= V;
}

static void setUsedBits(CodeGenModule &, QualType, int,
                        SmallVectorImpl<uint64_t> &);

// Set the bits in `Bits`, which correspond to the value representations of
// the actual members of the record type `RTy`. Note that this function does
// not handle base classes, virtual tables, etc, since they cannot happen in
// CMSE function arguments or return. The bit mask corresponds to the target
// memory layout, i.e. it's endian dependent.
static void setUsedBits(CodeGenModule &CGM, const RecordType *RTy, int Offset,
                        SmallVectorImpl<uint64_t> &Bits) {
  ASTContext &Context = CGM.getContext();
  int CharWidth = Context.getCharWidth();
  const RecordDecl *RD = RTy->getDecl()->getDefinition();
  const ASTRecordLayout &ASTLayout = Context.getASTRecordLayout(RD);
  const CGRecordLayout &Layout = CGM.getTypes().getCGRecordLayout(RD);

  int Idx = 0;
  for (auto I = RD->field_begin(), E = RD->field_end(); I != E; ++I, ++Idx) {
    const FieldDecl *F = *I;

    if (F->isUnnamedBitField() || F->isZeroLengthBitField() ||
        F->getType()->isIncompleteArrayType())
      continue;

    if (F->isBitField()) {
      const CGBitFieldInfo &BFI = Layout.getBitFieldInfo(F);
      setBitRange(Bits, Offset + BFI.StorageOffset.getQuantity(),
                  BFI.StorageSize / CharWidth, BFI.Offset, BFI.Size, CharWidth,
                  CGM.getDataLayout().isBigEndian());
      continue;
    }

    setUsedBits(CGM, F->getType(),
                Offset + ASTLayout.getFieldOffset(Idx) / CharWidth, Bits);
  }
}

// Set the bits in `Bits`, which correspond to the value representations of
// the elements of an array type `ATy`.
static void setUsedBits(CodeGenModule &CGM, const ConstantArrayType *ATy,
                        int Offset, SmallVectorImpl<uint64_t> &Bits) {
  const ASTContext &Context = CGM.getContext();

  QualType ETy = Context.getBaseElementType(ATy);
  int Size = Context.getTypeSizeInChars(ETy).getQuantity();
  SmallVector<uint64_t, 4> TmpBits(Size);
  setUsedBits(CGM, ETy, 0, TmpBits);

  for (int I = 0, N = Context.getConstantArrayElementCount(ATy); I < N; ++I) {
    auto Src = TmpBits.begin();
    auto Dst = Bits.begin() + Offset + I * Size;
    for (int J = 0; J < Size; ++J)
      *Dst++ |= *Src++;
  }
}

// Set the bits in `Bits`, which correspond to the value representations of
// the type `QTy`.
static void setUsedBits(CodeGenModule &CGM, QualType QTy, int Offset,
                        SmallVectorImpl<uint64_t> &Bits) {
  if (const auto *RTy = QTy->getAsCanonical<RecordType>())
    return setUsedBits(CGM, RTy, Offset, Bits);

  ASTContext &Context = CGM.getContext();
  if (const auto *ATy = Context.getAsConstantArrayType(QTy))
    return setUsedBits(CGM, ATy, Offset, Bits);

  int Size = Context.getTypeSizeInChars(QTy).getQuantity();
  if (Size <= 0)
    return;

  std::fill_n(Bits.begin() + Offset, Size,
              (uint64_t(1) << Context.getCharWidth()) - 1);
}

static uint64_t buildMultiCharMask(const SmallVectorImpl<uint64_t> &Bits,
                                   int Pos, int Size, int CharWidth,
                                   bool BigEndian) {
  assert(Size > 0);
  uint64_t Mask = 0;
  if (BigEndian) {
    for (auto P = Bits.begin() + Pos, E = Bits.begin() + Pos + Size; P != E;
         ++P)
      Mask = (Mask << CharWidth) | *P;
  } else {
    auto P = Bits.begin() + Pos + Size, End = Bits.begin() + Pos;
    do
      Mask = (Mask << CharWidth) | *--P;
    while (P != End);
  }
  return Mask;
}

// Emit code to clear the bits in a record, which aren't a part of any user
// declared member, when the record is a function return.
llvm::Value *CodeGenFunction::EmitCMSEClearRecord(llvm::Value *Src,
                                                  llvm::IntegerType *ITy,
                                                  QualType QTy) {
  assert(Src->getType() == ITy);
  assert(ITy->getScalarSizeInBits() <= 64);

  const llvm::DataLayout &DataLayout = CGM.getDataLayout();
  int Size = DataLayout.getTypeStoreSize(ITy);
  SmallVector<uint64_t, 4> Bits(Size);
  setUsedBits(CGM, QTy->castAsCanonical<RecordType>(), 0, Bits);

  int CharWidth = CGM.getContext().getCharWidth();
  uint64_t Mask =
      buildMultiCharMask(Bits, 0, Size, CharWidth, DataLayout.isBigEndian());

  return Builder.CreateAnd(Src, Mask, "cmse.clear");
}

// Emit code to clear the bits in a record, which aren't a part of any user
// declared member, when the record is a function argument.
llvm::Value *CodeGenFunction::EmitCMSEClearRecord(llvm::Value *Src,
                                                  llvm::ArrayType *ATy,
                                                  QualType QTy) {
  const llvm::DataLayout &DataLayout = CGM.getDataLayout();
  int Size = DataLayout.getTypeStoreSize(ATy);
  SmallVector<uint64_t, 16> Bits(Size);
  setUsedBits(CGM, QTy->castAsCanonical<RecordType>(), 0, Bits);

  // Clear each element of the LLVM array.
  int CharWidth = CGM.getContext().getCharWidth();
  int CharsPerElt =
      ATy->getArrayElementType()->getScalarSizeInBits() / CharWidth;
  int MaskIndex = 0;
  llvm::Value *R = llvm::PoisonValue::get(ATy);
  for (int I = 0, N = ATy->getArrayNumElements(); I != N; ++I) {
    uint64_t Mask = buildMultiCharMask(Bits, MaskIndex, CharsPerElt, CharWidth,
                                       DataLayout.isBigEndian());
    MaskIndex += CharsPerElt;
    llvm::Value *T0 = Builder.CreateExtractValue(Src, I);
    llvm::Value *T1 = Builder.CreateAnd(T0, Mask, "cmse.clear");
    R = Builder.CreateInsertValue(R, T1, I);
  }

  return R;
}

void CodeGenFunction::EmitFunctionEpilog(
    const CGFunctionInfo &FI, bool EmitRetDbgLoc, SourceLocation EndLoc,
    uint64_t RetKeyInstructionsSourceAtom) {
  if (FI.isNoReturn()) {
    // Noreturn functions don't return.
    EmitUnreachable(EndLoc);
    return;
  }

  if (CurCodeDecl && CurCodeDecl->hasAttr<NakedAttr>()) {
    // Naked functions don't have epilogues.
    Builder.CreateUnreachable();
    return;
  }

  // Functions with no result always return void.
  if (!ReturnValue.isValid()) {
    auto *I = Builder.CreateRetVoid();
    if (RetKeyInstructionsSourceAtom)
      addInstToSpecificSourceAtom(I, nullptr, RetKeyInstructionsSourceAtom);
    else
      addInstToNewSourceAtom(I, nullptr);
    return;
  }

  llvm::DebugLoc RetDbgLoc;
  llvm::Value *RV = nullptr;
  QualType RetTy = FI.getReturnType();
  const ABIArgInfo &RetAI = FI.getReturnInfo();

  switch (RetAI.getKind()) {
  case ABIArgInfo::InAlloca:
    // Aggregates get evaluated directly into the destination.  Sometimes we
    // need to return the sret value in a register, though.
    assert(hasAggregateEvaluationKind(RetTy));
    if (RetAI.getInAllocaSRet()) {
      llvm::Function::arg_iterator EI = CurFn->arg_end();
      --EI;
      llvm::Value *ArgStruct = &*EI;
      llvm::Value *SRet = Builder.CreateStructGEP(
          FI.getArgStruct(), ArgStruct, RetAI.getInAllocaFieldIndex());
      llvm::Type *Ty =
          cast<llvm::GetElementPtrInst>(SRet)->getResultElementType();
      RV = Builder.CreateAlignedLoad(Ty, SRet, getPointerAlign(), "sret");
    }
    break;

  case ABIArgInfo::Indirect: {
    auto AI = CurFn->arg_begin();
    if (RetAI.isSRetAfterThis())
      ++AI;
    switch (getEvaluationKind(RetTy)) {
    case TEK_Complex: {
      ComplexPairTy RT =
          EmitLoadOfComplex(MakeAddrLValue(ReturnValue, RetTy), EndLoc);
      EmitStoreOfComplex(RT, MakeNaturalAlignAddrLValue(&*AI, RetTy),
                         /*isInit*/ true);
      break;
    }
    case TEK_Aggregate:
      // Do nothing; aggregates get evaluated directly into the destination.
      break;
    case TEK_Scalar: {
      LValueBaseInfo BaseInfo;
      TBAAAccessInfo TBAAInfo;
      CharUnits Alignment =
          CGM.getNaturalTypeAlignment(RetTy, &BaseInfo, &TBAAInfo);
      Address ArgAddr(&*AI, ConvertType(RetTy), Alignment);
      LValue ArgVal =
          LValue::MakeAddr(ArgAddr, RetTy, getContext(), BaseInfo, TBAAInfo);
      EmitStoreOfScalar(
          EmitLoadOfScalar(MakeAddrLValue(ReturnValue, RetTy), EndLoc), ArgVal,
          /*isInit*/ true);
      break;
    }
    }
    break;
  }

  case ABIArgInfo::Extend:
  case ABIArgInfo::Direct:
    if (RetAI.getCoerceToType() == ConvertType(RetTy) &&
        RetAI.getDirectOffset() == 0) {
      // The internal return value temp always will have pointer-to-return-type
      // type, just do a load.

      // If there is a dominating store to ReturnValue, we can elide
      // the load, zap the store, and usually zap the alloca.
      if (llvm::StoreInst *SI = findDominatingStoreToReturnValue(*this)) {
        // Reuse the debug location from the store unless there is
        // cleanup code to be emitted between the store and return
        // instruction.
        if (EmitRetDbgLoc && !AutoreleaseResult)
          RetDbgLoc = SI->getDebugLoc();
        // Get the stored value and nuke the now-dead store.
        RV = SI->getValueOperand();
        SI->eraseFromParent();

      // Otherwise, we have to do a simple load.
      } else {
        RV = Builder.CreateLoad(ReturnValue);
      }
    } else {
      // If the value is offset in memory, apply the offset now.
      Address V = emitAddressAtOffset(*this, ReturnValue, RetAI);

      RV = CreateCoercedLoad(V, RetTy, RetAI.getCoerceToType(), *this);
    }

    // In ARC, end functions that return a retainable type with a call
    // to objc_autoreleaseReturnValue.
    if (AutoreleaseResult) {
#ifndef NDEBUG
      // Type::isObjCRetainabletype has to be called on a QualType that hasn't
      // been stripped of the typedefs, so we cannot use RetTy here. Get the
      // original return type of FunctionDecl, CurCodeDecl, and BlockDecl from
      // CurCodeDecl or BlockInfo.
      QualType RT;

      if (auto *FD = dyn_cast<FunctionDecl>(CurCodeDecl))
        RT = FD->getReturnType();
      else if (auto *MD = dyn_cast<ObjCMethodDecl>(CurCodeDecl))
        RT = MD->getReturnType();
      else if (isa<BlockDecl>(CurCodeDecl))
        RT = BlockInfo->BlockExpression->getFunctionType()->getReturnType();
      else
        llvm_unreachable("Unexpected function/method type");

      assert(getLangOpts().ObjCAutoRefCount && !FI.isReturnsRetained() &&
             RT->isObjCRetainableType());
#endif
      RV = emitAutoreleaseOfResult(*this, RV);
    }

    break;

  case ABIArgInfo::Ignore:
    break;

  case ABIArgInfo::CoerceAndExpand: {
    auto coercionType = RetAI.getCoerceAndExpandType();
    auto unpaddedCoercionType = RetAI.getUnpaddedCoerceAndExpandType();
    auto *unpaddedStruct = dyn_cast<llvm::StructType>(unpaddedCoercionType);

    // Load all of the coerced elements out into results.
    llvm::SmallVector<llvm::Value *, 4> results;
    Address addr = ReturnValue.withElementType(coercionType);
    unsigned unpaddedIndex = 0;
    for (unsigned i = 0, e = coercionType->getNumElements(); i != e; ++i) {
      auto coercedEltType = coercionType->getElementType(i);
      if (ABIArgInfo::isPaddingForCoerceAndExpand(coercedEltType))
        continue;

      auto eltAddr = Builder.CreateStructGEP(addr, i);
      llvm::Value *elt = CreateCoercedLoad(
          eltAddr, RetTy,
          unpaddedStruct ? unpaddedStruct->getElementType(unpaddedIndex++)
                         : unpaddedCoercionType,
          *this);
      results.push_back(elt);
    }

    // If we have one result, it's the single direct result type.
    if (results.size() == 1) {
      RV = results[0];

    // Otherwise, we need to make a first-class aggregate.
    } else {
      // Construct a return type that lacks padding elements.
      llvm::Type *returnType = RetAI.getUnpaddedCoerceAndExpandType();

      RV = llvm::PoisonValue::get(returnType);
      for (unsigned i = 0, e = results.size(); i != e; ++i) {
        RV = Builder.CreateInsertValue(RV, results[i], i);
      }
    }
    break;
  }
  case ABIArgInfo::TargetSpecific: {
    Address V = emitAddressAtOffset(*this, ReturnValue, RetAI);
    RV = CGM.getABIInfo().createCoercedLoad(V, RetAI, *this);
    break;
  }
  case ABIArgInfo::Expand:
  case ABIArgInfo::IndirectAliased:
    llvm_unreachable("Invalid ABI kind for return argument");
  }

  llvm::Instruction *Ret;
  if (RV) {
    if (CurFuncDecl && CurFuncDecl->hasAttr<CmseNSEntryAttr>()) {
      // For certain return types, clear padding bits, as they may reveal
      // sensitive information.
      // Small struct/union types are passed as integers.
      auto *ITy = dyn_cast<llvm::IntegerType>(RV->getType());
      if (ITy != nullptr && isa<RecordType>(RetTy.getCanonicalType()))
        RV = EmitCMSEClearRecord(RV, ITy, RetTy);
    }
    EmitReturnValueCheck(RV);
    Ret = Builder.CreateRet(RV);
  } else {
    Ret = Builder.CreateRetVoid();
  }

  if (RetDbgLoc)
    Ret->setDebugLoc(std::move(RetDbgLoc));

  llvm::Value *Backup = RV ? Ret->getOperand(0) : nullptr;
  if (RetKeyInstructionsSourceAtom)
    addInstToSpecificSourceAtom(Ret, Backup, RetKeyInstructionsSourceAtom);
  else
    addInstToNewSourceAtom(Ret, Backup);
}

void CodeGenFunction::EmitReturnValueCheck(llvm::Value *RV) {
  // A current decl may not be available when emitting vtable thunks.
  if (!CurCodeDecl)
    return;

  // If the return block isn't reachable, neither is this check, so don't emit
  // it.
  if (ReturnBlock.isValid() && ReturnBlock.getBlock()->use_empty())
    return;

  ReturnsNonNullAttr *RetNNAttr = nullptr;
  if (SanOpts.has(SanitizerKind::ReturnsNonnullAttribute))
    RetNNAttr = CurCodeDecl->getAttr<ReturnsNonNullAttr>();

  if (!RetNNAttr && !requiresReturnValueNullabilityCheck())
    return;

  // Prefer the returns_nonnull attribute if it's present.
  SourceLocation AttrLoc;
  SanitizerKind::SanitizerOrdinal CheckKind;
  SanitizerHandler Handler;
  if (RetNNAttr) {
    assert(!requiresReturnValueNullabilityCheck() &&
           "Cannot check nullability and the nonnull attribute");
    AttrLoc = RetNNAttr->getLocation();
    CheckKind = SanitizerKind::SO_ReturnsNonnullAttribute;
    Handler = SanitizerHandler::NonnullReturn;
  } else {
    if (auto *DD = dyn_cast<DeclaratorDecl>(CurCodeDecl))
      if (auto *TSI = DD->getTypeSourceInfo())
        if (auto FTL = TSI->getTypeLoc().getAsAdjusted<FunctionTypeLoc>())
          AttrLoc = FTL.getReturnLoc().findNullabilityLoc();
    CheckKind = SanitizerKind::SO_NullabilityReturn;
    Handler = SanitizerHandler::NullabilityReturn;
  }

  SanitizerDebugLocation SanScope(this, {CheckKind}, Handler);

  // Make sure the "return" source location is valid. If we're checking a
  // nullability annotation, make sure the preconditions for the check are met.
  llvm::BasicBlock *Check = createBasicBlock("nullcheck");
  llvm::BasicBlock *NoCheck = createBasicBlock("no.nullcheck");
  llvm::Value *SLocPtr = Builder.CreateLoad(ReturnLocation, "return.sloc.load");
  llvm::Value *CanNullCheck = Builder.CreateIsNotNull(SLocPtr);
  if (requiresReturnValueNullabilityCheck())
    CanNullCheck =
        Builder.CreateAnd(CanNullCheck, RetValNullabilityPrecondition);
  Builder.CreateCondBr(CanNullCheck, Check, NoCheck);
  EmitBlock(Check);

  // Now do the null check.
  llvm::Value *Cond = Builder.CreateIsNotNull(RV);
  llvm::Constant *StaticData[] = {EmitCheckSourceLocation(AttrLoc)};
  llvm::Value *DynamicData[] = {SLocPtr};
  EmitCheck(std::make_pair(Cond, CheckKind), Handler, StaticData, DynamicData);

  EmitBlock(NoCheck);

#ifndef NDEBUG
  // The return location should not be used after the check has been emitted.
  ReturnLocation = Address::invalid();
#endif
}

