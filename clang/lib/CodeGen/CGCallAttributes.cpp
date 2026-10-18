//===--- CGCallAttributes.cpp - Derive function and call attributes ---===//
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

static void AddAttributesFromFunctionProtoType(ASTContext &Ctx,
                                               llvm::AttrBuilder &FuncAttrs,
                                               const FunctionProtoType *FPT) {
  if (!FPT)
    return;

  if (!isUnresolvedExceptionSpec(FPT->getExceptionSpecType()) &&
      FPT->isNothrow())
    FuncAttrs.addAttribute(llvm::Attribute::NoUnwind);

  unsigned SMEBits = FPT->getAArch64SMEAttributes();
  if (SMEBits & FunctionType::SME_PStateSMEnabledMask)
    FuncAttrs.addAttribute("aarch64_pstate_sm_enabled");
  if (SMEBits & FunctionType::SME_PStateSMCompatibleMask)
    FuncAttrs.addAttribute("aarch64_pstate_sm_compatible");
  if (SMEBits & FunctionType::SME_AgnosticZAStateMask)
    FuncAttrs.addAttribute("aarch64_za_state_agnostic");

  // ZA
  if (FunctionType::getArmZAState(SMEBits) == FunctionType::ARM_Preserves)
    FuncAttrs.addAttribute("aarch64_preserves_za");
  if (FunctionType::getArmZAState(SMEBits) == FunctionType::ARM_In)
    FuncAttrs.addAttribute("aarch64_in_za");
  if (FunctionType::getArmZAState(SMEBits) == FunctionType::ARM_Out)
    FuncAttrs.addAttribute("aarch64_out_za");
  if (FunctionType::getArmZAState(SMEBits) == FunctionType::ARM_InOut)
    FuncAttrs.addAttribute("aarch64_inout_za");

  // ZT0
  if (FunctionType::getArmZT0State(SMEBits) == FunctionType::ARM_Preserves)
    FuncAttrs.addAttribute("aarch64_preserves_zt0");
  if (FunctionType::getArmZT0State(SMEBits) == FunctionType::ARM_In)
    FuncAttrs.addAttribute("aarch64_in_zt0");
  if (FunctionType::getArmZT0State(SMEBits) == FunctionType::ARM_Out)
    FuncAttrs.addAttribute("aarch64_out_zt0");
  if (FunctionType::getArmZT0State(SMEBits) == FunctionType::ARM_InOut)
    FuncAttrs.addAttribute("aarch64_inout_zt0");
}

static void AddAttributesFromOMPAssumes(llvm::AttrBuilder &FuncAttrs,
                                        const Decl *Callee) {
  if (!Callee)
    return;

  SmallVector<StringRef, 4> Attrs;

  for (const OMPAssumeAttr *AA : Callee->specific_attrs<OMPAssumeAttr>())
    AA->getAssumption().split(Attrs, ",");

  if (!Attrs.empty())
    FuncAttrs.addAttribute(llvm::AssumptionAttrKey,
                           llvm::join(Attrs.begin(), Attrs.end(), ","));
}

bool CodeGenModule::MayDropFunctionReturn(const ASTContext &Context,
                                          QualType ReturnType) const {
  // We can't just discard the return value for a record type with a
  // complex destructor or a non-trivially copyable type.
  if (const RecordType *RT =
          ReturnType.getCanonicalType()->getAsCanonical<RecordType>()) {
    if (const auto *ClassDecl = dyn_cast<CXXRecordDecl>(RT->getDecl()))
      return ClassDecl->hasTrivialDestructor();
  }
  return ReturnType.isTriviallyCopyableType(Context);
}

static bool HasStrictReturn(const CodeGenModule &Module, QualType RetTy,
                            const Decl *TargetDecl) {
  // As-is msan can not tolerate noundef mismatch between caller and
  // implementation. Mismatch is possible for e.g. indirect calls from C-caller
  // into C++. Such mismatches lead to confusing false reports. To avoid
  // expensive workaround on msan we enforce initialization event in uncommon
  // cases where it's allowed.
  if (Module.getLangOpts().Sanitize.has(SanitizerKind::Memory))
    return true;
  // C++ explicitly makes returning undefined values UB. C's rule only applies
  // to used values, so we never mark them noundef for now.
  if (!Module.getLangOpts().CPlusPlus)
    return false;
  if (TargetDecl) {
    if (const FunctionDecl *FDecl = dyn_cast<FunctionDecl>(TargetDecl)) {
      if (FDecl->isExternC())
        return false;
    } else if (const VarDecl *VDecl = dyn_cast<VarDecl>(TargetDecl)) {
      // Function pointer.
      if (VDecl->isExternC())
        return false;
    }
  }

  // We don't want to be too aggressive with the return checking, unless
  // it's explicit in the code opts or we're using an appropriate sanitizer.
  // Try to respect what the programmer intended.
  return Module.getCodeGenOpts().StrictReturn ||
         !Module.MayDropFunctionReturn(Module.getContext(), RetTy) ||
         Module.getLangOpts().Sanitize.has(SanitizerKind::Return);
}

/// Add denormal-fp-math and denormal-fp-math-f32 as appropriate for the
/// requested denormal behavior, accounting for the overriding behavior of the
/// -f32 case.
static void addDenormalModeAttrs(llvm::DenormalMode FPDenormalMode,
                                 llvm::DenormalMode FP32DenormalMode,
                                 llvm::AttrBuilder &FuncAttrs) {
  llvm::DenormalFPEnv FPEnv(FPDenormalMode, FP32DenormalMode);
  if (FPEnv != llvm::DenormalFPEnv::getDefault())
    FuncAttrs.addDenormalFPEnvAttr(FPEnv);
}

/// Add default attributes to a function, which have merge semantics under
/// -mlink-builtin-bitcode and should not simply overwrite any existing
/// attributes in the linked library.
static void
addMergableDefaultFunctionAttributes(const CodeGenOptions &CodeGenOpts,
                                     llvm::AttrBuilder &FuncAttrs) {
  addDenormalModeAttrs(CodeGenOpts.FPDenormalMode, CodeGenOpts.FP32DenormalMode,
                       FuncAttrs);
}

static void getTrivialDefaultFunctionAttributes(
    StringRef Name, bool HasOptnone, const CodeGenOptions &CodeGenOpts,
    const LangOptions &LangOpts, bool AttrOnCallSite,
    llvm::AttrBuilder &FuncAttrs) {
  // OptimizeNoneAttr takes precedence over -Os or -Oz. No warning needed.
  if (!HasOptnone) {
    if (CodeGenOpts.OptimizeSize)
      FuncAttrs.addAttribute(llvm::Attribute::OptimizeForSize);
    if (CodeGenOpts.OptimizeSize == 2)
      FuncAttrs.addAttribute(llvm::Attribute::MinSize);
  }

  if (CodeGenOpts.DisableRedZone)
    FuncAttrs.addAttribute(llvm::Attribute::NoRedZone);
  if (CodeGenOpts.IndirectTlsSegRefs)
    FuncAttrs.addAttribute("indirect-tls-seg-refs");
  if (CodeGenOpts.NoImplicitFloat)
    FuncAttrs.addAttribute(llvm::Attribute::NoImplicitFloat);

  if (AttrOnCallSite) {
    // Attributes that should go on the call site only.
    // FIXME: Look for 'BuiltinAttr' on the function rather than re-checking
    // the -fno-builtin-foo list.
    if (!CodeGenOpts.SimplifyLibCalls || LangOpts.isNoBuiltinFunc(Name))
      FuncAttrs.addAttribute(llvm::Attribute::NoBuiltin);
    if (!CodeGenOpts.TrapFuncName.empty())
      FuncAttrs.addAttribute("trap-func-name", CodeGenOpts.TrapFuncName);
  } else {
    switch (CodeGenOpts.getFramePointer()) {
    case CodeGenOptions::FramePointerKind::None:
      // This is the default behavior.
      break;
    case CodeGenOptions::FramePointerKind::Reserved:
    case CodeGenOptions::FramePointerKind::NonLeafNoReserve:
    case CodeGenOptions::FramePointerKind::NonLeaf:
    case CodeGenOptions::FramePointerKind::All:
      FuncAttrs.addAttribute("frame-pointer",
                             CodeGenOptions::getFramePointerKindName(
                                 CodeGenOpts.getFramePointer()));
    }

    if (CodeGenOpts.LessPreciseFPMAD)
      FuncAttrs.addAttribute("less-precise-fpmad", "true");

    if (CodeGenOpts.NullPointerIsValid)
      FuncAttrs.addAttribute(llvm::Attribute::NullPointerIsValid);

    if (LangOpts.getDefaultExceptionMode() == LangOptions::FPE_Ignore)
      FuncAttrs.addAttribute("no-trapping-math", "true");

    // TODO: Are these all needed?
    // unsafe/inf/nan/nsz are handled by instruction-level FastMathFlags.
    if (CodeGenOpts.SoftFloat)
      FuncAttrs.addAttribute("use-soft-float", "true");
    FuncAttrs.addAttribute("stack-protector-buffer-size",
                           llvm::utostr(CodeGenOpts.SSPBufferSize));
    if (LangOpts.NoSignedZero)
      FuncAttrs.addAttribute("no-signed-zeros-fp-math", "true");

    // TODO: Reciprocal estimate codegen options should apply to instructions?
    const std::vector<std::string> &Recips = CodeGenOpts.Reciprocals;
    if (!Recips.empty())
      FuncAttrs.addAttribute("reciprocal-estimates", llvm::join(Recips, ","));

    if (!CodeGenOpts.PreferVectorWidth.empty() &&
        CodeGenOpts.PreferVectorWidth != "none")
      FuncAttrs.addAttribute("prefer-vector-width",
                             CodeGenOpts.PreferVectorWidth);

    if (CodeGenOpts.StackRealignment)
      FuncAttrs.addAttribute("stackrealign");
    if (CodeGenOpts.Backchain)
      FuncAttrs.addAttribute("backchain");
    if (CodeGenOpts.EnableSegmentedStacks)
      FuncAttrs.addAttribute("split-stack");

    if (CodeGenOpts.SpeculativeLoadHardening)
      FuncAttrs.addAttribute(llvm::Attribute::SpeculativeLoadHardening);

    // Add zero-call-used-regs attribute.
    switch (CodeGenOpts.getZeroCallUsedRegs()) {
    case llvm::ZeroCallUsedRegs::ZeroCallUsedRegsKind::Skip:
      FuncAttrs.removeAttribute("zero-call-used-regs");
      break;
    case llvm::ZeroCallUsedRegs::ZeroCallUsedRegsKind::UsedGPRArg:
      FuncAttrs.addAttribute("zero-call-used-regs", "used-gpr-arg");
      break;
    case llvm::ZeroCallUsedRegs::ZeroCallUsedRegsKind::UsedGPR:
      FuncAttrs.addAttribute("zero-call-used-regs", "used-gpr");
      break;
    case llvm::ZeroCallUsedRegs::ZeroCallUsedRegsKind::UsedArg:
      FuncAttrs.addAttribute("zero-call-used-regs", "used-arg");
      break;
    case llvm::ZeroCallUsedRegs::ZeroCallUsedRegsKind::Used:
      FuncAttrs.addAttribute("zero-call-used-regs", "used");
      break;
    case llvm::ZeroCallUsedRegs::ZeroCallUsedRegsKind::AllGPRArg:
      FuncAttrs.addAttribute("zero-call-used-regs", "all-gpr-arg");
      break;
    case llvm::ZeroCallUsedRegs::ZeroCallUsedRegsKind::AllGPR:
      FuncAttrs.addAttribute("zero-call-used-regs", "all-gpr");
      break;
    case llvm::ZeroCallUsedRegs::ZeroCallUsedRegsKind::AllArg:
      FuncAttrs.addAttribute("zero-call-used-regs", "all-arg");
      break;
    case llvm::ZeroCallUsedRegs::ZeroCallUsedRegsKind::All:
      FuncAttrs.addAttribute("zero-call-used-regs", "all");
      break;
    }
  }

  if (LangOpts.assumeFunctionsAreConvergent()) {
    // Conservatively, mark all functions and calls in CUDA and OpenCL as
    // convergent (meaning, they may call an intrinsically convergent op, such
    // as __syncthreads() / barrier(), and so can't have certain optimizations
    // applied around them).  LLVM will remove this attribute where it safely
    // can.
    FuncAttrs.addAttribute(llvm::Attribute::Convergent);
  }

  // TODO: NoUnwind attribute should be added for other GPU modes HIP,
  // OpenMP offload. AFAIK, neither of them support exceptions in device code.
  if ((LangOpts.CUDA && LangOpts.CUDAIsDevice) || LangOpts.OpenCL ||
      LangOpts.SYCLIsDevice) {
    FuncAttrs.addAttribute(llvm::Attribute::NoUnwind);
  }

  if (CodeGenOpts.SaveRegParams && !AttrOnCallSite)
    FuncAttrs.addAttribute("save-reg-params");

  for (StringRef Attr : CodeGenOpts.DefaultFunctionAttrs) {
    StringRef Var, Value;
    std::tie(Var, Value) = Attr.split('=');
    FuncAttrs.addAttribute(Var, Value);
  }

  TargetInfo::BranchProtectionInfo BPI(LangOpts);
  TargetCodeGenInfo::initBranchProtectionFnAttributes(BPI, FuncAttrs);
}

/// Merges `target-features` from \TargetOpts and \F, and sets the result in
/// \FuncAttr
/// * features from \F are always kept
/// * a feature from \TargetOpts is kept if itself and its opposite are absent
/// from \F
static void
overrideFunctionFeaturesWithTargetFeatures(llvm::AttrBuilder &FuncAttr,
                                           const llvm::Function &F,
                                           const TargetOptions &TargetOpts) {
  auto FFeatures = F.getFnAttribute("target-features");

  llvm::StringSet<> MergedNames;
  SmallVector<StringRef> MergedFeatures;
  MergedFeatures.reserve(TargetOpts.Features.size());

  auto AddUnmergedFeatures = [&](auto &&FeatureRange) {
    for (StringRef Feature : FeatureRange) {
      if (Feature.empty())
        continue;
      assert(Feature[0] == '+' || Feature[0] == '-');
      StringRef Name = Feature.drop_front(1);
      bool Merged = !MergedNames.insert(Name).second;
      if (!Merged)
        MergedFeatures.push_back(Feature);
    }
  };

  if (FFeatures.isValid())
    AddUnmergedFeatures(llvm::split(FFeatures.getValueAsString(), ','));
  AddUnmergedFeatures(TargetOpts.Features);

  if (!MergedFeatures.empty()) {
    llvm::sort(MergedFeatures);
    FuncAttr.addAttribute("target-features", llvm::join(MergedFeatures, ","));
  }
}

void CodeGen::mergeDefaultFunctionDefinitionAttributes(
    llvm::Function &F, const CodeGenOptions &CodeGenOpts,
    const LangOptions &LangOpts, const TargetOptions &TargetOpts,
    bool WillInternalize) {

  llvm::AttrBuilder FuncAttrs(F.getContext());
  // Here we only extract the options that are relevant compared to the version
  // from GetCPUAndFeaturesAttributes.
  if (!TargetOpts.CPU.empty())
    FuncAttrs.addAttribute("target-cpu", TargetOpts.CPU);
  if (!TargetOpts.TuneCPU.empty())
    FuncAttrs.addAttribute("tune-cpu", TargetOpts.TuneCPU);

  ::getTrivialDefaultFunctionAttributes(F.getName(), F.hasOptNone(),
                                        CodeGenOpts, LangOpts,
                                        /*AttrOnCallSite=*/false, FuncAttrs);

  if (!WillInternalize && F.isInterposable()) {
    // Do not promote "dynamic" denormal-fp-math to this translation unit's
    // setting for weak functions that won't be internalized. The user has no
    // real control for how builtin bitcode is linked, so we shouldn't assume
    // later copies will use a consistent mode.
    F.addFnAttrs(FuncAttrs);
    return;
  }

  llvm::AttributeMask AttrsToRemove;

  llvm::DenormalFPEnv OptsFPEnv(CodeGenOpts.FPDenormalMode,
                                CodeGenOpts.FP32DenormalMode);
  llvm::DenormalFPEnv MergedFPEnv =
      OptsFPEnv.mergeCalleeMode(F.getDenormalFPEnv());

  if (MergedFPEnv == llvm::DenormalFPEnv::getDefault()) {
    AttrsToRemove.addAttribute(llvm::Attribute::DenormalFPEnv);
  } else {
    // Overwrite existing attribute
    FuncAttrs.addDenormalFPEnvAttr(MergedFPEnv);
  }

  F.removeFnAttrs(AttrsToRemove);

  overrideFunctionFeaturesWithTargetFeatures(FuncAttrs, F, TargetOpts);

  F.addFnAttrs(FuncAttrs);
}

void CodeGenModule::getTrivialDefaultFunctionAttributes(
    StringRef Name, bool HasOptnone, bool AttrOnCallSite,
    llvm::AttrBuilder &FuncAttrs) {
  ::getTrivialDefaultFunctionAttributes(Name, HasOptnone, getCodeGenOpts(),
                                        getLangOpts(), AttrOnCallSite,
                                        FuncAttrs);
}

void CodeGenModule::getDefaultFunctionAttributes(StringRef Name,
                                                 bool HasOptnone,
                                                 bool AttrOnCallSite,
                                                 llvm::AttrBuilder &FuncAttrs) {
  getTrivialDefaultFunctionAttributes(Name, HasOptnone, AttrOnCallSite,
                                      FuncAttrs);

  if (!AttrOnCallSite)
    TargetCodeGenInfo::initPointerAuthFnAttributes(CodeGenOpts.PointerAuth,
                                                   FuncAttrs);

  // If we're just getting the default, get the default values for mergeable
  // attributes.
  if (!AttrOnCallSite)
    addMergableDefaultFunctionAttributes(CodeGenOpts, FuncAttrs);
}

void CodeGenModule::addDefaultFunctionDefinitionAttributes(
    llvm::AttrBuilder &attrs) {
  getDefaultFunctionAttributes(/*function name*/ "", /*optnone*/ false,
                               /*for call*/ false, attrs);
  GetCPUAndFeaturesAttributes(GlobalDecl(), attrs);
}

static void addNoBuiltinAttributes(llvm::AttrBuilder &FuncAttrs,
                                   const LangOptions &LangOpts,
                                   const NoBuiltinAttr *NBA = nullptr) {
  auto AddNoBuiltinAttr = [&FuncAttrs](StringRef BuiltinName) {
    SmallString<32> AttributeName;
    AttributeName += "no-builtin-";
    AttributeName += BuiltinName;
    FuncAttrs.addAttribute(AttributeName);
  };

  // First, handle the language options passed through -fno-builtin.
  if (LangOpts.NoBuiltin) {
    // -fno-builtin disables them all.
    FuncAttrs.addAttribute("no-builtins");
    return;
  }

  // Then, add attributes for builtins specified through -fno-builtin-<name>.
  llvm::for_each(LangOpts.NoBuiltinFuncs, AddNoBuiltinAttr);

  // Now, let's check the __attribute__((no_builtin("...")) attribute added to
  // the source.
  if (!NBA)
    return;

  // If there is a wildcard in the builtin names specified through the
  // attribute, disable them all.
  if (llvm::is_contained(NBA->builtinNames(), "*")) {
    FuncAttrs.addAttribute("no-builtins");
    return;
  }

  // And last, add the rest of the builtin names.
  llvm::for_each(NBA->builtinNames(), AddNoBuiltinAttr);
}

static bool DetermineNoUndef(QualType QTy, CodeGenTypes &Types,
                             const llvm::DataLayout &DL, const ABIArgInfo &AI,
                             bool CheckCoerce = true) {
  llvm::Type *Ty = Types.ConvertTypeForMem(QTy);
  if (AI.getKind() == ABIArgInfo::Indirect ||
      AI.getKind() == ABIArgInfo::IndirectAliased)
    return true;
  if (AI.getKind() == ABIArgInfo::Extend && !AI.isNoExt())
    return true;
  if (!DL.typeSizeEqualsStoreSize(Ty))
    // TODO: This will result in a modest amount of values not marked noundef
    // when they could be. We care about values that *invisibly* contain undef
    // bits from the perspective of LLVM IR.
    return false;
  if (CheckCoerce && AI.canHaveCoerceToType()) {
    llvm::Type *CoerceTy = AI.getCoerceToType();
    if (llvm::TypeSize::isKnownGT(DL.getTypeSizeInBits(CoerceTy),
                                  DL.getTypeSizeInBits(Ty)))
      // If we're coercing to a type with a greater size than the canonical one,
      // we're introducing new undef bits.
      // Coercing to a type of smaller or equal size is ok, as we know that
      // there's no internal padding (typeSizeEqualsStoreSize).
      return false;
  }
  if (QTy->isBitIntType())
    return true;
  if (QTy->isReferenceType())
    return true;
  if (QTy->isNullPtrType())
    return false;
  if (QTy->isMemberPointerType())
    // TODO: Some member pointers are `noundef`, but it depends on the ABI. For
    // now, never mark them.
    return false;
  if (QTy->isScalarType()) {
    if (const ComplexType *Complex = dyn_cast<ComplexType>(QTy))
      return DetermineNoUndef(Complex->getElementType(), Types, DL, AI, false);
    return true;
  }
  if (const VectorType *Vector = dyn_cast<VectorType>(QTy))
    return DetermineNoUndef(Vector->getElementType(), Types, DL, AI, false);
  if (const MatrixType *Matrix = dyn_cast<MatrixType>(QTy))
    return DetermineNoUndef(Matrix->getElementType(), Types, DL, AI, false);
  if (const ArrayType *Array = dyn_cast<ArrayType>(QTy))
    return DetermineNoUndef(Array->getElementType(), Types, DL, AI, false);

  // TODO: Some structs may be `noundef`, in specific situations.
  return false;
}

/// Check if the argument of a function has maybe_undef attribute.
bool CodeGen::IsArgumentMaybeUndef(const Decl *TargetDecl,
                                 unsigned NumRequiredArgs, unsigned ArgNo) {
  const auto *FD = dyn_cast_or_null<FunctionDecl>(TargetDecl);
  if (!FD)
    return false;

  // Assume variadic arguments do not have maybe_undef attribute.
  if (ArgNo >= NumRequiredArgs)
    return false;

  // Check if argument has maybe_undef attribute.
  if (ArgNo < FD->getNumParams()) {
    const ParmVarDecl *Param = FD->getParamDecl(ArgNo);
    if (Param && Param->hasAttr<MaybeUndefAttr>())
      return true;
  }

  return false;
}

/// Test if it's legal to apply nofpclass for the given parameter type and it's
/// lowered IR type.
static bool canApplyNoFPClass(const ABIArgInfo &AI, QualType ParamType,
                              bool IsReturn) {
  // Should only apply to FP types in the source, not ABI promoted.
  if (!ParamType->hasFloatingRepresentation())
    return false;

  // The promoted-to IR type also needs to support nofpclass.
  llvm::Type *IRTy = AI.getCoerceToType();
  if (llvm::AttributeFuncs::isNoFPClassCompatibleType(IRTy))
    return true;

  if (llvm::StructType *ST = dyn_cast<llvm::StructType>(IRTy)) {
    return !IsReturn && AI.getCanBeFlattened() &&
           llvm::all_of(ST->elements(),
                        llvm::AttributeFuncs::isNoFPClassCompatibleType);
  }

  return false;
}

/// Return the nofpclass mask that can be applied to floating-point parameters.
static llvm::FPClassTest getNoFPClassTestMask(const LangOptions &LangOpts) {
  llvm::FPClassTest Mask = llvm::fcNone;
  if (LangOpts.NoHonorInfs)
    Mask |= llvm::fcInf;
  if (LangOpts.NoHonorNaNs)
    Mask |= llvm::fcNan;
  return Mask;
}

void CodeGenModule::AdjustMemoryAttribute(StringRef Name,
                                          CGCalleeInfo CalleeInfo,
                                          llvm::AttributeList &Attrs) {
  if (Attrs.getMemoryEffects().getModRef() == llvm::ModRefInfo::NoModRef) {
    Attrs = Attrs.removeFnAttribute(getLLVMContext(), llvm::Attribute::Memory);
    llvm::Attribute MemoryAttr = llvm::Attribute::getWithMemoryEffects(
        getLLVMContext(), llvm::MemoryEffects::writeOnly());
    Attrs = Attrs.addFnAttribute(getLLVMContext(), MemoryAttr);
  }
}

/// Construct the IR attribute list of a function or call.
///
/// When adding an attribute, please consider where it should be handled:
///
///   - getDefaultFunctionAttributes is for attributes that are essentially
///     part of the global target configuration (but perhaps can be
///     overridden on a per-function basis).  Adding attributes there
///     will cause them to also be set in frontends that build on Clang's
///     target-configuration logic, as well as for code defined in library
///     modules such as CUDA's libdevice.
///
///   - ConstructAttributeList builds on top of getDefaultFunctionAttributes
///     and adds declaration-specific, convention-specific, and
///     frontend-specific logic.  The last is of particular importance:
///     attributes that restrict how the frontend generates code must be
///     added here rather than getDefaultFunctionAttributes.
///
void CodeGenModule::ConstructAttributeList(StringRef Name,
                                           const CGFunctionInfo &FI,
                                           CGCalleeInfo CalleeInfo,
                                           llvm::AttributeList &AttrList,
                                           unsigned &CallingConv,
                                           bool AttrOnCallSite, bool IsThunk) {
  llvm::AttrBuilder FuncAttrs(getLLVMContext());
  llvm::AttrBuilder RetAttrs(getLLVMContext());

  // Collect function IR attributes from the CC lowering.
  // We'll collect the paramete and result attributes later.
  CallingConv = FI.getEffectiveCallingConvention();
  if (FI.isNoReturn())
    FuncAttrs.addAttribute(llvm::Attribute::NoReturn);
  if (FI.isCmseNSCall())
    FuncAttrs.addAttribute("cmse_nonsecure_call");

  // Collect function IR attributes from the callee prototype if we have one.
  AddAttributesFromFunctionProtoType(getContext(), FuncAttrs,
                                     CalleeInfo.getCalleeFunctionProtoType());
  const Decl *TargetDecl = CalleeInfo.getCalleeDecl().getDecl();

  // Attach assumption attributes to the declaration. If this is a call
  // site, attach assumptions from the caller to the call as well.
  AddAttributesFromOMPAssumes(FuncAttrs, TargetDecl);

  bool HasOptnone = false;
  // The NoBuiltinAttr attached to the target FunctionDecl.
  const NoBuiltinAttr *NBA = nullptr;

  // Some ABIs may result in additional accesses to arguments that may
  // otherwise not be present.
  std::optional<llvm::Attribute::AttrKind> MemAttrForPtrArgs;
  bool AddedPotentialArgAccess = false;
  auto AddPotentialArgAccess = [&]() {
    AddedPotentialArgAccess = true;
    llvm::Attribute A = FuncAttrs.getAttribute(llvm::Attribute::Memory);
    if (A.isValid())
      FuncAttrs.addMemoryAttr(A.getMemoryEffects() |
                              llvm::MemoryEffects::argMemOnly());
  };

  // Collect function IR attributes based on declaration-specific
  // information.
  // FIXME: handle sseregparm someday...
  if (TargetDecl) {
    if (TargetDecl->hasAttr<ReturnsTwiceAttr>())
      FuncAttrs.addAttribute(llvm::Attribute::ReturnsTwice);
    if (TargetDecl->hasAttr<NoThrowAttr>())
      FuncAttrs.addAttribute(llvm::Attribute::NoUnwind);
    if (TargetDecl->hasAttr<NoReturnAttr>())
      FuncAttrs.addAttribute(llvm::Attribute::NoReturn);
    if (TargetDecl->hasAttr<ColdAttr>())
      FuncAttrs.addAttribute(llvm::Attribute::Cold);
    if (TargetDecl->hasAttr<HotAttr>())
      FuncAttrs.addAttribute(llvm::Attribute::Hot);
    if (TargetDecl->hasAttr<NoDuplicateAttr>())
      FuncAttrs.addAttribute(llvm::Attribute::NoDuplicate);
    if (TargetDecl->hasAttr<ConvergentAttr>())
      FuncAttrs.addAttribute(llvm::Attribute::Convergent);

    if (const FunctionDecl *Fn = dyn_cast<FunctionDecl>(TargetDecl)) {
      AddAttributesFromFunctionProtoType(
          getContext(), FuncAttrs, Fn->getType()->getAs<FunctionProtoType>());
      if (AttrOnCallSite && Fn->isReplaceableGlobalAllocationFunction()) {
        // A sane operator new returns a non-aliasing pointer.
        auto Kind = Fn->getDeclName().getCXXOverloadedOperator();
        if (getCodeGenOpts().AssumeSaneOperatorNew &&
            (Kind == OO_New || Kind == OO_Array_New))
          RetAttrs.addAttribute(llvm::Attribute::NoAlias);
      }
      const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(Fn);
      const bool IsVirtualCall = MD && MD->isVirtual();
      // Don't use [[noreturn]], _Noreturn or [[no_builtin]] for a call to a
      // virtual function. These attributes are not inherited by overloads.
      if (!(AttrOnCallSite && IsVirtualCall)) {
        if (Fn->isNoReturn())
          FuncAttrs.addAttribute(llvm::Attribute::NoReturn);
        NBA = Fn->getAttr<NoBuiltinAttr>();
      }
    }

    if (isa<FunctionDecl>(TargetDecl) || isa<VarDecl>(TargetDecl)) {
      // Only place nomerge attribute on call sites, never functions. This
      // allows it to work on indirect virtual function calls.
      if (AttrOnCallSite && TargetDecl->hasAttr<NoMergeAttr>())
        FuncAttrs.addAttribute(llvm::Attribute::NoMerge);
    }

    // 'const', 'pure' and 'noalias' attributed functions are also nounwind.
    if (TargetDecl->hasAttr<ConstAttr>()) {
      FuncAttrs.addMemoryAttr(llvm::MemoryEffects::none());
      FuncAttrs.addAttribute(llvm::Attribute::NoUnwind);
      // gcc specifies that 'const' functions have greater restrictions than
      // 'pure' functions, so they also cannot have infinite loops.
      FuncAttrs.addAttribute(llvm::Attribute::WillReturn);
      MemAttrForPtrArgs = llvm::Attribute::ReadNone;
    } else if (TargetDecl->hasAttr<PureAttr>()) {
      FuncAttrs.addMemoryAttr(llvm::MemoryEffects::readOnly());
      FuncAttrs.addAttribute(llvm::Attribute::NoUnwind);
      // gcc specifies that 'pure' functions cannot have infinite loops.
      FuncAttrs.addAttribute(llvm::Attribute::WillReturn);
      MemAttrForPtrArgs = llvm::Attribute::ReadOnly;
    } else if (TargetDecl->hasAttr<NoAliasAttr>()) {
      FuncAttrs.addMemoryAttr(llvm::MemoryEffects::inaccessibleOrArgMemOnly());
      FuncAttrs.addAttribute(llvm::Attribute::NoUnwind);
    }
    if (const auto *RA = TargetDecl->getAttr<RestrictAttr>();
        RA && RA->getDeallocator() == nullptr)
      RetAttrs.addAttribute(llvm::Attribute::NoAlias);
    if (TargetDecl->hasAttr<ReturnsNonNullAttr>() &&
        !CodeGenOpts.NullPointerIsValid)
      RetAttrs.addAttribute(llvm::Attribute::NonNull);
    if (TargetDecl->hasAttr<AnyX86NoCallerSavedRegistersAttr>())
      FuncAttrs.addAttribute("no_caller_saved_registers");
    if (TargetDecl->hasAttr<AnyX86NoCfCheckAttr>())
      FuncAttrs.addAttribute(llvm::Attribute::NoCfCheck);
    if (TargetDecl->hasAttr<LeafAttr>())
      FuncAttrs.addAttribute(llvm::Attribute::NoCallback);
    if (TargetDecl->hasAttr<BPFFastCallAttr>())
      FuncAttrs.addAttribute("bpf_fastcall");

    HasOptnone = TargetDecl->hasAttr<OptimizeNoneAttr>();
    if (auto *AllocSize = TargetDecl->getAttr<AllocSizeAttr>()) {
      std::optional<unsigned> NumElemsParam;
      if (AllocSize->getNumElemsParam().isValid())
        NumElemsParam = AllocSize->getNumElemsParam().getLLVMIndex();
      FuncAttrs.addAllocSizeAttr(AllocSize->getElemSizeParam().getLLVMIndex(),
                                 NumElemsParam);
    }

    // OpenCL v2.0 Work groups may be whether uniform or not.
    // '-cl-uniform-work-group-size' compile option gets a hint
    // to the compiler that the global work-size be a multiple of
    // the work-group size specified to clEnqueueNDRangeKernel
    // (i.e. work groups are uniform).
    if (getLangOpts().OffloadUniformBlock)
      FuncAttrs.addAttribute("uniform-work-group-size");

    if (TargetDecl->hasAttr<ArmLocallyStreamingAttr>())
      FuncAttrs.addAttribute("aarch64_pstate_sm_body");

    if (auto *ModularFormat = TargetDecl->getAttr<ModularFormatAttr>()) {
      FormatAttr *Format = TargetDecl->getAttr<FormatAttr>();
      StringRef Type = Format->getType()->getName();
      std::string FormatIdx = std::to_string(Format->getFormatIdx());
      std::string FirstArg = std::to_string(Format->getFirstArg());
      SmallVector<StringRef> Args = {
          Type, FormatIdx, FirstArg,
          ModularFormat->getModularImplFn()->getName(),
          ModularFormat->getImplName()};
      llvm::append_range(Args, ModularFormat->aspects());
      FuncAttrs.addAttribute("modular-format", llvm::join(Args, ","));
    }
  }

  // Attach "no-builtins" attributes to:
  // * call sites: both `nobuiltin` and "no-builtins" or "no-builtin-<name>".
  // * definitions: "no-builtins" or "no-builtin-<name>" only.
  // The attributes can come from:
  // * LangOpts: -ffreestanding, -fno-builtin, -fno-builtin-<name>
  // * FunctionDecl attributes: __attribute__((no_builtin(...)))
  addNoBuiltinAttributes(FuncAttrs, getLangOpts(), NBA);

  // Collect function IR attributes based on global settiings.
  getDefaultFunctionAttributes(Name, HasOptnone, AttrOnCallSite, FuncAttrs);

  // Override some default IR attributes based on declaration-specific
  // information.
  if (TargetDecl) {
    if (TargetDecl->hasAttr<NoSpeculativeLoadHardeningAttr>())
      FuncAttrs.removeAttribute(llvm::Attribute::SpeculativeLoadHardening);
    if (TargetDecl->hasAttr<SpeculativeLoadHardeningAttr>())
      FuncAttrs.addAttribute(llvm::Attribute::SpeculativeLoadHardening);
    if (TargetDecl->hasAttr<NoSplitStackAttr>())
      FuncAttrs.removeAttribute("split-stack");
    if (TargetDecl->hasAttr<ZeroCallUsedRegsAttr>()) {
      // A function "__attribute__((...))" overrides the command-line flag.
      auto Kind =
          TargetDecl->getAttr<ZeroCallUsedRegsAttr>()->getZeroCallUsedRegs();
      FuncAttrs.removeAttribute("zero-call-used-regs");
      FuncAttrs.addAttribute(
          "zero-call-used-regs",
          ZeroCallUsedRegsAttr::ConvertZeroCallUsedRegsKindToStr(Kind));
    }

    // Add NonLazyBind attribute to function declarations when -fno-plt
    // is used.
    // FIXME: what if we just haven't processed the function definition
    // yet, or if it's an external definition like C99 inline?
    if (CodeGenOpts.NoPLT) {
      if (auto *Fn = dyn_cast<FunctionDecl>(TargetDecl)) {
        if (!Fn->isDefined() && !AttrOnCallSite) {
          FuncAttrs.addAttribute(llvm::Attribute::NonLazyBind);
        }
      }
    }
    // Remove 'convergent' if requested.
    if (TargetDecl->hasAttr<NoConvergentAttr>())
      FuncAttrs.removeAttribute(llvm::Attribute::Convergent);
  }

  // Add "sample-profile-suffix-elision-policy" attribute for internal linkage
  // functions with -funique-internal-linkage-names.
  if (TargetDecl && CodeGenOpts.UniqueInternalLinkageNames) {
    if (const auto *FD = dyn_cast_or_null<FunctionDecl>(TargetDecl)) {
      if (!FD->isExternallyVisible())
        FuncAttrs.addAttribute("sample-profile-suffix-elision-policy",
                               "selected");
    }
  }

  // Collect non-call-site function IR attributes from declaration-specific
  // information.
  if (!AttrOnCallSite) {
    if (TargetDecl && TargetDecl->hasAttr<CmseNSEntryAttr>())
      FuncAttrs.addAttribute("cmse_nonsecure_entry");

    // Whether tail calls are enabled.
    auto shouldDisableTailCalls = [&] {
      // Should this be honored in getDefaultFunctionAttributes?
      if (CodeGenOpts.DisableTailCalls)
        return true;

      if (!TargetDecl)
        return false;

      if (TargetDecl->hasAttr<DisableTailCallsAttr>() ||
          TargetDecl->hasAttr<AnyX86InterruptAttr>())
        return true;

      if (CodeGenOpts.NoEscapingBlockTailCalls) {
        if (const auto *BD = dyn_cast<BlockDecl>(TargetDecl))
          if (!BD->doesNotEscape())
            return true;
      }

      return false;
    };
    if (shouldDisableTailCalls())
      FuncAttrs.addAttribute("disable-tail-calls", "true");

    // These functions require the returns_twice attribute for correct codegen,
    // but the attribute may not be added if -fno-builtin is specified. We
    // explicitly add that attribute here.
    static const llvm::StringSet<> ReturnsTwiceFn{
        "_setjmpex", "setjmp",      "_setjmp", "vfork",
        "sigsetjmp", "__sigsetjmp", "savectx", "getcontext"};
    if (ReturnsTwiceFn.contains(Name))
      FuncAttrs.addAttribute(llvm::Attribute::ReturnsTwice);

    // CPU/feature overrides.  addDefaultFunctionDefinitionAttributes
    // handles these separately to set them based on the global defaults.
    GetCPUAndFeaturesAttributes(CalleeInfo.getCalleeDecl(), FuncAttrs);

    // Windows hotpatching support
    if (!MSHotPatchFunctions.empty()) {
      bool IsHotPatched = llvm::binary_search(MSHotPatchFunctions, Name);
      if (IsHotPatched)
        FuncAttrs.addAttribute("marked_for_windows_hot_patching");
    }
  }

  // Mark functions that are replaceable by the loader.
  if (CodeGenOpts.isLoaderReplaceableFunctionName(Name))
    FuncAttrs.addAttribute("loader-replaceable");

  // Collect attributes from arguments and return values.
  ClangToLLVMArgMapping IRFunctionArgs(getContext(), FI);

  QualType RetTy = FI.getReturnType();
  const ABIArgInfo &RetAI = FI.getReturnInfo();
  const llvm::DataLayout &DL = getDataLayout();

  // Determine if the return type could be partially undef
  if (CodeGenOpts.EnableNoundefAttrs &&
      HasStrictReturn(*this, RetTy, TargetDecl)) {
    if (!RetTy->isVoidType() && RetAI.getKind() != ABIArgInfo::Indirect &&
        DetermineNoUndef(RetTy, getTypes(), DL, RetAI))
      RetAttrs.addAttribute(llvm::Attribute::NoUndef);
  }

  switch (RetAI.getKind()) {
  case ABIArgInfo::Extend:
    if (RetAI.isSignExt())
      RetAttrs.addAttribute(llvm::Attribute::SExt);
    else if (RetAI.isZeroExt())
      RetAttrs.addAttribute(llvm::Attribute::ZExt);
    else
      RetAttrs.addAttribute(llvm::Attribute::NoExt);
    [[fallthrough]];
  case ABIArgInfo::TargetSpecific:
  case ABIArgInfo::Direct:
    if (RetAI.getInReg())
      RetAttrs.addAttribute(llvm::Attribute::InReg);

    if (canApplyNoFPClass(RetAI, RetTy, true))
      RetAttrs.addNoFPClassAttr(getNoFPClassTestMask(getLangOpts()));

    break;
  case ABIArgInfo::Ignore:
    break;

  case ABIArgInfo::InAlloca:
  case ABIArgInfo::Indirect: {
    // inalloca and sret disable readnone and readonly
    AddPotentialArgAccess();
    break;
  }

  case ABIArgInfo::CoerceAndExpand:
    break;

  case ABIArgInfo::Expand:
  case ABIArgInfo::IndirectAliased:
    llvm_unreachable("Invalid ABI kind for return argument");
  }

  if (!IsThunk) {
    // FIXME: fix this properly, https://reviews.llvm.org/D100388
    if (const auto *RefTy = RetTy->getAs<ReferenceType>()) {
      QualType PTy = RefTy->getPointeeType();
      if (!PTy->isIncompleteType() && PTy->isConstantSizeType())
        RetAttrs.addDereferenceableAttr(
            getMinimumObjectSize(PTy).getQuantity());
      if (getTypes().getTargetAddressSpace(PTy) == 0 &&
          !CodeGenOpts.NullPointerIsValid)
        RetAttrs.addAttribute(llvm::Attribute::NonNull);
      if (PTy->isObjectType()) {
        llvm::Align Alignment =
            getNaturalPointeeTypeAlignment(RetTy).getAsAlign();
        RetAttrs.addAlignmentAttr(Alignment);
      }
    }
  }

  bool hasUsedSRet = false;
  SmallVector<llvm::AttrBuilder, 4> ArgAttrs;
  for (unsigned I = 0; I < IRFunctionArgs.totalIRArgs(); ++I)
    ArgAttrs.emplace_back(getLLVMContext());

  // Attach attributes to sret.
  if (IRFunctionArgs.hasSRetArg()) {
    llvm::AttrBuilder &SRETAttrs = ArgAttrs[IRFunctionArgs.getSRetArgNo()];
    SRETAttrs.addStructRetAttr(getTypes().ConvertTypeForMem(RetTy));
    SRETAttrs.addAttribute(llvm::Attribute::Writable);
    SRETAttrs.addAttribute(llvm::Attribute::DeadOnUnwind);
    hasUsedSRet = true;
    if (RetAI.getInReg())
      SRETAttrs.addAttribute(llvm::Attribute::InReg);
    SRETAttrs.addAlignmentAttr(RetAI.getIndirectAlign().getQuantity());
  }

  // Attach attributes to inalloca argument.
  if (IRFunctionArgs.hasInallocaArg()) {
    ArgAttrs[IRFunctionArgs.getInallocaArgNo()].addInAllocaAttr(
        FI.getArgStruct());
  }

  // Apply `nonnull`, `dereferenceable(N)` and `align N` to the `this` argument,
  // unless this is a thunk function. Add dead_on_return to the `this` argument
  // in base class destructors to aid in DSE.
  // FIXME: fix this properly, https://reviews.llvm.org/D100388
  if (FI.isInstanceMethod() && !IRFunctionArgs.hasInallocaArg() &&
      !FI.arg_begin()->type->isVoidPointerType() && !IsThunk) {
    auto IRArgs = IRFunctionArgs.getIRArgs(0);

    assert(IRArgs.second == 1 && "Expected only a single `this` pointer.");

    llvm::AttrBuilder &Attrs = ArgAttrs[IRArgs.first];

    QualType ThisTy = FI.arg_begin()->type.getTypePtr()->getPointeeType();
    int64_t ThisSz = getMinimumObjectSize(ThisTy).getQuantity();

    if (!CodeGenOpts.NullPointerIsValid &&
        getTypes().getTargetAddressSpace(FI.arg_begin()->type) == 0) {
      Attrs.addAttribute(llvm::Attribute::NonNull);
      Attrs.addDereferenceableAttr(ThisSz);
    } else {
      // FIXME dereferenceable should be correct here, regardless of
      // NullPointerIsValid. However, dereferenceable currently does not always
      // respect NullPointerIsValid and may imply nonnull and break the program.
      // See https://reviews.llvm.org/D66618 for discussions.
      Attrs.addDereferenceableOrNullAttr(ThisSz);
    }

    llvm::Align Alignment =
        getNaturalTypeAlignment(ThisTy, /*BaseInfo=*/nullptr,
                                /*TBAAInfo=*/nullptr, /*forPointeeType=*/true)
            .getAsAlign();
    Attrs.addAlignmentAttr(Alignment);

    const auto *DD = dyn_cast_if_present<CXXDestructorDecl>(
        CalleeInfo.getCalleeDecl().getDecl());
    // Do not annotate vector deleting destructors with dead_on_return as the
    // this pointer in that case points to an array which we cannot
    // statically know the size of. Also do not mark deleting destructors
    // dead_on_return as then we might delete stores inside of a user-defined
    // operator delete implementation if it gets inlined, which would be
    // incorrect as the object's lifetime has already ended and the operator
    // delete implementation is allowed to manipulate the underlying storage.
    if (DD &&
        CalleeInfo.getCalleeDecl().getDtorType() !=
            CXXDtorType::Dtor_VectorDeleting &&
        CalleeInfo.getCalleeDecl().getDtorType() !=
            CXXDtorType::Dtor_Deleting &&
        CodeGenOpts.StrictLifetimes) {
      const CXXRecordDecl *ClassDecl =
          dyn_cast<CXXRecordDecl>(DD->getDeclContext());
      // We cannot add dead_on_return if we have virtual base classes because
      // they will generally still be live after the base object destructor.
      if (ClassDecl->getNumVBases() == 0)
        Attrs.addDeadOnReturnAttr(llvm::DeadOnReturnInfo(
            Context.getASTRecordLayout(ClassDecl).getDataSize().getQuantity()));
    }
  }

  unsigned ArgNo = 0;
  for (CGFunctionInfo::const_arg_iterator I = FI.arg_begin(), E = FI.arg_end();
       I != E; ++I, ++ArgNo) {
    QualType ParamType = I->type;
    const ABIArgInfo &AI = I->info;
    llvm::AttrBuilder Attrs(getLLVMContext());

    // Add attribute for padding argument, if necessary.
    if (IRFunctionArgs.hasPaddingArg(ArgNo)) {
      if (AI.getPaddingInReg()) {
        ArgAttrs[IRFunctionArgs.getPaddingArgNo(ArgNo)].addAttribute(
            llvm::Attribute::InReg);
      }
    }

    // Decide whether the argument we're handling could be partially undef
    if (CodeGenOpts.EnableNoundefAttrs &&
        DetermineNoUndef(ParamType, getTypes(), DL, AI)) {
      Attrs.addAttribute(llvm::Attribute::NoUndef);
    }

    // 'restrict' -> 'noalias' is done in EmitFunctionProlog when we
    // have the corresponding parameter variable.  It doesn't make
    // sense to do it here because parameters are so messed up.
    switch (AI.getKind()) {
    case ABIArgInfo::Extend:
      if (AI.isSignExt())
        Attrs.addAttribute(llvm::Attribute::SExt);
      else if (AI.isZeroExt())
        Attrs.addAttribute(llvm::Attribute::ZExt);
      else
        Attrs.addAttribute(llvm::Attribute::NoExt);
      [[fallthrough]];
    case ABIArgInfo::TargetSpecific:
    case ABIArgInfo::Direct:
      if (ArgNo == 0 && FI.isChainCall())
        Attrs.addAttribute(llvm::Attribute::Nest);
      else if (AI.getInReg())
        Attrs.addAttribute(llvm::Attribute::InReg);
      Attrs.addStackAlignmentAttr(llvm::MaybeAlign(AI.getDirectAlign()));

      if (canApplyNoFPClass(AI, ParamType, false))
        Attrs.addNoFPClassAttr(getNoFPClassTestMask(getLangOpts()));
      break;
    case ABIArgInfo::Indirect: {
      assert(!ParamType->isIncompleteType() &&
             "Pass-by-value parameter has incomplete definition?");

      if (AI.getInReg())
        Attrs.addAttribute(llvm::Attribute::InReg);

      // HLSL out and inout parameters must not be marked with ByVal or
      // DeadOnReturn attributes because stores to these parameters by the
      // callee are visible to the caller.
      if (auto ParamABI = FI.getExtParameterInfo(ArgNo).getABI();
          ParamABI != ParameterABI::HLSLOut &&
          ParamABI != ParameterABI::HLSLInOut) {

        // Depending on the ABI, this may be either a byval or a dead_on_return
        // argument.
        if (AI.getIndirectByVal()) {
          Attrs.addByValAttr(getTypes().ConvertTypeForMem(ParamType));
        } else {
          // Add dead_on_return when the object's lifetime ends in the callee.
          // This includes trivially-destructible objects, as well as objects
          // whose destruction / clean-up is carried out within the callee
          // (e.g., Obj-C ARC-managed structs, MSVC callee-destroyed objects).
          if (!ParamType.isDestructedType() || !ParamType->isRecordType() ||
              ParamType->castAsRecordDecl()->isParamDestroyedInCallee())
            Attrs.addDeadOnReturnAttr(llvm::DeadOnReturnInfo());
        }
      }

      auto *Decl = ParamType->getAsRecordDecl();
      if (CodeGenOpts.PassByValueIsNoAlias && Decl &&
          Decl->getArgPassingRestrictions() ==
              RecordArgPassingKind::CanPassInRegs)
        // When calling the function, the pointer passed in will be the only
        // reference to the underlying object. Mark it accordingly.
        Attrs.addAttribute(llvm::Attribute::NoAlias);

      // TODO: We could add the byref attribute if not byval, but it would
      // require updating many testcases.

      CharUnits Align = AI.getIndirectAlign();

      // In a byval argument, it is important that the required
      // alignment of the type is honored, as LLVM might be creating a
      // *new* stack object, and needs to know what alignment to give
      // it. (Sometimes it can deduce a sensible alignment on its own,
      // but not if clang decides it must emit a packed struct, or the
      // user specifies increased alignment requirements.)
      //
      // This is different from indirect *not* byval, where an aligned copy is
      // already created by the caller, and the align attribute is purely
      // informative. However, this can still be useful information for
      // optimizations, such as giving us one necessary condition for checking
      // if a load to this pointer can be speculatively executed.
      assert(!Align.isZero());
      Attrs.addAlignmentAttr(Align.getQuantity());

      // The `nofree` and `dereferenceable` attributes can already be inferred
      // for `byval` arguments. We'll need to provide additional hints
      // otherwise.
      if (!AI.getIndirectByVal()) {
        // Both 6.9.1 of the C standard and [basic.stc.auto] of the C++ standard
        // require parameters to have automatic storage duration. Therefore, the
        // underlying object of this pointer will not be freed during the
        // function's execution.
        Attrs.addAttribute(llvm::Attribute::NoFree);
        Attrs.addDereferenceableAttr(
            Context.getTypeSizeInChars(ParamType).getQuantity());
      }

      // byval disables readnone and readonly.
      AddPotentialArgAccess();
      break;
    }
    case ABIArgInfo::IndirectAliased: {
      CharUnits Align = AI.getIndirectAlign();
      Attrs.addByRefAttr(getTypes().ConvertTypeForMem(ParamType));
      Attrs.addAlignmentAttr(Align.getQuantity());
      break;
    }
    case ABIArgInfo::Ignore:
    case ABIArgInfo::Expand:
    case ABIArgInfo::CoerceAndExpand:
      break;

    case ABIArgInfo::InAlloca:
      // inalloca disables readnone and readonly.
      AddPotentialArgAccess();
      continue;
    }

    if (const auto *RefTy = ParamType->getAs<ReferenceType>()) {
      QualType PTy = RefTy->getPointeeType();
      if (!PTy->isIncompleteType() && PTy->isConstantSizeType())
        Attrs.addDereferenceableAttr(getMinimumObjectSize(PTy).getQuantity());
      if (getTypes().getTargetAddressSpace(PTy) == 0 &&
          !CodeGenOpts.NullPointerIsValid)
        Attrs.addAttribute(llvm::Attribute::NonNull);
      if (PTy->isObjectType()) {
        llvm::Align Alignment =
            getNaturalPointeeTypeAlignment(ParamType).getAsAlign();
        Attrs.addAlignmentAttr(Alignment);
      }
    }

    // From OpenCL spec v3.0.10 section 6.3.5 Alignment of Types:
    // > For arguments to a __kernel function declared to be a pointer to a
    // > data type, the OpenCL compiler can assume that the pointee is always
    // > appropriately aligned as required by the data type.
    if (TargetDecl &&
        DeviceKernelAttr::isOpenCLSpelling(
            TargetDecl->getAttr<DeviceKernelAttr>()) &&
        ParamType->isPointerType()) {
      QualType PTy = ParamType->getPointeeType();
      if (!PTy->isIncompleteType() && PTy->isConstantSizeType()) {
        llvm::Align Alignment =
            getNaturalPointeeTypeAlignment(ParamType).getAsAlign();
        Attrs.addAlignmentAttr(Alignment);
      }
    }

    switch (FI.getExtParameterInfo(ArgNo).getABI()) {
    case ParameterABI::HLSLOut:
    case ParameterABI::HLSLInOut:
      Attrs.addAttribute(llvm::Attribute::NoAlias);
      break;
    case ParameterABI::Ordinary:
      break;

    case ParameterABI::SwiftIndirectResult: {
      // Add 'sret' if we haven't already used it for something, but
      // only if the result is void.
      if (!hasUsedSRet && RetTy->isVoidType()) {
        Attrs.addStructRetAttr(getTypes().ConvertTypeForMem(ParamType));
        hasUsedSRet = true;
      }

      // Add 'noalias' in either case.
      Attrs.addAttribute(llvm::Attribute::NoAlias);

      // Add 'dereferenceable' and 'alignment'.
      auto PTy = ParamType->getPointeeType();
      if (!PTy->isIncompleteType() && PTy->isConstantSizeType()) {
        auto info = getContext().getTypeInfoInChars(PTy);
        Attrs.addDereferenceableAttr(info.Width.getQuantity());
        Attrs.addAlignmentAttr(info.Align.getAsAlign());
      }
      break;
    }

    case ParameterABI::SwiftErrorResult:
      Attrs.addAttribute(llvm::Attribute::SwiftError);
      break;

    case ParameterABI::SwiftContext:
      Attrs.addAttribute(llvm::Attribute::SwiftSelf);
      break;

    case ParameterABI::SwiftAsyncContext:
      Attrs.addAttribute(llvm::Attribute::SwiftAsync);
      break;
    }

    if (FI.getExtParameterInfo(ArgNo).isNoEscape())
      Attrs.addCapturesAttr(
          llvm::CaptureInfo(llvm::CaptureComponents::Address));

    if (Attrs.hasAttributes()) {
      unsigned FirstIRArg, NumIRArgs;
      std::tie(FirstIRArg, NumIRArgs) = IRFunctionArgs.getIRArgs(ArgNo);
      for (unsigned i = 0; i < NumIRArgs; i++)
        ArgAttrs[FirstIRArg + i].merge(Attrs);
    }
  }
  assert(ArgNo == FI.arg_size());

  // We can't see all potential arguments in a varargs declaration; treat them
  // as if they can access memory.
  if (!AttrOnCallSite && FI.isVariadic())
    AddPotentialArgAccess();

  ArgNo = 0;
  if (AddedPotentialArgAccess && MemAttrForPtrArgs) {
    llvm::FunctionType *FunctionType = getTypes().GetFunctionType(FI);
    for (CGFunctionInfo::const_arg_iterator I = FI.arg_begin(),
                                            E = FI.arg_end();
         I != E; ++I, ++ArgNo) {
      if (I->info.isDirect() || I->info.isExpand() ||
          I->info.isCoerceAndExpand()) {
        unsigned FirstIRArg, NumIRArgs;
        std::tie(FirstIRArg, NumIRArgs) = IRFunctionArgs.getIRArgs(ArgNo);
        for (unsigned i = FirstIRArg; i < FirstIRArg + NumIRArgs; ++i) {
          // The index may be out-of-bounds if the callee is a varargs
          // function.
          //
          // FIXME: We can compute the types of varargs arguments without going
          // through the function type, but the relevant code isn't exposed
          // in a way that can be called from here.
          if (i < FunctionType->getNumParams() &&
              FunctionType->getParamType(i)->isPointerTy()) {
            ArgAttrs[i].addAttribute(*MemAttrForPtrArgs);
          }
        }
      }
    }
  }

  SmallVector<llvm::AttributeSet, 4> ArgAttrSets;
  for (const llvm::AttrBuilder &Attrs : ArgAttrs)
    ArgAttrSets.push_back(llvm::AttributeSet::get(getLLVMContext(), Attrs));

  AttrList = llvm::AttributeList::get(
      getLLVMContext(), llvm::AttributeSet::get(getLLVMContext(), FuncAttrs),
      llvm::AttributeSet::get(getLLVMContext(), RetAttrs), ArgAttrSets);
}

/// An argument came in as a promoted argument; demote it back to its
/// declared type.
