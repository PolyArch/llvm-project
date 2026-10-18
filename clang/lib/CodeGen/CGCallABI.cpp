//===--- CGCallABI.cpp - Arrange function ABI layouts ---===//
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

unsigned CodeGenTypes::ClangCallConvToLLVMCallConv(CallingConv CC) {
  switch (CC) {
  case CC_C:
    // On SPIR/SPIR-V, CC_C is the AST-level default calling convention, but
    // it still needs to lower to spir_func so IR consumers can rely on the
    // calling convention to distinguish device functions.
    if (Target.getTriple().isSPIROrSPIRV())
      return llvm::CallingConv::SPIR_FUNC;
    return llvm::CallingConv::C;
  case CC_X86StdCall:
    return llvm::CallingConv::X86_StdCall;
  case CC_X86FastCall:
    return llvm::CallingConv::X86_FastCall;
  case CC_X86RegCall:
    return llvm::CallingConv::X86_RegCall;
  case CC_X86ThisCall:
    return llvm::CallingConv::X86_ThisCall;
  case CC_Win64:
    return llvm::CallingConv::Win64;
  case CC_X86_64SysV:
    return llvm::CallingConv::X86_64_SysV;
  case CC_AAPCS:
    return llvm::CallingConv::ARM_AAPCS;
  case CC_AAPCS_VFP:
    return llvm::CallingConv::ARM_AAPCS_VFP;
  case CC_IntelOclBicc:
    return llvm::CallingConv::Intel_OCL_BI;
  // TODO: Add support for __pascal to LLVM.
  case CC_X86Pascal:
    return llvm::CallingConv::C;
  // TODO: Add support for __vectorcall to LLVM.
  case CC_X86VectorCall:
    return llvm::CallingConv::X86_VectorCall;
  case CC_AArch64VectorCall:
    return llvm::CallingConv::AArch64_VectorCall;
  case CC_AArch64SVEPCS:
    return llvm::CallingConv::AArch64_SVE_VectorCall;
  case CC_DeviceKernel:
    return CGM.getTargetCodeGenInfo().getDeviceKernelCallingConv();
  case CC_PreserveMost:
    return llvm::CallingConv::PreserveMost;
  case CC_PreserveAll:
    return llvm::CallingConv::PreserveAll;
  case CC_Swift:
    return llvm::CallingConv::Swift;
  case CC_SwiftAsync:
    return llvm::CallingConv::SwiftTail;
  case CC_M68kRTD:
    return llvm::CallingConv::M68k_RTD;
  case CC_PreserveNone:
    return llvm::CallingConv::PreserveNone;
    // clang-format off
  case CC_RISCVVectorCall: return llvm::CallingConv::RISCV_VectorCall;
    // clang-format on
#define CC_VLS_CASE(ABI_VLEN)                                                  \
  case CC_RISCVVLSCall_##ABI_VLEN:                                             \
    return llvm::CallingConv::RISCV_VLSCall_##ABI_VLEN;
    CC_VLS_CASE(32)
    CC_VLS_CASE(64)
    CC_VLS_CASE(128)
    CC_VLS_CASE(256)
    CC_VLS_CASE(512)
    CC_VLS_CASE(1024)
    CC_VLS_CASE(2048)
    CC_VLS_CASE(4096)
    CC_VLS_CASE(8192)
    CC_VLS_CASE(16384)
    CC_VLS_CASE(32768)
    CC_VLS_CASE(65536)
#undef CC_VLS_CASE
  }
}

/// Derives the 'this' type for codegen purposes, i.e. ignoring method CVR
/// qualification. Either or both of RD and MD may be null. A null RD indicates
/// that there is no meaningful 'this' type, and a null MD can occur when
/// calling a method pointer.
CanQualType CodeGenTypes::DeriveThisType(const CXXRecordDecl *RD,
                                         const CXXMethodDecl *MD) {
  CanQualType RecTy;
  if (RD)
    RecTy = Context.getCanonicalTagType(RD);
  else
    RecTy = Context.VoidTy;

  if (MD)
    RecTy = CanQualType::CreateUnsafe(Context.getAddrSpaceQualType(
        RecTy, MD->getMethodQualifiers().getAddressSpace()));
  return Context.getPointerType(RecTy);
}

/// Returns the canonical formal type of the given C++ method.
static CanQual<FunctionProtoType> GetFormalType(const CXXMethodDecl *MD) {
  return MD->getType()
      ->getCanonicalTypeUnqualified()
      .getAs<FunctionProtoType>();
}

/// Returns the "extra-canonicalized" return type, which discards
/// qualifiers on the return type.  Codegen doesn't care about them,
/// and it makes ABI code a little easier to be able to assume that
/// all parameter and return types are top-level unqualified.
static CanQualType GetReturnType(QualType RetTy) {
  return RetTy->getCanonicalTypeUnqualified();
}

/// Arrange the argument and result information for a value of the given
/// unprototyped freestanding function type.
const CGFunctionInfo &
CodeGenTypes::arrangeFreeFunctionType(CanQual<FunctionNoProtoType> FTNP) {
  // When translating an unprototyped function type, always use a
  // variadic type.
  return arrangeLLVMFunctionInfo(FTNP->getReturnType().getUnqualifiedType(),
                                 FnInfoOpts::None, {}, FTNP->getExtInfo(), {},
                                 RequiredArgs(0), /*ABIInfoFD=*/nullptr);
}

static void addExtParameterInfosForCall(
    llvm::SmallVectorImpl<FunctionProtoType::ExtParameterInfo> &paramInfos,
    const FunctionProtoType *proto, unsigned prefixArgs, unsigned totalArgs) {
  assert(proto->hasExtParameterInfos());
  assert(paramInfos.size() <= prefixArgs);
  assert(proto->getNumParams() + prefixArgs <= totalArgs);

  paramInfos.reserve(totalArgs);

  // Add default infos for any prefix args that don't already have infos.
  paramInfos.resize(prefixArgs);

  // Add infos for the prototype.
  for (const auto &ParamInfo : proto->getExtParameterInfos()) {
    paramInfos.push_back(ParamInfo);
    // pass_object_size params have no parameter info.
    if (ParamInfo.hasPassObjectSize())
      paramInfos.emplace_back();
  }

  assert(paramInfos.size() <= totalArgs &&
         "Did we forget to insert pass_object_size args?");
  // Add default infos for the variadic and/or suffix arguments.
  paramInfos.resize(totalArgs);
}

/// Adds the formal parameters in FPT to the given prefix. If any parameter in
/// FPT has pass_object_size attrs, then we'll add parameters for those, too.
static void appendParameterTypes(
    const CodeGenTypes &CGT, SmallVectorImpl<CanQualType> &prefix,
    SmallVectorImpl<FunctionProtoType::ExtParameterInfo> &paramInfos,
    CanQual<FunctionProtoType> FPT) {
  // Fast path: don't touch param info if we don't need to.
  if (!FPT->hasExtParameterInfos()) {
    assert(paramInfos.empty() &&
           "We have paramInfos, but the prototype doesn't?");
    prefix.append(FPT->param_type_begin(), FPT->param_type_end());
    return;
  }

  unsigned PrefixSize = prefix.size();
  // In the vast majority of cases, we'll have precisely FPT->getNumParams()
  // parameters; the only thing that can change this is the presence of
  // pass_object_size. So, we preallocate for the common case.
  prefix.reserve(prefix.size() + FPT->getNumParams());

  auto ExtInfos = FPT->getExtParameterInfos();
  assert(ExtInfos.size() == FPT->getNumParams());
  for (unsigned I = 0, E = FPT->getNumParams(); I != E; ++I) {
    prefix.push_back(FPT->getParamType(I));
    if (ExtInfos[I].hasPassObjectSize())
      prefix.push_back(CGT.getContext().getCanonicalSizeType());
  }

  addExtParameterInfosForCall(paramInfos, FPT.getTypePtr(), PrefixSize,
                              prefix.size());
}

using ExtParameterInfoList =
    SmallVector<FunctionProtoType::ExtParameterInfo, 16>;

/// Arrange the LLVM function layout for a value of the given function
/// type, on top of any implicit parameters already stored.
static const CGFunctionInfo &
arrangeLLVMFunctionInfo(CodeGenTypes &CGT, bool instanceMethod,
                        SmallVectorImpl<CanQualType> &prefix,
                        CanQual<FunctionProtoType> FTP) {
  ExtParameterInfoList paramInfos;
  RequiredArgs Required = RequiredArgs::forPrototypePlus(FTP, prefix.size());
  appendParameterTypes(CGT, prefix, paramInfos, FTP);
  CanQualType resultType = FTP->getReturnType().getUnqualifiedType();

  FnInfoOpts opts =
      instanceMethod ? FnInfoOpts::IsInstanceMethod : FnInfoOpts::None;
  return CGT.arrangeLLVMFunctionInfo(resultType, opts, prefix,
                                     FTP->getExtInfo(), paramInfos, Required,
                                     /*ABIInfoFD=*/nullptr);
}

using CanQualTypeList = SmallVector<CanQualType, 16>;

/// Arrange the argument and result information for a value of the
/// given freestanding function type.
const CGFunctionInfo &
CodeGenTypes::arrangeFreeFunctionType(CanQual<FunctionProtoType> FTP) {
  CanQualTypeList argTypes;
  return ::arrangeLLVMFunctionInfo(*this, /*instanceMethod=*/false, argTypes,
                                   FTP);
}

CallingConv CodeGen::getCallingConventionForDecl(const ObjCMethodDecl *D,
                                               bool IsTargetDefaultMSABI) {
  // Set the appropriate calling convention for the Function.
  if (D->hasAttr<StdCallAttr>())
    return CC_X86StdCall;

  if (D->hasAttr<FastCallAttr>())
    return CC_X86FastCall;

  if (D->hasAttr<RegCallAttr>())
    return CC_X86RegCall;

  if (D->hasAttr<ThisCallAttr>())
    return CC_X86ThisCall;

  if (D->hasAttr<VectorCallAttr>())
    return CC_X86VectorCall;

  if (D->hasAttr<PascalAttr>())
    return CC_X86Pascal;

  if (PcsAttr *PCS = D->getAttr<PcsAttr>())
    return (PCS->getPCS() == PcsAttr::AAPCS ? CC_AAPCS : CC_AAPCS_VFP);

  if (D->hasAttr<AArch64VectorPcsAttr>())
    return CC_AArch64VectorCall;

  if (D->hasAttr<AArch64SVEPcsAttr>())
    return CC_AArch64SVEPCS;

  if (D->hasAttr<DeviceKernelAttr>())
    return CC_DeviceKernel;

  if (D->hasAttr<IntelOclBiccAttr>())
    return CC_IntelOclBicc;

  if (D->hasAttr<MSABIAttr>())
    return IsTargetDefaultMSABI ? CC_C : CC_Win64;

  if (D->hasAttr<SysVABIAttr>())
    return IsTargetDefaultMSABI ? CC_X86_64SysV : CC_C;

  if (D->hasAttr<PreserveMostAttr>())
    return CC_PreserveMost;

  if (D->hasAttr<PreserveAllAttr>())
    return CC_PreserveAll;

  if (D->hasAttr<M68kRTDAttr>())
    return CC_M68kRTD;

  if (D->hasAttr<PreserveNoneAttr>())
    return CC_PreserveNone;

  if (D->hasAttr<RISCVVectorCCAttr>())
    return CC_RISCVVectorCall;

  if (RISCVVLSCCAttr *PCS = D->getAttr<RISCVVLSCCAttr>()) {
    switch (PCS->getVectorWidth()) {
    default:
      llvm_unreachable("Invalid RISC-V VLS ABI VLEN");
#define CC_VLS_CASE(ABI_VLEN)                                                  \
  case ABI_VLEN:                                                               \
    return CC_RISCVVLSCall_##ABI_VLEN;
      CC_VLS_CASE(32)
      CC_VLS_CASE(64)
      CC_VLS_CASE(128)
      CC_VLS_CASE(256)
      CC_VLS_CASE(512)
      CC_VLS_CASE(1024)
      CC_VLS_CASE(2048)
      CC_VLS_CASE(4096)
      CC_VLS_CASE(8192)
      CC_VLS_CASE(16384)
      CC_VLS_CASE(32768)
      CC_VLS_CASE(65536)
#undef CC_VLS_CASE
    }
  }

  return CC_C;
}

/// Arrange the argument and result information for a call to an
/// unknown C++ non-static member function of the given abstract type.
/// (A null RD means we don't have any meaningful "this" argument type,
///  so fall back to a generic pointer type).
/// The member function must be an ordinary function, i.e. not a
/// constructor or destructor.
const CGFunctionInfo &
CodeGenTypes::arrangeCXXMethodType(const CXXRecordDecl *RD,
                                   const FunctionProtoType *FTP,
                                   const CXXMethodDecl *MD) {
  CanQualTypeList argTypes;

  // Add the 'this' pointer.
  argTypes.push_back(DeriveThisType(RD, MD));
  auto CanonicalFTP =
      FTP->getCanonicalTypeUnqualified().getAs<FunctionProtoType>();
  ExtParameterInfoList paramInfos;
  RequiredArgs required = RequiredArgs::forPrototypePlus(
      CanonicalFTP.getTypePtr(), argTypes.size());
  appendParameterTypes(*this, argTypes, paramInfos, CanonicalFTP);
  return arrangeLLVMFunctionInfo(
      CanonicalFTP->getReturnType().getUnqualifiedType(),
      FnInfoOpts::IsInstanceMethod, argTypes, CanonicalFTP->getExtInfo(),
      paramInfos, required, MD);
}

/// Set calling convention for CUDA/HIP kernel.
static void setCUDAKernelCallingConvention(CanQualType &FTy, CodeGenModule &CGM,
                                           const FunctionDecl *FD) {
  if (FD->hasAttr<CUDAGlobalAttr>()) {
    const FunctionType *FT = FTy->getAs<FunctionType>();
    CGM.getTargetCodeGenInfo().setCUDAKernelCallingConvention(FT);
    FTy = FT->getCanonicalTypeUnqualified();
  }
}

/// Arrange the argument and result information for a declaration or
/// definition of the given C++ non-static member function.  The
/// member function must be an ordinary function, i.e. not a
/// constructor or destructor.
const CGFunctionInfo &
CodeGenTypes::arrangeCXXMethodDeclaration(const CXXMethodDecl *MD) {
  assert(!isa<CXXConstructorDecl>(MD) && "wrong method for constructors!");
  assert(!isa<CXXDestructorDecl>(MD) && "wrong method for destructors!");

  CanQualType FT = GetFormalType(MD).getAs<Type>();
  setCUDAKernelCallingConvention(FT, CGM, MD);
  auto prototype = FT.getAs<FunctionProtoType>();

  if (MD->isImplicitObjectMemberFunction()) {
    // The abstract case is perfectly fine.
    const CXXRecordDecl *ThisType =
        getCXXABI().getThisArgumentTypeForMethod(MD);
    return arrangeCXXMethodType(ThisType, prototype.getTypePtr(), MD);
  }

  CanQualTypeList argTypes;
  ExtParameterInfoList paramInfos;
  appendParameterTypes(*this, argTypes, paramInfos, prototype);
  return arrangeLLVMFunctionInfo(
      prototype->getReturnType().getUnqualifiedType(), FnInfoOpts::None,
      argTypes, prototype->getExtInfo(), paramInfos,
      RequiredArgs::forPrototypePlus(prototype.getTypePtr(), 0), MD);
}

bool CodeGenTypes::inheritingCtorHasParams(
    const InheritedConstructor &Inherited, CXXCtorType Type) {
  // Parameters are unnecessary if we're constructing a base class subobject
  // and the inherited constructor lives in a virtual base.
  return Type == Ctor_Complete ||
         !Inherited.getShadowDecl()->constructsVirtualBase() ||
         !Target.getCXXABI().hasConstructorVariants();
}

const CGFunctionInfo &
CodeGenTypes::arrangeCXXStructorDeclaration(GlobalDecl GD) {
  auto *MD = cast<CXXMethodDecl>(GD.getDecl());

  CanQualTypeList argTypes;
  ExtParameterInfoList paramInfos;

  const CXXRecordDecl *ThisType = getCXXABI().getThisArgumentTypeForMethod(GD);
  argTypes.push_back(DeriveThisType(ThisType, MD));

  bool PassParams = true;

  if (auto *CD = dyn_cast<CXXConstructorDecl>(MD)) {
    // A base class inheriting constructor doesn't get forwarded arguments
    // needed to construct a virtual base (or base class thereof).
    if (auto Inherited = CD->getInheritedConstructor())
      PassParams = inheritingCtorHasParams(Inherited, GD.getCtorType());
  }

  CanQual<FunctionProtoType> FTP = GetFormalType(MD);

  // Add the formal parameters.
  if (PassParams)
    appendParameterTypes(*this, argTypes, paramInfos, FTP);

  CGCXXABI::AddedStructorArgCounts AddedArgs =
      getCXXABI().buildStructorSignature(GD, argTypes);
  if (!paramInfos.empty()) {
    // Note: prefix implies after the first param.
    if (AddedArgs.Prefix)
      paramInfos.insert(paramInfos.begin() + 1, AddedArgs.Prefix,
                        FunctionProtoType::ExtParameterInfo{});
    if (AddedArgs.Suffix)
      paramInfos.append(AddedArgs.Suffix,
                        FunctionProtoType::ExtParameterInfo{});
  }

  RequiredArgs required =
      (PassParams && MD->isVariadic() ? RequiredArgs(argTypes.size())
                                      : RequiredArgs::All);

  FunctionType::ExtInfo extInfo = FTP->getExtInfo();
  CanQualType resultType = getCXXABI().HasThisReturn(GD) ? argTypes.front()
                           : getCXXABI().hasMostDerivedReturn(GD)
                               ? CGM.getContext().VoidPtrTy
                               : Context.VoidTy;
  return arrangeLLVMFunctionInfo(resultType, FnInfoOpts::IsInstanceMethod,
                                 argTypes, extInfo, paramInfos, required, MD);
}

static CanQualTypeList getArgTypesForCall(ASTContext &ctx,
                                          const CallArgList &args) {
  CanQualTypeList argTypes;
  for (auto &arg : args)
    argTypes.push_back(ctx.getCanonicalParamType(arg.Ty));
  return argTypes;
}

static CanQualTypeList getArgTypesForDeclaration(ASTContext &ctx,
                                                 const FunctionArgList &args) {
  CanQualTypeList argTypes;
  for (auto &arg : args)
    argTypes.push_back(ctx.getCanonicalParamType(arg->getType()));
  return argTypes;
}

static ExtParameterInfoList
getExtParameterInfosForCall(const FunctionProtoType *proto, unsigned prefixArgs,
                            unsigned totalArgs) {
  ExtParameterInfoList result;
  if (proto->hasExtParameterInfos()) {
    addExtParameterInfosForCall(result, proto, prefixArgs, totalArgs);
  }
  return result;
}

/// Arrange a call to a C++ method, passing the given arguments.
///
/// ExtraPrefixArgs is the number of ABI-specific args passed after the `this`
/// parameter.
/// ExtraSuffixArgs is the number of ABI-specific args passed at the end of
/// args.
/// PassProtoArgs indicates whether `args` has args for the parameters in the
/// given CXXConstructorDecl.
const CGFunctionInfo &CodeGenTypes::arrangeCXXConstructorCall(
    const CallArgList &args, const CXXConstructorDecl *D, CXXCtorType CtorKind,
    unsigned ExtraPrefixArgs, unsigned ExtraSuffixArgs,
    const FunctionDecl *ABIInfoFD, bool PassProtoArgs) {
  CanQualTypeList ArgTypes;
  for (const auto &Arg : args)
    ArgTypes.push_back(Context.getCanonicalParamType(Arg.Ty));

  // +1 for implicit this, which should always be args[0].
  unsigned TotalPrefixArgs = 1 + ExtraPrefixArgs;

  CanQual<FunctionProtoType> FPT = GetFormalType(D);
  RequiredArgs Required = PassProtoArgs
                              ? RequiredArgs::forPrototypePlus(
                                    FPT, TotalPrefixArgs + ExtraSuffixArgs)
                              : RequiredArgs::All;

  GlobalDecl GD(D, CtorKind);
  CanQualType ResultType = getCXXABI().HasThisReturn(GD) ? ArgTypes.front()
                           : getCXXABI().hasMostDerivedReturn(GD)
                               ? CGM.getContext().VoidPtrTy
                               : Context.VoidTy;

  FunctionType::ExtInfo Info = FPT->getExtInfo();
  ExtParameterInfoList ParamInfos;
  // If the prototype args are elided, we should only have ABI-specific args,
  // which never have param info.
  if (PassProtoArgs && FPT->hasExtParameterInfos()) {
    // ABI-specific suffix arguments are treated the same as variadic arguments.
    addExtParameterInfosForCall(ParamInfos, FPT.getTypePtr(), TotalPrefixArgs,
                                ArgTypes.size());
  }

  return arrangeLLVMFunctionInfo(ResultType, FnInfoOpts::IsInstanceMethod,
                                 ArgTypes, Info, ParamInfos, Required,
                                 ABIInfoFD);
}

/// Arrange the argument and result information for the declaration or
/// definition of the given function.
const CGFunctionInfo &
CodeGenTypes::arrangeFunctionDeclaration(const GlobalDecl GD) {
  const FunctionDecl *FD = cast<FunctionDecl>(GD.getDecl());
  if (const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(FD))
    if (MD->isImplicitObjectMemberFunction())
      return arrangeCXXMethodDeclaration(MD);

  CanQualType FTy = FD->getType()->getCanonicalTypeUnqualified();

  assert(isa<FunctionType>(FTy));
  setCUDAKernelCallingConvention(FTy, CGM, FD);

  if (DeviceKernelAttr::isOpenCLSpelling(FD->getAttr<DeviceKernelAttr>()) &&
      GD.getKernelReferenceKind() == KernelReferenceKind::Stub) {
    const FunctionType *FT = FTy->getAs<FunctionType>();
    CGM.getTargetCodeGenInfo().setOCLKernelStubCallingConvention(FT);
    FTy = FT->getCanonicalTypeUnqualified();
  }

  // When declaring a function without a prototype, always use a
  // non-variadic type.
  if (CanQual<FunctionNoProtoType> noProto = FTy.getAs<FunctionNoProtoType>()) {
    return arrangeLLVMFunctionInfo(noProto->getReturnType(), FnInfoOpts::None,
                                   {}, noProto->getExtInfo(), {},
                                   RequiredArgs::All, FD);
  }

  CanQual<FunctionProtoType> FTP = FTy.castAs<FunctionProtoType>();
  CanQualTypeList argTypes;
  ExtParameterInfoList paramInfos;
  appendParameterTypes(*this, argTypes, paramInfos, FTP);
  return arrangeLLVMFunctionInfo(FTP->getReturnType().getUnqualifiedType(),
                                 FnInfoOpts::None, argTypes, FTP->getExtInfo(),
                                 paramInfos,
                                 RequiredArgs::forPrototypePlus(FTP, 0), FD);
}

/// Arrange the argument and result information for the declaration or
/// definition of an Objective-C method.
const CGFunctionInfo &
CodeGenTypes::arrangeObjCMethodDeclaration(const ObjCMethodDecl *MD) {
  // It happens that this is the same as a call with no optional
  // arguments, except also using the formal 'self' type.
  return arrangeObjCMessageSendSignature(MD, MD->getSelfDecl()->getType());
}

/// Arrange the argument and result information for the function type
/// through which to perform a send to the given Objective-C method,
/// using the given receiver type.  The receiver type is not always
/// the 'self' type of the method or even an Objective-C pointer type.
/// This is *not* the right method for actually performing such a
/// message send, due to the possibility of optional arguments.
const CGFunctionInfo &
CodeGenTypes::arrangeObjCMessageSendSignature(const ObjCMethodDecl *MD,
                                              QualType receiverType) {
  CanQualTypeList argTys;
  ExtParameterInfoList extParamInfos(MD->isDirectMethod() ? 1 : 2);
  argTys.push_back(Context.getCanonicalParamType(receiverType));
  if (!MD->isDirectMethod())
    argTys.push_back(Context.getCanonicalParamType(Context.getObjCSelType()));
  for (const auto *I : MD->parameters()) {
    argTys.push_back(Context.getCanonicalParamType(I->getType()));
    auto extParamInfo = FunctionProtoType::ExtParameterInfo().withIsNoEscape(
        I->hasAttr<NoEscapeAttr>());
    extParamInfos.push_back(extParamInfo);
  }

  FunctionType::ExtInfo einfo;
  bool IsTargetDefaultMSABI =
      getContext().getTargetInfo().getTriple().isOSWindows() ||
      getContext().getTargetInfo().getTriple().isUEFI();
  einfo = einfo.withCallingConv(
      getCallingConventionForDecl(MD, IsTargetDefaultMSABI));

  if (getContext().getLangOpts().ObjCAutoRefCount &&
      MD->hasAttr<NSReturnsRetainedAttr>())
    einfo = einfo.withProducesResult(true);

  RequiredArgs required =
      (MD->isVariadic() ? RequiredArgs(argTys.size()) : RequiredArgs::All);

  return arrangeLLVMFunctionInfo(GetReturnType(MD->getReturnType()),
                                 FnInfoOpts::None, argTys, einfo, extParamInfos,
                                 required, /*ABIInfoFD=*/nullptr);
}

const CGFunctionInfo &
CodeGenTypes::arrangeUnprototypedObjCMessageSend(QualType returnType,
                                                 const CallArgList &args) {
  CanQualTypeList argTypes = getArgTypesForCall(Context, args);
  FunctionType::ExtInfo einfo;

  return arrangeLLVMFunctionInfo(GetReturnType(returnType), FnInfoOpts::None,
                                 argTypes, einfo, {}, RequiredArgs::All,
                                 nullptr);
}

const CGFunctionInfo &CodeGenTypes::arrangeGlobalDeclaration(GlobalDecl GD) {
  // FIXME: Do we need to handle ObjCMethodDecl?
  if (isa<CXXConstructorDecl>(GD.getDecl()) ||
      isa<CXXDestructorDecl>(GD.getDecl()))
    return arrangeCXXStructorDeclaration(GD);

  return arrangeFunctionDeclaration(GD);
}

/// Arrange a thunk that takes 'this' as the first parameter followed by
/// varargs.  Return a void pointer, regardless of the actual return type.
/// The body of the thunk will end in a musttail call to a function of the
/// correct type, and the caller will bitcast the function to the correct
/// prototype.
const CGFunctionInfo &
CodeGenTypes::arrangeUnprototypedMustTailThunk(const CXXMethodDecl *MD) {
  assert(MD->isVirtual() && "only methods have thunks");
  CanQual<FunctionProtoType> FTP = GetFormalType(MD);
  CanQualType ArgTys[] = {DeriveThisType(MD->getParent(), MD)};
  return arrangeLLVMFunctionInfo(Context.VoidTy, FnInfoOpts::None, ArgTys,
                                 FTP->getExtInfo(), {}, RequiredArgs(1), MD);
}

const CGFunctionInfo &
CodeGenTypes::arrangeMSCtorClosure(const CXXConstructorDecl *CD,
                                   CXXCtorType CT) {
  assert(CT == Ctor_CopyingClosure || CT == Ctor_DefaultClosure);

  CanQual<FunctionProtoType> FTP = GetFormalType(CD);
  SmallVector<CanQualType, 2> ArgTys;
  const CXXRecordDecl *RD = CD->getParent();
  ArgTys.push_back(DeriveThisType(RD, CD));
  if (CT == Ctor_CopyingClosure)
    ArgTys.push_back(*FTP->param_type_begin());
  if (RD->getNumVBases() > 0)
    ArgTys.push_back(Context.IntTy);
  CallingConv CC = Context.getDefaultCallingConvention(
      /*IsVariadic=*/false, /*IsCXXMethod=*/true);
  return arrangeLLVMFunctionInfo(Context.VoidTy, FnInfoOpts::IsInstanceMethod,
                                 ArgTys, FunctionType::ExtInfo(CC), {},
                                 RequiredArgs::All, /*ABIInfoFD=*/nullptr);
}

/// Arrange a call as unto a free function, except possibly with an
/// additional number of formal parameters considered required.
static const CGFunctionInfo &
arrangeFreeFunctionLikeCall(CodeGenTypes &CGT, CodeGenModule &CGM,
                            const CallArgList &args, const FunctionType *fnType,
                            unsigned numExtraRequiredArgs, bool chainCall,
                            const FunctionDecl *ABIInfoFD) {
  assert(args.size() >= numExtraRequiredArgs);

  ExtParameterInfoList paramInfos;

  // In most cases, there are no optional arguments.
  RequiredArgs required = RequiredArgs::All;

  // If we have a variadic prototype, the required arguments are the
  // extra prefix plus the arguments in the prototype.
  if (const FunctionProtoType *proto = dyn_cast<FunctionProtoType>(fnType)) {
    if (proto->isVariadic())
      required = RequiredArgs::forPrototypePlus(proto, numExtraRequiredArgs);

    if (proto->hasExtParameterInfos())
      addExtParameterInfosForCall(paramInfos, proto, numExtraRequiredArgs,
                                  args.size());

  // If we don't have a prototype at all, but we're supposed to
  // explicitly use the variadic convention for unprototyped calls,
  // treat all of the arguments as required but preserve the nominal
  // possibility of variadics.
  } else if (CGM.getTargetCodeGenInfo().isNoProtoCallVariadic(
                 args, cast<FunctionNoProtoType>(fnType))) {
    required = RequiredArgs(args.size());
  }

  CanQualTypeList argTypes;
  for (const auto &arg : args)
    argTypes.push_back(CGT.getContext().getCanonicalParamType(arg.Ty));
  FnInfoOpts opts = chainCall ? FnInfoOpts::IsChainCall : FnInfoOpts::None;
  return CGT.arrangeLLVMFunctionInfo(GetReturnType(fnType->getReturnType()),
                                     opts, argTypes, fnType->getExtInfo(),
                                     paramInfos, required, ABIInfoFD);
}

/// Figure out the rules for calling a function with the given formal
/// type using the given arguments.  The arguments are necessary
/// because the function might be unprototyped, in which case it's
/// target-dependent in crazy ways.
const CGFunctionInfo &CodeGenTypes::arrangeFreeFunctionCall(
    const CallArgList &args, const FunctionType *fnType, bool chainCall,
    const FunctionDecl *ABIInfoFD) {
  return arrangeFreeFunctionLikeCall(*this, CGM, args, fnType,
                                     chainCall ? 1 : 0, chainCall, ABIInfoFD);
}

/// A block function is essentially a free function with an
/// extra implicit argument.
const CGFunctionInfo &
CodeGenTypes::arrangeBlockFunctionCall(const CallArgList &args,
                                       const FunctionType *fnType) {
  // FIXME: Pass the enclosing function's ABI information so block calls use
  // the caller's target features.
  return arrangeFreeFunctionLikeCall(*this, CGM, args, fnType, 1,
                                     /*chainCall=*/false, nullptr);
}

const CGFunctionInfo &
CodeGenTypes::arrangeBlockFunctionDeclaration(const FunctionProtoType *proto,
                                              const FunctionArgList &params) {
  ExtParameterInfoList paramInfos =
      getExtParameterInfosForCall(proto, 1, params.size());
  CanQualTypeList argTypes = getArgTypesForDeclaration(Context, params);

  // FIXME: Use the block's target features when arranging its invoke function.
  return arrangeLLVMFunctionInfo(
      GetReturnType(proto->getReturnType()), FnInfoOpts::None, argTypes,
      proto->getExtInfo(), paramInfos, RequiredArgs::forPrototypePlus(proto, 1),
      /*ABIInfoFD=*/nullptr);
}

const CGFunctionInfo &
CodeGenTypes::arrangeBuiltinFunctionCall(QualType resultType,
                                         const CallArgList &args) {
  CanQualTypeList argTypes;
  for (const auto &Arg : args)
    argTypes.push_back(Context.getCanonicalParamType(Arg.Ty));
  return arrangeLLVMFunctionInfo(GetReturnType(resultType), FnInfoOpts::None,
                                 argTypes, FunctionType::ExtInfo(),
                                 /*paramInfos=*/{}, RequiredArgs::All, nullptr);
}

const CGFunctionInfo &
CodeGenTypes::arrangeBuiltinFunctionDeclaration(QualType resultType,
                                                const FunctionArgList &args) {
  CanQualTypeList argTypes = getArgTypesForDeclaration(Context, args);

  return arrangeLLVMFunctionInfo(GetReturnType(resultType), FnInfoOpts::None,
                                 argTypes, FunctionType::ExtInfo(), {},
                                 RequiredArgs::All, /*ABIInfoFD=*/nullptr);
}

const CGFunctionInfo &CodeGenTypes::arrangeBuiltinFunctionDeclaration(
    CanQualType resultType, ArrayRef<CanQualType> argTypes) {
  return arrangeLLVMFunctionInfo(resultType, FnInfoOpts::None, argTypes,
                                 FunctionType::ExtInfo(), {}, RequiredArgs::All,
                                 /*ABIInfoFD=*/nullptr);
}

const CGFunctionInfo &CodeGenTypes::arrangeDeviceKernelCallerDeclaration(
    QualType resultType, const FunctionArgList &args) {
  CanQualTypeList argTypes = getArgTypesForDeclaration(Context, args);

  return arrangeLLVMFunctionInfo(GetReturnType(resultType), FnInfoOpts::None,
                                 argTypes,
                                 FunctionType::ExtInfo(CC_DeviceKernel),
                                 /*paramInfos=*/{}, RequiredArgs::All,
                                 /*ABIInfoFD=*/nullptr);
}

/// Arrange a call to a C++ method, passing the given arguments.
///
/// numPrefixArgs is the number of ABI-specific prefix arguments we have. It
/// does not count `this`.
const CGFunctionInfo &CodeGenTypes::arrangeCXXMethodCall(
    const CallArgList &args, const FunctionProtoType *proto,
    RequiredArgs required, unsigned numPrefixArgs,
    const FunctionDecl *ABIInfoFD) {
  assert(numPrefixArgs + 1 <= args.size() &&
         "Emitting a call with less args than the required prefix?");
  // Add one to account for `this`. It's a bit awkward here, but we don't count
  // `this` in similar places elsewhere.
  ExtParameterInfoList paramInfos =
      getExtParameterInfosForCall(proto, numPrefixArgs + 1, args.size());

  CanQualTypeList argTypes = getArgTypesForCall(Context, args);

  FunctionType::ExtInfo info = proto->getExtInfo();
  return arrangeLLVMFunctionInfo(GetReturnType(proto->getReturnType()),
                                 FnInfoOpts::IsInstanceMethod, argTypes, info,
                                 paramInfos, required, ABIInfoFD);
}

const CGFunctionInfo &CodeGenTypes::arrangeNullaryFunction() {
  return arrangeLLVMFunctionInfo(getContext().VoidTy, FnInfoOpts::None, {},
                                 FunctionType::ExtInfo(), {}, RequiredArgs::All,
                                 /*ABIInfoFD=*/nullptr);
}

const CGFunctionInfo &CodeGenTypes::arrangeCall(const CGFunctionInfo &signature,
                                                const CallArgList &args,
                                                const FunctionDecl *ABIInfoFD) {
  assert(signature.arg_size() <= args.size());
  unsigned X86ABIAVXLevel =
      CGM.getABIInfo().getX86ABIAVXLevel(ABIInfoFD, signature.getExtInfo());
  if (signature.arg_size() == args.size() &&
      signature.getX86ABIAVXLevel() == X86ABIAVXLevel)
    return signature;

  ExtParameterInfoList paramInfos;
  auto sigParamInfos = signature.getExtParameterInfos();
  if (!sigParamInfos.empty()) {
    paramInfos.append(sigParamInfos.begin(), sigParamInfos.end());
    paramInfos.resize(args.size());
  }

  CanQualTypeList argTypes = getArgTypesForCall(Context, args);

  assert(signature.getRequiredArgs().allowsOptionalArgs());
  FnInfoOpts opts = FnInfoOpts::None;
  if (signature.isInstanceMethod())
    opts |= FnInfoOpts::IsInstanceMethod;
  if (signature.isChainCall())
    opts |= FnInfoOpts::IsChainCall;
  if (signature.isDelegateCall())
    opts |= FnInfoOpts::IsDelegateCall;

  const CGFunctionInfo *newFI = findOrInsertCGFunctionInfo(
      signature.isInstanceMethod(), signature.isChainCall(),
      signature.isDelegateCall(), X86ABIAVXLevel, signature.getExtInfo(),
      paramInfos, signature.getRequiredArgs(), signature.getReturnType(),
      argTypes);
  return *newFI;
}

namespace clang {
namespace CodeGen {
void computeSPIRKernelABIInfo(CodeGenModule &CGM, CGFunctionInfo &FI);
} // namespace CodeGen
} // namespace clang

#ifndef NDEBUG
static const char *abiKindToString(ABIArgInfo::Kind K) {
  switch (K) {
  case ABIArgInfo::Direct:
    return "Direct";
  case ABIArgInfo::Extend:
    return "Extend";
  case ABIArgInfo::Indirect:
    return "Indirect";
  case ABIArgInfo::IndirectAliased:
    return "IndirectAliased";
  case ABIArgInfo::Ignore:
    return "Ignore";
  case ABIArgInfo::Expand:
    return "Expand";
  case ABIArgInfo::CoerceAndExpand:
    return "CoerceAndExpand";
  case ABIArgInfo::TargetSpecific:
    return "TargetSpecific";
  case ABIArgInfo::InAlloca:
    return "InAlloca";
  }
  llvm_unreachable("Unknown kind");
}
#endif

void CodeGenModule::computeABIInfoUsingLib(CGFunctionInfo &FI) {
  SmallVector<const llvm::abi::Type *> MappedArgTypes;
  MappedArgTypes.reserve(FI.arg_size());
  for (const auto &Arg : FI.arguments())
    MappedArgTypes.push_back(AbiMapper->convertType(Arg.type));

  RequiredArgs Required = FI.getRequiredArgs();
  llvm::abi::RequiredArgs AbiRequired = llvm::abi::RequiredArgs::All;
  if (Required.allowsOptionalArgs())
    AbiRequired = llvm::abi::RequiredArgs(Required.getNumRequiredArgs());

  auto AbiFI = llvm::abi::FunctionInfo::create(
      FI.getCallingConvention(), AbiMapper->convertType(FI.getReturnType()),
      MappedArgTypes, AbiRequired);

  getLLVMABITargetInfo(AbiMapper->getTypeBuilder()).computeInfo(*AbiFI);

#ifndef NDEBUG
  // With assertions enabled, also compute info using Clang ABI logic,
  // so we can ensure the results are consistent.
  getABIInfo().computeInfo(FI);

  auto ConvertABIArgInfo = [&](ABIArgInfo &Target,
                               const llvm::abi::ArgInfo &AbiInfo, QualType Type,
                               int ArgNo) {
    auto Check = [&](bool Cond, llvm::function_ref<void()> MessageFn) {
      if (Cond)
        return;
      if (ArgNo == -1)
        llvm::dbgs() << "For return value of type ";
      else
        llvm::dbgs() << "For argument " << ArgNo << " of type ";
      llvm::dbgs() << Type << ": ";
      MessageFn();
      llvm::dbgs() << "\n";
      abort();
    };
    auto CheckSimple = [&](auto TargetVal, auto ResVal, StringRef What) {
      Check(TargetVal == ResVal, [&]() {
        llvm::dbgs() << What << " mismatch (expected: " << TargetVal
                     << ", given: " << ResVal << ")";
      });
    };

    ABIArgInfo Res = convertABIArgInfo(AbiInfo, Type);
    Check(Target.getKind() == Res.getKind(), [&]() {
      llvm::dbgs() << "Kind mismatch (expected: "
                   << abiKindToString(Target.getKind())
                   << ", given: " << abiKindToString(Res.getKind()) << ")";
    });

    if (Res.canHaveCoerceToType()) {
      // Normalize nullptr types.
      llvm::Type *TargetType = Target.getCoerceToType();
      llvm::Type *ResType = Res.getCoerceToType();
      if (!TargetType)
        TargetType = getTypes().ConvertType(Type);
      if (!ResType)
        ResType = getTypes().ConvertType(Type);

      Check(TargetType == ResType, [&]() {
        llvm::dbgs() << "CoerceToType mismatch (expected: " << *TargetType
                     << ", given: " << *ResType << ")";
      });
    }

    switch (Res.getKind()) {
    case ABIArgInfo::Extend:
      CheckSimple(Target.isSignExt(), Res.isSignExt(), "SignExt");
      CheckSimple(Target.isZeroExt(), Res.isZeroExt(), "ZeroExt");
      [[fallthrough]];
    case ABIArgInfo::Direct:
      CheckSimple(Target.getDirectAlign(), Res.getDirectAlign(), "DirectAlign");
      CheckSimple(Target.getDirectOffset(), Res.getDirectOffset(),
                  "DirectOffset");
      break;
    case ABIArgInfo::Indirect:
      CheckSimple(Target.getIndirectByVal(), Res.getIndirectByVal(),
                  "IndirectByVal");
      [[fallthrough]];
    case ABIArgInfo::IndirectAliased:
      CheckSimple(Target.getIndirectAddrSpace(), Res.getIndirectAddrSpace(),
                  "IndirectAddrSpace");
      CheckSimple(Target.getIndirectRealign(), Res.getIndirectRealign(),
                  "IndirectRealign");
      Check(Target.getIndirectAlign() == Res.getIndirectAlign(), [&]() {
        llvm::dbgs() << "IndirectAlign mismatch (expected: "
                     << Target.getIndirectAlign().getQuantity()
                     << ", given: " << Res.getIndirectAlign().getQuantity()
                     << ")";
      });
      break;
    default:
      break;
    }

    Target = Res;
  };
#else
  auto ConvertABIArgInfo =
      [&](ABIArgInfo &Target, const llvm::abi::ArgInfo &AbiInfo, QualType Type,
          int ArgNo) { Target = convertABIArgInfo(AbiInfo, Type); };
#endif

  ConvertABIArgInfo(FI.getReturnInfo(), AbiFI->getReturnInfo(),
                    FI.getReturnType(), -1);

  int ArgNo = 0;
  for (auto [CGArg, AbiArg] :
       llvm::zip_equal(FI.arguments(), AbiFI->arguments()))
    ConvertABIArgInfo(CGArg.info, AbiArg.Info, CGArg.type, ArgNo++);
}

ABIArgInfo CodeGenModule::convertABIArgInfo(const llvm::abi::ArgInfo &AbiInfo,
                                            QualType Type) {
  switch (AbiInfo.getKind()) {
  case llvm::abi::ArgInfo::Direct: {
    llvm::Type *CoercedType = nullptr;
    if (AbiInfo.getCoerceToType())
      CoercedType = AbiReverseMapper->convertType(AbiInfo.getCoerceToType());
    if (!CoercedType)
      CoercedType = getTypes().ConvertType(Type);
    return ABIArgInfo::getDirect(CoercedType, AbiInfo.getDirectOffset());
  }
  case llvm::abi::ArgInfo::Extend: {
    llvm::Type *CoercedType = nullptr;
    if (AbiInfo.getCoerceToType())
      CoercedType = AbiReverseMapper->convertType(AbiInfo.getCoerceToType());
    if (!CoercedType)
      CoercedType = getTypes().ConvertType(Type);
    // A transparent union is passed as its first field, so the extend keys off
    // that field's integral type, matching the classifier's
    // useFirstFieldIfTransparentUnion.  Passing the union type to
    // ABIArgInfo::getSignExtend would trip its integral-type assert.
    QualType ExtendType = useFirstFieldIfTransparentUnion(Type);
    if (AbiInfo.isSignExt())
      return ABIArgInfo::getSignExtend(ExtendType, CoercedType);
    if (AbiInfo.isZeroExt())
      return ABIArgInfo::getZeroExtend(ExtendType, CoercedType);
    return ABIArgInfo::getExtend(ExtendType, CoercedType);
  }
  case llvm::abi::ArgInfo::Indirect: {
    CharUnits Alignment =
        CharUnits::fromQuantity(AbiInfo.getIndirectAlign().value());
    return ABIArgInfo::getIndirect(Alignment, AbiInfo.getIndirectAddrSpace(),
                                   AbiInfo.getIndirectByVal(),
                                   AbiInfo.getIndirectRealign());
  }
  case llvm::abi::ArgInfo::Ignore:
    return ABIArgInfo::getIgnore();
  }
  llvm_unreachable("Unexpected llvm::abi::ArgInfo kind");
}

/// Arrange the argument and result information for an abstract value
/// of a given function type.  This is the method which all of the
/// above functions ultimately defer to.
const CGFunctionInfo &CodeGenTypes::arrangeLLVMFunctionInfo(
    CanQualType resultType, FnInfoOpts opts, ArrayRef<CanQualType> argTypes,
    FunctionType::ExtInfo info,
    ArrayRef<FunctionProtoType::ExtParameterInfo> paramInfos,
    RequiredArgs required, const FunctionDecl *ABIInfoFD) {
  assert(llvm::all_of(argTypes,
                      [](CanQualType T) { return T.isCanonicalAsParam(); }));

  // Lookup or create unique function info.
  llvm::FoldingSetNodeID ID;
  bool isInstanceMethod =
      (opts & FnInfoOpts::IsInstanceMethod) == FnInfoOpts::IsInstanceMethod;
  bool isChainCall =
      (opts & FnInfoOpts::IsChainCall) == FnInfoOpts::IsChainCall;
  bool isDelegateCall =
      (opts & FnInfoOpts::IsDelegateCall) == FnInfoOpts::IsDelegateCall;
  unsigned X86ABIAVXLevel = CGM.getABIInfo().getX86ABIAVXLevel(ABIInfoFD, info);

  const CGFunctionInfo *newFI = findOrInsertCGFunctionInfo(
      isInstanceMethod, isChainCall, isDelegateCall, X86ABIAVXLevel, info,
      paramInfos, required, resultType, argTypes);
  return *newFI;
}

CGFunctionInfo *CodeGenTypes::findOrInsertCGFunctionInfo(
    bool isInstanceMethod, bool isChainCall, bool isDelegateCall,
    unsigned X86ABIAVXLevel, const FunctionType::ExtInfo &info,
    ArrayRef<FunctionProtoType::ExtParameterInfo> paramInfos,
    RequiredArgs required, CanQualType resultType,
    ArrayRef<CanQualType> argTypes) {
  llvm::FoldingSetNodeID ID;
  CGFunctionInfo::Profile(ID, isInstanceMethod, isChainCall, isDelegateCall,
                          X86ABIAVXLevel, info, paramInfos, required,
                          resultType, argTypes);

  void *insertPos = nullptr;
  CGFunctionInfo *FI = FunctionInfos.FindNodeOrInsertPos(ID, insertPos);
  if (FI)
    return FI;

  unsigned CC = ClangCallConvToLLVMCallConv(info.getCC());

  // Construct the function info.  We co-allocate the ArgInfos.
  FI = CGFunctionInfo::create(CC, isInstanceMethod, isChainCall, isDelegateCall,
                              X86ABIAVXLevel, info, paramInfos, resultType,
                              argTypes, required);
  FunctionInfos.InsertNode(FI, insertPos);

  bool inserted = FunctionsBeingProcessed.insert(FI).second;
  (void)inserted;
  assert(inserted && "Recursively being processed?");

  // Compute ABI information.
  if (info.getCC() == CC_DeviceKernel &&
      (CC == llvm::CallingConv::SPIR_KERNEL || CC == llvm::CallingConv::C)) {
    // Force target independent argument handling for the host visible
    // kernel functions.
    //
    // For CPU targets, this currently only works for OpenCL.
    assert(CC != llvm::CallingConv::C || getContext().getLangOpts().OpenCL);
    computeSPIRKernelABIInfo(CGM, *FI);
  } else if (info.getCC() == CC_Swift || info.getCC() == CC_SwiftAsync) {
    swiftcall::computeABIInfo(CGM, *FI);
  } else if (CGM.shouldUseLLVMABILowering(CC)) {
    CGM.computeABIInfoUsingLib(*FI);
  } else {
    CGM.getABIInfo().computeInfo(*FI);
  }

  // Loop over all of the computed argument and return value info.  If any of
  // them are direct or extend without a specified coerce type, specify the
  // default now.
  ABIArgInfo &retInfo = FI->getReturnInfo();
  if (retInfo.canHaveCoerceToType() && retInfo.getCoerceToType() == nullptr)
    retInfo.setCoerceToType(ConvertType(FI->getReturnType()));

  for (auto &I : FI->arguments())
    if (I.info.canHaveCoerceToType() && I.info.getCoerceToType() == nullptr)
      I.info.setCoerceToType(ConvertType(I.type));

  bool erased = FunctionsBeingProcessed.erase(FI);
  (void)erased;
  assert(erased && "Not in set?");

  return FI;
}

CGFunctionInfo *CGFunctionInfo::create(
    unsigned llvmCC, bool instanceMethod, bool chainCall, bool delegateCall,
    unsigned X86ABIAVXLevel, const FunctionType::ExtInfo &info,
    ArrayRef<ExtParameterInfo> paramInfos, CanQualType resultType,
    ArrayRef<CanQualType> argTypes, RequiredArgs required) {
  assert(paramInfos.empty() || paramInfos.size() == argTypes.size());
  assert(!required.allowsOptionalArgs() ||
         required.getNumRequiredArgs() <= argTypes.size());

  void *buffer = operator new(totalSizeToAlloc<ArgInfo, ExtParameterInfo>(
      argTypes.size() + 1, paramInfos.size()));

  CGFunctionInfo *FI = new (buffer) CGFunctionInfo();
  FI->CallingConvention = llvmCC;
  FI->EffectiveCallingConvention = llvmCC;
  FI->ASTCallingConvention = info.getCC();
  FI->InstanceMethod = instanceMethod;
  FI->ChainCall = chainCall;
  FI->DelegateCall = delegateCall;
  FI->CmseNSCall = info.getCmseNSCall();
  FI->NoReturn = info.getNoReturn();
  FI->ReturnsRetained = info.getProducesResult();
  FI->NoCallerSavedRegs = info.getNoCallerSavedRegs();
  FI->NoCfCheck = info.getNoCfCheck();
  FI->Required = required;
  FI->HasRegParm = info.getHasRegParm();
  FI->RegParm = info.getRegParm();
  FI->X86ABIAVXLevel = X86ABIAVXLevel;
  FI->ArgStruct = nullptr;
  FI->ArgStructAlign = 0;
  FI->NumArgs = argTypes.size();
  FI->HasExtParameterInfos = !paramInfos.empty();
  FI->getArgsBuffer()[0].type = resultType;
  FI->MaxVectorWidth = 0;
  for (unsigned i = 0, e = argTypes.size(); i != e; ++i)
    FI->getArgsBuffer()[i + 1].type = argTypes[i];
  for (unsigned i = 0, e = paramInfos.size(); i != e; ++i)
    FI->getExtParameterInfosBuffer()[i] = paramInfos[i];
  return FI;
}

/***/

bool CodeGenModule::ReturnTypeUsesSRet(const CGFunctionInfo &FI) {
  const auto &RI = FI.getReturnInfo();
  return RI.isIndirect() || (RI.isInAlloca() && RI.getInAllocaSRet());
}

bool CodeGenModule::ReturnTypeHasInReg(const CGFunctionInfo &FI) {
  const auto &RI = FI.getReturnInfo();
  return RI.getInReg();
}

bool CodeGenModule::ReturnSlotInterferesWithArgs(const CGFunctionInfo &FI) {
  return ReturnTypeUsesSRet(FI) &&
         getTargetCodeGenInfo().doesReturnSlotInterfereWithArgs();
}

bool CodeGenModule::ReturnTypeUsesFPRet(QualType ResultType) {
  if (const BuiltinType *BT = ResultType->getAs<BuiltinType>()) {
    switch (BT->getKind()) {
    default:
      return false;
    case BuiltinType::Float:
      return getTarget().useObjCFPRetForRealType(FloatModeKind::Float);
    case BuiltinType::Double:
      return getTarget().useObjCFPRetForRealType(FloatModeKind::Double);
    case BuiltinType::LongDouble:
      return getTarget().useObjCFPRetForRealType(FloatModeKind::LongDouble);
    }
  }

  return false;
}

bool CodeGenModule::ReturnTypeUsesFP2Ret(QualType ResultType) {
  if (const ComplexType *CT = ResultType->getAs<ComplexType>()) {
    if (const BuiltinType *BT = CT->getElementType()->getAs<BuiltinType>()) {
      if (BT->getKind() == BuiltinType::LongDouble)
        return getTarget().useObjCFP2RetForComplexLongDouble();
    }
  }

  return false;
}

llvm::FunctionType *CodeGenTypes::GetFunctionType(GlobalDecl GD) {
  const CGFunctionInfo &FI = arrangeGlobalDeclaration(GD);
  return GetFunctionType(FI);
}

llvm::FunctionType *CodeGenTypes::GetFunctionType(const CGFunctionInfo &FI) {

  bool Inserted = FunctionsBeingProcessed.insert(&FI).second;
  (void)Inserted;
  assert(Inserted && "Recursively being processed?");

  llvm::Type *resultType = nullptr;
  const ABIArgInfo &retAI = FI.getReturnInfo();
  switch (retAI.getKind()) {
  case ABIArgInfo::Expand:
  case ABIArgInfo::IndirectAliased:
    llvm_unreachable("Invalid ABI kind for return argument");

  case ABIArgInfo::TargetSpecific:
  case ABIArgInfo::Extend:
  case ABIArgInfo::Direct:
    resultType = retAI.getCoerceToType();
    break;

  case ABIArgInfo::InAlloca:
    if (retAI.getInAllocaSRet()) {
      // sret things on win32 aren't void, they return the sret pointer.
      QualType ret = FI.getReturnType();
      unsigned addressSpace = CGM.getTypes().getTargetAddressSpace(ret);
      resultType = llvm::PointerType::get(getLLVMContext(), addressSpace);
    } else {
      resultType = llvm::Type::getVoidTy(getLLVMContext());
    }
    break;

  case ABIArgInfo::Indirect:
  case ABIArgInfo::Ignore:
    resultType = llvm::Type::getVoidTy(getLLVMContext());
    break;

  case ABIArgInfo::CoerceAndExpand:
    resultType = retAI.getUnpaddedCoerceAndExpandType();
    break;
  }

  ClangToLLVMArgMapping IRFunctionArgs(getContext(), FI, true);
  SmallVector<llvm::Type *, 8> ArgTypes(IRFunctionArgs.totalIRArgs());

  // Add type for sret argument.
  if (IRFunctionArgs.hasSRetArg()) {
    ArgTypes[IRFunctionArgs.getSRetArgNo()] = llvm::PointerType::get(
        getLLVMContext(), FI.getReturnInfo().getIndirectAddrSpace());
  }

  // Add type for inalloca argument.
  if (IRFunctionArgs.hasInallocaArg())
    ArgTypes[IRFunctionArgs.getInallocaArgNo()] =
        llvm::PointerType::getUnqual(getLLVMContext());

  // Add in all of the required arguments.
  unsigned ArgNo = 0;
  CGFunctionInfo::const_arg_iterator it = FI.arg_begin(),
                                     ie = it + FI.getNumRequiredArgs();
  for (; it != ie; ++it, ++ArgNo) {
    const ABIArgInfo &ArgInfo = it->info;

    // Insert a padding type to ensure proper alignment.
    if (IRFunctionArgs.hasPaddingArg(ArgNo))
      ArgTypes[IRFunctionArgs.getPaddingArgNo(ArgNo)] =
          ArgInfo.getPaddingType();

    unsigned FirstIRArg, NumIRArgs;
    std::tie(FirstIRArg, NumIRArgs) = IRFunctionArgs.getIRArgs(ArgNo);

    switch (ArgInfo.getKind()) {
    case ABIArgInfo::Ignore:
    case ABIArgInfo::InAlloca:
      assert(NumIRArgs == 0);
      break;

    case ABIArgInfo::Indirect:
      assert(NumIRArgs == 1);
      // indirect arguments are always on the stack, which is alloca addr space.
      ArgTypes[FirstIRArg] = llvm::PointerType::get(
          getLLVMContext(), CGM.getDataLayout().getAllocaAddrSpace());
      break;
    case ABIArgInfo::IndirectAliased:
      assert(NumIRArgs == 1);
      ArgTypes[FirstIRArg] = llvm::PointerType::get(
          getLLVMContext(), ArgInfo.getIndirectAddrSpace());
      break;
    case ABIArgInfo::TargetSpecific:
    case ABIArgInfo::Extend:
    case ABIArgInfo::Direct: {
      // Fast-isel and the optimizer generally like scalar values better than
      // FCAs, so we flatten them if this is safe to do for this argument.
      llvm::Type *argType = ArgInfo.getCoerceToType();
      llvm::StructType *st = dyn_cast<llvm::StructType>(argType);
      if (st && ArgInfo.isDirect() && ArgInfo.getCanBeFlattened()) {
        assert(NumIRArgs == st->getNumElements());
        for (unsigned i = 0, e = st->getNumElements(); i != e; ++i)
          ArgTypes[FirstIRArg + i] = st->getElementType(i);
      } else {
        assert(NumIRArgs == 1);
        ArgTypes[FirstIRArg] = argType;
      }
      break;
    }

    case ABIArgInfo::CoerceAndExpand: {
      auto ArgTypesIter = ArgTypes.begin() + FirstIRArg;
      for (auto *EltTy : ArgInfo.getCoerceAndExpandTypeSequence()) {
        *ArgTypesIter++ = EltTy;
      }
      assert(ArgTypesIter == ArgTypes.begin() + FirstIRArg + NumIRArgs);
      break;
    }

    case ABIArgInfo::Expand:
      auto ArgTypesIter = ArgTypes.begin() + FirstIRArg;
      getExpandedTypes(it->type, ArgTypesIter);
      assert(ArgTypesIter == ArgTypes.begin() + FirstIRArg + NumIRArgs);
      break;
    }
  }

  bool Erased = FunctionsBeingProcessed.erase(&FI);
  (void)Erased;
  assert(Erased && "Not in set?");

  return llvm::FunctionType::get(resultType, ArgTypes, FI.isVariadic());
}

llvm::Type *CodeGenTypes::GetFunctionTypeForVTable(GlobalDecl GD) {
  const CXXMethodDecl *MD = cast<CXXMethodDecl>(GD.getDecl());
  const FunctionProtoType *FPT = MD->getType()->castAs<FunctionProtoType>();

  if (!isFuncTypeConvertible(FPT))
    return llvm::StructType::get(getLLVMContext());

  return GetFunctionType(GD);
}

