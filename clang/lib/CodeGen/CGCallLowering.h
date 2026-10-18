//===--- CGCallLowering.h - Call lowering internals ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_CGCALLLOWERING_H
#define LLVM_CLANG_LIB_CODEGEN_CGCALLLOWERING_H

#include "Address.h"
#include "clang/AST/Type.h"
#include "clang/CodeGen/CGFunctionInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DerivedTypes.h"

namespace clang {
class ASTContext;
class Decl;
class NonNullAttr;
class ObjCMethodDecl;
class ParmVarDecl;

namespace CodeGen {
class CallArgList;
class CodeGenFunction;

/// Encapsulates information about the way function arguments from
/// CGFunctionInfo should be passed to actual LLVM IR function.
class ClangToLLVMArgMapping {
  static const unsigned InvalidIndex = ~0U;
  unsigned InallocaArgNo;
  unsigned SRetArgNo;
  unsigned TotalIRArgs;

  /// Arguments of LLVM IR function corresponding to single Clang argument.
  struct IRArgs {
    unsigned PaddingArgIndex;
    // Argument is expanded to IR arguments at positions
    // [FirstArgIndex, FirstArgIndex + NumberOfArgs).
    unsigned FirstArgIndex;
    unsigned NumberOfArgs;

    IRArgs()
        : PaddingArgIndex(InvalidIndex), FirstArgIndex(InvalidIndex),
          NumberOfArgs(0) {}
  };

  SmallVector<IRArgs, 8> ArgInfo;

public:
  ClangToLLVMArgMapping(const ASTContext &Context, const CGFunctionInfo &FI,
                        bool OnlyRequiredArgs = false)
      : InallocaArgNo(InvalidIndex), SRetArgNo(InvalidIndex), TotalIRArgs(0),
        ArgInfo(OnlyRequiredArgs ? FI.getNumRequiredArgs() : FI.arg_size()) {
    construct(Context, FI, OnlyRequiredArgs);
  }

  bool hasInallocaArg() const { return InallocaArgNo != InvalidIndex; }
  unsigned getInallocaArgNo() const {
    assert(hasInallocaArg());
    return InallocaArgNo;
  }

  bool hasSRetArg() const { return SRetArgNo != InvalidIndex; }
  unsigned getSRetArgNo() const {
    assert(hasSRetArg());
    return SRetArgNo;
  }

  unsigned totalIRArgs() const { return TotalIRArgs; }

  bool hasPaddingArg(unsigned ArgNo) const {
    assert(ArgNo < ArgInfo.size());
    return ArgInfo[ArgNo].PaddingArgIndex != InvalidIndex;
  }
  unsigned getPaddingArgNo(unsigned ArgNo) const {
    assert(hasPaddingArg(ArgNo));
    return ArgInfo[ArgNo].PaddingArgIndex;
  }

  /// Returns index of first IR argument corresponding to ArgNo, and their
  /// quantity.
  std::pair<unsigned, unsigned> getIRArgs(unsigned ArgNo) const {
    assert(ArgNo < ArgInfo.size());
    return std::make_pair(ArgInfo[ArgNo].FirstArgIndex,
                          ArgInfo[ArgNo].NumberOfArgs);
  }

private:
  void construct(const ASTContext &Context, const CGFunctionInfo &FI,
                 bool OnlyRequiredArgs);
};

CallingConv getCallingConventionForDecl(const ObjCMethodDecl *D,
                                               bool IsTargetDefaultMSABI);

llvm::Value *CreatePFPCoercedLoad(Address Src, QualType SrcFETy,
                                         llvm::Type *Ty, CodeGenFunction &CGF);

llvm::Value *CreateCoercedLoad(Address Src, QualType SrcFETy,
                                      llvm::Type *Ty, CodeGenFunction &CGF);

bool CreatePFPCoercedStore(llvm::Value *Src, QualType SrcFETy,
                                  Address Dst, CodeGenFunction &CGF);

Address emitAddressAtOffset(CodeGenFunction &CGF, Address addr,
                                   const ABIArgInfo &info);

std::pair<llvm::Value *, bool>
CoerceScalableToFixed(CodeGenFunction &CGF, llvm::FixedVectorType *ToTy,
                      llvm::ScalableVectorType *FromTy, llvm::Value *V,
                      StringRef Name = "");

bool IsArgumentMaybeUndef(const Decl *TargetDecl,
                                 unsigned NumRequiredArgs, unsigned ArgNo);

const NonNullAttr *getNonNullAttr(const Decl *FD, const ParmVarDecl *PVD,
                                         QualType ArgType, unsigned ArgNo);

void deactivateArgCleanupsBeforeCall(CodeGenFunction &CGF,
                                            const CallArgList &CallArgs);

} // namespace CodeGen
} // namespace clang

#endif
