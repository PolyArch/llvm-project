//===- ModuleImportDetail.h - Shared LLVM import details ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_TARGET_LLVMIR_MODULEIMPORTDETAIL_H
#define MLIR_LIB_TARGET_LLVMIR_MODULEIMPORTDETAIL_H

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Attributes.h"
#include <string>

namespace llvm {
class Metadata;
class Module;
class Value;
}

namespace mlir::LLVM::detail {
std::string diag(const llvm::Value &value);
std::string diagMD(const llvm::Metadata *node, const llvm::Module *module);
ArrayAttr convertLLVMAttributesToMLIR(
    Location loc, MLIRContext *context, llvm::AttributeSet attributes,
    ArrayRef<StringLiteral> attributesToSkip = {},
    ArrayRef<StringLiteral> attributePrefixesToSkip = {});

/// Returns the name of the global_ctors global variables.
inline constexpr StringRef getGlobalCtorsVarName() {
  return "llvm.global_ctors";
}

/// Returns the name of the global_dtors global variables.
inline constexpr StringRef getGlobalDtorsVarName() {
  return "llvm.global_dtors";
}

} // namespace mlir::LLVM::detail

#endif
