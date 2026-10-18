//===--- SourceVariadicArguments.h - Source variadic values -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CODEGEN_SOURCEVARIADICARGUMENTS_H
#define LLVM_CLANG_CODEGEN_SOURCEVARIADICARGUMENTS_H

#include "llvm/ADT/StringRef.h"
#include <cstdint>

namespace clang::CodeGen::SourceVariadicArguments {

/// The call bundle contains the source variadic arguments, in source parameter
/// order, before target ABI coercion. Each value group begins with an i8 Kind
/// and an i64 source storage size in bytes. A scalar has one value operand, a
/// complex value has real and imaginary operands, and an aggregate has its
/// storage address. An ignored aggregate argument has no value operand. There
/// are no physical register or stack-layout fields.
inline constexpr llvm::StringLiteral CallBundle = "clang.source.varargs";

/// A source read is a void call taking the evaluated va_list address and its
/// result storage. Its bundle contains the i64 source byte size and the
/// local native reader function with the same void(ptr, ptr) signature.
/// Clang's ABI owner emits the native reader body; an ordinary target consumer
/// lowers the source read to that reader after preserving the payload. A source
/// execution consumer supplies the logical argument value to result storage
/// instead, including source-owned va_start, va_copy, and va_end lifecycle.
/// No target ABI list state is used by that source execution.
///
/// The read callee must be a direct external declaration in ReadMarkerPrefix.
/// The native reader must be a local definition in NativeReadPrefix. Both
/// have the exact non-variadic void(ptr, ptr) signature. A different callee,
/// body ownership, signature, or malformed bundle must be rejected rather than
/// erased or interpreted as a source read by a consumer.
inline constexpr llvm::StringLiteral ReadBundle = "clang.source.va_arg";
inline constexpr llvm::StringLiteral ReadMarkerPrefix = "__clang_source_va_arg";
inline constexpr llvm::StringLiteral NativeReadPrefix = "__clang_native_va_arg";

enum class Kind : std::uint8_t { Scalar, Complex, Aggregate, Ignored };

} // namespace clang::CodeGen::SourceVariadicArguments

#endif
