//===--- CGCallCoercion.cpp - Coerce values to and from ABI layouts ---===//
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

namespace {
// ABIArgInfo::Expand implementation.

// Specifies the way QualType passed as ABIArgInfo::Expand is expanded.
struct TypeExpansion {
  enum TypeExpansionKind {
    // Elements of constant arrays are expanded recursively.
    TEK_ConstantArray,
    // Record fields are expanded recursively (but if record is a union, only
    // the field with the largest size is expanded).
    TEK_Record,
    // For complex types, real and imaginary parts are expanded recursively.
    TEK_Complex,
    // All other types are not expandable.
    TEK_None
  };

  const TypeExpansionKind Kind;

  TypeExpansion(TypeExpansionKind K) : Kind(K) {}
  virtual ~TypeExpansion() {}
};

struct ConstantArrayExpansion : TypeExpansion {
  QualType EltTy;
  uint64_t NumElts;

  ConstantArrayExpansion(QualType EltTy, uint64_t NumElts)
      : TypeExpansion(TEK_ConstantArray), EltTy(EltTy), NumElts(NumElts) {}
  static bool classof(const TypeExpansion *TE) {
    return TE->Kind == TEK_ConstantArray;
  }
};

struct RecordExpansion : TypeExpansion {
  SmallVector<const CXXBaseSpecifier *, 1> Bases;

  SmallVector<const FieldDecl *, 1> Fields;

  RecordExpansion(SmallVector<const CXXBaseSpecifier *, 1> &&Bases,
                  SmallVector<const FieldDecl *, 1> &&Fields)
      : TypeExpansion(TEK_Record), Bases(std::move(Bases)),
        Fields(std::move(Fields)) {}
  static bool classof(const TypeExpansion *TE) {
    return TE->Kind == TEK_Record;
  }
};

struct ComplexExpansion : TypeExpansion {
  QualType EltTy;

  ComplexExpansion(QualType EltTy) : TypeExpansion(TEK_Complex), EltTy(EltTy) {}
  static bool classof(const TypeExpansion *TE) {
    return TE->Kind == TEK_Complex;
  }
};

struct NoExpansion : TypeExpansion {
  NoExpansion() : TypeExpansion(TEK_None) {}
  static bool classof(const TypeExpansion *TE) { return TE->Kind == TEK_None; }
};
} // namespace

static std::unique_ptr<TypeExpansion>
getTypeExpansion(QualType Ty, const ASTContext &Context) {
  if (const ConstantArrayType *AT = Context.getAsConstantArrayType(Ty)) {
    return std::make_unique<ConstantArrayExpansion>(AT->getElementType(),
                                                    AT->getZExtSize());
  }
  if (const auto *RD = Ty->getAsRecordDecl()) {
    SmallVector<const CXXBaseSpecifier *, 1> Bases;
    SmallVector<const FieldDecl *, 1> Fields;
    assert(!RD->hasFlexibleArrayMember() &&
           "Cannot expand structure with flexible array.");
    if (RD->isUnion()) {
      // Unions can be here only in degenerative cases - all the fields are same
      // after flattening. Thus we have to use the "largest" field.
      const FieldDecl *LargestFD = nullptr;
      CharUnits UnionSize = CharUnits::Zero();

      for (const auto *FD : RD->fields()) {
        if (FD->isZeroLengthBitField())
          continue;
        assert(!FD->isBitField() &&
               "Cannot expand structure with bit-field members.");
        CharUnits FieldSize = Context.getTypeSizeInChars(FD->getType());
        if (UnionSize < FieldSize) {
          UnionSize = FieldSize;
          LargestFD = FD;
        }
      }
      if (LargestFD)
        Fields.push_back(LargestFD);
    } else {
      if (const auto *CXXRD = dyn_cast<CXXRecordDecl>(RD)) {
        assert(!CXXRD->isDynamicClass() &&
               "cannot expand vtable pointers in dynamic classes");
        llvm::append_range(Bases, llvm::make_pointer_range(CXXRD->bases()));
      }

      for (const auto *FD : RD->fields()) {
        if (FD->isZeroLengthBitField())
          continue;
        assert(!FD->isBitField() &&
               "Cannot expand structure with bit-field members.");
        Fields.push_back(FD);
      }
    }
    return std::make_unique<RecordExpansion>(std::move(Bases),
                                             std::move(Fields));
  }
  if (const ComplexType *CT = Ty->getAs<ComplexType>()) {
    return std::make_unique<ComplexExpansion>(CT->getElementType());
  }
  return std::make_unique<NoExpansion>();
}

static int getExpansionSize(QualType Ty, const ASTContext &Context) {
  auto Exp = getTypeExpansion(Ty, Context);
  if (auto CAExp = dyn_cast<ConstantArrayExpansion>(Exp.get())) {
    return CAExp->NumElts * getExpansionSize(CAExp->EltTy, Context);
  }
  if (auto RExp = dyn_cast<RecordExpansion>(Exp.get())) {
    int Res = 0;
    for (auto BS : RExp->Bases)
      Res += getExpansionSize(BS->getType(), Context);
    for (auto FD : RExp->Fields)
      Res += getExpansionSize(FD->getType(), Context);
    return Res;
  }
  if (isa<ComplexExpansion>(Exp.get()))
    return 2;
  assert(isa<NoExpansion>(Exp.get()));
  return 1;
}

void CodeGenTypes::getExpandedTypes(
    QualType Ty, SmallVectorImpl<llvm::Type *>::iterator &TI) {
  auto Exp = getTypeExpansion(Ty, Context);
  if (auto CAExp = dyn_cast<ConstantArrayExpansion>(Exp.get())) {
    for (int i = 0, n = CAExp->NumElts; i < n; i++) {
      getExpandedTypes(CAExp->EltTy, TI);
    }
  } else if (auto RExp = dyn_cast<RecordExpansion>(Exp.get())) {
    for (auto BS : RExp->Bases)
      getExpandedTypes(BS->getType(), TI);
    for (auto FD : RExp->Fields)
      getExpandedTypes(FD->getType(), TI);
  } else if (auto CExp = dyn_cast<ComplexExpansion>(Exp.get())) {
    llvm::Type *EltTy = ConvertType(CExp->EltTy);
    *TI++ = EltTy;
    *TI++ = EltTy;
  } else {
    assert(isa<NoExpansion>(Exp.get()));
    *TI++ = ConvertType(Ty);
  }
}

static void forConstantArrayExpansion(CodeGenFunction &CGF,
                                      ConstantArrayExpansion *CAE,
                                      Address BaseAddr,
                                      llvm::function_ref<void(Address)> Fn) {
  for (int i = 0, n = CAE->NumElts; i < n; i++) {
    Address EltAddr = CGF.Builder.CreateConstGEP2_32(BaseAddr, 0, i);
    Fn(EltAddr);
  }
}

void CodeGenFunction::ExpandTypeFromArgs(QualType Ty, LValue LV,
                                         llvm::Function::arg_iterator &AI) {
  assert(LV.isSimple() &&
         "Unexpected non-simple lvalue during struct expansion.");

  auto Exp = getTypeExpansion(Ty, getContext());
  if (auto CAExp = dyn_cast<ConstantArrayExpansion>(Exp.get())) {
    forConstantArrayExpansion(
        *this, CAExp, LV.getAddress(), [&](Address EltAddr) {
          LValue LV = MakeAddrLValue(EltAddr, CAExp->EltTy);
          ExpandTypeFromArgs(CAExp->EltTy, LV, AI);
        });
  } else if (auto RExp = dyn_cast<RecordExpansion>(Exp.get())) {
    Address This = LV.getAddress();
    for (const CXXBaseSpecifier *BS : RExp->Bases) {
      // Perform a single step derived-to-base conversion.
      Address Base =
          GetAddressOfBaseClass(This, Ty->getAsCXXRecordDecl(), &BS, &BS + 1,
                                /*NullCheckValue=*/false, SourceLocation());
      LValue SubLV = MakeAddrLValue(Base, BS->getType());

      // Recurse onto bases.
      ExpandTypeFromArgs(BS->getType(), SubLV, AI);
    }
    for (auto FD : RExp->Fields) {
      // FIXME: What are the right qualifiers here?
      LValue SubLV = EmitLValueForFieldInitialization(LV, FD);
      ExpandTypeFromArgs(FD->getType(), SubLV, AI);
    }
  } else if (isa<ComplexExpansion>(Exp.get())) {
    auto realValue = &*AI++;
    auto imagValue = &*AI++;
    EmitStoreOfComplex(ComplexPairTy(realValue, imagValue), LV, /*init*/ true);
  } else {
    // Call EmitStoreOfScalar except when the lvalue is a bitfield to emit a
    // primitive store.
    assert(isa<NoExpansion>(Exp.get()));
    llvm::Value *Arg = &*AI++;
    if (LV.isBitField()) {
      EmitStoreThroughLValue(RValue::get(Arg), LV);
    } else {
      // TODO: currently there are some places are inconsistent in what LLVM
      // pointer type they use (see D118744). Once clang uses opaque pointers
      // all LLVM pointer types will be the same and we can remove this check.
      if (Arg->getType()->isPointerTy()) {
        Address Addr = LV.getAddress();
        Arg = Builder.CreateBitCast(Arg, Addr.getElementType());
      }
      EmitStoreOfScalar(Arg, LV);
    }
  }
}

void CodeGenFunction::ExpandTypeToArgs(
    QualType Ty, CallArg Arg, llvm::FunctionType *IRFuncTy,
    SmallVectorImpl<llvm::Value *> &IRCallArgs, unsigned &IRCallArgPos) {
  auto Exp = getTypeExpansion(Ty, getContext());
  if (auto CAExp = dyn_cast<ConstantArrayExpansion>(Exp.get())) {
    Address Addr = Arg.hasLValue() ? Arg.getKnownLValue().getAddress()
                                   : Arg.getKnownRValue().getAggregateAddress();
    forConstantArrayExpansion(*this, CAExp, Addr, [&](Address EltAddr) {
      CallArg EltArg =
          CallArg(convertTempToRValue(EltAddr, CAExp->EltTy, SourceLocation()),
                  CAExp->EltTy);
      ExpandTypeToArgs(CAExp->EltTy, EltArg, IRFuncTy, IRCallArgs,
                       IRCallArgPos);
    });
  } else if (auto RExp = dyn_cast<RecordExpansion>(Exp.get())) {
    Address This = Arg.hasLValue() ? Arg.getKnownLValue().getAddress()
                                   : Arg.getKnownRValue().getAggregateAddress();
    for (const CXXBaseSpecifier *BS : RExp->Bases) {
      // Perform a single step derived-to-base conversion.
      Address Base =
          GetAddressOfBaseClass(This, Ty->getAsCXXRecordDecl(), &BS, &BS + 1,
                                /*NullCheckValue=*/false, SourceLocation());
      CallArg BaseArg = CallArg(RValue::getAggregate(Base), BS->getType());

      // Recurse onto bases.
      ExpandTypeToArgs(BS->getType(), BaseArg, IRFuncTy, IRCallArgs,
                       IRCallArgPos);
    }

    LValue LV = MakeAddrLValue(This, Ty);
    for (auto FD : RExp->Fields) {
      CallArg FldArg =
          CallArg(EmitRValueForField(LV, FD, SourceLocation()), FD->getType());
      ExpandTypeToArgs(FD->getType(), FldArg, IRFuncTy, IRCallArgs,
                       IRCallArgPos);
    }
  } else if (isa<ComplexExpansion>(Exp.get())) {
    ComplexPairTy CV = Arg.getKnownRValue().getComplexVal();
    IRCallArgs[IRCallArgPos++] = CV.first;
    IRCallArgs[IRCallArgPos++] = CV.second;
  } else {
    assert(isa<NoExpansion>(Exp.get()));
    auto RV = Arg.getKnownRValue();
    assert(RV.isScalar() &&
           "Unexpected non-scalar rvalue during struct expansion.");

    // Insert a bitcast as needed.
    llvm::Value *V = RV.getScalarVal();
    if (IRCallArgPos < IRFuncTy->getNumParams() &&
        V->getType() != IRFuncTy->getParamType(IRCallArgPos))
      V = Builder.CreateBitCast(V, IRFuncTy->getParamType(IRCallArgPos));

    IRCallArgs[IRCallArgPos++] = V;
  }
}

/// Create a temporary allocation for the purposes of coercion.
static RawAddress CreateTempAllocaForCoercion(CodeGenFunction &CGF,
                                              llvm::Type *Ty,
                                              CharUnits MinAlign,
                                              const Twine &Name = "tmp") {
  // Don't use an alignment that's worse than what LLVM would prefer.
  auto PrefAlign = CGF.CGM.getDataLayout().getPrefTypeAlign(Ty);
  CharUnits Align = std::max(MinAlign, CharUnits::fromQuantity(PrefAlign));

  return CGF.CreateTempAlloca(Ty, Align, Name + ".coerce");
}

/// EnterStructPointerForCoercedAccess - Given a struct pointer that we are
/// accessing some number of bytes out of it, try to gep into the struct to get
/// at its inner goodness.  Dive as deep as possible without entering an element
/// with an in-memory size smaller than DstSize.
static Address EnterStructPointerForCoercedAccess(Address SrcPtr,
                                                  llvm::StructType *SrcSTy,
                                                  uint64_t DstSize,
                                                  CodeGenFunction &CGF) {
  // We can't dive into a zero-element struct.
  if (SrcSTy->getNumElements() == 0)
    return SrcPtr;

  llvm::Type *FirstElt = SrcSTy->getElementType(0);

  // If the first elt is at least as large as what we're looking for, or if the
  // first element is the same size as the whole struct, we can enter it. The
  // comparison must be made on the store size and not the alloca size. Using
  // the alloca size may overstate the size of the load.
  uint64_t FirstEltSize = CGF.CGM.getDataLayout().getTypeStoreSize(FirstElt);
  if (FirstEltSize < DstSize &&
      FirstEltSize < CGF.CGM.getDataLayout().getTypeStoreSize(SrcSTy))
    return SrcPtr;

  // GEP into the first element.
  SrcPtr = CGF.Builder.CreateStructGEP(SrcPtr, 0, "coerce.dive");

  // If the first element is a struct, recurse.
  llvm::Type *SrcTy = SrcPtr.getElementType();
  if (llvm::StructType *SrcSTy = dyn_cast<llvm::StructType>(SrcTy))
    return EnterStructPointerForCoercedAccess(SrcPtr, SrcSTy, DstSize, CGF);

  return SrcPtr;
}

/// CoerceIntOrPtrToIntOrPtr - Convert a value Val to the specific Ty where both
/// are either integers or pointers.  This does a truncation of the value if it
/// is too large or a zero extension if it is too small.
///
/// This behaves as if the value were coerced through memory, so on big-endian
/// targets the high bits are preserved in a truncation, while little-endian
/// targets preserve the low bits.
static llvm::Value *CoerceIntOrPtrToIntOrPtr(llvm::Value *Val, llvm::Type *Ty,
                                             CodeGenFunction &CGF) {
  if (Val->getType() == Ty)
    return Val;

  if (isa<llvm::PointerType>(Val->getType())) {
    // If this is Pointer->Pointer avoid conversion to and from int.
    if (isa<llvm::PointerType>(Ty))
      return CGF.Builder.CreateBitCast(Val, Ty, "coerce.val");

    // Convert the pointer to an integer so we can play with its width.
    Val = CGF.Builder.CreatePtrToInt(Val, CGF.IntPtrTy, "coerce.val.pi");
  }

  llvm::Type *DestIntTy = Ty;
  if (isa<llvm::PointerType>(DestIntTy))
    DestIntTy = CGF.IntPtrTy;

  if (Val->getType() != DestIntTy) {
    const llvm::DataLayout &DL = CGF.CGM.getDataLayout();
    if (DL.isBigEndian()) {
      // Preserve the high bits on big-endian targets.
      // That is what memory coercion does.
      uint64_t SrcSize = DL.getTypeSizeInBits(Val->getType());
      uint64_t DstSize = DL.getTypeSizeInBits(DestIntTy);

      if (SrcSize > DstSize) {
        Val = CGF.Builder.CreateLShr(Val, SrcSize - DstSize, "coerce.highbits");
        Val = CGF.Builder.CreateTrunc(Val, DestIntTy, "coerce.val.ii");
      } else {
        Val = CGF.Builder.CreateZExt(Val, DestIntTy, "coerce.val.ii");
        Val = CGF.Builder.CreateShl(Val, DstSize - SrcSize, "coerce.highbits");
      }
    } else {
      // Little-endian targets preserve the low bits. No shifts required.
      Val = CGF.Builder.CreateIntCast(Val, DestIntTy, false, "coerce.val.ii");
    }
  }

  if (isa<llvm::PointerType>(Ty))
    Val = CGF.Builder.CreateIntToPtr(Val, Ty, "coerce.val.ip");
  return Val;
}

llvm::Value *CodeGen::CreatePFPCoercedLoad(Address Src, QualType SrcFETy,
                                         llvm::Type *Ty, CodeGenFunction &CGF) {
  std::vector<PFPField> PFPFields = CGF.getContext().findPFPFields(SrcFETy);
  if (PFPFields.empty())
    return nullptr;

  auto LoadCoercedField = [&](CharUnits Offset,
                              llvm::Type *FieldType) -> llvm::Value * {
    // Check whether the field at Offset is a PFP field. This function is called
    // in ascending order of offset, and PFPFields is sorted by offset. This
    // means that we only need to check the first element (and remove it from
    // PFPFields if matching).
    if (!PFPFields.empty() && PFPFields[0].Offset == Offset) {
      auto FieldAddr = CGF.EmitAddressOfPFPField(Src, PFPFields[0]);
      llvm::Value *FieldVal = CGF.Builder.CreateLoad(FieldAddr);
      if (isa<llvm::IntegerType>(FieldType))
        FieldVal = CGF.Builder.CreatePtrToInt(FieldVal, FieldType);
      PFPFields.erase(PFPFields.begin());
      return FieldVal;
    }
    auto FieldAddr =
        CGF.Builder
            .CreateConstInBoundsByteGEP(Src.withElementType(CGF.Int8Ty), Offset)
            .withElementType(FieldType);
    return CGF.Builder.CreateLoad(FieldAddr);
  };

  // The types handled by this function are the only ones that may be generated
  // by AArch64ABIInfo::classify{Argument,Return}Type for struct types with
  // pointers. PFP is only supported on AArch64.
  if (isa<llvm::IntegerType>(Ty) || isa<llvm::PointerType>(Ty)) {
    auto Addr = CGF.EmitAddressOfPFPField(Src, PFPFields[0]);
    llvm::Value *Val = CGF.Builder.CreateLoad(Addr);
    if (isa<llvm::IntegerType>(Ty))
      Val = CGF.Builder.CreatePtrToInt(Val, Ty);
    return Val;
  }
  auto *AT = cast<llvm::ArrayType>(Ty);
  auto *ET = AT->getElementType();
  CharUnits WordSize = CGF.getContext().toCharUnitsFromBits(
      CGF.CGM.getDataLayout().getTypeSizeInBits(ET));
  CharUnits Offset = CharUnits::Zero();
  llvm::Value *Val = llvm::PoisonValue::get(AT);
  for (unsigned Idx = 0; Idx != AT->getNumElements(); ++Idx, Offset += WordSize)
    Val = CGF.Builder.CreateInsertValue(Val, LoadCoercedField(Offset, ET), Idx);
  return Val;
}

/// CreateCoercedLoad - Create a load from \arg SrcPtr interpreted as
/// a pointer to an object of type \arg Ty, known to be aligned to
/// \arg SrcAlign bytes.
///
/// This safely handles the case when the src type is smaller than the
/// destination type; in this situation the values of bits which not
/// present in the src are undefined.
llvm::Value *CodeGen::CreateCoercedLoad(Address Src, QualType SrcFETy,
                                      llvm::Type *Ty, CodeGenFunction &CGF) {
  llvm::Type *SrcTy = Src.getElementType();

  // If SrcTy and Ty are the same, just do a load.
  if (SrcTy == Ty)
    return CGF.Builder.CreateLoad(Src);

  if (llvm::Value *V = CreatePFPCoercedLoad(Src, SrcFETy, Ty, CGF))
    return V;

  llvm::TypeSize DstSize = CGF.CGM.getDataLayout().getTypeAllocSize(Ty);

  if (llvm::StructType *SrcSTy = dyn_cast<llvm::StructType>(SrcTy)) {
    Src = EnterStructPointerForCoercedAccess(Src, SrcSTy,
                                             DstSize.getFixedValue(), CGF);
    SrcTy = Src.getElementType();
  }

  llvm::TypeSize SrcSize = CGF.CGM.getDataLayout().getTypeAllocSize(SrcTy);

  // If the source and destination are integer or pointer types, just do an
  // extension or truncation to the desired type.
  if ((isa<llvm::IntegerType>(Ty) || isa<llvm::PointerType>(Ty)) &&
      (isa<llvm::IntegerType>(SrcTy) || isa<llvm::PointerType>(SrcTy))) {
    llvm::Value *Load = CGF.Builder.CreateLoad(Src);
    return CoerceIntOrPtrToIntOrPtr(Load, Ty, CGF);
  }

  // If load is legal, just bitcast the src pointer.
  if (!SrcSize.isScalable() && !DstSize.isScalable() &&
      SrcSize.getFixedValue() >= DstSize.getFixedValue()) {
    // Generally SrcSize is never greater than DstSize, since this means we are
    // losing bits. However, this can happen in cases where the structure has
    // additional padding, for example due to a user specified alignment.
    //
    // FIXME: Assert that we aren't truncating non-padding bits when have access
    // to that information.
    Src = Src.withElementType(Ty);
    return CGF.Builder.CreateLoad(Src);
  }

  // If coercing a fixed vector to a scalable vector for ABI compatibility, and
  // the types match, use the llvm.vector.insert intrinsic to perform the
  // conversion.
  if (auto *ScalableDstTy = dyn_cast<llvm::ScalableVectorType>(Ty)) {
    if (auto *FixedSrcTy = dyn_cast<llvm::FixedVectorType>(SrcTy)) {
      // If we are casting a fixed i8 vector to a scalable i1 predicate
      // vector, use a vector insert and bitcast the result.
      if (ScalableDstTy->getElementType()->isIntegerTy(1) &&
          FixedSrcTy->getElementType()->isIntegerTy(8)) {
        ScalableDstTy = llvm::ScalableVectorType::get(
            FixedSrcTy->getElementType(),
            llvm::divideCeil(
                ScalableDstTy->getElementCount().getKnownMinValue(), 8));
      }
      if (ScalableDstTy->getElementType() == FixedSrcTy->getElementType()) {
        auto *Load = CGF.Builder.CreateLoad(Src);
        auto *PoisonVec = llvm::PoisonValue::get(ScalableDstTy);
        llvm::Value *Result = CGF.Builder.CreateInsertVector(
            ScalableDstTy, PoisonVec, Load, uint64_t(0), "cast.scalable");
        ScalableDstTy = cast<llvm::ScalableVectorType>(
            llvm::VectorType::getWithSizeAndScalar(ScalableDstTy, Ty));
        if (Result->getType() != ScalableDstTy)
          Result = CGF.Builder.CreateBitCast(Result, ScalableDstTy);
        if (Result->getType() != Ty)
          Result = CGF.Builder.CreateExtractVector(Ty, Result, uint64_t(0));
        return Result;
      }
    }
  }

  // Otherwise do coercion through memory. This is stupid, but simple.
  RawAddress Tmp =
      CreateTempAllocaForCoercion(CGF, Ty, Src.getAlignment(), Src.getName());
  CGF.Builder.CreateMemCpy(
      Tmp.getPointer(), Tmp.getAlignment().getAsAlign(),
      Src.emitRawPointer(CGF), Src.getAlignment().getAsAlign(),
      llvm::ConstantInt::get(CGF.IntPtrTy, SrcSize.getKnownMinValue()));
  return CGF.Builder.CreateLoad(Tmp);
}

bool CodeGen::CreatePFPCoercedStore(llvm::Value *Src, QualType SrcFETy,
                                  Address Dst, CodeGenFunction &CGF) {
  std::vector<PFPField> PFPFields = CGF.getContext().findPFPFields(SrcFETy);
  if (PFPFields.empty())
    return false;

  llvm::Type *SrcTy = Src->getType();
  auto StoreCoercedField = [&](CharUnits Offset, llvm::Value *FieldVal) {
    if (!PFPFields.empty() && PFPFields[0].Offset == Offset) {
      auto FieldAddr = CGF.EmitAddressOfPFPField(Dst, PFPFields[0]);
      if (isa<llvm::IntegerType>(FieldVal->getType()))
        FieldVal = CGF.Builder.CreateIntToPtr(FieldVal, CGF.VoidPtrTy);
      CGF.Builder.CreateStore(FieldVal, FieldAddr);
      PFPFields.erase(PFPFields.begin());
    } else {
      auto FieldAddr = CGF.Builder
                           .CreateConstInBoundsByteGEP(
                               Dst.withElementType(CGF.Int8Ty), Offset)
                           .withElementType(FieldVal->getType());
      CGF.Builder.CreateStore(FieldVal, FieldAddr);
    }
  };

  // The types handled by this function are the only ones that may be generated
  // by AArch64ABIInfo::classify{Argument,Return}Type for struct types with
  // pointers. PFP is only supported on AArch64.
  if (isa<llvm::IntegerType>(SrcTy) || isa<llvm::PointerType>(SrcTy)) {
    if (isa<llvm::IntegerType>(SrcTy))
      Src = CGF.Builder.CreateIntToPtr(Src, CGF.VoidPtrTy);
    auto Addr = CGF.EmitAddressOfPFPField(Dst, PFPFields[0]);
    CGF.Builder.CreateStore(Src, Addr);
  } else {
    auto *AT = cast<llvm::ArrayType>(SrcTy);
    auto *ET = AT->getElementType();
    CharUnits WordSize = CGF.getContext().toCharUnitsFromBits(
        CGF.CGM.getDataLayout().getTypeSizeInBits(ET));
    CharUnits Offset = CharUnits::Zero();
    for (unsigned i = 0; i != AT->getNumElements(); ++i, Offset += WordSize)
      StoreCoercedField(Offset, CGF.Builder.CreateExtractValue(Src, i));
  }
  return true;
}

void CodeGenFunction::CreateCoercedStore(llvm::Value *Src, QualType SrcFETy,
                                         Address Dst, llvm::TypeSize DstSize,
                                         bool DstIsVolatile) {
  if (!DstSize)
    return;

  llvm::Type *SrcTy = Src->getType();
  llvm::TypeSize SrcSize = CGM.getDataLayout().getTypeAllocSize(SrcTy);

  // GEP into structs to try to make types match.
  // FIXME: This isn't really that useful with opaque types, but it impacts a
  // lot of regression tests.
  if (SrcTy != Dst.getElementType()) {
    if (llvm::StructType *DstSTy =
            dyn_cast<llvm::StructType>(Dst.getElementType())) {
      assert(!SrcSize.isScalable());
      Dst = EnterStructPointerForCoercedAccess(Dst, DstSTy,
                                               SrcSize.getFixedValue(), *this);
    }
  }

  if (CreatePFPCoercedStore(Src, SrcFETy, Dst, *this))
    return;

  if (SrcSize.isScalable() || SrcSize <= DstSize) {
    if (SrcTy->isIntegerTy() && Dst.getElementType()->isPointerTy() &&
        SrcSize == CGM.getDataLayout().getTypeAllocSize(Dst.getElementType())) {
      // If the value is supposed to be a pointer, convert it before storing it.
      Src = CoerceIntOrPtrToIntOrPtr(Src, Dst.getElementType(), *this);
      auto *I = Builder.CreateStore(Src, Dst, DstIsVolatile);
      addInstToCurrentSourceAtom(I, Src);
    } else if (llvm::StructType *STy =
                   dyn_cast<llvm::StructType>(Src->getType())) {
      // Prefer scalar stores to first-class aggregate stores.
      Dst = Dst.withElementType(SrcTy);
      for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i) {
        Address EltPtr = Builder.CreateStructGEP(Dst, i);
        llvm::Value *Elt = Builder.CreateExtractValue(Src, i);
        auto *I = Builder.CreateStore(Elt, EltPtr, DstIsVolatile);
        addInstToCurrentSourceAtom(I, Elt);
      }
    } else {
      auto *I =
          Builder.CreateStore(Src, Dst.withElementType(SrcTy), DstIsVolatile);
      addInstToCurrentSourceAtom(I, Src);
    }
  } else if (SrcTy->isIntegerTy()) {
    // If the source is a simple integer, coerce it directly.
    llvm::Type *DstIntTy = Builder.getIntNTy(DstSize.getFixedValue() * 8);
    Src = CoerceIntOrPtrToIntOrPtr(Src, DstIntTy, *this);
    auto *I =
        Builder.CreateStore(Src, Dst.withElementType(DstIntTy), DstIsVolatile);
    addInstToCurrentSourceAtom(I, Src);
  } else {
    // Otherwise do coercion through memory. This is stupid, but
    // simple.

    // Generally SrcSize is never greater than DstSize, since this means we are
    // losing bits. However, this can happen in cases where the structure has
    // additional padding, for example due to a user specified alignment.
    //
    // FIXME: Assert that we aren't truncating non-padding bits when have access
    // to that information.
    RawAddress Tmp =
        CreateTempAllocaForCoercion(*this, SrcTy, Dst.getAlignment());
    Builder.CreateStore(Src, Tmp);
    auto *I = Builder.CreateMemCpy(
        Dst.emitRawPointer(*this), Dst.getAlignment().getAsAlign(),
        Tmp.getPointer(), Tmp.getAlignment().getAsAlign(),
        Builder.CreateTypeSize(IntPtrTy, DstSize));
    addInstToCurrentSourceAtom(I, Src);
  }
}

Address CodeGen::emitAddressAtOffset(CodeGenFunction &CGF, Address addr,
                                   const ABIArgInfo &info) {
  if (unsigned offset = info.getDirectOffset()) {
    addr = addr.withElementType(CGF.Int8Ty);
    addr = CGF.Builder.CreateConstInBoundsByteGEP(
        addr, CharUnits::fromQuantity(offset));
    addr = addr.withElementType(info.getCoerceToType());
  }
  return addr;
}

std::pair<llvm::Value *, bool>
CodeGen::CoerceScalableToFixed(CodeGenFunction &CGF, llvm::FixedVectorType *ToTy,
                      llvm::ScalableVectorType *FromTy, llvm::Value *V,
                      StringRef Name) {
  // If we are casting a scalable i1 predicate vector to a fixed i8
  // vector, first bitcast the source.
  if (FromTy->getElementType()->isIntegerTy(1) &&
      ToTy->getElementType() == CGF.Builder.getInt8Ty()) {
    if (!FromTy->getElementCount().isKnownMultipleOf(8)) {
      FromTy = llvm::ScalableVectorType::get(
          FromTy->getElementType(),
          llvm::alignTo<8>(FromTy->getElementCount().getKnownMinValue()));
      llvm::Value *ZeroVec = llvm::Constant::getNullValue(FromTy);
      V = CGF.Builder.CreateInsertVector(FromTy, ZeroVec, V, uint64_t(0));
    }
    FromTy = llvm::ScalableVectorType::get(
        ToTy->getElementType(),
        FromTy->getElementCount().getKnownMinValue() / 8);
    V = CGF.Builder.CreateBitCast(V, FromTy);
  }
  if (FromTy->getElementType() == ToTy->getElementType()) {
    V->setName(Name + ".coerce");
    V = CGF.Builder.CreateExtractVector(ToTy, V, uint64_t(0), "cast.fixed");
    return {V, true};
  }
  return {V, false};
}

void ClangToLLVMArgMapping::construct(const ASTContext &Context,
                                      const CGFunctionInfo &FI,
                                      bool OnlyRequiredArgs) {
  unsigned IRArgNo = 0;
  bool SwapThisWithSRet = false;
  const ABIArgInfo &RetAI = FI.getReturnInfo();

  if (RetAI.getKind() == ABIArgInfo::Indirect) {
    SwapThisWithSRet = RetAI.isSRetAfterThis();
    SRetArgNo = SwapThisWithSRet ? 1 : IRArgNo++;
  }

  unsigned ArgNo = 0;
  unsigned NumArgs = OnlyRequiredArgs ? FI.getNumRequiredArgs() : FI.arg_size();
  for (CGFunctionInfo::const_arg_iterator I = FI.arg_begin(); ArgNo < NumArgs;
       ++I, ++ArgNo) {
    assert(I != FI.arg_end());
    QualType ArgType = I->type;
    const ABIArgInfo &AI = I->info;
    // Collect data about IR arguments corresponding to Clang argument ArgNo.
    auto &IRArgs = ArgInfo[ArgNo];

    if (AI.getPaddingType())
      IRArgs.PaddingArgIndex = IRArgNo++;

    switch (AI.getKind()) {
    case ABIArgInfo::TargetSpecific:
    case ABIArgInfo::Extend:
    case ABIArgInfo::Direct: {
      // FIXME: handle sseregparm someday...
      llvm::StructType *STy = dyn_cast<llvm::StructType>(AI.getCoerceToType());
      if (AI.isDirect() && AI.getCanBeFlattened() && STy) {
        IRArgs.NumberOfArgs = STy->getNumElements();
      } else {
        IRArgs.NumberOfArgs = 1;
      }
      break;
    }
    case ABIArgInfo::Indirect:
    case ABIArgInfo::IndirectAliased:
      IRArgs.NumberOfArgs = 1;
      break;
    case ABIArgInfo::Ignore:
    case ABIArgInfo::InAlloca:
      // ignore and inalloca doesn't have matching LLVM parameters.
      IRArgs.NumberOfArgs = 0;
      break;
    case ABIArgInfo::CoerceAndExpand:
      IRArgs.NumberOfArgs = AI.getCoerceAndExpandTypeSequence().size();
      break;
    case ABIArgInfo::Expand:
      IRArgs.NumberOfArgs = getExpansionSize(ArgType, Context);
      break;
    }

    if (IRArgs.NumberOfArgs > 0) {
      IRArgs.FirstArgIndex = IRArgNo;
      IRArgNo += IRArgs.NumberOfArgs;
    }

    // Skip over the sret parameter when it comes second.  We already handled it
    // above.
    if (IRArgNo == 1 && SwapThisWithSRet)
      IRArgNo++;
  }
  assert(ArgNo == ArgInfo.size());

  if (FI.usesInAlloca())
    InallocaArgNo = IRArgNo++;

  TotalIRArgs = IRArgNo;
}
