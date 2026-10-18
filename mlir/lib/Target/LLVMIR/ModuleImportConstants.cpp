//===- ModuleImportConstants.cpp - LLVM global and constant value import ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements LLVM global and constant value import into the LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/ModuleImport.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Target/LLVMIR/Import.h"

#include "AttrKindDetail.h"
#include "ModuleImportDetail.h"
#include "DebugImporter.h"
#include "LoopAnnotationImporter.h"

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Target/LLVMIR/DataLayoutImporter.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Comdat.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugProgramInstruction.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/ModRef.h"
#include <optional>

using namespace mlir;
using namespace mlir::LLVM;
using namespace mlir::LLVM::detail;

#include "mlir/Dialect/LLVMIR/LLVMConversionEnumsFromLLVM.inc"

/// Prefix used for symbols of nameless llvm globals.
static constexpr StringRef getNamelessGlobalPrefix() {
  return "mlir.llvm.nameless_global";
}

/// Returns `type` if it is a builtin integer or floating-point vector type that
/// can be used to create an attribute or nullptr otherwise. If provided,
/// `arrayShape` is added to the shape of the vector to create an attribute that
/// matches an array of vectors.
static Type getVectorTypeForAttr(Type type, ArrayRef<int64_t> arrayShape = {}) {
  if (!LLVM::isCompatibleVectorType(type))
    return {};

  llvm::ElementCount numElements = LLVM::getVectorNumElements(type);
  if (numElements.isScalable()) {
    emitError(UnknownLoc::get(type.getContext()))
        << "scalable vectors not supported";
    return {};
  }

  // An LLVM dialect vector can only contain scalars.
  Type elementType = cast<VectorType>(type).getElementType();
  if (!elementType.isIntOrFloat())
    return {};

  SmallVector<int64_t> shape(arrayShape);
  shape.push_back(numElements.getKnownMinValue());
  return VectorType::get(shape, elementType);
}

Type ModuleImport::getBuiltinTypeForAttr(Type type) {
  if (!type)
    return {};

  // Return builtin integer and floating-point types as is.
  if (type.isIntOrFloat())
    return type;

  // Return builtin vectors of integer and floating-point types as is.
  if (Type vectorType = getVectorTypeForAttr(type))
    return vectorType;

  // Multi-dimensional array types are converted to tensors or vectors,
  // depending on the innermost type being a scalar or a vector.
  SmallVector<int64_t> arrayShape;
  while (auto arrayType = dyn_cast<LLVMArrayType>(type)) {
    arrayShape.push_back(arrayType.getNumElements());
    type = arrayType.getElementType();
  }
  if (type.isIntOrFloat())
    return RankedTensorType::get(arrayShape, type);
  return getVectorTypeForAttr(type, arrayShape);
}

/// Returns an integer or float attribute for the provided scalar constant
/// `constScalar` or nullptr if the conversion fails.
static TypedAttr getScalarConstantAsAttr(OpBuilder &builder,
                                         llvm::Constant *constScalar) {
  MLIRContext *context = builder.getContext();

  if (constScalar->getType()->isVectorTy())
    return {};

  // Convert scalar integers.
  if (auto *constInt = dyn_cast<llvm::ConstantInt>(constScalar)) {
    return builder.getIntegerAttr(
        IntegerType::get(context, constInt->getBitWidth()),
        constInt->getValue());
  }

  // Convert scalar floats.
  if (auto *constFloat = dyn_cast<llvm::ConstantFP>(constScalar)) {
    llvm::Type *type = constFloat->getType();
    FloatType floatType =
        type->isBFloatTy()
            ? BFloat16Type::get(context)
            : LLVM::detail::getFloatType(context, type->getScalarSizeInBits());
    if (!floatType) {
      emitError(UnknownLoc::get(builder.getContext()))
          << "unexpected floating-point type";
      return {};
    }
    return builder.getFloatAttr(floatType, constFloat->getValueAPF());
  }
  return {};
}

/// Returns an integer or float attribute array for the provided constant
/// sequence `constSequence` or nullptr if the conversion fails.
static SmallVector<Attribute>
getSequenceConstantAsAttrs(OpBuilder &builder,
                           llvm::ConstantDataSequential *constSequence) {
  SmallVector<Attribute> elementAttrs;
  elementAttrs.reserve(constSequence->getNumElements());
  for (auto idx : llvm::seq<int64_t>(0, constSequence->getNumElements())) {
    llvm::Constant *constElement = constSequence->getElementAsConstant(idx);
    elementAttrs.push_back(getScalarConstantAsAttr(builder, constElement));
  }
  return elementAttrs;
}

Attribute ModuleImport::getConstantAsAttr(llvm::Constant *constant) {
  // Convert scalar constants.
  if (Attribute scalarAttr = getScalarConstantAsAttr(builder, constant))
    return scalarAttr;

  // Returns the static shape of the provided type if possible.
  auto getConstantShape = [&](llvm::Type *type) {
    return llvm::dyn_cast_if_present<ShapedType>(
        getBuiltinTypeForAttr(convertType(type)));
  };

  // Convert constant vector splat values.
  if (isa<llvm::ConstantInt, llvm::ConstantFP>(constant)) {
    assert(constant->getType()->isVectorTy() && "expected a vector splat");
    auto shape = getConstantShape(constant->getType());
    if (!shape)
      return {};
    Attribute splatAttr =
        getScalarConstantAsAttr(builder, constant->getSplatValue());
    return SplatElementsAttr::get(shape, splatAttr);
  }

  // Convert one-dimensional constant arrays or vectors that store 1/2/4/8-byte
  // integer or half/bfloat/float/double values.
  if (auto *constArray = dyn_cast<llvm::ConstantDataSequential>(constant)) {
    if (constArray->isString())
      return builder.getStringAttr(constArray->getAsString());
    auto shape = getConstantShape(constArray->getType());
    if (!shape)
      return {};
    // Convert splat constants to splat elements attributes.
    auto *constVector = dyn_cast<llvm::ConstantDataVector>(constant);
    if (constVector && constVector->isSplat()) {
      // A vector is guaranteed to have at least size one.
      Attribute splatAttr = getScalarConstantAsAttr(
          builder, constVector->getElementAsConstant(0));
      return SplatElementsAttr::get(shape, splatAttr);
    }
    // Convert non-splat constants to dense elements attributes.
    SmallVector<Attribute> elementAttrs =
        getSequenceConstantAsAttrs(builder, constArray);
    return DenseElementsAttr::get(shape, elementAttrs);
  }

  // Convert multi-dimensional constant aggregates that store all kinds of
  // integer and floating-point types.
  if (auto *constAggregate = dyn_cast<llvm::ConstantAggregate>(constant)) {
    auto shape = getConstantShape(constAggregate->getType());
    if (!shape)
      return {};
    // Collect the aggregate elements in depths first order.
    SmallVector<Attribute> elementAttrs;
    SmallVector<llvm::Constant *> workList = {constAggregate};
    while (!workList.empty()) {
      llvm::Constant *current = workList.pop_back_val();
      // Append any nested aggregates in reverse order to ensure the head
      // element of the nested aggregates is at the back of the work list.
      if (auto *constAggregate = dyn_cast<llvm::ConstantAggregate>(current)) {
        for (auto idx :
             reverse(llvm::seq<int64_t>(0, constAggregate->getNumOperands())))
          workList.push_back(constAggregate->getAggregateElement(idx));
        continue;
      }
      // Append the elements of nested constant arrays or vectors that store
      // 1/2/4/8-byte integer or half/bfloat/float/double values.
      if (auto *constArray = dyn_cast<llvm::ConstantDataSequential>(current)) {
        SmallVector<Attribute> attrs =
            getSequenceConstantAsAttrs(builder, constArray);
        elementAttrs.append(attrs.begin(), attrs.end());
        continue;
      }
      // Append nested scalar constants that store all kinds of integer and
      // floating-point types.
      if (Attribute scalarAttr = getScalarConstantAsAttr(builder, current)) {
        elementAttrs.push_back(scalarAttr);
        continue;
      }
      // Bail if the aggregate contains a unsupported constant type such as a
      // constant expression.
      return {};
    }
    return DenseElementsAttr::get(shape, elementAttrs);
  }

  // Convert zero aggregates.
  if (auto *constZero = dyn_cast<llvm::ConstantAggregateZero>(constant)) {
    auto shape = llvm::dyn_cast_if_present<ShapedType>(
        getBuiltinTypeForAttr(convertType(constZero->getType())));
    if (!shape)
      return {};
    // Convert zero aggregates with a static shape to splat elements attributes.
    Attribute splatAttr = builder.getZeroAttr(shape.getElementType());
    assert(splatAttr && "expected non-null zero attribute for scalar types");
    return SplatElementsAttr::get(shape, splatAttr);
  }
  return {};
}

FlatSymbolRefAttr
ModuleImport::getOrCreateNamelessSymbolName(llvm::GlobalVariable *globalVar) {
  assert(globalVar->getName().empty() &&
         "expected to work with a nameless global");
  auto [it, success] = namelessGlobals.try_emplace(globalVar);
  if (!success)
    return it->second;

  // Make sure the symbol name does not clash with an existing symbol.
  SmallString<128> globalName = SymbolTable::generateSymbolName<128>(
      getNamelessGlobalPrefix(),
      [this](StringRef newName) { return llvmModule->getNamedValue(newName); },
      namelessGlobalId);
  auto symbolRef = FlatSymbolRefAttr::get(context, globalName);
  it->getSecond() = symbolRef;
  return symbolRef;
}

OpBuilder::InsertionGuard ModuleImport::setGlobalInsertionPoint() {
  OpBuilder::InsertionGuard guard(builder);
  if (globalInsertionOp)
    builder.setInsertionPointAfter(globalInsertionOp);
  else
    builder.setInsertionPointToStart(mlirModule.getBody());
  return guard;
}

LogicalResult ModuleImport::convertAlias(llvm::GlobalAlias *alias) {
  // Insert the alias after the last one or at the start of the module.
  OpBuilder::InsertionGuard guard = setGlobalInsertionPoint();

  Type type = convertType(alias->getValueType());
  AliasOp aliasOp = AliasOp::create(
      builder, mlirModule.getLoc(), type,
      convertLinkageFromLLVM(alias->getLinkage()), alias->getName(),
      /*dsoLocal=*/alias->isDSOLocal(),
      convertThreadLocalModeFromLLVM(alias->getThreadLocalMode()),
      /*attrs=*/ArrayRef<NamedAttribute>());
  globalInsertionOp = aliasOp;

  clearRegionState();
  Block *block = builder.createBlock(&aliasOp.getInitializerRegion());
  setConstantInsertionPointToStart(block);
  FailureOr<Value> initializer = convertConstantExpr(alias->getAliasee());
  if (failed(initializer))
    return failure();
  ReturnOp::create(builder, aliasOp.getLoc(), *initializer);

  if (alias->hasAtLeastLocalUnnamedAddr())
    aliasOp.setUnnamedAddr(convertUnnamedAddrFromLLVM(alias->getUnnamedAddr()));
  aliasOp.setVisibility_(convertVisibilityFromLLVM(alias->getVisibility()));

  return success();
}

LogicalResult ModuleImport::convertIFunc(llvm::GlobalIFunc *ifunc) {
  OpBuilder::InsertionGuard guard = setGlobalInsertionPoint();

  Type type = convertType(ifunc->getValueType());
  llvm::Constant *resolver = ifunc->getResolver();
  Type resolverType = convertType(resolver->getType());
  IFuncOp::create(builder, mlirModule.getLoc(), ifunc->getName(), type,
                  resolver->getName(), resolverType,
                  convertLinkageFromLLVM(ifunc->getLinkage()),
                  ifunc->isDSOLocal(), ifunc->getAddressSpace(),
                  convertUnnamedAddrFromLLVM(ifunc->getUnnamedAddr()),
                  convertVisibilityFromLLVM(ifunc->getVisibility()));
  return success();
}

/// Converts LLVM attributes from `globalVar` into MLIR attributes and adds them
/// to `globalOp` as target-specific attributes.
static void processTargetSpecificAttrs(llvm::GlobalVariable *globalVar,
                                       GlobalOp globalOp) {
  ArrayAttr targetSpecificAttrs = convertLLVMAttributesToMLIR(
      globalOp.getLoc(), globalOp.getContext(), globalVar->getAttributes());
  if (!targetSpecificAttrs.empty())
    globalOp.setTargetSpecificAttrsAttr(targetSpecificAttrs);
}

LogicalResult ModuleImport::convertGlobal(llvm::GlobalVariable *globalVar) {
  // Insert the global after the last one or at the start of the module.
  OpBuilder::InsertionGuard guard = setGlobalInsertionPoint();

  Attribute valueAttr;
  if (globalVar->hasInitializer())
    valueAttr = getConstantAsAttr(globalVar->getInitializer());
  Type type = convertType(globalVar->getValueType());

  uint64_t alignment = 0;
  llvm::MaybeAlign maybeAlign = globalVar->getAlign();
  if (maybeAlign.has_value()) {
    llvm::Align align = *maybeAlign;
    alignment = align.value();
  }

  // Get the global expression associated with this global variable and convert
  // it.
  SmallVector<Attribute> globalExpressionAttrs;
  SmallVector<llvm::DIGlobalVariableExpression *> globalExpressions;
  globalVar->getDebugInfo(globalExpressions);

  for (llvm::DIGlobalVariableExpression *expr : globalExpressions) {
    DIGlobalVariableExpressionAttr globalExpressionAttr =
        debugImporter->translateGlobalVariableExpression(expr);
    globalExpressionAttrs.push_back(globalExpressionAttr);
  }

  // Workaround to support LLVM's nameless globals. MLIR, in contrast to LLVM,
  // always requires a symbol name.
  StringRef globalName = globalVar->getName();
  if (globalName.empty())
    globalName = getOrCreateNamelessSymbolName(globalVar).getValue();

  GlobalOp globalOp = GlobalOp::create(
      builder, mlirModule.getLoc(), type, globalVar->isConstant(),
      convertLinkageFromLLVM(globalVar->getLinkage()), StringRef(globalName),
      valueAttr, alignment, /*addrSpace=*/globalVar->getAddressSpace(),
      /*dsoLocal=*/globalVar->isDSOLocal(),
      convertThreadLocalModeFromLLVM(globalVar->getThreadLocalMode()),
      /*comdat=*/SymbolRefAttr(),
      /*attrs=*/ArrayRef<NamedAttribute>(), /*dbgExprs=*/globalExpressionAttrs);
  globalInsertionOp = globalOp;

  if (globalVar->hasInitializer() && !valueAttr) {
    clearRegionState();
    Block *block = builder.createBlock(&globalOp.getInitializerRegion());
    setConstantInsertionPointToStart(block);
    FailureOr<Value> initializer =
        convertConstantExpr(globalVar->getInitializer());
    if (failed(initializer))
      return failure();
    ReturnOp::create(builder, globalOp.getLoc(), *initializer);
  }
  if (globalVar->hasAtLeastLocalUnnamedAddr()) {
    globalOp.setUnnamedAddr(
        convertUnnamedAddrFromLLVM(globalVar->getUnnamedAddr()));
  }
  if (globalVar->hasSection())
    globalOp.setSection(globalVar->getSection());
  globalOp.setVisibility_(
      convertVisibilityFromLLVM(globalVar->getVisibility()));

  if (globalVar->hasComdat())
    globalOp.setComdatAttr(comdatMapping.lookup(globalVar->getComdat()));

  processTargetSpecificAttrs(globalVar, globalOp);

  return success();
}

LogicalResult
ModuleImport::convertGlobalCtorsAndDtors(llvm::GlobalVariable *globalVar) {
  if (!globalVar->hasInitializer() || !globalVar->hasAppendingLinkage())
    return failure();
  llvm::Constant *initializer = globalVar->getInitializer();

  bool knownInit = isa<llvm::ConstantArray>(initializer) ||
                   isa<llvm::ConstantAggregateZero>(initializer);
  if (!knownInit)
    return failure();

  // ConstantAggregateZero does not engage with the operand initialization
  // in the loop that follows - there should be no operands. This implies
  // empty ctor/dtor lists.
  if (auto *caz = dyn_cast<llvm::ConstantAggregateZero>(initializer)) {
    if (caz->getElementCount().getFixedValue() != 0)
      return failure();
  }

  SmallVector<Attribute> funcs;
  SmallVector<int32_t> priorities;
  SmallVector<Attribute> dataList;
  for (llvm::Value *operand : initializer->operands()) {
    auto *aggregate = dyn_cast<llvm::ConstantAggregate>(operand);
    if (!aggregate || aggregate->getNumOperands() != 3)
      return failure();

    auto *priority = dyn_cast<llvm::ConstantInt>(aggregate->getOperand(0));
    auto *func = dyn_cast<llvm::Function>(aggregate->getOperand(1));
    auto *data = dyn_cast<llvm::Constant>(aggregate->getOperand(2));
    if (!priority || !func || !data)
      return failure();

    auto *gv = dyn_cast_or_null<llvm::GlobalValue>(data);
    Attribute dataAttr;
    if (gv)
      dataAttr = FlatSymbolRefAttr::get(context, gv->getName());
    else if (data->isNullValue())
      dataAttr = ZeroAttr::get(context);
    else
      return failure();

    funcs.push_back(FlatSymbolRefAttr::get(context, func->getName()));
    priorities.push_back(priority->getValue().getZExtValue());
    dataList.push_back(dataAttr);
  }

  // Insert the global after the last one or at the start of the module.
  OpBuilder::InsertionGuard guard = setGlobalInsertionPoint();

  if (globalVar->getName() == getGlobalCtorsVarName()) {
    globalInsertionOp = LLVM::GlobalCtorsOp::create(
        builder, mlirModule.getLoc(), builder.getArrayAttr(funcs),
        builder.getI32ArrayAttr(priorities), builder.getArrayAttr(dataList));
    return success();
  }
  globalInsertionOp = LLVM::GlobalDtorsOp::create(
      builder, mlirModule.getLoc(), builder.getArrayAttr(funcs),
      builder.getI32ArrayAttr(priorities), builder.getArrayAttr(dataList));
  return success();
}

SetVector<llvm::Constant *>
ModuleImport::getConstantsToConvert(llvm::Constant *constant) {
  // Return the empty set if the constant has been translated before.
  if (valueMapping.contains(constant))
    return {};

  // Traverse the constants in post-order and stop the traversal if a constant
  // already has a `valueMapping` from an earlier constant translation or if the
  // constant is traversed a second time.
  SetVector<llvm::Constant *> orderedSet;
  SetVector<llvm::Constant *> workList;
  DenseMap<llvm::Constant *, SmallVector<llvm::Constant *>> adjacencyLists;
  workList.insert(constant);
  while (!workList.empty()) {
    llvm::Constant *current = workList.back();
    // References of global objects are just pointers to the object. Avoid
    // walking the elements of these here.
    if (isa<llvm::GlobalObject>(current) || isa<llvm::GlobalAlias>(current)) {
      orderedSet.insert(current);
      workList.pop_back();
      continue;
    }

    // Collect all dependencies of the current constant and add them to the
    // adjacency list if none has been computed before.
    auto [adjacencyIt, inserted] = adjacencyLists.try_emplace(current);
    if (inserted) {
      // Add all constant operands to the adjacency list and skip any other
      // values such as basic block addresses.
      for (llvm::Value *operand : current->operands())
        if (auto *constDependency = dyn_cast<llvm::Constant>(operand))
          adjacencyIt->getSecond().push_back(constDependency);
      // Use the getElementValue method to add the dependencies of zero
      // initialized aggregate constants since they do not take any operands.
      if (auto *constAgg = dyn_cast<llvm::ConstantAggregateZero>(current)) {
        unsigned numElements = constAgg->getElementCount().getFixedValue();
        for (unsigned i = 0, e = numElements; i != e; ++i)
          adjacencyIt->getSecond().push_back(constAgg->getElementValue(i));
      }
    }
    // Add the current constant to the `orderedSet` of the traversed nodes if
    // all its dependencies have been traversed before. Additionally, remove the
    // constant from the `workList` and continue the traversal.
    if (adjacencyIt->getSecond().empty()) {
      orderedSet.insert(current);
      workList.pop_back();
      continue;
    }
    // Add the next dependency from the adjacency list to the `workList` and
    // continue the traversal. Remove the dependency from the adjacency list to
    // mark that it has been processed. Only enqueue the dependency if it has no
    // `valueMapping` from an earlier translation and if it has not been
    // enqueued before.
    llvm::Constant *dependency = adjacencyIt->getSecond().pop_back_val();
    if (valueMapping.contains(dependency) || workList.contains(dependency) ||
        orderedSet.contains(dependency))
      continue;
    workList.insert(dependency);
  }

  return orderedSet;
}

FailureOr<Value> ModuleImport::convertConstant(llvm::Constant *constant) {
  Location loc = UnknownLoc::get(context);

  // Convert constants that can be represented as attributes.
  if (Attribute attr = getConstantAsAttr(constant)) {
    Type type = convertType(constant->getType());
    if (auto symbolRef = dyn_cast<FlatSymbolRefAttr>(attr)) {
      return AddressOfOp::create(builder, loc, type, symbolRef.getValue())
          .getResult();
    }
    return ConstantOp::create(builder, loc, type, attr).getResult();
  }

  // Convert null pointer constants.
  if (auto *nullPtr = dyn_cast<llvm::ConstantPointerNull>(constant)) {
    Type type = convertType(nullPtr->getType());
    return ZeroOp::create(builder, loc, type).getResult();
  }

  // Convert none token constants.
  if (isa<llvm::ConstantTokenNone>(constant)) {
    return NoneTokenOp::create(builder, loc).getResult();
  }

  // Convert poison.
  if (auto *poisonVal = dyn_cast<llvm::PoisonValue>(constant)) {
    Type type = convertType(poisonVal->getType());
    return PoisonOp::create(builder, loc, type).getResult();
  }

  // Convert undef.
  if (auto *undefVal = dyn_cast<llvm::UndefValue>(constant)) {
    Type type = convertType(undefVal->getType());
    return UndefOp::create(builder, loc, type).getResult();
  }

  // Convert dso_local_equivalent.
  if (auto *dsoLocalEquivalent = dyn_cast<llvm::DSOLocalEquivalent>(constant)) {
    Type type = convertType(dsoLocalEquivalent->getType());
    return DSOLocalEquivalentOp::create(
               builder, loc, type,
               FlatSymbolRefAttr::get(
                   builder.getContext(),
                   dsoLocalEquivalent->getGlobalValue()->getName()))
        .getResult();
  }

  // Convert global variable accesses.
  if (auto *globalObj = dyn_cast<llvm::GlobalObject>(constant)) {
    Type type = convertType(globalObj->getType());
    StringRef globalName = globalObj->getName();
    FlatSymbolRefAttr symbolRef;
    // Empty names are only allowed for global variables.
    if (globalName.empty())
      symbolRef =
          getOrCreateNamelessSymbolName(cast<llvm::GlobalVariable>(globalObj));
    else
      symbolRef = FlatSymbolRefAttr::get(context, globalName);
    return AddressOfOp::create(builder, loc, type, symbolRef).getResult();
  }

  // Convert global alias accesses.
  if (auto *globalAliasObj = dyn_cast<llvm::GlobalAlias>(constant)) {
    Type type = convertType(globalAliasObj->getType());
    StringRef aliaseeName = globalAliasObj->getName();
    FlatSymbolRefAttr symbolRef = FlatSymbolRefAttr::get(context, aliaseeName);
    return AddressOfOp::create(builder, loc, type, symbolRef).getResult();
  }

  // Convert constant expressions.
  if (auto *constExpr = dyn_cast<llvm::ConstantExpr>(constant)) {
    // Convert the constant expression to a temporary LLVM instruction and
    // translate it using the `processInstruction` method. Delete the
    // instruction after the translation and remove it from `valueMapping`,
    // since later calls to `getAsInstruction` may return the same address
    // resulting in a conflicting `valueMapping` entry.
    llvm::Instruction *inst = constExpr->getAsInstruction();
    llvm::scope_exit guard([&]() {
      assert(!noResultOpMapping.contains(inst) &&
             "expected constant expression to return a result");
      valueMapping.erase(inst);
      inst->deleteValue();
    });
    // Note: `processInstruction` does not call `convertConstant` recursively
    // since all constant dependencies have been converted before.
    assert(llvm::all_of(inst->operands(), [&](llvm::Value *value) {
      return valueMapping.contains(value);
    }));
    if (failed(processInstruction(inst)))
      return failure();
    return lookupValue(inst);
  }

  // Convert zero-initialized aggregates to ZeroOp.
  if (auto *aggregateZero = dyn_cast<llvm::ConstantAggregateZero>(constant)) {
    Type type = convertType(aggregateZero->getType());
    return ZeroOp::create(builder, loc, type).getResult();
  }

  // Convert aggregate constants.
  if (auto *constAgg = dyn_cast<llvm::ConstantAggregate>(constant)) {
    // Lookup the aggregate elements that have been converted before.
    SmallVector<Value> elementValues;

    elementValues.reserve(constAgg->getNumOperands());
    for (llvm::Value *operand : constAgg->operands())
      elementValues.push_back(lookupValue(operand));

    assert(llvm::count(elementValues, nullptr) == 0 &&
           "expected all elements have been converted before");

    // Generate an UndefOp as root value and insert the aggregate elements.
    Type rootType = convertType(constant->getType());
    bool isArrayOrStruct = isa<LLVMArrayType, LLVMStructType>(rootType);
    assert((isArrayOrStruct || LLVM::isCompatibleVectorType(rootType)) &&
           "unrecognized aggregate type");
    Value root = UndefOp::create(builder, loc, rootType);
    for (const auto &it : llvm::enumerate(elementValues)) {
      if (isArrayOrStruct) {
        root =
            InsertValueOp::create(builder, loc, root, it.value(), it.index());
      } else {
        Attribute indexAttr = builder.getI32IntegerAttr(it.index());
        Value indexValue =
            ConstantOp::create(builder, loc, builder.getI32Type(), indexAttr);
        root = InsertElementOp::create(builder, loc, rootType, root, it.value(),
                                       indexValue);
      }
    }
    return root;
  }

  if (auto *constTargetNone = dyn_cast<llvm::ConstantTargetNone>(constant)) {
    LLVMTargetExtType targetExtType =
        cast<LLVMTargetExtType>(convertType(constTargetNone->getType()));
    assert(targetExtType.hasProperty(LLVMTargetExtType::HasZeroInit) &&
           "target extension type does not support zero-initialization");
    // Create llvm.mlir.zero operation to represent zero-initialization of
    // target extension type.
    return LLVM::ZeroOp::create(builder, loc, targetExtType).getRes();
  }

  if (auto *blockAddr = dyn_cast<llvm::BlockAddress>(constant)) {
    auto fnSym =
        FlatSymbolRefAttr::get(context, blockAddr->getFunction()->getName());
    auto blockTag =
        BlockTagAttr::get(context, blockAddr->getBasicBlock()->getNumber());
    return BlockAddressOp::create(
               builder, loc, convertType(blockAddr->getType()),
               BlockAddressAttr::get(context, fnSym, blockTag))
        .getRes();
  }

  StringRef error = "";

  if (isa<llvm::ConstantPtrAuth>(constant))
    error = " since ptrauth(...) is unsupported";

  if (isa<llvm::NoCFIValue>(constant))
    error = " since no_cfi is unsupported";

  if (isa<llvm::GlobalValue>(constant))
    error = " since global value is unsupported";

  return emitError(loc) << "unhandled constant: " << diag(*constant) << error;
}

FailureOr<Value> ModuleImport::convertConstantExpr(llvm::Constant *constant) {
  // Only call the function for constants that have not been translated before
  // since it updates the constant insertion point assuming the converted
  // constant has been introduced at the end of the constant section.
  assert(!valueMapping.contains(constant) &&
         "expected constant has not been converted before");
  assert(constantInsertionBlock &&
         "expected the constant insertion block to be non-null");

  // Insert the constant after the last one or at the start of the entry block.
  OpBuilder::InsertionGuard guard(builder);
  if (!constantInsertionOp)
    builder.setInsertionPointToStart(constantInsertionBlock);
  else
    builder.setInsertionPointAfter(constantInsertionOp);

  // Convert all constants of the expression and add them to `valueMapping`.
  SetVector<llvm::Constant *> constantsToConvert =
      getConstantsToConvert(constant);
  for (llvm::Constant *constantToConvert : constantsToConvert) {
    FailureOr<Value> converted = convertConstant(constantToConvert);
    if (failed(converted))
      return failure();
    mapValue(constantToConvert, *converted);
  }

  // Update the constant insertion point and return the converted constant.
  Value result = lookupValue(constant);
  constantInsertionOp = result.getDefiningOp();
  return result;
}

FailureOr<Value> ModuleImport::convertValue(llvm::Value *value) {
  // Return the mapped value if it has been converted before.
  auto it = valueMapping.find(value);
  if (it != valueMapping.end())
    return it->getSecond();

  // `llvm::MetadataAsValue` operands (e.g. the rounding-mode / FP-exception
  // MDString arguments used by the constrained floating-point intrinsics, or
  // the named-register MDNode used by `llvm.read_register`) are lifted into a
  // `llvm.mlir.metadata_as_value` SSA op carrying the corresponding metadata
  // attribute.
  if (auto *mdAsVal = dyn_cast<llvm::MetadataAsValue>(value)) {
    llvm::Metadata *md = mdAsVal->getMetadata();
    Attribute mdAttr = convertMetadataToAttr(md);
    if (!mdAttr)
      return emitError(mlirModule.getLoc())
             << "unsupported metadata: " << diagMD(md, llvmModule.get());
    Value result =
        MetadataAsValueOp::create(builder, UnknownLoc::get(context), mdAttr)
            .getRes();
    mapValue(value, result);
    return result;
  }

  // Convert constants such as immediate values that have no mapping yet.
  if (auto *constant = dyn_cast<llvm::Constant>(value))
    return convertConstantExpr(constant);

  Location loc = UnknownLoc::get(context);
  if (auto *inst = dyn_cast<llvm::Instruction>(value))
    loc = translateLoc(inst->getDebugLoc());
  return emitError(loc) << "unhandled value: " << diag(*value);
}

FailureOr<Value> ModuleImport::convertMetadataValue(llvm::Value *value) {
  // A value may be wrapped as metadata, for example, when passed to a debug
  // intrinsic. Unwrap these values before the conversion.
  auto *nodeAsVal = dyn_cast<llvm::MetadataAsValue>(value);
  if (!nodeAsVal)
    return failure();
  auto *node = dyn_cast<llvm::ValueAsMetadata>(nodeAsVal->getMetadata());
  if (!node)
    return failure();
  value = node->getValue();

  // Return the mapped value if it has been converted before.
  auto it = valueMapping.find(value);
  if (it != valueMapping.end())
    return it->getSecond();

  // Convert constants such as immediate values that have no mapping yet.
  if (auto *constant = dyn_cast<llvm::Constant>(value))
    return convertConstantExpr(constant);
  return failure();
}

FailureOr<SmallVector<Value>>
ModuleImport::convertValues(ArrayRef<llvm::Value *> values) {
  SmallVector<Value> remapped;
  remapped.reserve(values.size());
  for (llvm::Value *value : values) {
    FailureOr<Value> converted = convertValue(value);
    if (failed(converted))
      return failure();
    remapped.push_back(*converted);
  }
  return remapped;
}

LogicalResult ModuleImport::convertIntrinsicArguments(
    ArrayRef<llvm::Value *> values, ArrayRef<llvm::OperandBundleUse> opBundles,
    bool requiresOpBundles, ArrayRef<unsigned> immArgPositions,
    ArrayRef<StringLiteral> immArgAttrNames, SmallVectorImpl<Value> &valuesOut,
    SmallVectorImpl<NamedAttribute> &attrsOut) {
  assert(immArgPositions.size() == immArgAttrNames.size() &&
         "LLVM `immArgPositions` and MLIR `immArgAttrNames` should have equal "
         "length");

  SmallVector<llvm::Value *> operands(values);
  for (auto [immArgPos, immArgName] :
       llvm::zip(immArgPositions, immArgAttrNames)) {
    auto &value = operands[immArgPos];
    auto *constant = llvm::cast<llvm::Constant>(value);
    auto attr = getScalarConstantAsAttr(builder, constant);
    assert(attr && attr.getType().isIntOrFloat() &&
           "expected immarg to be float or integer constant");
    auto nameAttr = StringAttr::get(attr.getContext(), immArgName);
    attrsOut.push_back({nameAttr, attr});
    // Mark matched attribute values as null (so they can be removed below).
    value = nullptr;
  }

  for (llvm::Value *value : operands) {
    if (!value)
      continue;
    auto mlirValue = convertValue(value);
    if (failed(mlirValue))
      return failure();
    valuesOut.push_back(*mlirValue);
  }

  SmallVector<int> opBundleSizes;
  SmallVector<Attribute> opBundleTagAttrs;
  if (requiresOpBundles) {
    opBundleSizes.reserve(opBundles.size());
    opBundleTagAttrs.reserve(opBundles.size());

    for (const llvm::OperandBundleUse &bundle : opBundles) {
      opBundleSizes.push_back(bundle.Inputs.size());
      opBundleTagAttrs.push_back(StringAttr::get(context, bundle.getTagName()));

      for (const llvm::Use &opBundleOperand : bundle.Inputs) {
        auto operandMlirValue = convertValue(opBundleOperand.get());
        if (failed(operandMlirValue))
          return failure();
        valuesOut.push_back(*operandMlirValue);
      }
    }

    auto opBundleSizesAttr = DenseI32ArrayAttr::get(context, opBundleSizes);
    auto opBundleSizesAttrNameAttr =
        StringAttr::get(context, LLVMDialect::getOpBundleSizesAttrName());
    attrsOut.push_back({opBundleSizesAttrNameAttr, opBundleSizesAttr});

    auto opBundleTagsAttr = ArrayAttr::get(context, opBundleTagAttrs);
    auto opBundleTagsAttrNameAttr =
        StringAttr::get(context, LLVMDialect::getOpBundleTagsAttrName());
    attrsOut.push_back({opBundleTagsAttrNameAttr, opBundleTagsAttr});
  }

  return success();
}

IntegerAttr ModuleImport::matchIntegerAttr(llvm::Value *value) {
  IntegerAttr integerAttr;
  FailureOr<Value> converted = convertValue(value);
  bool success = succeeded(converted) &&
                 matchPattern(*converted, m_Constant(&integerAttr));
  assert(success && "expected a constant integer value");
  (void)success;
  return integerAttr;
}

FloatAttr ModuleImport::matchFloatAttr(llvm::Value *value) {
  FloatAttr floatAttr;
  FailureOr<Value> converted = convertValue(value);
  bool success =
      succeeded(converted) && matchPattern(*converted, m_Constant(&floatAttr));
  assert(success && "expected a constant float value");
  (void)success;
  return floatAttr;
}

DILocalVariableAttr ModuleImport::matchLocalVariableAttr(
    llvm::PointerUnion<llvm::Value *, llvm::DILocalVariable *> valOrVariable) {
  llvm::DILocalVariable *node = nullptr;
  if (auto *value = dyn_cast<llvm::Value *>(valOrVariable)) {
    auto *nodeAsVal = cast<llvm::MetadataAsValue>(value);
    node = cast<llvm::DILocalVariable>(nodeAsVal->getMetadata());
  } else {
    node = cast<llvm::DILocalVariable *>(valOrVariable);
  }
  return debugImporter->translate(node);
}

DILabelAttr ModuleImport::matchLabelAttr(llvm::Value *value) {
  auto *nodeAsVal = cast<llvm::MetadataAsValue>(value);
  auto *node = cast<llvm::DILabel>(nodeAsVal->getMetadata());
  return debugImporter->translate(node);
}

FPExceptionBehaviorAttr
ModuleImport::matchFPExceptionBehaviorAttr(llvm::Value *value) {
  auto *metadata = cast<llvm::MetadataAsValue>(value);
  auto *mdstr = cast<llvm::MDString>(metadata->getMetadata());
  std::optional<llvm::fp::ExceptionBehavior> optLLVM =
      llvm::convertStrToExceptionBehavior(mdstr->getString());
  assert(optLLVM && "Expecting FP exception behavior");
  return builder.getAttr<FPExceptionBehaviorAttr>(
      convertFPExceptionBehaviorFromLLVM(*optLLVM));
}

RoundingModeAttr ModuleImport::matchRoundingModeAttr(llvm::Value *value) {
  auto *metadata = cast<llvm::MetadataAsValue>(value);
  auto *mdstr = cast<llvm::MDString>(metadata->getMetadata());
  std::optional<llvm::RoundingMode> optLLVM =
      llvm::convertStrToRoundingMode(mdstr->getString());
  assert(optLLVM && "Expecting rounding mode");
  return builder.getAttr<RoundingModeAttr>(
      convertRoundingModeFromLLVM(*optLLVM));
}

