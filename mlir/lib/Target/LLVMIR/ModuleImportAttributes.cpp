//===- ModuleImportAttributes.cpp - LLVM attribute import ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements LLVM attribute and metadata import into the LLVM
// dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Target/LLVMIR/ModuleImport.h"

#include "AttrKindDetail.h"
#include "DebugImporter.h"
#include "LoopAnnotationImporter.h"
#include "ModuleImportDetail.h"

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

// Utility to print an LLVM value as a string for passing to emitError().
// FIXME: Diagnostic should be able to natively handle types that have
// operator << (raw_ostream&) defined.
std::string mlir::LLVM::detail::diag(const llvm::Value &value) {
  std::string str;
  llvm::raw_string_ostream os(str);
  os << value;
  return str;
}

// Utility to print an LLVM metadata node as a string for passing
// to emitError(). The module argument is needed to print the nodes
// canonically numbered.
std::string mlir::LLVM::detail::diagMD(const llvm::Metadata *node,
                                       const llvm::Module *module) {
  std::string str;
  llvm::raw_string_ostream os(str);
  node->print(os, module, /*IsForDebug=*/true);
  return str;
}

FlatSymbolRefAttr
ModuleImport::getMetadataGlobalValueSymbolRef(llvm::GlobalValue *global) {
  if (auto *globalVar = dyn_cast<llvm::GlobalVariable>(global)) {
    StringRef name = globalVar->getName();
    if (name.empty())
      return getOrCreateNamelessSymbolName(globalVar);
    if (name == getGlobalCtorsVarName() || name == getGlobalDtorsVarName())
      return {};
  }

  if (auto *func = dyn_cast<llvm::Function>(global)) {
    // Intrinsics with a dedicated import conversion do not have an imported
    // function declaration that a metadata symbol reference could resolve to.
    if (func->isIntrinsic() &&
        iface.isConvertibleIntrinsic(func->getIntrinsicID()))
      return {};
  }

  if (global->getName().empty())
    return {};
  return FlatSymbolRefAttr::get(context, global->getName());
}

/// Depth-first conversion of the metadata node `md` to the matching LLVM
/// dialect metadata attribute. Returns a null attribute for shapes that the
/// dialect's metadata-attribute hierarchy does not currently model. `path`
/// holds the metadata nodes on the current depth-first search path. Cyclic
/// metadata graphs are valid in LLVM IR, but they cannot be expressed by the
/// immutable, structurally-uniqued metadata attributes built here. The `path`
/// set lets the traversal recognize such a back-edge and bail out. `attrMap`
/// caches the attributes of fully converted nodes so that shared subgraphs
/// are visited only once.
Attribute ModuleImport::convertMetadataToAttrImpl(
    const llvm::Metadata *md, SmallPtrSetImpl<const llvm::Metadata *> &path,
    DenseMap<const llvm::Metadata *, Attribute> &attrMap) {
  if (!md)
    return {};
  if (auto *mdStr = dyn_cast<llvm::MDString>(md))
    return MDStringAttr::get(context,
                             StringAttr::get(context, mdStr->getString()));
  if (auto *cam = dyn_cast<llvm::ConstantAsMetadata>(md)) {
    llvm::Constant *constant = cam->getValue();
    if (auto *global = dyn_cast<llvm::GlobalValue>(constant)) {
      if (FlatSymbolRefAttr symbolRef = getMetadataGlobalValueSymbolRef(global))
        return MDGlobalValueAttr::get(context, symbolRef);
    }
    if (auto *ci = dyn_cast<llvm::ConstantInt>(constant)) {
      auto intType = IntegerType::get(context, ci->getBitWidth());
      return MDConstantAttr::get(context,
                                 IntegerAttr::get(intType, ci->getValue()));
    }
    if (auto *nullPtr = dyn_cast<llvm::ConstantPointerNull>(constant))
      return MDNullAttr::get(context,
                             nullPtr->getType()->getPointerAddressSpace());
    if (auto *constExpr = dyn_cast<llvm::ConstantExpr>(constant)) {
      // Only `addrspacecast` is modelled; other constant expressions have no
      // metadata-attribute counterpart.
      if (constExpr->getOpcode() != llvm::Instruction::AddrSpaceCast)
        return {};
      Attribute argAttr = convertMetadataToAttrImpl(
          llvm::ConstantAsMetadata::get(constExpr->getOperand(0)), path,
          attrMap);
      if (!argAttr)
        return {};
      return MDAddrSpaceCastAttr::get(
          context, argAttr, constExpr->getType()->getPointerAddressSpace());
    }
    return {};
  }
  if (auto *node = dyn_cast<llvm::MDNode>(md)) {
    // Metadata attributes cannot preserve distinctness, so bail out.
    if (node->isDistinct())
      return {};
    if (Attribute cached = attrMap.lookup(node))
      return cached;
    // If `node` is already on the current search path, this is a back-edge into
    // a cyclic graph. While that's valid it isn't implemented yet, so bail out.
    if (!path.insert(node).second)
      return {};
    SmallVector<Attribute> operands;
    operands.reserve(node->getNumOperands());
    for (const llvm::MDOperand &op : node->operands()) {
      Attribute opAttr = convertMetadataToAttrImpl(op.get(), path, attrMap);
      if (!opAttr)
        return {};
      operands.push_back(opAttr);
    }
    path.erase(node);
    Attribute nodeAttr = MDNodeAttr::get(context, operands);
    attrMap.try_emplace(node, nodeAttr);
    return nodeAttr;
  }
  return {};
}

/// Converts the metadata node `md` to the matching LLVM dialect metadata
/// attribute. Returns a null attribute for shapes that the dialect's
/// metadata-attribute hierarchy does not currently model, including distinct
/// nodes and cyclic metadata graphs that the immutable metadata attributes
/// cannot express.
Attribute ModuleImport::convertMetadataToAttr(const llvm::Metadata *md) {
  SmallPtrSet<const llvm::Metadata *, 8> path;
  DenseMap<const llvm::Metadata *, Attribute> attrMap;
  return convertMetadataToAttrImpl(md, path, attrMap);
}

LogicalResult ModuleImport::processTBAAMetadata(const llvm::MDNode *node) {
  Location loc = mlirModule.getLoc();

  // If `node` is a valid TBAA root node, then return its optional identity
  // string, otherwise return failure.
  auto getIdentityIfRootNode =
      [&](const llvm::MDNode *node) -> FailureOr<std::optional<StringRef>> {
    // Root node, e.g.:
    //   !0 = !{!"Simple C/C++ TBAA"}
    //   !1 = !{}
    if (node->getNumOperands() > 1)
      return failure();
    // If the operand is MDString, then assume that this is a root node.
    if (node->getNumOperands() == 1)
      if (const auto *op0 = dyn_cast<const llvm::MDString>(node->getOperand(0)))
        return std::optional<StringRef>{op0->getString()};
    return std::optional<StringRef>{};
  };

  // If `node` looks like a TBAA type descriptor metadata,
  // then return true, if it is a valid node, and false otherwise.
  // If it does not look like a TBAA type descriptor metadata, then
  // return std::nullopt.
  // If `identity` and `memberTypes/Offsets` are non-null, then they will
  // contain the converted metadata operands for a valid TBAA node (i.e. when
  // true is returned).
  auto isTypeDescriptorNode = [&](const llvm::MDNode *node,
                                  StringRef *identity = nullptr,
                                  SmallVectorImpl<TBAAMemberAttr> *members =
                                      nullptr) -> std::optional<bool> {
    unsigned numOperands = node->getNumOperands();
    // Type descriptor, e.g.:
    //   !1 = !{!"int", !0, /*optional*/i64 0} /* scalar int type */
    //   !2 = !{!"agg_t", !1, i64 0} /* struct agg_t { int x; } */
    if (numOperands < 2)
      return std::nullopt;

    // TODO: support "new" format (D41501) for type descriptors,
    //       where the first operand is an MDNode.
    const auto *identityNode =
        dyn_cast<const llvm::MDString>(node->getOperand(0));
    if (!identityNode)
      return std::nullopt;

    // This should be a type descriptor node.
    if (identity)
      *identity = identityNode->getString();

    for (unsigned pairNum = 0, e = numOperands / 2; pairNum < e; ++pairNum) {
      const auto *memberNode =
          dyn_cast<const llvm::MDNode>(node->getOperand(2 * pairNum + 1));
      if (!memberNode) {
        emitError(loc) << "operand '" << 2 * pairNum + 1 << "' must be MDNode: "
                       << diagMD(node, llvmModule.get());
        return false;
      }
      int64_t offset = 0;
      if (2 * pairNum + 2 >= numOperands) {
        // Allow for optional 0 offset in 2-operand nodes.
        if (numOperands != 2) {
          emitError(loc) << "missing member offset: "
                         << diagMD(node, llvmModule.get());
          return false;
        }
      } else {
        auto *offsetCI = llvm::mdconst::dyn_extract<llvm::ConstantInt>(
            node->getOperand(2 * pairNum + 2));
        if (!offsetCI) {
          emitError(loc) << "operand '" << 2 * pairNum + 2
                         << "' must be ConstantInt: "
                         << diagMD(node, llvmModule.get());
          return false;
        }
        offset = offsetCI->getZExtValue();
      }

      if (members)
        members->push_back(TBAAMemberAttr::get(
            cast<TBAANodeAttr>(tbaaMapping.lookup(memberNode)), offset));
    }

    return true;
  };

  // If `node` looks like a TBAA access tag metadata,
  // then return true, if it is a valid node, and false otherwise.
  // If it does not look like a TBAA access tag metadata, then
  // return std::nullopt.
  // If the other arguments are non-null, then they will contain
  // the converted metadata operands for a valid TBAA node (i.e. when true is
  // returned).
  auto isTagNode = [&](const llvm::MDNode *node,
                       TBAATypeDescriptorAttr *baseAttr = nullptr,
                       TBAATypeDescriptorAttr *accessAttr = nullptr,
                       int64_t *offset = nullptr,
                       bool *isConstant = nullptr) -> std::optional<bool> {
    // Access tag, e.g.:
    //   !3 = !{!1, !1, i64 0} /* scalar int access */
    //   !4 = !{!2, !1, i64 0} /* agg_t::x access */
    //
    // Optional 4th argument is ConstantInt 0/1 identifying whether
    // the location being accessed is "constant" (see for details:
    // https://llvm.org/docs/LangRef.html#representation).
    unsigned numOperands = node->getNumOperands();
    if (numOperands != 3 && numOperands != 4)
      return std::nullopt;
    const auto *baseMD = dyn_cast<const llvm::MDNode>(node->getOperand(0));
    const auto *accessMD = dyn_cast<const llvm::MDNode>(node->getOperand(1));
    auto *offsetCI =
        llvm::mdconst::dyn_extract<llvm::ConstantInt>(node->getOperand(2));
    if (!baseMD || !accessMD || !offsetCI)
      return std::nullopt;
    // TODO: support "new" TBAA format, if needed (see D41501).
    // In the "old" format the first operand of the access type
    // metadata is MDString. We have to distinguish the formats,
    // because access tags have the same structure, but different
    // meaning for the operands.
    if (accessMD->getNumOperands() < 1 ||
        !isa<llvm::MDString>(accessMD->getOperand(0)))
      return std::nullopt;
    bool isConst = false;
    if (numOperands == 4) {
      auto *isConstantCI =
          llvm::mdconst::dyn_extract<llvm::ConstantInt>(node->getOperand(3));
      if (!isConstantCI) {
        emitError(loc) << "operand '3' must be ConstantInt: "
                       << diagMD(node, llvmModule.get());
        return false;
      }
      isConst = isConstantCI->getValue()[0];
    }
    if (baseAttr)
      *baseAttr = cast<TBAATypeDescriptorAttr>(tbaaMapping.lookup(baseMD));
    if (accessAttr)
      *accessAttr = cast<TBAATypeDescriptorAttr>(tbaaMapping.lookup(accessMD));
    if (offset)
      *offset = offsetCI->getZExtValue();
    if (isConstant)
      *isConstant = isConst;
    return true;
  };

  // Do a post-order walk over the TBAA Graph. Since a correct TBAA Graph is a
  // DAG, a post-order walk guarantees that we convert any metadata node we
  // depend on, prior to converting the current node.
  DenseSet<const llvm::MDNode *> seen;
  SmallVector<const llvm::MDNode *> workList;
  workList.push_back(node);
  while (!workList.empty()) {
    const llvm::MDNode *current = workList.back();
    if (tbaaMapping.contains(current)) {
      // Already converted. Just pop from the worklist.
      workList.pop_back();
      continue;
    }

    // If any child of this node is not yet converted, don't pop the current
    // node from the worklist but push the not-yet-converted children in the
    // front of the worklist.
    bool anyChildNotConverted = false;
    for (const llvm::MDOperand &operand : current->operands())
      if (auto *childNode = dyn_cast_or_null<const llvm::MDNode>(operand.get()))
        if (!tbaaMapping.contains(childNode)) {
          workList.push_back(childNode);
          anyChildNotConverted = true;
        }

    if (anyChildNotConverted) {
      // If this is the second time we failed to convert an element in the
      // worklist it must be because a child is dependent on it being converted
      // and we have a cycle in the graph. Cycles are not allowed in TBAA
      // graphs.
      if (!seen.insert(current).second)
        return emitError(loc) << "has cycle in TBAA graph: "
                              << diagMD(current, llvmModule.get());

      continue;
    }

    // Otherwise simply import the current node.
    workList.pop_back();

    FailureOr<std::optional<StringRef>> rootNodeIdentity =
        getIdentityIfRootNode(current);
    if (succeeded(rootNodeIdentity)) {
      StringAttr stringAttr = *rootNodeIdentity
                                  ? builder.getStringAttr(**rootNodeIdentity)
                                  : nullptr;
      // The root nodes do not have operands, so we can create
      // the TBAARootAttr on the first walk.
      tbaaMapping.insert({current, builder.getAttr<TBAARootAttr>(stringAttr)});
      continue;
    }

    StringRef identity;
    SmallVector<TBAAMemberAttr> members;
    if (std::optional<bool> isValid =
            isTypeDescriptorNode(current, &identity, &members)) {
      assert(isValid.value() && "type descriptor node must be valid");

      tbaaMapping.insert({current, builder.getAttr<TBAATypeDescriptorAttr>(
                                       identity, members)});
      continue;
    }

    TBAATypeDescriptorAttr baseAttr, accessAttr;
    int64_t offset;
    bool isConstant;
    if (std::optional<bool> isValid =
            isTagNode(current, &baseAttr, &accessAttr, &offset, &isConstant)) {
      assert(isValid.value() && "access tag node must be valid");
      tbaaMapping.insert(
          {current, builder.getAttr<TBAATagAttr>(baseAttr, accessAttr, offset,
                                                 isConstant)});
      continue;
    }

    return emitError(loc) << "unsupported TBAA node format: "
                          << diagMD(current, llvmModule.get());
  }
  return success();
}

LogicalResult
ModuleImport::processAccessGroupMetadata(const llvm::MDNode *node) {
  Location loc = mlirModule.getLoc();
  if (failed(loopAnnotationImporter->translateAccessGroup(node, loc)))
    return emitError(loc) << "unsupported access group node: "
                          << diagMD(node, llvmModule.get());
  return success();
}

LogicalResult
ModuleImport::processAliasScopeMetadata(const llvm::MDNode *node) {
  Location loc = mlirModule.getLoc();
  // Helper that verifies the node has a self reference operand.
  auto verifySelfRef = [](const llvm::MDNode *node) {
    return node->getNumOperands() != 0 &&
           node == dyn_cast<llvm::MDNode>(node->getOperand(0));
  };
  auto verifySelfRefOrString = [](const llvm::MDNode *node) {
    return node->getNumOperands() != 0 &&
           (node == dyn_cast<llvm::MDNode>(node->getOperand(0)) ||
            isa<llvm::MDString>(node->getOperand(0)));
  };
  // Helper that verifies the given operand is a string or does not exist.
  auto verifyDescription = [](const llvm::MDNode *node, unsigned idx) {
    return idx >= node->getNumOperands() ||
           isa<llvm::MDString>(node->getOperand(idx));
  };

  auto getIdAttr = [&](const llvm::MDNode *node) -> Attribute {
    if (verifySelfRef(node))
      return DistinctAttr::create(builder.getUnitAttr());

    auto *name = cast<llvm::MDString>(node->getOperand(0));
    return builder.getStringAttr(name->getString());
  };

  // Helper that creates an alias scope domain attribute.
  auto createAliasScopeDomainOp = [&](const llvm::MDNode *aliasDomain) {
    StringAttr description = nullptr;
    if (aliasDomain->getNumOperands() >= 2)
      if (auto *operand = dyn_cast<llvm::MDString>(aliasDomain->getOperand(1)))
        description = builder.getStringAttr(operand->getString());
    Attribute idAttr = getIdAttr(aliasDomain);
    return builder.getAttr<AliasScopeDomainAttr>(idAttr, description);
  };

  // Collect the alias scopes and domains to translate them.
  for (const llvm::MDOperand &operand : node->operands()) {
    if (const auto *scope = dyn_cast<llvm::MDNode>(operand)) {
      llvm::AliasScopeNode aliasScope(scope);
      const llvm::MDNode *domain = aliasScope.getDomain();

      // Verify the scope node points to valid scope metadata which includes
      // verifying its domain. Perform the verification before looking it up in
      // the alias scope mapping since it could have been inserted as a domain
      // node before.
      if (!verifySelfRefOrString(scope) || !domain ||
          !verifyDescription(scope, 2))
        return emitError(loc) << "unsupported alias scope node: "
                              << diagMD(scope, llvmModule.get());
      if (!verifySelfRefOrString(domain) || !verifyDescription(domain, 1))
        return emitError(loc) << "unsupported alias domain node: "
                              << diagMD(domain, llvmModule.get());

      if (aliasScopeMapping.contains(scope))
        continue;

      // Convert the domain metadata node if it has not been translated before.
      auto it = aliasScopeMapping.find(aliasScope.getDomain());
      if (it == aliasScopeMapping.end()) {
        auto aliasScopeDomainOp = createAliasScopeDomainOp(domain);
        it = aliasScopeMapping.try_emplace(domain, aliasScopeDomainOp).first;
      }

      // Convert the scope metadata node if it has not been converted before.
      StringAttr description = nullptr;
      if (!aliasScope.getName().empty())
        description = builder.getStringAttr(aliasScope.getName());
      Attribute idAttr = getIdAttr(scope);
      auto aliasScopeOp = builder.getAttr<AliasScopeAttr>(
          idAttr, cast<AliasScopeDomainAttr>(it->second), description);

      aliasScopeMapping.try_emplace(aliasScope.getNode(), aliasScopeOp);
    }
  }
  return success();
}

FailureOr<SmallVector<AliasScopeAttr>>
ModuleImport::lookupAliasScopeAttrs(const llvm::MDNode *node) const {
  SmallVector<AliasScopeAttr> aliasScopes;
  aliasScopes.reserve(node->getNumOperands());
  for (const llvm::MDOperand &operand : node->operands()) {
    auto *node = cast<llvm::MDNode>(operand.get());
    aliasScopes.push_back(
        dyn_cast_or_null<AliasScopeAttr>(aliasScopeMapping.lookup(node)));
  }
  // Return failure if one of the alias scope lookups failed.
  if (llvm::is_contained(aliasScopes, nullptr))
    return failure();
  return aliasScopes;
}

void ModuleImport::addDebugIntrinsic(llvm::CallInst *intrinsic) {
  debugIntrinsics.insert(intrinsic);
}

void ModuleImport::addDebugRecord(llvm::DbgVariableRecord *dbgRecord) {
  if (!dbgRecords.contains(dbgRecord))
    dbgRecords.insert(dbgRecord);
}

static Attribute convertCGProfileModuleFlagValue(ModuleOp mlirModule,
                                                 llvm::MDTuple *mdTuple) {
  auto getLLVMFunction =
      [&](const llvm::MDOperand &funcMDO) -> llvm::Function * {
    auto *f = cast_or_null<llvm::ValueAsMetadata>(funcMDO);
    // nullptr is a valid value for the function pointer.
    if (!f)
      return nullptr;
    auto *llvmFn = cast<llvm::Function>(f->getValue()->stripPointerCasts());
    return llvmFn;
  };

  // Each tuple element becomes one ModuleFlagCGProfileEntryAttr.
  SmallVector<Attribute> cgProfile;
  for (unsigned i = 0; i < mdTuple->getNumOperands(); i++) {
    const llvm::MDOperand &mdo = mdTuple->getOperand(i);
    auto *cgEntry = cast<llvm::MDNode>(mdo);
    llvm::Constant *llvmConstant =
        cast<llvm::ConstantAsMetadata>(cgEntry->getOperand(2))->getValue();
    uint64_t count = cast<llvm::ConstantInt>(llvmConstant)->getZExtValue();
    auto *fromFn = getLLVMFunction(cgEntry->getOperand(0));
    auto *toFn = getLLVMFunction(cgEntry->getOperand(1));
    // FlatSymbolRefAttr::get(mlirModule->getContext(), llvmFn->getName());
    cgProfile.push_back(ModuleFlagCGProfileEntryAttr::get(
        mlirModule->getContext(),
        fromFn ? FlatSymbolRefAttr::get(mlirModule->getContext(),
                                        fromFn->getName())
               : nullptr,
        toFn ? FlatSymbolRefAttr::get(mlirModule->getContext(), toFn->getName())
             : nullptr,
        count));
  }
  return ArrayAttr::get(mlirModule->getContext(), cgProfile);
}

/// Extract a two element `MDTuple` from a `MDOperand`. Emit a warning in case
/// something else is found.
static llvm::MDTuple *getTwoElementMDTuple(ModuleOp mlirModule,
                                           const llvm::Module *llvmModule,
                                           const llvm::MDOperand &md) {
  auto *tupleEntry = dyn_cast_or_null<llvm::MDTuple>(md);
  if (!tupleEntry || tupleEntry->getNumOperands() != 2)
    emitWarning(mlirModule.getLoc())
        << "expected 2-element tuple metadata: " << diagMD(md, llvmModule);
  return tupleEntry;
}

/// Extract a constant metadata value from a two element tuple (<key, value>).
/// Return nullptr if requirements are not met. A warning is emitted if the
/// `matchKey` is different from the tuple's key.
static llvm::ConstantAsMetadata *getConstantMDFromKeyValueTuple(
    ModuleOp mlirModule, const llvm::Module *llvmModule,
    const llvm::MDOperand &md, StringRef matchKey, bool optional = false) {
  llvm::MDTuple *tupleEntry = getTwoElementMDTuple(mlirModule, llvmModule, md);
  if (!tupleEntry)
    return nullptr;
  auto *keyMD = dyn_cast<llvm::MDString>(tupleEntry->getOperand(0));
  if (!keyMD || keyMD->getString() != matchKey) {
    if (!optional)
      emitWarning(mlirModule.getLoc())
          << "expected '" << matchKey << "' key, but found: "
          << diagMD(tupleEntry->getOperand(0), llvmModule);
    return nullptr;
  }

  return dyn_cast<llvm::ConstantAsMetadata>(tupleEntry->getOperand(1));
}

/// Extract an integer value from a two element tuple (<key, value>).
/// Fail if requirements are not met. A warning is emitted if the
/// found value isn't a LLVM constant integer.
static FailureOr<uint64_t>
convertInt64FromKeyValueTuple(ModuleOp mlirModule,
                              const llvm::Module *llvmModule,
                              const llvm::MDOperand &md, StringRef matchKey) {
  llvm::ConstantAsMetadata *valMD =
      getConstantMDFromKeyValueTuple(mlirModule, llvmModule, md, matchKey);
  if (!valMD)
    return failure();

  if (auto *cstInt = dyn_cast<llvm::ConstantInt>(valMD->getValue()))
    return cstInt->getZExtValue();

  emitWarning(mlirModule.getLoc())
      << "expected integer metadata value for key '" << matchKey
      << "': " << diagMD(md, llvmModule);
  return failure();
}

static std::optional<ProfileSummaryFormatKind>
convertProfileSummaryFormat(ModuleOp mlirModule, const llvm::Module *llvmModule,
                            const llvm::MDOperand &formatMD) {
  auto *tupleEntry = getTwoElementMDTuple(mlirModule, llvmModule, formatMD);
  if (!tupleEntry)
    return std::nullopt;

  llvm::MDString *keyMD = dyn_cast<llvm::MDString>(tupleEntry->getOperand(0));
  if (!keyMD || keyMD->getString() != "ProfileFormat") {
    emitWarning(mlirModule.getLoc())
        << "expected 'ProfileFormat' key: "
        << diagMD(tupleEntry->getOperand(0), llvmModule);
    return std::nullopt;
  }

  llvm::MDString *valMD = dyn_cast<llvm::MDString>(tupleEntry->getOperand(1));
  std::optional<ProfileSummaryFormatKind> fmtKind =
      symbolizeProfileSummaryFormatKind(valMD->getString());
  if (!fmtKind) {
    emitWarning(mlirModule.getLoc())
        << "expected 'SampleProfile', 'InstrProf' or 'CSInstrProf' values, "
           "but found: "
        << diagMD(valMD, llvmModule);
    return std::nullopt;
  }

  return fmtKind;
}

static FailureOr<SmallVector<ModuleFlagProfileSummaryDetailedAttr>>
convertProfileSummaryDetailed(ModuleOp mlirModule,
                              const llvm::Module *llvmModule,
                              const llvm::MDOperand &summaryMD) {
  auto *tupleEntry = getTwoElementMDTuple(mlirModule, llvmModule, summaryMD);
  if (!tupleEntry)
    return failure();

  llvm::MDString *keyMD = dyn_cast<llvm::MDString>(tupleEntry->getOperand(0));
  if (!keyMD || keyMD->getString() != "DetailedSummary") {
    emitWarning(mlirModule.getLoc())
        << "expected 'DetailedSummary' key: "
        << diagMD(tupleEntry->getOperand(0), llvmModule);
    return failure();
  }

  llvm::MDTuple *entriesMD = dyn_cast<llvm::MDTuple>(tupleEntry->getOperand(1));
  if (!entriesMD) {
    emitWarning(mlirModule.getLoc())
        << "expected tuple value for 'DetailedSummary' key: "
        << diagMD(tupleEntry->getOperand(1), llvmModule);
    return failure();
  }

  SmallVector<ModuleFlagProfileSummaryDetailedAttr> detailedSummary;
  for (auto &&entry : entriesMD->operands()) {
    llvm::MDTuple *entryMD = dyn_cast<llvm::MDTuple>(entry);
    if (!entryMD || entryMD->getNumOperands() != 3) {
      emitWarning(mlirModule.getLoc())
          << "'DetailedSummary' entry expects 3 operands: "
          << diagMD(entry, llvmModule);
      return failure();
    }

    auto *op0 = dyn_cast<llvm::ConstantAsMetadata>(entryMD->getOperand(0));
    auto *op1 = dyn_cast<llvm::ConstantAsMetadata>(entryMD->getOperand(1));
    auto *op2 = dyn_cast<llvm::ConstantAsMetadata>(entryMD->getOperand(2));
    if (!op0 || !op1 || !op2) {
      emitWarning(mlirModule.getLoc())
          << "expected only integer entries in 'DetailedSummary': "
          << diagMD(entry, llvmModule);
      return failure();
    }

    auto detaildSummaryEntry = ModuleFlagProfileSummaryDetailedAttr::get(
        mlirModule->getContext(),
        cast<llvm::ConstantInt>(op0->getValue())->getZExtValue(),
        cast<llvm::ConstantInt>(op1->getValue())->getZExtValue(),
        cast<llvm::ConstantInt>(op2->getValue())->getZExtValue());
    detailedSummary.push_back(detaildSummaryEntry);
  }
  return detailedSummary;
}

static Attribute
convertProfileSummaryModuleFlagValue(ModuleOp mlirModule,
                                     const llvm::Module *llvmModule,
                                     llvm::MDTuple *mdTuple) {
  unsigned profileNumEntries = mdTuple->getNumOperands();
  if (profileNumEntries < 8) {
    emitWarning(mlirModule.getLoc())
        << "expected at 8 entries in 'ProfileSummary': "
        << diagMD(mdTuple, llvmModule);
    return nullptr;
  }

  unsigned summayIdx = 0;
  auto checkOptionalPosition = [&](const llvm::MDOperand &md,
                                   StringRef matchKey) -> LogicalResult {
    // Make sure we won't step over the bound of the array of summary entries.
    // Since (non-optional) DetailedSummary always comes last, the next entry in
    // the tuple operand array must exist.
    if (summayIdx + 1 >= profileNumEntries) {
      emitWarning(mlirModule.getLoc())
          << "the last summary entry is '" << matchKey
          << "', expected 'DetailedSummary': " << diagMD(md, llvmModule);
      return failure();
    }

    return success();
  };

  auto getOptIntValue =
      [&](const llvm::MDOperand &md,
          StringRef matchKey) -> FailureOr<std::optional<uint64_t>> {
    if (!getConstantMDFromKeyValueTuple(mlirModule, llvmModule, md, matchKey,
                                        /*optional=*/true))
      return FailureOr<std::optional<uint64_t>>(std::nullopt);
    if (checkOptionalPosition(md, matchKey).failed())
      return failure();
    FailureOr<uint64_t> val =
        convertInt64FromKeyValueTuple(mlirModule, llvmModule, md, matchKey);
    if (failed(val))
      return failure();
    return val;
  };

  auto getOptDoubleValue = [&](const llvm::MDOperand &md,
                               StringRef matchKey) -> FailureOr<FloatAttr> {
    auto *valMD = getConstantMDFromKeyValueTuple(mlirModule, llvmModule, md,
                                                 matchKey, /*optional=*/true);
    if (!valMD)
      return FloatAttr{};
    if (auto *cstFP = dyn_cast<llvm::ConstantFP>(valMD->getValue())) {
      if (checkOptionalPosition(md, matchKey).failed())
        return failure();
      return FloatAttr::get(Float64Type::get(mlirModule.getContext()),
                            cstFP->getValueAPF());
    }
    emitWarning(mlirModule.getLoc())
        << "expected double metadata value for key '" << matchKey
        << "': " << diagMD(md, llvmModule);
    return failure();
  };

  // Build ModuleFlagProfileSummaryAttr by sequentially fetching elements in
  // a fixed order: format, total count, etc.
  std::optional<ProfileSummaryFormatKind> format = convertProfileSummaryFormat(
      mlirModule, llvmModule, mdTuple->getOperand(summayIdx++));
  if (!format.has_value())
    return nullptr;

  FailureOr<uint64_t> totalCount = convertInt64FromKeyValueTuple(
      mlirModule, llvmModule, mdTuple->getOperand(summayIdx++), "TotalCount");
  if (failed(totalCount))
    return nullptr;

  FailureOr<uint64_t> maxCount = convertInt64FromKeyValueTuple(
      mlirModule, llvmModule, mdTuple->getOperand(summayIdx++), "MaxCount");
  if (failed(maxCount))
    return nullptr;

  FailureOr<uint64_t> maxInternalCount = convertInt64FromKeyValueTuple(
      mlirModule, llvmModule, mdTuple->getOperand(summayIdx++),
      "MaxInternalCount");
  if (failed(maxInternalCount))
    return nullptr;

  FailureOr<uint64_t> maxFunctionCount = convertInt64FromKeyValueTuple(
      mlirModule, llvmModule, mdTuple->getOperand(summayIdx++),
      "MaxFunctionCount");
  if (failed(maxFunctionCount))
    return nullptr;

  FailureOr<uint64_t> numCounts = convertInt64FromKeyValueTuple(
      mlirModule, llvmModule, mdTuple->getOperand(summayIdx++), "NumCounts");
  if (failed(numCounts))
    return nullptr;

  FailureOr<uint64_t> numFunctions = convertInt64FromKeyValueTuple(
      mlirModule, llvmModule, mdTuple->getOperand(summayIdx++), "NumFunctions");
  if (failed(numFunctions))
    return nullptr;

  // Handle optional keys.
  FailureOr<std::optional<uint64_t>> isPartialProfile =
      getOptIntValue(mdTuple->getOperand(summayIdx), "IsPartialProfile");
  if (failed(isPartialProfile))
    return nullptr;
  if (isPartialProfile->has_value())
    summayIdx++;

  FailureOr<FloatAttr> partialProfileRatio =
      getOptDoubleValue(mdTuple->getOperand(summayIdx), "PartialProfileRatio");
  if (failed(partialProfileRatio))
    return nullptr;
  if (*partialProfileRatio)
    summayIdx++;

  // Handle detailed summary.
  FailureOr<SmallVector<ModuleFlagProfileSummaryDetailedAttr>> detailed =
      convertProfileSummaryDetailed(mlirModule, llvmModule,
                                    mdTuple->getOperand(summayIdx));
  if (failed(detailed))
    return nullptr;

  // Build the final profile summary attribute.
  return ModuleFlagProfileSummaryAttr::get(
      mlirModule->getContext(), *format, *totalCount, *maxCount,
      *maxInternalCount, *maxFunctionCount, *numCounts, *numFunctions,
      *isPartialProfile, *partialProfileRatio, *detailed);
}

/// Invoke specific handlers for each known module flag value, returns nullptr
/// if the key is unknown or unimplemented.
static Attribute
convertModuleFlagValueFromMDTuple(ModuleOp mlirModule,
                                  const llvm::Module *llvmModule, StringRef key,
                                  llvm::MDTuple *mdTuple) {
  if (key == LLVMDialect::getModuleFlagKeyCGProfileName())
    return convertCGProfileModuleFlagValue(mlirModule, mdTuple);
  if (key == LLVMDialect::getModuleFlagKeyProfileSummaryName())
    return convertProfileSummaryModuleFlagValue(mlirModule, llvmModule,
                                                mdTuple);
  // Handle MDTuples whose operands are all MDStrings (e.g. "riscv-isa").
  // Convert them to ArrayAttr of StringAttrs for a lossless round-trip.
  Builder builder(mlirModule->getContext());
  SmallVector<Attribute> strings;
  strings.reserve(mdTuple->getNumOperands());
  for (const llvm::MDOperand &operand : mdTuple->operands()) {
    auto *mdString = dyn_cast_if_present<llvm::MDString>(operand.get());
    if (!mdString)
      return nullptr;
    strings.push_back(builder.getStringAttr(mdString->getString()));
  }
  return builder.getArrayAttr(strings);
}

LogicalResult ModuleImport::convertModuleFlagsMetadata() {
  SmallVector<llvm::Module::ModuleFlagEntry> llvmModuleFlags;
  llvmModule->getModuleFlagsMetadata(llvmModuleFlags);

  SmallVector<Attribute> moduleFlags;
  for (const auto [behavior, key, val] : llvmModuleFlags) {
    Attribute valAttr = nullptr;
    if (auto *constInt = llvm::mdconst::dyn_extract<llvm::ConstantInt>(val)) {
      valAttr = builder.getI32IntegerAttr(constInt->getZExtValue());
    } else if (auto *mdString = dyn_cast<llvm::MDString>(val)) {
      valAttr = builder.getStringAttr(mdString->getString());
    } else if (auto *mdTuple = dyn_cast<llvm::MDTuple>(val)) {
      valAttr = convertModuleFlagValueFromMDTuple(mlirModule, llvmModule.get(),
                                                  key->getString(), mdTuple);
    }

    if (!valAttr) {
      emitWarning(mlirModule.getLoc())
          << "unsupported module flag value for key '" << key->getString()
          << "' : " << diagMD(val, llvmModule.get());
      continue;
    }

    moduleFlags.push_back(builder.getAttr<ModuleFlagAttr>(
        convertModFlagBehaviorFromLLVM(behavior),
        builder.getStringAttr(key->getString()), valAttr));
  }

  if (!moduleFlags.empty())
    LLVM::ModuleFlagsOp::create(builder, mlirModule.getLoc(),
                                builder.getArrayAttr(moduleFlags));

  return success();
}

LogicalResult ModuleImport::convertLinkerOptionsMetadata() {
  for (const llvm::NamedMDNode &named : llvmModule->named_metadata()) {
    if (named.getName() != "llvm.linker.options")
      continue;
    // llvm.linker.options operands are lists of strings.
    for (const llvm::MDNode *node : named.operands()) {
      SmallVector<StringRef> options;
      options.reserve(node->getNumOperands());
      for (const llvm::MDOperand &option : node->operands())
        options.push_back(cast<llvm::MDString>(option)->getString());
      LLVM::LinkerOptionsOp::create(builder, mlirModule.getLoc(),
                                    builder.getStrArrayAttr(options));
    }
  }
  return success();
}

LogicalResult ModuleImport::convertDependentLibrariesMetadata() {
  for (const llvm::NamedMDNode &named : llvmModule->named_metadata()) {
    if (named.getName() != "llvm.dependent-libraries")
      continue;
    SmallVector<StringRef> libraries;
    for (const llvm::MDNode *node : named.operands()) {
      if (node->getNumOperands() == 1)
        if (auto *mdString = dyn_cast<llvm::MDString>(node->getOperand(0)))
          libraries.push_back(mdString->getString());
    }
    if (!libraries.empty())
      mlirModule->setAttr(LLVM::LLVMDialect::getDependentLibrariesAttrName(),
                          builder.getStrArrayAttr(libraries));
  }
  return success();
}

LogicalResult ModuleImport::convertIdentMetadata() {
  for (const llvm::NamedMDNode &named : llvmModule->named_metadata()) {
    // llvm.ident should have a single operand. That operand is itself an
    // MDNode with a single string operand.
    if (named.getName() != LLVMDialect::getIdentAttrName())
      continue;

    if (named.getNumOperands() == 1)
      if (auto *md = dyn_cast<llvm::MDNode>(named.getOperand(0)))
        if (md->getNumOperands() == 1)
          if (auto *mdStr = dyn_cast<llvm::MDString>(md->getOperand(0)))
            mlirModule->setAttr(LLVMDialect::getIdentAttrName(),
                                builder.getStringAttr(mdStr->getString()));
  }
  return success();
}

LogicalResult ModuleImport::convertCommandlineMetadata() {
  for (const llvm::NamedMDNode &nmd : llvmModule->named_metadata()) {
    // llvm.commandline should have a single operand. That operand is itself an
    // MDNode with a single string operand.
    if (nmd.getName() != LLVMDialect::getCommandlineAttrName())
      continue;

    if (nmd.getNumOperands() == 1)
      if (auto *md = dyn_cast<llvm::MDNode>(nmd.getOperand(0)))
        if (md->getNumOperands() == 1)
          if (auto *mdStr = dyn_cast<llvm::MDString>(md->getOperand(0)))
            mlirModule->setAttr(LLVMDialect::getCommandlineAttrName(),
                                builder.getStringAttr(mdStr->getString()));
  }
  return success();
}

LogicalResult ModuleImport::convertMetadata() {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(mlirModule.getBody());
  for (const llvm::Function &func : llvmModule->functions()) {
    for (const llvm::Instruction &inst : llvm::instructions(func)) {
      // Convert access group metadata nodes.
      if (llvm::MDNode *node =
              inst.getMetadata(llvm::LLVMContext::MD_access_group))
        if (failed(processAccessGroupMetadata(node)))
          return failure();

      // Convert alias analysis metadata nodes.
      llvm::AAMDNodes aliasAnalysisNodes = inst.getAAMetadata();
      if (!aliasAnalysisNodes)
        continue;
      if (aliasAnalysisNodes.TBAA)
        if (failed(processTBAAMetadata(aliasAnalysisNodes.TBAA)))
          return failure();
      if (aliasAnalysisNodes.Scope)
        if (failed(processAliasScopeMetadata(aliasAnalysisNodes.Scope)))
          return failure();
      if (aliasAnalysisNodes.NoAlias)
        if (failed(processAliasScopeMetadata(aliasAnalysisNodes.NoAlias)))
          return failure();
    }
  }
  if (failed(convertLinkerOptionsMetadata()))
    return failure();
  if (failed(convertDependentLibrariesMetadata()))
    return failure();
  if (failed(convertModuleFlagsMetadata()))
    return failure();
  if (failed(convertIdentMetadata()))
    return failure();
  if (failed(convertCommandlineMetadata()))
    return failure();
  return success();
}

void ModuleImport::setNonDebugMetadataAttrs(llvm::Instruction *inst,
                                            Operation *op) {
  SmallVector<std::pair<unsigned, llvm::MDNode *>> allMetadata;
  inst->getAllMetadataOtherThanDebugLoc(allMetadata);
  for (auto &[kind, node] : allMetadata) {
    if (!iface.isConvertibleMetadata(kind))
      continue;
    if (failed(iface.setMetadataAttrs(builder, kind, node, op, *this))) {
      if (emitExpensiveWarnings) {
        Location loc = debugImporter->translateLoc(inst->getDebugLoc());
        emitWarning(loc) << "unhandled metadata: "
                         << diagMD(node, llvmModule.get()) << " on "
                         << diag(*inst);
      }
    }
  }
}

/// Converts LLVM string, integer, and enum attributes into MLIR attributes,
/// skipping those in `attributesToSkip` and emitting a warning at `loc` for
/// any other unsupported attributes.
ArrayAttr mlir::LLVM::detail::convertLLVMAttributesToMLIR(
    Location loc, MLIRContext *context, llvm::AttributeSet attributes,
    ArrayRef<StringLiteral> attributesToSkip,
    ArrayRef<StringLiteral> attributePrefixesToSkip) {
  SmallVector<Attribute> mlirAttributes;
  for (llvm::Attribute attr : attributes) {
    StringRef attrName;
    if (attr.isStringAttribute())
      attrName = attr.getKindAsString();
    else
      attrName = llvm::Attribute::getNameFromAttrKind(attr.getKindAsEnum());
    if (llvm::is_contained(attributesToSkip, attrName))
      continue;

    auto attrNameStartsWith = [attrName](StringLiteral sl) {
      return attrName.starts_with(sl);
    };
    if (attributePrefixesToSkip.end() !=
        llvm::find_if(attributePrefixesToSkip, attrNameStartsWith))
      continue;

    auto keyAttr = StringAttr::get(context, attrName);
    if (attr.isStringAttribute()) {
      StringRef val = attr.getValueAsString();
      if (val.empty()) {
        // For string attributes without values, add only the attribute name.
        mlirAttributes.push_back(keyAttr);
        continue;
      }
      // For string attributes with a value, create a [name, value] pair.
      mlirAttributes.push_back(
          ArrayAttr::get(context, {keyAttr, StringAttr::get(context, val)}));
      continue;
    }
    if (attr.isIntAttribute()) {
      // For integer attributes, convert the value to a string and create a
      // [name, value] pair.
      auto val = std::to_string(attr.getValueAsInt());
      mlirAttributes.push_back(
          ArrayAttr::get(context, {keyAttr, StringAttr::get(context, val)}));
      continue;
    }
    if (attr.isEnumAttribute()) {
      // For enum attributes, add only the attribute name.
      mlirAttributes.push_back(keyAttr);
      continue;
    }

    emitWarning(loc)
        << "'" << attrName
        << "' attribute is invalid on current operation, skipping it";
  }
  return ArrayAttr::get(context, mlirAttributes);
}

FailureOr<SmallVector<AliasScopeAttr>>
ModuleImport::matchAliasScopeAttrs(llvm::Value *value) {
  auto *nodeAsVal = cast<llvm::MetadataAsValue>(value);
  auto *node = cast<llvm::MDNode>(nodeAsVal->getMetadata());
  return lookupAliasScopeAttrs(node);
}

FlatSymbolRefAttr ModuleImport::getPersonalityAsAttr(llvm::Function *f) {
  if (!f->hasPersonalityFn())
    return nullptr;

  llvm::Constant *pf = f->getPersonalityFn();

  // If it directly has a name, we can use it.
  if (pf->hasName())
    return SymbolRefAttr::get(builder.getContext(), pf->getName());

  // If it doesn't have a name, currently, only function pointers that are
  // bitcast to i8* are parsed.
  if (auto *ce = dyn_cast<llvm::ConstantExpr>(pf)) {
    if (ce->getOpcode() == llvm::Instruction::BitCast &&
        ce->getType() == llvm::PointerType::getUnqual(f->getContext())) {
      if (auto *func = dyn_cast<llvm::Function>(ce->getOperand(0)))
        return SymbolRefAttr::get(builder.getContext(), func->getName());
    }
  }
  return FlatSymbolRefAttr();
}

static void processMemoryEffects(llvm::Function *func, LLVMFuncOp funcOp) {
  llvm::MemoryEffects memEffects = func->getMemoryEffects();

  auto othermem = convertModRefInfoFromLLVM(
      memEffects.getModRef(llvm::MemoryEffects::Location::Other));
  auto argMem = convertModRefInfoFromLLVM(
      memEffects.getModRef(llvm::MemoryEffects::Location::ArgMem));
  auto inaccessibleMem = convertModRefInfoFromLLVM(
      memEffects.getModRef(llvm::MemoryEffects::Location::InaccessibleMem));
  auto errnoMem = convertModRefInfoFromLLVM(
      memEffects.getModRef(llvm::MemoryEffects::Location::ErrnoMem));
  auto targetMem0 = convertModRefInfoFromLLVM(
      memEffects.getModRef(llvm::MemoryEffects::Location::TargetMem0));
  auto targetMem1 = convertModRefInfoFromLLVM(
      memEffects.getModRef(llvm::MemoryEffects::Location::TargetMem1));
  auto memAttr =
      MemoryEffectsAttr::get(funcOp.getContext(), othermem, argMem,
                             inaccessibleMem, errnoMem, targetMem0, targetMem1);
  // Only set the attr when it does not match the default value.
  if (memAttr.isReadWrite())
    return;
  funcOp.setMemoryEffectsAttr(memAttr);
}

static void processDenormalFPEnv(llvm::Function *func, LLVMFuncOp funcOp) {
  llvm::DenormalFPEnv denormalFpEnv = func->getDenormalFPEnv();
  // Only set the attr when it does not match the default value.
  if (denormalFpEnv == llvm::DenormalFPEnv::getDefault())
    return;

  llvm::DenormalMode defaultMode = denormalFpEnv.DefaultMode;
  llvm::DenormalMode floatMode = denormalFpEnv.F32Mode;

  auto denormalFpEnvAttr = DenormalFPEnvAttr::get(
      funcOp.getContext(), convertDenormalModeKindFromLLVM(defaultMode.Output),
      convertDenormalModeKindFromLLVM(defaultMode.Input),
      convertDenormalModeKindFromLLVM(floatMode.Output),
      convertDenormalModeKindFromLLVM(floatMode.Input));
  funcOp.setDenormalFpenvAttr(denormalFpEnvAttr);
}

// List of LLVM IR attributes that map to an explicit attribute on the MLIR
// LLVMFuncOp.
static constexpr std::array kExplicitLLVMFuncOpAttributes{
    StringLiteral("aarch64_in_za"),
    StringLiteral("aarch64_inout_za"),
    StringLiteral("aarch64_new_za"),
    StringLiteral("aarch64_out_za"),
    StringLiteral("aarch64_preserves_za"),
    StringLiteral("aarch64_pstate_sm_body"),
    StringLiteral("aarch64_pstate_sm_compatible"),
    StringLiteral("aarch64_pstate_sm_enabled"),
    StringLiteral("allocsize"),
    StringLiteral("alwaysinline"),
    StringLiteral("cold"),
    StringLiteral("convergent"),
    StringLiteral("fp-contract"),
    StringLiteral("frame-pointer"),
    StringLiteral("hot"),
    StringLiteral("inlinehint"),
    StringLiteral("instrument-function-entry"),
    StringLiteral("instrument-function-exit"),
    StringLiteral("modular-format"),
    StringLiteral("memory"),
    StringLiteral("minsize"),
    StringLiteral("no_caller_saved_registers"),
    StringLiteral("no-signed-zeros-fp-math"),
    StringLiteral("no-builtins"),
    StringLiteral("nocallback"),
    StringLiteral("noduplicate"),
    StringLiteral("noinline"),
    StringLiteral("noreturn"),
    StringLiteral("nounwind"),
    StringLiteral("optnone"),
    StringLiteral("optsize"),
    StringLiteral("returns_twice"),
    StringLiteral("save-reg-params"),
    StringLiteral("target-features"),
    StringLiteral("trap-func-name"),
    StringLiteral("tune-cpu"),
    StringLiteral("uwtable"),
    StringLiteral("vscale_range"),
    StringLiteral("willreturn"),
    StringLiteral("zero-call-used-regs"),
    StringLiteral("denormal_fpenv"),
};

// List of LLVM IR attributes that are handled by prefix to map onto an MLIR
// LLVMFuncOp.
static constexpr std::array kExplicitLLVMFuncOpAttributePrefixes{
    StringLiteral("no-builtin-"),
};

template <typename OpTy>
static void convertNoBuiltinAttrs(MLIRContext *ctx,
                                  const llvm::AttributeSet &attrs,
                                  OpTy target) {
  // 'no-builtins' is the complete collection, and overrides all the rest.
  if (attrs.hasAttribute("no-builtins")) {
    target.setNobuiltinsAttr(ArrayAttr::get(ctx, {}));
    return;
  }

  llvm::SetVector<Attribute> nbAttrs;
  for (llvm::Attribute attr : attrs) {
    // Attributes that are part of llvm directly (that is, have an AttributeKind
    // in the enum) shouldn't be checked.
    if (attr.hasKindAsEnum())
      continue;

    StringRef val = attr.getKindAsString();

    if (val.starts_with("no-builtin-"))
      nbAttrs.insert(
          StringAttr::get(ctx, val.drop_front(sizeof("no-builtin-") - 1)));
  }

  if (!nbAttrs.empty())
    target.setNobuiltinsAttr(ArrayAttr::get(ctx, nbAttrs.getArrayRef()));
}

template <typename OpTy>
static void convertAllocsizeAttr(MLIRContext *ctx,
                                 const llvm::AttributeSet &attrs, OpTy target) {
  llvm::Attribute attr = attrs.getAttribute(llvm::Attribute::AllocSize);
  if (!attr.isValid())
    return;

  auto [elemSize, numElems] = attr.getAllocSizeArgs();
  if (numElems) {
    target.setAllocsizeAttr(
        DenseI32ArrayAttr::get(ctx, {static_cast<int32_t>(elemSize),
                                     static_cast<int32_t>(*numElems)}));
  } else {
    target.setAllocsizeAttr(
        DenseI32ArrayAttr::get(ctx, {static_cast<int32_t>(elemSize)}));
  }
}

/// Converts LLVM attributes from `func` into MLIR attributes and adds them
/// to `funcOp` as passthrough attributes, skipping those listed in
/// `kExplicitLLVMFuncAttributes`.
static void processPassthroughAttrs(llvm::Function *func, LLVMFuncOp funcOp) {
  llvm::AttributeSet funcAttrs = func->getAttributes().getAttributes(
      llvm::AttributeList::AttrIndex::FunctionIndex);
  ArrayAttr passthroughAttr = convertLLVMAttributesToMLIR(
      funcOp.getLoc(), funcOp.getContext(), funcAttrs,
      kExplicitLLVMFuncOpAttributes, kExplicitLLVMFuncOpAttributePrefixes);
  if (!passthroughAttr.empty())
    funcOp.setPassthroughAttr(passthroughAttr);
}

void ModuleImport::processFunctionAttributes(llvm::Function *func,
                                             LLVMFuncOp funcOp) {
  processMemoryEffects(func, funcOp);
  processDenormalFPEnv(func, funcOp);
  processPassthroughAttrs(func, funcOp);

  if (func->hasFnAttribute(llvm::Attribute::NoInline))
    funcOp.setNoInline(true);
  if (func->hasFnAttribute(llvm::Attribute::AlwaysInline))
    funcOp.setAlwaysInline(true);
  if (func->hasFnAttribute(llvm::Attribute::InlineHint))
    funcOp.setInlineHint(true);
  if (func->hasFnAttribute(llvm::Attribute::OptimizeNone))
    funcOp.setOptimizeNone(true);
  if (func->hasFnAttribute(llvm::Attribute::Convergent))
    funcOp.setConvergent(true);
  if (func->hasFnAttribute(llvm::Attribute::NoUnwind))
    funcOp.setNoUnwind(true);
  if (func->hasFnAttribute(llvm::Attribute::WillReturn))
    funcOp.setWillReturn(true);
  if (func->hasFnAttribute(llvm::Attribute::NoReturn))
    funcOp.setNoreturn(true);
  if (func->hasFnAttribute(llvm::Attribute::OptimizeForSize))
    funcOp.setOptsize(true);
  if (func->hasFnAttribute("save-reg-params"))
    funcOp.setSaveRegParams(true);
  if (func->hasFnAttribute(llvm::Attribute::MinSize))
    funcOp.setMinsize(true);
  if (func->hasFnAttribute(llvm::Attribute::ReturnsTwice))
    funcOp.setReturnsTwice(true);
  if (func->hasFnAttribute(llvm::Attribute::Cold))
    funcOp.setCold(true);
  if (func->hasFnAttribute(llvm::Attribute::Hot))
    funcOp.setHot(true);
  if (func->hasFnAttribute(llvm::Attribute::NoDuplicate))
    funcOp.setNoduplicate(true);
  if (func->hasFnAttribute("no_caller_saved_registers"))
    funcOp.setNoCallerSavedRegisters(true);
  if (func->hasFnAttribute(llvm::Attribute::NoCallback))
    funcOp.setNocallback(true);
  if (llvm::Attribute attr = func->getFnAttribute("modular-format");
      attr.isStringAttribute())
    funcOp.setModularFormat(StringAttr::get(context, attr.getValueAsString()));
  if (llvm::Attribute attr = func->getFnAttribute("zero-call-used-regs");
      attr.isStringAttribute())
    funcOp.setZeroCallUsedRegsAttr(
        StringAttr::get(context, attr.getValueAsString()));

  if (func->hasFnAttribute("aarch64_pstate_sm_enabled"))
    funcOp.setArmStreaming(true);
  else if (func->hasFnAttribute("aarch64_pstate_sm_body"))
    funcOp.setArmLocallyStreaming(true);
  else if (func->hasFnAttribute("aarch64_pstate_sm_compatible"))
    funcOp.setArmStreamingCompatible(true);

  if (func->hasFnAttribute("aarch64_new_za"))
    funcOp.setArmNewZa(true);
  else if (func->hasFnAttribute("aarch64_in_za"))
    funcOp.setArmInZa(true);
  else if (func->hasFnAttribute("aarch64_out_za"))
    funcOp.setArmOutZa(true);
  else if (func->hasFnAttribute("aarch64_inout_za"))
    funcOp.setArmInoutZa(true);
  else if (func->hasFnAttribute("aarch64_preserves_za"))
    funcOp.setArmPreservesZa(true);

  convertNoBuiltinAttrs(context, func->getAttributes().getFnAttrs(), funcOp);
  convertAllocsizeAttr(context, func->getAttributes().getFnAttrs(), funcOp);

  llvm::Attribute attr = func->getFnAttribute(llvm::Attribute::VScaleRange);
  if (attr.isValid()) {
    MLIRContext *context = funcOp.getContext();
    auto intTy = IntegerType::get(context, 32);
    funcOp.setVscaleRangeAttr(LLVM::VScaleRangeAttr::get(
        context, IntegerAttr::get(intTy, attr.getVScaleRangeMin()),
        IntegerAttr::get(intTy, attr.getVScaleRangeMax().value_or(0))));
  }

  // Process frame-pointer attribute.
  if (func->hasFnAttribute("frame-pointer")) {
    StringRef stringRefFramePointerKind =
        func->getFnAttribute("frame-pointer").getValueAsString();
    funcOp.setFramePointerAttr(LLVM::FramePointerKindAttr::get(
        funcOp.getContext(), LLVM::framePointerKind::symbolizeFramePointerKind(
                                 stringRefFramePointerKind)
                                 .value()));
  }

  if (func->hasFnAttribute("use-sample-profile"))
    funcOp.setUseSampleProfile(true);

  if (llvm::Attribute attr = func->getFnAttribute("target-cpu");
      attr.isStringAttribute())
    funcOp.setTargetCpuAttr(StringAttr::get(context, attr.getValueAsString()));

  if (llvm::Attribute attr = func->getFnAttribute("tune-cpu");
      attr.isStringAttribute())
    funcOp.setTuneCpuAttr(StringAttr::get(context, attr.getValueAsString()));

  if (llvm::Attribute attr = func->getFnAttribute("target-features");
      attr.isStringAttribute())
    funcOp.setTargetFeaturesAttr(
        LLVM::TargetFeaturesAttr::get(context, attr.getValueAsString()));

  if (llvm::Attribute attr = func->getFnAttribute("reciprocal-estimates");
      attr.isStringAttribute())
    funcOp.setReciprocalEstimatesAttr(
        StringAttr::get(context, attr.getValueAsString()));

  if (llvm::Attribute attr = func->getFnAttribute("prefer-vector-width");
      attr.isStringAttribute())
    funcOp.setPreferVectorWidth(attr.getValueAsString());

  if (llvm::Attribute attr = func->getFnAttribute("instrument-function-entry");
      attr.isStringAttribute())
    funcOp.setInstrumentFunctionEntry(
        StringAttr::get(context, attr.getValueAsString()));

  if (llvm::Attribute attr = func->getFnAttribute("instrument-function-exit");
      attr.isStringAttribute())
    funcOp.setInstrumentFunctionExit(
        StringAttr::get(context, attr.getValueAsString()));

  if (llvm::Attribute attr = func->getFnAttribute("no-signed-zeros-fp-math");
      attr.isStringAttribute())
    funcOp.setNoSignedZerosFpMath(attr.getValueAsBool());

  if (llvm::Attribute attr = func->getFnAttribute("fp-contract");
      attr.isStringAttribute())
    funcOp.setFpContractAttr(StringAttr::get(context, attr.getValueAsString()));

  if (func->hasUWTable()) {
    ::llvm::UWTableKind uwtableKind = func->getUWTableKind();
    funcOp.setUwtableKindAttr(LLVM::UWTableKindAttr::get(
        funcOp.getContext(), convertUWTableKindFromLLVM(uwtableKind)));
  }
}

DictionaryAttr
ModuleImport::convertArgOrResultAttrSet(llvm::AttributeSet llvmAttrSet) {
  SmallVector<NamedAttribute> paramAttrs;
  for (auto [llvmKind, mlirName] : getAttrKindToNameMapping()) {
    auto llvmAttr = llvmAttrSet.getAttribute(llvmKind);
    // Skip attributes that are not attached.
    if (!llvmAttr.isValid())
      continue;

    // TODO: Import captures(none) as a nocapture unit attribute until the
    // LLVM dialect switches to the captures representation.
    if (llvmAttr.hasKindAsEnum() &&
        llvmAttr.getKindAsEnum() == llvm::Attribute::Captures) {
      if (llvm::capturesNothing(llvmAttr.getCaptureInfo()))
        paramAttrs.push_back(
            builder.getNamedAttr(mlirName, builder.getUnitAttr()));
      continue;
    }

    Attribute mlirAttr;
    if (llvmAttr.isTypeAttribute())
      mlirAttr = TypeAttr::get(convertType(llvmAttr.getValueAsType()));
    else if (llvmAttr.isIntAttribute())
      mlirAttr = builder.getI64IntegerAttr(llvmAttr.getValueAsInt());
    else if (llvmAttr.isEnumAttribute())
      mlirAttr = builder.getUnitAttr();
    else if (llvmAttr.isConstantRangeAttribute()) {
      const llvm::ConstantRange &value = llvmAttr.getValueAsConstantRange();
      mlirAttr = builder.getAttr<LLVM::ConstantRangeAttr>(value.getLower(),
                                                          value.getUpper());
    } else {
      llvm_unreachable("unexpected parameter attribute kind");
    }
    paramAttrs.push_back(builder.getNamedAttr(mlirName, mlirAttr));
  }

  return builder.getDictionaryAttr(paramAttrs);
}

void ModuleImport::convertArgAndResultAttrs(llvm::Function *func,
                                            LLVMFuncOp funcOp) {
  auto llvmAttrs = func->getAttributes();
  for (size_t i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
    llvm::AttributeSet llvmArgAttrs = llvmAttrs.getParamAttrs(i);
    funcOp.setArgAttrs(i, convertArgOrResultAttrSet(llvmArgAttrs));
  }
  // Convert the result attributes and attach them wrapped in an ArrayAttribute
  // to the funcOp.
  llvm::AttributeSet llvmResAttr = llvmAttrs.getRetAttrs();
  if (!llvmResAttr.hasAttributes())
    return;
  funcOp.setResAttrsAttr(
      builder.getArrayAttr({convertArgOrResultAttrSet(llvmResAttr)}));
}

void ModuleImport::convertArgAndResultAttrs(
    llvm::CallBase *call, ArgAndResultAttrsOpInterface attrsOp,
    ArrayRef<unsigned> immArgPositions) {
  // Compute the set of immediate argument positions.
  llvm::SmallDenseSet<unsigned> immArgPositionsSet(immArgPositions.begin(),
                                                   immArgPositions.end());
  // Convert the argument attributes and filter out immediate arguments.
  llvm::AttributeList llvmAttrs = call->getAttributes();
  SmallVector<llvm::AttributeSet> llvmArgAttrsSet;
  bool anyArgAttrs = false;
  for (size_t i = 0, e = call->arg_size(); i < e; ++i) {
    // Skip immediate arguments.
    if (immArgPositionsSet.contains(i))
      continue;
    llvmArgAttrsSet.emplace_back(llvmAttrs.getParamAttrs(i));
    if (llvmArgAttrsSet.back().hasAttributes())
      anyArgAttrs = true;
  }
  auto getArrayAttr = [&](ArrayRef<DictionaryAttr> dictAttrs) {
    SmallVector<Attribute> attrs;
    for (auto &dict : dictAttrs)
      attrs.push_back(dict ? dict : builder.getDictionaryAttr({}));
    return builder.getArrayAttr(attrs);
  };
  if (anyArgAttrs) {
    SmallVector<DictionaryAttr> argAttrs;
    for (auto &llvmArgAttrs : llvmArgAttrsSet)
      argAttrs.emplace_back(convertArgOrResultAttrSet(llvmArgAttrs));
    attrsOp.setArgAttrsAttr(getArrayAttr(argAttrs));
  }

  // Convert the result attributes.
  llvm::AttributeSet llvmResAttr = llvmAttrs.getRetAttrs();
  if (!llvmResAttr.hasAttributes())
    return;
  DictionaryAttr resAttrs = convertArgOrResultAttrSet(llvmResAttr);
  attrsOp.setResAttrsAttr(getArrayAttr({resAttrs}));
}

template <typename Op>
static LogicalResult convertCallBaseAttributes(llvm::CallBase *inst, Op op,
                                               ModuleImport &moduleImport) {
  op.setCConv(convertCConvFromLLVM(inst->getCallingConv()));
  if (inst->hasOperandBundles()) {
    SmallVector<Value> operands;
    SmallVector<int32_t> sizes;
    SmallVector<Attribute> tags;
    for (unsigned index = 0; index < inst->getNumOperandBundles(); ++index) {
      llvm::OperandBundleUse bundle = inst->getOperandBundleAt(index);
      sizes.push_back(bundle.Inputs.size());
      tags.push_back(StringAttr::get(op.getContext(), bundle.getTagName()));
      for (const llvm::Use &input : bundle.Inputs) {
        FailureOr<Value> value = moduleImport.convertValue(input.get());
        if (failed(value))
          return failure();
        operands.push_back(*value);
      }
    }
    op.getOpBundleOperandsMutable().join().assign(operands);
    op.setOpBundleSizesAttr(DenseI32ArrayAttr::get(op.getContext(), sizes));
    op.setOpBundleTagsAttr(ArrayAttr::get(op.getContext(), tags));
  }
  return success();
}

LogicalResult ModuleImport::convertInvokeAttributes(llvm::InvokeInst *inst,
                                                    InvokeOp op) {
  return convertCallBaseAttributes(inst, op, *this);
}

LogicalResult ModuleImport::convertCallAttributes(llvm::CallInst *inst,
                                                  CallOp op) {
  setFastmathFlagsAttr(inst, op.getOperation());
  // Query the attributes directly instead of using `inst->getFnAttr(Kind)`, the
  // latter does additional lookup to the parent and inherits, changing the
  // semantics too early.
  llvm::AttributeList callAttrs = inst->getAttributes();

  op.setTailCallKind(convertTailCallKindFromLLVM(inst->getTailCallKind()));
  op.setConvergent(callAttrs.getFnAttr(llvm::Attribute::Convergent).isValid());
  op.setNoUnwind(callAttrs.getFnAttr(llvm::Attribute::NoUnwind).isValid());
  op.setWillReturn(callAttrs.getFnAttr(llvm::Attribute::WillReturn).isValid());
  op.setNoreturn(callAttrs.getFnAttr(llvm::Attribute::NoReturn).isValid());
  op.setOptsize(
      callAttrs.getFnAttr(llvm::Attribute::OptimizeForSize).isValid());
  op.setSaveRegParams(callAttrs.getFnAttr("save-reg-params").isValid());
  op.setBuiltin(callAttrs.getFnAttr(llvm::Attribute::Builtin).isValid());
  op.setNobuiltin(callAttrs.getFnAttr(llvm::Attribute::NoBuiltin).isValid());
  op.setMinsize(callAttrs.getFnAttr(llvm::Attribute::MinSize).isValid());

  op.setReturnsTwice(
      callAttrs.getFnAttr(llvm::Attribute::ReturnsTwice).isValid());
  op.setHot(callAttrs.getFnAttr(llvm::Attribute::Hot).isValid());
  op.setCold(callAttrs.getFnAttr(llvm::Attribute::Cold).isValid());
  op.setNoduplicate(
      callAttrs.getFnAttr(llvm::Attribute::NoDuplicate).isValid());
  op.setNoCallerSavedRegisters(
      callAttrs.getFnAttr("no_caller_saved_registers").isValid());
  op.setNocallback(callAttrs.getFnAttr(llvm::Attribute::NoCallback).isValid());

  if (llvm::Attribute attr = callAttrs.getFnAttr("modular-format");
      attr.isStringAttribute())
    op.setModularFormat(StringAttr::get(context, attr.getValueAsString()));
  if (llvm::Attribute attr = callAttrs.getFnAttr("zero-call-used-regs");
      attr.isStringAttribute())
    op.setZeroCallUsedRegsAttr(
        StringAttr::get(context, attr.getValueAsString()));
  if (llvm::Attribute attr = callAttrs.getFnAttr("trap-func-name");
      attr.isStringAttribute())
    op.setTrapFuncNameAttr(StringAttr::get(context, attr.getValueAsString()));
  op.setNoInline(callAttrs.getFnAttr(llvm::Attribute::NoInline).isValid());
  op.setAlwaysInline(
      callAttrs.getFnAttr(llvm::Attribute::AlwaysInline).isValid());
  op.setInlineHint(callAttrs.getFnAttr(llvm::Attribute::InlineHint).isValid());

  llvm::MemoryEffects memEffects = inst->getMemoryEffects();
  ModRefInfo othermem = convertModRefInfoFromLLVM(
      memEffects.getModRef(llvm::MemoryEffects::Location::Other));
  ModRefInfo argMem = convertModRefInfoFromLLVM(
      memEffects.getModRef(llvm::MemoryEffects::Location::ArgMem));
  ModRefInfo inaccessibleMem = convertModRefInfoFromLLVM(
      memEffects.getModRef(llvm::MemoryEffects::Location::InaccessibleMem));
  ModRefInfo errnoMem = convertModRefInfoFromLLVM(
      memEffects.getModRef(llvm::MemoryEffects::Location::ErrnoMem));
  ModRefInfo targetMem0 = convertModRefInfoFromLLVM(
      memEffects.getModRef(llvm::MemoryEffects::Location::TargetMem0));
  ModRefInfo targetMem1 = convertModRefInfoFromLLVM(
      memEffects.getModRef(llvm::MemoryEffects::Location::TargetMem1));
  auto memAttr =
      MemoryEffectsAttr::get(op.getContext(), othermem, argMem, inaccessibleMem,
                             errnoMem, targetMem0, targetMem1);
  // Only set the attribute when it does not match the default value.
  if (!memAttr.isReadWrite())
    op.setMemoryEffectsAttr(memAttr);

  convertNoBuiltinAttrs(op.getContext(), callAttrs.getFnAttrs(), op);
  convertAllocsizeAttr(op.getContext(), callAttrs.getFnAttrs(), op);

  return convertCallBaseAttributes(inst, op, *this);
}

FailureOr<SmallVector<AccessGroupAttr>>
ModuleImport::lookupAccessGroupAttrs(const llvm::MDNode *node) const {
  return loopAnnotationImporter->lookupAccessGroupAttrs(node);
}

LoopAnnotationAttr
ModuleImport::translateLoopAnnotationAttr(const llvm::MDNode *node,
                                          Location loc) const {
  return loopAnnotationImporter->translateLoopAnnotation(node, loc);
}

FailureOr<DereferenceableAttr>
ModuleImport::translateDereferenceableAttr(const llvm::MDNode *node,
                                           unsigned kindID) {
  Location loc = mlirModule.getLoc();

  // The only operand should be a constant integer representing the number of
  // dereferenceable bytes.
  if (node->getNumOperands() != 1)
    return emitError(loc) << "dereferenceable metadata must have one operand: "
                          << diagMD(node, llvmModule.get());

  auto *numBytesMD = dyn_cast<llvm::ConstantAsMetadata>(node->getOperand(0));
  auto *numBytesCst = dyn_cast<llvm::ConstantInt>(numBytesMD->getValue());
  if (!numBytesCst || !numBytesCst->getValue().isNonNegative())
    return emitError(loc) << "dereferenceable metadata operand must be a "
                             "non-negative constant integer: "
                          << diagMD(node, llvmModule.get());

  bool mayBeNull = kindID == llvm::LLVMContext::MD_dereferenceable_or_null;
  auto derefAttr = builder.getAttr<DereferenceableAttr>(
      numBytesCst->getZExtValue(), mayBeNull);

  return derefAttr;
}
