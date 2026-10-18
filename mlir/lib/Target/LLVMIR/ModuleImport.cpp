//===- ModuleImport.cpp - LLVM to MLIR conversion ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the import of an LLVM IR module into an LLVM dialect
// module.
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

/// Returns the symbol name for the module-level comdat operation. It must not
/// conflict with the user namespace.
static constexpr StringRef getGlobalComdatOpName() {
  return "__llvm_global_comdat";
}

/// Converts the sync scope identifier of `inst` to the string representation
/// necessary to build an atomic LLVM dialect operation. Returns the empty
/// string if the operation has either no sync scope or the default system-level
/// sync scope attached. The atomic operations only set their sync scope
/// attribute if they have a non-default sync scope attached.
static StringRef getLLVMSyncScope(llvm::Instruction *inst) {
  std::optional<llvm::SyncScope::ID> syncScopeID =
      llvm::getAtomicSyncScopeID(inst);
  if (!syncScopeID)
    return "";

  // Search the sync scope name for the given identifier. The default
  // system-level sync scope thereby maps to the empty string.
  SmallVector<StringRef> syncScopeName;
  llvm::LLVMContext &llvmContext = inst->getContext();
  llvmContext.getSyncScopeNames(syncScopeName);
  auto *it = llvm::find_if(syncScopeName, [&](StringRef name) {
    return *syncScopeID == llvmContext.getOrInsertSyncScopeID(name);
  });
  if (it != syncScopeName.end())
    return *it;
  llvm_unreachable("incorrect sync scope identifier");
}

/// Converts an array of unsigned indices to a signed integer position array.
static SmallVector<int64_t> getPositionFromIndices(ArrayRef<unsigned> indices) {
  SmallVector<int64_t> position;
  llvm::append_range(position, indices);
  return position;
}

/// Converts the LLVM instructions that have a generated MLIR builder. Using a
/// static implementation method called from the module import ensures the
/// builders have to use the `moduleImport` argument and cannot directly call
/// import methods. As a result, both the intrinsic and the instruction MLIR
/// builders have to use the `moduleImport` argument and none of them has direct
/// access to the private module import methods.
static LogicalResult convertInstructionImpl(OpBuilder &odsBuilder,
                                            llvm::Instruction *inst,
                                            ModuleImport &moduleImport,
                                            LLVMImportInterface &iface) {
  // Copy the operands to an LLVM operands array reference for conversion.
  SmallVector<llvm::Value *> operands(inst->operands());
  ArrayRef<llvm::Value *> llvmOperands(operands);

  // Convert all instructions that provide an MLIR builder.
  if (iface.isConvertibleInstruction(inst->getOpcode()))
    return iface.convertInstruction(odsBuilder, inst, llvmOperands,
                                    moduleImport);
  // TODO: Implement the `convertInstruction` hooks in the
  // `LLVMDialectLLVMIRImportInterface` and move the following include there.
#include "mlir/Dialect/LLVMIR/LLVMOpFromLLVMIRConversions.inc"

  return failure();
}

/// Get a topologically sorted list of blocks for the given basic blocks.
static SetVector<llvm::BasicBlock *>
getTopologicallySortedBlocks(ArrayRef<llvm::BasicBlock *> basicBlocks) {
  SetVector<llvm::BasicBlock *> blocks;
  for (llvm::BasicBlock *basicBlock : basicBlocks) {
    if (!blocks.contains(basicBlock)) {
      llvm::ReversePostOrderTraversal<llvm::BasicBlock *> traversal(basicBlock);
      blocks.insert_range(traversal);
    }
  }
  assert(blocks.size() == basicBlocks.size() && "some blocks are not sorted");
  return blocks;
}

ModuleImport::ModuleImport(ModuleOp mlirModule,
                           std::unique_ptr<llvm::Module> llvmModule,
                           bool emitExpensiveWarnings,
                           bool importEmptyDICompositeTypes,
                           bool preferUnregisteredIntrinsics,
                           bool importStructsAsLiterals)
    : builder(mlirModule->getContext()), context(mlirModule->getContext()),
      mlirModule(mlirModule), llvmModule(std::move(llvmModule)),
      iface(mlirModule->getContext()),
      typeTranslator(*mlirModule->getContext(), importStructsAsLiterals),
      debugImporter(std::make_unique<DebugImporter>(
          mlirModule, importEmptyDICompositeTypes)),
      loopAnnotationImporter(
          std::make_unique<LoopAnnotationImporter>(*this, builder)),
      emitExpensiveWarnings(emitExpensiveWarnings),
      preferUnregisteredIntrinsics(preferUnregisteredIntrinsics) {
  builder.setInsertionPointToStart(mlirModule.getBody());
}

ComdatOp ModuleImport::getGlobalComdatOp() {
  if (globalComdatOp)
    return globalComdatOp;

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(mlirModule.getBody());
  globalComdatOp =
      ComdatOp::create(builder, mlirModule.getLoc(), getGlobalComdatOpName());
  globalInsertionOp = globalComdatOp;
  return globalComdatOp;
}

void ModuleImport::processComdat(const llvm::Comdat *comdat) {
  if (comdatMapping.contains(comdat))
    return;

  ComdatOp comdatOp = getGlobalComdatOp();
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(&comdatOp.getBody().back());
  auto selectorOp = ComdatSelectorOp::create(
      builder, mlirModule.getLoc(), comdat->getName(),
      convertComdatFromLLVM(comdat->getSelectionKind()));
  auto symbolRef =
      SymbolRefAttr::get(builder.getContext(), getGlobalComdatOpName(),
                         FlatSymbolRefAttr::get(selectorOp.getSymNameAttr()));
  comdatMapping.try_emplace(comdat, symbolRef);
}

LogicalResult ModuleImport::convertComdats() {
  for (llvm::GlobalVariable &globalVar : llvmModule->globals())
    if (globalVar.hasComdat())
      processComdat(globalVar.getComdat());
  for (llvm::Function &func : llvmModule->functions())
    if (func.hasComdat())
      processComdat(func.getComdat());
  return success();
}

LogicalResult ModuleImport::convertGlobals() {
  for (llvm::GlobalVariable &globalVar : llvmModule->globals()) {
    if (globalVar.getName() == getGlobalCtorsVarName() ||
        globalVar.getName() == getGlobalDtorsVarName()) {
      if (failed(convertGlobalCtorsAndDtors(&globalVar))) {
        return emitError(UnknownLoc::get(context))
               << "unhandled global variable: " << diag(globalVar);
      }
      continue;
    }
    if (failed(convertGlobal(&globalVar))) {
      return emitError(UnknownLoc::get(context))
             << "unhandled global variable: " << diag(globalVar);
    }
  }
  return success();
}

LogicalResult ModuleImport::convertAliases() {
  for (llvm::GlobalAlias &alias : llvmModule->aliases()) {
    if (failed(convertAlias(&alias))) {
      return emitError(UnknownLoc::get(context))
             << "unhandled global alias: " << diag(alias);
    }
  }
  return success();
}

LogicalResult ModuleImport::convertIFuncs() {
  for (llvm::GlobalIFunc &ifunc : llvmModule->ifuncs()) {
    if (failed(convertIFunc(&ifunc))) {
      return emitError(UnknownLoc::get(context))
             << "unhandled global ifunc: " << diag(ifunc);
    }
  }
  return success();
}

LogicalResult ModuleImport::convertDataLayout() {
  Location loc = mlirModule.getLoc();
  DataLayoutImporter dataLayoutImporter(
      context, llvmModule->getDataLayout().getStringRepresentation());
  if (!dataLayoutImporter.getDataLayoutSpec())
    return emitError(loc, "cannot translate data layout: ")
           << dataLayoutImporter.getLastToken();

  for (StringRef token : dataLayoutImporter.getUnhandledTokens())
    emitWarning(loc, "unhandled data layout token: ") << token;

  mlirModule->setAttr(DLTIDialect::kDataLayoutAttrName,
                      dataLayoutImporter.getDataLayoutSpec());
  return success();
}

void ModuleImport::convertTargetTriple() {
  mlirModule->setAttr(
      LLVM::LLVMDialect::getTargetTripleAttrName(),
      builder.getStringAttr(llvmModule->getTargetTriple().str()));
}

void ModuleImport::convertModuleLevelAsm() {
  llvm::SmallVector<mlir::Attribute> asmArrayAttr;

  for (const llvm::Module::GlobalAsmFragment &Frag :
       llvmModule->getModuleInlineAsm()) {
    // TODO: Preserve module asm properties.
    for (llvm::StringRef line : llvm::split(Frag.Asm, '\n'))
      if (!line.empty())
        asmArrayAttr.push_back(builder.getStringAttr(line));
  }

  mlirModule->setAttr(LLVM::LLVMDialect::getModuleLevelAsmAttrName(),
                      builder.getArrayAttr(asmArrayAttr));
}

LogicalResult ModuleImport::convertFunctions() {
  for (llvm::Function &func : llvmModule->functions())
    if (failed(processFunction(&func)))
      return failure();
  return success();
}

void ModuleImport::setIntegerOverflowFlags(llvm::Instruction *inst,
                                           Operation *op) const {
  auto iface = cast<IntegerOverflowFlagsInterface>(op);

  IntegerOverflowFlags value = {};
  value = bitEnumSet(value, IntegerOverflowFlags::nsw, inst->hasNoSignedWrap());
  value =
      bitEnumSet(value, IntegerOverflowFlags::nuw, inst->hasNoUnsignedWrap());

  iface.setOverflowFlags(value);
}

void ModuleImport::setExactFlag(llvm::Instruction *inst, Operation *op) const {
  auto iface = cast<ExactFlagInterface>(op);

  iface.setIsExact(inst->isExact());
}

void ModuleImport::setDisjointFlag(llvm::Instruction *inst,
                                   Operation *op) const {
  auto iface = cast<DisjointFlagInterface>(op);
  auto *instDisjoint = cast<llvm::PossiblyDisjointInst>(inst);

  iface.setIsDisjoint(instDisjoint->isDisjoint());
}

void ModuleImport::setNonNegFlag(llvm::Instruction *inst, Operation *op) const {
  auto iface = cast<NonNegFlagInterface>(op);

  iface.setNonNeg(inst->hasNonNeg());
}

void ModuleImport::setFastmathFlagsAttr(llvm::Instruction *inst,
                                        Operation *op) const {
  auto iface = cast<FastmathFlagsInterface>(op);

  // Even if the imported operation implements the fastmath interface, the
  // original instruction may not have fastmath flags set. Exit if an
  // instruction, such as a non floating-point function call, does not have
  // fastmath flags.
  if (!isa<llvm::FPMathOperator>(inst))
    return;
  llvm::FastMathFlags flags = inst->getFastMathFlags();

  // Set the fastmath bits flag-by-flag.
  FastmathFlags value = {};
  value = bitEnumSet(value, FastmathFlags::nnan, flags.noNaNs());
  value = bitEnumSet(value, FastmathFlags::ninf, flags.noInfs());
  value = bitEnumSet(value, FastmathFlags::nsz, flags.noSignedZeros());
  value = bitEnumSet(value, FastmathFlags::arcp, flags.allowReciprocal());
  value = bitEnumSet(value, FastmathFlags::contract, flags.allowContract());
  value = bitEnumSet(value, FastmathFlags::afn, flags.approxFunc());
  value = bitEnumSet(value, FastmathFlags::reassoc, flags.allowReassoc());
  FastmathFlagsAttr attr = FastmathFlagsAttr::get(builder.getContext(), value);
  iface->setAttr(iface.getFastmathAttrName(), attr);
}

Location ModuleImport::translateLoc(llvm::DILocation *loc) {
  return debugImporter->translateLoc(loc);
}

LogicalResult
ModuleImport::convertBranchArgs(llvm::Instruction *branch,
                                llvm::BasicBlock *target,
                                SmallVectorImpl<Value> &blockArguments) {
  for (auto inst = target->begin(); isa<llvm::PHINode>(inst); ++inst) {
    auto *phiInst = cast<llvm::PHINode>(&*inst);
    llvm::Value *value = phiInst->getIncomingValueForBlock(branch->getParent());
    FailureOr<Value> converted = convertValue(value);
    if (failed(converted))
      return failure();
    blockArguments.push_back(*converted);
  }
  return success();
}

FailureOr<SmallVector<Value>>
ModuleImport::convertCallOperands(llvm::CallBase *callInst,
                                  bool allowInlineAsm) {
  bool isInlineAsm = callInst->isInlineAsm();
  if (isInlineAsm && !allowInlineAsm)
    return failure();

  SmallVector<Value> operands;

  // Cannot use isIndirectCall() here because we need to handle Constant callees
  // that are not considered indirect calls by LLVM. However, in MLIR, they are
  // treated as indirect calls to constant operands that need to be converted.
  // Skip the callee operand if it's inline assembly, as it's handled separately
  // in InlineAsmOp.
  llvm::Value *calleeOperand = callInst->getCalledOperand();
  if (!isa<llvm::Function, llvm::GlobalIFunc>(calleeOperand) && !isInlineAsm) {
    FailureOr<Value> called = convertValue(calleeOperand);
    if (failed(called))
      return failure();
    operands.push_back(*called);
  }

  SmallVector<llvm::Value *> args(callInst->args());
  FailureOr<SmallVector<Value>> arguments = convertValues(args);
  if (failed(arguments))
    return failure();

  llvm::append_range(operands, *arguments);
  return operands;
}

/// Checks if `callType` and `calleeType` are compatible and can be represented
/// in MLIR.
static LogicalResult
checkFunctionTypeCompatibility(LLVMFunctionType callType,
                               LLVMFunctionType calleeType) {
  if (callType.getReturnType() != calleeType.getReturnType())
    return failure();

  if (calleeType.isVarArg()) {
    // For variadic functions, the call can have more types than the callee
    // specifies.
    if (callType.getNumParams() < calleeType.getNumParams())
      return failure();
  } else {
    // For non-variadic functions, the number of parameters needs to be the
    // same.
    if (callType.getNumParams() != calleeType.getNumParams())
      return failure();
  }

  // Check that all operands match.
  for (auto [operandType, argumentType] :
       llvm::zip(callType.getParams(), calleeType.getParams()))
    if (operandType != argumentType)
      return failure();

  return success();
}

FailureOr<LLVMFunctionType>
ModuleImport::convertFunctionType(llvm::CallBase *callInst,
                                  bool &isIncompatibleCall) {
  isIncompatibleCall = false;
  auto castOrFailure = [](Type convertedType) -> FailureOr<LLVMFunctionType> {
    auto funcTy = dyn_cast_or_null<LLVMFunctionType>(convertedType);
    if (!funcTy)
      return failure();
    return funcTy;
  };

  llvm::Value *calledOperand = callInst->getCalledOperand();
  FailureOr<LLVMFunctionType> callType =
      castOrFailure(convertType(callInst->getFunctionType()));
  if (failed(callType))
    return failure();
  auto *callee = dyn_cast<llvm::Function>(calledOperand);

  llvm::FunctionType *origCalleeType = nullptr;
  if (callee) {
    origCalleeType = callee->getFunctionType();
  } else if (auto *ifunc = dyn_cast<llvm::GlobalIFunc>(calledOperand)) {
    origCalleeType = cast<llvm::FunctionType>(ifunc->getValueType());
  }

  // For indirect calls, return the type of the call itself.
  if (!origCalleeType)
    return callType;

  FailureOr<LLVMFunctionType> calleeType =
      castOrFailure(convertType(origCalleeType));
  if (failed(calleeType))
    return failure();

  // Compare the types and notify users via `isIncompatibleCall` if they are not
  // compatible.
  if (failed(checkFunctionTypeCompatibility(*callType, *calleeType))) {
    isIncompatibleCall = true;
    Location loc = translateLoc(callInst->getDebugLoc());
    emitWarning(loc) << "incompatible call and callee types: " << *callType
                     << " and " << *calleeType;
    return callType;
  }

  return calleeType;
}

FlatSymbolRefAttr ModuleImport::convertCalleeName(llvm::CallBase *callInst) {
  llvm::Value *calledOperand = callInst->getCalledOperand();
  if (isa<llvm::Function, llvm::GlobalIFunc>(calledOperand))
    return SymbolRefAttr::get(context, calledOperand->getName());
  return {};
}

LogicalResult ModuleImport::convertIntrinsic(llvm::CallInst *inst) {
  if (succeeded(iface.convertIntrinsic(builder, inst, *this)))
    return success();

  Location loc = translateLoc(inst->getDebugLoc());
  return emitError(loc) << "unhandled intrinsic: " << diag(*inst);
}

ArrayAttr
ModuleImport::convertAsmInlineOperandAttrs(const llvm::CallBase &llvmCall) {
  const auto *ia = cast<llvm::InlineAsm>(llvmCall.getCalledOperand());
  unsigned argIdx = 0;
  SmallVector<mlir::Attribute> opAttrs;
  bool hasIndirect = false;

  for (const llvm::InlineAsm::ConstraintInfo &ci : ia->ParseConstraints()) {
    // Only deal with constraints that correspond to call arguments.
    if (ci.Type == llvm::InlineAsm::isLabel || !ci.hasArg())
      continue;

    // Only increment `argIdx` in terms of constraints containing arguments,
    // which are guaranteed to happen in the same order of the call arguments.
    if (ci.isIndirect) {
      if (llvm::Type *paramEltType = llvmCall.getParamElementType(argIdx)) {
        SmallVector<mlir::NamedAttribute> attrs;
        attrs.push_back(builder.getNamedAttr(
            mlir::LLVM::InlineAsmOp::getElementTypeAttrName(),
            mlir::TypeAttr::get(convertType(paramEltType))));
        opAttrs.push_back(builder.getDictionaryAttr(attrs));
        hasIndirect = true;
      }
    } else {
      opAttrs.push_back(builder.getDictionaryAttr({}));
    }
    argIdx++;
  }

  // Avoid emitting an array where all entries are empty dictionaries.
  return hasIndirect ? ArrayAttr::get(mlirModule->getContext(), opAttrs)
                     : nullptr;
}

LogicalResult ModuleImport::convertInstruction(llvm::Instruction *inst) {
  // Convert all instructions that do not provide an MLIR builder.
  Location loc = translateLoc(inst->getDebugLoc());
  if (auto *brInst = dyn_cast<llvm::UncondBrInst>(inst)) {
    llvm::BasicBlock *succ = brInst->getSuccessor();
    SmallVector<Value> blockArgs;
    if (failed(convertBranchArgs(brInst, succ, blockArgs)))
      return failure();

    auto brOp = LLVM::BrOp::create(builder, loc, blockArgs, lookupBlock(succ));
    mapNoResultOp(inst, brOp);
    return success();
  }
  if (auto *brInst = dyn_cast<llvm::CondBrInst>(inst)) {
    SmallVector<Block *> succBlocks;
    SmallVector<SmallVector<Value>> succBlockArgs;
    for (auto i : llvm::seq<unsigned>(0, brInst->getNumSuccessors())) {
      llvm::BasicBlock *succ = brInst->getSuccessor(i);
      SmallVector<Value> blockArgs;
      if (failed(convertBranchArgs(brInst, succ, blockArgs)))
        return failure();
      succBlocks.push_back(lookupBlock(succ));
      succBlockArgs.push_back(blockArgs);
    }

    FailureOr<Value> condition = convertValue(brInst->getCondition());
    if (failed(condition))
      return failure();
    auto condBrOp = LLVM::CondBrOp::create(
        builder, loc, *condition, succBlocks.front(), succBlockArgs.front(),
        succBlocks.back(), succBlockArgs.back());
    mapNoResultOp(inst, condBrOp);
    return success();
  }
  if (inst->getOpcode() == llvm::Instruction::Switch) {
    auto *swInst = cast<llvm::SwitchInst>(inst);
    // Process the condition value.
    FailureOr<Value> condition = convertValue(swInst->getCondition());
    if (failed(condition))
      return failure();
    SmallVector<Value> defaultBlockArgs;
    // Process the default case.
    llvm::BasicBlock *defaultBB = swInst->getDefaultDest();
    if (failed(convertBranchArgs(swInst, defaultBB, defaultBlockArgs)))
      return failure();

    // Process the cases.
    unsigned numCases = swInst->getNumCases();
    SmallVector<SmallVector<Value>> caseOperands(numCases);
    SmallVector<ValueRange> caseOperandRefs(numCases);
    SmallVector<APInt> caseValues(numCases);
    SmallVector<Block *> caseBlocks(numCases);
    for (const auto &it : llvm::enumerate(swInst->cases())) {
      const llvm::SwitchInst::CaseHandle &caseHandle = it.value();
      llvm::BasicBlock *succBB = caseHandle.getCaseSuccessor();
      if (failed(convertBranchArgs(swInst, succBB, caseOperands[it.index()])))
        return failure();
      caseOperandRefs[it.index()] = caseOperands[it.index()];
      caseValues[it.index()] = caseHandle.getCaseValue()->getValue();
      caseBlocks[it.index()] = lookupBlock(succBB);
    }

    auto switchOp = SwitchOp::create(builder, loc, *condition,
                                     lookupBlock(defaultBB), defaultBlockArgs,
                                     caseValues, caseBlocks, caseOperandRefs);
    mapNoResultOp(inst, switchOp);
    return success();
  }
  if (inst->getOpcode() == llvm::Instruction::PHI) {
    Type type = convertType(inst->getType());
    mapValue(inst, builder.getInsertionBlock()->addArgument(
                       type, translateLoc(inst->getDebugLoc())));
    return success();
  }
  if (inst->getOpcode() == llvm::Instruction::Call) {
    auto *callInst = cast<llvm::CallInst>(inst);
    llvm::Value *calledOperand = callInst->getCalledOperand();

    FailureOr<SmallVector<Value>> operands =
        convertCallOperands(callInst, /*allowInlineAsm=*/true);
    if (failed(operands))
      return failure();

    auto callOp = [&]() -> FailureOr<Operation *> {
      if (auto *asmI = dyn_cast<llvm::InlineAsm>(calledOperand)) {
        Type resultTy = convertType(callInst->getType());
        if (!resultTy)
          return failure();
        ArrayAttr operandAttrs = convertAsmInlineOperandAttrs(*callInst);
        return InlineAsmOp::create(
                   builder, loc, resultTy, *operands,
                   builder.getStringAttr(asmI->getAsmString()),
                   builder.getStringAttr(asmI->getConstraintString()),
                   asmI->hasSideEffects(), asmI->isAlignStack(),
                   convertTailCallKindFromLLVM(callInst->getTailCallKind()),
                   AsmDialectAttr::get(
                       mlirModule.getContext(),
                       convertAsmDialectFromLLVM(asmI->getDialect())),
                   operandAttrs)
            .getOperation();
      }
      bool isIncompatibleCall;
      FailureOr<LLVMFunctionType> funcTy =
          convertFunctionType(callInst, isIncompatibleCall);
      if (failed(funcTy))
        return failure();

      FlatSymbolRefAttr callee = nullptr;
      if (isIncompatibleCall) {
        // Use an indirect call (in order to represent valid and verifiable LLVM
        // IR). Build the indirect call by passing an empty `callee` operand and
        // insert into `operands` to include the indirect call target.
        FlatSymbolRefAttr calleeSym = convertCalleeName(callInst);
        Value indirectCallVal = LLVM::AddressOfOp::create(
            builder, loc, LLVM::LLVMPointerType::get(context), calleeSym);
        operands->insert(operands->begin(), indirectCallVal);
      } else {
        // Regular direct call using callee name.
        callee = convertCalleeName(callInst);
      }
      CallOp callOp = CallOp::create(builder, loc, *funcTy, callee, *operands);

      if (failed(convertCallAttributes(callInst, callOp)))
        return failure();

      // Handle parameter and result attributes unless it's an incompatible
      // call.
      if (!isIncompatibleCall)
        convertArgAndResultAttrs(callInst, callOp);
      return callOp.getOperation();
    }();

    if (failed(callOp))
      return failure();

    if (!callInst->getType()->isVoidTy())
      mapValue(inst, (*callOp)->getResult(0));
    else
      mapNoResultOp(inst, *callOp);
    return success();
  }
  if (inst->getOpcode() == llvm::Instruction::LandingPad) {
    auto *lpInst = cast<llvm::LandingPadInst>(inst);

    SmallVector<Value> operands;
    operands.reserve(lpInst->getNumClauses());
    for (auto i : llvm::seq<unsigned>(0, lpInst->getNumClauses())) {
      FailureOr<Value> operand = convertValue(lpInst->getClause(i));
      if (failed(operand))
        return failure();
      operands.push_back(*operand);
    }

    Type type = convertType(lpInst->getType());
    auto lpOp =
        LandingpadOp::create(builder, loc, type, lpInst->isCleanup(), operands);
    mapValue(inst, lpOp);
    return success();
  }
  if (inst->getOpcode() == llvm::Instruction::Invoke) {
    auto *invokeInst = cast<llvm::InvokeInst>(inst);

    if (invokeInst->isInlineAsm())
      return emitError(loc) << "invoke of inline assembly is not supported";

    FailureOr<SmallVector<Value>> operands = convertCallOperands(invokeInst);
    if (failed(operands))
      return failure();

    // Check whether the invoke result is an argument to the normal destination
    // block.
    bool invokeResultUsedInPhi = llvm::any_of(
        invokeInst->getNormalDest()->phis(), [&](const llvm::PHINode &phi) {
          return phi.getIncomingValueForBlock(invokeInst->getParent()) ==
                 invokeInst;
        });

    Block *normalDest = lookupBlock(invokeInst->getNormalDest());
    Block *directNormalDest = normalDest;
    if (invokeResultUsedInPhi) {
      // The invoke result cannot be an argument to the normal destination
      // block, as that would imply using the invoke operation result in its
      // definition, so we need to create a dummy block to serve as an
      // intermediate destination.
      OpBuilder::InsertionGuard g(builder);
      directNormalDest = builder.createBlock(normalDest);
    }

    SmallVector<Value> unwindArgs;
    if (failed(convertBranchArgs(invokeInst, invokeInst->getUnwindDest(),
                                 unwindArgs)))
      return failure();

    bool isIncompatibleInvoke;
    FailureOr<LLVMFunctionType> funcTy =
        convertFunctionType(invokeInst, isIncompatibleInvoke);
    if (failed(funcTy))
      return failure();

    FlatSymbolRefAttr calleeName = nullptr;
    if (isIncompatibleInvoke) {
      // Use an indirect invoke (in order to represent valid and verifiable LLVM
      // IR). Build the indirect invoke by passing an empty `callee` operand and
      // insert into `operands` to include the indirect invoke target.
      FlatSymbolRefAttr calleeSym = convertCalleeName(invokeInst);
      Value indirectInvokeVal = LLVM::AddressOfOp::create(
          builder, loc, LLVM::LLVMPointerType::get(context), calleeSym);
      operands->insert(operands->begin(), indirectInvokeVal);
    } else {
      // Regular direct invoke using callee name.
      calleeName = convertCalleeName(invokeInst);
    }
    // Create the invoke operation. Normal destination block arguments will be
    // added later on to handle the case in which the operation result is
    // included in this list.
    auto invokeOp = InvokeOp::create(
        builder, loc, *funcTy, calleeName, *operands, directNormalDest,
        ValueRange(), lookupBlock(invokeInst->getUnwindDest()), unwindArgs);

    if (failed(convertInvokeAttributes(invokeInst, invokeOp)))
      return failure();

    // Handle parameter and result attributes unless it's an incompatible
    // invoke.
    if (!isIncompatibleInvoke)
      convertArgAndResultAttrs(invokeInst, invokeOp);

    if (!invokeInst->getType()->isVoidTy())
      mapValue(inst, invokeOp.getResults().front());
    else
      mapNoResultOp(inst, invokeOp);

    SmallVector<Value> normalArgs;
    if (failed(convertBranchArgs(invokeInst, invokeInst->getNormalDest(),
                                 normalArgs)))
      return failure();

    if (invokeResultUsedInPhi) {
      // The dummy normal dest block will just host an unconditional branch
      // instruction to the normal destination block passing the required block
      // arguments (including the invoke operation's result).
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToStart(directNormalDest);
      LLVM::BrOp::create(builder, loc, normalArgs, normalDest);
    } else {
      // If the invoke operation's result is not a block argument to the normal
      // destination block, just add the block arguments as usual.
      assert(llvm::none_of(
                 normalArgs,
                 [&](Value val) { return val.getDefiningOp() == invokeOp; }) &&
             "An llvm.invoke operation cannot pass its result as a block "
             "argument.");
      invokeOp.getNormalDestOperandsMutable().append(normalArgs);
    }

    return success();
  }
  if (inst->getOpcode() == llvm::Instruction::GetElementPtr) {
    auto *gepInst = cast<llvm::GetElementPtrInst>(inst);
    Type sourceElementType = convertType(gepInst->getSourceElementType());
    FailureOr<Value> basePtr = convertValue(gepInst->getOperand(0));
    if (failed(basePtr))
      return failure();

    // Treat every indices as dynamic since GEPOp::build will refine those
    // indices into static attributes later. One small downside of this
    // approach is that many unused `llvm.mlir.constant` would be emitted
    // at first place.
    SmallVector<GEPArg> indices;
    for (llvm::Value *operand : llvm::drop_begin(gepInst->operand_values())) {
      FailureOr<Value> index = convertValue(operand);
      if (failed(index))
        return failure();
      indices.push_back(*index);
    }

    Type type = convertType(inst->getType());
    auto gepOp = GEPOp::create(
        builder, loc, type, sourceElementType, *basePtr, indices,
        static_cast<GEPNoWrapFlags>(gepInst->getNoWrapFlags().getRaw()));
    mapValue(inst, gepOp);
    return success();
  }

  if (inst->getOpcode() == llvm::Instruction::IndirectBr) {
    auto *indBrInst = cast<llvm::IndirectBrInst>(inst);

    FailureOr<Value> basePtr = convertValue(indBrInst->getAddress());
    if (failed(basePtr))
      return failure();

    SmallVector<Block *> succBlocks;
    SmallVector<SmallVector<Value>> succBlockArgs;
    for (auto i : llvm::seq<unsigned>(0, indBrInst->getNumSuccessors())) {
      llvm::BasicBlock *succ = indBrInst->getSuccessor(i);
      SmallVector<Value> blockArgs;
      if (failed(convertBranchArgs(indBrInst, succ, blockArgs)))
        return failure();
      succBlocks.push_back(lookupBlock(succ));
      succBlockArgs.push_back(blockArgs);
    }
    SmallVector<ValueRange> succBlockArgsRange =
        llvm::to_vector_of<ValueRange>(succBlockArgs);
    Location loc = translateLoc(inst->getDebugLoc());
    auto indBrOp = LLVM::IndirectBrOp::create(builder, loc, *basePtr,
                                              succBlockArgsRange, succBlocks);

    mapNoResultOp(inst, indBrOp);
    return success();
  }

  // Convert all instructions that have an mlirBuilder.
  if (succeeded(convertInstructionImpl(builder, inst, *this, iface)))
    return success();

  return emitError(loc) << "unhandled instruction: " << diag(*inst);
}

LogicalResult ModuleImport::processInstruction(llvm::Instruction *inst) {
  // FIXME: Support uses of SubtargetData.
  // FIXME: Add support for call / operand attributes.
  // FIXME: Add support for the cleanupret, catchret, catchswitch, callbr,
  // vaarg, catchpad, cleanuppad instructions.

  // Convert LLVM intrinsics calls to MLIR intrinsics.
  if (auto *intrinsic = dyn_cast<llvm::IntrinsicInst>(inst))
    return convertIntrinsic(intrinsic);

  // Process debug records attached to this instruction. Debug variable records
  // are stored for later processing after all SSA values are converted, while
  // debug label records can be converted immediately.
  if (inst->DebugMarker) {
    for (llvm::DbgRecord &dbgRecord : inst->DebugMarker->getDbgRecordRange()) {
      // Store debug variable records for later processing.
      if (auto *dbgVariableRecord =
              dyn_cast<llvm::DbgVariableRecord>(&dbgRecord)) {
        addDebugRecord(dbgVariableRecord);
        continue;
      }
      Location loc = translateLoc(dbgRecord.getDebugLoc());
      auto emitUnsupportedWarning = [&]() -> LogicalResult {
        if (!emitExpensiveWarnings)
          return success();
        std::string options;
        llvm::raw_string_ostream optionsStream(options);
        dbgRecord.print(optionsStream);
        emitWarning(loc) << "unhandled debug record " << optionsStream.str();
        return success();
      };
      // Convert the debug label records in-place.
      if (auto *dbgLabelRecord = dyn_cast<llvm::DbgLabelRecord>(&dbgRecord)) {
        DILabelAttr labelAttr =
            debugImporter->translate(dbgLabelRecord->getLabel());
        if (!labelAttr)
          return emitUnsupportedWarning();
        LLVM::DbgLabelOp::create(builder, loc, labelAttr);
        continue;
      }
      // Warn if an unsupported debug record is encountered.
      return emitUnsupportedWarning();
    }
  }

  // Convert all remaining LLVM instructions to MLIR operations.
  return convertInstruction(inst);
}

LogicalResult ModuleImport::processFunction(llvm::Function *func) {
  clearRegionState();

  auto functionType =
      dyn_cast<LLVMFunctionType>(convertType(func->getFunctionType()));
  if (func->isIntrinsic() &&
      iface.isConvertibleIntrinsic(func->getIntrinsicID()))
    return success();

  bool dsoLocal = func->isDSOLocal();
  CConv cconv = convertCConvFromLLVM(func->getCallingConv());

  // Insert the function at the end of the module.
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(mlirModule.getBody());

  Location loc = debugImporter->translateFuncLocation(func);
  LLVMFuncOp funcOp = LLVMFuncOp::create(
      builder, loc, func->getName(), functionType,
      convertLinkageFromLLVM(func->getLinkage()), dsoLocal, cconv);

  convertArgAndResultAttrs(func, funcOp);

  if (FlatSymbolRefAttr personality = getPersonalityAsAttr(func))
    funcOp.setPersonalityAttr(personality);
  else if (func->hasPersonalityFn())
    emitWarning(funcOp.getLoc(), "could not deduce personality, skipping it");

  if (func->hasGC())
    funcOp.setGarbageCollector(StringRef(func->getGC()));

  if (func->hasAtLeastLocalUnnamedAddr())
    funcOp.setUnnamedAddr(convertUnnamedAddrFromLLVM(func->getUnnamedAddr()));

  if (func->hasSection())
    funcOp.setSection(StringRef(func->getSection()));

  funcOp.setVisibility_(convertVisibilityFromLLVM(func->getVisibility()));

  if (func->hasComdat())
    funcOp.setComdatAttr(comdatMapping.lookup(func->getComdat()));

  if (llvm::MaybeAlign maybeAlign = func->getAlign())
    funcOp.setAlignment(maybeAlign->value());

  // Handle Function attributes.
  processFunctionAttributes(func, funcOp);

  // Convert non-debug metadata by using the dialect interface. Metadata without
  // a kind-specific conversion is preserved in the generic function metadata
  // carrier.
  SmallVector<std::pair<unsigned, llvm::MDNode *>> allMetadata;
  func->getAllMetadata(allMetadata);
  SmallVector<StringRef> metadataNames;
  llvmModule->getMDKindNames(metadataNames);
  SmallVector<Attribute> functionMetadata;
  for (auto &[kind, node] : allMetadata) {
    if (kind == llvm::LLVMContext::MD_dbg)
      continue;

    llvm::MDNode *metadataNode = node;
    auto emitUnhandledFunctionMetadataWarning = [&]() {
      emitWarning(funcOp.getLoc())
          << "unhandled function metadata: "
          << diagMD(metadataNode, llvmModule.get()) << " on " << diag(*func);
    };

    if (iface.isConvertibleMetadata(kind)) {
      if (succeeded(iface.setMetadataAttrs(builder, kind, metadataNode, funcOp,
                                           *this)))
        continue;
      emitUnhandledFunctionMetadataWarning();
      continue;
    }

    Attribute nodeAttr = convertMetadataToAttr(metadataNode);
    auto mdNodeAttr = dyn_cast_if_present<LLVM::MDNodeAttr>(nodeAttr);
    if (!mdNodeAttr || kind >= metadataNames.size()) {
      emitUnhandledFunctionMetadataWarning();
      continue;
    }

    functionMetadata.push_back(LLVM::FunctionMetadataAttr::get(
        context, builder.getStringAttr(metadataNames[kind]), mdNodeAttr));
  }
  if (!functionMetadata.empty())
    funcOp.setFunctionMetadataAttr(builder.getArrayAttr(functionMetadata));

  if (func->isDeclaration())
    return success();

  // Collect the set of basic blocks reachable from the function's entry block.
  // This step is crucial as LLVM IR can contain unreachable blocks that
  // self-dominate. As a result, an operation might utilize a variable it
  // defines, which the import does not support. Given that MLIR lacks block
  // label support, we can safely remove unreachable blocks, as there are no
  // indirect branch instructions that could potentially target these blocks.
  llvm::df_iterator_default_set<llvm::BasicBlock *> reachable;
  for (llvm::BasicBlock *basicBlock : llvm::depth_first_ext(func, reachable))
    (void)basicBlock;

  // Eagerly create all reachable blocks.
  SmallVector<llvm::BasicBlock *> reachableBasicBlocks;
  for (llvm::BasicBlock &basicBlock : *func) {
    // Skip unreachable blocks.
    if (!reachable.contains(&basicBlock)) {
      if (basicBlock.hasAddressTaken())
        return emitError(funcOp.getLoc())
               << "unreachable block '" << basicBlock.getName()
               << "' with address taken";
      continue;
    }
    Region &body = funcOp.getBody();
    Block *block = builder.createBlock(&body, body.end());
    mapBlock(&basicBlock, block);
    reachableBasicBlocks.push_back(&basicBlock);
  }

  // Add function arguments to the entry block.
  for (const auto &it : llvm::enumerate(func->args())) {
    BlockArgument blockArg = funcOp.getFunctionBody().addArgument(
        functionType.getParamType(it.index()), funcOp.getLoc());
    mapValue(&it.value(), blockArg);
  }

  // Process the blocks in topological order. The ordered traversal ensures
  // operands defined in a dominating block have a valid mapping to an MLIR
  // value once a block is translated.
  SetVector<llvm::BasicBlock *> blocks =
      getTopologicallySortedBlocks(reachableBasicBlocks);
  setConstantInsertionPointToStart(lookupBlock(blocks.front()));
  for (llvm::BasicBlock *basicBlock : blocks)
    if (failed(processBasicBlock(basicBlock, lookupBlock(basicBlock))))
      return failure();

  // Process the debug intrinsics that require a delayed conversion after
  // everything else was converted.
  if (failed(processDebugIntrinsics()))
    return failure();

  // Process the debug records that require a delayed conversion after
  // everything else was converted.
  if (failed(processDebugRecords()))
    return failure();

  return success();
}

/// Checks if `dbgIntr` is a kill location that holds metadata instead of an SSA
/// value.
static bool isMetadataKillLocation(llvm::DbgVariableIntrinsic *dbgIntr) {
  if (!dbgIntr->isKillLocation())
    return false;
  llvm::Value *value = dbgIntr->getArgOperand(0);
  auto *nodeAsVal = dyn_cast<llvm::MetadataAsValue>(value);
  if (!nodeAsVal)
    return false;
  return !isa<llvm::ValueAsMetadata>(nodeAsVal->getMetadata());
}

/// Ensure that the debug intrinsic is inserted right after the operand
/// definition. Otherwise, the operand might not necessarily dominate the
/// intrinsic. If the defining operation is a terminator, insert the intrinsic
/// into a dominated block.
static LogicalResult setDebugIntrinsicBuilderInsertionPoint(
    mlir::OpBuilder &builder, DominanceInfo &domInfo, Value argOperand) {
  if (Operation *op = argOperand.getDefiningOp();
      op && op->hasTrait<OpTrait::IsTerminator>()) {
    // Find a dominated block that can hold the debug intrinsic.
    auto dominatedBlocks = domInfo.getNode(op->getBlock())->children();
    // If no block is dominated by the terminator, this intrinisc cannot be
    // converted.
    if (dominatedBlocks.empty())
      return failure();
    // Set insertion point before the terminator, to avoid inserting something
    // before landingpads.
    Block *dominatedBlock = (*dominatedBlocks.begin())->getBlock();
    builder.setInsertionPoint(dominatedBlock->getTerminator());
  } else {
    Value insertPt = argOperand;
    if (auto blockArg = dyn_cast<BlockArgument>(argOperand)) {
      // The value might be coming from a phi node and is now a block argument,
      // which means the insertion point is set to the start of the block. If
      // this block is a target destination of an invoke, the insertion point
      // must happen after the landing pad operation.
      Block *insertionBlock = argOperand.getParentBlock();
      if (!insertionBlock->empty() &&
          isa<LandingpadOp>(insertionBlock->front()))
        insertPt = cast<LandingpadOp>(insertionBlock->front()).getRes();
    }

    builder.setInsertionPointAfterValue(insertPt);
  }
  return success();
}

std::tuple<DILocalVariableAttr, DIExpressionAttr, Value>
ModuleImport::processDebugOpArgumentsAndInsertionPt(
    Location loc,
    llvm::function_ref<FailureOr<Value>()> convertArgOperandToValue,
    llvm::Value *address,
    llvm::PointerUnion<llvm::Value *, llvm::DILocalVariable *> variable,
    llvm::DIExpression *expression, DominanceInfo &domInfo) {
  // Drop debug intrinsics if the associated debug information cannot be
  // translated due to an unsupported construct.
  DILocalVariableAttr localVarAttr = matchLocalVariableAttr(variable);
  if (!localVarAttr)
    return {};
  FailureOr<Value> argOperand = convertArgOperandToValue();
  if (failed(argOperand)) {
    emitError(loc) << "failed to convert a debug operand: " << diag(*address);
    return {};
  }

  if (setDebugIntrinsicBuilderInsertionPoint(builder, domInfo, *argOperand)
          .failed())
    return {};

  return {localVarAttr, debugImporter->translateExpression(expression),
          *argOperand};
}

LogicalResult
ModuleImport::processDebugIntrinsic(llvm::DbgVariableIntrinsic *dbgIntr,
                                    DominanceInfo &domInfo) {
  Location loc = translateLoc(dbgIntr->getDebugLoc());
  auto emitUnsupportedWarning = [&]() {
    if (emitExpensiveWarnings)
      emitWarning(loc) << "dropped intrinsic: " << diag(*dbgIntr);
    return success();
  };

  OpBuilder::InsertionGuard guard(builder);
  auto convertArgOperandToValue = [&]() {
    return convertMetadataValue(dbgIntr->getArgOperand(0));
  };

  // Drop debug intrinsics with an argument list.
  // TODO: Support this case.
  if (dbgIntr->hasArgList())
    return emitUnsupportedWarning();

  // Drop debug intrinsics with kill locations that have metadata nodes as
  // location operand, which cannot be converted to poison as the type cannot be
  // reconstructed.
  // TODO: Support this case.
  if (isMetadataKillLocation(dbgIntr))
    return emitUnsupportedWarning();

  auto [localVariableAttr, locationExprAttr, locVal] =
      processDebugOpArgumentsAndInsertionPt(
          loc, convertArgOperandToValue, dbgIntr->getArgOperand(0),
          dbgIntr->getArgOperand(1), dbgIntr->getExpression(), domInfo);

  if (!localVariableAttr)
    return emitUnsupportedWarning();

  if (!locVal) // Expected if localVariableAttr is present.
    return failure();

  Operation *op = nullptr;
  if (isa<llvm::DbgDeclareInst>(dbgIntr))
    op = LLVM::DbgDeclareOp::create(builder, loc, locVal, localVariableAttr,
                                    locationExprAttr);
  else if (isa<llvm::DbgValueInst>(dbgIntr))
    op = LLVM::DbgValueOp::create(builder, loc, locVal, localVariableAttr,
                                  locationExprAttr);
  else
    return emitUnsupportedWarning();

  mapNoResultOp(dbgIntr, op);
  setNonDebugMetadataAttrs(dbgIntr, op);
  return success();
}

LogicalResult
ModuleImport::processDebugRecord(llvm::DbgVariableRecord &dbgRecord,
                                 DominanceInfo &domInfo) {
  OpBuilder::InsertionGuard guard(builder);
  Location loc = translateLoc(dbgRecord.getDebugLoc());
  auto emitUnsupportedWarning = [&]() -> LogicalResult {
    if (!emitExpensiveWarnings)
      return success();
    std::string options;
    llvm::raw_string_ostream optionsStream(options);
    dbgRecord.print(optionsStream);
    emitWarning(loc) << "unhandled debug variable record "
                     << optionsStream.str();
    return success();
  };

  // Drop debug records with an argument list.
  // TODO: Support this case.
  if (dbgRecord.hasArgList())
    return emitUnsupportedWarning();

  // Drop all other debug records with a address operand that cannot be
  // converted to an SSA value such as an empty metadata node.
  // TODO: Support this case.
  if (!dbgRecord.getAddress())
    return emitUnsupportedWarning();

  auto convertArgOperandToValue = [&]() -> FailureOr<Value> {
    llvm::Value *value = dbgRecord.getAddress();

    // Return the mapped value if it has been converted before.
    auto it = valueMapping.find(value);
    if (it != valueMapping.end())
      return it->getSecond();

    // Convert constants such as immediate values that have no mapping yet.
    if (auto *constant = dyn_cast<llvm::Constant>(value))
      return convertConstantExpr(constant);
    return failure();
  };

  auto [localVariableAttr, locationExprAttr, locVal] =
      processDebugOpArgumentsAndInsertionPt(
          loc, convertArgOperandToValue, dbgRecord.getAddress(),
          dbgRecord.getVariable(), dbgRecord.getExpression(), domInfo);

  if (!localVariableAttr)
    return emitUnsupportedWarning();

  if (!locVal) // Expected if localVariableAttr is present.
    return failure();

  if (dbgRecord.isDbgDeclare())
    LLVM::DbgDeclareOp::create(builder, loc, locVal, localVariableAttr,
                               locationExprAttr);
  else if (dbgRecord.isDbgValue())
    LLVM::DbgValueOp::create(builder, loc, locVal, localVariableAttr,
                             locationExprAttr);
  else // isDbgAssign
    return emitUnsupportedWarning();

  return success();
}

LogicalResult ModuleImport::processDebugIntrinsics() {
  DominanceInfo domInfo;
  for (llvm::Instruction *inst : debugIntrinsics) {
    auto *intrCall = cast<llvm::DbgVariableIntrinsic>(inst);
    if (failed(processDebugIntrinsic(intrCall, domInfo)))
      return failure();
  }
  return success();
}

LogicalResult ModuleImport::processDebugRecords() {
  DominanceInfo domInfo;
  for (llvm::DbgVariableRecord *dbgRecord : dbgRecords)
    if (failed(processDebugRecord(*dbgRecord, domInfo)))
      return failure();
  dbgRecords.clear();
  return success();
}

LogicalResult ModuleImport::processBasicBlock(llvm::BasicBlock *bb,
                                              Block *block) {
  builder.setInsertionPointToStart(block);
  for (llvm::Instruction &inst : *bb) {
    if (failed(processInstruction(&inst)))
      return failure();

    // Skip additional processing when the instructions is a debug intrinsics
    // that was not yet converted.
    if (debugIntrinsics.contains(&inst))
      continue;

    // Set the non-debug metadata attributes on the imported operation and emit
    // a warning if an instruction other than a phi instruction is dropped
    // during the import.
    if (Operation *op = lookupOperation(&inst)) {
      setNonDebugMetadataAttrs(&inst, op);
    } else if (inst.getOpcode() != llvm::Instruction::PHI) {
      if (emitExpensiveWarnings) {
        Location loc = debugImporter->translateLoc(inst.getDebugLoc());
        emitWarning(loc) << "dropped instruction: " << diag(inst);
      }
    }
  }

  if (bb->hasAddressTaken()) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(block);
    BlockTagOp::create(builder, block->getParentOp()->getLoc(),
                       BlockTagAttr::get(context, bb->getNumber()));
  }
  return success();
}

OwningOpRef<ModuleOp> mlir::translateLLVMIRToModule(
    std::unique_ptr<llvm::Module> llvmModule, MLIRContext *context,
    bool emitExpensiveWarnings, bool dropDICompositeTypeElements,
    bool loadAllDialects, bool preferUnregisteredIntrinsics,
    bool importStructsAsLiterals) {
  // Preload all registered dialects to allow the import to iterate the
  // registered LLVMImportDialectInterface implementations and query the
  // supported LLVM IR constructs before starting the translation. Assumes the
  // LLVM and DLTI dialects that convert the core LLVM IR constructs have been
  // registered before.
  assert(llvm::is_contained(context->getAvailableDialects(),
                            LLVMDialect::getDialectNamespace()));
  assert(llvm::is_contained(context->getAvailableDialects(),
                            DLTIDialect::getDialectNamespace()));
  if (loadAllDialects)
    context->loadAllAvailableDialects();
  OwningOpRef<ModuleOp> module(ModuleOp::create(FileLineColLoc::get(
      StringAttr::get(context, llvmModule->getSourceFileName()), /*line=*/0,
      /*column=*/0)));

  ModuleImport moduleImport(module.get(), std::move(llvmModule),
                            emitExpensiveWarnings, dropDICompositeTypeElements,
                            preferUnregisteredIntrinsics,
                            importStructsAsLiterals);
  if (failed(moduleImport.initializeImportInterface()))
    return {};
  if (failed(moduleImport.convertDataLayout()))
    return {};
  if (failed(moduleImport.convertComdats()))
    return {};
  if (failed(moduleImport.convertMetadata()))
    return {};
  if (failed(moduleImport.convertGlobals()))
    return {};
  if (failed(moduleImport.convertFunctions()))
    return {};
  if (failed(moduleImport.convertAliases()))
    return {};
  if (failed(moduleImport.convertIFuncs()))
    return {};
  moduleImport.convertTargetTriple();
  moduleImport.convertModuleLevelAsm();
  return module;
}
