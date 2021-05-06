#include "llvm_common.h"

#include <cstdlib>
#include <sstream>
#include <fstream>
#include <queue>
#include <map>

#include "llvm/ADT/BreadthFirstIterator.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/Format.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"

#include "CodeXform.h"
#include "DfgAnalysis.h"
#include "DfgEntry.h"
#include "Transformation.h"
#include "Util.h"
#include "Pass.h"
#include "dsa/rf.h"

using namespace llvm;

#define DEBUG_TYPE "stream-specialize"


bool StreamSpecialize::runOnModule(Module &M) {

  for (auto &F : M) {
    if (F.isDeclaration()) {
      continue;
    }
    ACT = &getAnalysis<AssumptionCacheTracker>(F).getAssumptionCache(F);
    LI = &getAnalysis<LoopInfoWrapperPass>(F).getLoopInfo();
    SE = &getAnalysis<ScalarEvolutionWrapperPass>(F).getSE();
    DT = &getAnalysis<DominatorTreeWrapperPass>(F).getDomTree();
    AAR = &getAnalysis<AAResultsWrapperPass>(F).getAAResults();
    BFI = &getAnalysis<BlockFrequencyInfoWrapperPass>(F).getBFI();
    MSSA = &getAnalysis<MemorySSAWrapperPass>(F).getMSSA();
    runOnFunction(F);
  }

  // Clean up the code generated.
  llvm::PassManagerBuilder PMB;
  PMB.OptLevel = 3;
  llvm::legacy::FunctionPassManager fpass(&M);
  llvm::legacy::PassManager mpass;
  PMB.populateFunctionPassManager(fpass);
  PMB.populateModulePassManager(mpass);
  PMB.Inliner = llvm::createFunctionInliningPass(3, 0, false);
  for (auto &F : M) {
    runOnFunction(F);
  }
  fpass.doFinalization();
  mpass.run(M);

  return false;
}

bool StreamSpecialize::runOnFunction(Function &F) {

  if (!MF.TEMPORAL) {
    dsa::xform::EliminateTemporal(F);
  }

  LLVM_DEBUG(dbgs() << "Working on " << F.getName() << "\n");

  IRBuilder<> IB(F.getContext());
  IBPtr = &IB;

  {
    int Cnt = 0;
    auto ScopePairs = dsa::analysis::GatherConfigScope(F);
    if (!ScopePairs.empty()) {
      DSARegs = dsa::xform::InjectDSARegisterFile(F);
    }
    for (auto &Pair : ScopePairs) {
      std::ostringstream OSS;
      OSS << F.getName().bytes_begin() << "_dfg_" << Cnt++ << ".dfg";
      DfgFile DFG(OSS.str(), F, Pair.first, Pair.second, this);
      DFG.InspectSPADs();
      DFG.InitAllDfgs();
      DFG.EmitAndScheduleDfg();
      DFG.InjectStreamIntrinsics();
      if (MF.EXTRACT) {
        DFG.EmitAndScheduleDfg();
        continue;
      }
      DFG.EraseOffloadedInstructions();
    }
    for (auto &Pair : ScopePairs) {
      Pair.second->eraseFromParent();
      Pair.first->eraseFromParent();
    }
  }

  HandleMemIntrin(F);

  LLVM_DEBUG(
    llvm::errs() << "After transformation:\n";
    F.dump()
  );

  return false;
}

void StreamSpecialize::HandleMemIntrin(Function &F) {
  DfgFile Dummy("dummy", F, nullptr, nullptr, this);
  std::vector<IntrinsicInst*> ToInject;

  for (auto &BB : F) {
    for (auto &I : BB) {
      auto Intrin = dyn_cast<IntrinsicInst>(&I);
      if (!Intrin)
        continue;
      if (Intrin->getIntrinsicID() == Intrinsic::memcpy) {
        ToInject.push_back(Intrin);
      }
    }
  }

  for (auto Intrin : ToInject) {
    Value *Dst = Intrin->getOperand(0);
    Value *Src = Intrin->getOperand(1);
    Value *Size = Intrin->getOperand(2);

    AnalyzedStream AS0;
    AnalyzedStream AS1;

    AS0.Dimensions.emplace_back(Dst, Size, 0);
    AS1.Dimensions.emplace_back(Src, Size, 0);

    IntrinDfg Dfg(Dummy, Intrin);
    IBPtr->SetInsertPoint(Dfg.DefaultIP());
    dsa::inject::InjectLinearStream(IBPtr, DSARegs, MemScrPort, AS1, DMO_Read, DP_NoPadding, DMT_DMA, 1);
    dsa::inject::InjectLinearStream(IBPtr, DSARegs, MemScrPort, AS1, DMO_Read, DP_NoPadding, DMT_SPAD, 1);
    dsa::inject::DSAIntrinsicEmitter(IBPtr, DSARegs).SS_WAIT(~0ull);
    Intrin->eraseFromParent();
  }

}

void StreamSpecialize::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<AssumptionCacheTracker>();
  AU.addRequired<BlockFrequencyInfoWrapperPass>();
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addRequired<DependenceAnalysisWrapperPass>();
  AU.addRequired<LoopInfoWrapperPass>();
  AU.addRequired<ScalarEvolutionWrapperPass>();
  AU.addRequired<TargetTransformInfoWrapperPass>();
  AU.addRequired<AAResultsWrapperPass>();
  AU.addRequired<LoopAccessLegacyAnalysis>();
  AU.addRequired<OptimizationRemarkEmitterWrapperPass>();
  AU.addRequired<TargetLibraryInfoWrapperPass>();
  AU.addRequired<MemorySSAWrapperPass>();
}

char StreamSpecialize::ID = 0;

static RegisterPass<StreamSpecialize> X("stream-specialize", "Decoupld-Spatial Specialized Transformation");
