#include <cstdlib>
#include <sstream>
#include <fstream>
#include <queue>
#include <map>

#include "llvm/ADT/BreadthFirstIterator.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Support/Format.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"

using namespace llvm;

#define DEBUG_TYPE "dsa-peephole"

struct StreamSpecialize : public FunctionPass {

  StreamSpecialize() : FunctionPass(ID) {}

  // {
  bool runOnFunction(Function &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  // }


};

bool StreamSpecialize::runOnFunction(Function &F) {
  return false;
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
  AU.addPreserved<LoopInfoWrapperPass>();
  AU.addPreserved<DependenceAnalysisWrapperPass>();
}

char StreamSpecialize::ID = 0;

static RegisterPass<StreamSpecialize> X("dsa-peephole", "Peephole optimization of dsa program...");
