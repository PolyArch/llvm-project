#ifndef STREAM_SPECIALIZE_H_
#define STREAM_SPECIALIZE_H_

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/DDG.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"

#include "Util.h"

using namespace llvm;

/*!
 * \brief This is the entry class of the dsa specialized pass.
 */
struct StreamSpecialize : public ModulePass {

  /*
   * \brief These are the LLVM passes required for dsa transformation.
   */
  // {
  AssumptionCache *ACT;
  LoopInfo *LI;
  ScalarEvolution *SE;
  TargetLibraryInfo *TLI;
  DominatorTree *DT;
  AAResults *AAR;
  DDGInfo *DI;
  BlockFrequencyInfo *BFI;
  MemorySSA *MSSA;
  // }

  /*
   * \brief The IR builder for the IR manipulation.
   */
  IRBuilder<> *IBPtr{nullptr};
  /*
   * \brief Data structure for register stickiness optimization.
   */
  dsa::RegisterFile DSARegs;

  static char ID;

  StreamSpecialize() : ModulePass(ID) {}

  bool runOnFunction(Function &F);

  /*!
   * \brief These are the functions implemented for this pass.
   */
  // {
  bool runOnModule(Module &M) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  // }

};

#endif
