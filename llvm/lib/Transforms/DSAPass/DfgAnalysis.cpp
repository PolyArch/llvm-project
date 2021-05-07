#include "dsa/debug.h"

#include "./DfgAnalysis.h"

namespace dsa {
namespace analysis {

std::vector<std::pair<IntrinsicInst*, IntrinsicInst*>>
GatherConfigScope(Function &F) {
  std::vector<std::pair<IntrinsicInst*, IntrinsicInst*>> Res;
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto End = dyn_cast<IntrinsicInst>(&I)) {
        if (End->getIntrinsicID() == Intrinsic::ss_config_end) {
          auto Start = dyn_cast<IntrinsicInst>(End->getOperand(0));
          CHECK(Start && Start->getIntrinsicID() == Intrinsic::ss_config_start);
          Res.emplace_back(Start, End);
        }
      }
    }
  }
  return Res;
}

void ExtractDFGFromScope(DFGFile &DF,
                         IntrinsicInst *Start, IntrinsicInst *End,
                         DominatorTree *DT, LoopInfo *LI) {
  std::set<BasicBlock*> OutOfBound;
  for (auto BBN : breadth_first(DT->getNode(End->getParent()))) {
    if (BBN->getBlock() == End->getParent())
      continue;
    OutOfBound.insert(BBN->getBlock());
  }
  for (auto NestedLoop : *LI) {
    for (auto SubLoop : breadth_first(NestedLoop)) {
      if (!DT->dominates(Start, SubLoop->getBlocks()[0]))
        continue;
      if (!SubLoop->getLoopID() || SubLoop->getLoopID()->getNumOperands() == 0)
        continue;
      bool InBound = true;
      for (auto BB : SubLoop->getBlocks()) {
        if (OutOfBound.count(BB))
          InBound = false;
      }
      if (!InBound)
        continue;
      if (MDNode *MD = GetUnrollMetadata(SubLoop->getLoopID(), "llvm.loop.ss.dedicated")) {
        auto MDFactor = dyn_cast<ConstantAsMetadata>(MD->getOperand(1));
        int Factor =(int) MDFactor->getValue()->getUniqueInteger().getSExtValue();
        auto DD = new DedicatedDfg(&DF, SubLoop, Factor);
        DF.addDfg(DD);
      }
    }
  }

  for (auto BBN : breadth_first(DT->getNode(Start->getParent()))) {
    auto BB = BBN->getBlock();

    auto range = make_range(BB != Start->getParent() ? BB->begin() : Start->getIterator(),
                            BB != End->getParent() ? BB->end() : End->getIterator());
    for (auto &I : range) {
      auto TemporalStart = dyn_cast<IntrinsicInst>(&I);
      if (!TemporalStart || TemporalStart->getIntrinsicID() != Intrinsic::ss_temporal_region_start)
        continue;
      // TODO: Maybe later we need control flow in the temporal regions
      bool Found = false;
      for (Instruction *Cur =TemporalStart; Cur != &Cur->getParent()->back();
        Cur = Cur->getNextNode()) {
        auto TemporalEnd = dyn_cast<IntrinsicInst>(Cur);
        if (!TemporalEnd || TemporalEnd->getIntrinsicID() != Intrinsic::ss_temporal_region_end)
          continue;
        assert(TemporalEnd->getOperand(0) == TemporalStart);
        DF.addDfg(new TemporalDfg(&DF, TemporalStart, TemporalEnd));
        Found = true;
      }
      assert(Found);
    }

    if (BB == End->getParent()) {
      break;
    }
  }

}

}
}