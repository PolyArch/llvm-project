#include "dsa/debug.h"

#include "llvm/ADT/BreadthFirstIterator.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"
#include <sstream>

#include "StreamAnalysis.h"

#define DEBUG_TYPE "stream-analysis"

namespace dsa {
namespace analysis {

LinearInfo LinearInfo::LoopInvariant(ScalarEvolution *SE, int N, const SCEV *Base) {
  LinearInfo Res;
  Res.Base = Base;
  return Res;
}

const uint64_t *LinearInfo::ConstInt() const {
  if (!Invariant()) {
    return nullptr;
  }
  const auto *SC = dyn_cast<SCEVConstant>(Invariant());
  if (!SC) {
    return nullptr;
  }
  return SC->getAPInt().getRawData();
}

const SCEV *LinearInfo::Invariant() const {
  return Coef.empty() ? Base : nullptr;
}

std::string LinearInfo::toString(const std::vector<Loop *> &Loops) {
  std::string S;
  llvm::raw_string_ostream RSO(S);
  if (Coef.empty()) {
    RSO << *Base;
  } else {
    CHECK(Loops.empty() || Loops.size() == Coef.size())
        << Loops.size() << " ? " << Coef.size();
    RSO << "(";
    for (int i = 0, N = Coef.size(); i < N; ++i) { // NOLINT
      if (!Loops.empty()) {
        if (auto *IndVar = Loops[i]->getCanonicalInductionVariable()) {
          if (IndVar->hasName()) {
            RSO << IndVar->getName();
          } else {
            ModuleSlotTracker MST(
                IndVar->getParent()->getParent()->getParent());
            MST.incorporateFunction(*IndVar->getParent()->getParent());
            RSO << "%" << MST.getLocalSlot(IndVar);
          }
        } else {
          RSO << "i" << i;
        }
      } else {
        RSO << "i" << i;
      }
      RSO << " * " << Coef[i].toString(Loops) << " + ";
    }
    RSO << *Base << ")";
  }
  return RSO.str();
}

int LinearInfo::PatialInvariant() const {
  for (int i = 0, N = Coef.size(); i < N; ++i) { // NOLINT
    const auto *CI = Coef[i].ConstInt();
    if (CI && *CI == 0) {
      continue;
    }
    return i;
  }
  return Coef.size();
}

LinearInfo AnalyzeIndexExpr(ScalarEvolution *SE, const SCEV *Raw,
                            const std::vector<Loop *> &Loops) {
  LinearInfo Res;
  int i = 0; // NOLINT
  auto AppendZero = [SE, &Res, &Loops]() {
    auto Coef = LinearInfo::LoopInvariant(SE, Loops.size(), SE->getConstant(APInt(64, 0)));
    Res.Coef.push_back(Coef);
  };
  for (int N = Loops.size(); i < N; ++i) {
    if (!SE->isLoopInvariant(Raw, Loops[i])) {
      break;
    }
    AppendZero();
  }
  if (!isa<SCEVAddRecExpr>(Raw) || i == (int)Loops.size()) {
    CHECK(i == (int)Loops.size()) << i << " != " << Loops.size() << ": " << *Raw;
    Res.Coef.clear();
    Res.Coef.shrink_to_fit();
    Res.Base = Raw;
    return Res;
  }
  auto *Cur = Raw;
  for (int N = Loops.size(); i < N; ++i) {
    if (const auto *SARE = dyn_cast<SCEVAddRecExpr>(Cur)) {
      if (!Loops.back()->contains(SARE->getLoop())) {
        AppendZero();
        continue;
      }
      if (SARE->getLoop() != Loops[i]) {
        CHECK(SE->isLoopInvariant(Cur, Loops[i]))
          << "\n" << *Raw << "\n" << *Cur << "\nNot an invariant in loop " << *Loops[i]
          << "\nCurrent Loop: " << *SARE->getLoop()
          << "\nOuter Most: " << *Loops.back();
        auto Coef = LinearInfo::LoopInvariant(SE, Loops.size(), SE->getConstant(APInt(64, 0)));
        Res.Coef.push_back(Coef);
        continue;
      }
      Res.Coef.push_back(AnalyzeIndexExpr(SE, SARE->getStepRecurrence(*SE), Loops));
      Cur = SARE->getStart();
    } else {
        CHECK(SE->isLoopInvariant(Cur, Loops[i]))
          << "\n" << *Raw << "\n" << *Cur << "\nNot an invariant in loop " << *Loops[i];
      AppendZero();
    }
  }
  CHECK(Res.Coef.size() == Loops.size()) << Res.Coef.size() << " != " << Loops.size();
  Res.Base = Cur;
  return Res;
}

namespace test {

/*!
 * \brief Expose these functions in the .so for the purpose of test.
 */
struct StreamAnalysisPass : public FunctionPass {
  ScalarEvolution *SE{nullptr};
  LoopInfo *LI{nullptr};
  DependenceInfo *DI{nullptr};
  std::vector<Loop *> Loops;

  void AnalyzeLoopNest(const std::vector<Loop *> &Loops) { // NOLINT
    CHECK(!Loops.empty());
    auto *L = Loops.front();
    for (auto &BB : L->getBlocks()) {
      for (auto &I : *BB) {
        if (auto *Load = dyn_cast<LoadInst>(&I)) {
          auto *Index = Load->getPointerOperand();
          CHECK(SE->isSCEVable(Index->getType()));
          auto LI = AnalyzeIndexExpr(SE, SE->getSCEV(Index), Loops);
          LOG(LOOP) << LI.toString();
        }
      }
    }
    for (auto *L : Loops) {
      auto LI = AnalyzeIndexExpr(SE, SE->getBackedgeTakenCount(L), Loops);
      LOG(LOOP) << LI.toString();
    }
  }

  void Dfs(Loop *L) { // NOLINT
    Loops.insert(Loops.begin(), L);
    if (L->empty()) {
      if (GetUnrollMetadata(L->getLoopID(), "llvm.loop.ss.dedicated")) {
        LOG(LOOP) << "Offload: " << *L;
        auto I = Loops.begin();
        decltype(Loops) Slice;
        while (I < Loops.end()) {
          auto *IndVar = (*I)->getCanonicalInductionVariable();
          std::string S;
          llvm::raw_string_ostream OSS(S);
          if (IndVar) {
            OSS << *IndVar;
          } else if ((*I)->getInductionVariable(*SE)) {
            OSS << *(*I)->getInductionVariable(*SE);
          } else {
            OSS << "Non-Canonical";
          }
          LOG(LOOP) << "Loop: " << *(*I) << " " << S;
          if (GetUnrollMetadata((*I)->getLoopID(), "llvm.loop.ss.stream")) {
            Slice = decltype(Loops)(Loops.begin(), I + 1);
            break;
          }
          ++I;
        }
        AnalyzeLoopNest(Slice);
      }
    } else {
      for (auto *SubLoop : *L) {
        Dfs(SubLoop);
      }
    }
    Loops.erase(Loops.begin());
  }

  bool runOnFunction(Function &F) override {
    SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();
    LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    DI = &getAnalysis<DependenceAnalysisWrapperPass>().getDI();
    for (auto *L : *LI) {
      Dfs(L);
    }
    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<BlockFrequencyInfoWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<DependenceAnalysisWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addRequired<AAResultsWrapperPass>();
    // AU.addRequired<DemandedBitsWrapperPass>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addRequired<MemorySSAWrapperPass>();
    AU.addPreserved<LoopInfoWrapperPass>();
    // AU.addPreserved<BasicAAWrapperPass>();
    // AU.addPreserved<GlobalsAAWrapperPass>();
    AU.addPreserved<DependenceAnalysisWrapperPass>();
  }

  StreamAnalysisPass() : FunctionPass(ID) {}

  static char ID;
};

char StreamAnalysisPass::ID = 0;

static RegisterPass<StreamAnalysisPass>
    Y("analyze-stream", "Invoke stream analysis from streams...");

} // namespace test

} // namespace analysis
} // namespace dsa
