#include "dsa/debug.h"

#include "llvm/ADT/BreadthFirstIterator.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"
#include <sstream>

#include "StreamAnalysis.h"

#define DEBUG_TYPE "stream-analysis"

namespace dsa {
namespace analysis {

void IAVisitor::visit(IndexAnalysis*) {}

void IAVisitor::visit(LoopInvariantDim* LID) {
  IAVisitor::visit(static_cast<IndexAnalysis*>(LID));
}

void IAVisitor::visit(LinearCombination* LC) {
  IAVisitor::visit(static_cast<IndexAnalysis*>(LC));
}

  
std::string LoopInvariantDim::toString(const std::vector<Loop *> &Loops) const {
  std::string S;
  llvm::raw_string_ostream RSO(S);
  RSO << *Value;
  return S;
}

ExprComputable::ExprComputable(const SCEV *R) :
  IndexAnalysis(IAKind::kExprComputable), Root(R) {}

const uint64_t *LoopInvariantDim::constInt() const {
  if (auto *SC = dyn_cast<SCEVConstant>(Value)) {
    return SC->getAPInt().getRawData();
  }
  return nullptr;
}

std::string LinearCombination::toString(const std::vector<Loop *> &Loops) const {
  std::string S;
  llvm::raw_string_ostream RSO(S);
  if (!Loops.empty()) {
    CHECK(Loops.size() == Coef.size()) << Loops.size() << " ? " << Coef.size();
  }
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
    RSO << " * " << Coef[i]->toString(Loops) << " + ";
  }
  RSO << Base->toString() << ")";
  return RSO.str();
}

int LinearCombination::partialInvariant() const {
  for (int i = 0, N = Coef.size(); i < N; ++i) { // NOLINT
    if (const auto *LID = dyn_cast<LoopInvariantDim>(Coef[i])) {
      if (const auto *CI = LID->constInt()) {
        if (CI && *CI == 0) {
          continue;
        }
      }
    }
    return i;
  }
  return Coef.size();
}

IndexAnalysis *analyzeIndexExpr(ScalarEvolution *SE, const SCEV *Raw,
                            const std::vector<Loop *> &Loops) {
  LinearCombination Res;
  int i = 0; // NOLINT
  auto AppendZero = [SE, &Res]() {
    Res.Coef.push_back(new LoopInvariantDim(SE->getConstant(APInt(64, 0))));
  };
  for (int N = Loops.size(); i < N; ++i) {
    if (!SE->isLoopInvariant(Raw, Loops[i])) {
      break;
    }
    AppendZero();
  }
  if (!isa<SCEVAddRecExpr>(Raw)) {
    if (i == (int)Loops.size()) {
      return new LoopInvariantDim(Raw);
    }
    CHECK(false) << "Not supported yet! " << *Raw << " "
      << *SE->getAddExpr(Raw, SE->getOne(Raw->getType()));
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
        AppendZero();
        continue;
      }
      Res.Coef.push_back(analyzeIndexExpr(SE, SARE->getStepRecurrence(*SE), Loops));
      Cur = SARE->getStart();
    } else {
      CHECK(SE->isLoopInvariant(Cur, Loops[i]))
        << "\n" << *Raw << "\n" << *Cur << "\nNot an invariant in loop " << *Loops[i];
      AppendZero();
    }
  }
  CHECK(Res.Coef.size() == Loops.size()) << Res.Coef.size() << " != " << Loops.size();
  Res.Base = new LoopInvariantDim(Cur);
  bool AllZero = true;
  for (int i = 0; i < (int) Loops.size(); ++i) { // NOLINT
    DSA_LOG(STREAM) << i << ": " << Res.Coef[i]->toString();
    if (auto *LID = dyn_cast<LoopInvariantDim>(Res.Coef[i])) {
      if (*LID->constInt() == 0) {
        continue;
      }
    }
    AllZero = false;
    DSA_LOG(STREAM) << i << ": Non-zero!";
    break;
  }
  if (AllZero) {
    DSA_LOG(STREAM) << "All zero! loop invariant";
    return Res.Base;
  }
  return new LinearCombination(Res);
}


std::pair<IndexAnalysis*, std::vector<IndexAnalysis*>>
fuseInnerDimensions(IndexAnalysis *IdxLI, std::vector<IndexAnalysis*> LoopLI,
                    int DType, int Unroll, int P, IRBuilder<> *IB, ScalarEvolution &SE) {
  if (isa<LoopInvariantDim>(IdxLI)) {
    return {IdxLI, LoopLI};
  }
  auto ExtractConstInt = [](IndexAnalysis *IA) -> const uint64_t* {
    if (const auto *LID = dyn_cast<LoopInvariantDim>(IA)) {
      return LID->constInt();
    }
    return nullptr;
  };
  auto *LC = dyn_cast<LinearCombination>(IdxLI);
  while (LoopLI.size() > 1) {
    if (auto *S1D = ExtractConstInt(LC->Coef.front())) {
      if (((int) *S1D) == DType) {
        if (const auto *N = ExtractConstInt(LoopLI[0])) {
          if (*N % Unroll) {
            break;
          }
          if (const auto *S2D = ExtractConstInt(LC->Coef[1])) {
            // TODO(@were): Relax this condition.
            if (auto *N2D = dyn_cast<LoopInvariantDim>(LoopLI[1])) {
              if ((*S1D) * (*N) == *S2D) {
                LoopLI.erase(LoopLI.begin());
                LC->Coef.erase(LC->Coef.begin());
                LC->Coef.front() = new LoopInvariantDim(SE.getConstant(IB->getInt64(DType)));
                auto *SCEVN = SE.getConstant(IB->getInt64(*N));
                LoopLI[0] = new LoopInvariantDim(SE.getMulExpr(SCEVN, N2D->Value));
                // No modification, redefine.
                continue;
              }
            }
          }
        }
      }
    }
    break;
  }
  return {IdxLI, LoopLI};
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
          const auto *IA = analyzeIndexExpr(SE, SE->getSCEV(Index), Loops);
          DSA_LOG(LOOP) << IA->toString();
        }
      }
    }
    for (auto *L : Loops) {
      const auto *IA = analyzeIndexExpr(SE, SE->getBackedgeTakenCount(L), Loops);
      DSA_LOG(LOOP) << IA->toString();
    }
  }

  void Dfs(Loop *L) { // NOLINT
    Loops.insert(Loops.begin(), L);
    if (L->empty()) {
      if (GetUnrollMetadata(L->getLoopID(), "llvm.loop.ss.dedicated")) {
        DSA_LOG(LOOP) << "Offload: " << *L;
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
          DSA_LOG(LOOP) << "Loop: " << *(*I) << " " << S;
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
