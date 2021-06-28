#pragma once

#include "./llvm_common.h"

#include "./Transformation.h"

namespace dsa {
namespace analysis {

/*!
 * \brief Gather all the pairs of dsa.config.start/end
 * \param F The function to be analyzed.
 */
std::vector<std::pair<IntrinsicInst *, IntrinsicInst *>>
GatherConfigScope(Function &F);

/*!
 * \brief Extract dataflow graphs from the given scope
 * \param Start The start instrinsic annotates the scope
 * \param End The end intrinsic annotates the scope
 * \param DT The dominator tree information for analysis
 * \param LI The loop information for analysis
 */
void ExtractDFGFromScope(DFGFile &DF, IntrinsicInst *Start, IntrinsicInst *End,
                         DominatorTree *DT, LoopInfo *LI);

/*!
 * \brief Information for confiration injection.
 */
struct ConfigInfo {
  std::string Name;
  std::string Bitstream;
  int Size;
  ConfigInfo(const std::string &N, const std::string &B, int S)
      : Name(N), Bitstream(B), Size(S) {}
};

/*!
 * \brief Extract the port number from the generated schedule.
 * \param FName The emitted file to be analyzed.
 * \param Ports Write the result in the allocated vector buffer.
 */
ConfigInfo ExtractDFGPorts(std::string FName, DFGFile &DF);

/*!
 * \brief Analyze DFG loops for code generation.
 */
struct DFGLoopInfo {
  /*!
   * \brief The loop nest, from inner to outer.
   */
  std::vector<Loop*> LoopNest;
  /*!
   * \brief The trip count of each loop level, corresponding to the loop nest above.
   */
  std::vector<analysis::LinearInfo*> TripCount;

  DFGLoopInfo() {}

  DFGLoopInfo(const std::vector<Loop*> &L, const std::vector<analysis::LinearInfo*> &T) :
    LoopNest(L), TripCount(T) {}
};

/*!
 * \brief Gather the loop information of the loop nest that wraps the DFG.
 * \param DB The DFG to be analyzed.
 * \param SE The scalar evolution for analysis.
 */
DFGLoopInfo AnalyzeDFGLoops(DFGBase *DB, ScalarEvolution &SE);

/*!
 * \brief Disjoint union set.
 * \param Elem 
 * \param DSU 
 */
int DSUGetSet(int Elem, std::vector<int> &DSU);

/*!
 * \brief 
 * \param DB 
 * \param SE 
 */
template<typename T>
void GatherMemoryCoalescing(DFGBase *DB, ScalarEvolution &SE, std::vector<int> &DSU) {
  auto DLI = AnalyzeDFGLoops(DB, SE);
  auto AnalyzeIndexPair = [&SE, &DLI](const SCEV *SA, const SCEV *SB) -> int64_t {
    auto A = AnalyzeIndexExpr(&SE, SA, DLI.LoopNest);
    auto B = AnalyzeIndexExpr(&SE, SB, DLI.LoopNest);
    bool SameDims = true;
    CHECK(A->Coef.size() == B->Coef.size());
    for (int j = 0; j < (int) A->Coef.size(); ++j) {
      if (A->Coef[j]->Base != B->Coef[j]->Base) {
        DSA_INFO << *SE.getEqualPredicate(A->Coef[j]->Base, B->Coef[j]->Base);
        SameDims = false;
      }
    }
    if (SameDims) {
      if (auto Offset = dyn_cast<SCEVConstant>(SE.getAddExpr(A->Base, SE.getNegativeSCEV(B->Base)))) {
        DSA_INFO << *Offset << " " << Offset->getAPInt().getSExtValue();
        return abs(Offset->getAPInt().getSExtValue());
      }
    }
    return -1;
  };
  bool Iterative = true;
  while (Iterative) {
    Iterative = false;
    auto DLI = dsa::analysis::AnalyzeDFGLoops(DB, SE);
    for (int i = 0, n = DB->Entries.size(); i < n; ++i) {
      if (auto Node1 = dyn_cast<T>(DB->Entries[i])) {
        for (int j = i + 1; j < n; ++j) {
          if (auto Node2 = dyn_cast<T>(DB->Entries[j])) {
            auto IdxA = SE.getSCEV(Node1->UnderlyingInst()->getOperand(T::PtrOperandIdx));
            auto IdxB = SE.getSCEV(Node2->UnderlyingInst()->getOperand(T::PtrOperandIdx));
            auto Offset = AnalyzeIndexPair(IdxA, IdxB);
            DSA_INFO << *IdxA << " " << *IdxB << " " << Offset;
            if (Offset != -1 && Offset <= 64) {
              DSA_INFO << i << ", " << DSUGetSet(i, DSU) << "  " << j << ", "<< DSUGetSet(j, DSU);
              if (DSUGetSet(i, DSU) != DSUGetSet(j, DSU)) {
                DSU[DSUGetSet(i, DSU)] = DSUGetSet(j, DSU);
                Iterative = true;
                break;
              }
            }
          }
        }
      }
    }
  }
}
  

} // namespace analysis
} // namespace dsa