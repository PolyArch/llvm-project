#pragma once

#include "./llvm_common.h"

#include "./Transformation.h"

namespace dsa {

namespace xform {

struct CodeGenContext;

} // namespace xform

namespace analysis {

/*!
 * \brief Information of the bitstream.
 */
struct ConfigInfo {
  std::string Name;
  std::string Bitstream;
  int Size;
  ConfigInfo(const std::string &N, const std::string &B, int S)
      : Name(N), Bitstream(B), Size(S) {}
  ConfigInfo() : Name(""), Bitstream(""), Size(0) {}
};

/*! \brief The result of coalesced SLP memory operation. */
struct CoalMemoryInfo {
  /*!
   * \brief Which set of memory coalescing this memory operation belongs to.
   */
  std::vector<int> Belong;

  struct CoalescedEntry {
    /*!
     * \brief The ID of the DFGEntry.
     */
    int ID{-1};
    /*!
     * \brief The offset relative to the first element of the cluster.
     */
    int Offset{INT_MIN};

    CoalescedEntry(int ID) : ID(ID) {}
  };
  /*!
   * \brief Each vector is a cluster of coalesced memory.
   */
  std::vector<std::vector<CoalescedEntry>> Clusters;
  /*!
   * \brief The port number of each coalesced port.
   */
  std::vector<int> ClusterPortNum;
};

/*!
 * \brief The information of the scratchpads.
 */
struct SpadInfo {
  /*! \brief Offset The begin of arrays. */
  std::map<AllocaInst*, int> Offset;

  // [Linear Read Stream, Address Offset, Buffer Size, Load Port, Store Port]
  using BuffetEntry = std::tuple<MemPort*, int, int, int, int>;
  /*! \brief Buffet The information of buffet double buffering. */
  std::vector<BuffetEntry> Buffet;
  /*! \brief The total bytes occupied on SPAD. */
  int Total{0};

  /*!
   * \brief If the given pointer is on scratchpad.
   * \param Val The pointer to be inspected.
   */
  bool isSpad(Value *Val);

  SpadInfo(std::map<AllocaInst*, int> &O, std::vector<BuffetEntry> &B, int T) :
    Offset(O), Buffet(B), Total(T) {}

  SpadInfo() {}
};

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
  std::vector<analysis::LinearInfo> TripCount;

  DFGLoopInfo() {}

  DFGLoopInfo(const std::vector<Loop*> &L, const std::vector<analysis::LinearInfo> &T) :
    LoopNest(L), TripCount(T) {}
};

/*!
 * \brief The information of accumulator.
 */
struct AccumulatorInfo {
  struct Entry {
    /*!
     * \brief The index of the accumulator.
     */
    int AccIdx{-1};
    /*!
     * \brief The index of the input tag.
     */
    int TagIdx{-1};
    /*!
     * \brief The loop level of value reset and produce.
     */
    int ResetLevel{-1};
  };
  /*!
   * \brief The DFG this information belongs to.
   */
  DedicatedDFG *DD{nullptr};
  /*!
   * \brief The information accumulators.
   */
  std::vector<Entry> Entries;
  /*!
   * \brief The ID's of memory port to be zero padding.
   */
  // TODO(@were): Deprecate the consumed by acc in Transformation.cpp; Use this!
  std::vector<int> ZeroPadding;

  /*!
   * \brief Check if the given ID should be tagged.
   * \param ID The given ID.
   */
  bool isTagged(int ID);
};

/*! \brief The result of DFG analysis. */
struct DFGAnalysisResult {
  SpadInfo SI;
  ConfigInfo CI;
  std::vector<std::pair<IntrinsicInst *, IntrinsicInst *>> Scope;
  std::vector<CoalMemoryInfo> CMI;
  std::vector<DFGLoopInfo> DLI;
  std::vector<AccumulatorInfo> ACCI;
};

/*!
 * \brief Gather all the pairs of dsa.config.start/end
 * \param F The function to be analyzed.
 * \param DAR The result of scopes should go. Specifically, the `Scope` attr.
 */
void gatherConfigScope(Function &F, DFGAnalysisResult &DAR);

/*!
 * \brief Gather the loop information of the loop nest that wraps the DFG.
 * \param DF The scope to be analyzed.
 * \param CGC The LLVM utilities passed.
 * \param DAR The result of scopes should go. Specifically, the `DLI` attr.
 */
void analyzeDFGLoops(DFGFile &DF, xform::CodeGenContext &CGC, DFGAnalysisResult &DAR);

/*!
 * \brief Gather the the scratchpad local variables.
 * \param Start The starting scope of DFG.
 * \param Start The end scope of DFG.
 */
void extractSpadFromScope(DFGFile &DF, xform::CodeGenContext &CGC, DFGAnalysisResult &DAR);

/*!
 * \brief Extract dataflow graphs from the given scope
 * \param DF The DFG scope to be analyzed.
 * \param CGC The LLVM utilities passed.
 */
void extractDFGFromScope(DFGFile &DF, dsa::xform::CodeGenContext &CGC);

/*!
 * \brief Extract the port number from the generated schedule.
 * \param FName The emitted file to be analyzed.
 * \param Ports Write the result in the allocated vector buffer.
 * \param CMIs The coalesced memory.
 */
ConfigInfo extractDFGPorts(std::string FName, DFGFile &DF, std::vector<CoalMemoryInfo> &CMIs,
                           SpadInfo &SI);

/*!
 * \brief Analyze the offset between two memory operation indices.
 * \param SA The first index.
 * \param SB The second index.
 * \param SE Scalar evolution for memory stream analysis.
 * \param DLI The loop information associated to the DFG.
 * \param Signed If result of analysis is signed.
 */
int64_t indexPairOffset(const SCEV *SA, const SCEV *SB, ScalarEvolution &SE,
                        const DFGLoopInfo &DLI, bool Signed);

/*!
 * \brief Analyze the information of accumulator.
 * \param DF The DFG file to be analyzed.
 * \param CGC The context required for the analysis.
 * \param VACCI The result to be written to.
 */
void analyzeAccumulatorTags(DFGFile &DF, xform::CodeGenContext &CGC,
                            DFGAnalysisResult &DAR);

/*!
 * \brief The implementation of coalescing memory operations.
 * @tparam T The DFGEntry type to be gathered, either MemPort or PortMem.
 * \param DB The DFG to be analyzed.
 * \param SE Scalar evolution for memory stream analysis.
 * \param DLI The loop information associated to the DFG.
 * \param DSU The disjoint union to be updated by coalescing.
 */
template<typename T>
void gatherMemoryCoalescingImpl(DFGBase *DB, ScalarEvolution &SE, const DFGLoopInfo& DLI,
                                std::vector<int> &DSU) {
  bool Iterative = true;
  while (Iterative) {
    Iterative = false;
    for (int i = 0, n = DB->Entries.size(); i < n; ++i) { // NOLINT
      if (auto Node1 = dyn_cast<T>(DB->Entries[i])) {
        for (int j = i + 1; j < n; ++j) { // NOLINT
          if (auto Node2 = dyn_cast<T>(DB->Entries[j])) {
            auto PtrA = Node1->underlyingInst()->getOperand(T::PtrOperandIdx);
            auto PtrB = Node2->underlyingInst()->getOperand(T::PtrOperandIdx);
            if (PtrA->getType() != PtrB->getType()) {
              continue;
            }
            auto Offset = indexPairOffset(SE.getSCEV(PtrA), SE.getSCEV(PtrB), SE, DLI, false);
            if (Offset != -1 && Offset <= 64) {
              if (utils::DSUGetSet(i, DSU) != utils::DSUGetSet(j, DSU)) {
                DSU[utils::DSUGetSet(i, DSU)] = utils::DSUGetSet(j, DSU);
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

/*!
 * \brief Gather all the memory that can be SLP-coalesced in the given DFG file.
 *        By SLP, we mean a[i], a[i+1] can be coalesced into one a[2] vector port.
 * \param DF The DFG file to be analyzed.
 * \param SE The LLVM scalar evolution.
 * \param DAR The result of DFG analysis from previous pass.
 */
void gatherMemoryCoalescing(DFGFile &DF, ScalarEvolution &SE, DFGAnalysisResult &DAR);

/*!
 * \brief BFS back to all the instructions within/outside the DFG scope
 *        on which the given instruction depends.
 * \param DB The DFG scope.
 * \param From The given source instruction of BFS.
 * \param DT The dominator tree. Can be null, then domination is not checked.
 * \return A pair of sets of instructions. The instructions within the scope, and the instruction
 *         outside the scope.
 */
std::pair<std::set<Instruction *>, std::set<Instruction *>>
bfsOperands(DFGBase *DB, Instruction *From, DominatorTree *DT);

} // namespace analysis
} // namespace dsa
