#pragma once

#include "./llvm_common.h"

#include "./Transformation.h"

namespace dsa {
namespace analysis {

/*!
 * \brief Gather all the pairs of dsa.config.start/end
 * \param F The function to be analyzed.
 */
std::vector<std::pair<IntrinsicInst*, IntrinsicInst*>>
GatherConfigScope(Function &F);

/*!
 * \brief Extract dataflow graphs from the given scope
 * \param Start The start instrinsic annotates the scope
 * \param End The end intrinsic annotates the scope
 * \param DT The dominator tree information for analysis
 * \param LI The loop information for analysis
 */
void ExtractDFGFromScope(DFGFile &DF,
                         IntrinsicInst *Start, IntrinsicInst *End,
                         DominatorTree *DT, LoopInfo *LI);

}
}