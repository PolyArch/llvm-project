#pragma once

#include "Pass.h"
#include "Util.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/Casting.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include <map>
#include <memory>
#include <set>
#include <vector>

namespace dsa {
namespace analysis {

/*!
 * \brief Streams are analyzed by the ScalarEvolution module.
 *        In this pass, we need a helper wrapper to store the decoded SCEV information.
 */
struct SEWrapper {
  // TODO(@were): Let's see, I think I have all the specific site of using this instance.
  //              Thus, it is ok to be a union.
  /*!
   * \brief The DFG entry this SCEV derived.
   */
  union {
    DFGEntry *DE;
    Loop *L;
    SEWrapper *SE;
  } Parent;
  /*!
   * \brief The raw SCEV to be extracted.
   */
  const SCEV *Raw;
  /*!
   * \brief Dump to text for debugging.
   */
  virtual std::string toString();

  virtual ~SEWrapper() {}
};

/*!
 * \brief THis value is a loop invariant under the given loop.
 */
struct LoopInvariant : SEWrapper {
  /*!
   * \brief The trip count of loops above.
   */
  std::vector<SEWrapper*> TripCount;

  /*!
   * \brief If this value is a constant, return this value. Otherwise, return nullptr.
   */
  const int64_t *constInt();
};

/*!
 * \brief Assuming the index expression is affined,
 *        we have sum(Loops[i] * Coef[i]) + Base.
 *        More complicatedly, Base can recursively be another level of linear combine.
 */
struct LinearCombine : SEWrapper {
  /*!
   * \brief The coefficient of each loop variable.
   */
  std::vector<SEWrapper*> Coef;
  /*!
   * \brief The trip count of each loop variable.
   */
  std::vector<SEWrapper*> TripCount;
  /*!
   * \brief The base coefficient
   */
  SEWrapper *Base;
  /*!
   * \brief Dump this result of analysis for debugging.
   * \param Loops The loops on which analysis is based.
   */
  std::string toString();
  /*!
   * \brief Return the loop level from which, this value is no longer a loop
   * nest invariant.
   */
  int partialInvariant() const;
};

LinearCombine analyzeIndexExpr(ScalarEvolution *SE, const Value *Val,
                               const std::vector<Loop *> &LoopNest);

/*!
 * \brief Fuse continous inner-most memory access dimensions.
 * \param LI The memory access pattern to be fused.
 * \param DType The data type of memory access.
 * \param Unroll The unrolling degree of this access.
 * \param P The padding strategy of the access.
 * \param SE The SCEV analysis.
 * \param CutOff The dimension to stop fusing. Sometimes we do not want every dimension to be fused.
 */
void fuseInnerDimensions(LinearCombine &LI, int DType, int Unroll,
                         ScalarEvolution &SE, int CutOff);

} // namespace analysis
} // namespace dsa
