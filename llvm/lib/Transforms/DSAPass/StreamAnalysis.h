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

/*
 * \brief To fit in LLVM dyn_cast system. A type enumerator is required
 */
enum class IAKind {
  kLoopInvariant,
  kLinearCombine,
  kExprComputable,
  kUnknown
};

struct IndexAnalysis;
struct LoopInvariantDim;
struct LinearCombination;

/*!
 * \brief The visitor class for index analysis.
 */
struct IAVisitor {
  virtual void visit(IndexAnalysis*);
  virtual void visit(LoopInvariantDim*);
  virtual void visit(LinearCombination*);
};

/*!
 * \brief A base class of the analyzed SCEV wrapper.
 */
struct IndexAnalysis {

  IndexAnalysis(IAKind K = IAKind::kUnknown) : Kind(K) {}

  /*!
   * \brief Dump this to a string for debug.
   */
  virtual std::string toString(const std::vector<Loop *> &Loops = std::vector<Loop *>()) const = 0;

  /*!
   * \brief The entrance of a visitor pattern.
   */
  virtual void accept(IAVisitor *IAV) { IAV->visit(this); }

  virtual ~IndexAnalysis() {}

  /*!
   * \brief To fit in LLVM dyn_cast system. A type enumerator is required.
   */
  IAKind Kind;

  /*!
   * \brief To fit in LLVM dyn_cast system. A class enumerator is required.
   */
  static bool classof(const IndexAnalysis *IA) { return true; }
};

/*! \brief A loop invariant through all the loop levels. */
struct LoopInvariantDim : IndexAnalysis {
  /*! \brief The analyzed loop invariant. */
  const SCEV *Value;

  LoopInvariantDim(const SCEV *V) : IndexAnalysis(IAKind::kLoopInvariant), Value(V) {}

  /*!
   * \brief Dump this to a string for debug.
   */
  std::string toString(const std::vector<Loop *> &Loops = std::vector<Loop *>()) const override;

  void accept(IAVisitor *IAV) override { IAV->visit(this); }

  /*!
   * \brief To fit in LLVM dyn_cast system. A class enumerator is required.
   */
  static bool classof(const IndexAnalysis *IA) { return IA->Kind == IAKind::kLoopInvariant; }

  /*!
   * \brief If this loop invariant is a constant return the pointer of this constant.
   *        O.w return nullptr.
   */
  const uint64_t *constInt() const;
};

/*!
 * \brief Assuming the index expression is affined, we have sum(Loops[i] *
 * Coef[i]) + Base.
 */
struct LinearCombination : IndexAnalysis {

  LinearCombination() : IndexAnalysis(IAKind::kLinearCombine) {}

  LinearCombination(const LinearCombination &B) :
    IndexAnalysis(IAKind::kLinearCombine), Coef(B.Coef), Base(B.Base) {}

  /*!
   * \brief The coefficient of each loop variable.
   */
  std::vector<IndexAnalysis*> Coef;
  /*!
   * \brief The base constant, but in 2-d indirect, this can also be a linear combination.
   */
  IndexAnalysis *Base{nullptr};
  /*!
   * \brief Create loop invariant.
   * \param N The number of levels of loop nest.
   */
  static LinearCombination loopInvariant(ScalarEvolution *SE, int N, const SCEV *Base);
  /*!
   * \brief Dump this result of analysis for debugging.
   * \param Loops The loops on which analysis is based.
   */
  std::string toString(const std::vector<Loop *> &Loops = std::vector<Loop *>()) const override;
  /*!
   * \brief Return the loop level from which, this value is no longer a loop
   * nest invariant.
   */
  int partialInvariant() const;

  void accept(IAVisitor *IAV) override { IAV->visit(this); }

  static bool classof(const IndexAnalysis *IA) { return IA->Kind == IAKind::kLinearCombine; }
};

/*!
 * \brief The index is computable by a dataflow graph expression.
 */
struct ExprComputable : IndexAnalysis {
  /*! \brief The raw root of the value to be analyzed. */
  const SCEV *Root;
  /*! \brief The instructions to construct a DFG. */
  std::vector<Instruction*> Insts;

  ExprComputable(const SCEV *Root);

  void accept(IAVisitor *IAV) override { IAV->visit(this); }
};

IndexAnalysis *analyzeIndexExpr(ScalarEvolution *SE, const SCEV *Index,
                                const std::vector<Loop *> &LoopNest);

std::pair<IndexAnalysis*, std::vector<IndexAnalysis*>>
fuseInnerDimensions(IndexAnalysis *IdxLI, std::vector<IndexAnalysis*> LoopLI,
                    int DType, int Unroll, int P, IRBuilder<> *IB, ScalarEvolution &SE);

} // namespace analysis
} // namespace dsa
