#include "DFGEntry.h"
#include "Util.h"
// #include "dsa/arch/model.h"
// #include "dsa/core/singleton.h"
#include "dsa/debug.h"

#include "./CodeXform.h"
#include "./DFGAnalysis.h"
#include "./StreamAnalysis.h"

#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/Casting.h"
#include <string>

#define DEBUG_TYPE "dfg-analysis"

namespace dsa {
namespace analysis {

/*
 * \brief Find all the pairs of ss.config.start() and ss.config.end()
 */
std::vector<std::pair<IntrinsicInst *, IntrinsicInst *>> gatherConfigScope(Function &F) {
  std::vector<std::pair<IntrinsicInst *, IntrinsicInst *>> Res;
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto *End = dyn_cast<IntrinsicInst>(&I)) {
        if (End->getIntrinsicID() == Intrinsic::ss_config_end) {
          auto *Start = dyn_cast<IntrinsicInst>(End->getOperand(0));
          DSA_CHECK(Start && Start->getIntrinsicID() == Intrinsic::ss_config_start);
          Res.emplace_back(Start, End);
        }
      }
    }
  }
  return Res;
}

void DFGAnalysisResult::injectAnalyzedArrayHint(DFGFile &DF, dsa::xform::CodeGenContext &CGC) {
  // All the information gathered in the function `gatherArraySizeHints` will
  // be injected into the DFG data structure.
  for (int i = 0; i < (int) DF.DFGs.size(); ++i) { // NOLINT
    auto &Entries = DF.DFGs[i]->Entries;
    Instruction *Inst = nullptr;
    for (int j = 0; j < (int) Entries.size(); ++j) { // NOLINT
      auto *Entry = Entries[j];
      int DType = -1;
      if (auto *MP = dyn_cast<MemPort>(Entry)) {
        DType = MP->Load->getType()->getScalarSizeInBits() / 8;
        Inst = MP->Load;
      } else if (auto *PM = dyn_cast<PortMem>(Entry)) {
        DType = PM->Store->getValueOperand()->getType()->getScalarSizeInBits() / 8;
        Inst = PM->Store;
      } else if (auto *SMP = dyn_cast<SLPMemPort>(Entry)) {
        DType = SMP->Coal[0]->Load->getType()->getScalarSizeInBits() / 8;
        Inst = SMP->Coal[0]->Load;
      } else if (auto *SPM = dyn_cast<SLPPortMem>(Entry)) { 
        DType = SPM->Coal[0]->Store->getValueOperand()->getType()->getScalarSizeInBits() / 8;
        Inst = SPM->Coal[0]->Store;
      }
      if (DType != -1) {
        std::pair<float, float> PortMemReuse = dsa::analysis::getPortReuse(Entry, CGC, *this, DType);
        auto *Array = findArrayInvolved(Inst, ArraySize, SI);
        if (Array) {
          auto *Hint = ArraySize[Array];
          if (auto *Reuse = dyn_cast<ConstantFP>(Hint->getOperand(2))) {
            auto DReuse = Reuse->getValueAPF().convertToDouble();
            if (std::fabs(DReuse + 1) < 1e-5) {
              Hint->setOperand(2, ConstantFP::get(CGC.IB->getContext(), APFloat((double) PortMemReuse.first)));
              DSA_LOG(ARRAY_HINT) << "Injected hint: " << *Hint;
            }
          }
        }
      }
    }
  }
}

void gatherArraySizeHints(DFGFile &DF, dsa::xform::CodeGenContext &CGC, DFGAnalysisResult &DAR) {
  // In the dumped DFG, each port sourced from or destinated to memory can have
  // a set of meta-data to indicate the size of the array and the reuse rate
  // of the data.
  //
  // This is left to programmers to annotate in the intrinsic `arrayhint`.
  // Because in LLVM IR, no matter how you implement the function signature,
  // either `foo(int *a)` or `foo(int a[100])`, it will all be translated to
  // something like `function @foo(i32 *%a)`, which loses the information of
  // array size. Automatically doing this will require a complicated array-wise
  // alias and footprint analysis, which imposes excessive compiler work.
  auto *Start = DF.Config;
  auto *End = DF.Fence;
  auto *DT = CGC.DT;
  std::set<BasicBlock *> OutOfBound;
  for (auto *BBN : breadth_first(DT->getNode(End->getParent()))) {
    if (BBN->getBlock() == End->getParent())
      continue;
    OutOfBound.insert(BBN->getBlock());
  }
  std::vector<Instruction*> ToRemove;
  for (auto *BBN : breadth_first(DT->getNode(Start->getParent()))) {
    auto *BB = BBN->getBlock();
    auto Range = make_range(
        BB != Start->getParent() ? BB->begin() : Start->getIterator(),
        BB != End->getParent() ? BB->end() : End->getIterator());
    for (auto &I : Range) {
      if (auto *CI = dyn_cast<CallInst>(&I)) {
        if (auto *CF = CI->getCalledFunction()) {
          if (CF->getName() == "arrayhint") {
            auto *BCI = dyn_cast<BitCastInst>(CI->getOperand(0));
            DSA_CHECK(BCI) << *CI->getOperand(0);
            DSA_CHECK(isa<ConstantInt>(CI->getOperand(1))) << *CI->getOperand(1);
            DSA_CHECK(isa<ConstantFP>(CI->getOperand(2))) << *CI->getOperand(2);
            DAR.ArraySize[BCI->getOperand(0)] = CI;
          } else if (CF->getName() == "spadoverride") {
            auto *BCI = dyn_cast<BitCastInst>(CI->getOperand(0));
            auto *Const = dyn_cast<ConstantInt>(CI->getOperand(1));
            DSA_CHECK(Const) << *CI->getOperand(1);
            auto *Alloc = dyn_cast<AllocaInst>(BCI->getOperand(0));
            DAR.SI.Offset[Alloc] = Const->getSExtValue();
          }
        }
      }
    }
    if (BB == End->getParent()) {
      break;
    }
  }
}

/*
 * \brief Traverse all the downstream instructions from the `From` instruction.
 * \return {Visited, OutBound}: `Visited` is the set of instructions that are
 *                              within the scope of this DFG, `OutBound` is the
 *                              set of instructions that are out of the scope.
 */
std::pair<std::set<Instruction *>, std::set<Instruction *>>
bfsOperands(DFGBase *DB, Instruction *From, DominatorTree *DT) {
  std::set<Instruction *> Visited, OutBound;
  std::queue<Instruction *> Q;
  Visited.insert(From);
  Q.push(From);
  while (!Q.empty()) {
    auto *Cur = Q.front();
    Q.pop();
    for (size_t I = 0; I < Cur->getNumOperands(); ++I) {
      if (auto *Inst = dyn_cast<Instruction>(Cur->getOperand(I))) {
        if (DT && !DT->dominates(Inst, From)) {
          continue;
        }
        if (!Visited.count(Inst)) {
          if (DB->InThisScope(Inst)) {
            Visited.insert(Inst);
            Q.push(Inst);
          } else {
            OutBound.insert(Inst);
          }
        }
      }
    }
  }
  return {Visited, OutBound};
}

Value* findArrayInvolved(Instruction *Ptr, std::unordered_map<Value*, CallInst*> &AS,
                         SpadInfo &SI) {
  std::set<Instruction *> Visited;
  std::queue<Instruction *> Q;
  Visited.insert(Ptr);
  Q.push(Ptr);
  auto F = [&AS, &SI](Value *Val) -> Value* {
    {
      auto Iter = AS.find(Val);
      if (Iter != AS.end()) {
        return Val;
      }
    }
    {
      if (auto *AI = dyn_cast<AllocaInst>(Val)) {
        auto Iter = SI.Offset.find(AI);
        if (Iter != SI.Offset.end()) {
          return AI;
        }
      }
    }
    return nullptr;
  };
  while (!Q.empty()) {
    auto *Cur = Q.front();
    Q.pop();
    if (auto *Res = F(Cur)) {
      return Res;
    }
    for (size_t I = 0; I < Cur->getNumOperands(); ++I) {
      if (auto *Inst = dyn_cast<Instruction>(Cur->getOperand(I))) {
        if (Visited.find(Inst) == Visited.end()) {
          Q.push(Inst);
          Visited.insert(Inst);
        }
      } else {
        if (auto *Res = F(Cur->getOperand(I))) {
          return Res;
        }
      }
    }
  }
  return nullptr;
}


/*!
 * \brief How DFG Entries are analyzed and extracted.
 */
struct DFGEntryAnalyzer : DFGVisitor {
  /*!
   * \brief The DFG to be analyzed.
   */
  DFGBase *DBPtr;
  /*!
   * \brief The result of an analyzer.
   */
  DominatorTree *DT;
  /*!
   * \brief
   */
  ScalarEvolution &SE;

  /*!
   * \brief Find all the PHI function that is essentially the same value as the
   * given instruction. \param Inst The instruction to be analyzed. \param Equiv
   * The result to be stored.
   */
  void findEquivPhIs(Instruction *Inst, std::set<Instruction *> &Equiv) {
    std::queue<Instruction *> Q;
    Q.push(Inst);
    Equiv.insert(Inst);
    while (!Q.empty()) {
      if (auto *PHI = dyn_cast<PHINode>(Q.front())) {
        for (auto &Elem : PHI->incoming_values()) {
          auto *Casted = dyn_cast<Instruction>(Elem);
          if (!Casted)
            continue;
          if (Equiv.count(Casted))
            continue;
          Q.push(Casted);
          Equiv.insert(Casted);
        }
      }
      for (auto *User : Q.front()->users()) {
        auto *Phi = dyn_cast<PHINode>(User);
        if (!Phi)
          continue;
        if (Equiv.count(Phi))
          continue;
        Q.push(Phi);
        Equiv.insert(Phi);
      }
      Q.pop();
    }

    DSA_LOG(DFG) << "equiv of " << *Inst;
    for (auto *I : Equiv) { DSA_LOG(DFG) << *I; };
  }

  Predicate *findEquivPredicate(Value *LHS, Value *RHS) {
    auto &DB = *DBPtr;
    for (auto *Elem : DB.EntryFilter<Predicate>()) {
      if (Elem->Cond[0]->getOperand(0) == LHS &&
          Elem->Cond[0]->getOperand(1) == RHS)
        return Elem;
      if (Elem->Cond[0]->getOperand(1) == LHS &&
          Elem->Cond[0]->getOperand(0) == RHS)
        return Elem;
    }
    return nullptr;
  }

  /*!
   * TODO(@were): We can make this even more aggressive.
   *              Make index a general expression, and offload it
   *              to compute fabric.
   *              For now, only one level of load is supported.
   * \brief Differentiate this memory load is linear or indirect.
   */
  DFGEntry *differentiateMemoryStream(LoadInst *Load) {
    auto &DB = *DBPtr;
    // Analyze the pointer operand to see if it is indirect or conditional.
    // By conditional, this is currently very adhoc to kv-join.
    if (auto *GEP = dyn_cast<GetElementPtrInst>(Load->getPointerOperand())) {
      // Traverse back from the pointer operand.
      // Gather all the upstream instructions.
      auto BFSBack = bfsOperands(DBPtr, GEP, DT);

      // Return the indirect memory access node, if it is determined
      // as indirect. If not, return null.
      auto FLoad = [Load, BFSBack, &DB](Value *Val) -> DFGEntry * {
        LoadInst *IdxLoad = nullptr;
        for (auto *Elem : BFSBack.first) {
          // If another level of load is involved, it is indirect.
          if (auto *ThisLoad = dyn_cast<LoadInst>(Elem)) {
            DSA_CHECK(!IdxLoad)
              << "For now only one level indirect supported!"
              << *ThisLoad << "\n" << *IdxLoad << "\n" << *Load;
            IdxLoad = ThisLoad;
          }
        }
        if (IdxLoad) {
          return new IndMemPort(&DB, IdxLoad, Load);
        }
        return nullptr;
      };

      // This is to analyze the conditional memory access.
      auto FGather = [Load, GEP, this, &DB](Value *Val) -> DFGEntry * {
        auto *DD = dyn_cast<DedicatedDFG>(&DB);

        auto *Casted = dyn_cast<Instruction>(Val);
        if (!Casted)
          return nullptr;

        if (!DD) {
          return nullptr;
        }


        {
          // If constant or indirect, skip analysis.
          auto *IdxSE = SE.getSCEV(Load->getPointerOperand());
          if (isa<SCEVAddRecExpr>(IdxSE) || isa<SCEVConstant>(IdxSE)) {
            return nullptr;
          }
          // bool SimpleLoop = false;
          // auto *BI = dyn_cast<BranchInst>(&DD->InnerMost()->getLoopLatch()->back());
          // auto *Latch = DD->InnerMost()->getLoopLatch();
          // DSA_CHECK(BI);
          // for (auto *BB : BI->successors()) {
          //   if (BB == Latch) {
          //     SimpleLoop = true;
          //   }
          // }
          // if (SimpleLoop) {
          //   return nullptr;
          // }
        }

        std::set<Instruction *> Equiv;

        findEquivPhIs(Casted, Equiv);

        Value *TripCount = nullptr;
        SmallVector<std::pair<ICmpInst *, bool>, 0> Conditions;
        // Gather all the predications of this pointer, and the condition
        // of moving the pointer forward.
        for (auto *Elem : Equiv) {
          if (auto *PHI = dyn_cast<PHINode>(Elem)) {
            for (auto &Essense : PHI->incoming_values()) {
              auto *Inst = dyn_cast<Instruction>(Essense);
              if (Inst && !isa<PHINode>(Inst)) {
                if (Inst->getOpcode() == BinaryOperator::Add) {
                  auto *DomBB = DT->getNode(Inst->getParent())->getIDom()->getBlock();
                  DSA_LOG(DFG) << *Inst << " dominated by " << DomBB->back();
                  auto *BI = dyn_cast<BranchInst>(&DomBB->back());
                  assert(BI);
                  if (BI->isConditional()) {
                    auto *CondCmp = dyn_cast<ICmpInst>(BI->getCondition());
                    bool OK = true;
                    for (size_t I = 0; I < Conditions.size(); ++I) {
                      auto *AlreadyCmp = Conditions[I].first;
                      if (CondCmp->getNumOperands() !=
                          AlreadyCmp->getNumOperands()) {
                        OK = false;
                        break;
                      }
                      OK = OK && ((AlreadyCmp->getOperand(0) ==
                                       CondCmp->getOperand(0) &&
                                   AlreadyCmp->getOperand(1) ==
                                       CondCmp->getOperand(1)) ||
                                  (AlreadyCmp->getOperand(0) ==
                                       CondCmp->getOperand(1) &&
                                   AlreadyCmp->getOperand(1) ==
                                       CondCmp->getOperand(0)));
                    }
                    assert(OK && "Controlled by more than one predicate!");
                    Conditions.push_back(std::make_pair(
                        CondCmp, Inst->getParent() == BI->getSuccessor(0)));
                  }
                }
              }
            }
          }
          // This is very adhoc to analyze the trip count.
          // It finds the `n` of `i<n`, where `i` is NOT a canonical loop.
          if (DD && Elem->getParent() == DD->InnerMost()->getLoopLatch()) {
            DSA_LOG(DFG) << "Latch" << *Elem;
            for (auto *User : Elem->users()) {
              auto *Cmp = dyn_cast<ICmpInst>(User);
              if (!Cmp)
                continue;
              if (Cmp->getParent() != Elem->getParent())
                continue;
              assert(Cmp->getPredicate() == ICmpInst::Predicate::ICMP_SLT &&
                     "For now only support SLT");
              for (size_t I = 0; I < Cmp->getNumOperands(); ++I) {
                if (Cmp->getOperand(I) != Elem) {
                  TripCount = Cmp->getOperand(I);
                  DSA_LOG(DFG) << "TripCount" << *TripCount;
                }
              }
            }
          }
        }
        // If we have the trip count, and the controlled conditions,
        // it is determined as a conditional memory access.
        if (TripCount && !Conditions.empty()) {

          int Mask = 0;
          SmallVector<bool, 0> Reverse;

          auto *Pred = findEquivPredicate(Conditions[0].first->getOperand(0),
                                         Conditions[0].first->getOperand(1));

          if (!Pred) {
            Pred = new Predicate(&DB, Conditions[0].first);
            DB.Entries.push_back(Pred);
          }

          for (size_t I = 0; I < Conditions.size(); ++I) {
            Reverse.push_back(Pred->addCond(Conditions[I].first));
          }

          for (size_t I = 0; I < Conditions.size(); ++I) {
            auto Cond = Conditions[I];
            DSA_LOG(DFG) <<  "Moveforward: " << Cond.second << " " << *Cond.first;
            int SubMask = PredicateToInt(Cond.first->getPredicate(),
                                         Cond.second, Reverse[I]);
            Mask |= SubMask;
          }

          DSA_LOG(DFG) << "Forward mask: " << Mask;
          // FIXME: GEP for not is good enough, but we need a better way to
          //        figure out the start pointer later.
          return new CtrlMemPort(&DB, Load, GEP->getOperand(0), TripCount, Pred,
                                 Mask);
        }
        return nullptr;
      };

      // TODO(@were): Can I hoist BFSOperands out here?
      for (size_t I = 1; I < GEP->getNumOperands(); ++I) {
        if (auto *Res = FLoad(GEP->getOperand(I)))
          return Res;
        if (auto *Res = FGather(GEP->getOperand(I)))
          return Res;
        if (auto *Cast = dyn_cast<CastInst>(GEP->getOperand(I))) {
          for (size_t J = 0; J < Cast->getNumOperands(); ++J) {
            if (auto *Res = FGather(Cast->getOperand(J)))
              return Res;
          }
        }
      }
    }

    // If it is neither a indirect load nor a conditional load,
    // it is a normal linear load.
    auto *MP = new MemPort(&DB, Load);
    return MP;
  }

  /*!
   * \brief Starting with the given instructions, BFS out to find all
   *        instrucitons slices.
   * \param Q The given starting instructions.
   * \param Visited Push the found instruction, which is also the vector
   *        of the instruction to be put in the DFG.
   */
  void gatherEntryInsts(std::queue<Instruction *> &Q,
                        std::vector<Instruction *> &Visited) {
    auto &DB = *DBPtr;
    while (!Q.empty()) {
      auto *Cur = Q.front();
      int Idx = std::find(Visited.begin(), Visited.end(), Cur) - Visited.begin();
      Q.pop();
      // User is consumer.
      DSA_LOG(DFG) << "Analyze Users of: " << *Cur;
      for (auto *Elem : Cur->users()) {
        auto *Inst = dyn_cast<Instruction>(Elem);
        // If this consumer is not an instruction, skip it.
        if (!Inst) {
          DSA_LOG(DFG) << "Not a Inst: " << *Elem;
          continue;
        }
        // If this instruction is not in the scope of this DFG, skip it.
        if (!DB.Contains(Inst)) {
          DSA_LOG(DFG) << "Not in blocks: " << *Elem;
          continue;
        }
        // If this instruction cannot fit in the DFG, e.g. PHI, skip it.
        if (!CanBeAEntry(Inst)) {
          DSA_LOG(DFG) << "Not a entry: " << *Elem;
          continue;
        }
        // If this value is not put into the result yet,
        auto Iter = std::find(Visited.begin(), Visited.end(), Inst);
        if (Iter == Visited.end()) {
          // put it in.
          Q.push(Inst);
          Visited.push_back(Inst);
        } else {
          // o.w. reorder it to a later place to garantee the topological order.
          //
          // For example, c = a + b. It makes no sense to dump c before a & b.
          // If we are inspecting `b`, and `c` is already put into the result
          // vector (NOTE: Visited is also the result vector) by `a`, and here
          // we are on `b`. If we do nothing, `c` is before `b`, which will
          // cause topological issue.
          if (Iter - Visited.begin() < Idx) {
            Visited.erase(Iter);
            Q.push(Inst);
            Visited.push_back(Inst);
          }
        }
      }
    }
  }

  /*!
   * \brief Inspect the operands of an instruction, and wrap them by DFG entry.
   * \param Inst The instruction to be inspected.
   */
  void inspectOperands(Instruction *Inst) {
    // TODO(@were): Make this a InstVisitor
    auto &DB = *DBPtr;
    DSA_LOG(DFG) << "Analyze Entry: " << *Inst;

    bool IsAcc = false;

    // For a branch instruction, inspect its condition, since spatial
    // architecture already pre-buffers all the instructions.
    auto *ToOffload =
        !isa<BranchInst>(Inst)
            ? Inst
            : dyn_cast<Instruction>(dyn_cast<BranchInst>(Inst)->getCondition());

    for (size_t I = 0; I < ToOffload->getNumOperands(); ++I) {
      auto *Operand = ToOffload->getOperand(I);
      DSA_LOG(DFG) << "Operand: " << *Operand;
      if (DB.InThisDFG(Operand)) {
        DSA_LOG(DFG) << "Already-in skip!";
        continue;
      }
      // Accumulator
      if (auto *Phi = dyn_cast<PHINode>(Operand)) {

        std::queue<PHINode *> Q;
        std::set<PHINode *> Visited;

        for (Q.push(Phi), Visited.insert(Phi); !Q.empty();) {
          auto *Poll = Q.front();
          Q.pop();
          for (auto &InComing : Poll->incoming_values()) {
            if (InComing == Inst) {
              IsAcc = true;
              break;
            }
            if (auto *AnotherPhi = dyn_cast<PHINode>(InComing)) {
              if (!Visited.count(AnotherPhi)) {
                Q.push(AnotherPhi);
                Visited.insert(AnotherPhi);
              }
            }
          }
        }

        // FIXME: I am not sure if this will suffice. I relax the constraint
        // of detecting a accumulation by checking the BFS successors.
        DSA_CHECK(IsAcc) << "Phi should be an accumulator! " << *Phi;

      // Handle indirect memory access.
      } else if (auto *Load = dyn_cast<LoadInst>(Operand)) {
        DB.Entries.push_back(differentiateMemoryStream(Load));
      // Handle stream-in values.
      } else if (auto *Consumee = dyn_cast<Instruction>(Operand)) {
        DSA_LOG(DFG) << "Not in this Nest: " << DB.Contains(Consumee);
        if (!DB.Contains(Consumee)) {
          if (DB.BelongOtherDFG(Consumee)) {
            DB.Entries.push_back(new StreamInPort(&DB, Consumee));
            DSA_LOG(DFG) << "Upstream: " << *Consumee;
          } else {
            DB.Entries.push_back(new InputConst(&DB, Consumee));
            DSA_LOG(DFG) << "Outer loop invariant: " << *Consumee;
          }
        }
      // Handle constant values.
      } else if (!isa<Constant>(Operand)) {
        DB.Entries.push_back(new InputConst(&DB, Operand));
        DSA_LOG(DFG) << "Other non const: " << *Operand;
      }
    }

    DFGEntry *Entry = nullptr;
    if (!IsAcc) {
      // If this instruction is conditionally executed, associate it with
      // the DFG node.
      if (auto *BI = dyn_cast<BranchInst>(Inst)) {
        assert(BI->isConditional());
        if (ICmpInst *Cmp = dyn_cast<ICmpInst>(BI->getCondition())) {
          Predicate *PredEntry =
              findEquivPredicate(Cmp->getOperand(0), Cmp->getOperand(1));
          if (!PredEntry) {
            PredEntry = new Predicate(&DB, Cmp);
            DB.Entries.push_back(PredEntry);
          } else {
            PredEntry->addCond(Cmp);
          }
        } else if (Instruction *Cond =
                       dyn_cast<Instruction>(BI->getCondition())) {
          DB.Entries.push_back(new Predicate(&DB, Cond));
        }
        Entry = DB.Entries.back();
      } else {
        // Normal instruction.
        auto *CB = new ComputeBody(&DB, Inst);
        DB.Entries.push_back(CB);
        Entry = CB;
        DSA_LOG(DFG) << "Plain Inst: " << *Inst;
      }
    } else {
      // Accumulation.
      auto *Acc = new Accumulator(&DB, Inst);
      DB.Entries.push_back(Acc);
      Entry = Acc;
      DSA_LOG(DFG) << "Accumulator: " << *Inst;
    }
    DSA_CHECK(Entry);
  }

  /*!
   * \brief Inspect the ends up of an instruction, and wrap them by DFG entry.
   * \param Inst The instruction to be inspected.
   */
  void inspectConsumers(Instruction *Inst) {
    auto &DB = *DBPtr;
    auto &Entries = DB.Entries;
    std::set<Instruction *> Equiv;
    findEquivPhIs(Inst, Equiv);
    DSA_LOG(DFG) << "Inspect consumers of: " << *Inst;
    for (auto *Elem : Equiv) {
      DSA_LOG(DFG) << "Phi Equiv: " << *Elem;
      for (auto *User : Elem->users()) {
        DSA_LOG(DFG) << "User: " << *User;
        if (auto *Store = dyn_cast<StoreInst>(User)) {
          LoadInst *Load = nullptr;
          // This is for data move, a[i] = b[i].
          if (auto *GEP = dyn_cast<GetElementPtrInst>(Store->getPointerOperand())) {
            auto BFSBack = bfsOperands(DBPtr, GEP, DT);
            for (auto *Elem : BFSBack.first) {
              if ((bool)(Load = dyn_cast<LoadInst>(Elem))) {
                break;
              }
            }
          }
          // This is for atomic analysis, a[b[i]] += c[i].
          if (Load && DB.Contains(Load)) {
            if (auto *BO = dyn_cast<BinaryOperator>(Store->getValueOperand())) {
              bool IsUpdate = false;
              Value *AtomicOperand = nullptr;
              if (BO->getOpcode() == Instruction::BinaryOps::Add) {
                for (size_t I = 0; I < BO->getNumOperands(); ++I) {
                  auto *Operand = dyn_cast<LoadInst>(BO->getOperand(I));
                  if (Operand && Operand->getPointerOperand() == Store->getPointerOperand()) {
                    IsUpdate = true;
                  } else {
                    AtomicOperand = BO->getOperand(I);
                  }
                }
              }
              if (!IsUpdate) {
                AtomicOperand = nullptr;
              } else {
                DSA_CHECK(AtomicOperand);
                DSA_LOG(DFG) << "Operation: " << *AtomicOperand;
              }
              Entries.push_back(new AtomicPortMem(
                    &DB, Load, Store, IsUpdate ? 0 : 3, Inst, AtomicOperand));
            } else {
              Entries.push_back(new AtomicPortMem(&DB, Load, Store, 3, Inst, nullptr));
            }
            DSA_LOG(DFG) << "AtomicMem: " << *Inst;
          } else {
            // Write to memory normally.
            auto *Out = new PortMem(&DB, Store);
            Entries.push_back(Out);
            DSA_LOG(DFG) << "PortMem: " << *Inst;
          }
        } else if (CanBeAEntry(User) && DB.BelongOtherDFG(dyn_cast<Instruction>(User))) {
          // Support downstream data consumption, stream out.
          auto *Consume = dyn_cast<Instruction>(User);
          auto *Out = new StreamOutPort(&DB, Inst);
          Entries.push_back(Out);
          DSA_LOG(DFG) << "Stream " << *Inst << " to " << *Consume;
        } else if (auto *UserInst = dyn_cast<Instruction>(User)) {
          if (isa<PHINode>(UserInst))
            continue;
          // Write this value to register not memory.
          if (!DB.BelongOtherDFG(UserInst) && !DB.InThisDFG(UserInst) && !DB.Contains(UserInst)) {
            Entries.push_back(new OutputPort(&DB, Inst));
            DSA_LOG(DFG) << "Write to register: " << *Inst;
          }
        }
      }
    }
  }

  /*
   * \brief The entrance of analysing the entries of a dedicated DFG.
   */
  void Visit(DedicatedDFG *DDPtr) override {
    std::vector<Instruction *> Visited;
    DBPtr = DDPtr;
    auto &DD = *DDPtr;
    std::queue<Instruction *> Q;

    // Inspect all the blocks of the inner most loop.
    for (auto *BB : DD.InnerMost()->getBlocks()) {

      // This is to handle the predication based execution.
      // For each code block, if its immediate dominator is within this
      // inner most loop, use the dominator's considiton as the predicate
      // of this block. Therefore, we need to inspect this condition
      // instruction.
      auto *DomBB = DT->getNode(BB)->getIDom()->getBlock();
      if (DD.InnerMost()->getBlocksSet().count(DomBB)) {
        auto *BI = dyn_cast<BranchInst>(&DomBB->back());
        if (BI->isConditional()) {
          Visited.push_back(BI);
        }
      }
      // Starting with load instructions,
      // find all their consumers to build the DFG.
      for (auto &Inst : *BB) {
        if (auto *Load = dyn_cast<LoadInst>(&Inst)) {
          Visited.push_back(Load);
          Q.push(Load);
        }
      }
      // Use a breadth first search to traverse all the consumers, to build
      // the DFG.
      gatherEntryInsts(Q, Visited);
    }

    for (auto *Inst : Visited) {
      if (!isa<LoadInst>(Inst)) {
        inspectOperands(Inst);
        inspectConsumers(Inst);
      }
    }

    // Inspect a[i] = b[i], and a[b[i]] ?= c[i].
    if (DD.Entries.empty()) {
      DSA_CHECK(Visited.size() == 1 && isa<LoadInst>(Visited[0]));
      DD.Entries.push_back(differentiateMemoryStream(dyn_cast<LoadInst>(Visited[0])));
      inspectConsumers(Visited[0]);
    }

    DSA_CHECK(!DD.Entries.empty());
  }

  void Visit(TemporalDFG *TD) override {
    DBPtr = TD;
    std::queue<Instruction *> Q;
    std::vector<Instruction *> Visited;
    DSA_CHECK(TD->getBlocks().size() == 1);

    for (auto *I = TD->Begin->getNextNode(); I != TD->End;
         I = I->getNextNode()) {
      if (CanBeAEntry(I)) {
        Q.push(I);
        Visited.push_back(I);
      }
    }
    gatherEntryInsts(Q, Visited);

    // Besides the traverse, the input operands and the consumers should
    // also be inspected again for special cases.
    for (auto *Inst : Visited) {
      if (!isa<LoadInst>(Inst)) {
        // 1. For constant coefficients, these values are not instructions
        //    at all. If we do not inspect them here, they have no chance
        //    to be further inspected.
        // 2. For accumulators, their consumers are PHI node, which should be
        //    treated as a special case.
        // 3. For an indirect stream, its input operand can be another level
        //    of memory access, which also involves special rules.
        // 4. Look at consumers below, this is the consumer side of the story.
        //    We need to expose a in-port for the values to stream in.
        inspectOperands(Inst);
        // The current DFG is the producer, and this value can be consumed
        // by external DFG. The consumer instruction will not
        // be considered as a part of this DFG, but we still
        // need to expose a port for these inter-DFG communication.
        inspectConsumers(Inst);
      }
    }
  }

  DFGEntryAnalyzer(xform::CodeGenContext &CGC) : DT(CGC.DT), SE(CGC.SE) {}
};

void extractSpadFromScope(DFGFile &DF, xform::CodeGenContext &CGC, DFGAnalysisResult &DAR) {
  Instruction *Start = DF.Config;
  Instruction *End = DF.Fence;
  auto &Offset = DAR.SI.Offset;
  auto &Total = DAR.SI.Total;
  // Compactly map memory addresses to scratchpad addresses.
  // NOTE: This is just a allocate pass, no code replacement is done.
  //       Code replacement will be done in stream generation.
  for (auto *BB : breadth_first(Start->getParent())) {
    iterator_range<BasicBlock::InstListType::iterator> Range(BB->begin(), BB->end());
    if (BB == Start->getParent()) {
      Range = iterator_range<BasicBlock::InstListType::iterator>(Start->getIterator(), BB->end());
    } else if (BB == End->getParent()) {
      Range = iterator_range<BasicBlock::InstListType::iterator>(BB->begin(), End->getIterator());
    }
    for (auto &Inst : Range) {
      if (auto *GEP = dyn_cast<GetElementPtrInst>(&Inst)) {
        if (auto *Alloc = dyn_cast<AllocaInst>(GEP->getPointerOperand())) {
          if (auto *AT = dyn_cast<ArrayType>(Alloc->getType()->getElementType())) {
            if (!Offset.count(Alloc)) {
              int Size = AT->getNumElements() * AT->getElementType()->getScalarSizeInBits() / 8;
              Offset[Alloc] = Total;
              DSA_LOG(SPAD) << *Alloc << ": " << Total;
              Total += Size;
            } else {
              DSA_LOG(SPAD) << *Alloc << " already assigned to " << Offset[Alloc];
            }
          }
        }
      }
    }
  }

  // for (auto &I : Range) {
  //   if (auto *CI = dyn_cast<CallInst>(&I)) {
  //     if (auto *CF = CI->getCalledFunction()) {
  //       if (CF->getName() == "arrayhint") {
  //         auto *BCI = dyn_cast<BitCastInst>(CI->getOperand(0));
  //         DSA_CHECK(BCI) << *CI->getOperand(0);
  //         DSA_CHECK(isa<ConstantInt>(CI->getOperand(1))) << *CI->getOperand(1);
  //         DSA_CHECK(isa<ConstantFP>(CI->getOperand(2))) << *CI->getOperand(2);
  //         DAR.ArraySize[BCI->getOperand(0)] = CI;
  //       }
  //     }
  //   }
  // }

  // If DOUBLE_BUFFER flag is not enabled, skip the buffet analysis.
  if (!utils::ModuleContext().DOUBLE_BUFFER) {
    return;
  }
  // Buffet analysis basically support such a 3-d pattern:
  // for i in 0..ext.i
  //   for j in 0..ext.j
  //     for k in 0..ext.k
  //       which means it is reused across j and will move forward on i
  //       access a[i,k]
  auto DFGs = DF.DFGFilter<DFGBase>();
  for (int i = 0; i < (int) DFGs.size(); ++i) { // NOLINT
    auto *DFG = DFGs[i];
    struct Analyzer : DFGEntryVisitor {
      void Visit(IndMemPort *) override {}
      void Visit(AtomicPortMem *) override {}

      void buffetImpl(LinearCombine *LC, Type *Ty, InputPort *IP) {
        std::vector<const SCEV*> N;
        std::vector<const SCEV*> Stride;
        int PI = LC->partialInvariant();
        auto &TripCnt = LC->TripCount;
        if (LC->Coef.size() - PI == 3) {
          for (int j = PI; j < (int) TripCnt.size(); ++j) { // NOLINT
            N.push_back(TripCnt[j]->Raw);
            Stride.push_back(LC->Coef[j]->Raw);
          }
          if (auto *CIS2D = dyn_cast<SCEVConstant>(Stride[1])) {
            if (auto *CIN1D = dyn_cast<SCEVConstant>(N[0])) {
              if (auto *CIS3D = dyn_cast<SCEVConstant>(Stride[2])) {
                // which is the j above.
                if (CIS2D->getAPInt().getSExtValue() == 0 && CIS3D->getAPInt().getSExtValue() != 0) {
                  int DType = Ty->getScalarSizeInBits() / 8;
                  int BufferSize = (CIN1D->getAPInt().getSExtValue()) * 2 * DType;
                  DSA_LOG(BUFFET)
                    << SI.Buffet.size() << ": " << LC->toString() << ", "
                    << (CIN1D->getAPInt().getSExtValue()) << " * " << DType
                    << " * 2 = " << BufferSize << "\n" << LC->toString() << "\n";
                  // MP, Total, BufferSize, -1
                  SI.Buffet.emplace_back(IP, SI.Total, BufferSize, -1, -1);
                  SI.Total += BufferSize;
                }
              }
            }
          }
        }
      }

      void Visit(MemPort *MP) override {
        auto *IdxLI = DAR.affineMemoryAccess(MP, CGC.SE, false);
        if (auto *LC = dyn_cast<LinearCombine>(IdxLI)) {
          buffetImpl(LC, MP->Load->getType(), MP);
        }
      }

      void Visit(SLPMemPort *SMP) override {
        auto *IdxLI = DAR.affineMemoryAccess(SMP, CGC.SE, false);
        if (auto *LC = dyn_cast<LinearCombine>(IdxLI)) {
          buffetImpl(LC, SMP->Coal[0]->Load->getType(), SMP);
        }
      }

      int Unroll;
      DFGAnalysisResult &DAR;
      SpadInfo &SI;
      DFGLoopInfo &DLI;
      IRBuilder<> *IB;
      xform::CodeGenContext &CGC;
      ScalarEvolution &SE;
      Analyzer(int U, DFGAnalysisResult &DAR, SpadInfo &SI, DFGLoopInfo &DLI,
               xform::CodeGenContext &CGC) :
        Unroll(U), DAR(DAR), SI(SI), DLI(DLI), IB(CGC.IB), CGC(CGC), SE(CGC.SE) {}
    };
    Analyzer A(DFG->getUnroll(), DAR, DAR.SI, DAR.DLI[i], CGC);
    for (auto &Elem : DFG->Entries) {
      Elem->accept(&A);
    }
  }
}

void extractDFGFromScope(DFGFile &DF, dsa::xform::CodeGenContext &CGC) {
  auto *Start = DF.Config;
  auto *End = DF.Fence;
  auto *DT = CGC.DT;
  auto *LI = CGC.LI;
  // Extract the sketch of dedicated and temporal DFGs respectively.

  // This is something like a 2-pass analysis. The first pass get the blocks
  // involved in the DFG. The second pass fill in the instructions involved
  // in the DFG.
  //
  // We need a 2-pass analysis because we do not want to run into the
  // chicken-egg paradox: say DFG 1 and DFG 2 communicates through
  // an instruction value, but when analyzing DFG 1 the DFG 2
  // does not exist yet. Therefore, we first gather all the blocks of
  // each DFG. Then, when analysing DFG 1, if we find one instruction value
  // belongs to DFG 1 also used in DFG 2, this can be later handled.

  // Pass 1: Rule out all the blocks should not be involved in.

  // Get all the blocks dominated by the end intrinsic. They anyways should not
  // be in the DFG.
  std::set<BasicBlock *> OutOfBound;
  for (auto *BBN : breadth_first(DT->getNode(End->getParent()))) {
    if (BBN->getBlock() == End->getParent())
      continue;
    OutOfBound.insert(BBN->getBlock());
  }

  for (auto *NestedLoop : *LI) {
    for (auto *SubLoop : breadth_first(NestedLoop)) {
      // Filter out all the loops not involved in this DFG.
      if (!DT->dominates(Start, SubLoop->getBlocks()[0]))
        continue;
      if (!SubLoop->getLoopID() || SubLoop->getLoopID()->getNumOperands() == 0)
        continue;
      bool InBound = true;
      for (auto *BB : SubLoop->getBlocks()) {
        if (OutOfBound.count(BB))
          InBound = false;
      }
      if (!InBound)
        continue;
      // Get the unrolling degree.
      if (MDNode *MD = GetUnrollMetadata(SubLoop->getLoopID(),
                                         "llvm.loop.ss.dedicated")) {
        auto *MDFactor = dyn_cast<ConstantAsMetadata>(MD->getOperand(1));
        DSA_CHECK(MDFactor);
        int Factor = (int)MDFactor->getValue()->getUniqueInteger().getSExtValue();
        if (Factor == 0) {
          Factor = 1;
        }
        auto *DD = new DedicatedDFG(&DF, SubLoop, Factor);
        DF.addDFG(DD);
      }
    }
  }
  // Gather all the blocks wrapped by temporal.begin/end.
  for (auto *BBN : breadth_first(DT->getNode(Start->getParent()))) {
    auto *BB = BBN->getBlock();
    auto Range = make_range(
        BB != Start->getParent() ? BB->begin() : Start->getIterator(),
        BB != End->getParent() ? BB->end() : End->getIterator());
    for (auto &I : Range) {
      auto *TemporalStart = dyn_cast<IntrinsicInst>(&I);
      if (!TemporalStart || TemporalStart->getIntrinsicID() !=
                                Intrinsic::ss_temporal_region_start)
        continue;
      // TODO: Maybe later we need control flow in the temporal regions
      bool Found = false;
      for (Instruction *Cur = TemporalStart; Cur != &Cur->getParent()->back();
           Cur = Cur->getNextNode()) {
        if (auto *TemporalEnd = dyn_cast<IntrinsicInst>(Cur)) {
          if (TemporalEnd->getIntrinsicID() !=
              Intrinsic::ss_temporal_region_end)
            continue;
          DSA_CHECK(TemporalEnd->getOperand(0) == TemporalStart);
          auto *TD = new TemporalDFG(&DF, TemporalStart, TemporalEnd);
          DF.addDFG(TD);
          Found = true;
          break;
        }
      }
      DSA_CHECK(Found);
    }
    if (BB == End->getParent()) {
      break;
    }
  }
  // Pass 2: After all the blocks are gathered, fill in the instructions.
  auto DFGs = DF.DFGFilter<DFGBase>();
  // All the sketches are extracted. Fill in the entries.
  for (int i = 0; i < (int) DFGs.size(); ++i) { // NOLINT
    DFGEntryAnalyzer DEA(CGC);
    DFGs[i]->accept(&DEA);
  }
}

ConfigInfo extractDFGPorts(DFGFile &DF, SpadInfo &SI, bool Schedule) {
  auto *SBCONFIG = getenv("SBCONFIG");
  DSA_CHECK(SBCONFIG);
  std::string ExtraFlag = "";
  if (dsa::utils::ModuleContext().DUMMY) {
    ExtraFlag += " --dummy";
  }
  bool BitStream = dsa::utils::ModuleContext().BITSTREAM;
  if (BitStream) {
    ExtraFlag += " -b";
  }
  if (dsa::utils::ModuleContext().COMPAT_ADG) {
    ExtraFlag += " -a";
  }
  auto FName = DF.FileName;
  std::string Stripped(FName);
  while (Stripped.back() != '.')
    Stripped.pop_back();
  Stripped.pop_back();
  int Seed = 0;
  if (dsa::utils::ModuleContext().SEED == -1) {
    Seed = time(0);
  }

  if (Schedule) {
    utils::ModuleContext().TP.beginRoi();
    auto Cmd = formatv("ss_sched --verbose --max-iters=1024 {0} {1} {2} {3} > {4}.log 2>&1",
                       ExtraFlag, SBCONFIG, FName, "-e " + std::to_string(Seed), Stripped).str();
    DSA_LOG(CONFIG) << Cmd;
    int RetCode = system(Cmd.c_str());
    utils::ModuleContext().TP.endRoi();
    if (RetCode != 0) {
      return ConfigInfo(Stripped, "", RetCode);
    }
  }

  std::ifstream Ifs(FName + (BitStream ? ".bits" : "") + ".h");
  auto DFGs = DF.DFGFilter<DFGBase>();

  int ConfigSize = 0;
  std::string ConfigString;

  std::string Line;
  std::string PortPrefix(formatv("P_{0}_", Stripped).str());
  std::string IOPortPrefix = PortPrefix + "sub";
  std::string IClusterPrefix = PortPrefix + "ICluster";
  std::string OClusterPrefix = PortPrefix + "OCluster";
  std::string IndirectPrefix = PortPrefix + "indirect_";
  std::string IBuffetPrefix = PortPrefix + "IBuffet";
  std::string OBuffetPrefix = PortPrefix + "OBuffet";
  std::string L2DPrefix = PortPrefix + "l2d_";
  std::string SignalSuffix = "_signal";
  while (std::getline(Ifs, Line)) {
    std::istringstream Iss(Line);
    std::string Token;
    Iss >> Token;
    // #define xxx
    if (Token == "#define") {
      Iss >> Token;
      // #define P_dfgX_subY_vZ
      if (Token.find(IOPortPrefix) == 0) {
        int X, Y;
        auto Stripped = Token.substr(IOPortPrefix.size());
        bool IsSignal = false;
        auto Pos = Stripped.find(SignalSuffix);
        if (Pos != std::string::npos) {
          if (Pos + SignalSuffix.size() == Stripped.size()) {
            Stripped = std::string(Stripped.begin(), Stripped.end() - SignalSuffix.size());
            IsSignal = true;
          }
        }
        DSA_CHECK(sscanf(Stripped.c_str(), "%d_v%d", &X, &Y) == 2);
        int Port;
        Iss >> Port;
        DSA_LOG(CONFIG) << "sub" << X << "v" << Y << " -> " << Port;
        DSA_CHECK(X >= 0 && X < (int) DFGs.size()) << Token << ": " << X << ", " << DFGs.size();
        DSA_CHECK(Y >= 0 && Y < (int) DFGs[X]->Entries.size());
        auto *Entry = DFGs[X]->Entries[Y];
        if (auto *PB = dyn_cast<PortBase>(Entry)) {
          if (!IsSignal) {
            PB->SoftPortNum = Port;
          } else if (auto *IMP = dyn_cast<IndMemPort>(PB)) {
            IMP->SignalPort = Port;
          }
        } else if (auto *CB = dyn_cast<ComputeBody>(Entry)) {
          DSA_CHECK(false) << Token << ": " << X << " " << Y << " This should be deprecated!";
          if (!CB->isImmediateAtomic()) {
            DSA_LOG(CONFIG) << "OutputPorts:";
            auto OutPorts = CB->getOutPorts();
            for (auto *Port : CB->getOutPorts()) {
              (void) Port;
              LLVM_DEBUG(Port->underlyingInst()->dump());
            }
            // FIXME: For now only one destination is supported
            // Later divergence will be supported
            DSA_CHECK(OutPorts.size() == 1)
                << *CB->underlyingInst() << " " << OutPorts.size();
            OutPorts[0]->SoftPortNum = Port;

            // Fill in the latency of each node.
            //{
            //  OutputPort *OP = OutPorts[0];
            //  std::string ON = OP->Parent->InThisDFG(OP->Output)->NameInDFG();
            //  for (auto SDVO : dfg.nodes<SSDFGVecOutput*>()) {
            //    if (SDVO->name() == ON) {
            //      OP->Latency = sched->latOf(SDVO);
            //      DSA_LOG(DFG) << "[lat] " << ON << ": " << OP->Latency
            //      << "\n"); break;
            //    }
            //  }
            //}

          } else {
            CB->SoftPortNum = Port;
          }
        } else {
          DSA_CHECK(false) << "DSAPass port information gathering unreachable!";
        }
        // #define dfgx_size size
      } else if (Token.find(IClusterPrefix) == 0 || Token.find(OClusterPrefix) == 0) {
        Token = Token.substr(IClusterPrefix.size());
        int X, Y, Port;
        DSA_CHECK(sscanf(Token.c_str(), "_%d_%d_", &X, &Y) == 2);
        Iss >> Port;
        auto *SLP = dyn_cast<PortBase>(DFGs[X]->Entries[Y]);
        DSA_CHECK(SLP) << DFGs[X]->Entries[Y]->dump();
        SLP->SoftPortNum = Port;
      } else if (Token.find(formatv("{0}_size", Stripped).str()) == 0) {
        Iss >> ConfigSize;
      } else if (Token.find(IndirectPrefix) == 0) {
        Token = Token.substr(IndirectPrefix.size());
        int X, Y;
        if (sscanf(Token.c_str(), "in_%d_%d_", &X, &Y) == 2) {
          auto *IMP = dyn_cast<IndMemPort>(DFGs[X]->Entries[Y]);
          DSA_CHECK(IMP);
          Iss >> IMP->Index->SoftPortNum;
        } else {
          DSA_CHECK(sscanf(Token.c_str(), "out_%d_%d_", &X, &Y) == 2);
          auto *IMP = dyn_cast<IndMemPort>(DFGs[X]->Entries[Y]);
          DSA_CHECK(IMP);
          Iss >> IMP->IndexOutPort;
        }
      } else if (Token.find(PortPrefix + "Operand_") == 0) {
        Token = Token.substr(std::string(PortPrefix + "Operand_").size());
        int X, Y;
        DSA_CHECK(sscanf(Token.c_str(), "sub%d_v%d_", &X, &Y) == 2);
        auto *APM = dyn_cast<AtomicPortMem>(DFGs[X]->Entries[Y]);
        DSA_CHECK(APM);
        Iss >> APM->OperandPort;
      } else if (Token.find(IBuffetPrefix) == 0 || Token.find(OBuffetPrefix) == 0) {
        bool IsInput = Token.find(IBuffetPrefix) == 0;
        Token = Token.substr(IBuffetPrefix.size());
        int X;
        DSA_CHECK(sscanf(Token.c_str(), "_%d_", &X));
        if (IsInput) {
          Iss >> std::get<3>(SI.Buffet[X]);
          DSA_LOG(BUFFET) << "Buffet Read Port: " << std::get<3>(SI.Buffet[X]);
        } else {
          Iss >> std::get<4>(SI.Buffet[X]);
          DSA_LOG(BUFFET) << "Buffet Write Port: " << std::get<4>(SI.Buffet[X]);
        }
      } else if (Token.find(L2DPrefix)) {
        DSA_CHECK(false) << "Support this: " << Token;
      } else {
        DSA_CHECK(false) << "Unrecognized token: " << Token;
      }
      // char dfgx_config[size] = "filename:dfgx.sched";
    } else if (Token == "char" || Token == "uint64_t") {
      if (!BitStream) {
        // dfgx_config[size]
        Iss >> Token;
        // =
        Iss >> Token;
        // "filename:dfgx.sched";
        Iss >> Token;
        ConfigString = Token.substr(1, Token.size() - 3);
        if ((int) ConfigString.size() > ConfigSize) {
          ConfigString += std::string(ConfigSize - ConfigString.size() - 1, '\0');
        } else {
          ConfigSize = ConfigString.size() + 1;
        }
      } else {
        std::string Raw;
        for (int i = 0; i < ConfigSize; ++i) { // NOLINT
          DSA_CHECK(std::getline(Ifs, Raw));
          while (isspace(Raw[0])) {
            Raw.erase(Raw.begin());
          }
          while (Raw.back() != ',') {
            Raw.pop_back();
          }
          Raw.pop_back();
          uint64_t Word = 0;
          if (Raw[0] == '0' && Raw[1] == 'b') {
            for (int j = 2; j < (int) Raw.size(); ++j) {  // NOLINT
              Word = (Word << 1) | (Raw[j] - '0');
            }
          } else {
            std::istringstream Iss(Raw);
            DSA_CHECK(Iss >> Word);
          }
          std::string WordByte((char*)&Word, (char*)&Word + 8);
          ConfigString.insert(ConfigString.end(), WordByte.begin(), WordByte.end());
        }
        DSA_CHECK(std::getline(Ifs, Raw));
      }
    }
  }

  ConfigInfo Res(Stripped, ConfigString, ConfigSize);
  DSA_LOG(DFG) << "[Config] " << ConfigSize << ": " << ConfigString << "\n";
  {
    std::ifstream Fin(Stripped + ".log");
    std::string Raw;
    while (Fin >> Raw) {
      if (Raw == "Performance:") {
        Fin >> Res.EstimatedILP;
      }
    }
  }
  return Res;
}

void analyzeDFGLoops(DFGFile &DF, xform::CodeGenContext &CGC, DFGAnalysisResult &DAR) {
  struct DFGLoopAnalyzer : DFGVisitor {
    // For now, we only assume dedidated regions require excessive computing
    // power, so we only specialize and analyze loops for dedicated regions.
    void Visit(DedicatedDFG *DD) override {
      auto &LoopNest = DD->LoopNest;
      DLI.LoopNest = LoopNest;
      for (int i = (int) LoopNest.size() - 1; i >= 0; --i) { // NOLINT
        // This returns the times of the backedge is branched, which is `n-1`.
        auto *NSCEV = SE.getBackedgeTakenCount(LoopNest[i]);
        // If the loop bound is not analyzable at all, skip it.
        if (isa<SCEVCouldNotCompute>(NSCEV)) {
          DLI.TripCount.emplace_back();
          DSA_WARNING << "Trip count of Loop " << *LoopNest[i] << " could not compute!";
        } else {
          // Add one, to offset the back edge taken count.
          NSCEV = SE.getAddExpr(NSCEV, SE.getConstant(APInt(64, 1, true)));
          if (const auto *MME = dyn_cast<SCEVMinMaxExpr>(NSCEV)) {
            DSA_CHECK(isa<SCEVUMaxExpr>(MME) || isa<SCEVSMaxExpr>(MME));
            if (const auto *SC = dyn_cast<SCEVConstant>(MME->getOperand(0))) {
              if (SC->getAPInt().getSExtValue() == 1) {
                NSCEV = MME->getOperand(1);
              }
            }
          }
          // Analyze if the loop bound is a linear combination of
          // the loop variables above.
          DSA_LOG(AFFINE) << *NSCEV;
          auto LoopSlice = std::vector<Loop*>(LoopNest.begin() + i + 1, LoopNest.end());
          DSA_LOG(AFFINE) << LoopSlice.size() << " loops above";
          DLI.TripCount.push_back(analysis::analyzeIndexExpr(&SE, NSCEV, LoopNest[i],
                                                             LoopSlice, DLI.TripCount, LinearOverride));
          DLI.TripCount.back()->Parent.L = LoopNest[i];
          DSA_LOG(AFFINE) << i << ": " << DLI.TripCount.back()->toString();
        }
      }
      // We analyze from outer to inner, but we use them
      // for the codegen from inner to outer.
      std::reverse(DLI.TripCount.begin(), DLI.TripCount.end());
    }

    void Visit(TemporalDFG *TD) override {}

    ScalarEvolution &SE;
    DFGLoopInfo &DLI;
    std::unordered_map<const SCEV*, const SCEV*> &LinearOverride;

    DFGLoopAnalyzer(ScalarEvolution &SE, DFGLoopInfo &DLI,
                    std::unordered_map<const SCEV*, const SCEV*> &L) : SE(SE), DLI(DLI), LinearOverride(L) {}
  };

  for (auto *DB : DF.DFGFilter<DFGBase>()) {
    DAR.DLI.emplace_back();
    DFGLoopAnalyzer DLA(CGC.SE, DAR.DLI.back(), DAR.LinearOverride);
    DB->accept(&DLA);
  }
}


/*!
 * \brief Calulate the difference between two index. (B-A)
 *        If their difference is a constant, return it.
 *        If not, return -1.
 * \param A The result of affine analysis of A.
 * \param A The result of affine analysis of B.
 * \param SE The LLVM scalar evolution analyzer.
 * \param Signed If the result returned is abs.
 *        If abs is true, and -1 is returned, it is not a constant.
 *        If abs is false, constant should be garanteed to call.
 */
int64_t indexPairOffset(SEWrapper *ASW, SEWrapper *BSW, ScalarEvolution &SE, bool Signed) {
  bool SameDims = true;
  const SCEV *ABase = nullptr;
  const SCEV *BBase = nullptr;
  if (auto *A = dyn_cast<LinearCombine>(ASW)) {
    if (auto *B = dyn_cast<LinearCombine>(BSW)) {
      if (A->Coef.size() != B->Coef.size()) {
        DSA_CHECK(!Signed);
        return -1;
      }
      for (int j = 0; j < (int) A->Coef.size(); ++j) { // NOLINT
        if (A->Coef[j]->Raw != B->Coef[j]->Raw) {
          DSA_LOG(COAL) << *SE.getEqualPredicate(A->Coef[j]->Raw, B->Coef[j]->Raw);
          SameDims = false;
        }
      }
      if (SameDims /* && isa<LoopInvariant>(A->Base) && isa<LoopInvariant>(B->Base) */) {
        ABase = A->Base->Raw;
        BBase = B->Base->Raw;
      }
    }
  } else {
    if (auto *A = dyn_cast<LoopInvariant>(ASW)) {
      if (auto *B = dyn_cast<LoopInvariant>(BSW)) {
        ABase = A->Raw;
        BBase = B->Raw;
      }
    }
  }
  if (ABase && BBase) {
    if (auto *Offset = dyn_cast<SCEVConstant>(SE.getAddExpr(BBase, SE.getNegativeSCEV(ABase)))) {
      auto Res = Offset->getAPInt().getSExtValue();
      return Signed ? Res : std::abs(Res);
    }
  }
  DSA_CHECK(!Signed) << "If signed, same access pattern should be garanteed!";
  return -1;
}

void DFGAnalysisResult::initAffineCache(DFGFile &DF, ScalarEvolution &SE) {
  for (auto *DFG : DF.DFGs) {
    for (auto *Elem : DFG->Entries) {
      affineMemoryAccess(Elem, SE, true);
    }
  }
}

SEWrapper *DFGAnalysisResult::affineMemoryAccess(DFGEntry *DE, ScalarEvolution &SE, bool AllowNull) {

  struct AffineExtractor : DFGEntryVisitor {
    void Visit(MemPort *MP) override {
      analyze(MP->Load->getPointerOperand(), MP);
      DSA_CHECK(SW);
      auto &LoopNest = DAR.DLI[MP->Parent->ID].LoopNest;
      if (auto *LC = dyn_cast<LinearCombine>(SW)) {
        for (int i = 0; i < (int) LoopNest.size(); ++i) { // NOLINT
          auto *CurLoop = LoopNest[i];
          auto Uses = [CurLoop](LoadInst *Load) {
            for (auto *BB : CurLoop->getBlocks()) {
              for (auto &Inst : *BB) {
                for (int j = 0; j < (int) Inst.getNumOperands(); ++j) { // NOLINT
                  if (Inst.getOperand(j) == Load || &Inst == Load) {
                    return true;
                  }
                }
              }
            }
            return false;
          };
          // Remove all the inner loops un-related to save dimensions. (1)
          if (!Uses(MP->Load)) {
            LC->Coef.erase(LC->Coef.begin());
            LC->TripCount.erase(LC->TripCount.begin());
            continue;
          }
          break;
        }
      }
    }
    void Visit(PortMem *PM) override {
      analyze(PM->Store->getPointerOperand(), PM);
      DSA_CHECK(SW);
      if (auto *LC = dyn_cast<LinearCombine>(SW)) {
        // Remove all the inner loops un-related to save dimensions.
        // This branch should serve the same functionality as (1) above.
        auto PI = LC->partialInvariant();
        LC->Coef.erase(LC->Coef.begin(), LC->Coef.begin() + PI);
        LC->TripCount.erase(LC->TripCount.begin(), LC->TripCount.begin() + PI);
      } else if (auto *LI = dyn_cast<LoopInvariant>(SW)) {
        LI->TripCount.clear();
        LI->TripCount.push_back(LoopInvariant::get(PM, SE.getConstant(APInt(64, 1))));
      }
    }
    void Visit(IndMemPort *IMP) override {
      analyze(IMP->Load->getPointerOperand(), IMP);
    }
    void Visit(AtomicPortMem *APM) override {
      analyze(APM->Store->getPointerOperand(), APM);
    }
    void analyze(Value *Ptr, DFGEntry *DE) {
      auto *Raw = SE.getSCEV(Ptr);
      auto &DLI = DAR.DLI[DE->Parent->ID];
      DSA_CHECK(DE->Parent->ID >= 0 && DE->Parent->ID < (int) DAR.DLI.size())
        << DAR.DLI.size() << " " << DE->Parent->ID;
      DSA_LOG(AFFINE) << *Ptr << " " << *Raw;
      auto *Res = analyzeIndexExpr(&SE, Raw, DE, DLI.LoopNest, DLI.TripCount, DAR.LinearOverride);
      SW = Res;
      DSA_LOG(AFFINE) << DE->dump() << ": " << Res->toString();
    }
    SEWrapper *SW{nullptr};
    DFGAnalysisResult &DAR;
    ScalarEvolution &SE;
    AffineExtractor(DFGAnalysisResult &DAR, ScalarEvolution &SE) : DAR(DAR), SE(SE) {}
  };

  auto &AI = AffineInfoCache;
  // If it is already analyzed, return the result.
  auto Iter = AI.find(DE);
  if (Iter != AI.end()) {
    return Iter->second;
  }
  // If not, analyze it.
  AffineExtractor AE(*this, SE);
  DE->accept(&AE);
  // TODO(@were): Revisit this later.
  // I did not remember why I allow null result.
  // It seems all the call sites do not allow null except memory coalescing
  // analysis.
  if (!AllowNull) {
    DSA_CHECK(AE.SW) << DE->dump();
  }
  if (AE.SW) {
    AI[DE] = AE.SW;
  }
  return AE.SW;
}

template<typename T>
void gatherMemoryCoalescingImpl(std::vector<DFGEntry*> &Entries, ScalarEvolution &SE,
                                std::vector<int> &DSU, DFGAnalysisResult &DAR) {
  bool Iterative = true;
  while (Iterative) {
    Iterative = false;
    for (int i = 0; i < (int) Entries.size(); ++i) { // NOLINT
      if (auto *A = dyn_cast<T>(Entries[i])) {
        for (int j = i + 1; j < (int) Entries.size(); ++j) { // NOLINT
          if (auto *B = dyn_cast<T>(Entries[j])) {
            auto PtrA = A->underlyingInst()->getOperand(T::PtrOperandIdx);
            auto PtrB = B->underlyingInst()->getOperand(T::PtrOperandIdx);
            if (PtrA->getType() != PtrB->getType()) {
              continue;
            }
            auto Offset = indexPairOffset(DAR.affineMemoryAccess(Entries[i], SE, false),
                                          DAR.affineMemoryAccess(Entries[j], SE, false), SE, false);
            if (Offset == -1) {
              continue;
            }
            int DBits = PtrA->getType()->getPointerElementType()->getScalarSizeInBits();
            DSA_LOG(SLP) << *PtrA << " - " << *PtrB << ": " << Offset;
            // If we find two memory access, whose offset is the same as their
            // data types, coalsece them. We iteratively coalsece them and
            // use a DSU to represent the result of coalescing.
            //
            // DSU is disjoint set union. At the very beginning, each node
            // belongs to their own set and each set has transitive
            // unioned. Say `a[i+0]` and `a[i+1]` are coalesced, and `a[i+1]`
            // and `a[i+2]` are coalesced, then `a[i+0]` and `a[i+2]` should
            // also be coalesced. DSU does this easily.
            if (Offset == DBits / 8) {
              DSA_LOG(SLP) << "Coalesce!";
              if (utils::DSUGetSet(i, DSU) != utils::DSUGetSet(j, DSU)) {
                DSU[utils::DSUGetSet(i, DSU)] = DSU[utils::DSUGetSet(j, DSU)];
                Iterative = true;
              }
            }
          }
        }
      }
    }
  }
}

template<typename TSLP, typename TEntry>
void constructCoalescedMemory(DFGBase *DB, std::vector<DFGEntry*> &Entries,
                              std::vector<int> &DSU, std::vector<int> &DSize,
                              ScalarEvolution &SE, DFGAnalysisResult &DAR,
                              std::vector<int> &Remove) {
  // For each DSU element, if dsu[i] == i, it is the master of this DSU.
  // Therefore, we use this node to construct coalsece.
  for (int j = 0; j < (int) Entries.size(); ++j) { // NOLINT
    if (auto *Entry = dyn_cast<TEntry>(Entries[j])) {
      int DSet = utils::DSUGetSet(j, DSU);
      if (DSet == j && DSize[DSet] != 1) {
        auto *Res = new TSLP(DB);
        Res->Coal.push_back(Entry);
        Entries[j] = Res;
      }
    }
  }
  // If we recursively climb up the DSU tree, we will find the master of
  // each set, and coalsece this node to the mater.
  for (int j = 0; j < (int) Entries.size(); ++j) { // NOLINT
    if (auto *Entry = dyn_cast<TEntry>(Entries[j])) {
      int DSet = utils::DSUGetSet(j, DSU);
      if (DSet != j && DSize[DSet] != 1) {
        auto *SLP = dyn_cast<TSLP>(Entries[DSet]);
        DSA_CHECK(SLP);
        DSA_LOG(SLP) << DSet << "(" << DSize[DSet] << "): " << Entry->dump();
        SLP->Coal.push_back(Entry);
        Remove.push_back(j);
      }
    }
  }
  for (int i = 0; i < (int) Entries.size(); ++i) { // NOLINT
    if (auto *Entry = dyn_cast<TSLP>(Entries[i])) {
      auto &V = Entry->Coal;
      auto FCmp = [&V, &SE, &DAR](TEntry *A, TEntry *B) {
        auto *SW0 = DAR.affineMemoryAccess(V[0], SE, false);
        auto KeyA = indexPairOffset(SW0, DAR.affineMemoryAccess(A, SE, false), SE, true);
        auto KeyB = indexPairOffset(SW0, DAR.affineMemoryAccess(B, SE, false), SE, true);
        return KeyA < KeyB;
      };
      // This is to sort the coalseced memory accesses.
      // Say we have a[i+1], a[i+0], a[i+3], a[i+2],
      // they should be represented in the order of 0,1,2,3.
      std::sort(V.begin(), V.end(), FCmp);
      for (int j = 0; j < (int) V.size(); ++j) { // NOLINT
        V[j]->ID = j;
      }
      auto *Raw = DAR.affineMemoryAccess(Entry->Coal[0], SE, false);
      LinearCombine *L = nullptr;
      if (isa<LinearCombine>(Raw)) {
        L = dyn_cast<LinearCombine>(Raw);
      } else if (auto LI = dyn_cast<LoopInvariant>(Raw)) {
        L = LI->toLinearCombine(&SE);
      }
      DSA_CHECK(L);
      // Append another dimension to the linear combine.
      // Say before, we have a[i*4+0], a[i*4+1], a[i*4+2], a[i*4+3],
      // For the first one, the memory access analyzed is
      // (a+0) + (i*4) where `i` goes n/4 iterations.
      // Now, we append a fake `j` after `i`, and the memory access
      // (a+0) + (i*4) + j, where `i` stays the same, and `j` goes 4 iterations.
      L = new LinearCombine(*L);
      auto Ptr = Entry->Coal[0]->underlyingInst()->getOperand(TEntry::PtrOperandIdx);
      int DBits = Ptr->getType()->getPointerElementType()->getScalarSizeInBits();
      auto &TripCount = L->TripCount;
      const auto *ExtraTC = SE.getConstant(APInt(64, Entry->Coal.size(), true));
      TripCount.insert(TripCount.begin(), LoopInvariant::get(Entry, ExtraTC));
      auto &Coef = L->Coef;
      const auto *ExtraCoef = SE.getConstant(APInt(64, DBits / 8, true));
      Coef.insert(L->Coef.begin(), LoopInvariant::get(Entry, ExtraCoef));
      // If it used to be a constant, we need to make the higher dimensions to be zero-padded.
      if (Coef.size() == 1) {
        auto *SCEVZero = SE.getConstant(APInt(64, 0, true));
        auto *ZeroCoef = LoopInvariant::get(Entry, SCEVZero);
        Coef.reserve(TripCount.size());
        while (Coef.size() < TripCount.size()) {
          Coef.push_back(ZeroCoef);
        }
      }
      DAR.overrideStream(Entry, L);
    }
  }
}

void gatherMemoryCoalescing(DFGFile &DF, ScalarEvolution &SE, DFGAnalysisResult &DAR) {
  auto DFGs = DF.DFGFilter<DFGBase>();
  DSA_CHECK(DAR.DLI.size() == DFGs.size());
  for (int i = 0; i < (int) DFGs.size(); ++i) { // NOLINT
    auto &Entries = DFGs[i]->Entries;
    std::vector<int> DSU;
    DSU.resize(Entries.size());
    for (int j = 0; j < (int) Entries.size(); ++j) { // NOLINT
      DSU[j] = j;
    }
    // Analyze the memory accesses that should be coalesced.
    gatherMemoryCoalescingImpl<MemPort>(Entries, SE, DSU, DAR);
    gatherMemoryCoalescingImpl<PortMem>(Entries, SE, DSU, DAR);
    // TODO(@were): Make too wide port splitted later.
    std::vector<int> DSize(Entries.size(), 0);
    for (int j = 0; j < (int) Entries.size(); ++j) { // NOLINT
      ++DSize[utils::DSUGetSet(j, DSU)];
    }
    std::vector<int> Remove;
    // Coalesce the analyzed accesses.
    constructCoalescedMemory<SLPMemPort, MemPort>(DFGs[i], Entries, DSU, DSize, SE, DAR, Remove);
    constructCoalescedMemory<SLPPortMem, PortMem>(DFGs[i], Entries, DSU, DSize, SE, DAR, Remove);
    std::sort(Remove.begin(), Remove.end());
    for (int j = 0; j < (int) Remove.size(); ++j) { // NOLINT
      Entries.erase(Entries.begin() + Remove[j] - j);
    }
    for (int j = 0; j < (int) Entries.size(); ++j) { // NOLINT
      Entries[j]->ID = j;
    }
  }
}

bool SpadInfo::isSpad(Value *Ptr) {
  auto Cond = [this](Value *Val) -> bool {
    if (auto *AI = dyn_cast<AllocaInst>(Val)) {
      return Offset.count(AI);
    }
    return false;
  };
  if (Cond(Ptr)) {
    return true;
  }
  if (auto *Inst = dyn_cast<Instruction>(Ptr)) {
    std::queue<Instruction*> Q;
    std::set<Instruction*> Visited;
    Q.push(Inst);
    Visited.insert(Inst);
    while (!Q.empty()) {
      auto *Cur = Q.front();
      Q.pop();
      for (auto &Operand: Cur->operands()) {
        if (auto *Prev = dyn_cast<Instruction>(Operand)) {
          if (Cond(Prev)) {
            return true;
          }
          if (!Visited.count(Prev)) {
            Q.push(Prev);
            Visited.insert(Prev);
          }
        }
      }
    }
  }
  return false;
}

void analyzeAccumulatorTags(DFGFile &DF, xform::CodeGenContext &CGC,
                            DFGAnalysisResult &DAR) {
  auto Graphs = DF.DFGFilter<DFGBase>();
  for (int i = 0; i < (int) Graphs.size(); ++i) { // NOLINT
    auto &Loops = DAR.DLI[i].LoopNest;
    std::unordered_map<DFGEntry*, int> Cnt;
    if (auto *DD = dyn_cast<DedicatedDFG>(Graphs[i])) {
      for (auto *Acc : DD->EntryFilter<Accumulator>()) {
        // Insert it into DAR.
        DAR.AI.operator[](Acc);
        // For an accumulator there are two key loop levels involved:
        // for i in 0..n
        //   acc = 0 // reset level
        //   for j in 0..n
        //     acc += ...
        //   use acc // produce level
        // In this case, `acc` is reset when entering loop `j`,
        // and produced when exiting loop `j`.
        //
        // This pass first finds the loop produce and reset the accumulator.
        {
          std::set<Instruction*> Insts;
          FindEquivPHIs(Acc->Operation, Insts);
          for (auto *Elem : Insts) { // NOLINT
            if (auto *Phi = dyn_cast<PHINode>(Elem)) {
              for (int j = 0; j < (int) Phi->getNumOperands(); ++j) { // NOLINT
                auto *Val = dyn_cast<Value>(Phi->getOperand(j));
                if (/*Loops[0]->isLoopInvariant(Val)*/ isa<ConstantData>(Val)) {
                  auto *BB = Phi->getIncomingBlock(j);
                  DSA_LOG(ACC)
                    << Acc->dump() << " reset by block " << BB->getName() << ", " << *Val;
                  for (int k = 0; k < (int) Loops.size(); ++k) { // NOLINT
                    if (Loops[k]->contains(BB)) {
                      DSA_CHECK(k);
                      if (DAR.AI[Acc].ResetLevel == -1) {
                        DAR.AI[Acc].ResetLevel = k - 1;
                      } else {
                        DAR.AI[Acc].ResetLevel = std::max(k - 1, DAR.AI[Acc].ResetLevel);
                        // DSA_CHECK(DAR.AI[Acc].ResetLevel == k - 1)
                        //  << *Loops[k] << ", " << DAR.AI[Acc].ResetLevel << " != " << (k - 1);
                      }
                      break;
                    }
                  }
                }
              }
            }
          }
          if (DAR.AI[Acc].ResetLevel == -1) {
            DAR.AI[Acc].ResetLevel = Loops.size() - 1;
            DSA_LOG(ACC) << Loops.size() << " reset when exiting!";
          }
        }
        // Then this pass uses a BFS to find a memory access that is used
        // throughout the expression that is used to accumulate,
        // associate this memory access with this accumulator.
        //
        // If the memory access already has an accumulator, the compiler
        // tends to associate more accumulators with it.
        auto V = analysis::bfsOperands(DD, Acc->Operation, nullptr).first;
        InputPort *Tag = nullptr;
        auto F = [&Tag, &Acc] (InputPort *IP) {
          if (!Tag) {
            IP->Tagged.push_back(Acc);
            Tag = IP;
          } else if (IP->Tagged.size() >= Tag->Tagged.size()) {
            DSA_CHECK(Acc == Tag->Tagged.back());
            Tag->Tagged.pop_back();
            IP->Tagged.push_back(Acc);
          }
        };
        for (auto &Elem : V) {
          auto *DE = Acc->Parent->InThisDFG(Elem);
          if (!DE) {
            continue;
          }
          if (auto *IP = dyn_cast<InputPort>(DE)) {
            auto *SW = DAR.affineMemoryAccess(IP, CGC.SE, true);
            if (SW) {
              if (auto *IMP = dyn_cast<IndMemPort>(IP)) {
                auto *DS = extractStreamIntrinsics(IMP, CGC, DAR);
                if (auto *Ind1D = dyn_cast<xform::Indirect1D>(DS)) {
                  auto *IP = dyn_cast<IndirectPointer>(Ind1D->Idx);
                  DSA_CHECK(IP) << IP->toString();
                  auto *LC = dyn_cast<LinearCombine>(IP->Pointer);
                  DSA_CHECK(LC) << Ind1D->Idx->toString();
                  if (LC->partialInvariant() == 0) {
                    F(IMP);
                  }
                } else if (isa<xform::Indirect2D>(DS)) {
                  F(IMP);
                }
              } else if (auto *LC = dyn_cast<LinearCombine>(SW)) {
                if (LC->partialInvariant() == 0) {
                  F(IP);
                }
              }
            }
          }
        }
        // This finds the produce level. The produce level means the accumulated
        // value is finalized and used at that level.
        if (Tag) {
          for (auto *OP : Acc->Parent->EntryFilter<OutputPort>()) {
            if (!OP->Output) continue;
            auto Upstream = bfsOperands(OP->Parent, OP->Output, CGC.DT).first;
            for (auto *Elem : Upstream) {
              if (auto *DE = OP->Parent->InThisDFG(Elem)) {
                if (DE == Acc) {
                  auto *Consumer = OP->underlyingInsts()[0];
                  // ...
                  if (Consumer == Acc->Operation) {
                    continue;
                  }
                  for (int j = 0; j < (int) Loops.size(); ++j) { // NOLINT
                    if (Loops[j]->contains(Consumer)) {
                      DSA_LOG(ACC) << Acc->dump() << " Produce@" << (j - 1);
                      if (DAR.AI[Acc].ProduceLevel == INT_MAX) {
                        DAR.AI[Acc].ProduceLevel = j - 1;
                      } else {
                        DSA_CHECK(DAR.AI[Acc].ProduceLevel == j - 1)
                          << DAR.AI[Acc].ProduceLevel << " != " << (j - 1);
                      }
                      break;
                    }
                  }
                }
              }
            }
          }
          // @{
          for (auto *User : Acc->Operation->users()) {
            if (auto *Inst = dyn_cast<Instruction>(User)) {
              bool Skip = isa<PHINode>(Inst);
              for (int j = 0; j < (int) Acc->Operation->getNumOperands() && !Skip; ++j) { // NOLINT
                if (auto *OpInst = dyn_cast<Instruction>(Acc->Operation->getOperand(j))) {
                  if (OpInst == Inst) {
                    Skip = true;
                  }
                }
              }
              if (!Skip) { DSA_LOG(ACC) << "Used by: " << *Inst;
                for (int j = 0; j < (int) Loops.size(); ++j) { // NOLINT
                  if (Loops[j]->contains(Inst)) {
                    DSA_LOG(ACC) << Acc->dump() << " Produce@" << j - 1;
                    if (DAR.AI[Acc].ProduceLevel == INT_MAX) {
                      DAR.AI[Acc].ProduceLevel = j - 1;
                    } else {
                      DSA_CHECK(DAR.AI[Acc].ProduceLevel == j - 1) <<
                        DAR.AI[Acc].ProduceLevel << " != " << (j - 1);
                    }
                    break;
                  }
                }
              } else {
                DSA_LOG(ACC) << "Skip: " << *Inst;
              }
            }
          }
          // }
          if (DAR.AI[Acc].ProduceLevel == INT_MAX) {
            DAR.AI[Acc].ProduceLevel = Loops.size() - 1;
            DSA_LOG(ACC) << Loops.size() << " produce when exiting!";
          }
        }
      }
    }
    // If the associated memory access is a coalesced memory access, we need
    // to add the level of reset and produce by 1, because of
    // the extra dimension appended.
    //
    // Similar thing also happens to dimension fusion. The fused dimension
    // cannot be higher than the tag level. If fused, we lose the information
    // of leaving and entering a loop. (2)
    for (auto *IP : Graphs[i]->EntryFilter<InputPort>()) {
      if (!IP->Tagged.empty()) {
        DSA_LOG(ACC) << IP->dump(); 
        // TODO(@were): Enable this redundancy later.
        // auto *LC = dyn_cast<LinearCombine>(Iter->second);
        // DSA_CHECK(LC);
        for (auto *Acc : IP->Tagged) {
          auto &Entry = DAR.AI[Acc];
          if (isa<SLPMemPort>(IP)) {
            ++Entry.ResetLevel;
            ++Entry.ProduceLevel;
          }
          // DSA_CHECK(Acc->ResetLevel < (int) LC->Coef.size());
          DSA_LOG(ACC)
            << Acc->dump() << ", Reset@" << Entry.ResetLevel
            << ", Produce@" << Entry.ProduceLevel;
        }
        DSA_LOG(ACC) << "Finalized reset: " << resetLevel(IP, DAR);
      }
    }
  }
}

int cutoffLevel(InputPort *IP, DFGAnalysisResult &DAR) {
  // As mentioned at (2), The fused dimension cannot be higher than
  // the tag level. If fused, we lose the information of
  // leaving and entering a loop.

  auto FRed = [&DAR](int X, Accumulator *A) {
    return std::min(X, std::min(DAR.AI[A].ResetLevel, DAR.AI[A].ProduceLevel));
  };
  if (IP->Tagged.empty()) {
    return -1;
  }
  return std::accumulate(IP->Tagged.begin(), IP->Tagged.end(), INT_MAX, FRed);
}


void DFGAnalysisResult::fuseAffineDimensions(ScalarEvolution &SE) {
  struct FuseInfoExtractor : DFGEntryVisitor {
    void Visit(MemPort *MP) override {
      DBits = MP->Load->getType()->getScalarSizeInBits();
      Padding = MP->fillMode();
      CutOff = cutoffLevel(MP, DAR);
    }
    void Visit(PortMem *PM) override {
      DBits = PM->Store->getValueOperand()->getType()->getScalarSizeInBits();
    }
    void Visit(SLPMemPort *SMP) override {
      auto *MP = SMP->Coal[0];
      DBits = MP->Load->getType()->getScalarSizeInBits();
      CutOff = cutoffLevel(SMP, DAR);
      SLP = true;

      // TODO(@were): Revisit this later. I did not remember the exact workload.
      // This is to handle such a case:
      // for i in 0..n
      //   for j in 0..m
      //     for k is coalseced (say 0..4)
      //       access a[i,k]
      // In this case, we are essentially fusing dimension `i` and `k`.
      // Therefore, here we reorder `j` and `k`.
      if (const auto *LI0 = dyn_cast<LoopInvariant>(LC->TripCount[0])) {
        if (auto *PortWidth = LI0->constInt()) {
          DSA_CHECK(*PortWidth == SMP->Coal.size()); // Port is this wide.
          if (const auto *LI1 = dyn_cast<LoopInvariant>(LC->Coef[1])) {
            if (auto *Stride = LI1->constInt()) {
              if (*Stride == 0) { // Outer stride is 0, which is a repeat.
                std::swap(LC->Coef[0], LC->Coef[1]);
                std::swap(LC->TripCount[0], LC->TripCount[1]);
              }
            }
          }
        }
      }
    }
    void Visit(SLPPortMem *SPM) override {
      auto *PM = SPM->Coal[0];
      DBits = PM->Store->getValueOperand()->getType()->getScalarSizeInBits();
      SLP = true;
    }
    int DBits{-1};
    int Padding{0};
    int CutOff{-1};
    bool SLP{false};
    DFGAnalysisResult &DAR;
    LinearCombine *LC;
    FuseInfoExtractor(DFGAnalysisResult &DAR, LinearCombine *LC) : DAR(DAR), LC(LC) {}
  };
  for (auto &Elem : AffineInfoCache) {
    if (auto *Pattern = dyn_cast<LinearCombine>(Elem.second)) {
      FuseInfoExtractor FIE(*this, Pattern);
      Elem.first->accept(&FIE);
      if (FIE.CutOff == -1) {
        FIE.CutOff = Pattern->TripCount.size() - 1;
      }
      DSA_CHECK(FIE.DBits != -1);
      DSA_LOG(FUSE) << Pattern << " " << Elem.first->dump();
      DSA_LOG(FUSE) << "Before: " << Pattern->toString() << ", CutOff: " << FIE.CutOff;
      int Before = Pattern->Coef.size();
      fuseInnerDimensions(*Pattern, FIE.DBits / 8, Elem.first->Parent->getUnroll(), SE,
                          Pattern->TripCount.size() - FIE.CutOff, FIE.SLP);
      DSA_LOG(FUSE) << "After: " << Pattern->toString();
      if (auto *IP = dyn_cast<InputPort>(Elem.first)) {
        int Delta = Before - Pattern->Coef.size();
        DSA_LOG(FUSE) << "Fuse Delta: " << Delta;
        DSA_LOG(FUSE) << IP->dump();
        // Fix the accumulator signal associated with this port.
        // After inner dimensions are gone, the reset level and produce level
        // should alos be decreased. We adopt a from-inner-to-outer order.
        if (!IP->Tagged.empty()) {
          for (auto *Acc : IP->Tagged) {
            auto &Entry = AI[Acc];
            Entry.ResetLevel -= Delta;
            Entry.ProduceLevel -= Delta;
            DSA_LOG(FUSE) << Acc->dump() << ": " << Entry.ResetLevel;
          }
        } else if (auto *SMP = dyn_cast<SLPMemPort>(IP)) {
          std::ostringstream OSS;
          SMP->dump(OSS);
          // An peephole optimization for codegen.
          if (Pattern->Coef.size() > 1 && SMP->Parent->getUnroll() == 1 /*Unroll not supported yet!*/) {
            // Inner most dimension is not fused.
            if (auto *CoefConst = dyn_cast<SCEVConstant>(Pattern->Coef[0]->Raw)) {
              int DBytes = SMP->underlyingInsts()[0]->getType()->getScalarSizeInBits() / 8;
              // The inner most dimension has the same stride as the data type.
              if (CoefConst->getAPInt().getSExtValue() == DBytes) {
                if (auto *TripConst = dyn_cast<SCEVConstant>(Pattern->TripCount[0]->Raw)) {
                  // The inner most dimension has the same lanes as the manual unrolling degree.
                  if (TripConst->getAPInt().getSExtValue() == (int64_t) SMP->Coal.size()) {
                    // Outer dimension is repeat.
                    if (auto *CoefOuter = dyn_cast<SCEVConstant>(Pattern->Coef[1]->Raw)) {
                      // Outer dimension coef is 0.
                      if (CoefOuter->getAPInt().getSExtValue() == 0) {
                        std::swap(Pattern->Coef[0], Pattern->Coef[1]);
                        std::swap(Pattern->TripCount[0], Pattern->TripCount[1]);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

int resetLevel(InputPort *IP, DFGAnalysisResult &DAR) {
  if (IP->Tagged.empty()) {
    return -1;
  }
  auto FRed = [&DAR](int X, Accumulator *A) { return std::min(X, DAR.AI[A].ResetLevel); };
  return std::accumulate(IP->Tagged.begin(), IP->Tagged.end(), INT_MAX, FRed);
}

const SCEV *productTripCount(std::vector<analysis::SEWrapper*> &TC, ScalarEvolution *SE) {
  const SCEV *Res = SE->getConstant(APInt(64, 1, true));
  for (int i = 0, N = TC.size(); i < N; ++i) { // NOLINT
    Res = SE->getMulExpr(Res, TC[i]->Raw);
  }
  return Res;
}

xform::DataStream *
extractStreamIntrinsics(DFGEntry *DE, xform::CodeGenContext &CGC, DFGAnalysisResult &DAR) {
  struct StreamExtractor : DFGEntryVisitor {
    void Visit(AtomicPortMem *APM) override {
      auto *SW = DAR.affineMemoryAccess(APM, CGC.SE, false);
      int DType = APM->Store->getValueOperand()->getType()->getScalarSizeInBits() / 8;
      analyze(SW, DType, APM);
    }
    void Visit(IndMemPort *IMP) override {
      auto *SW = DAR.affineMemoryAccess(IMP, CGC.SE, false);
      int DType = IMP->Load->getType()->getScalarSizeInBits() / 8;
      analyze(SW, DType, IMP);
    }
    void analyze(SEWrapper *IdxLI, int DType, DFGEntry *DE) {
      int IType = 8;
      analysis::SEWrapper *Idx = nullptr;
      if (auto *SA = dyn_cast<analysis::SWBinary>(IdxLI)) {
        DSA_CHECK(SA && SA->Op == 0) << IdxLI->toString();
        Idx = SA->A;
        DSA_CHECK(isa<LoopInvariant>(SA->B));
        const SCEV *SAR = SA->B->Raw;
        // Strip the Coef
        if (DType == 1) {
          DSA_CHECK(false) << Idx->toString();
        } else {
          auto *SM = dyn_cast<analysis::SWBinary>(Idx);
          DSA_CHECK(SM);
          if (const auto *SC = dyn_cast<SCEVConstant>(SM->A->Raw)) {
            DSA_CHECK(SC->getAPInt().getSExtValue() == DType);
          } else {
            DSA_CHECK(false);
          }
          Idx = SM->B;
        }
        if (const auto *SCE = dyn_cast<analysis::SWCast>(Idx)) {
          IType = SCE->srcTy()->getScalarSizeInBits() / 8;
          Idx = SCE->Stripped;
        }
        auto *IP = dyn_cast<IndirectPointer>(Idx);
        DSA_CHECK(IP) << Idx->toString() << "\n" << SA->toString();
        auto *LC = dyn_cast<LinearCombine>(IP->Pointer);
        DSA_CHECK(LC) << IdxLI->toString() << "\n" << Idx->toString();
        auto *L1D = productTripCount(LC->TripCount, &CGC.SE);
        DS = new xform::Indirect1D(DType, IType, SAR, L1D, Idx);
      } else if (auto *LC = dyn_cast<analysis::LinearCombine>(IdxLI)) {
        auto *Add = dyn_cast<analysis::SWBinary>(LC->Base);
        DSA_CHECK(Add) << LC->Base->toString();
        auto *SAR = dyn_cast<analysis::LoopInvariant>(Add->B);
        auto *Mul = dyn_cast<analysis::SWBinary>(Add->A);
        DSA_CHECK(Mul) << Add->A->toString();
        auto *IndPtr = dyn_cast<analysis::IndirectPointer>(Mul->B);
        DSA_CHECK(IndPtr) << Mul->B->toString();
        DS = new xform::Indirect2D(DType, IType, SAR, IndPtr->Pointer, LC->TripCount[0],
                                   SAR->TripCount[0]->Raw);
      }
    }

    xform::DataStream *DS{nullptr};

    xform::CodeGenContext &CGC;
    DFGAnalysisResult &DAR;
    StreamExtractor(xform::CodeGenContext &CGC, DFGAnalysisResult &DAR) :
      CGC(CGC), DAR(DAR) {}
  };
  auto Iter = DAR.Streams.find(DE);
  if (Iter != DAR.Streams.end()) {
    return Iter->second;
  }
  StreamExtractor SE(CGC, DAR);
  DE->accept(&SE);
  // DSA_CHECK(SE.DS);
  DAR.Streams[DE] = SE.DS;
  return SE.DS;
}

DFGUnroll::DFGUnroll(DFGFile &DF, xform::CodeGenContext &CGC) : DF(DF) {
  // dsa::ContextFlags::Global().adg_compat = dsa::utils::ModuleContext().COMPAT_ADG;
  auto *SBCONFIG = getenv("SBCONFIG");
  DSA_CHECK(SBCONFIG);
  // SSModel Model(SBCONFIG);
  auto *BFI = CGC.BFI;
  for (auto *Elem : DF.DFGs) {
    Idx.push_back(0);
    Freq.push_back(0);
    if (Elem->getUnroll() == -1) {
      Degrees.push_back({8, 4, 2, 1});
      Explored.push_back(true);
      auto *DD = dyn_cast<DedicatedDFG>(Elem);
      for (auto *BB : DD->InnerMost()->getBlocks()) {
        Freq.back() += BFI->getBlockFreq(BB).getMaxFrequency();
      }
    } else {
      Explored.push_back(false);
      Degrees.push_back({Elem->getUnroll()});
    }
  }
}

bool DFGUnroll::hasNext() {
  DSA_CHECK(!Idx.empty());
  return Idx.back() < (int) Degrees.back().size();
}

std::string DFGUnroll::toString() {
  std::ostringstream OSS;
  for (int i = 0; i < (int) Idx.size(); ++i) { // NOLINT
    OSS << "_" << Degrees[i][Idx[i]];
  }
  return OSS.str();
}

bool DFGUnroll::next(bool Apply) {
  DSA_CHECK(hasNext());
  do {
    ++Cnt;
    auto &DFGs = DF.DFGs;
    bool Carry = true;
    DF.FileName = DF.Func.getName().str() + "_" + std::to_string(DF.ID) + toString() + ".dfg";
    for (int i = 0; i < (int) DFGs.size(); ++i) { // NOLINT
      auto *DFG = DF.DFGs[i];
      if (auto *DD = dyn_cast<DedicatedDFG>(DFG)) {
        if (Apply) {
          DD->UnrollFactor = Degrees[i][Idx[i]];
        }
        if (Carry) {
          ++Idx[i];
          Carry = false;
        }
        if (Idx[i] == (int) Degrees[i].size()) {
          Idx[i] = 0;
          Carry = true;
        }
      }
    }
    if (Carry) {
      Idx.back() = Degrees.back().size();
    }
    bool Good = true;
    for (int i = 0; i < (int) DFGs.size() && Good; ++i) { // NOLINT
      if (auto *DDI = dyn_cast<DedicatedDFG>(DFGs[i])) {
        if (!Explored[i]) {
          continue;
        }
        for (int j = 0; j < (int) DFGs.size() && Good; ++j) { // NOLINT
          if (!Explored[j]) {
            continue;
          }
          if (i == j) {
            continue;
          }
          if (auto *DDJ = dyn_cast<DedicatedDFG>(DFGs[j])) {
            if (Freq[i] <= Freq[j] && DDI->UnrollFactor > DDJ->UnrollFactor) {
              Good = false;
            }
          }
        }
      }
    }
    if (!Good) {
      ++Skip;
      continue;
    }
    DAR.emplace_back();
    return true;
  } while (hasNext());
  return false;
}

DFGAnalysisResult &DFGUnroll::best() {
  int Best = 0;
  for (int i = 1; i < (int) DAR.size(); ++i) { // NOLINT
    if (DAR[Best].CI.EstimatedILP < DAR[i].CI.EstimatedILP) {
      Best = i;
    }
  }
  return DAR[Best];
}

std::unordered_map<const SCEV*, const SCEV*>
gatherLinearOverride(IntrinsicInst *Start, IntrinsicInst *End, xform::CodeGenContext &CGC) {
  std::unordered_map<const SCEV*, const SCEV*> Res;
  std::vector<IntrinsicInst*> ToRemove;
  // Consider the loop nest below, it will generate a[0:n] instead of a[0:m].
  //
  // #prgma ss dfg fifo(a:n)
  // for (int i = 0; i < m; ++i) {
  //   // use a[i]
  // }
  //
  // The clang frontend will encode and embed this hint in a ss_fifo intrinsic.
  // This function below is to extract these hints.
  // The hints will be put into a <k,v> map.
  // The key is the array pointer, i.e. a in this case,
  // and the value is the size of the array, i.e. n in this case.
  auto F = [&Res, &ToRemove, &CGC](Instruction *I) {
    auto &SE = CGC.SE;
    // The parser already encodes 
    if (auto *II = dyn_cast<IntrinsicInst>(I)) {
      // If it is ss_fifo, gather the hint.
      if (II->getIntrinsicID() == Intrinsic::ss_fifo) {
        Value *Stripped = II->getOperand(0);
        if (auto *BCI = dyn_cast<BitCastInst>(Stripped)) {
          Stripped = BCI->getOperand(0);
        }
        const auto *Ky = SE.getSCEV(Stripped);
        const auto *V = SE.getSCEV(II->getOperand(1));
        Res[Ky] = V;
        ToRemove.push_back(II);
      }
    }
  };
  // This function designates the scope of the instructions to be traversed.
  // All the instructions between the config.start/end.
  traverseAndApply(Start, End, CGC.DT, F);
  // Remove all the hint intrinsics.
  for (auto *Elem : ToRemove) {
    Elem->eraseFromParent();
  }
  return Res;
}


float getPortDataTraffic(LinearCombine *LC, float PortDType) {
  float Traffic = 1.0f;
  for (auto *TC : LC->TripCount) { //NOLINT 
    if (auto *SCEVTripCount = dyn_cast<SCEVConstant>(TC->Raw)) {
      Traffic *= SCEVTripCount->getAPInt().getSExtValue();
    }
    else {
      Traffic = 0.0f;
      break;
    }
  }
  return Traffic * PortDType;
}


float getReuseTrafficAtLevel(LinearCombine *LC, int Lvl, float BaseStream) {

  float ReuseTraffic = 0.0f;
  if (auto *SCEVStride = dyn_cast<SCEVConstant>(LC->Coef[Lvl]->Raw))
    if (auto *SCEVTripCount = dyn_cast<SCEVConstant>(LC->TripCount[Lvl]->Raw)) {
      float Stride = static_cast <float> (SCEVStride->getAPInt().getSExtValue());
      float TripCount = static_cast <float> (SCEVTripCount->getAPInt().getSExtValue());
      if (Stride == 0.0f)
        ReuseTraffic = BaseStream;
      else if (Stride < BaseStream)
        ReuseTraffic = Stride * (TripCount - 1) + BaseStream;
      else
        ReuseTraffic = BaseStream * TripCount;
    }
  return ReuseTraffic;
}


std::pair<float, float> getPortReuse(DFGEntry *DE, xform::CodeGenContext &CGC, DFGAnalysisResult &DAR, float PortDType) { 
  std::pair<float, float> RRAT(0.0f, 0.0f);
  SEWrapper *SW = DAR.affineMemoryAccess(DE, CGC.SE, false);
  if (auto *LC = dyn_cast<LinearCombine>(SW)) {
    DSA_LOG(ARRAY_HINT) << LC->toString();
    float PortDataTraffic = getPortDataTraffic(LC, PortDType);
    if (PortDataTraffic == 0.0f) {
      return RRAT;
    }
    if (auto *SCEVInnerMostStride = dyn_cast<SCEVConstant>(LC->Coef[0]->Raw)) {
      float InnerMostStride = static_cast <float> (SCEVInnerMostStride->getAPInt().getSExtValue());
      if (InnerMostStride != PortDType) 
        return RRAT;
      if (auto *SCEVInnerMostTripCount = dyn_cast<SCEVConstant>(LC->TripCount[0]->Raw)) {
        float InnerMostTripCount = static_cast <float> (SCEVInnerMostTripCount->getAPInt().getSExtValue());
        float BaseStream = InnerMostStride * InnerMostTripCount;
        float ReuseTraffic = BaseStream;
        if (LC->Coef.size() == 1) {
          RRAT.second = ReuseTraffic;
          return RRAT;
        }
        for (unsigned int lvl = 1; lvl < LC->Coef.size(); ++lvl) // NOLINT
          ReuseTraffic = getReuseTrafficAtLevel(LC, lvl, ReuseTraffic);
        RRAT.second = ReuseTraffic;
      }
    }
    RRAT.first = 1.0f - RRAT.second / PortDataTraffic;
  }
  return RRAT;
}

} // namespace analysis
} // namespace dsa
