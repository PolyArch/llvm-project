#include "dsa/debug.h"

#include "./DfgAnalysis.h"

#define DEBUG_TYPE "dfg-analysis"

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


/*!
 * \brief How DFG Entries are analyzed and extracted.
 */
struct DFGEntryAnalyzer : DFGVisitor {
  /*!
   * \brief The DFG to be analyzed.
   */
  DfgBase *DBPtr;
  /*!
   * \brief The result of an analyzer.
   */
  DominatorTree *DT;

  void FindEquivPHIs(Instruction *Inst, std::set<Instruction*> &Equiv) {
    std::queue<Instruction*> Q;
    Q.push(Inst);
    Equiv.insert(Inst);
    while (!Q.empty()) {
      if (auto PHI = dyn_cast<PHINode>(Q.front())) {
        for (auto &Elem : PHI->incoming_values()) {
          auto Casted = dyn_cast<Instruction>(Elem);
          if (!Casted)
            continue;
          if (Equiv.count(Casted))
            continue;
          Q.push(Casted);
          Equiv.insert(Casted);
        }
      }
      for (auto User : Q.front()->users()) {
        auto Phi = dyn_cast<PHINode>(User);
        if (!Phi)
          continue;
        if (Equiv.count(Phi))
          continue;
        Q.push(Phi);
        Equiv.insert(Phi);
      }
      Q.pop();
    }

    LLVM_DEBUG(
      errs() << "equiv of "; Inst->dump();
      for (auto I : Equiv) {
        I->dump();
      }
    );
  }

  Predicate* FindEquivPredicate(Value *LHS, Value *RHS) {
    auto &DB = *DBPtr;
    for (auto Elem : DB.EntryFilter<Predicate>()) {
      if (Elem->Cond[0]->getOperand(0) == LHS && Elem->Cond[0]->getOperand(1) == RHS)
        return Elem;
      if (Elem->Cond[0]->getOperand(1) == LHS && Elem->Cond[0]->getOperand(0) == RHS)
        return Elem;
    }
    return nullptr;
  }

  DfgEntry *DifferentiateMemoryStream(LoadInst *Load) {
    auto &DB = *DBPtr;
    if (auto GEP = dyn_cast<GetElementPtrInst>(Load->getPointerOperand())) {

      auto BFSBack = BFSOperands(GEP);

      auto fLoad = [this, Load, BFSBack, &DB] (Value *Val) -> DfgEntry* {
        LoadInst *IdxLoad = nullptr;
        for (auto Elem : BFSBack.first) {
          if (auto ThisLoad = dyn_cast<LoadInst>(Elem)) {
            assert(!IdxLoad && "For now only one level indirect supported!");
            IdxLoad = ThisLoad;
          }
        }
        if (IdxLoad) {
          return new IndMemPort(&DB, IdxLoad, Load);
        }
        return nullptr;
      };

      auto fGather = [Load, GEP, this, &DB] (Value *Val) -> DfgEntry* {
        auto DD = dyn_cast<DedicatedDfg>(&DB);

        auto Casted = dyn_cast<Instruction>(Val);
        if (!Casted)
          return nullptr;

        if (!DD)
          return nullptr;

        {
          bool simpleLoop = false;
          auto BI = dyn_cast<BranchInst>(&DD->InnerMost()->getLoopLatch()->back());
          auto Latch = DD->InnerMost()->getLoopLatch();
          assert(BI);
          for (auto BB : BI->successors()) {
            if (BB == Latch) {
              simpleLoop = true;
            }
          }
          if (simpleLoop) {
            return nullptr;
          }
        }

        std::set<Instruction*> Equiv;
        LLVM_DEBUG(dbgs() << "Equiv of: ");
        FindEquivPHIs(Casted, Equiv);

        Value *TripCount = nullptr;
        SmallVector<std::pair<ICmpInst*, bool>, 0> Conditions;
        for (auto Elem : Equiv) {
          if (auto PHI = dyn_cast<PHINode>(Elem)) {
            for (auto &Essense : PHI->incoming_values()) {
              auto Inst = dyn_cast<Instruction>(Essense);
              if (Inst && !isa<PHINode>(Inst)) {
                if (Inst->getOpcode() == BinaryOperator::Add) {
                  auto DomBB = DT->getNode(Inst->getParent())->getIDom()->getBlock();
                  LLVM_DEBUG(DomBB->back().dump(); Inst->dump());
                  auto BI = dyn_cast<BranchInst>(&DomBB->back());
                  assert(BI);
                  if (BI->isConditional()) {
                    auto CondCmp = dyn_cast<ICmpInst>(BI->getCondition());
                    bool OK = true;
                    for (size_t i = 0; i < Conditions.size(); ++i) {
                      auto AlreadyCmp = Conditions[i].first;
                      if (CondCmp->getNumOperands() != AlreadyCmp->getNumOperands()) {
                        OK = false;
                        break;
                      }
                      OK = OK && ((AlreadyCmp->getOperand(0) == CondCmp->getOperand(0) &&
                                  AlreadyCmp->getOperand(1) == CondCmp->getOperand(1)) ||
                                  (AlreadyCmp->getOperand(0) == CondCmp->getOperand(1) &&
                                  AlreadyCmp->getOperand(1) == CondCmp->getOperand(0)));
                    }
                    assert(OK && "Controlled by more than one predicate!");
                    Conditions.push_back(std::make_pair(CondCmp,
                                                        Inst->getParent() == BI->getSuccessor(0)));
                  }
                }
              }
            }
          }
          if (DD && Elem->getParent() == DD->InnerMost()->getLoopLatch()) {
            LLVM_DEBUG(dbgs() << "Latch"; Elem->dump());
            for (auto Elem_ : Elem->users()) {
              auto Cmp = dyn_cast<ICmpInst>(Elem_);
              if (!Cmp)
                continue;
              if (Cmp->getParent() != Elem->getParent())
                continue;
              assert(Cmp->getPredicate() == ICmpInst::Predicate::ICMP_SLT &&
                     "For now only support SLT");
              for (size_t i = 0; i < Cmp->getNumOperands(); ++i) {
                if (Cmp->getOperand(i) != Elem) {
                  TripCount = Cmp->getOperand(i);
                  LLVM_DEBUG(dbgs() << "TripCount"; TripCount->dump());
                }
              }
            }
          }
        }
        if (TripCount && !Conditions.empty()) {

          int Mask = 0;
          SmallVector<bool, 0> Reverse;

          auto Pred =
            FindEquivPredicate(Conditions[0].first->getOperand(0), Conditions[0].first->getOperand(1));

          if (!Pred) {
            Pred = new Predicate(&DB, Conditions[0].first);
            DB.Entries.push_back(Pred);
          }

          for (size_t i = 0; i < Conditions.size(); ++i) {
            Reverse.push_back(Pred->addCond(Conditions[i].first));
          }

          for (size_t i = 0; i < Conditions.size(); ++i) {
            auto Cond = Conditions[i];
            LLVM_DEBUG(dbgs() << "Moveforward: " << Cond.second << " "; Cond.first->dump());
            int SubMask = PredicateToInt(Cond.first->getPredicate(), Cond.second, Reverse[i]);
            Mask |= SubMask;
          }

          LLVM_DEBUG(dbgs() << "Forward mask: " << Mask << "\n");
          // FIXME: GEP for not is good enough, but we need a better way to figure out the start
          //        pointer later.
          return new CtrlMemPort(&DB, Load, GEP->getOperand(0), TripCount, Pred, Mask);
        }
        LLVM_DEBUG(dbgs() << "\n");
        return nullptr;
      };

      for (size_t i = 1; i < GEP->getNumOperands(); ++i) {
        if (auto Res = fLoad(GEP->getOperand(i)))
          return Res;
        if (auto Res = fGather(GEP->getOperand(i)))
          return Res;
        if (auto Cast = dyn_cast<CastInst>(GEP->getOperand(i))) {
          for (size_t j = 0; j < Cast->getNumOperands(); ++j) {
            if (auto Res = fGather(Cast->getOperand(j)))
              return Res;
          }
        }
      }
    }
    return new MemPort(&DB, Load);
  }

  void InstsToEntries(iterator_range<BasicBlock::iterator> IRange, std::set<Instruction*> &Visited) {
    auto &DB = *DBPtr;
    std::queue<Instruction*> Q;
    for (auto &Inst : IRange) {

      if (auto Load = dyn_cast<LoadInst>(&Inst)) {
        Q.push(Load);
        Visited.insert(Load);
      }

      // FIXME(@were): Can I delete the Temporal DFG clause?
      // Jian Weng on May 11th, 2021: I think it is safe to remove.
      // if (isa<TemporalDfg>(&DB) && CanBeAEntry(&Inst)){
      //   for (size_t i = 0; i < Inst.getNumOperands(); ++i) {
      //     if (auto InstOperand = dyn_cast<Instruction>(Inst.getOperand(i))) {
      //       if (!DB.Contains(InstOperand)) {
      //         // A exteneral stream-in value
      //         Q.push(&Inst);
      //         Visited.insert(&Inst);
      //       }
      //     }
      //   }
      // }

      while (!Q.empty()) {
        auto Cur = Q.front();
        Q.pop();
        LLVM_DEBUG(dbgs() << "Analyze Users of: "; Cur->dump());
        for (auto Elem : Cur->users()) {
          auto Inst = dyn_cast<Instruction>(Elem);
          if (!Inst) {
            LLVM_DEBUG(dbgs() << "Not a inst: "; Elem->dump());
            continue;
          }
          if (!DB.Contains(Inst)) {
            LLVM_DEBUG(dbgs() << "Not in blocks: "; Elem->dump());
            continue;
          }
          if (!CanBeAEntry(Inst)) {
            LLVM_DEBUG(dbgs() << "Not a entry: "; Elem->dump());
            continue;
          }
          if (Visited.find(Inst) == Visited.end()) {
            Q.push(Inst);
            Visited.insert(Inst);
          }
        }
      }
    }

  }


  void AnalyzeEntryOp(Instruction *Inst) {
    auto &DB = *DBPtr;
    LLVM_DEBUG(dbgs() << "Analyze Entry: "; Inst->dump());

    bool isAcc = false;

    auto ToOffload = !isa<BranchInst>(Inst) ? Inst :
      dyn_cast<Instruction>(dyn_cast<BranchInst>(Inst)->getCondition());

    for (size_t i = 0; i < ToOffload->getNumOperands(); ++i) {
      auto Operand = ToOffload->getOperand(i);
      LLVM_DEBUG(dbgs() << "Operand: "; Operand->dump());
      if (DB.InThisDFG(Operand)) {
        LLVM_DEBUG(dbgs() << "Already-in skip!\n");
        continue;
      }
      if (auto Phi = dyn_cast<PHINode>(Operand)) {

        std::queue<PHINode*> Q;
        std::set<PHINode*> Visited;

        for (Q.push(Phi), Visited.insert(Phi); !Q.empty(); ) {
          auto Poll = Q.front(); Q.pop();
          for (auto &InComing : Poll->incoming_values()) {
            if (InComing == Inst) {
              isAcc = true;
              break;
            }
            if (auto AnotherPhi = dyn_cast<PHINode>(InComing)) {
              if (!Visited.count(AnotherPhi)) {
                Q.push(AnotherPhi);
                Visited.insert(AnotherPhi);
              }
            }
          }
        }

        // FIXME: I am not sure if this will suffice. I relax the constraint
        // of detecting a accumulation by checking the BFS successors.
        if (!isAcc) {
          llvm::errs() << "PhiNode: "; Phi->dump();
          assert(false && "Accumulator check goes wrong.");
        }

      } else if (auto Load = dyn_cast<LoadInst>(Operand)) {
        DB.Entries.push_back(DifferentiateMemoryStream(Load));
      } else if (auto Consumee = dyn_cast<Instruction>(Operand)) {
        LLVM_DEBUG(dbgs() << "Not in this Nest: " << DB.Contains(Consumee) << "\n");
        if (!DB.Contains(Consumee)) {
          if (DB.BelongOtherDFG(Consumee)) {
            DB.Entries.push_back(new StreamInPort(&DB, Consumee));
            LLVM_DEBUG(dbgs() << "Upstream:"; Consumee->dump());
          } else {
            DB.Entries.push_back(new InputConst(&DB, Consumee));
            LLVM_DEBUG(dbgs() << "Outer loop invariant:"; Consumee->dump());
          }
        }
      } else if (!isa<Constant>(Operand)) {
        DB.Entries.push_back(new InputConst(&DB, Operand));
        LLVM_DEBUG(dbgs() << "Other non const:"; Operand->dump());
      }
    }

    DfgEntry *Entry = nullptr;
    if (!isAcc) {
      if (auto BI = dyn_cast<BranchInst>(Inst)) {
        assert(BI->isConditional());
        if (ICmpInst* Cmp = dyn_cast<ICmpInst>(BI->getCondition())) {
          Predicate *PredEntry = FindEquivPredicate(Cmp->getOperand(0), Cmp->getOperand(1));
          if (!PredEntry) {
            PredEntry = new Predicate(&DB, Cmp);
            DB.Entries.push_back(PredEntry);
          } else {
            PredEntry->addCond(Cmp);
          }
        } else if (Instruction *Cond = dyn_cast<Instruction>(BI->getCondition())) {
          DB.Entries.push_back(new Predicate(&DB, Cond));
        }
        Entry = DB.Entries.back();
      } else {
        auto CB = new ComputeBody(&DB, Inst);
        DB.Entries.push_back(CB);
        Entry = CB;
        LLVM_DEBUG(dbgs() << "Plain Inst: "; Inst->dump());
      }
    } else {
      assert(Inst->getOpcode() == BinaryOperator::Add ||
             Inst->getOpcode() == BinaryOperator::FAdd);
      auto CS = new CtrlSignal(&DB);
      DB.Entries.push_back(CS);
      auto Acc = new Accumulator(&DB, Inst, CS);
      DB.Entries.push_back(Acc);
      CS->Controlled = Acc;
      Entry = Acc;
      LLVM_DEBUG(dbgs() << "Accumulator: "; Inst->dump());
    }
    CHECK(Entry);
  }

  void InspectConsumers(Instruction *Inst) {
    auto &DB = *DBPtr;
    auto DD = dyn_cast<DedicatedDfg>(&DB);
    auto &Entries = DB.Entries;

    for (auto User : Inst->users()) {
      if (auto Store = dyn_cast<StoreInst>(User)) {
        LoadInst *Load = nullptr;

        if (auto GEP = dyn_cast<GetElementPtrInst>(Store->getPointerOperand())) {
          for (size_t i = 0; i < GEP->getNumOperands(); ++i) {
            if ((bool) (Load = dyn_cast<LoadInst>(GEP->getOperand(i)))) {
              break;
            }
          }
        }

        if (Load && DD && DB.Contains(Load)) {

          if (auto BO = dyn_cast<BinaryOperator>(Store->getValueOperand())) {
            bool isUpdate = false;
            Value *AtomicOperand = nullptr;
            if (BO->getOpcode() == Instruction::BinaryOps::Add) {
              for (size_t i = 0; i < BO->getNumOperands(); ++i) {
                auto Operand = dyn_cast<LoadInst>(BO->getOperand(i));
                if (Operand && Operand->getPointerOperand() == Store->getPointerOperand()) {
                  isUpdate = true;
                } else {
                  AtomicOperand = BO->getOperand(i);
                }
              }
            }
            Entries.push_back(new AtomicPortMem(&DB, Load, Store, isUpdate ? 0 : 3,
                                                Inst, AtomicOperand));
          } else {
            Entries.push_back(new AtomicPortMem(&DB, Load, Store, 3, Inst, nullptr));
          }
          LLVM_DEBUG(dbgs() << "AtomicMem: "; Inst->dump());

        } else {
          auto Out = new PortMem(&DB, Store);
          Entries.push_back(Out);
          LLVM_DEBUG(dbgs() << "PortMem: "; Inst->dump());
        }
      // Support downstream data consumption
      } else if (CanBeAEntry(User)) {
        auto Consume = dyn_cast<Instruction>(User);
        LLVM_DEBUG(dbgs() << "Comsumed by: "; Consume->dump());
        if (DB.BelongOtherDFG(Consume)) {
          auto Out = new StreamOutPort(&DB, Inst);
          Entries.push_back(Out);
          LLVM_DEBUG(dbgs() << "DownStream: "; Inst->dump(); Consume->dump());
        }
      }
    }
  }

  std::pair<std::set<Instruction*>, std::set<Instruction*>>
  BFSOperands(Instruction *From) {
    auto &DB = *DBPtr;
    std::set<Instruction*> Visited, OutBound;
    std::queue<Instruction*> Q;
    Visited.insert(From);
    Q.push(From);
    while (!Q.empty()) {
      auto Cur = Q.front();
      Q.pop();
      for (size_t i = 0; i < Cur->getNumOperands(); ++i) {
        if (auto Inst = dyn_cast<Instruction>(Cur->getOperand(i))) {
          if (!DT->dominates(Inst, From)) {
            continue;
          }
          if (!Visited.count(Inst)) {
            if (DB.InThisScope(Inst)) {
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

  void Visit(DedicatedDfg *DDPtr) {
    std::set<Instruction*> Visited;
    DBPtr = DDPtr;
    auto &DD = *DDPtr;
    auto &Entries = DD.Entries;

    for (auto BB : DD.InnerMost()->getBlocks()) {

      // I am not sure if this is a safe assumption: All the blocks have its own immediate dom.
      auto DomBB = DT->getNode(BB)->getIDom()->getBlock();

      // Is the assumption too strong here?
      // A intruction with a idom conditional instruction which does not belong to this DFG indicates
      // the predicate is always true.
      if (DD.InnerMost()->getBlocksSet().count(DomBB)) {
        auto BI = dyn_cast<BranchInst>(&DomBB->back());
        if (BI->isConditional())
          Visited.insert(dyn_cast<Instruction>(BI));
      }

      InstsToEntries(iterator_range<BasicBlock::iterator>(BB->begin(), BB->end()), Visited);
    }

    for (auto Inst : Visited) {
      if (!isa<LoadInst>(Inst)) {
        AnalyzeEntryOp(Inst);
        std::set<Instruction*> Equiv;
        FindEquivPHIs(Inst, Equiv);
        for (auto EquivInst : Equiv) {
          InspectConsumers(EquivInst);
        }
      }
    }

    /// To handle data move DFGs.
    for (auto Inst : Visited) {
      auto Load = dyn_cast<LoadInst>(Inst);

      if (!Load)
        continue;

      if (!DD.InThisDFG(Load)) {
        auto MP = dyn_cast<PortBase>(DifferentiateMemoryStream(Load));
        assert(MP);
        Entries.push_back(MP);
        size_t j = Entries.size();
        DD.InspectConsumers(Load);
        // FIXME: For now we only support one consumer.
        CHECK(j + 1 == DD.Entries.size()) << "A load without usage?";
        for (; j < Entries.size(); ++j) {
          auto Entry = Entries[j];
          if (auto PM = dyn_cast<PortMem>(Entry)) {
            // FIXME(@were): Deprecate the reserved ports.
            (void) PM; // Pretend I used this variable to avoid warnings.
            CHECK(false) << "Data movement not supported yet!";
          } else if (auto APM = dyn_cast<AtomicPortMem>(Entry)) {
            assert(APM->Store->getValueOperand() == MP->UnderlyingInst());
            APM->Op = MP->UnderlyingInst();
            // FIXME: This is not correct...
            // MP->SoftPortNum = getNextReserved();
          } else {
            assert(false && "This should not happen");
          }
        }
      }
    }

    CleanupEntryDependences();
    ReorderEntries();
  }

  void Visit(TemporalDfg *TD) override {
    DBPtr = TD;
    std::set<Instruction*> Visited;
    CHECK(TD->getBlocks().size() == 1);

    InstsToEntries(make_range(TD->Begin, TD->End), Visited);

    for (auto Inst : Visited) {
      if (!isa<LoadInst>(Inst)) {
        AnalyzeEntryOp(Inst);
        std::set<Instruction*> Equiv;
        FindEquivPHIs(Inst, Equiv);
        for (auto EquivInst : Equiv)
          InspectConsumers(EquivInst);
      }
    }

    CleanupEntryDependences();
    ReorderEntries();
  }

  void CleanupEntryDependences() {
    auto &DB = *DBPtr;
    {
      std::set<DfgEntry*> ToRemove;
      for (auto Entry : DB.EntryFilter<IndMemPort>()) {
        for (auto Elem : BFSOperands(Entry->UnderlyingInst()).first) {
          if (Elem == Entry->UnderlyingInst())
            continue;
          if (auto Entry = DB.InThisDFG(Elem)) {
            if (!isa<ComputeBody>(Entry))
              continue;
            ToRemove.insert(Entry);
          }
        }
      }
      std::vector<int> Ids;
      for (auto Elem : ToRemove) {
        for (size_t i = 0; i < DB.Entries.size(); ++i) {
          if (Elem == DB.Entries[i])
            Ids.push_back(i);
        }
      }
      std::sort(Ids.begin(), Ids.end());
      for (size_t i = 0; i < Ids.size(); ++i) {
        DB.Entries.erase(DB.Entries.begin() + Ids[i] - i);
      }
    }

    /// Figure out entry write to registers
    {
      std::vector<DfgEntry*> ToAdd;
      for (auto Entry : DB.Entries) {
        if (isa<ComputeBody>(Entry)) {
          auto Val = Entry->UnderlyingValue();
          std::set<Value*> ToInspect;
          if (auto Inst = dyn_cast<Instruction>(Val)) {
            std::set<Instruction*> Equivs;
            FindEquivPHIs(Inst, Equivs);
            for (auto Iter : Equivs)
              ToInspect.insert(Iter);
          } else {
            ToInspect.insert(Val);
          }
          bool Written = false;
          for (auto Iter : ToInspect) {
            for (auto User : Iter->users()) {
              if (auto UserInst = dyn_cast<Instruction>(User)) {
                if (isa<PHINode>(UserInst))
                  continue;
                if (!DB.BelongOtherDFG(UserInst) && !DB.InThisDFG(UserInst)) {
                  ToAdd.push_back(new OutputPort(&DB, Val));
                  LLVM_DEBUG(dbgs() << "Write to register: ";
                             Iter->dump(); Val->dump(); UserInst->dump());
                  Written = true;
                  break;
                }
              }
            }
            if (Written)
              break;
          }
        }
      }
      DB.Entries.insert(DB.Entries.end(), ToAdd.begin(), ToAdd.end());
    }

  }

  void ReorderEntries() {
    auto &DB = *DBPtr;
    auto &Entries = DB.Entries;

    LLVM_DEBUG(
      dbgs() << "Before reordering: " << DB.Entries.size() << "\n";
      for (auto &Entry : DB.Entries)
        dbgs() << Entry->dump() << "\n";
      dbgs() << "\n";
    );

    // Reorder instructions in a topological order
    // Hoist all the input entries to the very first ones
    for (size_t i = 0; i < Entries.size(); ++i)
      if (!isa<InputPort>(Entries[i]))
        for (size_t j = i + 1; j < Entries.size(); ++j)
          if (isa<InputPort>(Entries[j]))
            std::swap(Entries[i], Entries[j]);

    // Postpone all the output entries to the very last ones
    for (int i = Entries.size() - 1; i >= 0; --i)
      if (!isa<OutputPort>(Entries[i]))
        for (int j = i - 1; j >= 0; --j)
          if (isa<OutputPort>(Entries[j]))
            std::swap(Entries[i], Entries[j]);

    std::set<Value*> Ready;
    for (size_t i = 0; i < Entries.size(); ++i) {

      if (isa<CtrlSignal>(Entries[i]))
        continue;

      if (isa<InputPort>(Entries[i])) {
        Ready.insert(Entries[i]->UnderlyingValue());
        LLVM_DEBUG(dbgs() << "Already OK: "; Entries[i]->UnderlyingValue()->dump());
        continue;
      }


      // TODO: Refactor this by replacing the `operand ready' by dominators
      if (!isa<OutputPort>(Entries[i])) {
        if (Entries[i]->OperandReady(Ready)) {
          LLVM_DEBUG(dbgs() << "Already OK: "; Entries[i]->UnderlyingValue()->dump());
          Ready.insert(Entries[i]->UnderlyingValue());
        } else {
          LLVM_DEBUG(dbgs() << "Looking for replacement of: ";
                     Entries[i]->UnderlyingInst()->dump());
          bool Found = false;
          for (size_t j = i + 1; j < Entries.size() && !Found && !isa<OutputPort>(Entries[j]); ++j) {
            if (Entries[j]->OperandReady(Ready)) {
              Ready.insert(Entries[j]->UnderlyingValue());
              std::swap(Entries[i], Entries[j]);
              Found = true;
            }
          }
          if (!Found) {
            FinalizeEntryID();
            errs() << Entries[i]->dump() << "\n";
            std::ostringstream oss;
            DB.dump(oss);
            errs() << oss.str() << "\n";
            assert(false && "NOT acyclic DFG?");
          }
        }
        if (auto Pred = dyn_cast<Predicate>(Entries[i])) {
          for (auto Cond : Pred->Cond)
            Ready.insert(Cond);
        }
      } else {
        break;
      }
    }

    LLVM_DEBUG(
      dbgs() << "After reordering: " << Entries.size() << "\n";
      for (auto &Entry : Entries)
        dbgs() << Entry->dump() << "\n";
      dbgs() << "\n";
    );

    FinalizeEntryID();

  }

  void FinalizeEntryID() {
    auto &DB = *DBPtr;
    auto &Entries = DB.Entries;
    for (int i = 0, n = Entries.size(); i < n; ++i) {
      Entries[i]->ID = i;
    }
    int j = Entries.size();
    for (auto Elem : DB.EntryFilter<IndMemPort>()) {
      Elem->Index->ID = j++;
    }
  }

  DFGEntryAnalyzer(DominatorTree *DT_) : DT(DT_) {}
};


void ExtractDFGFromScope(DFGFile &DF,
                         IntrinsicInst *Start, IntrinsicInst *End,
                         DominatorTree *DT, LoopInfo *LI) {
  // Extract the sketch of dedicated and temporal DFGs respectively.
  // The instruction entries in each DFG cannot be instantiated until all the
  // sketches are extracted, because we need this to analyze inter-DFG communication in one pass.
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
        if (auto TemporalEnd = dyn_cast<IntrinsicInst>(Cur)) {
          if (TemporalEnd->getIntrinsicID() != Intrinsic::ss_temporal_region_end)
            continue;
          CHECK(TemporalEnd->getOperand(0) == TemporalStart);
          auto TD = new TemporalDfg(&DF, TemporalStart, TemporalEnd);
          DF.addDfg(TD);
          Found = true;
          break;
        }
      }
      CHECK(Found);
    }
    if (BB == End->getParent()) {
      break;
    }
  }
  // All the sketches are extracted. Fill in the entries.
  for (auto Elem : DF.DFGFilter<DfgBase>()) {
    DFGEntryAnalyzer DEA(DT);
    Elem->Accept(&DEA);
  }

}

}
}