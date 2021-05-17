#include "CodeXform.h"

#include "dsa/rf.h"


// raw_ostream &operator<<(raw_ostream &os, DfgEntry &DE) {
//   os << "# [" << dsa::xform::KindStr[DE.Kind] << "]" << " DFG" << DE.Parent->ID << " Entry" << DE.ID;
//   if (DE.UnderlyingInsts().size() == 1) {
//     os << " " << *DE.UnderlyingInst();
//   } else {
//     os << "\n";
//     for (auto Elem : DE.UnderlyingInsts()) {
//       os << *Elem << "\n";
//     }
//   }
//   return os;
// }
// 
// raw_ostream &operator<<(raw_ostream &os, DfgBase &DB) {
//   std::ostringstream oss;
//   DB.dump(oss);
//   os << oss.str();
//   return os;
// }

namespace dsa {
namespace xform {

namespace {

/*!
 * \brief To avoid some unexpected iterator issue.
 *        We delete the instructions together.
 * \param Insts The instructions to be removed.
 */
void EliminateInstructions(const std::vector<Instruction*> &Insts) {
  for (auto I : Insts) {
    I->replaceAllUsesWith(UndefValue::get(I->getType()));
    I->eraseFromParent();
  }
}

}

void EliminateTemporal(Function &F) {
  std::vector<Instruction*> ToRemove;
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto Intrin = dyn_cast<IntrinsicInst>(&I)) {
        switch (Intrin->getIntrinsicID()) {
          case Intrinsic::ss_temporal_region_start:
          case Intrinsic::ss_temporal_region_end:
            ToRemove.push_back(Intrin);
            break;
          default:
            break;
        }
      }
    }
  }
  EliminateInstructions(ToRemove);
}


std::vector<utils::StickyRegister>
InjectDSARegisterFile(Function &F) {
  IRBuilder<> IB(F.getContext());
  IB.SetInsertPoint(&F.getEntryBlock().back());
  std::vector<utils::StickyRegister> Res;
  std::string TyName("reg.type");
  auto RegType = StructType::create(TyName, IB.getInt64Ty(), IB.getInt8Ty());
  for (int i = 0; i < DSARF::TOTAL_REG; ++i) {
    auto A = IB.CreateAlloca(RegType, nullptr, REG_NAMES[i]);
    auto R = IB.CreateInBoundsGEP(A, {IB.getInt32(0), IB.getInt32(0)}, std::string(REG_NAMES[i]) + ".r.ptr");
    auto S = IB.CreateInBoundsGEP(A, {IB.getInt32(0), IB.getInt32(1)}, std::string(REG_NAMES[i]) + ".s.ptr");
    IB.CreateStore(IB.getInt64(REG_STICKY[i]), R);
    IB.CreateStore(IB.getInt8(i == DSARF::TBC), S);
    Res.emplace_back(A, R, S);
  }
  return Res;
}

// TODO(@were): I did not remember what is this function for.
bool HasOtherUser(InputPort *IP, Value *Val) {
  for (auto User : Val->users()) {
    if (auto Entry = IP->Parent->InThisDFG(User)) {
      if (auto CB = dyn_cast<ComputeBody>(Entry)) {
        if (!CB->isAtomic())
          return true;
      }
      if (isa<Predicate>(Entry)) {
        return true;
      }
    }
  }
  return false;
}

std::string ValueToOperandText(Value *Val) {

  if (auto CI = dyn_cast<ConstantInt>(Val)) {
    return formatv("{0}", CI->getValue().getSExtValue());
  } else if (auto CFP = dyn_cast<ConstantFP>(Val)) {
    //TODO: Support more data types
    double FPV = CFP->getValueAPF().convertToDouble();
    uint64_t *RI = reinterpret_cast<uint64_t*>(&FPV);
    return formatv("{0}", *RI);
  }

  CHECK(false) << "Cannot dump " << *Val;
  return "";
}


std::string GetOperationStr(Instruction *Inst, bool isAcc, bool predicated) {
  std::string OpStr;
  int BitWidth = Inst->getType()->getScalarSizeInBits();

  if (auto Cmp = dyn_cast<CmpInst>(Inst)) {
    // TODO: Support floating point
    OpStr = "compare";
    BitWidth = Cmp->getOperand(0)->getType()->getScalarSizeInBits();
  } else if (auto Call = dyn_cast<CallInst>(Inst)) {
    OpStr = Call->getCalledFunction()->getName().str();
  } else {
    OpStr = Inst->getOpcodeName();
    if (isAcc) {
      if (!predicated) {
        OpStr = OpStr[0] == 'f' ? "facc" : "acc";
      } else {
        OpStr = OpStr[0] == 'f' ? "faccumulate" : "accumulate";
      }
    }
  }

  for (int i = 0, e = OpStr[0] == 'f' ? 2 : 1; i < e; ++i) {
    OpStr[i] -= 'a';
    OpStr[i] += 'A';
  }

  return formatv("{0}{1}", OpStr, BitWidth);
}

const char *KindStr[] = {
  "DfgEntry",

  // Computation {
  "ComputeStarts",
  "ComputeBody",
  "Duplicate",
  "Accumulator",
  "ComputeEnds",
  // }

  // Predicate {
  "Predicate",
  // }

  // Port {
  "PortStarts",
  "PortBase",

  // InPort {
  "InPortStarts",
  "InputPort",
  "MemPort",
  "IndMemPort",
  "CtrlMemPort",
  "InputConst",
  "StreamInPort",
  "CtrlSignal",
  "InPortEnds",
  // }

  // OutPort {
  "OutPortStarts",
  "OutputPort",
  "PortMem",
  "AtomicPortMem",
  "StreamOutPort",
  "OutPortEnds",
  // }

  "PortEnds"
  // }
};


struct DFGPrinter : dsa::DFGVisitor {

  struct ControlBit {
    std::ostringstream SubCtrl[3];
    Predicate *Pred{nullptr};

    void addControlledMemoryStream(int Idx, DfgEntry *DE) {
      if (auto CMP = dyn_cast<CtrlMemPort>(DE)) {
         if (Pred == nullptr) {
           Pred = CMP->Pred;
         } else {
           CHECK(Pred == CMP->Pred)
              << "An instruction cannot be controlled by more than one predication.";
         }
         assert(Pred);
         for (int j = 0; j < 3; ++j) {
           if (~CMP->Mask >> j & 1) {
             if (!SubCtrl[j].str().empty()) {
               SubCtrl[j] << "|";
             }
             SubCtrl[j] << "b" << (Idx + 1);
           }
         }
      }
    }

    bool empty() {
      for (int i = 0; i < 3; ++i)
        if (!SubCtrl[i].str().empty())
          return false;
      return true;
    }

    std::string finalize() {
      std::ostringstream Ctrl;
      for (int i = 0; i < 3; ++i) {
        if (!SubCtrl[i].str().empty()) {
          if (!Ctrl.str().empty()) Ctrl << ", ";
          Ctrl << i << ":" << SubCtrl[i].str();
        }
      }
      return Ctrl.str();
    }

    void updateAbstain(int x, bool isAcc) {
      for (int i = 0; i < 3; ++i) {
        if (x >> i & 1) {
          if (!SubCtrl[i].str().empty()) {
            SubCtrl[i] << "|";
          }
          SubCtrl[i] << "a";
        } else if (isAcc) {
          if (!SubCtrl[i].str().empty()) {
            SubCtrl[i] << "|";
          }
          SubCtrl[i] << "d";
        }
      }
    }
  };


  /*!
   * \brief I anyways need the DFG entries to be reordered in which the order the are emitted. 
   * \param DB The DFG to reorder the entries.
   */
  static std::vector<DfgEntry*> ReorderEntries(DfgBase *DB) {
    std::vector<DfgEntry*> Res;
    Res.reserve(DB->Entries.size());
    for (auto Elem : DB->EntryFilter<InputPort>()) {
      Res.push_back(Elem);
    }
    for (auto Elem : DB->Entries) {
      if (isa<InputPort>(Elem) || isa<OutputPort>(Elem)) {
        continue;
      }
      Res.push_back(Elem);
    }
    for (auto Elem : DB->EntryFilter<OutputPort>()) {
      Res.push_back(Elem);
    }
    return Res;
  }


  void Visit(DedicatedDfg *DD) override {
    if (utils::ModuleContext().TRIGGER) {
      CHECK(utils::ModuleContext().TEMPORAL)
        << "Trigger cannot be enabled without temporal";
      os << "#pragma group temporal\n";
    }
    DFGVisitor::Visit(DD);
  }

  void Visit(TemporalDfg *TD) override {
    os << "#pragma group temporal\n";
    DFGVisitor::Visit(TD);
  }

  void Visit(DfgBase *DB) {
    using dfg::MetaPort;
    // FIXME(@were): Add block freqency an attribute of DFG.
    // auto BF = Parent->Query->BFI->getBlockFreq(getBlocks()[0]);
    // os << "#pragma group frequency " << 1 << "\n";
    os << "#pragma group unroll " << DB->getUnroll() << "\n";

    auto Reordered = ReorderEntries(DB);

    struct DFGEntryEmitter : DFGEntryVisitor {
      void Visit(DfgEntry *DE) override {
        os << "# [" << KindStr[DE->Kind] << "]: DFG" << DE->Parent->ID << " Entry" << DE->ID << "\n";
        int i = 0;
        auto Insts = DE->UnderlyingInsts();
        for (auto Elem : Insts) {
          os << "# Inst";
          if(Insts.size() != 1) {
            os << i++;
          }
          os << ": " << *Elem << "\n";
        }
      }

      void Visit(Accumulator *Acc) {

        Visit(cast<DfgEntry>(Acc));
        auto Inst = Acc->Operation;
        std::queue<std::string> Q;
        auto Reduce = GetOperationStr(Inst, false, false);
        ControlBit CtrlBit;

        for (int vec = 0; vec < Acc->Parent->getUnroll(); ++vec) {
          for (size_t j = 0; j < 2; ++j) {
            auto Operand = Inst->getOperand(j);
            if (!isa<PHINode>(Operand)) {
              auto Entry = Acc->Parent->InThisDFG(Operand);
              CHECK(Entry);
              Q.push(Entry->Name(vec));
            }
          }
        }

        while (Q.size() > 1) {
          auto A = Q.front(); Q.pop(); assert(!Q.empty());
          auto B = Q.front(); Q.pop();
          os << formatv("TMP{0} = {1}({2}, {3})", Parent->TmpCnt, Reduce, A, B).str() << "\n";
          Q.push(formatv("TMP{0}", Parent->TmpCnt++));
        }

        auto P = Acc->getPredicate();
        os << Acc->Name() << " = " << GetOperationStr(Inst, true, P != nullptr) << "(" << Q.front()
           << ", ";

        if (dsa::utils::ModuleContext().PRED) {
          if (P || !CtrlBit.empty()) {
            CtrlBit.updateAbstain(Acc->getAbstainBit(), true);
            os << "ctrl=" << P->Name(-1) << "{" << CtrlBit.finalize() << "}";
          } else {
            os << Acc->Ctrl->Name();
          }
        } else {
          os << "ctrl=" << Acc->Ctrl->Name() << "{2:d}";
        }

        os << ")\n";
      }

      void Visit(CtrlSignal *CS) {
        if (!CS->Controlled->getPredicate() || !dsa::utils::ModuleContext().PRED) {
          Visit(cast<DfgEntry>(CS));
          os << "Input: " << CS->Name() << "\n";
        }
      }

      void Visit(InputConst *IC) {
        if (!HasOtherUser(IC, IC->Val)) {
          return;
        }
        Visit(cast<DfgEntry>(IC));
        os << "Input" << IC->Val->getType()->getScalarSizeInBits() << ": " << IC->Name() << "\n";
      }

      void Visit(InputPort *IP) override {
        if (!utils::ModuleContext().PRED) {
          for (auto User : IP->UnderlyingInst()->users()) {
            auto Entry = IP->Parent->InThisDFG(User);
            if (Entry && !isa<Predicate>(Entry)) {
              return;
            }
          }
        }
        if (auto IMP = dyn_cast<IndMemPort>(IP)) {
          if (IMP->duplicated()) {
            return;
          }
        }
        if (!HasOtherUser(IP, IP->UnderlyingInst())) {
          return;
        }
        Visit(cast<DfgEntry>(IP));
        {
          auto Meta = IP->Meta;
          if (Meta.source != MetaPort::Data::Unknown) {
            os << "#pragma src=" << MetaPort::DataText[(int) Meta.source] << "\n";
          }
          if (Meta.dest != MetaPort::Data::Unknown) {
            os << "#pragma dest=" << MetaPort::DataText[(int) Meta.dest] << "\n";
          }
          if (!Meta.dest_port.empty()) {
            os << "#pragma dest=" << Meta.dest_port << "\n";
          }
          for (int i = 0; i < (int) MetaPort::Operation::Unknown; ++i) {
            if (Meta.op >> i & 1) {
              os << "#pragma op=" << MetaPort::OperationText[i] << "\n";
            }
          }
          if (Meta.conc) {
            os << "#pragma conc=" << Meta.conc << "\n";
          }
          os << "#pragma cmd=" << Meta.cmd << "\n";
          os << "#pragma repeat=" << Meta.repeat << "\n";
        }

        os << "Input" << IP->UnderlyingInst()->getType()->getScalarSizeInBits()
                      << ": " << IP->Name();
        if (IP->ShouldUnroll()) {
          os << "[" << IP->Parent->getUnroll() << "]";
        }
        os << "\n";
      }

      void Visit(ComputeBody *CB) override {
        // If this ComputeBody is handled by a atomic operation skip it!
        if (CB->isAtomic())
          return;

        Visit(cast<DfgEntry>(CB));

        int Degree = (CB->ShouldUnroll() ? CB->Parent->getUnroll() : 1);
        for (int vec = 0; vec < Degree; ++vec) {
          os << CB->Name(vec) << " = " << GetOperationStr(CB->Operation, false, false) << "(";
          ControlBit CtrlBit;

          bool isCall = isa<CallInst>(CB->Operation);
          for (int i = 0; i < (int) (CB->Operation->getNumOperands() - isCall); ++i) {
            auto Val = CB->Operation->getOperand(i);
            if (i) os << ", ";
            if (auto EntryOp = CB->Parent->InThisDFG(Val)) {
              os << EntryOp->Name(vec);
              CtrlBit.addControlledMemoryStream(i, EntryOp);
            } else {
              os << ValueToOperandText(Val);
            }
          }

          if (utils::ModuleContext().PRED) {
            if (CB->getPredicate())
              CtrlBit.updateAbstain(CB->getAbstainBit(), false);
            if (!CtrlBit.empty()) {
              os << ", ctrl=" << CtrlBit.Pred->Name(vec) << "{" << CtrlBit.finalize() << "}";
            }
          }

          os << ")\n";
        }

      }

      void Visit(Predicate *P) override {
        if (!utils::ModuleContext().PRED) {
          return;
        }

        {
          // TODO(@were): Finalize this.
          os << "# [Predication] Combination\n";
          for (auto I : P->Cond) {
            os << "# " << *I << "\n";
          }
        }

        ControlBit CtrlBit;
        os << P->Name(-1) << " = Compare" << P->Cond[0]->getOperand(0)->getType()->getScalarSizeInBits()
           << "(";
        for (size_t i = 0; i < P->Cond[0]->getNumOperands(); ++i) {
          auto Val = P->Cond[0]->getOperand(i);
          if (i) os << ", ";
          if (auto EntryOp = P->Parent->InThisDFG(Val)) {
            os << EntryOp->Name(-1);
            CtrlBit.addControlledMemoryStream(i, EntryOp);
            if (auto CMP = dyn_cast<CtrlMemPort>(EntryOp))
              CMP->ForPredicate = true;
          } else {
            os << ValueToOperandText(Val);
          }
        }

        if (!CtrlBit.empty()) {
          if (CtrlBit.Pred == P)
            os << ", self=";
          else
            os << ", ctrl=" << CtrlBit.Pred->Name(-1);
          os << "{" << CtrlBit.finalize() << "}";
        }

        os << ")\n";

      }

      void Visit(OutputPort *OP) {
        Visit(cast<DfgEntry>(OP));
        std::ostringstream oss;
        OP->Meta.to_pragma(oss);
        os << oss.str();
        auto Entry = OP->Parent->InThisDFG(OP->Output);
        CHECK(Entry);
        for (int i = 0; i < (Entry->ShouldUnroll() ? OP->Parent->getUnroll() : 1); ++i) {
          os << OP->Name(i) << " = " << Entry->Name(i) << "\n";
        }
        os << "Output" << OP->Output->getType()->getScalarSizeInBits() << ": " << OP->Name();
        if (Entry->ShouldUnroll())
          os << "[" << OP->Parent->getUnroll() << "]";
        os << "\n";
      }

      raw_ostream &os;
      DFGPrinter *Parent;

      DFGEntryEmitter(raw_ostream &os_, DFGPrinter *Parent_) : os(os_), Parent(Parent_) {}

    };

    DFGEntryEmitter DEE(os, this);
    for (auto Elem : DB->Entries) {
      Elem->Accept(&DEE);
    }

    // TODO(@were): Bing this back for atomic operations.
    // /// Emit intermediate result for atomic operation
    // for (auto Elem : EntryFilter<ComputeBody>())
    //   Elem->EmitAtomic(os);

    // TODO(@were): Move this to a DFG.
    for (auto Elem : DB->EntryFilter<IndMemPort>()) {
      os << "\n\n----\n\n";
      os << "Input" << Elem->Index->Load->getType()->getScalarSizeInBits()
         << ": indirect_in_" << DB->ID << "_" << Elem->ID << "\n";
      os << "indirect_out_" << DB->ID << "_" << Elem->ID << " = "
         << "indirect_in_" << DB->ID << "_" << Elem->ID << "\n";
      os << "Output" << Elem->Index->Load->getType()->getScalarSizeInBits()
         << ": indirect_out_" << DB->ID << "_" << Elem->ID << "\n";
    }
  }

  raw_ostream &os;
  /*!
   * \brief Count the ticker of temp variables for accumulator reduction.
   */
  int TmpCnt{0};

  DFGPrinter(raw_ostream &os_) : os(os_) {}
};


void EmitDFG(raw_ostream *os, DFGFile *DFG) {
  bool DeleteOS = false;
  if (!os) {
    std::error_code EC;
    os = new raw_fd_ostream(DFG->getName(), EC);
    DeleteOS = true;
  }
  DFGPrinter DP(*os);
  auto Graphs = DFG->DFGFilter<DfgBase>();
  for (int i = 0; i < (int) Graphs.size(); ++i) {
    auto Elem = Graphs[i];
    if (i) {
      *os << "----\n";
    }
    Elem->Accept(&DP);
  }
  if (DeleteOS) {
    delete os;
  }
}

}
}
