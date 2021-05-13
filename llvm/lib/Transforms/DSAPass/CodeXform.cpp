#include "CodeXform.h"

#include "dsa/rf.h"

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
    auto R = IB.CreateInBoundsGEP(A, {IB.getInt32(0), IB.getInt32(0)});
    auto S = IB.CreateInBoundsGEP(A, {IB.getInt32(0), IB.getInt32(1)});
    IB.CreateStore(IB.getInt64(REG_STICKY[i]), R);
    IB.CreateStore(IB.getInt8(i == DSARF::TBC), S);
    Res.emplace_back(A, R, S);
  }
  return Res;
}

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

}
}

std::ostream &operator<<(std::ostream &os, DfgEntry &DE) {
  return os;
}

std::ostream &operator<<(std::ostream &os, DfgBase &DB) {
  std::ostringstream oss;

  struct DFGPrinter : dsa::DFGVisitor {
    void Visit(DedicatedDfg *DD) override {
      if (!First) {
        os << "----\n";
      }
      DFGVisitor::Visit(DD);
    }

    void Visit(TemporalDfg *TD) override {
      if (!First) {
        os << "----\n";
      }
      DFGVisitor::Visit(TD);
    }

    void Visit(DfgBase *DB) {
      // FIXME(@were): Add block freqency an attribute of DFG.
      // auto BF = Parent->Query->BFI->getBlockFreq(getBlocks()[0]);
      // os << "#pragma group frequency " << 1 << "\n";
      os << "#pragma group unroll " << DB->getUnroll() << "\n";

      for (auto Elem : DB->EntryFilter<InputPort>()) {
        if (!dsa::utils::ModuleContext().PRED) {
          bool OnlyPredicate{true};
          for (auto User : Elem->UnderlyingInst()->users()) {
            auto Entry = DB->InThisDFG(User);
            if (Entry && !isa<Predicate>(Entry)) {
              OnlyPredicate = false;
              break;
            }
          }
          if (OnlyPredicate) {
            return;
          }
        }

        // FIXME: We need also inspect all the consumers to make sure that
        //        this indirect memory is not used by other instructions.
        if (auto IMP = dyn_cast<IndMemPort>(Elem)) {
          if (IMP->duplicated()) {
            return;
          }
        }

        if (!dsa::xform::HasOtherUser(Elem, Elem->UnderlyingInst())) {
          return;
        }

        os << "# From Inst: " << Elem->dump() << "\n";
        std::ostringstream oss;
        Elem->Meta.to_pragma(oss);
        os << oss.str();

        os << "Input" << Elem->UnderlyingInst()->getType()->getScalarSizeInBits()
                      << ": " << Elem->Name();
        if (Elem->ShouldUnroll())
          os << "[" << DB->getUnroll() << "]";
        os << "\n";

      }

      // for (auto Elem : Entries) {
      //   if (auto CB = dyn_cast<ComputeBody>(Elem))
      //     CB->EmitCB(os);
      //   if (auto P = dyn_cast<Predicate>(Elem))
      //     P->EmitCond(os);
      // }

      // /// Emit intermediate result for atomic operation
      // for (auto Elem : EntryFilter<ComputeBody>())
      //   Elem->EmitAtomic(os);

      // for (auto Elem : EntryFilter<OutputPort>())
      //   Elem->EmitOutPort(os);

      // // TODO(@were): Move this to a DFG.
      // for (auto Elem : EntryFilter<IndMemPort>()) {
      //   os << "\n\n----\n\n";
      //   os << "Input" << Elem->Index->Load->getType()->getScalarSizeInBits()
      //      << ": indirect_in_" << ID << "_" << Elem->ID << "\n";
      //   os << "indirect_out_" << ID << "_" << Elem->ID << " = "
      //      << "indirect_in_" << ID << "_" << Elem->ID << "\n";
      //   os << "Output" << Elem->Index->Load->getType()->getScalarSizeInBits()
      //      << ": indirect_out_" << ID << "_" << Elem->ID << "\n";
      // }
    }

    bool First{true};
    std::ostream &os;

    DFGPrinter(std::ostream &os_) : os(os_) {}
  };

  DB.dump(oss);
  os << oss.str();
  return os;
}