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

}
}