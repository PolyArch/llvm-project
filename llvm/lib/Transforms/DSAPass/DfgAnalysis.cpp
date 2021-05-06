#include "dsa/debug.h"

#include "./DfgAnalysis.h"

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

}
}