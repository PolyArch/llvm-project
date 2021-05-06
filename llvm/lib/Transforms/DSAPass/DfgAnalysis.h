#pragma once

#include "./llvm_common.h"

namespace dsa {
namespace analysis {

/*!
 * \brief Gather all the pairs of dsa.config.start/end
 * \param F The function to be analyzed.
 */
std::vector<std::pair<IntrinsicInst*, IntrinsicInst*>>
GatherConfigScope(Function &F);

}
}