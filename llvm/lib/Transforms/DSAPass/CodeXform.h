#pragma once

#include "./llvm_common.h"

#include "Util.h"

namespace dsa {
namespace xform {

/*!
 * \brief Eliminate temporal intrinsics when temporal is not supported.
 * \param F The function to be transformed.
 */
void EliminateTemporal(Function &F);

/*!
 * \brief Inject DSA registers.
          These memory buffer will be used for register stickiness optimization.
 * \param F The function to be transformed.
 */
std::vector<utils::StickyRegister> InjectDSARegisterFile(Function &F);

}
}