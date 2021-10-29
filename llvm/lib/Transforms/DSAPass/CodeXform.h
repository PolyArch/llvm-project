#pragma once

#include <iostream>

#include "./llvm_common.h"

#include "DFGAnalysis.h"
#include "DFGIR.h"
#include "Util.h"

namespace dsa {
namespace xform {

/*!
 * \brief Eliminate temporal intrinsics when temporal is not supported.
 * \param F The function to be transformed.
 */
void eliminateTemporal(Function &F);

/*!
 * \brief Inject DSA registers.
          These memory buffer will be used for register stickiness optimization.
 * \param F The function to be transformed.
 */
std::vector<utils::StickyRegister> injectDSARegisterFile(Function &F);

/*!
 * \brief Emit DFG to the given destination.
 * \param os Where the given DFG is emitted. If null, dump to the filename of
 * the given DFGFile.
 */
void emitDFG(raw_ostream &OS, DFGFile *DFG, analysis::DFGAnalysisResult &DAR, CodeGenContext &CGC);

struct CodeGenContext {
  /*!
   * \brief The IR injection wrapper.
   */
  IRBuilder<> *IB;
  /*!
   * \brief The DSA register file for peephole injection.
   */
  const RegisterFile &Regs;
  /*!
   * \brief Loop scalar analyzer.
   */
  ScalarEvolution &SE;
  /*!
   * \brief SCEV expander.
   */
  SCEVExpander &SEE;
  /*!
   * \brief Instruction domination.
   */
  DominatorTree *DT;
  /*!
   * \brief Loop information.
   */
  LoopInfo *LI;
  /*!
   * \brief The injected intrinsics.
   */
  std::vector<CallInst *> Res;

  /*!
   * \brief Implemeting the register to make it compatible with intrin_impl.
   */
  class REG {
    // I am not sure if it is a good design decision.
    Value *Val{nullptr};
    uint64_t I{0};

  public:
    REG() : Val() {}
    // If the value fed is already a llvm::Value, just feed it into Val.
    REG(Value *V) : Val(V) {}
    // If it is a C-PoD int, store it temporarily in I.
    REG(uint64_t V) : I(V) {}
    // This value should be finalized into a llvm::Value when accessing.
    Value *&value(IRBuilder<> *IB) {
      if (!Val) {
        Val = IB->getInt64(I);
      }
      return Val;
    }
  };

  CodeGenContext(IRBuilder<> *IB, const RegisterFile &Regs, ScalarEvolution &SE, SCEVExpander &SEE,
                 DominatorTree *DT, LoopInfo *LI) :
    IB(IB), Regs(Regs), SE(SE), SEE(SEE), DT(DT), LI(LI) {}

  /*!
   * \brief Inject load/store instructions so that the compiler can
   * automatically analyze the stickiness.
   */
  void injectFusionHints(std::string Mnemonic, REG &A, REG &B, int C);

  /*!
   * \brief Injector of intrinsic instructions.
   * \param Mnemonic The mnmonic of the instruction.
   * \param OpConstrain The operand constrain of the instructions.
   * \param Args The argument of the instruction call.
   * \param ResTy The result of the instruction.
   */
  void intrinsicImpl(const std::string &Mnemonic,
                     const std::string &OpConstrain,
                     const std::vector<Value *> &Args, Type *ResTy);

  /// Implementing the interfaces for intrinsic injection.
  /// @{
  REG DIV(REG a, REG b) { return IB->CreateUDiv(a.value(IB), b.value(IB)); } // NOLINT

  REG SUB(REG a, REG b) { return IB->CreateSub(a.value(IB), b.value(IB)); } // NOLINT

  REG SHL(REG a, REG b) { return IB->CreateShl(a.value(IB), b.value(IB)); } // NOLINT

  REG MUL(REG a, REG b) { return IB->CreateMul(a.value(IB), b.value(IB)); } // NOLINT

  void INTRINSIC_DRI(std::string Mnemonic, REG &A, REG B, int C) { // NOLINT
    intrinsicImpl(Mnemonic, "=r,r,i", {B.value(IB), IB->getInt64(C)},
                  IB->getInt64Ty());
    A.value(IB) = Res.back();
  }

  void INTRINSIC_RRI(std::string Mnemonic, REG A, REG B, int C) { // NOLINT
    injectFusionHints(Mnemonic, A, B, C);
    intrinsicImpl(Mnemonic, "r,r,i",
                  {A.value(IB), B.value(IB), IB->getInt64(C)}, IB->getVoidTy());
  }

  void INTRINSIC_RI(std::string Mnemonic, REG A, int B) { // NOLINT
    intrinsicImpl(Mnemonic, "r,i", {A.value(IB), IB->getInt64(B)},
                  IB->getVoidTy());
  }

  void INTRINSIC_R(std::string Mnemonic, REG A) { // NOLINT
    intrinsicImpl(Mnemonic, "r", {A.value(IB)}, IB->getVoidTy());
  }
  /// @}

#include "intrin_impl.h"
};


/*!
 * \brief The base class of indirect stream emission.
 */
struct IndirectStream {
  /*!
   * \brief The data type of index.
   */
  int IType;
  /*!
   * \brief The data type of data accessed.
   */
  int DType;
  /*!
   * \brief Emit this stream to program intrinsic.
   */
  virtual void emitIntrinsic(CodeGenContext &CGC) = 0;
};

/*!
 * \brief The extracted 1D stream. a[b[i]].
 */
// TODO(@were): Support more complicated pattern than b[i].
struct Indirect1D : IndirectStream {
  /*!
   * \brief The base address of indirect load.
   */
  const SCEV *SAR;
  /*!
   * \brief The trip count of loading.
   */
  const SCEV *L1D;
  /*!
   * \brief Stream of b[i].
   */
  const analysis::SEWrapper *Idx;
  /*!
   * \brief Emit this stream to program intrinsic.
   */
  void emitIntrinsic(CodeGenContext &CGC) {}
};

/*!
 * \brief The extracted 1D stream. (a + b[i])[idx[j] + j * stride1d].
 */
struct Indirect2D : IndirectStream {
  /*!
   * \brief The port where (a + b[i]) comes.
   */
  int OffsetPort{-1};
  /*!
   * \brief The begin address can either be a stream or a constant.
   */
  const analysis::SEWrapper *SAR;
  /*
   * \brief The port where idx[j] comes.
   */
  int IdxPort{-1};
  /*!
   * \brief The index can either be a stream or a constant 0.
   */
  const analysis::SEWrapper *Idx;
  /*!
   * \brief The port where the trip count of L1D comes.
   */
  int LengthPort;
  /*!
   * \brief The 1D trip count can either be a stream or a constant from L1D register.
   */
  const analysis::SEWrapper *L1D;
  /*!
   * \brief The outer loop trip count.
   */
  const SCEV *L2D;
  /*!
   * \brief Emit this stream to program intrinsic.
   */
  void emitIntrinsic(CodeGenContext &CGC) {}
};

/*!
 * \brief Inject configuration instruction.
 * \param CGC The injection wrapper.
 * \param CI The result of config analysis.
 * \param Start Where the config is injected.
 * \param End Where the wait all is injected.
 */
void injectConfiguration(CodeGenContext &CGC, analysis::ConfigInfo &CI,
                         Instruction *Start, Instruction *End);

/*!
 * \brief Inject stream intrinsics.
 * \param CGC The context of code injection.
 */
void injectStreamIntrinsics(CodeGenContext &CGC, DFGFile &DF, analysis::DFGAnalysisResult &DAR);

/*!
 *  \brief Erase offloaded instructions.
 *  \param DF
 *  \param CGC
 */
void eraseOffloadedInstructions(DFGFile &DF, CodeGenContext &CGC);

} // namespace xform
} // namespace dsa
