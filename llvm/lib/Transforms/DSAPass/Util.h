#pragma once

#include <functional>
#include <map>
#include <set>
#include <string>
#include <sys/select.h>
#include <sys/time.h>
#include <tuple>

#include "DFGEntry.h"
#include "dsa/debug.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"


using namespace llvm;

namespace dsa {
namespace utils {

/*!
 * \brief A time profiler to profile the time of each region of interest.
 *        This profiler adopts a tree-style profiling. For example, if you have
 *
 *        tp.beginRoi();
 *        // balah0..
 *        tp.beginRoi();
 *        // balah1..
 *        tp.endRoi();
 *        // balah2..
 *        tp.endRoi();
 *
 *        You will have ((balah1: time) balah0+balah1+balah2: time)
 */
struct TimeProfiler {

  /*!
   * \brief The region of interests to be pranthesis closed.
   */
  std::vector<std::pair<uint64_t, std::vector<int>>> TimeStack;

  struct Node {
    int64_t TimeEllapsed;
    std::vector<int> Child;
  };

  /*!
   * \brief Time of each span ellapsed.
   */
  std::vector<Node> Buffer;

  static uint64_t currentTime();

  /*!
   * \brief Enter an ROI.
   */
  void beginRoi();

  /*!
   * \brief Leave an ROI.
   */
  void endRoi();

  /*!
   * \brief Dump the time.
   */
  std::string toString();

  void dfsImpl(int X, std::ostringstream &OSS);

  ~TimeProfiler() { DSA_CHECK(TimeStack.empty()); }
};

/*!
 * \brief The compilation transformations enabled modularly.
 */
struct ModuleFlags {

#define DEFINE_FLAG(x, default_)            \
  int64_t x{default_};                      \
  void getFlag##x() {                       \
    char *env = getenv(#x);                 \
    if (env) x = std::stol(env);            \
    DSA_LOG(FLAG) << #x << " set to " << x; \
  }
  /*!
   * \brief Indirect memory access like a[b[i]].
   */
  DEFINE_FLAG(IND, 1)
  /*!
   * \brief Optimization for recurrent update.
   */
  DEFINE_FLAG(REC, 1)
  /*!
   * \brief Converting control flow to data flow predication.
   */
  DEFINE_FLAG(PRED, 1)
  /*!
   * \brief Do not remove temporal regions, but offload them to dedicated PE.
   */
  DEFINE_FLAG(TEMPORAL_FALLBACK, -1)
  /*!
   * \brief If we want to move all the processing elements on triggered PE (even
   * if they are not annotated with temporal).
   */
  DEFINE_FLAG(TRIGGER, 0)
  /*!
   * \brief If we just want to extract the DFG without further transformation.
   */
  DEFINE_FLAG(EXTRACT, 0)
  /*!
   * \brief If we want to execute the peephole optimization over dsa register
   * file instructions.
   */
  DEFINE_FLAG(FUSION, 0)
  /*!
   * \brief If we want a dummy mapping for DFGs.
   */
  DEFINE_FLAG(DUMMY, 0)
  /*!
   * \brief If we want to dump the bitstream.
   */
  DEFINE_FLAG(BITSTREAM, 0)
  /*!
   * \brief If ADG is old version.
   */
  DEFINE_FLAG(COMPAT_ADG, 0)
  /*!
   * \brief Perform transformation without O3 cleanup.
   */
  DEFINE_FLAG(RAW, 0)
  /*!
   * \brief Perform transformation without O3 cleanup.
   */
  DEFINE_FLAG(SEED, -1)
  /*!
   * \brief If we want to execute the peephole optimization over dsa register
   * file instructions.
   */
  DEFINE_FLAG(FUSE_STREAM, 1)
  /*!
   * \brief If we want to execute the peephole optimization over dsa register
   * file instructions.
   */
  DEFINE_FLAG(SLP_STREAM, 1)
  /*!
   * \brief The most fine grainularity.
   */
  DEFINE_FLAG(GRANULARITY, -1)
  /*!
   * \brief If the register bitstream is faked.
   */
  DEFINE_FLAG(FAKE, 0)
  /*!
   * \brief If disable array node meta info generation.
   */
  DEFINE_FLAG(NOARRAY, 0)
  /*!
   * \brief If apply buffet double buffering.
   */
  DEFINE_FLAG(DOUBLE_BUFFER, 1)
#undef DEFINE_FLAG

  TimeProfiler TP;

  bool TemporalFound{false};

  ModuleFlags() {
    getFlagIND();
    getFlagREC();
    getFlagPRED();
    getFlagTEMPORAL_FALLBACK();
    getFlagTRIGGER();
    getFlagEXTRACT();
    getFlagFUSION();
    getFlagDUMMY();
    getFlagBITSTREAM();
    getFlagCOMPAT_ADG();
    getFlagRAW();
    getFlagFUSE_STREAM();
    getFlagSLP_STREAM();
    getFlagGRANULARITY();
    getFlagFAKE();
    getFlagSEED();
    getFlagNOARRAY();
    getFlagDOUBLE_BUFFER();
  }
};

/*!
 * \brief Return the flags of this pass, a singlton.
 */
ModuleFlags &ModuleContext();

/*!
 * \brief The data structure bundle for DSA registers.
 */
struct StickyRegister {
  /*!
   * \brief The struct allocation instruction.
   */
  Value *Alloca;
  /*!
   * \brief The memory that holds the value of the register.
   */
  Value *Reg;
  /*!
   * \brief The memory that holds the value of the stickiness.
   */
  Value *Sticky;

  StickyRegister(Value *A, Value *R, Value *S) : Alloca(A), Reg(R), Sticky(S) {}
};

/*!
 * \brief Disjoint union set.
 * \param Elem The element to get the set to which it belongs.
 * \param DSU The DSU data structure.
 */
int DSUGetSet(int Elem, std::vector<int> &DSU);

/*!
 * \brief Convert the DSU to explicitly a vector of sets.
 * \param DSU The DSU to convert.
 */
std::vector<std::vector<int>> DSU2Sets(std::vector<int> &DSU);

/*!
 * \brief The loop level this input value is consumed.
 * \param IP The input port.
 * \param The loops.
 */
int consumerLevel(Value *Val, const std::vector<DFGEntry*> &Entries,
                  const std::vector<Loop*> &Loops);
/*!
 * \brief A wrapper for the array dumping.
 * \param Func The function scope this array belongs to.
 * \param Array The array to get the name.
 */
std::string nameOfLLVMValue(Function &Func, Value *Name);

} // namespace utils

using RegisterFile = std::vector<utils::StickyRegister>;

} // namespace dsa

#undef assert
#define assert(x)                                                              \
  do {                                                                         \
    if (!(x))                                                                  \
      exit(1);                                                                 \
  } while (false)

const std::string OffloadPrefix = "__ss_offload";
const std::string TemporalPrefix = "__ss_temporal";
const double Eps = 1e-5;
const int ReservedPorts[] = {23, 24, 25, 26};
const int MemScrPort = 23;

/*
 * \brief Legacy functions before I know IRBuilder.
 */
// {
CallInst *createAssembleCall(Type *Ty, StringRef Inst, StringRef Operand,
                             ArrayRef<Value *> Args, Instruction *Before);

Constant *createConstant(LLVMContext &Context, uint64_t Val, int Bits = 64);

Value *CeilDiv(Value *A, Value *B, Instruction *InsertBefore);

Value *CeilDiv(Value *A, Value *B, IRBuilder<> *IB);
// }

Value *stripCast(Value *);

const SCEV *stripCast(const SCEV *SE);

std::string funcNameToDFGName(const StringRef &Name);

template <typename Desired, typename Iteratable>
std::vector<Desired *> TypeFilter(const Iteratable &List) {
  std::vector<Desired *> Res;
  for (auto Elem : List) {
    if (auto Val = dyn_cast<Desired>(Elem)) {
      Res.push_back(Val);
    }
  }
  return Res;
}

const std::map<std::string, std::string> &spatialIntrinics();

std::string bitwiseRename(Instruction *Inst);

/*
 * \brief If the given value can be a DFG entry (i.e. instruction node).
 */
bool CanBeAEntry(Value *Inst);

/*
 * \brief If the given value is constant int one.
 */
bool isOne(Value *Val);

/*
 * \brief For the given instruction, put all the PHI node that are essentially
 *        the same values in the result set.
 */
void FindEquivPHIs(Instruction *inst, std::set<Instruction *> &res);

/*
 * \brief Convert the LLVM comparison predicate to our DFG integer.
 * \param Pred The predicate to convert.
 * \param TF TODO(@were): I do not remember this parameter.
 * \param Reverse If the predicate is reversed.
 *        For example, a > b pred will be converted to a < b when true.
 */
int PredicateToInt(ICmpInst::Predicate Pred, bool TF, bool Reverse);

/*
 * \brief Find the exit block of the given loop.
 */
BasicBlock *FindLoopEpilogue(Loop *L);


/*!
 * \brief Starting with instruction `Start` ends with the block in which `End` is,
 *        apply function `F` onto all the instructions between.
 */
void traverseAndApply(Instruction *Start, Instruction *End, DominatorTree *DT,
                      std::function<void(Instruction*)> F);

class DFGEntry;

raw_ostream &operator<<(raw_ostream &OS, DFGEntry &DE);
