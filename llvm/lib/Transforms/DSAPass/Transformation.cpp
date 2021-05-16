#include <cstdio>
#include <fstream>
#include <queue>
#include <set>
#include <sstream>
#include <utility>

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/ADT/BreadthFirstIterator.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"

#include "CodeXform.h"
#include "StreamAnalysis.h"
#include "Transformation.h"
#include "Util.h"

#include "dsa/rf.h"
#include "dsa/spec.h"
//#include "dsa/mapper/scheduler.h"
//#include "dsa/mapper/scheduler_sa.h"

using namespace llvm;

#define DEBUG_TYPE "stream-specialize"

namespace dsa {

void DFGVisitor::Visit(DfgBase *Node) {}
void DFGVisitor::Visit(DedicatedDfg *Node) { Visit(static_cast<DfgBase*>(Node)); }
void DFGVisitor::Visit(TemporalDfg *Node) { Visit(static_cast<DfgBase*>(Node)); }

namespace inject {

std::stack<IRBuilder<>*> DSAIntrinsicEmitter::REG::IBStack;

MemoryType
InjectLinearStream(IRBuilder<> *IB,
                   const std::vector<dsa::utils::StickyRegister> &Regs,
                   int PortNum, const AnalyzedStream &Stream,
                   MemoryOperation MO, Padding PP, MemoryType MT, int DType) {
  DSAIntrinsicEmitter DIE(IB, Regs);
  Value *Start = std::get<0>(Stream.Dimensions.back());
  Value *Bytes = std::get<1>(Stream.Dimensions.back());
  Value *DTyValue = IB->getInt64(DType);
  if (Stream.Dimensions.size() == 1) {
    DIE.INSTANTIATE_1D_STREAM(Start, DType, DIE.DIV(Bytes, DTyValue),
                              PortNum, PP, DSA_Access, MO, MT, DType, 0);
    return DMT_DMA;
  } else if (Stream.Dimensions.size() == 2) {
    Value *Stride = std::get<0>(Stream.Dimensions[0]);
    Value *N = std::get<1>(Stream.Dimensions[0]);
    int Stretch = std::get<2>(Stream.Dimensions[0]);
    DIE.INSTANTIATE_2D_STREAM(Start, DType,
                              DIE.DIV(Bytes, DTyValue),
                              DIE.DIV(Stride, DTyValue),
                              IB->getInt64(Stretch / DType),
                              N, PortNum, PP, DSA_Access, MO, MT, DType, 0);
    return DMT_DMA;
  } else {
    llvm::errs() << Stream.Dimensions.size();
  }
  llvm_unreachable("Unsupported stream dimension");
}

}
}

const std::string &DFGFile::getName() {
  return FileName;
}



/// Check if a given SCEV is loop invariant
bool CheckLoopInvariant(const SCEV *S, int From, const SmallVector<Loop*, 0> &LoopNest,
                        ScalarEvolution *SE) {
  for (size_t i = From; i < LoopNest.size(); ++i) {
    auto LI = LoopNest[i];
    if (!SE->isLoopInvariant(S, LI)) {
      LLVM_DEBUG(
        S->dump();
        dbgs() << "Is NOT a loop invariant under ";
        LI->dump()
      );
      return false;
    }
  }
  return true;
}

Value *DfgBase::ComputeRepeat(Value *Prime, Value *Wrapper, bool isVectorized,
                              bool isPortConfig) {
  SCEVExpander Expander(*Parent->Query->SE, Parent->Func.getParent()->getDataLayout(), "");
  auto &IB = *Parent->Query->IBPtr;
  if (IB.GetInsertPoint() != DefaultIP()->getIterator())
    IB.SetInsertPoint(DefaultIP());
  if (isVectorized) {
    if (isPortConfig) {
      if (isOne(Wrapper)) {
        auto ShiftedPrime = IB.CreateShl(Prime, 3);
        auto VectorizedPrime = IB.CreateUDiv(ShiftedPrime, UnrollConstant());
        return VectorizedPrime;
      } else {
        auto CD = CeilDiv(Prime, UnrollConstant(), DefaultIP());
        return IB.CreateShl(IB.CreateMul(CD, Wrapper), 3);
      }
    } else {
      auto CD = CeilDiv(Prime, UnrollConstant(), DefaultIP());
      return IB.CreateMul(CD, Wrapper);
    }
  } else {
    auto Res = IB.CreateMul(Prime, Wrapper);
    if (isPortConfig) {
      return IB.CreateShl(Res, 3);
    }
    return Res;
  }
}

Value *DfgBase::ComputeRepeat(const AnalyzedRepeat &AR, bool isVectorized, bool isPortConfig) {
  SCEVExpander Expander(*Parent->Query->SE, Parent->Func.getParent()->getDataLayout(), "");
  if (!AR.Prime)
    return createConstant(getCtx(), 1);
  return ComputeRepeat(Expander.expandCodeFor(AR.Prime, nullptr, DefaultIP()),
                       AR.Wrapped, isVectorized, isPortConfig);
}

/// Chech if a SCEV can be a repeat stretch
bool CanBeStretched(int x, const SmallVector<Loop*, 0> &LoopNest, ScalarEvolution *SE) {
  if (x != 1) {
    LLVM_DEBUG(dbgs() << "[Can be stretch] x != 1\n");
    return false;
  }
  if (LoopNest.size() <= 1) {
    LLVM_DEBUG(dbgs() << "[Can be stretch] Too few loops to stretch!\n");
    return false;
  }
  const SCEV *S = SE->getBackedgeTakenCount(LoopNest[0]);
  if (S->getSCEVType() != scAddRecExpr) {
    LLVM_DEBUG(dbgs() << "[Can be stretch] trip count not stretch!\n");
    return false;
  }
  if (!SE->isLoopInvariant(S, LoopNest[1])) {
    for (size_t i = 2; i < LoopNest.size(); ++i) {
      auto LI = LoopNest[i];
      if (!SE->isLoopInvariant(S, LI)) {
        LLVM_DEBUG(
          S->dump();
          dbgs() << "[Can be stretch] Is NOT a loop invariant under ";
          LI->dump()
        );
        return false;
      }
    }
    auto SARE = dyn_cast<SCEVAddRecExpr>(S)->getStart();
    if (!CheckLoopInvariant(SARE, 1, LoopNest, SE)) {
      LLVM_DEBUG(dbgs() << "[Can be stretch] trip count not invariant!\n");
      return false;
    } else {
      return true;
    }
  }
  LLVM_DEBUG(dbgs() << "[Can be stretch] invariant for the right outside loop!\n");
  return false;
}

DfgEntry *DfgBase::InThisDFG(Value *Val) {
  for (auto &Elem : Entries) {
    for (auto &Iter : Elem->UnderlyingValues()) {
      if (Iter == Val) {
        return Elem;
      }
    }
  }
  return nullptr;
}

Instruction *DfgBase::DefaultIP() {
  assert(false && "Not supported yet!");
}

Instruction *TemporalDfg::DefaultIP() {
  return IP;
}

Instruction *DedicatedDfg::DefaultIP() {
  return &Preheader->back();
}

bool DedicatedDfg::ConsumedByAccumulator(MemPort *MP) {
  std::set<Value*> Visited{MP->UnderlyingInst()};
  std::queue<Value*> Q;
  Q.push(MP->UnderlyingInst());
  while (!Q.empty()) {
    auto Cur = Q.front(); Q.pop();
    for (auto User : Cur->users()) {
      if (auto DE = InThisDFG(User)) {
        if (isa<Accumulator>(DE)) {
          return true;
        }
      }
      if (Visited.find(User) == Visited.end()) {
        Q.push(User);
        Visited.insert(User);
      }
    }
  }
  return false;
}

int ScalarBits2T(int Bits) {
  switch (Bits) {
    case 64: return 0;
    case 32: return 1;
    case 16: return 2;
    case 8: return 3;
  }
  assert(false);
}

void DFGFile::InjectStreamIntrinsics() {
  auto IB = Query->IBPtr;

  for (auto &DFG : DFGs) {
    for (auto Port : DFG->EntryFilter<InputPort>()) {
      LLVM_DEBUG(dbgs() << "Inject for "; dbgs() << Port->dump());
      Port->InjectStreamIntrinsic(IB);
    }
  }

  for (auto &DFG : DFGs) {
    for (auto Port : DFG->EntryFilter<OutputPort>()) {
      LLVM_DEBUG(dbgs() << "Inject for "; dbgs() << Port->dump());
      Port->InjectStreamIntrinsic(IB);
    }
  }

  for (auto &DD : DFGs) {
    for (auto &PB : DD->EntryFilter<PortBase>()) {
      CHECK(PB->IntrinInjected);
    }
  }

  IB->SetInsertPoint(Fence);
  dsa::inject::DSAIntrinsicEmitter(IB, Query->DSARegs).SS_WAIT_ALL();

}

Instruction *DfgBase::InjectRepeat(const AnalyzedRepeat &AR, Value *Val, int Port) {
  SCEVExpander Expander(*Parent->Query->SE, Parent->Func.getParent()->getDataLayout(), "");

  if (!AR.Prime) {
    LLVM_DEBUG(dbgs() << "No repeat!\n");
    return nullptr;
  }

  auto SE = Parent->Query->SE;
  auto IB = Parent->Query->IBPtr;
  auto One = createConstant(getCtx(), 1);

  IB->SetInsertPoint(DefaultIP());

  if (AR.PT == AnalyzedRepeat::PlainRepeat) {
    auto PrimeRepeat = Expander.expandCodeFor(AR.Prime, nullptr, DefaultIP());
    return InjectRepeat(IB->CreateAdd(PrimeRepeat, createConstant(getCtx(), 1)),
                        AR.Wrapped, 0, Val, Port);
  } else if (AR.PT == AnalyzedRepeat::StretchedRepeat) {
    assert(AR.Prime->getSCEVType() == scAddRecExpr);
    auto SARE = dyn_cast<SCEVAddRecExpr>(AR.Prime);
    auto BT =
      Expander.expandCodeFor(SE->getBackedgeTakenCount(SARE->getLoop()), nullptr, DefaultIP());
    auto SV = Expander.expandCodeFor(SARE->getStepRecurrence(*SE), nullptr, DefaultIP());
    assert(isa<ConstantInt>(SV));
    return InjectRepeat(IB->CreateAdd(BT, One), AR.Wrapped,
                        dyn_cast<ConstantInt>(SV)->getSExtValue(), Val, Port);
  } else if (AR.PT == AnalyzedRepeat::SumOfStretchedRepeat) {
    // FIXME: Kinda ugly here.
    // Once we have a better hardware support to do this we no longer need this
    assert(AR.Prime->getSCEVType() == scAddRecExpr);
    auto SARE = dyn_cast<SCEVAddRecExpr>(AR.Prime);
    auto First = IB->CreateAdd(Expander.expandCodeFor(SARE->getStart(), nullptr, DefaultIP()), One);
    auto BT =
      Expander.expandCodeFor(SE->getBackedgeTakenCount(SARE->getLoop()), nullptr, DefaultIP());
    auto SV = Expander.expandCodeFor(SARE->getStepRecurrence(*SE), nullptr, DefaultIP());
    auto SVCI = dyn_cast<ConstantInt>(SV);
    CHECK(SVCI);
    auto SVInt = SVCI->getSExtValue();
    CHECK(SVInt == 1 || SVInt == -1);
    auto Last = IB->CreateAdd(First, IB->CreateMul(BT, SV));

    switch (getUnroll()) {
      case 1: {
        // (First + Last) * (BT + 1) / 2
        auto Prime = IB->CreateUDiv(IB->CreateMul(IB->CreateAdd(First, Last), IB->CreateAdd(BT, One)),
                                    createConstant(getCtx(), 2));
        return InjectRepeat(Prime, AR.Wrapped, /*I am not sure if put 0 here is correct*/0,
                            Val, Port);
      }
      case 2: {
        Value *Range;
        if (SVInt == -1) {
          //FIXME: Support last != 0 case later.
          auto F2 = IB->CreateAdd(First, IB->getInt64(2));
          auto FF = IB->CreateUDiv(IB->CreateMul(F2, F2), createConstant(getCtx(), 4));
          Range = FF;
        } else {
          assert(false && "Not supported yet...");
        }
        Range = IB->CreateShl(Range, 1);
        return InjectRepeat(Range, AR.Wrapped, /*Same as above*/0, Val, Port);
      }
      default: {
        llvm_unreachable("Not supported vectorization width");
      }
    }
    llvm_unreachable("Not supported yet...");
    return nullptr;
  }

  llvm_unreachable("Unknown kind of AnalyzedRepeat's Repeat Type");
}

Instruction *DfgBase::InjectRepeat(Value *Prime, Value *Wrapper, int64_t Stretch,
                                   Value *Val, int Port) {

  auto IP = DefaultIP();
  auto &IB = *Parent->Query->IBPtr;
  IB.SetInsertPoint(IP);

  auto ShiftedStretch = createConstant(getCtx(), Stretch << 3);
  auto VectorizedStretch = IB.CreateSDiv(ShiftedStretch, UnrollConstant());
  auto ActualStretch = IB.CreateShl(VectorizedStretch, 1);
  (void) ActualStretch;

  dsa::inject::DSAIntrinsicEmitter DIE(Parent->Query->IBPtr, Parent->Query->DSARegs);
  if (Val == nullptr) {
    auto Repeat = IB.CreateAdd(
      IB.CreateSDiv(IB.CreateSub(Prime, IB.getInt64(1)), UnrollConstant()),
      IB.getInt64(1));
    DIE.SS_CONFIG_PORT(Port, DPF_PortRepeat, Repeat);
    if (Stretch) {
      DIE.SS_CONFIG_PORT(Port, DPF_PortPeriod, getUnroll());
      DIE.SS_CONFIG_PORT(Port, DPF_PortRepeatStretch, Stretch);
    }
    return DIE.res.back();
  } else {
    LLVM_DEBUG(dbgs() << "Inject Repeat: "; Val->dump();
               dbgs() << "For port: " << Port);
    assert(Port != -1);
    DIE.SS_CONST(Port, Val,
                 /*Times=*/ComputeRepeat(Prime, Wrapper, true, false),
                 /*Const Type*/Val->getType()->getScalarSizeInBits() / 8);
    return DIE.res.back();
  }
}

DfgBase *DfgBase::BelongOtherDFG(Instruction *I) {
  for (auto &DFG : Parent->DFGs) {
    if (DFG == this)
      continue;
    if (DFG->Contains(I))
      return DFG;
  }
  return nullptr;
}

std::pair<std::set<Instruction*>, std::set<Instruction*>>
BFSOperands(DfgBase *DFG, DominatorTree *DT, Instruction *From) {

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
          if (DFG->InThisScope(Inst)) {
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

Predicate* DfgBase::FindEquivPredicate(Value *LHS, Value *RHS) {
  for (auto Elem : EntryFilter<Predicate>()) {
    if (Elem->Cond[0]->getOperand(0) == LHS && Elem->Cond[0]->getOperand(1) == RHS)
      return Elem;
    if (Elem->Cond[0]->getOperand(1) == LHS && Elem->Cond[0]->getOperand(0) == RHS)
      return Elem;
  }
  return nullptr;
}

bool DedicatedDfg::Contains(const Loop *L) {
  for (auto Elem : LoopNest) {
    if (Elem == L)
      return true;
  }
  return false;
}


Value *DedicatedDfg::TripCount(int x, Instruction *InsertBefore) {
  SCEVExpander Expander(*Parent->Query->SE, Parent->Func.getParent()->getDataLayout(), "");
  auto IB = Parent->Query->IBPtr;
  auto BackSE = Parent->Query->SE->getBackedgeTakenCount(LoopNest[x]);
  assert(CheckLoopInvariant(BackSE, x + 1, LoopNest, Parent->Query->SE));
  auto BackTaken = Expander.expandCodeFor(BackSE, nullptr, InsertBefore);
  IB->SetInsertPoint(InsertBefore);
  auto Trip = IB->CreateAdd(BackTaken, createConstant(getCtx(), 1), "trip.count");
  return Trip;
}

Value *DedicatedDfg::ProdTripCount(int l, int r, Instruction *InsertBefore) {
  Value *Prod = createConstant(getCtx(), 1);
  for (int i = l; i < r; ++i) {
    Prod = BinaryOperator::Create(BinaryOperator::Mul, Prod, TripCount(i, InsertBefore),
                                  "trip.prod", InsertBefore);
  }
  return Prod;
}

Value *DedicatedDfg::ProdTripCount(int x, Instruction *InsertBefore) {
  return ProdTripCount(0, x, InsertBefore);
}

DedicatedDfg::DedicatedDfg(DFGFile *Parent_, Loop *LI, int Unroll)
  : DfgBase(Parent_), UnrollFactor(std::max(1, Unroll)) {

  Kind = DfgBase::Dedicated;

  if (GetUnrollMetadata(LI->getLoopID(), "llvm.loop.ss.datamove")) {
    LLVM_DEBUG(dbgs() << "A data-move 'DFG'\n");
  }

  LoopNest.push_back(LI);
  while (LoopNest.back()->getLoopDepth() > 1) {
    auto Cur = LoopNest.back();
    LLVM_DEBUG(dbgs() << "Loop level: "; Cur->dump());

    if (GetUnrollMetadata(Cur->getLoopID(), "llvm.loop.ss.stream"))
      break;

    LoopNest.push_back(Cur->getParentLoop());
  }

  assert(GetUnrollMetadata(LoopNest.back()->getLoopID(), "llvm.loop.ss.stream") &&
         "A stream level required!");

  Loop *InnerLoop = nullptr;
  std::set<BasicBlock*> Exclude;

  for (auto LI : LoopNest) {
    LLVM_DEBUG(dbgs() << "Loop: "; LI->dump());
    for (auto Child : LI->getSubLoops()) {
      if (Child != InnerLoop) {
        for (auto &Block : Child->getBlocks())
          Exclude.insert(Block);
      }
    }

    for (auto &Block : LI->getBlocks()) {
      if (!Exclude.count(Block))
        Blocks.insert(Block);
    }

    InnerLoop = LI;
  }


  LLVM_DEBUG(
    dbgs() << "DFG" << (void*)(this) << ": ";
    for (auto Block : Blocks) {
      dbgs() << (void*) Block << ", ";
    }
  );


  LLVM_DEBUG(dbgs() << "Init blocks done\n");

  auto OuterMost = LoopNest.back();
  if (!(Preheader = OuterMost->getLoopPreheader())) {
    Preheader = InsertPreheaderForLoop(OuterMost, Parent->Query->DT, Parent->Query->LI, nullptr, false);
  }

  auto Prologue = FindLoopPrologue(LoopNest.back());
  assert(Prologue);
  PrologueIP = &Prologue->back();
}


void DfgBase::dump(std::ostringstream &os) {

  if (!Entries.empty()) {
    auto BF = Parent->Query->BFI->getBlockFreq(getBlocks()[0]);
    os << "#pragma group frequency " << BF.getFrequency() << "\n";
  }

  if (getUnroll() != 1) {
    os << "#pragma group unroll " << getUnroll() << "\n";
  }

  for (auto Elem : EntryFilter<InputPort>())
    Elem->EmitInPort(os);

  for (auto Elem : Entries) {
    if (auto CB = dyn_cast<ComputeBody>(Elem))
      CB->EmitCB(os);
    if (auto P = dyn_cast<Predicate>(Elem))
      P->EmitCond(os);
  }

  /// Emit intermediate result for atomic operation
  for (auto Elem : EntryFilter<ComputeBody>())
    Elem->EmitAtomic(os);

  for (auto Elem : EntryFilter<OutputPort>())
    Elem->EmitOutPort(os);

  // TODO(@were): Move this to a DFG.
  for (auto Elem : EntryFilter<IndMemPort>()) {
    os << "\n\n----\n\n";
    os << "Input" << Elem->Index->Load->getType()->getScalarSizeInBits()
       << ": indirect_in_" << ID << "_" << Elem->ID << "\n";
    os << "indirect_out_" << ID << "_" << Elem->ID << " = "
       << "indirect_in_" << ID << "_" << Elem->ID << "\n";
    os << "Output" << Elem->Index->Load->getType()->getScalarSizeInBits()
       << ": indirect_out_" << ID << "_" << Elem->ID << "\n";
  }

}

void DFGFile::addDfg(DfgBase *DB) {
  assert(!AllInitialized && "You can no longer modify the dfg after scheduling");
  DFGs.push_back(DB);
}

void DFGFile::EmitAndScheduleDfg() {

  dsa::xform::EmitDFG(nullptr, this);


  if (dsa::utils::ModuleContext().EXTRACT) {
    return;
  }

  auto SCHEDULED = getenv("SCHEDULED");
  std::string NameOnly;

  if (SCHEDULED) {
    std::string SCHEDString(SCHEDULED);
    int From = SCHEDString.size(), Len = 0;
    while (SCHEDString[From - 1] != '/') {
      --From;
      ++Len;
    }
    NameOnly = SCHEDString.substr(From, Len);
  } else {
    auto SBCONFIG = getenv("SBCONFIG");
    CHECK(SBCONFIG) << "Please specify CGRA config environment variable $SBCONFIG";
    LLVM_DEBUG(
      dbgs() << "Emitted DFG:\n";
      CHECK(system(formatv("cat {0} 1>&2", FileName).str().c_str()) == 0);
      dbgs() << "\n\n";
    );

    //SS_CONFIG::SSModel model(SBCONFIG);
    //SSDfg dfg(FileName);
    //SchedulerSimulatedAnnealing *sa = new SchedulerSimulatedAnnealing(&model);
    //sa->verbose = DebugFlag && isCurrentDebugType(DEBUG_TYPE);
    //sa->str_subalg = "";
    //sa->set_max_iters(Query->MF.EXTRACT ? 1 : 20000);
    //sa->setTimeout(20);
    //sa->setGap(0.1, 1.0);
    //Schedule *sched = sa->invoke(&model, &dfg, false);
    std::string Verbose = isCurrentDebugType(DEBUG_TYPE) ? "-v" : "";
    auto Cmd = formatv("ss_sched {0} {1} {2} -e 0 > /dev/null", Verbose, SBCONFIG, FileName);
    LLVM_DEBUG(dbgs() << Cmd);
    llvm::errs() << Cmd;
    if (system(Cmd.str().c_str()) != 0) {
      // TODO(@were): throw exception here.
    }
  }


  std::ifstream ifs(SCHEDULED ? std::string(SCHEDULED) : formatv("{0}.h", FileName).str());

  std::string Stripped(SCHEDULED ? NameOnly.substr(0, NameOnly.size() - 2) : FileName);
  while (Stripped.back() != '.') Stripped.pop_back();
  Stripped.pop_back();
  LLVM_DEBUG(dbgs() << "Stripped: " << Stripped << "\n");

  int ConfigSize = 0;
  std::string ConfigString;

  std::string Line;
  std::string PortPrefix(formatv("P_{0}_", Stripped).str());
  while (std::getline(ifs, Line)) {
    std::istringstream iss(Line);
    std::string Token;
    iss >> Token;
    // #define xxx
    if (Token == "#define") {
      iss >> Token;
      // #define P_dfgX_subY_vZ
      if (Token.find(PortPrefix + "sub") == 0) {
        int X, Y;
        Token = Token.substr(PortPrefix.size() + 3, Token.size());
        CHECK(sscanf(Token.c_str(), "%d_v%d", &X, &Y) == 2);
        int Port;
        iss >> Port;
        LLVM_DEBUG(dbgs() << "sub" << X << "v" << Y << " -> " << Port << "\n");
        auto Entry = DFGs[X]->Entries[Y];
        if (auto PB = dyn_cast<PortBase>(Entry)) {
          PB->SoftPortNum = Port;
        } else if (auto CB = dyn_cast<ComputeBody>(Entry)) {
          if (!CB->isImmediateAtomic()) {
            LLVM_DEBUG(llvm::dbgs() << "OutputPorts:\n");
            auto OutPorts = CB->GetOutPorts();
            for (auto &Port : CB->GetOutPorts()) {
              LLVM_DEBUG(Port->UnderlyingInst()->dump());
            }
            // FIXME: For now only one destination is supported
            // Later divergence will be supported
            CHECK(OutPorts.size() == 1) << *CB->UnderlyingInst() << " " << OutPorts.size();
            OutPorts[0]->SoftPortNum = Port;

            // Fill in the latency of each node.
            //{
            //  OutputPort *OP = OutPorts[0];
            //  std::string ON = OP->Parent->InThisDFG(OP->Output)->NameInDfg();
            //  for (auto SDVO : dfg.nodes<SSDfgVecOutput*>()) {
            //    if (SDVO->name() == ON) {
            //      OP->Latency = sched->latOf(SDVO);
            //      LLVM_DEBUG(dbgs() << "[lat] " << ON << ": " << OP->Latency << "\n");
            //      break;
            //    }
            //  }
            //}

          } else {
            CB->SoftPortNum = Port;
          }
        } else {
          assert(false && "What's going on?");
        }
      // #define dfgx_size size
      } else if (Token.find(formatv("{0}_size", Stripped).str()) == 0) {
        iss >> ConfigSize;
      } else if (Token.find(PortPrefix + "indirect_") == 0) {
        Token = Token.substr(std::string(PortPrefix + "indirect_").size());
        int X, Y;
        int Port;
        if (sscanf(Token.c_str(), "in_%d_%d", &X, &Y) == 2) {
          auto IMP = dyn_cast<IndMemPort>(DFGs[X]->Entries[Y]);
          CHECK(IMP);
          iss >> Port;
          IMP->Index->SoftPortNum = Port;
        } else {
          CHECK(sscanf(Token.c_str(), "out_%d_%d", &X, &Y) == 2);
          auto IMP = dyn_cast<IndMemPort>(DFGs[X]->Entries[Y]);
          CHECK(IMP);
          iss >> Port;
          IMP->IndexOutPort = Port;
        }
      }
    // char dfgx_config[size] = "filename:dfgx.sched";
    } else if (Token == "char") {
      // dfgx_config[size]
      iss >> Token;
      // =
      iss >> Token;
      // "filename:dfgx.sched";
      iss >> Token;
      ConfigString = Token.substr(1, Token.size() - 3);
      ConfigString += std::string(ConfigSize - ConfigString.size() - 1, '\0');
    }
  }

  LLVM_DEBUG(dbgs() << "[Config] " << ConfigSize << ": " << ConfigString << "\n");

  auto Module = Func.getParent();
  auto ConfigData = ConstantDataArray::getString(getCtx(), ConfigString);
  auto GV = new GlobalVariable(*Module, ConfigData->getType(), false, GlobalValue::ExternalLinkage,
                               ConfigData, formatv("{0}_config", Stripped));
  GV->setAlignment(llvm::MaybeAlign(1));
  GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Local);

  LLVM_DEBUG(dbgs() << "Inject global configuration value "; GV->dump());

  auto APConfigSize = APInt(64, ConfigSize);
  auto ConstConfig = Constant::getIntegerValue(Type::getInt64Ty(getCtx()), APConfigSize);

  auto IB = Query->IBPtr;
  IB->SetInsertPoint(Config);
  dsa::inject::DSAIntrinsicEmitter DIE(IB, Query->DSARegs);
  DIE.SS_CONFIG(GV, ConstConfig);
  LLVM_DEBUG(dbgs() << "Inject config asm "; DIE.res.back()->dump());
}

void DFGFile::EraseOffloadedInstructions() {

  std::set<Instruction*> Unique;

  for (auto &DFG : DFGs) {
    for (auto Entry: DFG->Entries) {

      // Skip instructions cannot be offloaded because of flags
      if (isa<IndMemPort>(Entry) && !dsa::utils::ModuleContext().IND) {
        continue;
      }
      if (isa<AtomicPortMem>(Entry) && !dsa::utils::ModuleContext().IND) {
        continue;
      }
      if ((isa<CtrlMemPort>(Entry) || isa<Predicate>(Entry)) && !dsa::utils::ModuleContext().PRED) {
        continue;
      }

      auto F = [&Unique] (Instruction *Inst) {
        bool SS = false;
        LLVM_DEBUG(Inst->dump());
        for (auto User : Inst->users()) {
          LLVM_DEBUG(dbgs() << "user: "; User->dump());
          if (auto Call = dyn_cast<CallInst>(User)) {
            if (auto IA = dyn_cast<InlineAsm>(Call->getCalledOperand())) {
              SS |= IA->getAsmString().find("ss_") == 0;
            }
          }
        }
        for (size_t i = 0; i < Inst->getNumOperands(); ++i) {
          auto Use = Inst->getOperand(i);
          LLVM_DEBUG(dbgs() << "use: "; Use->dump());
          if (auto Call = dyn_cast<CallInst>(Use)) {
            if (auto IA = dyn_cast<InlineAsm>(Call->getCalledOperand())) {
              SS |= IA->getAsmString().find("ss_") == 0;
            }
          }
        }
        if (!SS) {
          Unique.insert(Inst);
          LLVM_DEBUG(dbgs() << "DELETE: "; Inst->dump());
        }
        LLVM_DEBUG(dbgs() << "\n");
      };

      if (auto P = dyn_cast<Predicate>(Entry)) {
        for (auto Elem : P->Cond)
          F(Elem);
      } else if (Entry->UnderlyingInst()) {
        F(Entry->UnderlyingInst());
      }
    }

    if (auto DD = dyn_cast<DedicatedDfg>(DFG)) {
      auto StreamMD = GetUnrollMetadata(DD->LoopNest.back()->getLoopID(), "llvm.loop.ss.stream");
      auto BarrierMD = dyn_cast<ConstantAsMetadata>(StreamMD->getOperand(1));
      if (BarrierMD->getValue()->getUniqueInteger().getSExtValue()) {
        // Find the "break" finger, and inject the wait at the "break" finger block's end
        auto OutBB = FindLoopPrologue(DD->LoopNest.back());
        Instruction *I = &OutBB->back();
        for (; !isa<PHINode>(I); I = I->getPrevNode()) {
          bool Found = false;
          for (auto Elem : DD->InjectedCode) {
            if (I == Elem) {
              Found = true;
              break;
            }
          }
          if (isa<PHINode>(I) || Found) {
            auto IB = Query->IBPtr;
            IB->SetInsertPoint(I->getNextNode());
            dsa::inject::DSAIntrinsicEmitter(IB, Query->DSARegs).SS_WAIT_ALL();
            break;
          }
          if (I == &I->getParent()->front()) {
            auto IB = Query->IBPtr;
            IB->SetInsertPoint(I);
            dsa::inject::DSAIntrinsicEmitter(IB, Query->DSARegs).SS_WAIT_ALL();
            break;
          }
        }

      } else {
        LLVM_DEBUG(dbgs() << "No barrier required!\n");
      }
    } else if (auto TD = dyn_cast<TemporalDfg>(DFG)) {
      TD->End->eraseFromParent();
      TD->Begin->eraseFromParent();
    }
  }

  for (auto Inst : Unique) {
    Inst->replaceAllUsesWith(UndefValue::get(Inst->getType()));
    Inst->eraseFromParent();
  }

  LLVM_DEBUG(dbgs() << "Rip out all original instrucitons\n");

}

Loop *DedicatedDfg::InnerMost() {
  return LoopNest.front();
}

Loop *DedicatedDfg::OuterMost() {
  return LoopNest.back();
}

DfgBase::DfgBase(DFGFile *Parent_) :
  Parent(Parent_) {
  ID = Parent_->DFGs.size();
  Kind = DfgBase::Unknown;
}

DfgBase::DfgKind DfgBase::getKind() const {
  return Kind;
}

TemporalDfg::TemporalDfg(DFGFile *Parent, IntrinsicInst *Begin, IntrinsicInst *End)
  : DfgBase(Parent), Begin(Begin), End(End) {

  std::set<Instruction*> users;
  for (Instruction *I = Begin; I != End; I = I->getNextNode()) {
    for (auto Usr : I->users()) {
      if (auto Inst = dyn_cast<Instruction>(Usr)) {
        if (Inst->getParent() == Begin->getParent()) {
          users.insert(Inst);
        }
      }
    }
  }

  for (auto &Inst : *Begin->getParent()) {
    if (users.count(&Inst)) {
      IP = &Inst;
    }
  }

  Kind = DfgBase::Temporal;
}

void TemporalDfg::dump(std::ostringstream &os) {
  if (!os.str().empty())
    os << "\n----\n\n";
  os << "#pragma group temporal\n";
  DfgBase::dump(os);
}

void DedicatedDfg::dump(std::ostringstream &os) {
  if (!os.str().empty())
    os << "\n----\n\n";
  if (dsa::utils::ModuleContext().TRIGGER) {
    CHECK(dsa::utils::ModuleContext().TEMPORAL) << "Trigger cannot be enabled without temporal";
    os << "#pragma group temporal\n";
  }
  DfgBase::dump(os);
}

bool DedicatedDfg::Contains(Instruction *Inst) {

  // FIXME(@were): I forgot what this is doing.

  // if (Inst->getParent() == Parent->Config->getParent()) {
  //   for (Instruction *I = Parent->Config, *E = &Parent->Config->getParent()->back();
  //        I != E; I = I->getNextNode()) {
  //     if (I == Inst)
  //       return true;
  //   }
  // }

  // if (Inst->getParent() == Parent->Fence->getParent()) {
  //   for (Instruction *I = Inst, *E = &Parent->Fence->getParent()->back();
  //        I != E; I = I->getNextNode()) {
  //     if (I == Inst)
  //       return true;
  //   }
  // }

  return Contains(Inst->getParent());
}

bool DedicatedDfg::Contains(BasicBlock *BB) {
  // FIXME: I hope for now it is good enough.
  //        Later maybe we need to dive into the paritally contain thing.
  return Blocks.find(BB) != Blocks.end();
}

bool TemporalDfg::Contains(Instruction *Inst) {
  for (Instruction *I = Begin; I != End; I = I->getNextNode())
    if (I == Inst)
      return true;
  return false;
}

bool TemporalDfg::Contains(BasicBlock *BB) {
  return BB == Begin->getParent();
}

int TemporalDfg::getUnroll() {
  return 1;
}

int DedicatedDfg::getUnroll() {
  return UnrollFactor;
}

AnalyzedStream DedicatedDfg::AnalyzeDataStream(Value *Index, int Bytes) {
  return AnalyzeDataStream(Index, Bytes, false, nullptr);
}

AnalyzedStream
DedicatedDfg::AnalyzeDataStream(Value *Index, int Bytes, bool DoOuterRepeat,
                                Instruction *InsertBefore) {

  SCEVExpander Expander(*Parent->Query->SE, Parent->Func.getParent()->getDataLayout(), "");
  auto IB = Parent->Query->IBPtr;

  if (InsertBefore == nullptr)
    InsertBefore = DefaultIP();

  IB->SetInsertPoint(InsertBefore);

  assert(InsertBefore);

  LLVM_DEBUG(dbgs() << "Analyzing index "; Index->dump();
             dbgs() << "Bytes: " << Bytes << "\n");

  auto SE = Parent->Query->SE;
  AnalyzedStream Res;
  Value *One = createConstant(getCtx(), 1);
  Res.AR.Wrapped = One;

  size_t i = 0;
  for (; i < LoopNest.size(); ++i) {
    auto LI = LoopNest[i];
    LLVM_DEBUG(llvm::dbgs() << "Loop: "; LI->dump());
    if (!LI->isLoopInvariant(Index)) {
      LLVM_DEBUG(dbgs() << "Non-Invariant here!\n");
      break;
    }
  }

  if (i) {
    Res.AR.Prime = SE->getBackedgeTakenCount(LoopNest[0]);
  }
  if (CanBeStretched(i, LoopNest, SE)) {
    Res.AR.PT = AnalyzedRepeat::StretchedRepeat;
  } else {
    if (CanBeStretched(1, LoopNest, SE)) {
      Res.AR.PT = AnalyzedRepeat::SumOfStretchedRepeat;
      Res.AR.Wrapped = ProdTripCount(2, i, InsertBefore);
    } else if (i) {
      Res.AR.Wrapped = ProdTripCount(1, i, InsertBefore);
    }
  }

  if (!SE->isSCEVable(Index->getType())) {
    assert(DoOuterRepeat);
    Res.OuterRepeat = ProdTripCount(i, LoopNest.size(), InsertBefore);
    return Res;
  }

  auto EV = SE->getSCEV(Index);
  LLVM_DEBUG(EV->dump(); dbgs() << "Index (" << EV->getSCEVType() << "): "; EV->dump());

  if (i == LoopNest.size()) {
    LLVM_DEBUG(dbgs() << "Prime Repeat: "; Res.AR.Prime->dump();
               dbgs() << "Wrapped Repeat: "; Res.AR.Wrapped->dump();
               dbgs() << "\n");
    return Res;
  }

  if (EV->getSCEVType() != scAddRecExpr) {
    assert(DoOuterRepeat);
    Res.OuterRepeat = ProdTripCount(i, LoopNest.size(), InsertBefore);
    return Res;
  }

  assert(EV->getSCEVType() == scAddRecExpr && "The loop should starts with a polyhedral");

  int RawSize = Bytes;
  Value *Cnt = createConstant(getCtx(), RawSize);

  int Stretch = 0;

  for (bool FirstExec = true; i < LoopNest.size(); ++i, FirstExec = false) {

    auto Cur = dyn_cast<SCEVAddRecExpr>(EV);
    LLVM_DEBUG(EV->dump());

    if (DoOuterRepeat && (!Cur || Cur->getLoop() != LoopNest[i])) {
      IB->SetInsertPoint(InsertBefore);
      Res.OuterRepeat = ProdTripCount(i + 1, LoopNest.size(), InsertBefore);
      LLVM_DEBUG(dbgs() << "Loop gap detected! Get outer repeat!\n");
      break;
    }

    Value *Step = nullptr;
    if (Cur) {
      assert(CheckLoopInvariant(Cur->getStepRecurrence(*SE), i + 2, LoopNest, SE));
      Step = Expander.expandCodeFor(Cur->getStepRecurrence(*SE), nullptr, InsertBefore);
      auto L = Cur->getLoop();
      LLVM_DEBUG(
        dbgs() << "Cur: "; Cur->getLoop()->dump();
        dbgs() << "Nest:"; LoopNest[i]->dump();
      );
      if (L != LoopNest[i]) {
        // FIXME: Later check required, but I think it is correct for now.
        assert(L->contains(LoopNest[i]));
        Step = createConstant(getCtx(), 0);
      } else {
        EV = Cur->getStart();
      }
      LLVM_DEBUG(dbgs() << "======= Loop Induction =======\n");
    } else if (SE->isLoopInvariant(EV, LoopNest[i])) {
      Step = createConstant(getCtx(), 0);
      LLVM_DEBUG(dbgs() << "======= Loop Invariant =======\n");
    } else if (auto SAE = dyn_cast<SCEVAddExpr>(EV)) {
      SAE->dump();
      Index->dump();
      assert(false);
    }

    int StretchInt = 0;
    auto BackTaken = SE->getBackedgeTakenCount(LoopNest[i]);
    Value *DimTripCount = nullptr;

    if (CheckLoopInvariant(BackTaken, i + 1, LoopNest, SE)) {
      assert(CheckLoopInvariant(BackTaken, i + 2, LoopNest, SE));
      DimTripCount = TripCount(i, InsertBefore);
    } else {
      /// Support triangular streams
      if (!SE->isLoopInvariant(BackTaken, LoopNest[i + 1])) {
        assert(CheckLoopInvariant(BackTaken, i + 2, LoopNest, SE));
      }
      auto BackTaken_ = dyn_cast<SCEVAddRecExpr>(BackTaken);
      assert(BackTaken_);
      auto Cnt = Expander.expandCodeFor(BackTaken_->getStart(), nullptr, InsertBefore);
      assert(CheckLoopInvariant(BackTaken_->getStart(), i + 1, LoopNest, SE));
      IB->SetInsertPoint(InsertBefore);
      DimTripCount = IB->CreateAdd(Cnt, One, "trip.count");
      Value *NextStretch = nullptr;
      if (i + 1 < LoopNest.size() && BackTaken_->getLoop() == LoopNest[i + 1]) {
        NextStretch = Expander.expandCodeFor(BackTaken_->getStepRecurrence(*SE), nullptr,
                                             InsertBefore);
      } else {
        NextStretch = createConstant(getCtx(), 0);
      }
      auto StretchConst = dyn_cast<Constant>(NextStretch);
      assert(StretchConst);
      StretchInt = StretchConst->getUniqueInteger().getSExtValue();
      LLVM_DEBUG(dbgs() << "Stretch: "; NextStretch->dump());
    }

    /// Coalease the inner most dimension
    if (FirstExec) {
      bool Coalease = false;
      if (isa<Constant>(Cnt)) {
        int64_t CntInt = dyn_cast<Constant>(Cnt)->getUniqueInteger().getSExtValue();
        StretchInt *= CntInt;
        if (isa<Constant>(Step)) {
          int64_t StepInt = dyn_cast<Constant>(Step)->getUniqueInteger().getSExtValue();
          if (CntInt == StepInt) {
            IB->SetInsertPoint(InsertBefore);
            Cnt = IB->CreateMul(Step, DimTripCount);
            Coalease = true;
          }
        }
      }

      if (Coalease) {
        Stretch = StretchInt;
        LLVM_DEBUG(dbgs() << "Coaleased!\nStretch: " << StretchInt << "\n");
        continue;
      }
    }

    Res.Dimensions.emplace_back(Step, DimTripCount, Stretch);

    LLVM_DEBUG(
      dbgs() << "Injected: "; InsertBefore->dump();
      dbgs() << "Current: "; Step->dump();
      dbgs() << "Stretch: " << Stretch;
      dbgs() << "\n"
    );

    Stretch = StretchInt;
  }

  LLVM_DEBUG(
    dbgs() << "After exiting: ";
    EV->dump()
  );

  Value* Start = nullptr;

  Start = Expander.expandCodeFor(EV, nullptr, &Preheader->back());
  if (auto SARE = dyn_cast<SCEVAddRecExpr>(EV)) {
    if (SARE->getLoop() != LoopNest.back() && SARE->getLoop()->contains(LoopNest.back())) {
      Start = Expander.expandCodeFor(SARE->getStart(), nullptr, &Preheader->back());
      auto IndVar = SARE->getLoop()->getCanonicalInductionVariable();
      assert(IndVar);
      auto Stride = Expander.expandCodeFor(SARE->getStepRecurrence(*SE), nullptr, InsertBefore);
      IB->SetInsertPoint(InsertBefore);
      auto Delta = IB->CreateMul(IndVar, Stride);
      auto Casted = IB->CreatePtrToInt(Start, Delta->getType());
      Start = IB->CreateAdd(Casted, Delta, "", InsertBefore);
      LLVM_DEBUG(dbgs() << "Due to out of loop contribution: "; Start->dump());
    } else {
      // FIXME: Why did I leave this branch empty?
    }
  }

  Res.Dimensions.emplace_back(Start, Cnt, 0);

  LLVM_DEBUG(dbgs() << "\n");

  return Res;
}

Value *AnalyzedStream::BytesFromMemory(IRBuilder<> *IB) {
  Value *Res = createConstant(IB->getContext(), 1);
  for (size_t i = 0; i < Dimensions.size(); ++i)
    Res = IB->CreateMul(Res, std::get<1>(Dimensions[i]), "");
  return Res;
}


Value *DfgBase::UnrollConstant() {
  return Parent->Query->IBPtr->getInt64(getUnroll());
}


LLVMContext &DfgBase::getCtx() {
  return Parent->getCtx();
}


LLVMContext &DFGFile::getCtx() {
  return Query->IBPtr->getContext();
}

template<typename DfgType>
void AddDfg(DFGFile *DF, MDNode *Ptr, Loop *LI) {
  assert(Ptr->getNumOperands() == 2);
  auto MDFactor = dyn_cast<ConstantAsMetadata>(Ptr->getOperand(1));
  assert(MDFactor);
  int Factor =(int) MDFactor->getValue()->getUniqueInteger().getSExtValue();
  DF->addDfg(new DfgType(DF, LI, Factor));
}

DFGFile::DFGFile(StringRef Name, IntrinsicInst *Start, IntrinsicInst *End, StreamSpecialize *Query_)
  : FileName(Name), Func(*Start->getParent()->getParent()), Config(Start), Fence(End), Query(Query_) {}

int getPortImpl() {
  static const int Ports[] = {23, 24, 25, 26, 27, 28, 29, 30, 31};
  static const int N = (sizeof Ports) / (sizeof(int));
  static int Cur = 0;
  int Res = Ports[Cur];
  Cur = (Cur + 1) % N;
  return Res;
}

int DfgBase::getNextIND() {
  return getPortImpl();
  //static const int Ports[] = {27, 28, 29, 30, 31};
  //static const int N = (sizeof Ports) / (sizeof(int));
  //static int Cur = 0;
  //int Res = Ports[Cur];
  //Cur = (Cur + 1) % N;
  //return Res;
}

int DfgBase::getNextReserved() {
  return getPortImpl();
  //static const int Ports[] = {23, 24, 25, 26};
  //static const int N = (sizeof Ports) / (sizeof(int));
  //static int Cur = 0;
  //int Res = Ports[Cur];
  //Cur = (Cur + 1) % N;
  //return Res;
}

SmallVector<BasicBlock*, 0> DedicatedDfg::getBlocks() {
  SmallVector<BasicBlock*, 0> Res(InnerMost()->getBlocks().begin(),
                                  InnerMost()->getBlocks().end());
  return Res;
}

SmallVector<BasicBlock*, 0> TemporalDfg::getBlocks() {
  SmallVector<BasicBlock*, 0> Res{Begin->getParent()};
  return Res;
}

AnalyzedStream TemporalDfg::AnalyzeDataStream(Value *Index, int Bytes) {
  AnalyzedStream res;
  auto &Ctx = Parent->getCtx();
  res.AR.Wrapped = createConstant(Ctx, 1);
  res.Dimensions.emplace_back(Index, createConstant(Ctx, Bytes), 0);
  return res;
}

bool DedicatedDfg::InThisScope(Instruction *I) {
  return LoopNest.back()->getBlocksSet().count(I->getParent());
}

bool TemporalDfg::InThisScope(Instruction *Inst) {
  for (Instruction *I = Begin; I != End; I = I->getNextNode()) {
    if (I == Inst) {
      return true;
    }
  }
  return false;
}

void DFGFile::InspectSPADs() {

  AddrsOnSpad.clear();

  Function *start = nullptr, *end = nullptr;

  for (auto &Iter : *Func.getParent()) {
    if (Iter.getName().startswith("llvm.lifetime.start")) {
      start = &Iter;
    }
    if (Iter.getName().startswith("llvm.lifetime.end")) {
      end = &Iter;
    }
  }

  auto hasAlloc = [](Function &F) {
    for (auto &BB : F) {
      for (auto &I : BB) {
        if (isa<AllocaInst>(&I)) {
          return true;
        }
      }
    }
    return false;
  };

  if (!hasAlloc(Func)) {
    return;
  }

  if (!start) {
    assert(!end);
    return;
  }

  auto IP = Config;
  auto IBPtr = Query->IBPtr;
  if (!isa<BranchInst>(IP))
    IP = IP->getNextNode();
  IBPtr->SetInsertPoint(IP);
  auto SpadPtr = IBPtr->CreateAlloca(IBPtr->getInt64Ty(), IBPtr->getInt64(1));
  auto SpadI8Ptr = IBPtr->CreateBitCast(SpadPtr, IBPtr->getInt8PtrTy());
  auto Self = IBPtr->CreateCall(start, {IBPtr->getInt64(8), SpadI8Ptr});
  auto GEP = IBPtr->CreateGEP(SpadPtr, IBPtr->getInt64(0));
  IBPtr->CreateStore(IBPtr->getInt64(0), GEP);

  std::vector<Instruction*> ToInject;
  auto DT = Query->DT;

  for (auto &BB : Func) {
    if (!DT->dominates(&BB.front(), Fence))
      continue;
    if (&BB != Config->getParent() && !DT->dominates(Config, &BB))
      continue;
    auto range = make_range(&BB != Config->getParent() ? BB.begin() : Config->getIterator(),
                            &BB != Fence->getParent() ? BB.end() : Fence->getIterator());
    for (auto &Inst : range) {
      if (isa<ReturnInst>(Inst)) {
        ToInject.push_back(&Inst);
      }

      auto Intrin = dyn_cast<IntrinsicInst>(&Inst);
      if (!Intrin)
        continue;
      if (Intrin == Self)
        continue;

      if (Intrin->getIntrinsicID() == Intrinsic::lifetime_start ||
          Intrin->getIntrinsicID() == Intrinsic::lifetime_end) {
        ToInject.push_back(&Inst);
      }

    }

  }

  for (auto Inst : ToInject) {
    if (isa<ReturnInst>(Inst)) {
      IBPtr->SetInsertPoint(Inst);
      IBPtr->CreateCall(end, {IBPtr->getInt64(8), SpadI8Ptr});
    }
    if (auto Intrin = dyn_cast<IntrinsicInst>(Inst)) {

      auto f = [this, GEP, IBPtr](IntrinsicInst *Intrin) {
        assert(Intrin->getNumOperands() == 3);
        auto Size = Intrin->getOperand(0);
        // Calling an LLVM function will put this function as the last argument of the
        // CallSite.
        auto CI = dyn_cast<CastInst>(Intrin->getOperand(1));
        assert(CI);
        auto AI = dyn_cast<AllocaInst>(CI->getOperand(0));
        IBPtr->SetInsertPoint(Intrin);
        auto Ptr = IBPtr->CreateLoad(GEP);
        Value *Val = nullptr;
        switch (Intrin->getIntrinsicID()) {
        case Intrinsic::lifetime_start:
          AddrsOnSpad[AI] = Ptr;
          AddrsOnSpad[CI] = Ptr;
          Val = IBPtr->CreateAdd(Ptr, Size);
          break;
        case Intrinsic::lifetime_end:
          Val = IBPtr->CreateSub(Ptr, Size);
          break;
        default:
          assert(0);
        }
        IBPtr->CreateStore(Val, GEP);
      };

      f(Intrin);
    }
  }

}

void DfgBase::Accept(dsa::DFGVisitor *Visitor) { Visitor->Visit(this); }
void DedicatedDfg::Accept(dsa::DFGVisitor *Visitor) { Visitor->Visit(this); }
void TemporalDfg::Accept(dsa::DFGVisitor *Visitor) { Visitor->Visit(this); }
