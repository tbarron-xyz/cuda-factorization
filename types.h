#ifndef TYPES_H
#define TYPES_H

#include "config.h"
#include <cstdio>

// Forward declarations for functions that will be defined in memory.h
struct HostAndDeviceAllocations;
HostAndDeviceAllocations allocateEverything(int bufferNumElements, int bufferSize, int outputsSize, int memoSize);
void freeEverything(HostAndDeviceAllocations& a);

// Type aliases for factorizations
template <int D>
using Factorization2 = int[D];

using Factorization = Factorization2<DIM>;
using MemoFactorization = int[MEMO_DIM];
using MemoEntry = Factorization2<MEMO_DIM>[MEMO_cardinality];

// Worker state structure
struct FactorizationCandidateState {
  int element = 0;
  Factorization lastCandidate = {};
  int dim = 0;
  bool wasValid = false;
  Factorization bound = {};
  Factorization generators = {};
  bool endOfStream = true;
  int bufferIndex = 0; // the last used index in this worker's buffer
};

// Memory allocation structure
struct HostAndDeviceAllocations {
  FactorizationCandidateState *d_states = NULL;
  FactorizationCandidateState* h_states;// = new FactorizationCandidateState[TOTAL_THREADS](); // []() zero-initializes
  MemoEntry* h_memo;// = new MemoEntry[MEMO_maxElement](); // zero initialize
  MemoEntry* d_memo = NULL;
  Factorization* d_outputs = NULL;
  Factorization *d_buffer = NULL;
  Factorization* h_buffer;// = new Factorization[bufferNumElements](); // []() zero-initializes
  int* d_notFinishedCounter;
  int* d_orderDelta;
  int* d_fullCount;
};

// RAII wrapper for allocations
struct ManagedAllocations {
  HostAndDeviceAllocations a;
  ManagedAllocations(int bufferNumElements, int bufferSize, int outputsSize, int memoSize) {
    a = allocateEverything(bufferNumElements, bufferSize,  outputsSize,  memoSize);
  }
  ~ManagedAllocations() {
    printf("Freeing allocations");
    freeEverything(a);
  }
};

#endif // TYPES_H