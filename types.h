#ifndef TYPES_H
#define TYPES_H

#include <cstddef>
#include <cstdio>

// Constants
#define ELEMENT 1500
#define DIM 8
#define MEMO_cardinality 2048
#define MEMO_maxElement 5000
#define MEMO_DIM 4
#define BLOCKS 136
#define THREADS_PER_BLOCK 8
#define TOTAL_THREADS (BLOCKS * THREADS_PER_BLOCK)
#define GridIndex() (blockIdx.x * blockDim.x + threadIdx.x)

typedef int Factorization[DIM];
typedef int MemoFactorization[MEMO_DIM];
typedef MemoFactorization MemoEntry[MEMO_cardinality];

struct FactorizationCandidateState {
  int element;
  Factorization lastCandidate;
  int dim;
  bool wasValid;
  Factorization bound;
  Factorization generators;
  bool endOfStream;
  int bufferIndex;
};

struct HostAndDeviceAllocations {
  FactorizationCandidateState *d_states;
  FactorizationCandidateState* h_states;
  MemoEntry* h_memo;
  MemoEntry* d_memo;
  Factorization* d_outputs;
  Factorization *d_buffer;
  Factorization* h_buffer;
  int* d_notFinishedCounter;
  int* d_orderDelta;
  int* d_fullCount;
};

#endif // TYPES_H