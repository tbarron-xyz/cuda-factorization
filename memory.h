#ifndef MEMORY_H
#define MEMORY_H

#include "config.h"
#include "types.h"
#include "cuda_utils.h"
#include <cstdio>

#ifdef __CUDACC__

// Memory allocation function
HostAndDeviceAllocations allocateEverything(int bufferNumElements, int bufferSize, int outputsSize, int memoSize) {
  HostAndDeviceAllocations a;
  cudaError_t err = cudaSuccess;
  size_t size = THREADS_PER_BLOCK * BLOCKS * sizeof(FactorizationCandidateState);

  // cudamalloc d_states
  FactorizationCandidateState *&d_states = a.d_states;
  err = cudaMalloc((void **)&d_states, size);
  if (err != cudaSuccess) {
    vfprintf(stderr, "Failed to allocate device states (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  cudaMemset(d_states, 0, size);
  // vfprintf(stdout, "Number of total states: %i\n", totalThreads);
  vfprintf(stdout, "Size of one state: %i\n", sizeof(FactorizationCandidateState));
  vfprintf(stdout, "Allocated %i on the device for states\n", size);
  // #pragma unroll //todo make this a kernel?

  // malloc h_states
  a.h_states = new FactorizationCandidateState[TOTAL_THREADS](); // []() zero-initializes

  // cudamalloc for d_buffer
  Factorization *&d_buffer = a.d_buffer;

  err = cudaMalloc((void **)&d_buffer, bufferSize);
  if (err != cudaSuccess) {
    vfprintf(stderr, "Failed to allocate buffer (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  vfprintf(stdout, "Allocated %d on the device for buffer\n", bufferSize);
  zeroDeviceBuffer(d_buffer, bufferSize);

  //malloc h_buffer
  a.h_buffer = new Factorization[bufferNumElements](); // []() zero-initializes
  vfprintf(stdout, "Allocated host buffer %d * %d\n", bufferNumElements, sizeof(Factorization));


  // cudamalloc d_outputs
  Factorization*& d_outputs = a.d_outputs;
  err = cudaMalloc((void **)&d_outputs, outputsSize);
  if (err != cudaSuccess) {
    vfprintf(stderr, "Failed to allocate d_outputs (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  vfprintf(stdout, "Allocated %d on the device for d_outputs\n", outputsSize);
  zeroDeviceBuffer(d_outputs, outputsSize);

  //cudamalloc d_notfinishedcounter
  int*& d_notFinishedCounter = a.d_notFinishedCounter;
  err = cudaMalloc((void**)&d_notFinishedCounter, sizeof(int));
  if (err != cudaSuccess) {
    vfprintf(stderr, "Failed to malloc notfinishedcounter \n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  int*& d_orderDelta = a.d_orderDelta;
  err = cudaMalloc((void**)&d_orderDelta, sizeof(int));
  if (err != cudaSuccess) {
    vfprintf(stderr, "Failed to malloc orderdelta \n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // malloc and cudamalloc for memo
  a.h_memo = new MemoEntry[MEMO_maxElement](); // zero initialize
  MemoEntry* &d_memo = a.d_memo;
  err = cudaMalloc((void **)&d_memo, memoSize);
  if (err != cudaSuccess) {
    vfprintf(stderr, "Failed to allocate memo (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  cudaMemset(d_memo, 0, memoSize, "d_memo");
  vfprintf(stdout, "Allocated %d on the device for memo\n", memoSize);

  int*& d_fullCount = a.d_fullCount;
  err = cudaMalloc((void**)&d_fullCount, sizeof(int));
  if (err != cudaSuccess) {
    vfprintf(stderr, "Failed to cudamalloc d_fullcount\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  // printf("d_fullcount initial value: %d", d_fullCount);

  err = cudaMemset(d_fullCount, 0, sizeof(int), "d_fullCount");
  return a;
}

// Memory deallocation function
void freeEverything(HostAndDeviceAllocations& a) {
  cudaError_t err = cudaFree(a.d_fullCount, "d_fullCount");
  err = cudaFree(a.d_memo, "d_memo");
  err = cudaFree(a.d_outputs, "d_outputs");
  err = cudaFree(a.d_buffer, "d_buffer");
  err = cudaFree(a.d_notFinishedCounter, "d_notFinishedCounter");
  err = cudaFree(a.d_orderDelta, "d_orderDelta");
  delete(a.h_buffer);
  delete(a.h_memo);
  delete(a.h_states);
  

  FactorizationCandidateState *d_states = NULL;
  FactorizationCandidateState* h_states;// = new FactorizationCandidateState[TOTAL_THREADS](); // []() zero-initializes
  MemoEntry* h_memo = new MemoEntry[MEMO_maxElement](); // zero initialize
  MemoEntry* d_memo = NULL;
  Factorization* d_outputs = NULL;
  Factorization *d_buffer = NULL;
  Factorization* h_buffer;// = new Factorization[bufferNumElements](); // []() zero-initializes
  int* d_notFinishedCounter;
  int* d_orderDelta;
}

#endif // __CUDACC__

#endif // MEMORY_H