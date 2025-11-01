#ifndef KERNELS_H
#define KERNELS_H

#include "config.h"
#include "types.h"
#include "utils.h"
#include "cuda_utils.h"
#include "factorization.h"
#include <cstdio>

#ifdef __CUDACC__

// Buffer management kernel
__global__ void d_anyBufferIsFull(FactorizationCandidateState* d_states, int* d_fullCount, size_t bufferPerThread) { // this could use atomicOr instead of atomicAdd
  int gridIndex = blockIdx.x * blockDim.x + threadIdx.x;
  FactorizationCandidateState& state = d_states[gridIndex];
  if (state.bufferIndex + MEMO_cardinality >= bufferPerThread) { atomicAdd(d_fullCount, 1); }
}

// Main computation kernel
__global__ void nextCandidateLexicographic_inPlace_grid(FactorizationCandidateState* d_states, int* d_orderDelta, const MemoEntry* d_memo, Factorization* d_outputs, int actualTopOfMemo) {
  int element = (*d_states).element;
  int dim = (*d_states).dim;
  int gridIndex = blockIdx.x * blockDim.x + threadIdx.x;
  FactorizationCandidateState& state = d_states[gridIndex];
  nextCandidateLexicographic_inPlace(element, dim,
    &state.lastCandidate, &state.generators,
    &state.wasValid, &state.bound, &state.endOfStream, d_orderDelta, MEMO_DIM, d_memo, &(d_outputs[gridIndex * MEMO_cardinality]), actualTopOfMemo);
}

// Buffer operation kernels
__global__ void copyLastCandidateToBufferIfValid_grid(Factorization* d_buffer, FactorizationCandidateState* d_states, size_t bufferPerThread) {
  int gridIndex = blockIdx.x * blockDim.x + threadIdx.x;
  FactorizationCandidateState& state = d_states[gridIndex];
  if (state.wasValid && !state.endOfStream) {//currently does not include end of stream as that was already saved during bound splitting
    d_copy(&state.lastCandidate, &d_buffer[gridIndex * bufferPerThread + state.bufferIndex], DIM);
    state.bufferIndex ++;
  }
}

__global__ void copyOutputsToBufferIfValid_grid(Factorization* d_buffer, FactorizationCandidateState* d_states, Factorization* d_outputs, int bufferPerThread) {
  int gridIndex = blockIdx.x * blockDim.x + threadIdx.x;
  FactorizationCandidateState& state = d_states[gridIndex];
  Factorization* currentOutputs = &(d_outputs[gridIndex * MEMO_cardinality]);
  for (int i = 0; i < MEMO_cardinality; i++) {
    Factorization& currentOutput = currentOutputs[i];
    if (d_isAllZeroes(currentOutput, DIM)) {
      break;
    }
    if (lexicogLeq(&currentOutput, &(state.bound))) {
      continue;
    }
    Factorization* bufferTarget = &(d_buffer[gridIndex * bufferPerThread + state.bufferIndex]);
    d_copy(&currentOutput, bufferTarget, DIM);
    state.bufferIndex ++;
  }
}

// Completion check kernel
__global__ void allThreadsAreFinished_grid(FactorizationCandidateState* d_states, int* d_notFinishedCounter) {
  int gridIndex = blockIdx.x * blockDim.x + threadIdx.x;
  FactorizationCandidateState& state = d_states[gridIndex];
  if (!state.endOfStream) {
    atomicAdd(d_notFinishedCounter, 1); } else { 
  }
}

// Host-side buffer management functions
bool anyBufferIsFull(FactorizationCandidateState* d_states, size_t bufferPerThread, int* d_fullCount) {
  int fullCount = 0;
  cudaError_t err = cudaMemset(d_fullCount, 0, sizeof(int));
  if (err != cudaSuccess) {
    vfprintf(stderr, "Failed to zero d_fullcount in anyBufferIsFull: %s\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  d_anyBufferIsFull<<<BLOCKS, THREADS_PER_BLOCK>>>(d_states, d_fullCount, bufferPerThread);
  err = cudaMemcpy(&fullCount, d_fullCount, sizeof(int), cudaMemcpyDeviceToHost, "d_fullCount");

  return fullCount > 0;
}

void copyBufferDeviceToHostAndClear(Factorization* d_buffer, Factorization* h_buffer, FactorizationCandidateState* h_states, FactorizationCandidateState* d_states, size_t bufferSize) {
  cudaError_t err = cudaSuccess;
  // copy device to host
  err = cudaMemcpy(h_buffer, d_buffer, bufferSize, cudaMemcpyDeviceToHost, "d_buffer");
  // zero device
  zeroDeviceBuffer(d_buffer, bufferSize);

  // set all bufferIndex to zero on d_states
  err = cudaMemcpy(h_states,d_states,  sizeof(FactorizationCandidateState) * TOTAL_THREADS, cudaMemcpyDeviceToHost, "d_states (2)");
  for (int i = 0; i < TOTAL_THREADS; i++) {
    h_states[i].bufferIndex = 0;
  }
  err = cudaMemcpy(d_states, h_states, sizeof(FactorizationCandidateState) * TOTAL_THREADS, cudaMemcpyHostToDevice, "h_states (2)");
}

#endif // __CUDACC__

#endif // KERNELS_H