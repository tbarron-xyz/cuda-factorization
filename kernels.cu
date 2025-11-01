#include "types.h"
#include <cuda_runtime.h>

__device__ void nextCandidateLexicographic_inPlace(
  const int element, const int dim,
  Factorization *lastCandidate, const Factorization *generators,
  bool* wasValid, const Factorization* bound, bool* endOfStream,
   int* orderDelta, // for modulo optimization
   const int memoDim, const MemoEntry* d_memo, Factorization* output, int actualTopOfMemo) {
    if (*endOfStream) { return; } // we weren't invited to this party
    const Factorization& b = *bound;
    Factorization& a = *lastCandidate;//alias
    const Factorization& g = *generators;

    int i = -1;
    for (int j = 0; j < dim - 1; j++) { 
      if (a[j] > 0) i = j; 
    } //setting i to rightmost nonzero index, excluding the final
    if (i == -1) {
      // last candidate was the lexigocraphically final candidate; we are done
      // printfacln(a, "Final candidate");
      *wasValid = false;
      *endOfStream = true;
      return;
    }


    // TODO: Having this step enabled may be preventing the running of cases with MEMO_DIM = DIM - 1
    if (useModuloOptimization && dim > 3 && *wasValid && i == dim - 2) {
      //"slopes" optimization
      if (a[i] <= *orderDelta) {a[i] = 0;}
      else {
        a[i] -= *orderDelta;
      }
    } else { //normal decrement
      (a[i])--;
    }
    a[dim-1] = 0; // may have been nonzero from previous candidate but is now zeroed

    // memo step
    int phiResult = d_phi(lastCandidate, generators);
    if (i == dim - 1 - memoDim && element - phiResult < actualTopOfMemo) {
      // copy memo contents to buffer, then prepend lastCandidate to each
      const MemoEntry& memoEntry = d_memo[element - phiResult];
      int lastValidIndex = -1;
      for (int memoIndex = 0; memoIndex < MEMO_cardinality; memoIndex++) {
        const MemoFactorization& memoFac = memoEntry[memoIndex];
        Factorization& outputFac = output[memoIndex];
        if (d_isAllZeroes(memoFac, MEMO_DIM) && !(memoIndex == 0 && (element - phiResult == 0))) {
          break;
        } else {
          lastValidIndex = memoIndex;
          d_copy2(memoFac, &(outputFac[dim - memoDim]), memoDim);
          d_copy2(a, outputFac, dim - memoDim);
          if (d_phi(&outputFac, generators) != element) {    //sanity check.
            printf("INVALID: =%i\n", d_phi(&outputFac, generators));
            d_printfac(a); d_printfac(memoFac, memoDim);
          }
        } 
      }
      // copy last returned fac to prev candidate
      *wasValid = false; // ignore lastCandidate if outputs were used
    } else {
      int p = element - phiResult;   // the deficiency
      int m = (p + g[i+1] - 1) / (g[i+1]); // division sans remainder
      int result = m * g[i+1] + phiResult;
      (a[i+1]) = m;
      if (result != element) {
        *wasValid = false;
      } else {
        *wasValid = true;
      }
    }

    if (lexicogLeq(&a, bound)) {
      // printf("LT: "); printfac(a); printfacln((*bound), "is less than ");
      // *wasValid = false;
      // if (!facIsEqual(&a, bound)) {
      //   printfac(a); printfacln(b, "LT");
      //   printf("LT BOUND;");
      // }
      *endOfStream = true;
    }
    // if (*wasValid && !*endOfStream && d_phi(lastCandidate, generators) != element) {
    //   // printf("INVALID");
    //   printfacln(a, "Invalid return");
    // }
}

__global__ void nextCandidateLexicographic_inPlace_grid(FactorizationCandidateState* d_states, int* d_orderDelta, const MemoEntry* d_memo, Factorization* d_outputs, int actualTopOfMemo) {
  int element = (*d_states).element;
  int dim = (*d_states).dim;
  int gridIndex = blockIdx.x * blockDim.x + threadIdx.x;
  FactorizationCandidateState& state = d_states[gridIndex];
  nextCandidateLexicographic_inPlace(element, dim,
    &state.lastCandidate, &state.generators,
    &state.wasValid, &state.bound, &state.endOfStream, d_orderDelta, MEMO_DIM, d_memo, &(d_outputs[gridIndex * MEMO_cardinality]), actualTopOfMemo);
}

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

__global__ void allThreadsAreFinished_grid(FactorizationCandidateState* d_states, int* d_notFinishedCounter) {
  int gridIndex = blockIdx.x * blockDim.x + threadIdx.x;
  FactorizationCandidateState& state = d_states[gridIndex];
  if (!state.endOfStream) {
    atomicAdd(d_notFinishedCounter, 1); } else { 
  }
}