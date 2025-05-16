#include <stdio.h>
#include <ctime>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
// #include <cuda/std/chrono>
#include <chrono>
#include <string>
#include <exception>
#include <iostream>
#include <fstream>
#include <format>
#include <filesystem>

#include <helper_cuda.h>

// To build, place this file in the NVIDIA "cuda-samples" repository and run "nvcc .\factorize.cu -o factorize.exe -allow-unsupported-compiler -I Common\"

// constexpr int ELEMENT = 
#ifndef ELEMENT
#define ELEMENT 1500
// 120;
// 500;//-43*3;
// 1000;
// 1500 // good
// 2000;
// 2500;
// 3000;
// 4000;
// 5000;
// 7000;
// 9000;
// 10000;
// 12000;
// 13000;
// 17000;
// 20000;
// 23000;
// 27000;
// 45000;
// 70000;
// 150000;
// 225000;
// 300000;
// 500000;
#endif

// constexpr int DIM = 

#ifndef DIM
#define DIM \
8
// 3;
// 4;
// 5;
// 6;
// 7;
// 9;
#endif

// using Factorization = int[DIM];

template <int D>
using Factorization2 = int[D];

using Factorization = Factorization2<DIM>;


// good to make this larger than 1x or 2x of MEMO_cardinality, so that it can hold several kernels worth of outputs //5000
// make this too large and it make become slower even without running out of memory
constexpr size_t bufferPerThread = 
// 5000;
40000;  

// It is runtime-advantageous to make this as small as possible for your use case.
#ifndef MEMO_cardinality
#define MEMO_cardinality 2048
// 4096;
// 2048;
// 1024;
#endif

#ifndef MEMO_maxElement
#define MEMO_maxElement 5000
// ELEMENT;
// 5000;
#endif

#ifndef MEMO_DIM
#define MEMO_DIM 4
// 6;
// 5;
// 4;
// 3;
// 2;
#endif


using MemoFactorization = int[MEMO_DIM];

using MemoEntry = Factorization2<MEMO_DIM>[MEMO_cardinality];
// using Memo = MemoEntry[MEMO_maxElement];

constexpr bool useModuloOptimization = 
// false;
true;


constexpr int kernelsBetweenFreshBounds = 
// 4;
// 8;
// 16;
// 32;
// 64;
// 128;
// 256;
// 512;
1024; // good
// 2048;

constexpr int BLOCKS = 
// 1;
// 2;
// 8;
// 16;
// 32;
// 68;
68 * 2;//good
//68*
// 2;// good
// 4;
// 8;
// 100;
constexpr int THREADS_PER_BLOCK = 
// 1;
// 2;
// 4;
8; // good
// 16;
// 32;
// 128;
constexpr int TOTAL_THREADS = BLOCKS * THREADS_PER_BLOCK;
#define GridIndex() blockIdx.x * blockDim.x + threadIdx.x

// End constants



int orderOfAinB(int a, int b) {
  int mod = a % b;
  int result = 0;
  for (int i = 1; i <= b; i++) {
    result += mod;
    if (result % b == 0) {
      return i;
    }
  }
}
bool isAllZeroes(const Factorization* fP, int dim = DIM) {
  const Factorization& f = *fP;
  for (int i = 0; i < dim; i++) {
    if (f[i] != 0) { return false; }
  }
  return true;
}
bool isAllZeroes2(const int* fP, int dim = DIM) {
  const int* f = fP;
  for (int i = 0; i < dim; i++) {
    if (f[i] != 0) { return false; }
  }
  return true;
  // return f[0] == 0 && f[1] == 0 && f[2] == 0 && f[3] == 0;
}
__device__ bool d_isAllZeroes(const int* fP, int dim = MEMO_DIM) {
  const int* f = fP;
  for (int i = 0; i < dim; i++) {
    if (f[i] != 0) { return false; }
  }
  return true;
  // return f[0] =
}
__host__ __device__ bool isZeroesOnLeft(const int* f, int i) {
  for (int j = 0; j < i; j++) {
    if (j < i) {
      if (f[j] != 0) {
        return false;
      }
    }
  }
  return true;
}


#define vfprintf fprintf  // macro for optionally disabling all printing
#define vprintf printf
void printfac(const int* f, std::string prefix, std::string postfix = "") {
  vprintf(prefix.c_str());
  vprintf(": ("); 
  for (int i = 0; i < DIM; i++) {
  printf("%d, ", f[i]);
  }
  vprintf("\b\b)");
  vprintf(postfix.c_str());
}
void printfacln(const int* f, std::string prefix) { printfac(f, prefix, "\n"); }

__device__ void d_printfac(const int* f, int dim = DIM) {
  for (int i = 0; i < dim; i++) {
    printf("%d, ", f[i]);
  }
}


void copy(const Factorization* source, Factorization* target, int dim) { // todo should  just use memcpy/cudamemcpy directly
  for (int j = 0; j < dim; j++) {
    (*target)[j] = (*source)[j];
  }
}
__host__ __device__ void copy2(int* source, int* target, int dim) {
  for (int j = 0; j < dim; j++) {
    target[j] = source[j];
  }
}
//unused.
__host__ __device__ bool facIsEqual(const Factorization* a, const Factorization* b) {
  const Factorization& A = *a; const Factorization& B = *b;
  for (int i = 0; i < DIM; i++) {
    if (A[i] != B[i]) { return false;}
  }
  return true;
}





__device__ void d_copy(const Factorization* d_source, Factorization* d_target, int dim) { // todo this should just use memcpy/cudamemcpy directly
  #pragma unroll
  for (int j = 0; j < dim; j++) {
    (*d_target)[j] = (*d_source)[j];
  }
}

__device__ void d_copy2(const int* d_source, int* d_target, int dim) {
  #pragma unroll
  for (int j = 0; j < dim; j++) {
    d_target[j] = d_source[j];
  }
}




__host__ __device__ bool lexicogLeq(const Factorization* a, const Factorization* b) { 
  const Factorization& A = *a; const Factorization& B = *b;
  // for loop based comparison
  for (int i = 0; i < DIM; i++) {
    if (A[i] < B[i]) { return true; }
    else if (A[i] > B[i]) { return false; }
    else { // equality is the only case remaining
      if (i == DIM - 1) {
        return true;
      } // else, continue
    }
  }

  // Now this should be unreachable
  printf("Lexicog comparison fail?");
  return true;
}

__device__ int d_phi(const Factorization* factorization, const Factorization* generators) {
  int sum = 0;
  // #pragma unroll // can't unroll addition without atomicAdd.
  for (int i = 0; i < DIM; i++) {
    sum += (*factorization)[i] * (*generators)[i];
  }
  return sum;
}

int phi(const Factorization* factorization, const Factorization* generators) {
  int sum = 0;
  for (int i = 0; i < DIM; i++) {
    sum += (*factorization)[i] * (*generators)[i];
  }
  return sum;
}

__host__ __device__ void incrementIndex(int* f, int index) {
  f[index] ++;
}

// extra argument triggers checking errors
cudaError_t cudaMemcpy( void* dst, const void* src, size_t count, cudaMemcpyKind kind, std::string name) {
  cudaError_t err = cudaMemcpy(dst, src, count, kind);
  if (err != cudaSuccess) {
    vfprintf(stderr, "Failed to memcpy %s (error code %s)!\n",
            name.c_str(),
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  return err;
}

// End what should be true invariants







int dynamicFactorizations_inner(int element, const Factorization2<MEMO_DIM> memoGenerators, Factorization2<MEMO_DIM>* output, MemoEntry* memo) {
  if (element == 0) {
    return 1; // output already zero
  } else {
    // dependencies, incremented; check for exceeding cardinality, if so call it quits TODO
    int counter = 0;
    for (int i = 0; i < MEMO_DIM; i++) {
      if (counter >= MEMO_cardinality) {
        return -1;
      }
      int generator = memoGenerators[i];
      if (element < generator) {
        continue; // factorization set is empty
      } else if (element == generator) {
        if (counter >= MEMO_cardinality) {
          return -1;
        }
        MemoFactorization& outputFac = output[counter];
        incrementIndex(outputFac, i);//done
        counter++;
      } else {
        MemoEntry& prevEntry = memo[element - generator];
        // copy preventry to output
        // increment output
        // increase counter
        for (int j = 0; j < MEMO_cardinality; j++) {
          if (counter >= MEMO_cardinality) {
            return -1;
          }
          MemoFactorization& prevFac = prevEntry[j];
          MemoFactorization& outputFac = output[counter];
          if (isAllZeroes2(prevFac, MEMO_DIM)) {
            break;
          } else if (isZeroesOnLeft(prevFac, i)) {
            copy2(prevFac, outputFac, MEMO_DIM);
            incrementIndex(outputFac, i);
            counter++;
          }
        }
      }
    }
        // printf("Element %i had %i facs\n", element, counter);
    return counter;
  }
}

int populateDynamicFactorizations_sequential(MemoEntry* memo, const Factorization2<MEMO_DIM> memoGenerators
  //dim, max element, cardinality in globals
) {
  int lastSuccessfulElement = 0;
  for (int element = 0; element < MEMO_maxElement; element++) {
    int result = dynamicFactorizations_inner(element, memoGenerators, memo[element], memo);
    if (result == -1) {
      break;
    }
    lastSuccessfulElement = element;
    // printf("Z(%d, (%d, %d, %d)) had %d results\n", element, memoGenerators[0], memoGenerators[1], memoGenerators[2], result);
    // for (int i = 0; !isAllZeroes2(memo[element][i], MEMO_DIM) && i < MEMO_cardinality; i++) {
    //   printfacln(memo[element][i], "a");
    // }
  }
  printf("Dynamic factorizations (CPU) up to %d stored for Z(-, (", lastSuccessfulElement);
  for (int i = 0; i < MEMO_DIM; i++) {
    printf("%d, ", memoGenerators[i]);
  }
  printf("\b\b))\n");
  return lastSuccessfulElement;
}


__global__ void copyAndIncrementFactorizations_grid(int generatorIndex, Factorization2<MEMO_DIM>* source, Factorization2<MEMO_DIM>* target, int cardinality) {
  int gridIndex = GridIndex();
  if (gridIndex >= cardinality) { return; }
  Factorization2<MEMO_DIM>& sourceFac = source[gridIndex];
  Factorization2<MEMO_DIM>& targetFac = target[gridIndex];
  copy2(sourceFac, targetFac, MEMO_DIM);
  incrementIndex(targetFac, generatorIndex);
}




int dynamicFactorizations_factorizationwiseParallel_inner(int element, const Factorization2<MEMO_DIM> memoGenerators, Factorization2<MEMO_DIM>* d_output, MemoEntry* d_memo, MemoFactorization* cardinalities) {
  if (element == 0) {
    return 1; // output already zero
  } else {
    // dependencies, incremented; check for exceeding cardinality, if so call it quits TODO
    int counter = 0;
    for (int i = 0; i < MEMO_DIM; i++) {
      int cardinalityThisIndex = 0;
      if (counter >= MEMO_cardinality) {
        return -1;
      }
      int generator = memoGenerators[i];

      if (element < generator) {
        continue; // factorization set is empty
      } else if (element == generator) {

        // single element: does not require a kernel launch, we can do it ourselves
        if (counter >= MEMO_cardinality) {
          return -1;
        }
        MemoFactorization* outputFac = &(d_output[counter]);

        copyAndIncrementFactorizations_grid<<<1,1>>>(i, &(outputFac[0]), &(outputFac[0]), 1);

        cardinalityThisIndex = 1;
      } else { // element > generator
        int startIndex = 0; // number of factorizations preceding the first we want
        for (int j = 0; j < MEMO_DIM; j++) {
          if (j < i) startIndex += cardinalities[element-generator][j];
          else cardinalityThisIndex += cardinalities[element-generator][j];
        }
        if (counter + cardinalityThisIndex >= MEMO_cardinality) { 
          return -1;
        }
        // experiment with blocks and threads here. <<<BLOCKS, THREADS_PER_BLOCK>>>
        int split = 16;
        copyAndIncrementFactorizations_grid<<<(cardinalityThisIndex + split - 1) / split, split>>>(i, d_memo[element-generator], &(d_output[counter]), cardinalityThisIndex);
      }
      cardinalities[element][i] = cardinalityThisIndex;
      counter += cardinalityThisIndex;
    }
    // printf("Element %i had %i facs\n", element, counter);
    return counter;
  }
}


int populateDynamicFactorizations_factorizationwiseParallel(MemoEntry* d_memo, const Factorization2<MEMO_DIM> memoGenerators) {
  MemoFactorization* cardinalities = new MemoFactorization[MEMO_maxElement]();  // always remember to zero your initialization []()
  int lastSuccessfulElement = 0;
  for (int element = 0; element < MEMO_maxElement; element++) {
    int result = dynamicFactorizations_factorizationwiseParallel_inner(element, memoGenerators, d_memo[element], d_memo, cardinalities);
    if (result == -1) {
      break;
    }
    lastSuccessfulElement = element;
    // printf("Z(%d, (%d, %d, %d)) had %d results\n", element, memoGenerators[0], memoGenerators[1], memoGenerators[2], result);
    // for (int i = 0; !isAllZeroes2(memo[element][i], MEMO_DIM) && i < MEMO_cardinality; i++) {
    //   printfacln(memo[element][i], "a");
    // }
  }
  printf("Dynamic factorizations (GPU) up to %d stored for Z(-, (", lastSuccessfulElement);
  for (int i = 0; i < MEMO_DIM; i++) {
    printf("%d, ", memoGenerators[i]);
  }
  printf("\b\b))\n");
  delete cardinalities;
  return lastSuccessfulElement;
}













int savedFactorizationCounter = 0;
Factorization* savedFacs = new Factorization[1000](); // []() zero-initializes
void saveSingleFactorization(const Factorization& f) {  // circular in-memory save buffer; this could save to a file
  // printfacln(f, "Saving single");
  copy(&f, &(savedFacs[savedFactorizationCounter % 1000]), DIM);
  savedFactorizationCounter = (savedFactorizationCounter + 1);// % 1000;
}



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


void saveHostBufferToFakeDisk(const Factorization* h_buffer, int bufferNumElements) {
  for (int i = 0; i < bufferNumElements; i++) {
    const Factorization& bufferElement = h_buffer[i];
    if (!isAllZeroes(&bufferElement)) {
      saveSingleFactorization(bufferElement);
    }
  }
}








/*
Worker State Management
*/



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




bool newBoundFromExistingState_inPlace(int element, const Factorization* generators, const Factorization* bound, const Factorization* lastCandidate, //FactorizationCandidateState* state,
   Factorization* newBound, int dim, const int memoDim) {
  // set j leftmost nonzero such that a_i neq b_i
  int j = -1;
  const Factorization& g = *generators;
  const Factorization& l = *lastCandidate;
  const Factorization& b = *bound;
  Factorization& nB = *newBound;

  try {
    //leftmost nonzero index not matching the bound, excluding memodim at the end
    for (int tempJ = dim-1 - memoDim; tempJ >= 0; tempJ--) {
      if (l[tempJ] > 0 && l[tempJ] != (*bound)[tempJ]) { j = tempJ; }
    }
  } catch (std::exception& e) {
    vfprintf(stderr, "Caught something: %s", e.what());
  }
  if (j == dim - 1 || j == -1) { return false; } // The last candidate is equal to the bound (excepting possibly the final); this case means the work cannot be split
  // new bound is subtracting one from a_j and solving for a_{j+1}, zeroes afterward
  copy(lastCandidate, newBound, dim);
  if (nB[j] == 0) { printf("nbj == 0; this shouldnt happen"); }
  nB[j] --;
  // zero everything to the right of the next one
  for (int k = j + 1; k < dim; k++) {
    if (k < dim) { nB[k] = 0; }
  }

  const Factorization& a =*lastCandidate;
  int dividend = element - phi(&nB, generators);   // the deficiency
  int nextGenerator = g[j+1];

  int quotient = dividend / nextGenerator; // division sans remainder
  int r = dividend - quotient * (g[j+1]); // remainder
  bool wasValid = true; //, temporarily
  if (r != 0) {
    quotient++;
    wasValid = false;
  }
  nB[j+1] = quotient;

  //done setting new bound values
  if (lexicogLeq(newBound, bound)) { return false; }  // the supposed new bound is worse than the existing bound
  if (wasValid) {
    // printf("Saving in newBound"); printfacln((*newBound));
    saveSingleFactorization(*newBound); 
  } // both workers are left and right exclusive so this needs to be saved now.

  return true;
}

void giveFreshBounds(FactorizationCandidateState* h_states, int dim) {
  // split work from the last one (which has bound zero) to the second to last, and so forth
  for (int i = 0; i < TOTAL_THREADS/* 8 */; i++) {
    int newStateIndex = TOTAL_THREADS - 1 - i;
    FactorizationCandidateState& newState = h_states[newStateIndex];
    if (!newState.endOfStream) { 
      continue;
     }// doesnt need a new bound
    
    bool success = false;
    FactorizationCandidateState* oldStatePointer = NULL;//temporarily
    
    int offset = 1;
    int oldStateIndex = -1;
    for (; !success && offset <= i; offset++) {  // try to split work starting with the previous worker; if that fails, working backwards
      oldStateIndex = TOTAL_THREADS - 1 - i + offset;
      oldStatePointer = &(h_states[oldStateIndex]);
      FactorizationCandidateState& oldState = *oldStatePointer;
      if (oldState.endOfStream) continue;// we can't split an inactive stream
      success = newBoundFromExistingState_inPlace(oldState.element, &oldState.generators, &oldState.bound, &oldState.lastCandidate, &newState.bound, dim, MEMO_DIM);
    }
    if (!success) { 
      // couldn't set a bound for this guy, boo hoo;
      continue;
    }
    FactorizationCandidateState& oldState = *oldStatePointer;
    

    
    // old version: bound from first to second; new bound to last candidate of second; copy new bound to bound of first
    // current: old keeps its bound, receives new last candidate; new gets new bound, takes old last candidate
    copy(&oldState.lastCandidate, &newState.lastCandidate, DIM);
    copy(&newState.bound, &oldState.lastCandidate, DIM);
    

    newState.endOfStream = false;
  };
}

cudaError_t cudaMemset ( void* devPtr, int  value, size_t count , std::string name) {//extra arg adds error checking
  cudaError_t err = cudaMemset(devPtr, value, count);
  if (err != cudaSuccess) {
    vfprintf(stderr, "Failed to zero device buffer: %s\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  return err;
}

void zeroDeviceBuffer(Factorization* d_buffer, size_t bufferSize) {
  // printf("Zeroing device buffer");
  cudaError_t err = cudaSuccess;
  err = cudaMemset(d_buffer, 0, bufferSize);
  if (err != cudaSuccess) {
    vfprintf(stderr, "Failed to zero device buffer: %s\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
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



__global__ void d_anyBufferIsFull(FactorizationCandidateState* d_states, int* d_fullCount, size_t bufferPerThread) { // this could use atomicOr instead of atomicAdd
  int gridIndex = blockIdx.x * blockDim.x + threadIdx.x;
  FactorizationCandidateState& state = d_states[gridIndex];
  if (state.bufferIndex + MEMO_cardinality >= bufferPerThread) { atomicAdd(d_fullCount, 1); }
}

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


// extra arg adds error checking
cudaError_t 	cudaFree ( void* devPtr, std::string name ) {
  cudaError_t err = cudaFree(devPtr);
  if (err != cudaSuccess) {
    vfprintf(stderr, "Failed to cudafree %s: %s\n",
      name.c_str(),
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  return err;
}

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

struct ManagedAllocations {
  HostAndDeviceAllocations a;
  ManagedAllocations(int bufferNumElements, int bufferSize, int outputsSize, int memoSize) {
    a = allocateEverything(bufferNumElements, bufferSize,  outputsSize,  memoSize);
  }
  ~ManagedAllocations() {
    vfprintf(stdout, "Freeing allocations");
    freeEverything(a);
  }
};




bool fileExists(const char *fileName)
{
    std::ifstream infile(fileName);
    return infile.good();
}

int runParallelDynamicLexicographic(int element, const Factorization2<DIM> generators, bool hostMemoPopulation) {
  cudaError_t err = cudaSuccess;
  size_t bufferNumElements = TOTAL_THREADS * bufferPerThread;
  size_t bufferSize = bufferNumElements * sizeof(Factorization);
  size_t outputsSize = sizeof(Factorization) * MEMO_cardinality * TOTAL_THREADS;
  size_t memoSize = sizeof(MemoEntry) * MEMO_maxElement;
  //todo specialize variable args into constexpr
  ManagedAllocations ma(bufferNumElements,  bufferSize,  outputsSize,  memoSize);
  auto& allocations = ma.a;
  FactorizationCandidateState*& d_states = allocations.d_states;
  FactorizationCandidateState*& h_states = allocations.h_states;
  Factorization *&d_buffer = allocations.d_buffer;
  Factorization*& h_buffer = allocations.h_buffer;
  MemoEntry*& h_memo = allocations.h_memo;
  MemoEntry*& d_memo = allocations.d_memo;
  Factorization*& d_outputs = allocations.d_outputs;

  // set generators, element, and dim on all h_states
  // todo: this is unnecessary to store separately for each state, they can all reference one copy in device memory
  for (int i = 0; i < TOTAL_THREADS; i++) {
    h_states[i].element = element;
    h_states[i].dim = DIM;
    for (int j = 0; j < DIM; j++) { h_states[i].generators[j] = generators[j]; }
  }

  // set the first factorization
  FactorizationCandidateState& firstState = h_states[TOTAL_THREADS - 1];

  firstState.lastCandidate[0] = (element + generators[0] - 1) / generators[0];
  firstState.endOfStream = false;
  printf("Element: %i\n", element);
  printfacln(generators, "Generators");
  printfacln(firstState.lastCandidate, "First state lastCandidate");
  if (firstState.lastCandidate[0] * generators[0] == element) { saveSingleFactorization(firstState.lastCandidate); }
  // the rest of the coordinates are already zero from the initializer.
  // first worker's bound is all zeroes, which is set by the initializer.


  // copy h_states to d_states directly after setting instead of multiple calls
  err = cudaMemcpy(d_states, h_states, sizeof(FactorizationCandidateState) * TOTAL_THREADS, cudaMemcpyHostToDevice, "h_states");


  // Start the timer before dynamic factorizations
  auto steadyClock = std::chrono::steady_clock();
  auto steadyStartTime = steadyClock.now();

// THIS IS RUNNING BOTH CPU AND GPU MEMO POPULATION FOR COMPARISON PURPOSES.

  // populate host memo and copy host to device
  // bool hostMemoPopulation = true;
  int actualTopOfMemo;
  // if (hostMemoPopulation) {
    actualTopOfMemo = populateDynamicFactorizations_sequential(h_memo, &(generators[DIM - MEMO_DIM]));
    auto memoPopulationTime_cpu = steadyClock.now();
    // err = cudaMemcpy(d_memo, h_memo, memoSize, cudaMemcpyHostToDevice, "h_memo");
  // } else {
    actualTopOfMemo = populateDynamicFactorizations_factorizationwiseParallel(d_memo, &(generators[DIM - MEMO_DIM]));
  // }
  auto memoPopulationTime_gpu = steadyClock.now();


  // exit(0);

  // calculate next candidate in a loop on grid
  bool hasCompleted = false;
  int*& d_notFinishedCounter = allocations.d_notFinishedCounter;

  int orderD = orderOfAinB(generators[DIM-2],generators[DIM-1]);
  printf("orderD %i\n", orderD);
  int*& d_orderDelta = allocations.d_orderDelta;
  err = cudaMemcpy(d_orderDelta, &orderD, sizeof(int), cudaMemcpyHostToDevice, "orderD");
  int counter = 0;
  time_t startTime = time(NULL);

  int h_notFinishedCounter;
    // for (int iterations = 0; iterations < 50 && !hasCompleted; iterations++) {
  while (!hasCompleted) {
    if (counter % kernelsBetweenFreshBounds == 0) {
      err = cudaMemcpy(h_states,d_states,  sizeof(FactorizationCandidateState) * TOTAL_THREADS, cudaMemcpyDeviceToHost, "d_states (fresh bounds)");
      giveFreshBounds(h_states, DIM);
      err = cudaMemcpy(d_states, h_states, sizeof(FactorizationCandidateState) * TOTAL_THREADS, cudaMemcpyHostToDevice, "h_states (fresh bounds)");
    }

    
    if (counter % 1000 == 0) { vfprintf(stdout, "Launching kernel %d\n", counter); }
    // zero outputs first
    
    err = cudaMemset(d_outputs, 0, outputsSize);
    if (err != cudaSuccess) {vfprintf(stderr, "Failed to zero outputs\n",cudaGetErrorString(err));exit(EXIT_FAILURE);}
    nextCandidateLexicographic_inPlace_grid<<<BLOCKS, THREADS_PER_BLOCK>>>(d_states, d_orderDelta, d_memo, d_outputs, actualTopOfMemo);
    copyLastCandidateToBufferIfValid_grid<<<BLOCKS, THREADS_PER_BLOCK>>>(
      d_buffer, d_states, bufferPerThread
    );
    copyOutputsToBufferIfValid_grid<<<BLOCKS, THREADS_PER_BLOCK>>>(
      d_buffer, d_states, d_outputs, bufferPerThread
    );

    int zero = 0;
    err = cudaMemcpy(d_notFinishedCounter, &zero, sizeof(int), cudaMemcpyHostToDevice, "zero");
    allThreadsAreFinished_grid<<<BLOCKS,THREADS_PER_BLOCK>>>(d_states, d_notFinishedCounter);
    err = cudaMemcpy(&h_notFinishedCounter, d_notFinishedCounter, sizeof(int), cudaMemcpyDeviceToHost, "d_notFinishedCounter");
    hasCompleted = h_notFinishedCounter == 0;
    if (anyBufferIsFull(d_states, bufferPerThread, allocations.d_fullCount)) {
      printf("Buffer full, copying\n");
      copyBufferDeviceToHostAndClear(d_buffer, h_buffer, h_states, d_states, bufferSize);
      saveHostBufferToFakeDisk(h_buffer, bufferNumElements);
    }

    // // useful for debugging step: copy d_states to host and print on screen
    // err = cudaMemcpy(h_states, d_states,  sizeof(FactorizationCandidateState) * TOTAL_THREADS, cudaMemcpyDeviceToHost);
    // if (err != cudaSuccess) {
    //   vfprintf(stderr, "Failed to memcpy debug\n",
    //           cudaGetErrorString(err));
    //   exit(EXIT_FAILURE);
    // }
    // for (int i = 0; i < TOTAL_THREADS; i++) {
    // //   printf("Index %i\n", i);
    //   printfacln(h_states[i].lastCandidate, "h_states Last candidate");
    //   printfacln(h_states[i].bound, "h_states bound");

    // //   // printf("Index %i last: \n", i); printfac(h_states[i].lastCandidate);
    //   printf("Was valid: %s \n", h_states[0].wasValid ? "True" : "false");
    //   printf("endOfStream %s \n", h_states[i].endOfStream ? "true" : "false");
    // }
    // printf("\n");

    counter++  ;

  }

  copyBufferDeviceToHostAndClear(d_buffer, h_buffer, h_states, d_states, bufferSize);
  saveHostBufferToFakeDisk(h_buffer, bufferNumElements);

  printf("Has completed.\n");
  

  printf("Kernels launched: %i\n", counter);
  time_t endTime = time(NULL);
  auto steadyEndTime = steadyClock.now();
  printf("Duration (`time`): %i\n", endTime - startTime);
  int millis = std::chrono::duration_cast<std::chrono::milliseconds>(steadyEndTime - steadyStartTime).count();
  printf("Duration (steady_clock) ms: %i\n", millis);
  int cpu_us = std::chrono::duration_cast<std::chrono::microseconds>(memoPopulationTime_cpu - steadyStartTime).count();
  int gpu_us = std::chrono::duration_cast<std::chrono::microseconds>(memoPopulationTime_gpu - memoPopulationTime_cpu).count();
  printf("Memo population (cpu) us: %i\n", cpu_us);
  printf("Memo population (gpu) us: %i\n", gpu_us);

  if (!fileExists("runtimes.csv")) {
    std::ofstream outputFile("runtimes.csv");
    outputFile << "dim,memo_dim,element,num_results,cpu_memo_us,gpu_memo_us,runtime_ms" << std::endl;
    outputFile.close();
  }

  std::ofstream outputFile("runtimes.csv", std::ios_base::app);
  int d = DIM;
  int md = MEMO_DIM;
  char c[50]; sprintf(c, "%i, %i, %i, %i, %i, %i, %i", d, md, element, savedFactorizationCounter, cpu_us, gpu_us, millis);
  outputFile << c << std::endl;
  outputFile.close();

  printf("Number of results: %i\n", savedFactorizationCounter);

  return savedFactorizationCounter;
}


/**
 * Host main routine
 */
int main(int argc, char *argv[]) {

  int element = argc > 1 ? std::stoi(argv[1]) : ELEMENT;
  Factorization generators;
for (int i = 0; i < DIM; i++) {
  if (argc > i + 2) { generators[i] = std::stoi(argv[i+2]); }
  else { generators[i] = i == 0 ? 13 : i > 2 ? 37 + i : 36 + i;  }
}

  int numberOfFactorizations = runParallelDynamicLexicographic(element, generators, argc > DIM + 2);
}
