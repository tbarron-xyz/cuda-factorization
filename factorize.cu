#include "types.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <chrono>
#include <string>
#include <iostream>
#include <fstream>
#include <filesystem>

int savedFactorizationCounter = 0;
Factorization* savedFacs = new Factorization[1000](); // []() zero-initializes
void saveSingleFactorization(const Factorization& f) {  // circular in-memory save buffer; this could save to a file
  // printfacln(f, "Saving single");
  copy(&f, &(savedFacs[savedFactorizationCounter % 1000]), DIM);
  savedFactorizationCounter = (savedFactorizationCounter + 1);// % 1000;
}

void saveHostBufferToFakeDisk(const Factorization* h_buffer, int bufferNumElements) {
  for (int i = 0; i < bufferNumElements; i++) {
    const Factorization& bufferElement = h_buffer[i];
    if (!isAllZeroes(&bufferElement)) {
      saveSingleFactorization(bufferElement);
    }
  }
}

bool newBoundFromExistingState_inPlace(int element, const Factorization* generators, const Factorization* bound, const Factorization* lastCandidate, //FactorizationCandidateState* state,
   Factorization* newBound, int dim, const int memoDim) {
  // set j leftmost nonzero such that a_i neq b_i
  int j = -1;
  const Factorization& g = *generators;
  const Factorization& l = *lastCandidate;
  const Factorization& b = *bound;

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

int runParallelDynamicLexicographic(int element, const Factorization generators, bool hostMemoPopulation) {
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
    err = cudaMemcpy(d_memo, h_memo, memoSize, cudaMemcpyHostToDevice, "h_memo");
  // } else {
    // actualTopOfMemo = populateDynamicFactorizations_factorizationwiseParallel(d_memo, &(generators[DIM - MEMO_DIM]));
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