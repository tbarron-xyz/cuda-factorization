#include <stdio.h>
#include <ctime>
#include <chrono>
#include <string>
#include <exception>
#include <iostream>
#include <fstream>
#include <format>
#include <filesystem>

#include <helper_cuda.h>

#include "config.h"
#include "types.h"
#include "utils.h"
#include "cuda_utils.h"
#include "printing.h"
#include "memo.h"
#include "factorization.h"
#include "kernels.h"
#include "memory.h"

// To build, place this file in the NVIDIA "cuda-samples" repository and run "nvcc .\factorize.cu -o factorize.exe -allow-unsupported-compiler -I Common\"

// Global state for saving factorizations
int savedFactorizationCounter = 0;
Factorization* savedFacs = new Factorization[1000](); // []() zero-initializes

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