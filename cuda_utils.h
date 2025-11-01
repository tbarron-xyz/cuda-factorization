#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include "config.h"
#include "types.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <fstream>

// CUDA includes - only include when compiling with CUDA
#ifdef __CUDACC__
#include <cuda_runtime.h>

// Device-side utilities
__device__ bool d_isAllZeroes(const int* fP, int dim = MEMO_DIM) {
  const int* f = fP;
  for (int i = 0; i < dim; i++) {
    if (f[i] != 0) { return false; }
  }
  return true;
  // return f[0] =
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

// Device-side phi function
__device__ int d_phi(const Factorization* factorization, const Factorization* generators) {
  int sum = 0;
  // #pragma unroll // can't unroll addition without atomicAdd.
  for (int i = 0; i < DIM; i++) {
    sum += (*factorization)[i] * (*generators)[i];
  }
  return sum;
}

#endif // __CUDACC__

// CUDA error handling wrappers (only available when CUDA is present)
#ifdef __CUDACC__
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

cudaError_t cudaMemset ( void* devPtr, int  value, size_t count , std::string name) {//extra arg adds error checking
  cudaError_t err = cudaMemset(devPtr, value, count);
  if (err != cudaSuccess) {
    vfprintf(stderr, "Failed to zero device buffer: %s\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  return err;
}

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

// Device buffer management
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
#endif // __CUDACC__

// File utilities (available in both CUDA and non-CUDA builds)
bool fileExists(const char *fileName)
{
    std::ifstream infile(fileName);
    return infile.good();
}

#endif // CUDA_UTILS_H