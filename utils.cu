#include "types.h"
#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <exception>

#define vfprintf fprintf  // macro for optionally disabling all printing
#define vprintf printf

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
}

__device__ bool d_isAllZeroes(const int* fP, int dim = MEMO_DIM) {
  const int* f = fP;
  for (int i = 0; i < dim; i++) {
    if (f[i] != 0) { return false; }
  }
  return true;
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