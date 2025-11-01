#ifndef UTILS_H
#define UTILS_H

#include "config.h"
#include "types.h"
#include <cstdio>

// Forward declarations for CUDA functions that will be defined in cuda_utils.h
// Note: CUDA functions will be defined in cuda_utils.h to avoid compilation issues in pure C++ files

// Mathematical utilities
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

bool isZeroesOnLeft(const int* f, int i) {
  for (int j = 0; j < i; j++) {
    if (j < i) {
      if (f[j] != 0) {
        return false;
      }
    }
  }
  return true;
}

// Copy and comparison functions
void copy(const Factorization* source, Factorization* target, int dim) { // todo should  just use memcpy/cudamemcpy directly
  for (int j = 0; j < dim; j++) {
    (*target)[j] = (*source)[j];
  }
}

void copy2(int* source, int* target, int dim) {
  for (int j = 0; j < dim; j++) {
    target[j] = source[j];
  }
}

//unused.
bool facIsEqual(const Factorization* a, const Factorization* b) {
  const Factorization& A = *a; const Factorization& B = *b;
  for (int i = 0; i < DIM; i++) {
    if (A[i] != B[i]) { return false;}
  }
  return true;
}

// Lexicographic comparison
bool lexicogLeq(const Factorization* a, const Factorization* b) { 
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

// Phi functions (host version)
int phi(const Factorization* factorization, const Factorization* generators) {
  int sum = 0;
  for (int i = 0; i < DIM; i++) {
    sum += (*factorization)[i] * (*generators)[i];
  }
  return sum;
}

// Index manipulation
void incrementIndex(int* f, int index) {
  f[index] ++;
}

#endif // UTILS_H