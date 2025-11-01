#ifndef PRINTING_H
#define PRINTING_H

#include "config.h"
#include "types.h"
#include "utils.h"
#include <cstdio>
#include <string>

// Print macros
#define vfprintf fprintf  // macro for optionally disabling all printing
#define vprintf printf

// Print functions for factorizations
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

#ifdef __CUDACC__
// Device-side print function
__device__ void d_printfac(const int* f, int dim = DIM) {
  for (int i = 0; i < dim; i++) {
    printf("%d, ", f[i]);
  }
}
#endif

// Save functions
extern int savedFactorizationCounter;
extern Factorization* savedFacs;

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

#endif // PRINTING_H