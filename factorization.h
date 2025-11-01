#ifndef FACTORIZATION_H
#define FACTORIZATION_H

#include "config.h"
#include "types.h"
#include "utils.h"
#include "printing.h"
#include "cuda_utils.h"
#include <cstdio>
#include <exception>

#ifdef __CUDACC__
// Main factorization algorithm (device)
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
#endif

// Bound calculation (host)
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

// Work distribution (host)
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

#endif // FACTORIZATION_H