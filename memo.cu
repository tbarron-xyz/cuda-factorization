#include "types.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>

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
        // copy preventry to output, then increment output
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