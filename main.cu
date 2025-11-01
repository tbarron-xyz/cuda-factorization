#include "types.h"
#include <string>

int main(int argc, char *argv[]) {

  int element = argc > 1 ? std::stoi(argv[1]) : ELEMENT;
  Factorization generators;
 for (int i = 0; i < DIM; i++) {
   if (argc > i + 2) { generators[i] = std::stoi(argv[i+2]); }
   else { generators[i] = i == 0 ? 13 : i > 2 ? 37 + i : 36 + i;  }
 }

  int numberOfFactorizations = runParallelDynamicLexicographic(element, generators, argc > DIM + 2);
}