#ifndef CONFIG_H
#define CONFIG_H

#include <cstddef>

// Element configuration
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

// Dimension configuration
#ifndef DIM
#define DIM 8
// 3;
// 4;
// 5;
// 6;
// 7;
// 9;
#endif

// Buffer configuration
// good to make this larger than 1x or 2x of MEMO_cardinality, so that it can hold several kernels worth of outputs //5000
// make this too large and it make become slower even without running out of memory
constexpr size_t bufferPerThread = 
// 5000;
40000;  

// Memoization configuration
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

// Optimization flags
constexpr bool useModuloOptimization = 
// false;
true;

// Kernel execution configuration
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

// CUDA grid configuration
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

// Grid index macro
#define GridIndex() blockIdx.x * blockDim.x + threadIdx.x

#endif // CONFIG_H