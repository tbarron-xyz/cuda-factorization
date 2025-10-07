[![build](https://github.com/tbarron-xyz/cuda-factorization/actions/workflows/build.yml/badge.svg)](https://github.com/tbarron-xyz/cuda-factorization/actions/workflows/build.yml)

Parallelizing the lexicographic algorithm: https://arxiv.org/abs/2405.07989

Parallelizing the dynamic algorithm, adding dynamic steps to the parallel lexicographic algorithm: https://arxiv.org/abs/2407.20474

# cuda-factorization

This project implements a CUDA-accelerated algorithm for computing all possible factorizations of a given integer into a specified set of generators, using memoization and parallel processing to optimize performance.

## Project Structure

- `memofactorize.cu`: The main CUDA source file containing the factorization algorithm, memoization logic, and GPU kernel implementations.
- `.github/workflows/build.yml`: GitHub Actions workflow for automated building and testing of the project.
- `runmemo.all.bat`: Windows batch script that compiles multiple variants of the program with different dimension and memoization settings, then runs benchmarks on various input sizes.
- `runmemo.one.bat`: Windows batch script that compiles and runs the default configuration of the program.

## Purpose

The program solves the problem of finding all ways to express a target number as a sum of multiples of given generator values, where the order of terms matters. It uses:

- **Dynamic Programming with Memoization**: Precomputes factorizations for smaller numbers to avoid redundant calculations.
- **GPU Parallelization**: Distributes the computation across CUDA threads to handle large search spaces efficiently.
- **Lexicographic Search**: Systematically explores the factorization space in lexicographical order.
- **Memory Management**: Uses buffers and device memory to manage large result sets without running out of GPU memory.

Key features include:
- Configurable dimensions (DIM) for the factorization space
- Adjustable memoization depth (MEMO_DIM) and cardinality
- Support for different generator sets
- Performance benchmarking and CSV output
- Work splitting and bound management for parallel threads

The algorithm is particularly suited for computational number theory applications involving counting or enumerating factorizations in abelian or free monoid settings.

## Build and Run

### Prerequisites
- NVIDIA CUDA Toolkit (tested with 12.5.0)
- Access to NVIDIA CUDA Samples repository (for Common/ headers)

### Building
To build the default configuration (DIM=8, MEMO_DIM=4):
```bash
mkdir bin
nvcc ./memofactorize.cu -DDIM=8 -DMEMO_DIM=4 -o bin/memofactorize8_4 -allow-unsupported-compiler -I ./cuda-samples/Common/
```

For other configurations, adjust the `-DDIM` and `-DMEMO_DIM` defines as needed (see `runmemo.all.bat` for examples).

### Running
Execute the compiled binary with the target number and optional generator values:
```bash
./bin/memofactorize8_4 [element] [gen1] [gen2] ... [gen8]
```

If no arguments are provided, it uses default values (element=1500, generators based on DIM).

Example:
```bash
./bin/memofactorize8_4 1000 13 36 37 38 39 40 41 42
```

The program outputs factorization results, performance metrics, and appends timing data to `runtimes.csv`.
