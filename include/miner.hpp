#pragma once
#include <cstdint>
#include <array>
#include <string>
#include "keccak.hpp"

#ifdef HAVE_MPI
#include <mpi.h>
#endif

using U160    = std::array<uint8_t,20>;
using ScoreFn = int32_t(*)(const U160&);

struct salt_result {
    uint64_t salt_lo;
    uint64_t salt_hi;
    uint32_t score;
};

struct LaunchCfg {
    uint64_t    start     = 0;
    uint64_t    step      = 0;
    uint64_t    target    = 0;
    int32_t     scoreMode = 0;
    const uint8_t* deployer = nullptr;
    const uint8_t* initHash = nullptr;
};

// CUDA kernel entry point (only when compiling with nvcc)
#ifdef __CUDACC__
__global__ void mine(uint64_t start,
                     uint64_t step,
                     uint64_t target,
                     int scoreMode,
                     salt_result* out,
                     uint64_t* perfCounters,
                     uint32_t deviceIdx,
                     volatile int* host_should_exit);
#endif

// Host‐side launcher
#ifdef HAVE_CUDA
void run_kernel(const LaunchCfg& cfg,
                uint32_t blocks,
                uint32_t threads,
                bool use_mpi,
                int rank,
                int size);
#endif

void run_cpu_mining(const LaunchCfg& cfg,
                    uint32_t num_threads,
                    bool use_mpi,
                    int rank,
                    int size);
