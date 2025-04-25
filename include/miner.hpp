#pragma once
#include <cstdint>
#include <array>
#include <string>
#include "keccak.hpp"

using U160    = std::array<uint8_t,20>;
using ScoreFn = int32_t(*)(const U160&);

int pick_score_mode(const std::string &s);

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
void run_kernel(const LaunchCfg& cfg,
                uint32_t blocks,
                uint32_t threads);

