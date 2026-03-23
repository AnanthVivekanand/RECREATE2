#pragma once
#include <cstdint>
#include <array>
#include <string>
#include "keccak.hpp"
#include "targets.hpp"

#ifdef HAVE_MPI
#include <mpi.h>
#endif

using U160    = std::array<uint8_t,20>;
using ScoreFn = int32_t(*)(const U160&);

struct LaunchCfg {
    uint64_t    start     = 0;
    uint64_t    step      = 0;
    uint64_t    target    = 0;
    int32_t     scoreMode = 0;
    const uint8_t* deployer = nullptr;   // factory address for CREATE3
    const uint8_t* initHash = nullptr;   // unused for CREATE3 (constant proxy hash)
    bool        create3   = false;
    uint8_t     prefixBytes[20] = {};
    int         prefixNibbles   = 0;     // 0 = leading-zero scoring mode

    // Multi-target mode
    bool        multi_target = false;
    uint32_t    num_targets  = 0;
    TargetSpec  targets[MAX_TARGETS] = {};
};

// Solady CREATE3 proxy initcode hash (constant across all Solady factories)
static constexpr uint8_t SOLADY_PROXY_INITCODE_HASH[32] = {
    0x21, 0xc3, 0x5d, 0xbe, 0x1b, 0x34, 0x4a, 0x24,
    0x88, 0xcf, 0x33, 0x21, 0xd6, 0xce, 0x54, 0x2f,
    0x8e, 0x9f, 0x30, 0x55, 0x44, 0xff, 0x09, 0xe4,
    0x99, 0x3a, 0x62, 0x31, 0x9a, 0x49, 0x7c, 0x1f
};

// CUDA kernel entry points (only when compiling with nvcc)
#ifdef __CUDACC__
__global__ void mine(uint64_t start,
                     uint64_t step,
                     uint64_t target,
                     int scoreMode,
                     salt_result* out,
                     uint64_t* perfCounters,
                     uint32_t deviceIdx,
                     volatile int* host_should_exit);

__global__ void mine_create3(uint64_t start,
                             uint64_t step,
                             uint64_t target,
                             int scoreMode,
                             salt_result* out,
                             uint64_t* perfCounters,
                             uint32_t deviceIdx,
                             volatile int* host_should_exit);

__global__ void mine_create3_multi(uint64_t start,
                                   MultiResult* out,
                                   uint64_t* perfCounters,
                                   uint32_t deviceIdx,
                                   volatile int* device_should_exit);
#endif

// Host-side launchers
#ifdef HAVE_CUDA
void run_kernel(const LaunchCfg& cfg,
                uint32_t blocks,
                uint32_t threads,
                bool use_mpi,
                int rank,
                int size);

void run_kernel_multi(const LaunchCfg& cfg,
                      uint32_t blocks,
                      uint32_t threads);
#endif

void run_cpu_mining(const LaunchCfg& cfg,
                    uint32_t num_threads,
                    bool use_mpi,
                    int rank,
                    int size);

void run_cpu_mining_multi(const LaunchCfg& cfg, uint32_t num_threads);
