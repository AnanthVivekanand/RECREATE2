#include "miner.hpp"
#include "keccak.hpp"
#include <cuda_runtime.h>
#include <iostream>

// first 136 bytes of the Keccak rate block
__constant__ uint8_t template85[136];

// extract bytes 12–31 from the 200‑byte sponge output
__device__ __forceinline__
U160 tail20bytes(const State &s) {
    U160 out;
    // reinterpret the raw state as bytes
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&s);
    #pragma unroll
    for (int i = 0; i < 20; i++) {
        out[i] = ptr[12 + i];
    }
    return out;
}

__global__ void mine(uint64_t start,
                     uint64_t step,
                     uint64_t target,
                     ScoreFn score,
                     uint64_t* out)
{
    uint64_t gid        = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t salt       = start + gid;
    uint64_t bestPacked = 0;

    State base = load_template();
    for (;; salt += step) {
        State s = base;
        put_salt(s, salt, 0);          // high half always zero
        keccak_f1600_unrolled(s);
        U160 addr = tail20bytes(s);
        int32_t sc = score(addr);
        if (sc > int32_t(bestPacked >> 32)) {
            bestPacked = (uint64_t(sc) << 32) | (salt & 0xffffffffu);
            if (sc >= int32_t(target)) break;
        }
    }
    atomicMax((unsigned long long*)out, bestPacked);
}

void run_kernel(const LaunchCfg& cfg, uint32_t blocks, uint32_t threads)
{
    // 1) build & upload the 136‑byte template
    uint8_t hostTpl[136] = {};
    hostTpl[0] = 0xff;
    cudaMemcpy(hostTpl + 1,    cfg.deployer, 20, cudaMemcpyHostToHost);
    // salt bytes left as zero
    cudaMemcpy(hostTpl + 53,   cfg.initHash, 32, cudaMemcpyHostToHost);
    hostTpl[85]  = 0x01;
    hostTpl[135] = 0x80;
    cudaMemcpyToSymbol(template85, hostTpl, 136);

    // 2) pick block count if unspecified
    if (!blocks) {
        int maxPerSM;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxPerSM, (void*)mine, threads, 0);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        blocks = maxPerSM * prop.multiProcessorCount;
    }

    // 3) launch & retrieve result
    uint64_t* d_best;
    cudaMalloc(&d_best, sizeof(uint64_t));
    cudaMemset(d_best, 0, sizeof(uint64_t));

    mine<<<blocks,threads>>>(cfg.start,
                             cfg.step ? cfg.step : uint64_t(blocks)*threads,
                             cfg.target,
                             cfg.score,
                             d_best);
    cudaDeviceSynchronize();

    uint64_t bestPacked;
    cudaMemcpy(&bestPacked, d_best, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_best);

    uint32_t bestScore = bestPacked >> 32;
    uint32_t bestSalt  = uint32_t(bestPacked);
    std::cout << "Found score=" << bestScore
              << " salt=" << bestSalt << "\n";
}
