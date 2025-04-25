#ifndef __CUDACC__
  #ifndef __host__
    #define __host__
  #endif
  #ifndef __device__
    #define __device__
  #endif
#endif

#pragma once
#include <cstdint>
#include <cstddef>          // for size_t
#include <array>

// 25×64‑bit state
using State = std::array<uint64_t,25>;

// Keccak‑f[1600] round constants:
// Device builds (__CUDA_ARCH__ defined) get the __constant__ copy; host builds get the static one.
#if defined(__CUDA_ARCH__)
__device__ __constant__ static const uint64_t RC[24] = {
  0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
  0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
  0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
  0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
  0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
  0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
}; 
#else
static const uint64_t RC[24] = {
  0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
  0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
  0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
  0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
  0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
  0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};
#endif

#ifdef __CUDACC__
  #define KDEV __device__
  #define KDEV_INLINE __device__ inline
#else
  #define KDEV
  #define KDEV_INLINE inline
#endif

// load the 136‑B CREATE2 template into a State (rate=136 B)
KDEV State load_template();

// overwrite bytes 21..52 of the rate block with `salt` counter
KDEV void put_salt(State &s, uint64_t salt_lo, uint64_t salt_hi = 0);

// full 24‑round unrolled Keccak‑f[1600], available on both host and device
__host__ __device__ void keccak_f1600_unrolled(const State &s, State &t);

// CPU helpers (for testing path)
std::array<uint8_t,32> keccak256_cpu(const uint8_t *data, size_t len);

// CREATE2 address calc on GPU via one keccak call
std::array<uint8_t,20> create2_address_gpu(const uint8_t deployer[20],
                                           const uint8_t salt[32],
                                           const uint8_t initHash[32]);

// CREATE2 address calc on CPU (calls keccak256_cpu)
std::array<uint8_t,20> create2_address_cpu(const uint8_t deployer[20],
                                           const uint8_t salt[32],
                                           const uint8_t initHash[32]);
