#include "keccak.hpp"
#include <cstring>

// The initial 0xff byte followed by deployer (20B) + pre-allocated space for salt (32B) + initHash (32B)
extern __constant__ uint8_t template85[136];

__constant__ uint64_t RC[24] = {
  0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
  0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
  0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
  0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
  0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
  0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

// safe 64‑bit rotate‑left (handles r==0 cleanly)
inline __device__ __host__
uint64_t rotl64(uint64_t v, unsigned r) {
    return (v << (r & 63)) | (v >> ((64 - r) & 63));
}

// Keccak round constants must be accessible on device
__device__ const int KECCAK_R[25] = {
    0,36,3,41,18,1,44,10,45,2,62,6,43,15,61,28,55,25,21,56,27,20,39,8,14
};

__host__ __device__ void keccak_f1600_unrolled(State &A) {
    static const int R[25] = {
        0, 36,  3, 41, 18,
        1, 44, 10, 45,  2,
        62,  6, 43, 15, 61,
        28, 55, 25, 21, 56,
        27, 20, 39,  8, 14
    };

    for (int r = 0; r < 24; ++r) {
        // Θ step
        uint64_t C[5], D;
        for (int x = 0; x < 5; ++x) {
            C[x] = A[x] ^ A[x + 5] ^ A[x + 10] ^ A[x + 15] ^ A[x + 20];
        }
        for (int x = 0; x < 5; ++x) {
            D = C[(x + 4) % 5] ^ ((C[(x + 1) % 5] << 1) | (C[(x + 1) % 5] >> 63));
            for (int y = 0; y < 25; y += 5) {
                A[x + y] ^= D;
            }
        }

        // Rho + Pi
        State B;
        for (int i = 0; i < 25; ++i) {
            int x = i % 5;
            int y = i / 5;
            uint64_t v = A[i];
            v = rotl64(v, R[i]); // rotate by R[x+5*y]

            // Pi mapping: (x, y) → (x' = y, y' = (2x + 3y) mod 5)
            int xP = y;
            int yP = (2 * x + 3 * y) % 5;
            int j  = xP + 5 * yP;
            B[j] = v;
        }

        // χ step
        for (int y = 0; y < 25; y += 5) {
            uint64_t T[5];
            for (int x = 0; x < 5; ++x) {
                T[x] = B[y + x];
            }
            for (int x = 0; x < 5; ++x) {
                A[y + x] = T[x] ^ ((~T[(x + 1) % 5]) & T[(x + 2) % 5]);
            }
        }

        // ι step
        A[0] ^= RC[r];
    }
}

// Remove __forceinline__ to make function visible to other compilation units
__device__
State load_template() {
    State s{};
    // template[] in constant memory, rate=136 B => 17 lanes
    const uint64_t *tmpl = reinterpret_cast<const uint64_t*>(template85);
    #pragma unroll
    for(int i=0; i<17; i++) s[i] = tmpl[i];
    // rest are zero by default
    return s;
}

// Remove __forceinline__ to make function visible to other compilation units
__device__
void put_salt(State &s, uint64_t salt_lo, uint64_t salt_hi) {
    // bytes 21–52 cover lanes 2‒4 (partial)
    // assume little-endian: lane2 ^= salt_lo, lane3 ^= salt_hi
    s[2] ^= salt_lo;
    s[3] ^= salt_hi;
}

// Host-side wrapper that handles GPU device calls
std::array<uint8_t,20> create2_address_gpu(const uint8_t deployer[20],
                                           const uint8_t salt[32],
                                           const uint8_t initHash[32]) {
    // Create a host-side version that mimics the device-side processing
    State s{};
    
    // Manually load template-like data
    uint8_t buf[136] = {0};
    buf[0] = 0xff;
    memcpy(buf+1, deployer, 20);
    memcpy(buf+21, salt, 32);
    memcpy(buf+53, initHash, 32);
    buf[85]      = 0x01;    // start-of-pad
    buf[135]    |= 0x80;    // end-of-pad 
    
    // Convert to lanes and load into state
    uint64_t* s_ptr = reinterpret_cast<uint64_t*>(s.data());
    const uint64_t* buf_ptr = reinterpret_cast<const uint64_t*>(buf);
    for(int i=0; i<17; i++) {
        s_ptr[i] = buf_ptr[i];
    }
    
    // Process
    keccak_f1600_unrolled(s);
    
    // Extract address
    std::array<uint8_t,20> out;
    uint8_t *p = reinterpret_cast<uint8_t*>(s.data());
    memcpy(out.data(), p+12, 20);
    return out;
}
