#include "keccak.hpp"
#include <cstring>
#include <cstdio>

// The initial 0xff byte followed by deployer (20B) + pre‑allocated space for salt (32B) + initHash (32B)
extern __constant__ uint64_t template85[17];

__device__ inline void ROL(uint64_t &x, int r) {
    if (r == 0) return;
    x = (x << r) | (x >> (64 - r));
}

__device__ __attribute__((always_inline)) void keccak_f1600_unrolled(const State &input, State &output) {
/*
        for (int i = 0; i < 25; ++i) {
            for (int j = 0; j < 8; j++) {
                printf("%02x", (input[i] >> (8 * j) & 0xFF));
            }
            printf(" ");
        }
        printf("\n");
*/
    static const int R[25] = {
         0,  1, 62, 28, 27,
        36, 44,  6, 55, 20,
         3, 10, 43, 25, 39,
        41, 45, 15, 21,  8,
        18,  2, 61, 56, 14
    };

    uint64_t C[5], D[5], B[25], T[5];

    // First round: read from input, write to output
    // Theta
    #pragma unroll
    for (int x = 0; x < 5; ++x) {
        C[x] = input[x] ^ input[x + 5] ^ input[x + 10] ^ input[x + 15] ^ input[x + 20];
    }
    #pragma unroll
    for (int x = 0; x < 5; ++x) {
        D[x] = C[(x + 4) % 5] ^ ((C[(x + 1) % 5] << 1) | (C[(x + 1) % 5] >> 63));
    }
    // Rho + Pi
    #pragma unroll
    for (int i = 0; i < 25; ++i) {
        int x = i % 5;
        int y = i / 5;
        uint64_t v = input[i] ^ D[x];
        ROL(v, R[i]);
        int xP = y;
        int yP = (2 * x + 3 * y) % 5;
        int j = xP + 5 * yP;
        B[j] = v;
    }
    // Chi
    #pragma unroll
    for (int y = 0; y < 25; y += 5) {
        #pragma unroll
        for (int x = 0; x < 5; ++x) T[x] = B[y + x];
        #pragma unroll
        for (int x = 0; x < 5; ++x) output[y + x] = T[x] ^ ((~T[(x + 1) % 5]) & T[(x + 2) % 5]);
    }
    // Iota
    output[0] ^= RC[0];

    // Rounds 1 to 23: in-place on output
    #pragma unroll
    for (int r = 1; r < 24; ++r) {
        // Theta
        #pragma unroll
        for (int x = 0; x < 5; ++x) {
            C[x] = output[x] ^ output[x + 5] ^ output[x + 10] ^ output[x + 15] ^ output[x + 20];
        }
        #pragma unroll
        for (int x = 0; x < 5; ++x) {
            D[x] = C[(x + 4) % 5] ^ ((C[(x + 1) % 5] << 1) | (C[(x + 1) % 5] >> 63));
            #pragma unroll
            for (int y = 0; y < 25; y += 5) {
                output[x + y] ^= D[x];
            }
        }
        // Rho + Pi
        #pragma unroll
        for (int i = 0; i < 25; ++i) {
            int x = i % 5;
            int y = i / 5;
            uint64_t v = output[i];
            ROL(v, R[i]);
            int xP = y;
            int yP = (2 * x + 3 * y) % 5;
            int j = xP + 5 * yP;
            B[j] = v;
        }
        // Chi
        #pragma unroll
        for (int y = 0; y < 25; y += 5) {
            #pragma unroll
            for (int x = 0; x < 5; ++x) T[x] = B[y + x];
            #pragma unroll
            for (int x = 0; x < 5; ++x) output[y + x] = T[x] ^ ((~T[(x + 1) % 5]) & T[(x + 2) % 5]);
        }
        // Iota
        output[0] ^= RC[r];
    }
}

__device__
State load_template() {
    State s{};
    #pragma unroll
    for(int i = 0; i < 17; i++) {
        s[i] = template85[i];
    }
    // remaining lanes are already zero
    return s;
}
