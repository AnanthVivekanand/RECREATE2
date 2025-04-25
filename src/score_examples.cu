#include "miner.hpp"
#include <string>

__device__ __host__
int32_t score_lz(const U160 &addr) {
    int32_t cnt = 0;
    for (int i = 0; i < 20; i++) {
        uint8_t hi = addr[i] >> 4, lo = addr[i] & 0xF;
        if (hi == 0) {
            cnt++;
            if (lo == 0) cnt++;
            else break;
        } else break;
    }
    return cnt;
}

// Host helper: map string to integer mode
int pick_score_mode(const std::string &s) {
    if (s == "lz") return 0;
    // add more modes here...
    return 0;
}