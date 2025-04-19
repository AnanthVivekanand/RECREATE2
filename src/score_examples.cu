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

ScoreFn pick_score(const std::string &s) {
    if (s == "lz") return score_lz;
    // add more here...
    return score_lz;
}
