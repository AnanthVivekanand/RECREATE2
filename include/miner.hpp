#pragma once
#include <cstdint>
#include <array>
#include "keccak.hpp"       // for State, create2_address_gpu
#include <string>

using U160    = std::array<uint8_t,20>;
using ScoreFn = int32_t(*)(const U160&);

ScoreFn pick_score(const std::string &s);

struct LaunchCfg {
    uint64_t    start     = 0;
    uint64_t    step      = 0;
    uint64_t    target    = 0;
    ScoreFn     score     = nullptr;
    const uint8_t* deployer = nullptr;
    const uint8_t* initHash = nullptr;
};

void run_kernel(const LaunchCfg& cfg, uint32_t blocks = 0, uint32_t threads = 256);
