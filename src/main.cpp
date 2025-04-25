#include <iostream>
#include <vector>
#include <thread>
#include <cuda_runtime.h>
#include <argparse/argparse.hpp>
#include "miner.hpp"
#include "util.hpp"

int main(int argc, char** argv)
{
    argparse::ArgumentParser p("create2_miner");
    p.add_argument("--deployer").required();
    p.add_argument("--init-hash").required();
    p.add_argument("--score").default_value(0);   // leading‑zeros
    p.add_argument("--threshold").default_value(32).scan<'i',int>();
    p.add_argument("--threads").default_value(0).scan<'i',int>();
    p.add_argument("--blocks").scan<'i',int>();       // optional override
    p.add_argument("--test-salt");
    p.parse_args(argc, argv);

    auto dep  = hex_to_bytes<20>(p.get("--deployer"));
    auto init = hex_to_bytes<32>(p.get("--init-hash"));

    // single‑shot test mode
    if (p.present("--test-salt")) {
        auto salt = hex_to_bytes<32>(p.get("--test-salt"));
        auto addr = create2_address_gpu(dep.data(), salt.data(), init.data());
        std::cout << "0x" << to_hex(addr) << '\n';
        return 0;
    }

    LaunchCfg cfg;
    cfg.start     = 0;
    cfg.target    = p.get<int>("--threshold");
    cfg.scoreMode = p.get<int>("--score");
    cfg.deployer  = dep.data();
    cfg.initHash  = init.data();
    cfg.step      = 1;

    std::cerr << "[INFO] threshold=" << cfg.target << "\n";

    uint32_t threads = p.get<int>("--threads");
    uint32_t blocks  = p.present("--blocks")
                        ? p.get<int>("--blocks")
                        : 0;

    run_kernel(cfg, blocks, threads);

    return 0;
}
