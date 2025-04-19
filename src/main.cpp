#include <argparse/argparse.hpp>
#include "miner.hpp"
#include "util.hpp"

int main(int argc, char** argv)
{
    argparse::ArgumentParser p("create2_miner");
    p.add_argument("--deployer").required();
    p.add_argument("--init-hash").required();
    p.add_argument("--score").default_value(std::string("lz"));   // leading‑zeros
    p.add_argument("--threshold").scan<'i',int>().default_value(32);
    p.add_argument("--test-salt");
    p.parse_args(argc, argv);

    auto dep  = hex_to_bytes<20>(p.get("--deployer"));
    auto init = hex_to_bytes<32>(p.get("--init-hash"));

    if (p.present("--test-salt")) {          // single‑shot test
        auto salt = hex_to_bytes<32>(p.get("--test-salt"));
        auto addr = create2_address_gpu(dep.data(), salt.data(), init.data());
        std::cout << "0x" << to_hex(addr) << '\n';
        return 0;
    }

    LaunchCfg cfg;
    cfg.start   = 0;
    cfg.target  = p.get<int>("--threshold");
    cfg.score   = pick_score(p.get("--score"));
    cfg.deployer = dep.data();
    cfg.initHash = init.data();
    run_kernel(cfg);
    return 0;
}
