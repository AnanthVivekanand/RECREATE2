#include <iostream>
#include <vector>
#include <thread>
#include <argparse/argparse.hpp>
#include "miner.hpp"
#include "util.hpp"
#include "benchmark.hpp"

#ifdef HAVE_MPI
#include <mpi.h>
#endif

int main(int argc, char** argv)
{
    argparse::ArgumentParser p("create2_miner");
    p.add_argument("--deployer");
    p.add_argument("--init-hash");
    p.add_argument("--create3").default_value(false).implicit_value(true)
        .help("CREATE3 mode: deployer is the factory address, init-hash is ignored (uses Solady proxy hash)");
    p.add_argument("--score").default_value(0);   // leading‑zeros
    p.add_argument("--threshold").default_value(32).scan<'i',int>();
    p.add_argument("--threads").default_value(0).scan<'i',int>();
    p.add_argument("--blocks").scan<'i',int>();       // optional override
    p.add_argument("--test-salt");
    p.add_argument("--prefix").help("Mine for address with this hex prefix (e.g. 0xC0FFEE)");
    p.add_argument("--device").default_value("gpu").help("Device type: gpu or cpu");
    p.add_argument("--mpi").default_value(false).implicit_value(true).help("Enable MPI for distributed computing");
    p.add_argument("--benchmark").default_value(false).implicit_value(true).help("Run benchmark mode with built-in test values");
    p.parse_args(argc, argv);

    bool benchmark_mode = p.get<bool>("--benchmark");
    
    if (benchmark_mode) {
        std::string device = p.get<std::string>("--device");
        bool c3 = p.get<bool>("--create3");
        bool use_prefix = p.present("--prefix").has_value();
        run_benchmark(device, c3, use_prefix);
        return 0;
    }
    
    bool create3_mode = p.get<bool>("--create3");

    if (!p.present("--deployer")) {
        std::cerr << "[ERR] --deployer is required (factory address for CREATE3)\n";
        return 1;
    }
    if (!create3_mode && !p.present("--init-hash")) {
        std::cerr << "[ERR] --init-hash is required in CREATE2 mode\n";
        return 1;
    }

    auto dep = hex_to_bytes<20>(p.get("--deployer"));
    std::array<uint8_t,32> init{};
    if (!create3_mode) {
        init = hex_to_bytes<32>(p.get("--init-hash"));
    }

    // single‑shot test mode
    if (p.present("--test-salt")) {
        auto salt = hex_to_bytes<32>(p.get("--test-salt"));
        if (create3_mode) {
            auto addr = create3_address_cpu(dep.data(), salt.data());
            std::cout << "0x" << to_hex(addr) << '\n';
        } else {
            auto addr = create2_address_cpu(dep.data(), salt.data(), init.data());
            std::cout << "0x" << to_hex(addr) << '\n';
        }
        return 0;
    }

    LaunchCfg cfg;
    cfg.start     = 0;
    cfg.target    = p.get<int>("--threshold");
    cfg.scoreMode = p.get<int>("--score");
    cfg.deployer  = dep.data();
    cfg.initHash  = init.data();
    cfg.step      = 1;
    cfg.create3   = create3_mode;

    if (p.present("--prefix")) {
        std::string prefix_str = p.get("--prefix");
        if (prefix_str.size() >= 2 && prefix_str[0] == '0' && prefix_str[1] == 'x')
            prefix_str = prefix_str.substr(2);
        cfg.prefixNibbles = prefix_str.size();
        if (cfg.prefixNibbles > 40) {
            std::cerr << "[ERR] Prefix too long (max 40 hex chars)\n";
            return 1;
        }
        for (int i = 0; i < cfg.prefixNibbles; i += 2) {
            uint8_t hi = hex_char(prefix_str[i]);
            uint8_t lo = (i + 1 < cfg.prefixNibbles) ? hex_char(prefix_str[i + 1]) : 0;
            cfg.prefixBytes[i / 2] = (hi << 4) | lo;
        }
        cfg.target = cfg.prefixNibbles;
        std::cout << "[INFO] Mining for prefix: 0x" << prefix_str << " (" << cfg.prefixNibbles << " nibbles)\n";
    }

    if (create3_mode) {
        std::cout << "[INFO] CREATE3 mode enabled (Solady proxy hash)\n";
    }

    bool use_mpi = p.get<bool>("--mpi");
    int rank = 0, size = 1;
    if (use_mpi) {
#ifdef HAVE_MPI
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        cfg.start = uint64_t(rank) * 1000000000000ULL; // 10^12 per rank
        std::cout << "[INFO] MPI enabled: rank " << rank << " of " << size << "\n";
#else
        std::cerr << "[ERR] MPI support not compiled in\n";
        return 1;
#endif
    }

    uint32_t threads = p.get<int>("--threads");
    uint32_t blocks  = p.present("--blocks")
                        ? p.get<int>("--blocks")
                        : 0;

    std::string device = p.get<std::string>("--device");
    if (device == "gpu") {
#ifdef HAVE_CUDA
        std::cout << "[INFO] Using GPU device\n";
        run_kernel(cfg, blocks, threads, use_mpi, rank, size);
#else
        std::cerr << "[ERR] GPU support not compiled in\n";
        return 1;
#endif
    } else if (device == "cpu") {
        int num_threads = threads ? threads : std::thread::hardware_concurrency();
        std::cout << "[INFO] Using " << num_threads << " CPU threads\n";
        run_cpu_mining(cfg, num_threads, use_mpi, rank, size);
    } else {
        std::cerr << "[ERR] Invalid device type: " << device << "\n";
        return 1;
    }

#ifdef HAVE_MPI
    if (use_mpi) {
        MPI_Finalize();
    }
#endif

    return 0;
}
