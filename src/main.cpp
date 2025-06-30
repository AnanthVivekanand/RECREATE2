#include <iostream>
#include <vector>
#include <thread>
#include <argparse/argparse.hpp>
#include "miner.hpp"
#include "util.hpp"

#ifdef HAVE_MPI
#include <mpi.h>
#endif

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
    p.add_argument("--device").required().default_value("gpu").help("Device type: gpu or cpu");
    p.add_argument("--mpi").default_value(false).implicit_value(true).help("Enable MPI for distributed computing");
    p.parse_args(argc, argv);

    auto dep  = hex_to_bytes<20>(p.get("--deployer"));
    auto init = hex_to_bytes<32>(p.get("--init-hash"));

    // single‑shot test mode
    if (p.present("--test-salt")) {
        auto salt = hex_to_bytes<32>(p.get("--test-salt"));
        auto addr = create2_address_cpu(dep.data(), salt.data(), init.data());
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
