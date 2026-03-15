#include "benchmark.hpp"
#include "miner.hpp"
#include "util.hpp"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <signal.h>
#include <unistd.h>
#include <thread>

static void timeout_handler(int sig) {
    std::cout << "\n[BENCHMARK] 10-second benchmark completed\n";
    exit(0);
}

void run_benchmark(const std::string& device, bool create3, bool use_prefix) {
    std::string mode = create3 ? "CREATE3" : "CREATE2";
    std::string match_mode = use_prefix ? "prefix" : "leading-zero";
    std::cout << "[BENCHMARK] Starting " << mode << " (" << match_mode << ") benchmark mode\n";

    std::cout << "[BENCHMARK] Running correctness verification...\n";
    int test_result = system("./tests");
    if (test_result != 0) {
        std::cerr << "[BENCHMARK] Correctness verification failed!\n";
        exit(1);
    }
    std::cout << "[BENCHMARK] ✓ Correctness verification passed\n\n";

    signal(SIGALRM, timeout_handler);
    alarm(10);

    std::cout << "[BENCHMARK] Starting 10-second " << device << " " << mode << " (" << match_mode << ") performance test...\n";

    auto deployer = hex_to_bytes<20>("0x48E516B34A1274f49457b9C6182097796D0498Cb");
    auto init_hash = hex_to_bytes<32>("0x94d114296a5af85c1fd2dc039cdaa32f1ed4b0fe0868f02d888bfc91feb645d9");

    LaunchCfg cfg;
    cfg.start = 0;
    cfg.target = 9999;
    cfg.scoreMode = 0;
    cfg.deployer = deployer.data();
    cfg.initHash = init_hash.data();
    cfg.step = 1;
    cfg.create3 = create3;

    if (use_prefix) {
        // Unreachable 40-nibble prefix for sustained benchmarking
        auto prefix = hex_to_bytes<20>("0xDEADBEEFCAFEBABE0123456789ABCDEF01234567");
        memcpy(cfg.prefixBytes, prefix.data(), 20);
        cfg.prefixNibbles = 40;
        cfg.target = 40;
    }

    if (device == "gpu") {
#ifdef HAVE_CUDA
        run_kernel(cfg, 0, 0, false, 0, 1);
#else
        std::cerr << "[BENCHMARK] GPU support not compiled in\n";
        exit(1);
#endif
    } else if (device == "cpu") {
        uint32_t num_threads = std::thread::hardware_concurrency();
        run_cpu_mining(cfg, num_threads, false, 0, 1);
    } else {
        std::cerr << "[BENCHMARK] Invalid device: " << device << "\n";
        exit(1);
    }
} 