#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <argparse/argparse.hpp>
#include <nlohmann/json.hpp>
#include "miner.hpp"
#include "util.hpp"
#include "benchmark.hpp"

#ifdef HAVE_MPI
#include <mpi.h>
#endif

using json = nlohmann::json;

// Parse a hex prefix string (e.g. "0xC0FFEE" or "C0FFEE") into bytes and nibble count.
static void parse_prefix(const std::string& raw, uint8_t* out_bytes, int& out_nibbles) {
    std::string s = raw;
    if (s.size() >= 2 && s[0] == '0' && s[1] == 'x') s = s.substr(2);
    out_nibbles = static_cast<int>(s.size());
    if (out_nibbles > 40) {
        std::cerr << "[ERR] Prefix too long (max 40 hex chars)\n";
        exit(1);
    }
    for (int i = 0; i < out_nibbles; i += 2) {
        uint8_t hi = hex_char(s[i]);
        uint8_t lo = (i + 1 < out_nibbles) ? hex_char(s[i + 1]) : 0;
        out_bytes[i / 2] = (hi << 4) | lo;
    }
}

// Parse --config JSON file and populate LaunchCfg for multi-target mode.
static bool parse_config(const std::string& path, LaunchCfg& cfg, std::array<uint8_t,20>& factory_out) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[ERR] Cannot open config file: " << path << "\n";
        return false;
    }

    json j;
    try { j = json::parse(f); }
    catch (const json::parse_error& e) {
        std::cerr << "[ERR] JSON parse error: " << e.what() << "\n";
        return false;
    }

    if (!j.contains("factory")) {
        std::cerr << "[ERR] Config missing 'factory' field\n";
        return false;
    }
    factory_out = hex_to_bytes<20>(j["factory"].get<std::string>());
    cfg.deployer = factory_out.data();
    cfg.create3 = true;
    cfg.multi_target = true;

    if (!j.contains("targets") || !j["targets"].is_array()) {
        std::cerr << "[ERR] Config missing 'targets' array\n";
        return false;
    }

    auto& targets = j["targets"];
    if (targets.size() > MAX_TARGETS) {
        std::cerr << "[ERR] Too many targets (max " << MAX_TARGETS << ")\n";
        return false;
    }

    cfg.num_targets = static_cast<uint32_t>(targets.size());

    for (uint32_t i = 0; i < cfg.num_targets; i++) {
        auto& tj = targets[i];
        TargetSpec& spec = cfg.targets[i];
        memset(&spec, 0, sizeof(TargetSpec));
        spec.id = static_cast<uint8_t>(i);

        // Name
        std::string name = tj.value("name", "target_" + std::to_string(i));
        strncpy(spec.name, name.c_str(), sizeof(spec.name) - 1);

        // Type
        std::string type_str = tj.value("type", "prefix");
        if (type_str == "prefix") spec.type = TGT_PREFIX;
        else if (type_str == "leading_zeros") spec.type = TGT_LEADING_ZEROS;
        else if (type_str == "prefix_plus_zeros") spec.type = TGT_PREFIX_PLUS_ZEROS;
        else {
            std::cerr << "[ERR] Unknown target type: " << type_str << "\n";
            return false;
        }

        // Prefix (for PREFIX and PREFIX_PLUS_ZEROS types)
        if (tj.contains("prefix")) {
            uint8_t prefix_bytes[20] = {};
            int prefix_nibbles = 0;
            parse_prefix(tj["prefix"].get<std::string>(), prefix_bytes, prefix_nibbles);
            precompute_target_masks(spec, prefix_bytes, prefix_nibbles);
        }

        // Threshold
        if (tj.contains("threshold")) {
            spec.threshold = tj["threshold"].get<uint32_t>();
        } else if (spec.type == TGT_PREFIX) {
            // Binary prefix: any match satisfies
            spec.threshold = spec.prefix_nibbles;
        } else {
            spec.threshold = 1;
        }

        std::cout << "[INFO] Target " << i << ": " << spec.name
                  << " (type=" << type_str
                  << ", nibbles=" << (int)spec.prefix_nibbles
                  << ", threshold=" << spec.threshold << ")\n";
    }

    return true;
}

int main(int argc, char** argv)
{
    argparse::ArgumentParser p("create2_miner");
    p.add_argument("--deployer");
    p.add_argument("--init-hash");
    p.add_argument("--create3").default_value(false).implicit_value(true)
        .help("CREATE3 mode: deployer is the factory address, init-hash is ignored (uses Solady proxy hash)");
    p.add_argument("--score").default_value(0);   // leading-zeros
    p.add_argument("--threshold").default_value(32).scan<'i',int>();
    p.add_argument("--threads").default_value(0).scan<'i',int>();
    p.add_argument("--blocks").scan<'i',int>();       // optional override
    p.add_argument("--test-salt");
    p.add_argument("--prefix").help("Mine for address with this hex prefix (e.g. 0xC0FFEE)");
    p.add_argument("--device").default_value("gpu").help("Device type: gpu or cpu");
    p.add_argument("--mpi").default_value(false).implicit_value(true).help("Enable MPI for distributed computing");
    p.add_argument("--benchmark").default_value(false).implicit_value(true).help("Run benchmark mode with built-in test values");
    p.add_argument("--config").help("JSON config file for multi-target mining");
    p.parse_args(argc, argv);

    bool benchmark_mode = p.get<bool>("--benchmark");

    if (benchmark_mode) {
        std::string device = p.get<std::string>("--device");
        bool c3 = p.get<bool>("--create3");
        bool use_prefix = p.present("--prefix").has_value();
        run_benchmark(device, c3, use_prefix);
        return 0;
    }

    // Multi-target config mode
    if (p.present("--config")) {
        std::string config_path = p.get("--config");
        LaunchCfg cfg;
        std::array<uint8_t,20> factory;

        if (!parse_config(config_path, cfg, factory)) return 1;

        uint32_t threads = p.get<int>("--threads");
        uint32_t blocks  = p.present("--blocks") ? p.get<int>("--blocks") : 0;

        std::string device = p.get<std::string>("--device");
        if (device == "gpu") {
#ifdef HAVE_CUDA
            run_kernel_multi(cfg, blocks, threads);
#else
            std::cerr << "[ERR] GPU support not compiled in\n";
            return 1;
#endif
        } else if (device == "cpu") {
            int num_threads = threads ? threads : std::thread::hardware_concurrency();
            std::cout << "[INFO] Using " << num_threads << " CPU threads\n";
            run_cpu_mining_multi(cfg, num_threads);
        } else {
            std::cerr << "[ERR] Invalid device type: " << device << "\n";
            return 1;
        }
        return 0;
    }

    // Single-target mode (original behavior)
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

    // single-shot test mode
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
