#include "miner.hpp"
#include "keccak.hpp"
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include <chrono>
#include <iostream>
#include <array>
#include <cstring>
#include "util.hpp"
#include <ctime>
#include <csignal>
#include <fstream>

#define LOG_INTERVAL 5000

struct SharedData {
    std::atomic<int> should_exit{0};
    std::mutex mutex;
    salt_result best{0, 0, 0};
    std::vector<std::atomic<uint64_t>> perfCounters;

    explicit SharedData(size_t n_threads)
        : perfCounters(n_threads) {}
};

int32_t score_lz(const U160 &addr) {
    int32_t sc = 0;
    for (int i = 0; i < 20; i++) {
        if (addr[i] == 0) {
            sc += 8;
        } else if (addr[i] <= 0x0F) {
            sc += 4;
            break;
        } else {
            break;
        }
    }
    return sc;
}

inline int32_t compute_score(const U160 &addr, const LaunchCfg &cfg) {
    if (cfg.prefixNibbles > 0) {
        int n = cfg.prefixNibbles;
        int full_bytes = n / 2;
        for (int i = 0; i < full_bytes; i++) {
            if (addr[i] != cfg.prefixBytes[i]) return 0;
        }
        if (n & 1) {
            if ((addr[full_bytes] & 0xF0) != (cfg.prefixBytes[full_bytes] & 0xF0)) return 0;
        }
        return n;
    }
    return score_lz(addr);
}

// Optimized Keccak-f[1600] for CPU with loop unrolling
void keccak_f1600_cpu(const State& input, State& output) {
    static const uint64_t RC[24] = {
        0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
        0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
        0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
        0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
        0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
        0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
    };
    
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
    for (int x = 0; x < 5; ++x) {
        C[x] = input[x] ^ input[x + 5] ^ input[x + 10] ^ input[x + 15] ^ input[x + 20];
    }
    for (int x = 0; x < 5; ++x) {
        D[x] = C[(x + 4) % 5] ^ ((C[(x + 1) % 5] << 1) | (C[(x + 1) % 5] >> 63));
    }
    // Rho + Pi
    for (int i = 0; i < 25; ++i) {
        int x = i % 5;
        int y = i / 5;
        uint64_t v = input[i] ^ D[x];
        if (R[i] != 0) {
            v = (v << R[i]) | (v >> (64 - R[i]));
        }
        int xP = y;
        int yP = (2 * x + 3 * y) % 5;
        int j = xP + 5 * yP;
        B[j] = v;
    }
    // Chi
    for (int y = 0; y < 25; y += 5) {
        for (int x = 0; x < 5; ++x) {
            T[x] = B[y + x];
        }
        for (int x = 0; x < 5; ++x) {
            output[y + x] = T[x] ^ ((~T[(x + 1) % 5]) & T[(x + 2) % 5]);
        }
    }
    // Iota
    output[0] ^= RC[0];

    // Rounds 1 to 23: in-place on output
    for (int r = 1; r < 24; ++r) {
        // Theta
        for (int x = 0; x < 5; ++x) {
            C[x] = output[x] ^ output[x + 5] ^ output[x + 10] ^ output[x + 15] ^ output[x + 20];
        }
        for (int x = 0; x < 5; ++x) {
            D[x] = C[(x + 4) % 5] ^ ((C[(x + 1) % 5] << 1) | (C[(x + 1) % 5] >> 63));
            for (int y = 0; y < 25; y += 5) {
                output[x + y] ^= D[x];
            }
        }
        // Rho + Pi
        for (int i = 0; i < 25; ++i) {
            int x = i % 5;
            int y = i / 5;
            uint64_t v = output[i];
            if (R[i] != 0) {
                v = (v << R[i]) | (v >> (64 - R[i]));
            }
            int xP = y;
            int yP = (2 * x + 3 * y) % 5;
            int j = xP + 5 * yP;
            B[j] = v;
        }
        // Chi
        for (int y = 0; y < 25; y += 5) {
            for (int x = 0; x < 5; ++x) {
                T[x] = B[y + x];
            }
            for (int x = 0; x < 5; ++x) {
                output[y + x] = T[x] ^ ((~T[(x + 1) % 5]) & T[(x + 2) % 5]);
            }
        }
        // Iota
        output[0] ^= RC[r];
    }
}

inline void cpu_mine(uint64_t base, int thread_id, const LaunchCfg& cfg, SharedData& shared, uint32_t epoch) {
    uint64_t salt_hi = ((base + thread_id) << 52) | uint64_t(epoch);
    uint64_t salt_lo = 0;
    uint64_t localCount = 0;
    State s{};
    State result{};
    uint8_t* s8 = reinterpret_cast<uint8_t*>(&s);

    // Initialize template
    s8[0] = 0xff;
    std::memcpy(s8 + 1, cfg.deployer, 20);
    if (cfg.create3) {
        std::memcpy(s8 + 53, SOLADY_PROXY_INITCODE_HASH, 32);
    } else {
        std::memcpy(s8 + 53, cfg.initHash, 32);
    }
    s8[85] = 0x01;
    s8[135] = 0x80;

    // Set salt_hi
    for (int i = 0; i < 8; ++i) {
        s8[44 - i] = (salt_hi >> (8 * i)) & 0xff;
    }

    while (!shared.should_exit.load()) {
        // Set salt_lo
        for (int i = 0; i < 8; ++i) {
            s8[52 - i] = (salt_lo >> (8 * i)) & 0xff;
        }

        keccak_f1600_cpu(s, result);

        int32_t sc;
        if (cfg.create3) {
            // Second keccak: RLP CREATE from proxy address
            uint64_t r1 = result[1], r2 = result[2], r3 = result[3];
            State rlp{};
            rlp[0]  = 0x94d6ULL | ((r1 >> 32) << 16) | ((r2 & 0xFFFFULL) << 48);
            rlp[1]  = (r2 >> 16) | ((r3 & 0xFFFFULL) << 48);
            rlp[2]  = (r3 >> 16) | (0x0101ULL << 48);
            rlp[16] = 0x8000000000000000ULL;

            State final_res{};
            keccak_f1600_cpu(rlp, final_res);

            U160 addr;
            std::memcpy(addr.data(), reinterpret_cast<uint8_t*>(&final_res) + 12, 20);
            sc = compute_score(addr, cfg);
        } else {
            U160 addr;
            std::memcpy(addr.data(), reinterpret_cast<uint8_t*>(&result) + 12, 20);
            sc = compute_score(addr, cfg);
        }

        localCount++;
        if (localCount == LOG_INTERVAL) {
            shared.perfCounters[thread_id].fetch_add(localCount);
            localCount = 0;
            if (shared.should_exit.load()) break;
        }

        if (sc >= cfg.target) {
            std::lock_guard<std::mutex> lock(shared.mutex);
            if (sc > shared.best.score) {
                shared.best.score = sc;
                shared.best.salt_lo = salt_lo;
                shared.best.salt_hi = salt_hi;
            }
            shared.should_exit.store(1);
            break;
        }
        salt_lo++;
    }
    if (localCount) {
        shared.perfCounters[thread_id].fetch_add(localCount);
    }
}

void run_cpu_mining(const LaunchCfg& cfg, uint32_t num_threads, bool use_mpi, int rank, int size) {
    uint32_t epoch = (uint32_t)time(nullptr);
    std::cout << "[INFO] Epoch seed: " << epoch << "\n";

    SharedData shared(num_threads);
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (uint32_t i = 0; i < num_threads; ++i) {
        threads.emplace_back(cpu_mine, /*base=*/i, /*thread_id=*/static_cast<int>(i), std::cref(cfg), std::ref(shared), epoch);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
#ifdef HAVE_MPI
    int global_should_exit = 0;
    MPI_Request stop_request;
#endif
    int stop_flag = 0;
#ifdef HAVE_MPI
    if (use_mpi) {
        MPI_Irecv(&global_should_exit, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &stop_request);
    }
#endif

    uint64_t prev_total = 0;

    while (!shared.should_exit.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        uint64_t total = 0;
        for (const auto& c : shared.perfCounters) {
            total += c.load();
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double rate = (double(total) - prev_total) / std::chrono::duration<double>(t1 - t0).count();
        std::cout << "[PERF] CPU (rank " << rank << "): " << rate/1e6 << " M hashes/s\n";
        t0 = t1;

        prev_total = total;

#ifdef HAVE_MPI
        if (use_mpi) {
            MPI_Test(&stop_request, &global_should_exit, MPI_STATUS_IGNORE);
            if (global_should_exit) {
                shared.should_exit.store(1);
            }
        }
#endif

        if (shared.best.score >= cfg.target && !stop_flag) {
            shared.should_exit.store(1);
            if (use_mpi) {
#ifdef HAVE_MPI
                for (int r = 0; r < size; r++) {
                    if (r != rank) {
                        int flag = 1;
                        MPI_Send(&flag, 1, MPI_INT, r, 1, MPI_COMM_WORLD);
                    }
                }
#endif
                stop_flag = 1;
            }
        }
    }

    for (auto& t : threads) t.join();

#ifdef HAVE_MPI
    if (use_mpi) {
        struct { uint32_t score; int rank; } local_best = {static_cast<uint32_t>(shared.best.score), rank};
        struct { uint32_t score; int rank; } global_best;
        MPI_Reduce(&local_best, &global_best, 1, MPI_2INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            if (global_best.rank != rank) {
                std::printf("[INFO] Best result from rank %d with score %d\n", global_best.rank, global_best.score);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
#endif

    if (shared.best.score > 0) {
        std::printf("[INFO] Rank %d Best salt: 0x%016llx%016llx, score: %d\n",
               rank, shared.best.salt_hi, shared.best.salt_lo, shared.best.score);
        std::array<uint8_t, 32> saltArr{};
        uint64_t salt_lo = shared.best.salt_lo;
        uint64_t salt_hi = shared.best.salt_hi;
        for (int i = 0; i < 8; ++i) {
            saltArr[31 - i] = static_cast<uint8_t>(salt_lo & 0xff);
            salt_lo >>= 8;
            saltArr[23 - i] = static_cast<uint8_t>(salt_hi & 0xff);
            salt_hi >>= 8;
        }
        if (cfg.create3) {
            auto addr = create3_address_cpu(cfg.deployer, saltArr.data());
            std::cout << "CREATE3 Address: 0x" << to_hex(addr) << std::endl;
        } else {
            auto addr = create2_address_cpu(cfg.deployer, saltArr.data(), cfg.initHash);
            std::cout << "Address: 0x" << to_hex(addr) << std::endl;
        }
    }
}

// ============================================================
// Multi-target CPU mining
// ============================================================

// CPU-side zero-byte counting (same formula as GPU)
static inline int count_zero_bytes_cpu(const uint8_t* addr, int len) {
    int count = 0;
    for (int i = 0; i < len; i++) {
        if (addr[i] == 0) count++;
    }
    return count;
}

// CPU-side multi-target check using mask/value pairs on raw lane words
static inline void check_targets_cpu(const uint8_t* addr,
                                     uint64_t salt_lo, uint64_t salt_hi,
                                     const LaunchCfg& cfg,
                                     salt_result* results,
                                     std::mutex& mtx) {
    // Extract words in same LE lane order as GPU
    uint32_t w0;
    uint64_t w1, w2;
    memcpy(&w0, addr, 4);
    memcpy(&w1, addr + 4, 8);
    memcpy(&w2, addr + 12, 8);

    for (uint32_t t = 0; t < cfg.num_targets; t++) {
        const TargetSpec& tgt = cfg.targets[t];

        bool prefix_ok = ((w0 & tgt.mask0) == tgt.val0)
                       & ((w1 & tgt.mask1) == tgt.val1)
                       & ((w2 & tgt.mask2) == tgt.val2);
        if (!prefix_ok) continue;

        uint32_t score;
        if (tgt.type == TGT_PREFIX) {
            score = tgt.prefix_nibbles;
        } else if (tgt.type == TGT_LEADING_ZEROS) {
            score = score_lz(*(const U160*)addr);
        } else { // TGT_PREFIX_PLUS_ZEROS
            score = count_zero_bytes_cpu(addr, 20);
        }

        if (score >= tgt.threshold && score > results[t].score) {
            std::lock_guard<std::mutex> lock(mtx);
            if (score > results[t].score) {
                results[t].score = score;
                results[t].salt_lo = salt_lo;
                results[t].salt_hi = salt_hi;
            }
        }
    }
}

struct MultiSharedData {
    std::atomic<int> should_exit{0};
    std::mutex mutex;
    salt_result results[MAX_TARGETS] = {};
    std::vector<std::atomic<uint64_t>> perfCounters;

    explicit MultiSharedData(size_t n_threads) : perfCounters(n_threads) {}
};

static inline void cpu_mine_multi(uint64_t base, int thread_id,
                                  const LaunchCfg& cfg,
                                  MultiSharedData& shared, uint32_t epoch) {
    uint64_t salt_hi = ((base + thread_id) << 52) | uint64_t(epoch);
    uint64_t salt_lo = 0;
    uint64_t localCount = 0;
    State s{};
    State result{};
    uint8_t* s8 = reinterpret_cast<uint8_t*>(&s);

    // Initialize CREATE3 template
    s8[0] = 0xff;
    std::memcpy(s8 + 1, cfg.deployer, 20);
    std::memcpy(s8 + 53, SOLADY_PROXY_INITCODE_HASH, 32);
    s8[85] = 0x01;
    s8[135] = 0x80;

    for (int i = 0; i < 8; ++i) {
        s8[44 - i] = (salt_hi >> (8 * i)) & 0xff;
    }

    while (!shared.should_exit.load()) {
        for (int i = 0; i < 8; ++i) {
            s8[52 - i] = (salt_lo >> (8 * i)) & 0xff;
        }

        // First keccak: proxy address
        keccak_f1600_cpu(s, result);

        // Second keccak: final address via RLP CREATE
        uint64_t r1 = result[1], r2 = result[2], r3 = result[3];
        State rlp{};
        rlp[0]  = 0x94d6ULL | ((r1 >> 32) << 16) | ((r2 & 0xFFFFULL) << 48);
        rlp[1]  = (r2 >> 16) | ((r3 & 0xFFFFULL) << 48);
        rlp[2]  = (r3 >> 16) | (0x0101ULL << 48);
        rlp[16] = 0x8000000000000000ULL;

        State final_res{};
        keccak_f1600_cpu(rlp, final_res);

        const uint8_t* addr = reinterpret_cast<const uint8_t*>(&final_res) + 12;
        check_targets_cpu(addr, salt_lo, salt_hi, cfg, shared.results, shared.mutex);

        localCount++;
        if (localCount == LOG_INTERVAL) {
            shared.perfCounters[thread_id].fetch_add(localCount);
            localCount = 0;
            if (shared.should_exit.load()) break;
        }
        salt_lo++;
    }
    if (localCount) {
        shared.perfCounters[thread_id].fetch_add(localCount);
    }
}

static volatile sig_atomic_t g_cpu_sigint = 0;
static void cpu_sigint_handler(int) { g_cpu_sigint = 1; }

void run_cpu_mining_multi(const LaunchCfg& cfg, uint32_t num_threads) {
    signal(SIGINT, cpu_sigint_handler);

    uint32_t epoch = (uint32_t)time(nullptr);
    std::cout << "[INFO] Epoch seed: " << epoch << "\n";
    std::cout << "=== MULTI-TARGET CREATE3 MINING (CPU) ===\n";

    MultiSharedData shared(num_threads);
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (uint32_t i = 0; i < num_threads; ++i) {
        threads.emplace_back(cpu_mine_multi, i, static_cast<int>(i),
                             std::cref(cfg), std::ref(shared), epoch);
    }

    std::ofstream results_log("results.txt", std::ios::app);
    if (!results_log.is_open()) {
        std::cerr << "[WARN] Could not open results.txt for writing\n";
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    uint64_t prev_total = 0;
    std::vector<uint32_t> lastScores(cfg.num_targets, 0);

    while (!shared.should_exit.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(2));

        if (g_cpu_sigint) {
            std::cout << "\n[INFO] SIGINT received, stopping...\n";
            shared.should_exit.store(1);
            break;
        }

        uint64_t total = 0;
        for (const auto& c : shared.perfCounters) total += c.load();

        auto t1 = std::chrono::high_resolution_clock::now();
        double rate = double(total - prev_total) / std::chrono::duration<double>(t1 - t0).count();
        std::printf("[PERF] CPU: %.1f M hashes/s\n", rate / 1e6);
        t0 = t1;
        prev_total = total;

        for (uint32_t t = 0; t < cfg.num_targets; t++) {
            if (shared.results[t].score > lastScores[t]) {
                const auto& tgt = cfg.targets[t];
                auto& r = shared.results[t];

                std::array<uint8_t, 32> saltArr{};
                uint64_t slo = r.salt_lo, shi = r.salt_hi;
                for (int j = 0; j < 8; ++j) {
                    saltArr[31 - j] = static_cast<uint8_t>(slo & 0xff); slo >>= 8;
                    saltArr[23 - j] = static_cast<uint8_t>(shi & 0xff); shi >>= 8;
                }
                auto addr = create3_address_cpu(cfg.deployer, saltArr.data());

                std::string line = "[TARGET " + std::to_string(t) + "] " + tgt.name
                                 + ": score=" + std::to_string(r.score)
                                 + " salt=0x" + to_hex(saltArr);
                std::string addr_str = "     Address: 0x" + to_hex(addr);

                std::cout << line << "\n" << addr_str << "\n";

                if (results_log.is_open()) {
                    results_log << line << "\n" << addr_str << "\n";
                    results_log.flush();
                }
                lastScores[t] = r.score;
            }
        }
    }

    shared.should_exit.store(1);
    for (auto& t : threads) t.join();

    // Final results with verification
    std::cout << "\n=== FINAL RESULTS ===\n";
    if (results_log.is_open()) results_log << "\n=== FINAL RESULTS ===\n";
    for (uint32_t t = 0; t < cfg.num_targets; t++) {
        auto& r = shared.results[t];
        const auto& tgt = cfg.targets[t];

        if (r.score == 0) {
            std::printf("[%d] %s: no result found\n", t, tgt.name);
            continue;
        }

        std::array<uint8_t, 32> saltArr{};
        uint64_t slo = r.salt_lo, shi = r.salt_hi;
        for (int i = 0; i < 8; ++i) {
            saltArr[31 - i] = static_cast<uint8_t>(slo & 0xff); slo >>= 8;
            saltArr[23 - i] = static_cast<uint8_t>(shi & 0xff); shi >>= 8;
        }

        auto addr = create3_address_cpu(cfg.deployer, saltArr.data());
        std::string line = "[" + std::to_string(t) + "] " + tgt.name
                         + ": score=" + std::to_string(r.score)
                         + " salt=0x" + to_hex(saltArr);
        std::string addr_str = "     Address: 0x" + to_hex(addr);

        std::cout << line << "\n" << addr_str << "\n";

        if (results_log.is_open()) {
            results_log << line << "\n" << addr_str << "\n";
        }
    }
    if (results_log.is_open()) results_log.flush();
}