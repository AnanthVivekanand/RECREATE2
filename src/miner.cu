#include "miner.hpp"
#include "keccak.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <array>
#include "util.hpp"
#include <cstdio>
#include <thread>
#include <curand_kernel.h>
#include <vector>

#define LOG_INTERVAL 5000

// first 136 bytes of the Keccak rate block
__constant__ uint64_t template85[17];

// extract bytes 12–31 from the 200-byte sponge output
__device__ 
int32_t score_lz(const State &s) {
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&s) + 12;

    uint32_t w0 = (uint32_t(ptr[0]) << 24)
                | (uint32_t(ptr[1]) << 16)
                | (uint32_t(ptr[2]) <<  8)
                | (uint32_t(ptr[3]));
    if (w0 != 0) {
        return __builtin_clz(w0);
    }
    int32_t lz = 32;

    uint64_t w1 = (uint64_t(ptr[4]) << 56)
                | (uint64_t(ptr[5]) << 48)
                | (uint64_t(ptr[6]) << 40)
                | (uint64_t(ptr[7]) << 32)
                | (uint64_t(ptr[8]) << 24)
                | (uint64_t(ptr[9]) << 16)
                | (uint64_t(ptr[10])<<  8)
                | (uint64_t(ptr[11]));
    if (w1 != 0) {
        return lz + __builtin_clzll(w1);
    }
    lz += 64;

    uint64_t w2 = (uint64_t(ptr[12]) << 56)
                | (uint64_t(ptr[13]) << 48)
                | (uint64_t(ptr[14]) << 40)
                | (uint64_t(ptr[15]) << 32)
                | (uint64_t(ptr[16]) << 24)
                | (uint64_t(ptr[17]) << 16)
                | (uint64_t(ptr[18]) <<  8)
                | (uint64_t(ptr[19]));
    if (w2 != 0) {
        return lz + __builtin_clzll(w2);
    }

    return lz + 64;
}

__global__ void mine(uint64_t start,
                     uint64_t step,
                     uint64_t target,
                     int scoreMode,
                     salt_result* out,
                     uint64_t* perfCounters,
                     uint32_t deviceIdx,
                     volatile int* device_should_exit)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t salt_hi = ((start + gid) << 32) | (uint64_t(deviceIdx) << 52);
    uint64_t salt_lo = 0;
    uint64_t localCount = 0;

    if (gid % 1000 == 0) {
        printf("[DBG] thread %d, start=%llu, step=%llu, target=%llu\n",
               gid, start, step, target);
    }

    while (true) {
        if (*device_should_exit != 0) {
            break;
        }

        State base = load_template();
        uint8_t* s8 = reinterpret_cast<uint8_t*>(&base);

        for (int i = 0; i < 8; i++) {
            s8[52 - i] = (salt_lo >> (8 * i)) & 0xff;
        }
        for (int i = 0; i < 8; i++) {
            s8[44 - i] = (salt_hi >> (8 * i)) & 0xff;
        }

        keccak_f1600_unrolled(base, base);
        int32_t sc = score_lz(base);

        localCount++;
        if (localCount == LOG_INTERVAL) {
            atomicAdd((unsigned long long*)&perfCounters[blockIdx.x],
                      (unsigned long long)localCount);
            localCount = 0;
            if (*device_should_exit != 0) {
                break;
            }
        }

        if (sc >= int32_t(target)) {
            printf("[DBG] thread %d, salt_lo=%016llx, salt_hi=%016llx, score=%d\n",
                   gid, salt_lo, salt_hi, sc);
            if (*device_should_exit == 0) {
                out->score = sc;
                out->salt_lo = salt_lo;
                out->salt_hi = salt_hi;
                *device_should_exit = 1;
                break;
            }
        }
        salt_lo += 1;
    }

    if (localCount) {
        atomicAdd((unsigned long long*)&perfCounters[blockIdx.x],
                  (unsigned long long)localCount);
    }
}

void run_kernel(const LaunchCfg& cfg,
                uint32_t blocks,
                uint32_t threads,
                bool use_mpi,
                int rank,
                int size)
{
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);

    if (num_gpus < 1) {
        std::cerr << "[ERR] No CUDA-capable devices found.\n";
        return;
    }
    std::cout << "[INFO] Detected " << num_gpus << " GPU(s).\n";

    // prepare template
    uint64_t hostTpl[17] = {};
    uint8_t* ptr = reinterpret_cast<uint8_t*>(hostTpl);
    ptr[0] = 0xff;
    cudaMemcpy(ptr + 1,    cfg.deployer, 20, cudaMemcpyHostToHost);
    cudaMemcpy(ptr + 53,   cfg.initHash, 32, cudaMemcpyHostToHost);
    ptr[85]  = 0x01;
    ptr[135] = 0x80;

    int should_exit = 0;

    struct GPUContext {
        cudaStream_t stream;
        cudaStream_t copyStream;
        salt_result* d_best;
        uint64_t* d_perfCounters;
        uint64_t* h_perfCounters;
        int* d_should_exit;
    };
    std::vector<GPUContext> contexts(num_gpus);

    for (uint32_t i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);

        cudaError_t err = cudaMemcpyToSymbol(template85, hostTpl, 136);
        if (err != cudaSuccess) {
            std::cerr << "[ERR] cudaMemcpyToSymbol failed: " << cudaGetErrorString(err) << "\n";
            exit(1);
        }

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        if (!threads) threads = 256;
        if (!blocks) {
            int maxPerSM;
            err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &maxPerSM, (void*)mine, threads, 0);
            if (err != cudaSuccess) {
                std::cerr << "[ERR] Occupancy calculation failed for GPU " << i << ": "
                        << cudaGetErrorString(err) << "\n";
                return;
            }
            blocks = maxPerSM * prop.multiProcessorCount;
            printf("[INFO] GPU %d: maxPerSM=%d, multiProcessorCount=%d\n",
                   i, maxPerSM, prop.multiProcessorCount);
        }

        cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 10*1024*1024);
        
        GPUContext& ctx = contexts[i];

        cudaStreamCreate(&ctx.stream);

        cudaMemcpyToSymbol(template85, hostTpl, 136);

        cudaMalloc(&ctx.d_best, sizeof(salt_result));
        cudaMemset(ctx.d_best, 0, sizeof(salt_result));

        cudaMalloc(&ctx.d_perfCounters, blocks * sizeof(uint64_t));
        cudaMemset(ctx.d_perfCounters, 0, blocks * sizeof(uint64_t));

        err = cudaHostAlloc(&ctx.h_perfCounters, blocks * sizeof(uint64_t),
                      cudaHostAllocPortable);
        if (err != cudaSuccess) {
            std::cerr << "[ERR] cudaHostAlloc failed for GPU " << i
                    << ": " << cudaGetErrorString(err) << "\n";
            return;
        }
        memset(ctx.h_perfCounters, 0, blocks * sizeof(uint64_t));

        cudaMalloc(&ctx.d_should_exit, sizeof(int));
        cudaMemset(ctx.d_should_exit, 0, sizeof(int));

        cudaStreamCreateWithFlags(&ctx.copyStream, cudaStreamNonBlocking);

        std::cout << "[INFO] GPU " << i << ": Launching mine<<<" << blocks
                  << "," << threads << ">>> (" << uint64_t(blocks)*threads
                  << " threads)\n";

        mine<<<blocks, threads, 0, ctx.stream>>>(
            cfg.start,
            cfg.step,
            cfg.target,
            cfg.scoreMode,
            ctx.d_best,
            ctx.d_perfCounters,
            i,  // deviceIdx
            ctx.d_should_exit
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "[ERR] GPU " << i << " kernel launch failed: "
                      << cudaGetErrorString(err) << "\n";
        }
    }

    // Polling loop
    std::vector<uint64_t> lastTotals(num_gpus, 0);
    auto t0 = std::chrono::high_resolution_clock::now();
    bool done = false;
    
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

    while (!done) {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        for (int i = 0; i < num_gpus; i++) {
            cudaSetDevice(i);
            GPUContext& ctx = contexts[i];
            salt_result result;
            cudaMemcpyAsync(ctx.h_perfCounters, ctx.d_perfCounters,
                            blocks * sizeof(uint64_t),
                            cudaMemcpyDeviceToHost, ctx.copyStream);
            cudaMemcpyAsync(&result, ctx.d_best, sizeof(salt_result),
                            cudaMemcpyDeviceToHost, ctx.copyStream);
            cudaStreamSynchronize(ctx.copyStream);

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "[ERR] Runtime error: " << cudaGetErrorString(err) << "\n";
                break;
            }

            uint64_t total = 0;
            for (uint32_t b = 0; b < blocks; b++)
                total += ctx.h_perfCounters[b];

            auto t1 = std::chrono::high_resolution_clock::now();
            double rate = double(total - lastTotals[i]) /
                          std::chrono::duration<double>(t1 - t0).count();
            std::cout << "[PERF] GPU " << i << ": " << rate/1e6 << " M hashes/s\n";
            lastTotals[i] = total;

            uint32_t score = result.score;
            if (score >= cfg.target) {
                should_exit = 1;
                if (use_mpi && !stop_flag) {
#ifdef HAVE_MPI
                    for (int r = 0; r < size; r++) {
                        if (r != rank) {
                            MPI_Send(&should_exit, 1, MPI_INT, r, 1, MPI_COMM_WORLD);
                        }
                    }
#endif
                    stop_flag = 1;
                }
            }
        }

#ifdef HAVE_MPI
        if (use_mpi) {
            MPI_Test(&stop_request, &global_should_exit, MPI_STATUS_IGNORE);
            if (global_should_exit) {
                should_exit = 1;
            }
        }
#endif

        t0 = std::chrono::high_resolution_clock::now();
        if (should_exit) done = true;
    }

    printf("[INFO] Stopping all GPUs...\n");

    // Set should_exit flag on all GPUs
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        GPUContext& ctx = contexts[i];
        cudaMemcpyAsync(ctx.d_should_exit, &should_exit,
                        sizeof(int), cudaMemcpyHostToDevice,
                        ctx.copyStream);
        cudaStreamSynchronize(ctx.copyStream);
        printf("[INFO] GPU %d: should_exit set to %d\n", i, should_exit);
    }

    // Retrieve best result across all GPUs
    salt_result best_result{};
    int best_gpu = 0;
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        GPUContext& ctx = contexts[i];
        cudaStreamSynchronize(ctx.stream);
        salt_result result;
        cudaMemcpy(&result, ctx.d_best, sizeof(salt_result),
                   cudaMemcpyDeviceToHost);
        if (result.score > best_result.score) {
            best_result = result;
            best_gpu = i;
        }
    }

    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        GPUContext& ctx = contexts[i];
        cudaStreamDestroy(ctx.stream);
        cudaFree(ctx.d_best);
        cudaFree(ctx.d_perfCounters);
        cudaFreeHost(ctx.h_perfCounters);
        cudaFree(ctx.d_should_exit);
    }

    uint32_t bestScore = best_result.score;
    uint64_t bestSaltLo  = best_result.salt_lo;
    uint64_t bestSaltHi  = best_result.salt_hi;
    printf("[INFO] Best salt: 0x%016llx%016llx, score: %d\n",
           bestSaltHi, bestSaltLo, bestScore);

#ifdef HAVE_MPI
    if (use_mpi) {
        struct { uint32_t score; int rank; } local_best = {bestScore, rank};
        struct { uint32_t score; int rank; } global_best;
        MPI_Reduce(&local_best, &global_best, 1, MPI_2INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            if (global_best.rank != rank) {
                printf("[INFO] Best result from rank %d with score %d\n", global_best.rank, global_best.score);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
#endif

    std::array<uint8_t,32> saltArr{};
    for (int i = 0; i < 8; ++i) {
        saltArr[31 - i] = static_cast<uint8_t>(bestSaltLo & 0xff);
        bestSaltLo >>= 8;
        saltArr[23 - i] = static_cast<uint8_t>(bestSaltHi & 0xff);
        bestSaltHi >>= 8;
    }
    std::cout << "\n";
    auto addr = create2_address_gpu(
        cfg.deployer, saltArr.data(), cfg.initHash
    );
    std::cout << "Address: 0x" << to_hex(addr) << std::endl;
}