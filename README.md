# RECREATE2

RECREATE2 is a high-performance CUDA/MPI miner for Ethereum smart contract deployment addresses using CREATE2 and CREATE3 opcodes. It brute-forces salts to find deployment addresses matching a target prefix or leading-zero count, achieving billions of hashes/sec per GPU.

## Features

- **CREATE2 and CREATE3**: Mine vanity addresses for both deployment methods. CREATE3 uses the Solady proxy initcode hash.
- **Multi-target mining**: Mine for multiple address patterns simultaneously from a single JSON config. One Keccak computation per salt checks all targets with <1% overhead.
- **Prefix mining**: Search for any hex prefix (e.g. `--prefix 0xC0FFEE`), not just leading zeros.
- **Branchless target checking**: Word-level mask+compare for prefix matching, Hacker's Delight zero-byte detection with `__popcll`, and `__clz` intrinsics for leading-zero scoring — all operating directly on raw Keccak lane values.
- **GPU-accelerated**: CUDA kernels with ~1.8B hashes/sec (CREATE2) and ~800M hashes/sec (CREATE3) on an RTX A6000.
- **Distributed support**: Optional MPI for multi-node scaling.
- **CPU fallback**: Runs on multicore CPUs if CUDA is unavailable.
- **Epoch namespacing**: Each run uses a timestamp seed so concurrent or repeated runs explore different salt space.
- **Tested**: Built-in unit tests verify CPU/GPU agreement for both CREATE2 and CREATE3.

## Prerequisites

- C++20 compiler
- CMake >= 3.22
- Optional:
  - CUDA toolkit (GPU support)
  - MPI library (distributed support)

## Build

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

If `nvcc` is not on your PATH:
```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

Targets:
- `create2_miner`: main executable
- `tests`: unit tests

## Usage

### Single-target mode

```sh
./create2_miner \
  --deployer    0x<20-byte hex>   # deployer/factory address
  --init-hash   0x<32-byte hex>   # initcode hash (CREATE2 only)
  --create3                       # CREATE3 mode (uses Solady proxy hash)
  --prefix      0x<hex>           # mine for this address prefix
  --threshold   <int>             # minimum leading-zero score (default: 32)
  --threads     <n>               # CPU threads or CUDA threads per block
  --blocks      <n>               # CUDA blocks override
  --device      gpu|cpu           # device type (default: gpu)
  --mpi                           # enable MPI
  --benchmark                     # run 10-second benchmark
  --test-salt   0x<32-byte hex>   # single-shot address test
```

### Multi-target mode

Mine for multiple address patterns simultaneously using a JSON config file:

```sh
./create2_miner --config targets.json --device gpu
```

All targets must share the same CREATE3 factory. The kernel computes one address per salt and checks it against all targets with branchless word-level comparisons.

**Config format** (`targets.json`):

```json
{
    "factory": "0x<20-byte factory address>",
    "targets": [
        {
            "name": "my-token",
            "type": "prefix",
            "prefix": "0xC0FFEE"
        },
        {
            "name": "my-exchange",
            "type": "prefix_plus_zeros",
            "prefix": "0xBEEF",
            "threshold": 3
        },
        {
            "name": "my-factory",
            "type": "leading_zeros",
            "threshold": 16
        }
    ]
}
```

**Target types:**

| Type | Description | Score |
|------|-------------|-------|
| `prefix` | Exact hex prefix match | Number of matched nibbles (first match wins) |
| `leading_zeros` | Maximize leading zero bytes | 8 per zero byte + 4 per zero nibble |
| `prefix_plus_zeros` | Prefix must match, then maximize zero bytes anywhere | Count of `0x00` bytes in full 20-byte address |

- `threshold`: minimum score to record a result. Defaults to `prefix_nibbles` for `prefix` type, `1` otherwise.
- Mining runs until Ctrl+C. Results print as they improve.

### Examples

Mine a CREATE2 address with leading zeros:
```sh
./create2_miner \
  --deployer 0x48E516B34A1274f49457b9C6182097796D0498Cb \
  --init-hash 0x94d114296a5af85c1fd2dc039cdaa32f1ed4b0fe0868f02d888bfc91feb645d9 \
  --threshold 40 \
  --device gpu
```

Mine a CREATE3 address with a custom prefix:
```sh
./create2_miner \
  --deployer 0xE6171dF4BcF566a91D8E309A5FcF9Fb102C09eC2 \
  --create3 \
  --prefix 0xC0FFEE \
  --device gpu
```

Run benchmarks:
```sh
./create2_miner --benchmark --device gpu              # CREATE2
./create2_miner --benchmark --create3 --device gpu    # CREATE3
```

## Performance

RTX A6000 (Ampere, sm_86):

| Mode | Throughput |
|------|-----------|
| CREATE2 | ~1,830 M hashes/s |
| CREATE3 | ~800 M hashes/s |
| CREATE3 multi-target (5 targets) | ~800 M hashes/s |

CREATE3 is roughly half the speed of CREATE2 due to two Keccak-f[1600] permutations per salt. Multi-target adds <1% overhead since the target checks (~40 instructions) are negligible compared to the two Keccak permutations (~9,600 instructions).

## License

Licensed under GNU AGPLv3.
