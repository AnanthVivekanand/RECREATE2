# RECREATE2

RECREATE2 is a high-performance CUDA/MPI miner for Ethereum smart contract deployment addresses using CREATE2 and CREATE3 opcodes. It brute-forces salts to find deployment addresses matching a target prefix or leading-zero count, achieving billions of hashes/sec per GPU.

## Features

- **CREATE2 and CREATE3**: Mine vanity addresses for both deployment methods. CREATE3 uses the Solady proxy initcode hash.
- **Prefix mining**: Search for any hex prefix (e.g. `--prefix 0xC0FFEE`), not just leading zeros.
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
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

If `nvcc` is not on your PATH:
```sh
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

Targets:
- `create2_miner`: main executable
- `tests`: unit tests

## Usage

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

CREATE3 is roughly half the speed of CREATE2 due to two Keccak-f[1600] permutations per salt.

## License

Licensed under GNU AGPLv3.
