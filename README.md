# RECREATE2

RECREATE2 is a high-performance CUDA/MPI “miner” for Ethereum smart contract deployment addresses using the CREATE2 opcode. It brute‑forces salts to find deployment addresses with many leading zeros, achieving billions of salts/sec per GPU.

## Features

- **GPU-accelerated**: First(?) open-source CREATE2 miner implemented in CUDA.
- **Distributed support**: Optional MPI for multi-node scaling.
- **CPU fallback**: Runs on multicore CPUs if CUDA is unavailable.
- **Configurable**: Threshold, scoring, threading, blocks, device choice.
- **Tested**: Built-in unit test for correctness against reference values.

## Prerequisites

- C++20 compiler
- CMake ≥ 3.24
- Optional:
  - CUDA toolkit (GPU support)
  - MPI library (distributed support)

## Build

```sh
make           # runs CMake in Release mode and builds targets
```

- **Targets**:
  - `create2_miner`: main executable
  - `tests`: unit tests

```sh
make test      # run unit tests
make clean     # remove build artifacts
```

## Usage

```sh
./build/create2_miner \
  --deployer    0x<20-byte hex> \   # contract deployer address
  --init-hash    0x<32-byte hex> \   # initcode hash
  --threshold    <int>    \          # minimum leading-zero score (default: 32)
  --score        <mode>   \          # scoring mode (default: 0)
  --threads      <n>      \          # CPU threads or CUDA threads per block
  --blocks       <n>      \          # CUDA blocks override
  --device       gpu|cpu  \          # device type (default: gpu)
  --mpi                   \          # enable MPI
  --test-salt   0x<32-byte hex>      # single-shot address test
```

### Example

```sh
./build/create2_miner \
  --deployer 0x48E516B34A1274f49457b9C6182097796D0498Cb \
  --init-hash 0x94d114296a5af85c1fd2dc039cdaa32f1ed4b0fe0868f02d888bfc91feb645d9 \
  --threshold 40 \
  --device gpu
```

## Performance

- Capable of **billions of salts/sec per GPU** for SHA-3 (Keccak)
- **88–105%** of performance compared to similar open-source miners

## License

Licensed under GNU AGPLv3.