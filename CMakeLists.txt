cmake_minimum_required(VERSION 3.24)
project(create2_miner LANGUAGES CXX CUDA)

include(FetchContent)
FetchContent_Declare(
  argparse
  GIT_REPOSITORY https://github.com/p-ranav/argparse.git
  GIT_TAG        v2.9
)
FetchContent_MakeAvailable(argparse)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 86 89 90)

# core library
add_library(core STATIC
  src/miner.cu
  src/keccak_device.cu
  src/keccak_cpu.cpp
  src/score_examples.cu
)

# Compile keccak_cpu.cpp with NVCC so it sees __host__/__device__
set_source_files_properties(
  src/keccak_cpu.cpp
  PROPERTIES LANGUAGE CUDA
)

target_include_directories(core PUBLIC include)
target_compile_options(core
  PRIVATE
    -O3
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math --expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CXX>:-ffast-math>
)
# allow __device__ symbols (like load_template) to be resolved across files
set_target_properties(core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(create2_miner src/main.cpp)
target_link_libraries(create2_miner PRIVATE core argparse::argparse)
target_include_directories(create2_miner PRIVATE ${argparse_SOURCE_DIR}/include)


add_executable(tests test/test.cpp)
target_link_libraries(tests PRIVATE core argparse::argparse)
target_include_directories(tests PRIVATE ${argparse_SOURCE_DIR}/include)