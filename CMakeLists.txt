cmake_minimum_required(VERSION 3.24)
project(create2_miner LANGUAGES CXX)

find_package(MPI)
if (MPI_FOUND)
    add_definitions(-DHAVE_MPI)
    include_directories(${MPI_INCLUDE_PATH})
else()
    message(WARNING "MPI not found. MPI support will be disabled.")
endif()

find_package(CUDA QUIET)
if (CUDA_FOUND)
    enable_language(CUDA)
    add_definitions(-DHAVE_CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_ARCHITECTURES 70)
else()
    message(WARNING "CUDA not found. GPU support will be disabled.")
endif()

include(FetchContent)
FetchContent_Declare(
  argparse
  GIT_REPOSITORY https://github.com/p-ranav/argparse.git
  GIT_TAG        v2.9
)
FetchContent_MakeAvailable(argparse)

set(CMAKE_CXX_STANDARD 20)

add_library(core STATIC
  src/miner_cpu.cpp
  src/keccak_cpu.cpp
)

if (CUDA_FOUND)
    target_sources(core PRIVATE
        src/miner.cu
        src/keccak_device.cu
    )
    set_target_properties(core PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_compile_options(core PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math --expt-relaxed-constexpr>
    )
endif()

target_compile_options(core PRIVATE
    -O3
    $<$<COMPILE_LANGUAGE:CXX>:-ffast-math>
)
target_include_directories(core PUBLIC include)

# Main executable
add_executable(create2_miner src/main.cpp)
target_link_libraries(create2_miner PRIVATE core argparse::argparse)
if (MPI_FOUND)
    target_link_libraries(create2_miner PRIVATE ${MPI_LIBRARIES})
endif()
target_include_directories(create2_miner PRIVATE ${argparse_SOURCE_DIR}/include)

# Tests executable
add_executable(tests test/test.cpp)
target_link_libraries(tests PRIVATE core argparse::argparse)
if (MPI_FOUND)
    target_link_libraries(tests PRIVATE ${MPI_LIBRARIES})
endif()
target_include_directories(tests PRIVATE ${argparse_SOURCE_DIR}/include)