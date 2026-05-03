cmake_minimum_required(VERSION 3.16)
project(transformer_kernels VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-mavx512f -mavx512bw -mavx512vl" COMPILER_SUPPORTS_AVX512)

find_package(pybind11 REQUIRED)

add_library(tkernels STATIC
    src/softmax.cpp src/rmsnorm.cpp src/rope.cpp
)
target_include_directories(tkernels PUBLIC include)
target_compile_options(tkernels PRIVATE -O3 -ffast-math)

if(COMPILER_SUPPORTS_AVX2)
    target_compile_options(tkernels PRIVATE -mavx2 -mfma)
    target_compile_definitions(tkernels PUBLIC HAVE_AVX2)
    message(STATUS "AVX2 enabled")
endif()
if(COMPILER_SUPPORTS_AVX512)
    target_compile_definitions(tkernels PUBLIC HAVE_AVX512)
    message(STATUS "AVX-512 enabled")
endif()

pybind11_add_module(transformer_kernels_ext src/bindings.cpp)
target_link_libraries(transformer_kernels_ext PRIVATE tkernels)
target_include_directories(transformer_kernels_ext PRIVATE include)
target_compile_options(transformer_kernels_ext PRIVATE -O3 -ffast-math)
if(COMPILER_SUPPORTS_AVX2)
    target_compile_options(transformer_kernels_ext PRIVATE -mavx2 -mfma)
endif()
set_target_properties(transformer_kernels_ext PROPERTIES
    OUTPUT_NAME "transformer_kernels_ext"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/transformer_kernels"
)

option(BUILD_BENCH "Build standalone C++ benchmark" OFF)
if(BUILD_BENCH)
    add_executable(bench_cpp benchmarks/bench_main.cpp)
    target_link_libraries(bench_cpp PRIVATE tkernels)
    target_compile_options(bench_cpp PRIVATE -O3 -ffast-math -mavx2 -mfma)
endif()
