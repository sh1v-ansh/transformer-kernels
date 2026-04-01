cmake_minimum_required(VERSION 3.16)
project(transformer_kernels VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 REQUIRED)

add_library(tkernels STATIC
    src/softmax.cpp
    src/rmsnorm.cpp
    src/rope.cpp
)
target_include_directories(tkernels PUBLIC include)
target_compile_options(tkernels PRIVATE -O2)

pybind11_add_module(transformer_kernels_ext src/bindings.cpp)
target_link_libraries(transformer_kernels_ext PRIVATE tkernels)
target_include_directories(transformer_kernels_ext PRIVATE include)
