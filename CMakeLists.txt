cmake_minimum_required(VERSION 3.15)
project(tensorstats LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release flags — compiler-specific
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
  # MSVC: /O2=optimize speed, /arch:AVX2=AVX2 SIMD, /fp:fast=fast-math
  add_compile_options(/O2 /arch:AVX2 /fp:fast)
else()
  # GCC/Clang
  add_compile_options(-O3 -march=native -ffast-math)
endif()

find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(
    tensorstats_core
    STABLE_ABI
    src/tensorstats/core.cpp
)

install(TARGETS tensorstats_core LIBRARY DESTINATION tensorstats)
