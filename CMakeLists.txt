cmake_minimum_required(VERSION 3.15)
project(tensorstats LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release flags
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-O3 -march=native -ffast-math)

find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(
    tensorstats_core
    STABLE_ABI
    src/tensorstats/core.cpp
)

install(TARGETS tensorstats_core LIBRARY DESTINATION tensorstats)
