cmake_minimum_required(VERSION 3.28)
project(sycl_probe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER dpcpp)

add_executable(xmx_probe probe.cpp)
target_compile_options(xmx_probe PRIVATE -fsycl -O2)

add_executable(kvarn_test kvarn_test.cpp)
target_compile_options(kvarn_test PRIVATE -fsycl -O2)
