cmake_minimum_required(VERSION 3.28)
project(sycl_probe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(sycl REQUIRED)

add_executable(xmx_probe probe.cpp)
target_link_libraries(xmx_probe PRIVATE sycl::sycl)
target_compile_options(xmx_probe PRIVATE -fsycl -O2)
