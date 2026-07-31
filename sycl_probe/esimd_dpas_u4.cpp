// ESIMD DPAS U4 probe — can systolic engine do low-bit dequant+matmul?
// Build: icpx -fsycl -O2 -o esimd_dpas_u4 esimd_dpas_u4.cpp

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/intel/esimd/xmx/common.hpp>
#include <sycl/sycl.hpp>

#include <iostream>
#include <cmath>
#include <string>

using namespace sycl::ext::intel::esimd::xmx;

// Test S4 × S4 → F32
void run_s4_s4_test(sycl::queue &q) {
    constexpr int SD = 8;
    constexpr int RC = 1;
    constexpr int N = 16;
    // For S4: OpsPerChannel = 8
    constexpr int K = SD * 8; // 64

    // S4 packed 2 per byte
    constexpr int AN = RC * K * 4 / 8; // 32 bytes
    constexpr int BN = K * N * 4 / 8;  // 512 bytes
    constexpr int CN = RC * N;

    auto C = sycl::malloc_shared<float>(CN, q);
    q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(CN), [=](sycl::id<1> i) { C[i] = 0.0f; });
    });
    q.wait();

    try {
        constexpr size_t sub_group_size = 16;
        q.submit([&](sycl::handler &h) {
            h.parallel_for(sycl::nd_range<1>({RC * sub_group_size}, {sub_group_size}),
                [=](sycl::nd_item<1> item) {
                    int local_id = item.get_local_id()[0];
                    if (local_id == 0) {
                        __ESIMD_NS::simd<signed char, AN> a_simd(0x55);
                        __ESIMD_NS::simd<signed char, BN> b_simd(0x55);
                        __ESIMD_NS::simd<float, CN> c_simd(0.0f);

                        c_simd = dpas<SD, RC, float, float, signed char, signed char,
                                      dpas_argument_type::s4, dpas_argument_type::s4>(
                            c_simd, b_simd, a_simd);

                        for (int i = 0; i < CN; i++) C[i] = c_simd[i];
                    }
                    item.barrier();
                });
        });
        q.wait();

        float expected = 5.0f * 5.0f * K;
        bool ok = std::abs(C[0] - expected) < 1.0f;
        std::cout << "S4×S4: " << (ok ? "PASS" : "FAIL")
                  << " (expected=" << expected << ", got=" << C[0] << ")\n";
    } catch (const sycl::exception &e) {
        std::cout << "S4×S4: ERROR " << std::string(e.what()).substr(0, 100) << "\n";
    }

    sycl::free(C, q);
}

// Test U4 × U4 → F32 (pure low-bit matmul)
void run_u4_u4_test(sycl::queue &q) {
    constexpr int SD = 8;
    constexpr int RC = 1;
    constexpr int N = 16;
    // For U4: OpsPerChannel = min(8, 32/4) = 8
    constexpr int K = SD * 8; // 64

    // U4 packed 2 per byte: simd bytes = elements * 4 / 8
    constexpr int AN = RC * K * 4 / 8; // 32 bytes for 64 u4 values
    constexpr int BN = K * N * 4 / 8;  // 512 bytes for 1024 u4 values
    constexpr int CN = RC * N;

    auto C = sycl::malloc_shared<float>(CN, q);
    q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(CN), [=](sycl::id<1> i) { C[i] = 0.0f; });
    });
    q.wait();

    try {
        constexpr size_t sub_group_size = 16;
        q.submit([&](sycl::handler &h) {
            h.parallel_for(sycl::nd_range<1>({RC * sub_group_size}, {sub_group_size}),
                [=](sycl::nd_item<1> item) {
                    int local_id = item.get_local_id()[0];
                    if (local_id == 0) {
                        __ESIMD_NS::simd<unsigned char, AN> a_simd(0x55u);
                        __ESIMD_NS::simd<unsigned char, BN> b_simd(0x55u);
                        __ESIMD_NS::simd<float, CN> c_simd(0.0f);

                        c_simd = dpas<SD, RC, float, float, unsigned char, unsigned char,
                                      dpas_argument_type::u4, dpas_argument_type::u4>(
                            c_simd, b_simd, a_simd);

                        for (int i = 0; i < CN; i++) C[i] = c_simd[i];
                    }
                    item.barrier();
                });
        });
        q.wait();

        float expected = 5.0f * 5.0f * K;
        bool ok = std::abs(C[0] - expected) < 1.0f;
        std::cout << "U4×U4: " << (ok ? "PASS" : "FAIL")
                  << " (expected=" << expected << ", got=" << C[0] << ")\n";
    } catch (const sycl::exception &e) {
        std::cout << "U4×U4: ERROR " << std::string(e.what()).substr(0, 100) << "\n";
    }

    sycl::free(C, q);
}

int main() {
    try {
        sycl::queue q{sycl::default_selector_v};
        auto dev = q.get_device();

        std::cout << "Device: " << dev.get_info<sycl::info::device::name>() << "\n";
        std::cout << "Driver: " << dev.get_info<sycl::info::device::driver_version>() << "\n\n";

        std::cout << "Low-bit DPAS Tests (same-type only):\n";
        run_u4_u4_test(q);
        run_s4_s4_test(q);
        std::cout << "\nNote: Mixed precision (U4×F16) not supported by DPAS\n";

    } catch (const sycl::exception &e) {
        std::cerr << "SYCL error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
