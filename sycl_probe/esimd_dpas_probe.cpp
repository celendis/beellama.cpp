// ESIMD DPAS probe — tests xmx::dpas() on Arc Pro B70
// Build: icpx -fsycl -O2 -o esimd_dpas_probe esimd_dpas_probe.cpp

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/intel/esimd/xmx/common.hpp>
#include <sycl/sycl.hpp>

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <string>

using namespace sycl::ext::intel::esimd::xmx;

// DPAS constraints:
// - SystolicDepth must be 8
// - OpsPerChannel = min(8, 32/MaxElemBitSize) = 2 for F16
// - _K = SD * OpsPerChannel = 16 (for F16)
// - _M = RepeatCount (1-8)
// - _N = ExecutionSize (8 or 16)

template <int RC, int N>
void run_f16_test(sycl::queue &q, const std::string &label) {
    constexpr int SD = 8;
    constexpr int K = SD * 2; // 16 for F16
    constexpr int M = RC;
    constexpr int AN = M * K;
    constexpr int BN = K * N;
    constexpr int CN = M * N;

    auto C = sycl::malloc_shared<float>(CN, q);
    q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(CN), [=](sycl::id<1> i) { C[i] = 0.0f; });
    });
    q.wait();

    try {
        constexpr size_t sub_group_size = 16;
        // DPAS kernel — must be standalone
        q.submit([&](sycl::handler &h) {
            h.parallel_for(sycl::nd_range<1>({M * sub_group_size}, {sub_group_size}),
                [=](sycl::nd_item<1> item) {
                    int local_id = item.get_local_id()[0];
                    if (local_id == 0) {
                        __ESIMD_NS::simd<sycl::half, AN> a_simd(sycl::half(1.0f));
                        __ESIMD_NS::simd<sycl::half, BN> b_simd(sycl::half(1.0f));
                        __ESIMD_NS::simd<float, CN> c_simd(0.0f);

                        c_simd = dpas<SD, RC, float, float, sycl::half, sycl::half>(c_simd, b_simd, a_simd);

                        for (int i = 0; i < CN; i++) C[i] = c_simd[i];
                    }
                    item.barrier();
                });
        });
        q.wait();

        float expected = (float)K;
        bool ok = true;
        for (int i = 0; i < CN; i++) {
            if (std::abs(C[i] - expected) > 1.0f) { ok = false; break; }
        }
        std::cout << label << ": "
                  << (ok ? "PASS" : "FAIL")
                  << " (expected=" << expected << ", got=" << C[0] << ")\n";
    } catch (const sycl::exception &e) {
        std::cout << label << ": ERROR "
                  << std::string(e.what()).substr(0, 100) << "\n";
    }

    sycl::free(C, q);
}

int main() {
    try {
        sycl::queue q{sycl::default_selector_v};
        auto dev = q.get_device();

        std::cout << "Device: " << dev.get_info<sycl::info::device::name>() << "\n";
        std::cout << "Driver: " << dev.get_info<sycl::info::device::driver_version>() << "\n";

        auto aspects = dev.get_info<sycl::info::device::aspects>();
        bool has_matrix = false;
        for (const auto &a : aspects) {
            if (a == sycl::aspect::ext_intel_matrix) { has_matrix = true; break; }
        }
        std::cout << "ext_intel_matrix aspect: " << (has_matrix ? "YES" : "NO") << "\n\n";

        std::cout << "F16 DPAS Tests (SD=8, K=16):\n";
        run_f16_test<1, 16>(q, "RC=1 N=16");
        run_f16_test<2, 16>(q, "RC=2 N=16");
        run_f16_test<4, 16>(q, "RC=4 N=16");
        run_f16_test<8, 16>(q, "RC=8 N=16");
        run_f16_test<1, 8>(q, "RC=1 N=8");
        run_f16_test<2, 8>(q, "RC=2 N=8");

    } catch (const sycl::exception &e) {
        std::cerr << "SYCL error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
