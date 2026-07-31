// XMX probe — try shapes that might work on Battlemage
// Build: icpx -fsycl -O2 -fenable-matrix -o probe probe.cpp

#include <sycl/sycl.hpp>
#include <iostream>
#include <algorithm>
#include <vector>

#include <sycl/ext/oneapi/matrix/matrix.hpp>

using namespace sycl::ext::oneapi::experimental::matrix;

int main() {
    try {
        sycl::queue q{sycl::default_selector_v};
        auto dev = q.get_device();

        std::cout << "Device: " << dev.get_info<sycl::info::device::name>() << "\n";
        std::cout << "Driver: " << dev.get_info<sycl::info::device::driver_version>() << "\n\n";

        // Try shapes common on Xe2
        // Battlemage likely supports: 16x16, 16x32, 32x16, 16x64, 64x16, 32x32

        // 16x16x16
        {
            constexpr int M = 16, N = 16, K = 16;
            auto A = sycl::malloc_shared<sycl::half>(M * K, q);
            auto B = sycl::malloc_shared<sycl::half>(K * N, q);
            auto C = sycl::malloc_shared<float>(M * N, q);
            for (int i = 0; i < M * K; i++) A[i] = sycl::half(1.0f);
            for (int i = 0; i < K * N; i++) B[i] = sycl::half(1.0f);
            for (int i = 0; i < M * N; i++) C[i] = 0.0f;

            try {
                q.submit([&](sycl::handler& h) {
                    h.parallel_for<>(sycl::nd_range<1>(M * N, 16),
                        [=](sycl::nd_item<1> item) {
                        auto sg = item.get_sub_group();
                        joint_matrix<sycl::sub_group, sycl::half, use::a, M, K> matA;
                        joint_matrix<sycl::sub_group, sycl::half, use::b, K, N> matB;
                        joint_matrix<sycl::sub_group, float, use::accumulator, M, N> matC;
                        joint_matrix<sycl::sub_group, float, use::accumulator, M, N> matD;
                        joint_matrix_load(sg, matA, sycl::multi_ptr<const sycl::half, sycl::access::address_space::global_space>(A), K);
                        joint_matrix_load(sg, matB, sycl::multi_ptr<const sycl::half, sycl::access::address_space::global_space>(B), N);
                        joint_matrix_fill(sg, matC, 0.0f);
                        joint_matrix_mad(sg, matD, matA, matB, matC);
                        joint_matrix_store(sg, matD, sycl::multi_ptr<float, sycl::access::address_space::global_space>(C), N, layout::row_major);
                    });
                });
                q.wait();
                bool ok = true;
                for (int i = 0; i < M * N; i++) {
                    if (std::abs(C[i] - K) > 0.01f) { ok = false; break; }
                }
                std::cout << "16x16x16: " << (ok ? "PASS" : "FAIL") << "\n";
            } catch (const sycl::exception& e) {
                std::cout << "16x16x16: " << std::string(e.what()).substr(0, 100) << "\n";
            }
            sycl::free(A, q); sycl::free(B, q); sycl::free(C, q);
        }

        // 16x32x16
        {
            constexpr int M = 16, N = 32, K = 16;
            auto A = sycl::malloc_shared<sycl::half>(M * K, q);
            auto B = sycl::malloc_shared<sycl::half>(K * N, q);
            auto C = sycl::malloc_shared<float>(M * N, q);
            for (int i = 0; i < M * K; i++) A[i] = sycl::half(1.0f);
            for (int i = 0; i < K * N; i++) B[i] = sycl::half(1.0f);
            for (int i = 0; i < M * N; i++) C[i] = 0.0f;

            try {
                q.submit([&](sycl::handler& h) {
                    h.parallel_for<>(sycl::nd_range<1>(M * N, 16),
                        [=](sycl::nd_item<1> item) {
                        auto sg = item.get_sub_group();
                        joint_matrix<sycl::sub_group, sycl::half, use::a, M, K> matA;
                        joint_matrix<sycl::sub_group, sycl::half, use::b, K, N> matB;
                        joint_matrix<sycl::sub_group, float, use::accumulator, M, N> matC;
                        joint_matrix<sycl::sub_group, float, use::accumulator, M, N> matD;
                        joint_matrix_load(sg, matA, sycl::multi_ptr<const sycl::half, sycl::access::address_space::global_space>(A), K);
                        joint_matrix_load(sg, matB, sycl::multi_ptr<const sycl::half, sycl::access::address_space::global_space>(B), N);
                        joint_matrix_fill(sg, matC, 0.0f);
                        joint_matrix_mad(sg, matD, matA, matB, matC);
                        joint_matrix_store(sg, matD, sycl::multi_ptr<float, sycl::access::address_space::global_space>(C), N, layout::row_major);
                    });
                });
                q.wait();
                bool ok = true;
                for (int i = 0; i < M * N; i++) {
                    if (std::abs(C[i] - K) > 0.01f) { ok = false; break; }
                }
                std::cout << "16x32x16: " << (ok ? "PASS" : "FAIL") << "\n";
            } catch (const sycl::exception& e) {
                std::cout << "16x32x16: " << std::string(e.what()).substr(0, 100) << "\n";
            }
            sycl::free(A, q); sycl::free(B, q); sycl::free(C, q);
        }

        // 32x16x16
        {
            constexpr int M = 32, N = 16, K = 16;
            auto A = sycl::malloc_shared<sycl::half>(M * K, q);
            auto B = sycl::malloc_shared<sycl::half>(K * N, q);
            auto C = sycl::malloc_shared<float>(M * N, q);
            for (int i = 0; i < M * K; i++) A[i] = sycl::half(1.0f);
            for (int i = 0; i < K * N; i++) B[i] = sycl::half(1.0f);
            for (int i = 0; i < M * N; i++) C[i] = 0.0f;

            try {
                q.submit([&](sycl::handler& h) {
                    h.parallel_for<>(sycl::nd_range<1>(M * N, 16),
                        [=](sycl::nd_item<1> item) {
                        auto sg = item.get_sub_group();
                        joint_matrix<sycl::sub_group, sycl::half, use::a, M, K> matA;
                        joint_matrix<sycl::sub_group, sycl::half, use::b, K, N> matB;
                        joint_matrix<sycl::sub_group, float, use::accumulator, M, N> matC;
                        joint_matrix<sycl::sub_group, float, use::accumulator, M, N> matD;
                        joint_matrix_load(sg, matA, sycl::multi_ptr<const sycl::half, sycl::access::address_space::global_space>(A), K);
                        joint_matrix_load(sg, matB, sycl::multi_ptr<const sycl::half, sycl::access::address_space::global_space>(B), N);
                        joint_matrix_fill(sg, matC, 0.0f);
                        joint_matrix_mad(sg, matD, matA, matB, matC);
                        joint_matrix_store(sg, matD, sycl::multi_ptr<float, sycl::access::address_space::global_space>(C), N, layout::row_major);
                    });
                });
                q.wait();
                bool ok = true;
                for (int i = 0; i < M * N; i++) {
                    if (std::abs(C[i] - K) > 0.01f) { ok = false; break; }
                }
                std::cout << "32x16x16: " << (ok ? "PASS" : "FAIL") << "\n";
            } catch (const sycl::exception& e) {
                std::cout << "32x16x16: " << std::string(e.what()).substr(0, 100) << "\n";
            }
            sycl::free(A, q); sycl::free(B, q); sycl::free(C, q);
        }

        // 16x64x16
        {
            constexpr int M = 16, N = 64, K = 16;
            auto A = sycl::malloc_shared<sycl::half>(M * K, q);
            auto B = sycl::malloc_shared<sycl::half>(K * N, q);
            auto C = sycl::malloc_shared<float>(M * N, q);
            for (int i = 0; i < M * K; i++) A[i] = sycl::half(1.0f);
            for (int i = 0; i < K * N; i++) B[i] = sycl::half(1.0f);
            for (int i = 0; i < M * N; i++) C[i] = 0.0f;

            try {
                q.submit([&](sycl::handler& h) {
                    h.parallel_for<>(sycl::nd_range<1>(M * N, 16),
                        [=](sycl::nd_item<1> item) {
                        auto sg = item.get_sub_group();
                        joint_matrix<sycl::sub_group, sycl::half, use::a, M, K> matA;
                        joint_matrix<sycl::sub_group, sycl::half, use::b, K, N> matB;
                        joint_matrix<sycl::sub_group, float, use::accumulator, M, N> matC;
                        joint_matrix<sycl::sub_group, float, use::accumulator, M, N> matD;
                        joint_matrix_load(sg, matA, sycl::multi_ptr<const sycl::half, sycl::access::address_space::global_space>(A), K);
                        joint_matrix_load(sg, matB, sycl::multi_ptr<const sycl::half, sycl::access::address_space::global_space>(B), N);
                        joint_matrix_fill(sg, matC, 0.0f);
                        joint_matrix_mad(sg, matD, matA, matB, matC);
                        joint_matrix_store(sg, matD, sycl::multi_ptr<float, sycl::access::address_space::global_space>(C), N, layout::row_major);
                    });
                });
                q.wait();
                bool ok = true;
                for (int i = 0; i < M * N; i++) {
                    if (std::abs(C[i] - K) > 0.01f) { ok = false; break; }
                }
                std::cout << "16x64x16: " << (ok ? "PASS" : "FAIL") << "\n";
            } catch (const sycl::exception& e) {
                std::cout << "16x64x16: " << std::string(e.what()).substr(0, 100) << "\n";
            }
            sycl::free(A, q); sycl::free(B, q); sycl::free(C, q);
        }

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
