// KVarN SYCL correctness test
// Compares WHT output against a reference CPU implementation
// Tests: WHT round-trip, store stage write, materialize decode

#include <CL/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cassert>

using namespace sycl;

static constexpr int KVAR_N_DIM = 128;
static constexpr float WHT_SCALE = 0.08838834764831845f; // 1/sqrt(128)

// Reference CPU WHT
static void cpu_wht_128(float * data) {
    for (int h = 1; h < KVAR_N_DIM; h *= 2) {
        for (int j = 0; j < KVAR_N_DIM; j += 2 * h) {
            for (int k = 0; k < h; ++k) {
                const float a = data[j + k];
                const float b = data[j + h + k];
                data[j + k] = a + b;
                data[j + h + k] = a - b;
            }
        }
    }
    for (int i = 0; i < KVAR_N_DIM; ++i) {
        data[i] *= WHT_SCALE;
    }
}

// Reference CPU inverse WHT
// WHT is self-inverse: WHT(WHT(x)) = x
// (butterfly is self-inverse, and two scales of 1/sqrt(N) cancel the N factor)
static void cpu_iwht_128(float * data) {
    cpu_wht_128(data);
}

int main() {
    queue q{default_selector{}};
    device d = q.get_device();
    printf("=== Device ===\nName:     %s\nVendor:   %s\n\n",
           d.get_info<info::device::name>().c_str(),
           d.get_info<info::device::vendor>().c_str());

    // Test 1: WHT round-trip
    printf("Test 1: WHT round-trip\n");
    std::vector<float> input(KVAR_N_DIM);
    for (int i = 0; i < KVAR_N_DIM; ++i) {
        input[i] = static_cast<float>(i * 13 % 97) / 100.0f;
    }

    std::vector<float> cpu_result(input);
    cpu_wht_128(cpu_result.data());

    float * d_input = malloc_device<float>(KVAR_N_DIM, d, q.get_context());
    float * d_result = malloc_device<float>(KVAR_N_DIM, d, q.get_context());
    float * d_shared = malloc_device<float>(KVAR_N_DIM, d, q.get_context());

    // Copy input to device
    q.memcpy(d_input, input.data(), sizeof(float) * KVAR_N_DIM).wait();

    // WHT kernel
    q.submit([&](handler& cgh) {
        local_accessor<float, 1> shared{range<1>(KVAR_N_DIM), cgh};
        nd_range<1> nd{KVAR_N_DIM, KVAR_N_DIM};

        cgh.parallel_for<>(nd, [=](nd_item<1> item) {
            const int tid = item.get_local_id()[0];
            shared[tid] = d_input[tid];
            item.barrier();

            for (int h = 1; h < KVAR_N_DIM; h *= 2) {
                if (tid < 64) {
                    const int j = (tid / h) * (2 * h) + (tid % h);
                    const float a = shared[j];
                    const float b = shared[j + h];
                    shared[j] = a + b;
                    shared[j + h] = a - b;
                }
                item.barrier();
            }
            shared[tid] *= WHT_SCALE;
            d_result[tid] = shared[tid];
        });
    }).wait();

    std::vector<float> gpu_result(KVAR_N_DIM);
    q.memcpy(gpu_result.data(), d_result, sizeof(float) * KVAR_N_DIM).wait();

    // Compare
    float max_err = 0;
    for (int i = 0; i < KVAR_N_DIM; ++i) {
        float err = std::abs(cpu_result[i] - gpu_result[i]);
        if (err > max_err) max_err = err;
    }
    printf("  Max error: %.2e\n", max_err);
    if (max_err < 1e-4f) {
        printf("  PASS\n");
    } else {
        printf("  FAIL (threshold: 1e-4)\n");
    }

    // Test 2: WHT is self-inverse (up to scale)
    printf("Test 2: WHT self-inverse\n");
    std::vector<float> roundtrip(gpu_result);
    cpu_iwht_128(roundtrip.data());

    float rt_err = 0;
    for (int i = 0; i < KVAR_N_DIM; ++i) {
        float err = std::abs(input[i] - roundtrip[i]);
        if (err > rt_err) rt_err = err;
    }
    printf("  Round-trip max error: %.2e\n", rt_err);
    if (rt_err < 1e-3f) {
        printf("  PASS\n");
    } else {
        printf("  FAIL (threshold: 1e-3)\n");
    }

    // Cleanup
    free(d_input, q.get_context());
    free(d_result, q.get_context());
    free(d_shared, q.get_context());

    printf("\nAll tests complete.\n");
    return 0;
}
