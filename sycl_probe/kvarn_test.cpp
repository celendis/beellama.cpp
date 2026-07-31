// KVarN SYCL correctness test
#include <CL/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cassert>

using namespace sycl;

static constexpr int KVAR_N_DIM = 128;
static constexpr int KVAR_N_TILE_VALUES = KVAR_N_DIM * KVAR_N_DIM;
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

// CPU quantize row
static void cpu_quantize_row(float * tile, uint8_t * record, int bits, int row) {
    float lo = 3.402823466e+38F;
    float hi = -3.402823466e+38F;
    for (int col = 0; col < KVAR_N_DIM; ++col) {
        float x = tile[row * KVAR_N_DIM + col];
        lo = fminf(lo, x);
        hi = fmaxf(hi, x);
    }
    int qmax = (1 << bits) - 1;
    float scale = fmaxf((hi - lo) / qmax, 1e-10f);
    int row_bytes = KVAR_N_DIM * bits / 8;
    uint8_t * row_payload = record + row * row_bytes;
    for (int i = 0; i < row_bytes; ++i) row_payload[i] = 0;
    for (int col = 0; col < KVAR_N_DIM; ++col) {
        float x = tile[row * KVAR_N_DIM + col];
        uint8_t q = (uint8_t)fminf(fmaxf(roundf((x - lo) / scale), 0.0f), (float)qmax);
        int bit_offset = col * bits;
        for (int bit = 0; bit < bits; ++bit) {
            int dst_bit = bit_offset + bit;
            row_payload[dst_bit / 8] |= ((q >> bit) & 1u) << (dst_bit % 8);
        }
    }
    int payload_bytes = KVAR_N_TILE_VALUES * bits / 8;
    float * meta = (float *)(record + payload_bytes);
    meta[row * 3 + 0] = scale;
    meta[row * 3 + 1] = lo;
    meta[row * 3 + 2] = 1.0f;
}

// CPU dequantize value
static float cpu_dequantize(const uint8_t * record, int bits, int row, int col) {
    int payload_bytes = KVAR_N_TILE_VALUES * bits / 8;
    size_t bit_offset = size_t(row * KVAR_N_DIM + col) * size_t(bits);
    uint8_t q = 0;
    for (int bit = 0; bit < bits; ++bit) {
        size_t src_bit = bit_offset + size_t(bit);
        q |= uint8_t(((record[src_bit / 8] >> (src_bit % 8)) & 1u) << bit);
    }
    const float * meta = (const float *)(record + payload_bytes);
    return (float(q) * meta[row * 3 + 0] + meta[row * 3 + 1]) * meta[row * 3 + 2];
}

int main() {
    queue q{default_selector{}};
    device d = q.get_device();
    printf("=== Device ===\nName:     %s\nVendor:   %s\n\n",
           d.get_info<info::device::name>().c_str(),
           d.get_info<info::device::vendor>().c_str());

    // Test 1: WHT round-trip
    printf("Test 1: WHT round-trip\n");
    {
        std::vector<float> input(KVAR_N_DIM);
        for (int i = 0; i < KVAR_N_DIM; ++i)
            input[i] = static_cast<float>(i * 13 % 97) / 100.0f;

        std::vector<float> cpu_result(input);
        cpu_wht_128(cpu_result.data());

        float * d_input = malloc_device<float>(KVAR_N_DIM, d, q.get_context());
        float * d_result = malloc_device<float>(KVAR_N_DIM, d, q.get_context());

        q.memcpy(d_input, input.data(), sizeof(float) * KVAR_N_DIM).wait();

        q.submit([&](handler& cgh) {
            local_accessor<float, 1> shared{range<1>(KVAR_N_DIM), cgh};
            nd_range<1> nd{KVAR_N_DIM, KVAR_N_DIM};
            cgh.parallel_for<>(nd, [=](nd_item<1> item) {
                int tid = item.get_local_id()[0];
                shared[tid] = d_input[tid];
                item.barrier();
                for (int h = 1; h < KVAR_N_DIM; h *= 2) {
                    if (tid < 64) {
                        int j = (tid / h) * (2 * h) + (tid % h);
                        float a = shared[j], b = shared[j + h];
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

        float max_err = 0;
        for (int i = 0; i < KVAR_N_DIM; ++i)
            max_err = fmaxf(max_err, std::abs(cpu_result[i] - gpu_result[i]));
        printf("  Max error: %.2e %s\n", max_err, max_err < 1e-4f ? "PASS" : "FAIL");

        free(d_input, q.get_context());
        free(d_result, q.get_context());
    }

    // Test 2: WHT self-inverse
    printf("Test 2: WHT self-inverse\n");
    {
        std::vector<float> data(KVAR_N_DIM);
        for (int i = 0; i < KVAR_N_DIM; ++i)
            data[i] = static_cast<float>(i * 13 % 97) / 100.0f;
        cpu_wht_128(data.data());
        cpu_wht_128(data.data());
        float rt_err = 0;
        for (int i = 0; i < KVAR_N_DIM; ++i) {
            float expected = static_cast<float>(i * 13 % 97) / 100.0f;
            rt_err = fmaxf(rt_err, std::abs(data[i] - expected));
        }
        printf("  Round-trip max error: %.2e %s\n", rt_err, rt_err < 1e-3f ? "PASS" : "FAIL");
    }

    // Test 3: Quantize/Dequantize round-trip
    printf("Test 3: Quantize/Dequantize round-trip\n");
    {
        int bits = 4;
        int payload_bytes = KVAR_N_TILE_VALUES * bits / 8;
        int meta_bytes = KVAR_N_DIM * 3 * sizeof(float);
        int record_bytes = payload_bytes + meta_bytes;

        // Generate random tile
        std::vector<float> tile(KVAR_N_TILE_VALUES);
        for (int i = 0; i < KVAR_N_TILE_VALUES; ++i)
            tile[i] = static_cast<float>(rand() % 1000 - 500) / 1000.0f;

        // CPU quantize (no WHT to match GPU)
        std::vector<uint8_t> cpu_record(record_bytes, 0);
        for (int row = 0; row < KVAR_N_DIM; ++row)
            cpu_quantize_row(tile.data(), cpu_record.data(), bits, row);

        // CPU dequantize
        std::vector<float> cpu_output(KVAR_N_TILE_VALUES);
        for (int row = 0; row < KVAR_N_DIM; ++row)
            for (int col = 0; col < KVAR_N_DIM; ++col)
                cpu_output[row * KVAR_N_DIM + col] = cpu_dequantize(cpu_record.data(), bits, row, col);

        // GPU quantize
        float * d_tile = malloc_device<float>(KVAR_N_TILE_VALUES, d, q.get_context());
        uint8_t * d_record = malloc_device<uint8_t>(record_bytes, d, q.get_context());
        q.memset(d_record, 0, record_bytes).wait();
        q.memcpy(d_tile, tile.data(), sizeof(float) * KVAR_N_TILE_VALUES).wait();

        q.submit([&](handler& cgh) {
            nd_range<1> nd{KVAR_N_DIM, KVAR_N_DIM};
            cgh.parallel_for<>(nd, [=](nd_item<1> item) {
                int row = item.get_local_id()[0];
                float lo = 3.402823466e+38F;
                float hi = -3.402823466e+38F;
                for (int col = 0; col < KVAR_N_DIM; ++col) {
                    float x = d_tile[row * KVAR_N_DIM + col];
                    lo = fminf(lo, x);
                    hi = fmaxf(hi, x);
                }
                int qmax = (1 << bits) - 1;
                float scale = fmaxf((hi - lo) / qmax, 1e-10f);
                int row_bytes = KVAR_N_DIM * bits / 8;
                int payload_bytes = KVAR_N_TILE_VALUES * bits / 8;
                uint8_t * row_payload = d_record + row * row_bytes;
                for (int i = 0; i < row_bytes; ++i) row_payload[i] = 0;
                for (int col = 0; col < KVAR_N_DIM; ++col) {
                    float x = d_tile[row * KVAR_N_DIM + col];
                    uint8_t q_val = (uint8_t)fminf(fmaxf(roundf((x - lo) / scale), 0.0f), (float)qmax);
                    int bit_offset = col * bits;
                    for (int bit = 0; bit < bits; ++bit) {
                        int dst_bit = bit_offset + bit;
                        row_payload[dst_bit / 8] |= ((q_val >> bit) & 1u) << (dst_bit % 8);
                    }
                }
                float * meta = (float *)(d_record + payload_bytes);
                meta[row * 3 + 0] = scale;
                meta[row * 3 + 1] = lo;
                meta[row * 3 + 2] = 1.0f;
            });
        }).wait();

        // GPU dequantize
        float * d_output = malloc_device<float>(KVAR_N_TILE_VALUES, d, q.get_context());
        q.submit([&](handler& cgh) {
            nd_range<1> nd{KVAR_N_TILE_VALUES, 1024};
            cgh.parallel_for<>(nd, [=](nd_item<1> item) {
                int idx = item.get_global_id()[0];
                int row = idx / KVAR_N_DIM;
                int col = idx % KVAR_N_DIM;
                int payload_bytes = KVAR_N_TILE_VALUES * bits / 8;
                size_t bit_offset = size_t(row * KVAR_N_DIM + col) * size_t(bits);
                uint8_t q_val = 0;
                for (int bit = 0; bit < bits; ++bit) {
                    size_t src_bit = bit_offset + size_t(bit);
                    q_val |= uint8_t(((d_record[src_bit / 8] >> (src_bit % 8)) & 1u) << bit);
                }
                const float * meta = (const float *)(d_record + payload_bytes);
                d_output[idx] = (float(q_val) * meta[row * 3 + 0] + meta[row * 3 + 1]) * meta[row * 3 + 2];
            });
        }).wait();

        std::vector<float> gpu_output(KVAR_N_TILE_VALUES);
        q.memcpy(gpu_output.data(), d_output, sizeof(float) * KVAR_N_TILE_VALUES).wait();

        // Copy GPU record to CPU and dequantize on CPU to isolate quantize vs dequantize
        std::vector<uint8_t> gpu_record(record_bytes);
        q.memcpy(gpu_record.data(), d_record, record_bytes).wait();
        std::vector<float> gpu_record_cpu_output(KVAR_N_TILE_VALUES);
        for (int row = 0; row < KVAR_N_DIM; ++row)
            for (int col = 0; col < KVAR_N_DIM; ++col)
                gpu_record_cpu_output[row * KVAR_N_DIM + col] = cpu_dequantize(gpu_record.data(), bits, row, col);

        // Compare raw records byte-by-byte
        int byte_diffs = 0;
        for (int i = 0; i < record_bytes; ++i)
            if (cpu_record[i] != gpu_record[i]) byte_diffs++;
        printf("  Record byte diffs: %d / %d\n", byte_diffs, record_bytes);

        // Compare CPU vs GPU record
        float record_err = 0;
        for (int i = 0; i < KVAR_N_TILE_VALUES; ++i)
            record_err = fmaxf(record_err, std::abs(cpu_output[i] - gpu_record_cpu_output[i]));
        // Threshold: up to 2 quantization levels (4-bit, range ~10, so ~0.125)
        printf("  CPU vs GPU record (quantize): %.2e %s\n", record_err, record_err < 1e-1f ? "PASS" : "FAIL");

        // Compare CPU vs GPU dequantize
        float dq_err = 0;
        for (int i = 0; i < KVAR_N_TILE_VALUES; ++i)
            dq_err = fmaxf(dq_err, std::abs(gpu_record_cpu_output[i] - gpu_output[i]));
        printf("  GPU record vs GPU dequantize: %.2e %s\n", dq_err, dq_err < 1e-3f ? "PASS" : "FAIL");

        // Overall CPU vs GPU
        float max_err = 0;
        for (int i = 0; i < KVAR_N_TILE_VALUES; ++i)
            max_err = fmaxf(max_err, std::abs(cpu_output[i] - gpu_output[i]));
        printf("  CPU vs GPU total: %.2e %s\n", max_err, max_err < 1e-1f ? "PASS" : "FAIL");

        // Quantization error (CPU vs original)
        float q_err = 0;
        for (int i = 0; i < KVAR_N_TILE_VALUES; ++i)
            q_err = fmaxf(q_err, std::abs(tile[i] - cpu_output[i]));
        printf("  Quantization error (expected): %.2e\n", q_err);

        free(d_tile, q.get_context());
        free(d_record, q.get_context());
        free(d_output, q.get_context());
    }

    printf("\nAll tests complete.\n");
    return 0;
}
