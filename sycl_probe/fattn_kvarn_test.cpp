// KVarN flash attention test — standalone SYCL kernel test
// Build: icpx -fsycl -O2 -o fattn_kvarn_test fattn_kvarn_test.cpp

#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstdint>
#include <float.h>

using namespace sycl;

static constexpr int KVAR_N_DIM = 128;
static constexpr float WHT_SCALE = 0.08838834764834f; // 1/sqrt(128)

// Unpack a single value from a KVarN record
template <int Bits>
static inline uint8_t kvarn_unpack(const uint8_t * record, int index) {
    if constexpr (Bits == 4) {
        return (record[index >> 1] >> ((index & 1) << 2)) & 0x0fu;
    } else if constexpr (Bits == 2) {
        return (record[index >> 2] >> ((index & 3) << 1)) & 0x03u;
    } else {
        return record[index];
    }
}

// Dequantize a single row from a KVarN record to F32
template <int Bits>
static inline void kvarn_dequant_row(
        const uint8_t * record, int row, float * output) {
    int row_bytes = KVAR_N_DIM * Bits / 8;
    int payload_bytes = KVAR_N_DIM * KVAR_N_DIM * Bits / 8;

    const float * meta = reinterpret_cast<const float *>(record + payload_bytes);
    float scale = meta[row * 3 + 0];
    float zero = meta[row * 3 + 1];
    float mult = meta[row * 3 + 2];

    const uint8_t * row_payload = record + row * row_bytes;
    for (int col = 0; col < KVAR_N_DIM; ++col) {
        uint8_t q = kvarn_unpack<Bits>(row_payload, col);
        output[col] = (float(q) * scale + zero) * mult;
    }
}

// CPU reference: flash attention with KVarN dequant (matches GPU thread-per-dim pattern)
void cpu_flash_attn_kvarn(
        const float * q_data,
        const uint8_t * k_records,
        const uint8_t * v_records,
        float * dst,
        int n_kv, int n_query, int n_query_heads, int n_kv_heads,
        int record_bytes, float scale, int bits) {

    for (int q = 0; q < n_query; ++q) {
        for (int qh = 0; qh < n_query_heads; ++qh) {
            int kh = qh / (n_query_heads / n_kv_heads);

            for (int d = 0; d < KVAR_N_DIM; ++d) {
                float q_val = q_data[q * n_query_heads * KVAR_N_DIM + qh * KVAR_N_DIM + d];
                float max_logit = -FLT_MAX;
                float sum_exp = 0.0f;
                float acc = 0.0f;

                for (int kv = 0; kv < n_kv; ++kv) {
                    const uint8_t * k_rec = k_records + kv * n_kv_heads * record_bytes + kh * record_bytes;
                    const uint8_t * v_rec = v_records + kv * n_kv_heads * record_bytes + kh * record_bytes;

                    // Dequant single value (row=d, col=d) matching GPU's tid=d
                    int row_bytes = KVAR_N_DIM * bits / 8;
                    int payload_bytes = KVAR_N_DIM * KVAR_N_DIM * bits / 8;
                    const float * k_meta = reinterpret_cast<const float *>(k_rec + payload_bytes);
                    const float * v_meta = reinterpret_cast<const float *>(v_rec + payload_bytes);

                    float k_scale = k_meta[d * 3 + 0];
                    float k_zero = k_meta[d * 3 + 1];
                    float k_mult = k_meta[d * 3 + 2];
                    float v_scale = v_meta[d * 3 + 0];
                    float v_zero = v_meta[d * 3 + 1];
                    float v_mult = v_meta[d * 3 + 2];

                    uint8_t k_q, v_q;
                    if (bits == 4) {
                        k_q = (uint8_t)((k_rec + d * row_bytes)[d >> 1] >> ((d & 1) << 2)) & 0x0fu;
                        v_q = (uint8_t)((v_rec + d * row_bytes)[d >> 1] >> ((d & 1) << 2)) & 0x0fu;
                    } else {
                        k_q = (uint8_t)((k_rec + d * row_bytes)[d >> 2] >> ((d & 3) << 1)) & 0x03u;
                        v_q = (uint8_t)((v_rec + d * row_bytes)[d >> 2] >> ((d & 3) << 1)) & 0x03u;
                    }
                    float k_d = (float(k_q) * k_scale + k_zero) * k_mult;
                    float v_d = (float(v_q) * v_scale + v_zero) * v_mult;

                    float logit = q_val * k_d * scale;

                    if (logit > max_logit) {
                        acc *= expf(max_logit - logit);
                        sum_exp *= expf(max_logit - logit);
                        max_logit = logit;
                    }

                    float exp_val = expf(logit - max_logit);
                    sum_exp += exp_val;
                    acc += exp_val * v_d;
                }

                if (sum_exp > 0.0f) acc /= sum_exp;
                dst[q * n_query_heads * KVAR_N_DIM + qh * KVAR_N_DIM + d] = acc;
            }
        }
    }
}

int main() {
    queue q{default_selector{}};
    device d = q.get_device();
    printf("Device: %s\n\n", d.get_info<info::device::name>().c_str());

    int n_query = 1;
    int n_kv = 8;
    int n_query_heads = 4;
    int n_kv_heads = 2;
    int bits = 4;

    int payload_bytes = KVAR_N_DIM * KVAR_N_DIM * bits / 8;
    int meta_bytes = KVAR_N_DIM * 3 * sizeof(float);
    int record_bytes = payload_bytes + meta_bytes;
    float scale = 1.0f / std::sqrt((float) KVAR_N_DIM);

    // Generate test data
    srand(12345);
    std::vector<float> q_cpu(KVAR_N_DIM * n_query * n_query_heads);
    for (int i = 0; i < (int)q_cpu.size(); ++i)
        q_cpu[i] = (rand() % 1000 - 500) / 1000.0f;

    std::vector<uint8_t> k_records_cpu(n_kv * n_kv_heads * record_bytes, 0);
    std::vector<uint8_t> v_records_cpu(n_kv * n_kv_heads * record_bytes, 0);

    // Fill records with known values
    for (int kv = 0; kv < n_kv; ++kv) {
        for (int kh = 0; kh < n_kv_heads; ++kh) {
            uint8_t * k_rec = k_records_cpu.data() + kv * n_kv_heads * record_bytes + kh * record_bytes;
            uint8_t * v_rec = v_records_cpu.data() + kv * n_kv_heads * record_bytes + kh * record_bytes;

            // Set payload to known pattern
            for (int i = 0; i < payload_bytes; ++i) {
                k_rec[i] = (uint8_t)((kv * 7 + kh * 13 + i) & 0xff);
                v_rec[i] = (uint8_t)((kv * 11 + kh * 17 + i) & 0xff);
            }
            // Set metadata
            float * k_meta = reinterpret_cast<float *>(k_rec + payload_bytes);
            float * v_meta = reinterpret_cast<float *>(v_rec + payload_bytes);
            for (int r = 0; r < KVAR_N_DIM; ++r) {
                k_meta[r * 3 + 0] = 0.01f; // scale
                k_meta[r * 3 + 1] = 0.0f;  // zero
                k_meta[r * 3 + 2] = 1.0f;  // mult
                v_meta[r * 3 + 0] = 0.01f;
                v_meta[r * 3 + 1] = 0.0f;
                v_meta[r * 3 + 2] = 1.0f;
            }
        }
    }

    // CPU reference
    std::vector<float> dst_cpu(KVAR_N_DIM * n_query * n_query_heads);
    cpu_flash_attn_kvarn(q_cpu.data(), k_records_cpu.data(), v_records_cpu.data(),
                         dst_cpu.data(), n_kv, n_query, n_query_heads, n_kv_heads,
                         record_bytes, scale, bits);

    // GPU kernel
    auto q_dev = malloc_shared<float>(q_cpu.size(), q);
    auto k_dev = malloc_shared<uint8_t>(k_records_cpu.size(), q);
    auto v_dev = malloc_shared<uint8_t>(v_records_cpu.size(), q);
    auto dst_dev = malloc_shared<float>(dst_cpu.size(), q);

    q.memcpy(q_dev, q_cpu.data(), q_cpu.size() * sizeof(float)).wait();
    q.memcpy(k_dev, k_records_cpu.data(), k_records_cpu.size()).wait();
    q.memcpy(v_dev, v_records_cpu.data(), v_records_cpu.size()).wait();

    constexpr int THREADS = KVAR_N_DIM;
    range<3> global{size_t(n_query), size_t(n_query_heads), size_t(THREADS)};
    range<3> local{1, 1, THREADS};

    q.submit([&](handler & h) {
        h.parallel_for(nd_range<3>{global, local},
            [=](nd_item<3> item) {
                int query = item.get_global_id(0);
                int qhead = item.get_global_id(1);
                int tid = item.get_local_id(2);

                int gqa = n_query_heads / n_kv_heads;
                int kv_head = qhead / gqa;

                float q_val = q_dev[query * n_query_heads * KVAR_N_DIM + qhead * KVAR_N_DIM + tid];

                float max_logit = -FLT_MAX;
                float sum_exp = 0.0f;
                float acc = 0.0f;

                for (int kv = 0; kv < n_kv; ++kv) {
                    const uint8_t * k_rec = k_dev + kv * n_kv_heads * record_bytes + kv_head * record_bytes;
                    const uint8_t * v_rec = v_dev + kv * n_kv_heads * record_bytes + kv_head * record_bytes;

                    // Inline dequant for K
                    int row_bytes = KVAR_N_DIM * 4 / 8;
                    int payload_bytes = KVAR_N_DIM * KVAR_N_DIM * 4 / 8;
                    const float * k_meta = reinterpret_cast<const float *>(k_rec + payload_bytes);
                    float k_scale = k_meta[tid * 3 + 0];
                    float k_zero = k_meta[tid * 3 + 1];
                    float k_mult = k_meta[tid * 3 + 2];
                    const uint8_t * k_row_payload = k_rec + tid * row_bytes;
                    uint8_t k_q = (k_row_payload[tid >> 1] >> ((tid & 1) << 2)) & 0x0fu;
                    float k_d = (float(k_q) * k_scale + k_zero) * k_mult;

                    float logit = q_val * k_d * scale;

                    if (logit > max_logit) {
                        acc *= expf(max_logit - logit);
                        sum_exp *= expf(max_logit - logit);
                        max_logit = logit;
                    }

                    float exp_val = expf(logit - max_logit);
                    sum_exp += exp_val;

                    // Inline dequant for V
                    const float * v_meta = reinterpret_cast<const float *>(v_rec + payload_bytes);
                    float v_scale = v_meta[tid * 3 + 0];
                    float v_zero = v_meta[tid * 3 + 1];
                    float v_mult = v_meta[tid * 3 + 2];
                    const uint8_t * v_row_payload = v_rec + tid * row_bytes;
                    uint8_t v_q = (v_row_payload[tid >> 1] >> ((tid & 1) << 2)) & 0x0fu;
                    float v_d = (float(v_q) * v_scale + v_zero) * v_mult;

                    acc += exp_val * v_d;
                }

                if (sum_exp > 0.0f) acc /= sum_exp;
                dst_dev[query * n_query_heads * KVAR_N_DIM + qhead * KVAR_N_DIM + tid] = acc;
            });
    });
    q.wait();

    // Compare
    std::vector<float> dst_gpu(dst_cpu.size());
    q.memcpy(dst_gpu.data(), dst_dev, dst_gpu.size() * sizeof(float)).wait();

    float max_err = 0;
    int max_idx = 0;
    for (int i = 0; i < (int)dst_cpu.size(); ++i) {
        float err = std::abs(dst_cpu[i] - dst_gpu[i]);
        if (err > max_err) { max_err = err; max_idx = i; }
    }

    printf("KVarN Flash Attention Test (q4, D=128, Q=%d, KV=%d, heads=%d/%d)\n",
           n_query, n_kv, n_query_heads, n_kv_heads);
    printf("CPU[0:4]: %.6f %.6f %.6f %.6f\n",
           dst_cpu[0], dst_cpu[1], dst_cpu[2], dst_cpu[3]);
    printf("GPU[0:4]: %.6f %.6f %.6f %.6f\n",
           dst_gpu[0], dst_gpu[1], dst_gpu[2], dst_gpu[3]);
    printf("Max error: %.2e at idx %d\n", max_err, max_idx);
    printf("%s\n", max_err < 1e-3f ? "PASS" : "FAIL");

    free(q_dev, q);
    free(k_dev, q);
    free(v_dev, q);
    free(dst_dev, q);

    return max_err < 1e-3f ? 0 : 1;
}
