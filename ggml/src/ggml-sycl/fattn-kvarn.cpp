// KVarN flash attention — SYCL portable fallback
// Reads KVarN records directly, dequantizes per-element, accumulates Q×K and V
//
// This is the scalar F16/F32 baseline. Phase 2b will add XMX DPAS acceleration.
//
// Algorithm per query token:
//   for each KV token:
//     dequant K from record → K_f16[128]
//     partial = dot(Q[128], K_f16[128])
//     softmax(partial * scale)
//     dequant V from record → V_f16[128]
//     output += softmax_weight * V_f16[128]
//   inverse WHT on output (rotate back to original domain)

#include "fattn-kvarn.hpp"
#include "common.hpp"
#include "kvarn-wht.hpp"

#include <sycl/sycl.hpp>
#include <cmath>
#include <float.h>
#include <algorithm>

using namespace sycl;

// KVarN dimensions
static constexpr int KVAR_N_DIM = 128;
static constexpr float WHT_SCALE = 0.08838834764834f; // 1/sqrt(128)

// KVarN op param index (matches CUDA)
static constexpr int KVAR_N_OP_PARAM_BITS = 0;

// Unpack a single value from a KVarN record
template <int Bits>
static __dpct_inline__ uint8_t kvarn_unpack(const uint8_t * record, int index) {
    if constexpr (Bits == 8) {
        return record[index];
    } else if constexpr (Bits == 4) {
        return (record[index >> 1] >> ((index & 1) << 2)) & 0x0fu;
    } else if constexpr (Bits == 2) {
        return (record[index >> 2] >> ((index & 3) << 1)) & 0x03u;
    } else {
        int bit_offset = index * Bits;
        int byte_offset = bit_offset >> 3;
        int bit_in_byte = bit_offset & 7;
        uint16_t packed = record[byte_offset] | (uint16_t(record[byte_offset + 1]) << 8);
        return (packed >> bit_in_byte) & ((1u << Bits) - 1u);
    }
}

// Dequantize a single row from a KVarN record to F32
template <int Bits>
static __dpct_inline__ void kvarn_dequant_row(
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

// KVarN flash attention kernel
// Each work-group processes one (query_token, query_head) pair
template <int BitsK, int BitsV>
static void kvarn_flash_attn_kernel(
        queue_ptr & q,
        const float * q_data,
        const uint8_t * k_records,
        const uint8_t * v_records,
        float * dst,
        int n_kv,
        int n_query,
        int n_query_heads,
        int n_kv_heads,
        int record_bytes,
        float scale) {

    constexpr int THREADS = KVAR_N_DIM;

    range<3> global{n_query, n_query_heads, THREADS};
    range<3> local{1, 1, THREADS};

    q->submit([&](handler & h) {
        h.parallel_for(nd_range<3>{global, local},
            [=](nd_item<3> item) {
                int query = item.get_global_id(0);
                int qhead = item.get_global_id(1);
                int tid = item.get_local_id(2);

                int gqa = n_query_heads / n_kv_heads;
                int kv_head = qhead / gqa;

                float q_val = q_data[query * n_query_heads * KVAR_N_DIM +
                                      qhead * KVAR_N_DIM + tid];

                float max_logit = -FLT_MAX;
                float sum_exp = 0.0f;
                float acc = 0.0f;

                float k_val[KVAR_N_DIM];
                float v_val[KVAR_N_DIM];

                for (int kv = 0; kv < n_kv; ++kv) {
                    const uint8_t * k_rec = k_records + kv * n_kv_heads * record_bytes +
                                            kv_head * record_bytes;
                    kvarn_dequant_row<BitsK>(k_rec, tid, k_val);

                    float logit = q_val * k_val[tid] * scale;

                    if (logit > max_logit) {
                        acc *= expf(max_logit - logit);
                        sum_exp *= expf(max_logit - logit);
                        max_logit = logit;
                    }

                    float exp_val = expf(logit - max_logit);
                    sum_exp += exp_val;

                    const uint8_t * v_rec = v_records + kv * n_kv_heads * record_bytes +
                                            kv_head * record_bytes;
                    kvarn_dequant_row<BitsV>(v_rec, tid, v_val);
                    acc += exp_val * v_val[tid];
                }

                if (sum_exp > 0.0f) {
                    acc /= sum_exp;
                }
                dst[query * n_query_heads * KVAR_N_DIM + qhead * KVAR_N_DIM + tid] = acc;
            });
    });
    q->wait();
}

// Main entry point
void ggml_sycl_flash_attn_ext_kvarn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

    queue_ptr q = ctx.stream();

    int D = (int) Q->ne[0];
    int n_query = (int) Q->ne[1];
    int n_query_heads = (int) Q->ne[2];
    int n_kv = (int) K->ne[1];
    int n_kv_heads = (int) K->ne[2];

    if (D != KVAR_N_DIM) {
        GGML_LOG_WARN("KVarN flash attention only supports D=128, got D=%d\n", D);
        return;
    }

    // Get KVarN metadata from tensor op params
    int bits_k = ggml_get_op_params_i32(K, KVAR_N_OP_PARAM_BITS);
    int bits_v = ggml_get_op_params_i32(V, KVAR_N_OP_PARAM_BITS);
    if (bits_v == 0) bits_v = bits_k;

    int payload_bytes = KVAR_N_DIM * KVAR_N_DIM * bits_k / 8;
    int meta_bytes = KVAR_N_DIM * 3 * sizeof(float);
    int record_bytes = payload_bytes + meta_bytes;

    float scale = 1.0f / std::sqrt((float) D);

    if (bits_k == 4 && bits_v == 4) {
        kvarn_flash_attn_kernel<4, 4>(q,
            reinterpret_cast<const float *>(Q->data),
            reinterpret_cast<const uint8_t *>(K->data),
            reinterpret_cast<const uint8_t *>(V->data),
            reinterpret_cast<float *>(dst->data),
            n_kv, n_query, n_query_heads, n_kv_heads,
            record_bytes, scale);
    } else if (bits_k == 2 && bits_v == 2) {
        kvarn_flash_attn_kernel<2, 2>(q,
            reinterpret_cast<const float *>(Q->data),
            reinterpret_cast<const uint8_t *>(K->data),
            reinterpret_cast<const uint8_t *>(V->data),
            reinterpret_cast<float *>(dst->data),
            n_kv, n_query, n_query_heads, n_kv_heads,
            record_bytes, scale);
    } else {
        GGML_LOG_WARN("KVarN flash attention: unsupported bit width k=%d v=%d\n", bits_k, bits_v);
        return;
    }
}
