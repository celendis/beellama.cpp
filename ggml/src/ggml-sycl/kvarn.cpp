#include "kvarn.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

// KVarN constants (match llama-kvarn.h)
static constexpr int KVAR_N_DIM = 128;
static constexpr int KVAR_N_TILE_VALUES = KVAR_N_DIM * KVAR_N_DIM;

// Shared memory layout for headwide variant (F32 tile + metadata)
static constexpr int KVAR_N_REDUCE_FLOATS = 4 * 4;
static constexpr int KVAR_N_SHARED_FLOATS = KVAR_N_TILE_VALUES + 8 * KVAR_N_DIM + 2 + KVAR_N_REDUCE_FLOATS;
static constexpr int KVAR_N_SHARED_BYTES = KVAR_N_SHARED_FLOATS * sizeof(float);

// Op param indices (match ggml-cuda/kvarn.cu)
static constexpr int KVAR_N_OP_PARAM_BITS = 0;
static constexpr int KVAR_N_OP_PARAM_ITERS = 1;
static constexpr int KVAR_N_OP_PARAM_STORE_VALUE = 2;
static constexpr int KVAR_N_OP_PARAM_STORE_SWA = 4;
static constexpr int KVAR_N_OP_PARAM_HEAD_SLICES = 5;
static constexpr int KVAR_N_OP_PARAM_STAGE_GROUPS = 7;
static constexpr int KVAR_N_OP_PARAM_TAIL_GROUPS = 8;
static constexpr int KVAR_N_OP_PARAM_EAGER_RECORDS = 9;

static int kvarn_resolve_stage_groups(const ggml_tensor * dst) {
    return ggml_get_op_params_i32(dst, KVAR_N_OP_PARAM_STAGE_GROUPS);
}

static int kvarn_resolve_tail_groups(const ggml_tensor * dst, int stage_groups) {
    const int tail_groups = ggml_get_op_params_i32(dst, KVAR_N_OP_PARAM_TAIL_GROUPS);
    return tail_groups > 0 ? tail_groups : stage_groups - 1;
}

// Shared memory offsets (match CUDA layout)
static constexpr int OFF_TILE = 0;
static constexpr int OFF_LOG_S_COL = KVAR_N_TILE_VALUES;
static constexpr int OFF_LOG_S_ROW = OFF_LOG_S_COL + KVAR_N_DIM;
static constexpr int OFF_S_COL = OFF_LOG_S_ROW + KVAR_N_DIM;
static constexpr int OFF_S_ROW = OFF_S_COL + KVAR_N_DIM;
static constexpr int OFF_BEST_COL = OFF_S_ROW + KVAR_N_DIM;
static constexpr int OFF_BEST_ROW = OFF_BEST_COL + KVAR_N_DIM;
static constexpr int OFF_COL_STD = OFF_BEST_ROW + KVAR_N_DIM;
static constexpr int OFF_ROW_STD = OFF_COL_STD + KVAR_N_DIM;
static constexpr int OFF_BEST_IMBALANCE = OFF_ROW_STD + KVAR_N_DIM;
static constexpr int OFF_BETTER = OFF_BEST_IMBALANCE + 1;
static constexpr int OFF_REDUCE = OFF_BETTER + 1;

// Compute std dev for a single column (thread = column)
static inline __attribute__((always_inline)) float kvarn_std_col(const float * tile, const float * s_col, const float * s_row, int col) {
    float sum = 0.0f;
    float sum_sq = 0.0f;
    const float sc = s_col[col];
    for (int row = 0; row < KVAR_N_DIM; ++row) {
        const float value = tile[row * KVAR_N_DIM + col] / (sc * s_row[row]);
        sum += value;
        sum_sq += value * value;
    }
    const float mean = sum / KVAR_N_DIM;
    return sqrtf(fmaxf((sum_sq - KVAR_N_DIM * mean * mean) / (KVAR_N_DIM - 1), 0.0f));
}

// Compute std dev for a single row (thread = row)
static inline __attribute__((always_inline)) float kvarn_std_row(const float * tile, const float * s_col, const float * s_row, int row) {
    float sum = 0.0f;
    float sum_sq = 0.0f;
    const float sr = s_row[row];
    for (int col = 0; col < KVAR_N_DIM; ++col) {
        const float value = tile[row * KVAR_N_DIM + col] / (s_col[col] * sr);
        sum += value;
        sum_sq += value * value;
    }
    const float mean = sum / KVAR_N_DIM;
    return sqrtf(fmaxf((sum_sq - KVAR_N_DIM * mean * mean) / (KVAR_N_DIM - 1), 0.0f));
}

// Warp-level min reduction (4 warps of 32 lanes each in 128-thread block)
static inline __attribute__((always_inline)) float kvarn_warp_min(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fminf(value, value); // SYCL doesn't have shfl_down; use shared mem instead
    }
    return value;
}

// Warp-level max reduction
static inline __attribute__((always_inline)) float kvarn_warp_max(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, value);
    }
    return value;
}

// Reduce std ranges across warps using shared memory
static void kvarn_reduce_std_ranges(const float * col_std, const float * row_std, float * reduce, int tid, sycl::nd_item<1> item) {
    const int lane = tid & 31;
    const int warp = tid >> 5;

    // Intra-warp reduction via shared memory (no shuffle in oneAPI 2026.1)
    float col_min = col_std[tid];
    float col_max = col_std[tid];
    float row_min = row_std[tid];
    float row_max = row_std[tid];

    // Write to shared, reduce within warp
    const int base = OFF_REDUCE + warp * 4;
    reduce[base + 0] = col_min;
    reduce[base + 1] = col_max;
    reduce[base + 2] = row_min;
    reduce[base + 3] = row_max;
    item.barrier();

    // Cross-warp reduction (only 4 threads needed)
    if (tid < 4) {
        const int metric = tid;
        float value = reduce[OFF_REDUCE + metric];
        for (int w = 1; w < 4; ++w) {
            const float next = reduce[OFF_REDUCE + w * 4 + metric];
            value = (metric == 0 || metric == 2) ? fminf(value, next) : fmaxf(value, next);
        }
        reduce[OFF_REDUCE + metric] = value;
    }
    item.barrier();
}

// Update best scales if current imbalance is better
static void kvarn_update_best_from_std(
        const float * candidate_col, const float * candidate_row,
        bool candidate_is_log,
        float * best_col, float * best_row,
        float * col_std, float * row_std,
        float * best_imbalance, float * better, float * reduce,
        int tid, sycl::nd_item<1> item) {

    kvarn_reduce_std_ranges(col_std, row_std, reduce, tid, item);

    if (tid == 0) {
        const float col_min = reduce[OFF_REDUCE + 0];
        const float col_max = reduce[OFF_REDUCE + 1];
        const float row_min = reduce[OFF_REDUCE + 2];
        const float row_max = reduce[OFF_REDUCE + 3];
        const float imbalance =
            col_max / fmaxf(col_min, 1e-8f) +
            row_max / fmaxf(row_min, 1e-8f);
        *better = imbalance <= *best_imbalance ? 1.0f : 0.0f;
        if (*better != 0.0f) {
            *best_imbalance = imbalance;
        }
    }
    item.barrier();

    if (*better != 0.0f) {
        best_col[tid] = candidate_is_log ? expf(candidate_col[tid]) : candidate_col[tid];
        best_row[tid] = candidate_is_log ? expf(candidate_row[tid]) : candidate_row[tid];
    }
    item.barrier();
}

// Quantize a 128×128 tile from shared memory into a record
static void kvarn_quantize_tile(float * tile, uint8_t * record, int bits, int iterations,
                                 float * shared, int tid, sycl::nd_item<1> item) {
    float * log_s_col = shared + OFF_LOG_S_COL;
    float * log_s_row = shared + OFF_LOG_S_ROW;
    float * s_col = shared + OFF_S_COL;
    float * s_row = shared + OFF_S_ROW;
    float * best_col = shared + OFF_BEST_COL;
    float * best_row = shared + OFF_BEST_ROW;
    float * col_std = shared + OFF_COL_STD;
    float * row_std = shared + OFF_ROW_STD;
    float * best_imbalance = shared + OFF_BEST_IMBALANCE;
    float * better = shared + OFF_BETTER;
    float * reduce = shared + OFF_REDUCE;

    // Initialize scales
    log_s_col[tid] = 0.0f;
    log_s_row[tid] = 0.0f;
    s_col[tid] = 1.0f;
    s_row[tid] = 1.0f;
    best_col[tid] = 1.0f;
    best_row[tid] = 1.0f;
    item.barrier();

    // Initial std computation
    col_std[tid] = kvarn_std_col(tile, s_col, s_row, tid);
    row_std[tid] = kvarn_std_row(tile, s_col, s_row, tid);
    item.barrier();

    if (tid == 0) {
        *best_imbalance = 3.402823466e+38F;
    }
    item.barrier();

    kvarn_update_best_from_std(s_col, s_row, false, best_col, best_row, col_std, row_std, best_imbalance, better, reduce, tid, item);

    // Iterative Sinkhorn-like optimization
    for (int iter = 0; iter < iterations; ++iter) {
        const float col = fminf(fmaxf(col_std[tid], 1e-3f), 1e3f);
        log_s_col[tid] = fminf(fmaxf(log_s_col[tid] + logf(col), -0.3f), 10.0f);
        s_col[tid] = expf(log_s_col[tid]);
        item.barrier();

        row_std[tid] = kvarn_std_row(tile, s_col, s_row, tid);
        item.barrier();

        const float row = fminf(fmaxf(row_std[tid], 1e-3f), 1e3f);
        log_s_row[tid] = fminf(fmaxf(log_s_row[tid] + logf(row), -0.3f), 10.0f);
        s_row[tid] = expf(log_s_row[tid]);
        item.barrier();

        row_std[tid] = kvarn_std_row(tile, s_col, s_row, tid);
        col_std[tid] = kvarn_std_col(tile, s_col, s_row, tid);
        item.barrier();

        kvarn_update_best_from_std(s_col, s_row, false, best_col, best_row, col_std, row_std, best_imbalance, better, reduce, tid, item);
    }

    // Find min/max for uniform quantization
    const int row = tid;
    float lo = 3.402823466e+38F;
    float hi = -3.402823466e+38F;
    for (int col = 0; col < KVAR_N_DIM; ++col) {
        const float x = tile[row * KVAR_N_DIM + col] / (best_col[col] * best_row[row]);
        lo = fminf(lo, x);
        hi = fmaxf(hi, x);
    }

    const int qmax = (1 << bits) - 1;
    const float scale = fmaxf((hi - lo) / qmax, 1e-10f);
    const int row_bytes = KVAR_N_DIM * bits / 8;
    uint8_t * row_payload = record + row * row_bytes;
    for (int i = 0; i < row_bytes; ++i) {
        row_payload[i] = 0;
    }
    for (int col = 0; col < KVAR_N_DIM; ++col) {
        const float x = tile[row * KVAR_N_DIM + col] / (best_col[col] * best_row[row]);
        const uint8_t q = (uint8_t)fminf(fmaxf(roundf((x - lo) / scale), 0.0f), (float)qmax);
        const int bit_offset = col * bits;
        for (int bit = 0; bit < bits; ++bit) {
            const int dst_bit = bit_offset + bit;
            row_payload[dst_bit / 8] |= ((q >> bit) & 1u) << (dst_bit % 8);
        }
    }

    // Write scale/zero-point/other metadata
    const int payload_bytes = KVAR_N_TILE_VALUES * bits / 8;
    sycl::half * scale_axis = (sycl::half *)(record + payload_bytes);
    sycl::half * zp_axis = scale_axis + KVAR_N_DIM;
    sycl::half * other_axis = zp_axis + KVAR_N_DIM;
    scale_axis[row] = static_cast<sycl::half>(best_row[row] * scale);
    zp_axis[row] = static_cast<sycl::half>(best_row[row] * lo);
    other_axis[row] = static_cast<sycl::half>(best_col[row]);
    item.barrier();
}

// Quantize a stage tile from the stage buffer
static void kvarn_quantize_stage(
        const sycl::half * stage, uint8_t * record,
        int n_heads, int head, int stage_base, int stage_group,
        int bits, int iterations, bool value, bool swa,
        int stage_groups, int tail_groups,
        float * shared, int tid, sycl::nd_item<1> item) {

    float * tile = shared + OFF_TILE;
    const int stage_slot = swa ? (stage_group % stage_groups) : (1 + ((stage_group - 1) % tail_groups));

    // Load tile from stage buffer
    for (int i = tid; i < KVAR_N_TILE_VALUES; i += KVAR_N_DIM) {
        const int row = i / KVAR_N_DIM;
        const int col = i % KVAR_N_DIM;
        const int token = value ? row : col;
        const int dim = value ? col : row;
        const int stage_pos = stage_base + stage_slot * KVAR_N_DIM + token;
        tile[i] = static_cast<float>(stage[(stage_pos * n_heads + head) * KVAR_N_DIM + dim]);
    }
    item.barrier();

    kvarn_quantize_tile(tile, record, bits, iterations, shared, tid, item);
}

// Phase 1a: hishmem kernel skeleton
// One work-group per head, 128 threads, full shared memory
// For now: load F32 → WHT (placeholder) → write F16 stage
// No quantization yet — just verify data flow
static void kvarn_store_kernel_hishmem(
        const float * __restrict__ current,
        const int64_t * __restrict__ indices,
        sycl::half * __restrict__ stage,
        uint8_t * __restrict__ records,
        int n_heads,
        int n_tokens,
        int n_stream,
        int groups_per_stream,
        int record_bytes,
        int bits,
        int iterations,
        bool value,
        bool swa,
        int stage_groups,
        int tail_groups,
        bool eager_records,
        queue_ptr q_ptr) {

    sycl::queue q = *q_ptr;
    sycl::nd_range<1> nd_range{n_heads * 128, 128};

    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> shared{sycl::range<1>(KVAR_N_SHARED_FLOATS), cgh};
        float * shared_ptr = shared.get_pointer();

        cgh.parallel_for<>(nd_range, [=](sycl::nd_item<1> item) {
            const int head = item.get_group(0);
            if (head >= n_heads) return;

            const int tid = item.get_local_id()[0];

            for (int token = 0; token < n_tokens; ++token) {
                const int64_t idx = indices[token];
                const int group_global = (int)(idx / KVAR_N_DIM);
                const int pos = (int)(idx % KVAR_N_DIM);
                const int stream = swa ? 0 : group_global / groups_per_stream;
                const int group = swa ? group_global : group_global - stream * groups_per_stream;
                if (stream < 0 || stream >= n_stream || group < 0 || (!swa && group >= groups_per_stream)) {
                    return;
                }

                const int stage_base = stream * KVAR_N_DIM * stage_groups;

                // Flush old record when stage group fills (delayed mode)
                if (!eager_records && pos == 0 && (swa ? group >= tail_groups : group > tail_groups)) {
                    const int flush_group = group - tail_groups;
                    const int flush_ring = swa ? flush_group % groups_per_stream : flush_group;
                    const int flush_record_group = stream * groups_per_stream + flush_ring;
                    uint8_t * record = records + (flush_record_group * n_heads + head) * record_bytes;
                    kvarn_quantize_stage(stage, record, n_heads, head, stage_base, flush_group,
                                         bits, iterations, value, swa, stage_groups, tail_groups,
                                         shared_ptr, tid, item);
                }

                // Load F32 into shared memory
                shared_ptr[tid] = current[(token * n_heads + head) * KVAR_N_DIM + tid];
                item.barrier();

                // WHT: 7-stage butterfly transform
                for (int h = 1; h < KVAR_N_DIM; h *= 2) {
                    if (tid < 64) {
                        const int j = (tid / h) * (2 * h) + (tid % h);
                        const float a = shared_ptr[j];
                        const float b = shared_ptr[j + h];
                        shared_ptr[j] = a + b;
                        shared_ptr[j + h] = a - b;
                    }
                    item.barrier();
                }

                // Scale by 1/√128
                shared_ptr[tid] *= 0.08838834764831845f;
                item.barrier();

                // Write to F16 stage buffer
                const int stage_slot = swa ? (group % stage_groups) : (group == 0 ? 0 : 1 + ((group - 1) % tail_groups));
                const int stage_pos = stage_base + stage_slot * KVAR_N_DIM + pos;
                stage[(stage_pos * n_heads + head) * KVAR_N_DIM + tid] =
                    static_cast<sycl::half>(shared_ptr[tid]);

                item.barrier();

                // Eager record write when group completes
                if (eager_records && pos == KVAR_N_DIM - 1 && (swa || group > 0)) {
                    const int record_ring = swa ? group % groups_per_stream : group;
                    const int record_group = stream * groups_per_stream + record_ring;
                    uint8_t * record = records + (record_group * n_heads + head) * record_bytes;
                    kvarn_quantize_stage(stage, record, n_heads, head, stage_base, group,
                                         bits, iterations, value, swa, stage_groups, tail_groups,
                                         shared_ptr, tid, item);
                }
            }
        });
    });
}

// Headwide kernel: one work-group processes head_slices heads
// Shared memory holds SLICES × 128 floats for the tile
// Uses cross-slice Hadamard for multi-slice heads
template<int SLICES>
static void kvarn_store_kernel_headwide(
        const float * __restrict__ current,
        const int64_t * __restrict__ indices,
        sycl::half * __restrict__ stage,
        uint8_t * __restrict__ records,
        int n_heads,
        int n_tokens,
        int n_stream,
        int groups_per_stream,
        int record_bytes,
        int bits,
        int iterations,
        bool value,
        bool swa,
        int stage_groups,
        int tail_groups,
        bool eager_records,
        queue_ptr q_ptr) {

    sycl::queue q = *q_ptr;
    const int n_groups = n_heads / SLICES;
    sycl::nd_range<1> nd_range{n_groups * 128, 128};

    q.submit([&](sycl::handler& cgh) {
        // Shared memory: SLICES × 128 tile + metadata
        sycl::local_accessor<float, 1> shared{sycl::range<1>(SLICES * KVAR_N_DIM + KVAR_N_SHARED_FLOATS), cgh};
        float * shared_ptr = shared.get_pointer();

        cgh.parallel_for<>(nd_range, [=](sycl::nd_item<1> item) {
            const int group = item.get_group(0);
            if (group >= n_groups) return;

            const int head0 = group * SLICES;
            if (head0 + SLICES > n_heads) return;

            const int tid = item.get_local_id()[0];

            for (int token = 0; token < n_tokens; ++token) {
                const int64_t idx = indices[token];
                const int group_global = (int)(idx / KVAR_N_DIM);
                const int pos = (int)(idx % KVAR_N_DIM);
                const int stream = swa ? 0 : group_global / groups_per_stream;
                const int group_idx = swa ? group_global : group_global - stream * groups_per_stream;
                if (stream < 0 || stream >= n_stream || group_idx < 0 || (!swa && group_idx >= groups_per_stream)) {
                    return;
                }

                const int stage_base = stream * KVAR_N_DIM * stage_groups;

                // Flush old records (delayed mode)
                if (!eager_records && pos == 0 && (swa ? group_idx >= tail_groups : group_idx > tail_groups)) {
                    const int flush_group = group_idx - tail_groups;
                    const int flush_ring = swa ? flush_group % groups_per_stream : flush_group;
                    const int flush_record_group = stream * groups_per_stream + flush_ring;
                    for (int slice = 0; slice < SLICES; ++slice) {
                        const int head = head0 + slice;
                        uint8_t * record = records + (flush_record_group * n_heads + head) * record_bytes;
                        // Load tile from stage into shared tile area
                        float * tile = shared_ptr + slice * KVAR_N_DIM;
                        const int stage_slot = swa ? (flush_group % stage_groups) : (1 + ((flush_group - 1) % tail_groups));
                        for (int i = tid; i < KVAR_N_DIM; i += KVAR_N_DIM) {
                            const int token_val = value ? i : tid;
                            const int dim = value ? tid : i;
                            const int stage_pos = stage_base + stage_slot * KVAR_N_DIM + token_val;
                            tile[i] = static_cast<float>(stage[(stage_pos * n_heads + head) * KVAR_N_DIM + dim]);
                        }
                        item.barrier();
                        kvarn_quantize_tile(tile, record, bits, iterations, shared_ptr + SLICES * KVAR_N_DIM, tid, item);
                    }
                }

                // Load + WHT per slice
                float values[4] = {0};
                for (int slice = 0; slice < SLICES; ++slice) {
                    const int head = head0 + slice;
                    float * tile = shared_ptr + slice * KVAR_N_DIM;
                    tile[tid] = current[(token * n_heads + head) * KVAR_N_DIM + tid];
                    item.barrier();

                    // WHT butterfly
                    for (int h = 1; h < KVAR_N_DIM; h *= 2) {
                        if (tid < 64) {
                            const int j = (tid / h) * (2 * h) + (tid % h);
                            const float a = tile[j];
                            const float b = tile[j + h];
                            tile[j] = a + b;
                            tile[j + h] = a - b;
                        }
                        item.barrier();
                    }
                    tile[tid] *= 0.08838834764831845f;
                    item.barrier();

                    values[slice] = tile[tid];
                }

                // Cross-slice Hadamard
                if (SLICES == 2) {
                    const float a = values[0];
                    const float b = values[1];
                    values[0] = (a + b) * 0.7071067811865475f;
                    values[1] = (a - b) * 0.7071067811865475f;
                } else if (SLICES == 4) {
                    const float a0 = values[0];
                    const float a1 = values[1];
                    const float a2 = values[2];
                    const float a3 = values[3];
                    const float b0 = a0 + a1;
                    const float b1 = a0 - a1;
                    const float b2 = a2 + a3;
                    const float b3 = a2 - a3;
                    values[0] = (b0 + b2) * 0.5f;
                    values[1] = (b1 + b3) * 0.5f;
                    values[2] = (b0 - b2) * 0.5f;
                    values[3] = (b1 - b3) * 0.5f;
                }

                // Write to stage buffer
                const int stage_slot = swa ? (group_idx % stage_groups) : (group_idx == 0 ? 0 : 1 + ((group_idx - 1) % tail_groups));
                const int stage_pos = stage_base + stage_slot * KVAR_N_DIM + pos;
                for (int slice = 0; slice < SLICES; ++slice) {
                    const int head = head0 + slice;
                    stage[(stage_pos * n_heads + head) * KVAR_N_DIM + tid] =
                        static_cast<sycl::half>(values[slice]);
                }
                item.barrier();

                // Eager record write
                if (eager_records && pos == KVAR_N_DIM - 1 && (swa || group_idx > 0)) {
                    const int record_ring = swa ? group_idx % groups_per_stream : group_idx;
                    const int record_group = stream * groups_per_stream + record_ring;
                    for (int slice = 0; slice < SLICES; ++slice) {
                        const int head = head0 + slice;
                        uint8_t * record = records + (record_group * n_heads + head) * record_bytes;
                        float * tile = shared_ptr + slice * KVAR_N_DIM;
                        kvarn_quantize_tile(tile, record, bits, iterations, shared_ptr + SLICES * KVAR_N_DIM, tid, item);
                    }
                }
            }
        });
    });
}

// Phase 1a dispatch — hishmem only, head_slices == 1
void ggml_sycl_op_kvarn_store(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * current = dst->src[0];
    const ggml_tensor * indices = dst->src[1];
    ggml_tensor * stage = dst->src[2];
    ggml_tensor * records = dst->src[3];

    GGML_ASSERT(ggml_is_contiguous(current));
    GGML_ASSERT(ggml_is_contiguous(indices));
    GGML_ASSERT(ggml_is_contiguous(stage));
    GGML_ASSERT(ggml_is_contiguous(records));

    const int bits = ggml_get_op_params_i32(dst, KVAR_N_OP_PARAM_BITS);
    const int iterations = ggml_get_op_params_i32(dst, KVAR_N_OP_PARAM_ITERS);
    const bool value = ggml_get_op_params_i32(dst, KVAR_N_OP_PARAM_STORE_VALUE) != 0;
    const bool swa = ggml_get_op_params_i32(dst, KVAR_N_OP_PARAM_STORE_SWA) != 0;
    const int head_slices_param = ggml_get_op_params_i32(dst, KVAR_N_OP_PARAM_HEAD_SLICES);
    const int head_slices = head_slices_param > 0 ? head_slices_param : 1;
    const int stage_groups = kvarn_resolve_stage_groups(dst);
    const int tail_groups = kvarn_resolve_tail_groups(dst, stage_groups);
    const bool eager_records = ggml_get_op_params_i32(dst, KVAR_N_OP_PARAM_EAGER_RECORDS) != 0;

    GGML_ASSERT(head_slices == 1 || head_slices == 2 || head_slices == 4);
    GGML_ASSERT((KVAR_N_TILE_VALUES * bits) % 8 == 0);
    GGML_ASSERT((KVAR_N_DIM * bits) % 8 == 0);
    GGML_ASSERT(stage_groups >= 2);
    GGML_ASSERT(tail_groups >= 1 && tail_groups <= stage_groups);
    GGML_ASSERT(stage->ne[2] % (KVAR_N_DIM * stage_groups) == 0);

    const int n_stream = (int)(stage->ne[2] / (KVAR_N_DIM * stage_groups));
    GGML_ASSERT(n_stream > 0);
    GGML_ASSERT(records->ne[2] % n_stream == 0);
    const int groups_per_stream = (int)(records->ne[2] / n_stream);

    if (swa) {
        GGML_ASSERT(n_stream == 1 && "SWA KVarN ring requires a single stream");
    }

    const int n_heads = (int)current->ne[1];
    GGML_ASSERT(n_heads % head_slices == 0);
    const int n_tokens = (int)current->ne[2];

    // Dispatch based on head_slices
    if (head_slices == 1) {
        kvarn_store_kernel_hishmem(
            (const float *)current->data,
            (const int64_t *)indices->data,
            (sycl::half *)stage->data,
            (uint8_t *)records->data,
            n_heads,
            n_tokens,
            n_stream,
            groups_per_stream,
            (int)records->ne[0],
            bits,
            iterations,
            value,
            swa,
            stage_groups,
            tail_groups,
            eager_records,
            ctx.stream()
        );
    } else if (head_slices == 2) {
        kvarn_store_kernel_headwide<2>(
            (const float *)current->data,
            (const int64_t *)indices->data,
            (sycl::half *)stage->data,
            (uint8_t *)records->data,
            n_heads,
            n_tokens,
            n_stream,
            groups_per_stream,
            (int)records->ne[0],
            bits,
            iterations,
            value,
            swa,
            stage_groups,
            tail_groups,
            eager_records,
            ctx.stream()
        );
    } else if (head_slices == 4) {
        kvarn_store_kernel_headwide<4>(
            (const float *)current->data,
            (const int64_t *)indices->data,
            (sycl::half *)stage->data,
            (uint8_t *)records->data,
            n_heads,
            n_tokens,
            n_stream,
            groups_per_stream,
            (int)records->ne[0],
            bits,
            iterations,
            value,
            swa,
            stage_groups,
            tail_groups,
            eager_records,
            ctx.stream()
        );
    }
}

// Placeholder — materialize not yet implemented
void ggml_sycl_op_kvarn_materialize(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("KVarN materialize not yet implemented for SYCL");
}
