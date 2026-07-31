#include "kvarn.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

// KVarN constants (match llama-kvarn.h)
static constexpr int KVAR_N_DIM = 128;
static constexpr int KVAR_N_TILE_VALUES = KVAR_N_DIM * KVAR_N_DIM;

// Shared memory layout for headwide variant (F32 tile + metadata)
static constexpr int KVAR_N_SHARED_FLOATS = KVAR_N_TILE_VALUES + 8 * KVAR_N_DIM + 2;
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

                // Phase 1a: load F32 into shared memory
                shared[tid] = current[(token * n_heads + head) * KVAR_N_DIM + tid];
                item.barrier();

                // Phase 1b placeholder: WHT will go here
                // For now just pass through

                // Write to F16 stage buffer
                const int stage_slot = swa ? (group % stage_groups) : (group == 0 ? 0 : 1 + ((group - 1) % tail_groups));
                const int stage_pos = stage_base + stage_slot * KVAR_N_DIM + pos;
                stage[(stage_pos * n_heads + head) * KVAR_N_DIM + tid] =
                    static_cast<sycl::half>(shared[tid]);

                item.barrier();

                // Phase 1c/1d placeholder: quantization will go here
                // For now skip record writing
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

    // Phase 1a: hishmem only, head_slices == 1
    // TODO: add headwide, lowshmem, workspace, direct paths
    GGML_ASSERT(head_slices == 1 && "head_slices > 1 not yet supported in SYCL");

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
}

// Placeholder — materialize not yet implemented
void ggml_sycl_op_kvarn_materialize(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("KVarN materialize not yet implemented for SYCL");
}
