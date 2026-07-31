#include "kvarn-wht.hpp"

#if defined(__INTEL_LLVM_COMPILER) && __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
    #include <sycl/ext/oneapi/bfloat16.hpp>
#endif

// Convert input type to float
static inline __attribute__((always_inline)) float kvarn_wht_to_float(float value) {
    return value;
}

static inline __attribute__((always_inline)) float kvarn_wht_to_float(sycl::half value) {
    return static_cast<float>(value);
}

#ifdef GGML_SYCL_HAS_BF16
static inline __attribute__((always_inline)) float kvarn_wht_to_float(sycl::ext::oneapi::bfloat16 value) {
    return static_cast<float>(value);
}
#endif

// Convert float to output type
template<typename T>
static inline __attribute__((always_inline)) T kvarn_wht_from_float(float value) {
    return static_cast<T>(value);
}

// WHT kernel using shared (local) memory
// Each work-group processes one group (128 threads, one per element of the 128-vector)
// Direct translation of CUDA kvarn_wht_shared_kernel
template<typename T, int SLICES>
static void kvarn_wht_kernel(
        const T * __restrict__ src,
        T * __restrict__ dst,
        int64_t n_groups,
        queue_ptr q_ptr) {

    sycl::queue q = *q_ptr;
    sycl::nd_range<1> nd_range{n_groups * 128, 128};

    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> buf{sycl::range<1>(SLICES * 128), cgh};

        cgh.parallel_for<>(nd_range, [=](sycl::nd_item<1> item) {
            const int64_t group = item.get_group(0);
            if (group >= n_groups) return;

            const int tid = item.get_local_id()[0];
            const int64_t offset = group * (SLICES * 128);

            // Load into local memory
            for (int slice = 0; slice < SLICES; ++slice) {
                buf[slice * 128 + tid] = kvarn_wht_to_float(src[offset + slice * 128 + tid]);
            }
            item.barrier();

            // 7-stage butterfly transform
            for (int h = 1; h < 128; h *= 2) {
                if (tid < 64) {
                    const int j = (tid / h) * (2 * h) + (tid % h);
                    for (int slice = 0; slice < SLICES; ++slice) {
                        const float a = buf[slice * 128 + j];
                        const float b = buf[slice * 128 + j + h];
                        buf[slice * 128 + j] = a + b;
                        buf[slice * 128 + j + h] = a - b;
                    }
                }
                item.barrier();
            }

            // Scale and collect
            float x[4] = {0};
            constexpr float inv_sqrt_128 = 0.08838834764831845f;
            for (int slice = 0; slice < SLICES; ++slice) {
                x[slice] = buf[slice * 128 + tid] * inv_sqrt_128;
            }

            // Cross-slice Hadamard for multi-slice heads
            if (SLICES == 2) {
                const float a = x[0];
                const float b = x[1];
                x[0] = (a + b) * 0.7071067811865475f;
                x[1] = (a - b) * 0.7071067811865475f;
            } else if (SLICES == 4) {
                const float a0 = x[0];
                const float a1 = x[1];
                const float a2 = x[2];
                const float a3 = x[3];
                const float b0 = a0 + a1;
                const float b1 = a0 - a1;
                const float b2 = a2 + a3;
                const float b3 = a2 - a3;
                x[0] = (b0 + b2) * 0.5f;
                x[1] = (b1 + b3) * 0.5f;
                x[2] = (b0 - b2) * 0.5f;
                x[3] = (b1 - b3) * 0.5f;
            }

            // Write output
            for (int slice = 0; slice < SLICES; ++slice) {
                dst[offset + slice * 128 + tid] = kvarn_wht_from_float<T>(x[slice]);
            }
        });
    });
}

void ggml_sycl_op_kvarn_wht(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16
#ifdef GGML_SYCL_HAS_BF16
                || src0->type == GGML_TYPE_BF16
#endif
                );
    GGML_ASSERT(dst->type == src0->type);

    int head_width;
    memcpy(&head_width, dst->op_params, sizeof(head_width));
    GGML_ASSERT(head_width == 128 || head_width == 256 || head_width == 512);

    const int64_t n_elements = ggml_nelements(src0);
    GGML_ASSERT(n_elements % head_width == 0);
    const int64_t n_groups = n_elements / head_width;
    if (n_groups == 0) {
        return;
    }

    queue_ptr q_ptr = ctx.stream();

#define GGML_SYCL_KVARN_WHT_LAUNCH(T, SLICES) \
    kvarn_wht_kernel<T, SLICES>( \
        (const T *) src0->data, (T *) dst->data, n_groups, q_ptr)

#define GGML_SYCL_KVARN_WHT_TYPE(T) \
    do { \
        switch (head_width) { \
            case 128: GGML_SYCL_KVARN_WHT_LAUNCH(T, 1); break; \
            case 256: GGML_SYCL_KVARN_WHT_LAUNCH(T, 2); break; \
            case 512: GGML_SYCL_KVARN_WHT_LAUNCH(T, 4); break; \
            default: GGML_ABORT("unsupported KVarN WHT head width"); \
        } \
    } while (0)

    switch (src0->type) {
        case GGML_TYPE_F32:  GGML_SYCL_KVARN_WHT_TYPE(float); break;
        case GGML_TYPE_F16:  GGML_SYCL_KVARN_WHT_TYPE(sycl::half); break;
#ifdef GGML_SYCL_HAS_BF16
        case GGML_TYPE_BF16: GGML_SYCL_KVARN_WHT_TYPE(sycl::ext::oneapi::bfloat16); break;
#endif
        default: GGML_ABORT("unsupported KVarN WHT input type");
    }

#undef GGML_SYCL_KVARN_WHT_TYPE
#undef GGML_SYCL_KVARN_WHT_LAUNCH
}
