#include "kvarn-wht.hpp"

#include <sycl/ext/oneapi/experimental/shuffle.hpp>

#if defined(__INTEL_LLVM_COMPILER) && __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
    #include <sycl/ext/oneapi/bfloat16.hpp>
#endif

// Convert input type to float
static SYCL_INLINE float kvarn_wht_to_float(float value) {
    return value;
}

static SYCL_INLINE float kvarn_wht_to_float(sycl::half value) {
    return static_cast<float>(value);
}

#ifdef GGML_SYCL_HAS_BF16
static SYCL_INLINE float kvarn_wht_to_float(sycl::ext::oneapi::bfloat16 value) {
    return static_cast<float>(value);
}
#endif

// Convert float to output type
template<typename T>
static SYCL_INLINE T kvarn_wht_from_float(float value) {
    return static_cast<T>(value);
}

// WHT kernel using shared (local) memory — direct translation of CUDA shared variant
// Each work-group processes one group (128 threads, one per element of the 128-vector)
template<typename T, int SLICES>
static void kvarn_wht_shared_kernel(
        const T * __restrict__ src,
        T * __restrict__ dst,
        int64_t n_groups,
        sycl::queue & q_ptr) {

    auto q = *q_ptr;
    sycl::nd_range<1> nd_range{n_groups * 128, 128};

    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> buf{sycl::range<1>(SLICES * 128), cgh};

        cgh.parallel_for<>(nd_range, [=](sycl::nd_item<1> item) {
            const int64_t group = item.get_group_id()[0];
            if (group >= n_groups) return;

            const int tid = item.get_local_id()[0];
            const int64_t offset = group * (SLICES * 128);

            // Load into local memory
            for (int slice = 0; slice < SLICES; ++slice) {
                buf[slice * 128 + tid] = kvarn_wht_to_float(src[offset + slice * 128 + tid]);
            }
            item.get_group().barrier();

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
                item.get_group().barrier();
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

// WHT kernel using sub-group shuffles — translation of CUDA wave variant
// Uses shuffle_xor for intra-wave communication (stages 1-3 for 16-lane sub-groups)
// and local memory for cross-wave stages
template<typename T, int SLICES>
static void kvarn_wht_wave_kernel(
        const T * __restrict__ src,
        T * __restrict__ dst,
        int64_t n_groups,
        sycl::queue & q_ptr) {

    auto q = *q_ptr;
    sycl::nd_range<1> nd_range{n_groups * 128, 128};

    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> cross_wave{sycl::range<1>(SLICES * 128), cgh};

        cgh.parallel_for<>(nd_range, [=](sycl::nd_item<1> item) {
            const int64_t group = item.get_group_id()[0];
            if (group >= n_groups) return;

            const int tid = item.get_local_id()[0];
            auto sg = item.get_sub_group();
            const int wave_size = sg.get_local_range()[0];
            const int lane = item.get_local_id()[0] % wave_size;
            const int64_t offset = group * (SLICES * 128);

            float x[4] = {0};

            // Load directly into registers
            for (int slice = 0; slice < SLICES; ++slice) {
                x[slice] = kvarn_wht_to_float(src[offset + slice * 128 + tid]);
            }

            // Intra-wave butterfly using shuffle_xor
            for (int h = 1; h < wave_size; h *= 2) {
                for (int slice = 0; slice < SLICES; ++slice) {
                    const float partner = sycl::ext::oneapi::experimental::shuffle_xor(sg, x[slice], h);
                    x[slice] = (lane & h) != 0 ? partner - x[slice] : x[slice] + partner;
                }
            }

            // Cross-wave butterfly using local memory
            for (int h = wave_size; h < 128; h *= 2) {
                // Write to local memory
                for (int slice = 0; slice < SLICES; ++slice) {
                    cross_wave[slice * 128 + tid] = x[slice];
                }
                sg.barrier();

                // Read partner from local memory
                float partner[4] = {0};
                for (int slice = 0; slice < SLICES; ++slice) {
                    partner[slice] = cross_wave[slice * 128 + (tid ^ h)];
                }
                sg.barrier();

                // Apply butterfly
                for (int slice = 0; slice < SLICES; ++slice) {
                    x[slice] = (tid & h) != 0 ? partner[slice] - x[slice] : x[slice] + partner[slice];
                }
            }

            // Scale
            constexpr float inv_sqrt_128 = 0.08838834764831845f;
            for (int slice = 0; slice < SLICES; ++slice) {
                x[slice] *= inv_sqrt_128;
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

    sycl::queue * q_ptr = ctx.sycl_queue;
    const bool use_shared = false; // wave kernel is default for SYCL (matches CUDA default)

#define GGML_SYCL_KVARN_WHT_LAUNCH(T, SLICES) \
    do { \
        if (use_shared) { \
            kvarn_wht_shared_kernel<T, SLICES>( \
                (const T *) src0->data, (T *) dst->data, n_groups, q_ptr); \
        } else { \
            kvarn_wht_wave_kernel<T, SLICES>( \
                (const T *) src0->data, (T *) dst->data, n_groups, q_ptr); \
        } \
    } while (0)

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
