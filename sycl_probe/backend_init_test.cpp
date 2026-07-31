#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cstring>

static constexpr int KVAR_N_DIM = 128;

int main() {
    printf("Starting...\n");
    fflush(stdout);
    
    ggml_backend_t sycl_backend = nullptr;
    for (int i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            sycl_backend = ggml_backend_dev_init(dev, nullptr);
            break;
        }
    }
    if (!sycl_backend) {
        fprintf(stderr, "SYCL backend not available\n");
        return 1;
    }
    printf("Backend: %s\n", ggml_backend_name(sycl_backend));
    fflush(stdout);
    
    // Create context
    ggml_init_params ctx_params = {
        /* mem_size   = */ 128 * 1024 * 1024,
        /* mem        = */ nullptr,
        /* no_alloc   = */ true
    };
    struct ggml_context * ctx = ggml_init(ctx_params);
    
    // Create tensors
    int n_heads = 8, n_tokens = 128;
    int bits = 4, sinkhorn_iters = 10, stage_groups = 2, n_stream = 1, groups_per_stream = 2;
    int record_bytes = (KVAR_N_DIM * KVAR_N_DIM * bits / 8) + (KVAR_N_DIM * 3 * 2);
    
    ggml_tensor * current = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, KVAR_N_DIM, n_heads, n_tokens);
    ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    ggml_tensor * stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, KVAR_N_DIM, n_heads, KVAR_N_DIM * stage_groups * n_stream);
    ggml_tensor * records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, record_bytes, n_heads, groups_per_stream * n_stream);
    
    ggml_tensor * store_op = ggml_kvarn_store(ctx, current, indices, stage, records,
                                               bits, sinkhorn_iters, false, stage_groups);
    
    // Allocate
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(sycl_backend);
    ggml_gallocr_t galloc = ggml_gallocr_new(buft);
    
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, store_op);
    ggml_gallocr_alloc_graph(galloc, graph);
    
    // Upload
    std::vector<float> data(KVAR_N_DIM * n_heads * n_tokens);
    srand(42);
    for (int i = 0; i < (int)data.size(); ++i) data[i] = (rand() % 1000 - 500) / 500.0f;
    ggml_backend_tensor_set(current, data.data(), 0, data.size() * sizeof(float));
    std::vector<int64_t> idx_data(n_tokens);
    for (int i = 0; i < n_tokens; ++i) idx_data[i] = i;
    ggml_backend_tensor_set(indices, idx_data.data(), 0, n_tokens * sizeof(int64_t));
    
    printf("Computing kvarn_store...\n");
    fflush(stdout);
    ggml_backend_graph_compute(sycl_backend, graph);
    printf("Store complete\n");
    fflush(stdout);
    
    // Materialize
    ggml_tensor * mat_indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    ggml_tensor * output = ggml_kvarn_materialize(ctx, records, stage, mat_indices,
                                                   n_tokens, 0, n_stream, bits, false, stage_groups);
    printf("Materialize op created\n");
    fflush(stdout);
    
    ggml_cgraph * graph2 = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph2, output);
    ggml_gallocr_alloc_graph(galloc, graph2);
    
    ggml_backend_tensor_set(mat_indices, idx_data.data(), 0, n_tokens * sizeof(int64_t));
    
    printf("Computing kvarn_materialize...\n");
    fflush(stdout);
    ggml_backend_graph_compute(sycl_backend, graph2);
    printf("Materialize complete\n");
    fflush(stdout);
    
    // Read back (output is F16, read as F16 then convert)
    printf("Output dims: %lld x %lld x %lld x %lld\n",
           (long long)output->ne[0], (long long)output->ne[1],
           (long long)output->ne[2], (long long)output->ne[3]);
    int64_t output_size = ggml_nelements(output);
    printf("Output elements: %lld, nbytes: %lld, type: %d\n", (long long)output_size, (long long)ggml_nbytes(output), (int)output->type);
    
    // Read as F16
    std::vector<uint16_t> output_f16(output_size);
    ggml_backend_tensor_get(output, output_f16.data(), 0, output_f16.size() * sizeof(uint16_t));
    
    // Convert F16 to F32 (IEEE 754)
    std::vector<float> output_data(output_size);
    for (int i = 0; i < (int)output_size; ++i) {
        uint16_t f16 = output_f16[i];
        int32_t sign = (f16 >> 15) & 0x1;
        int32_t exp = (f16 >> 10) & 0x1F;
        int32_t frac = f16 & 0x3FF;
        float f32;
        if (exp == 0) {
            f32 = ldexpf((frac ? frac : 0) * (sign ? -1.0f : 1.0f), -24);
        } else if (exp == 31) {
            f32 = INFINITY;
        } else {
            f32 = ldexpf((1.0f + frac / 1024.0f) * (sign ? -1.0f : 1.0f), exp - 15);
        }
        output_data[i] = f32;
    }
    printf("First 5 output values: %.4f %.4f %.4f %.4f %.4f\n",
           output_data[0], output_data[1], output_data[2], output_data[3], output_data[4]);
    printf("Output read: %lld values\n", (long long)output_data.size());
    fflush(stdout);
    
    // Compare
    float max_err = 0;
    int max_idx = 0;
    for (int i = 0; i < (int)output_data.size(); ++i) {
        float err = std::abs(data[i] - output_data[i]);
        if (err > max_err) { max_err = err; max_idx = i; }
    }
    printf("Max err at idx %d: input=%.4f output=%.4f\n", max_idx, data[max_idx], output_data[max_idx]);
    printf("Round-trip max error: %.2e\n", max_err);
    printf("%s\n", max_err < 0.15f ? "PASS" : "FAIL");
    fflush(stdout);
    
    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    ggml_backend_free(sycl_backend);
    
    return max_err < 0.15f ? 0 : 1;
}
