// test-sycl-kvarn.cpp — Integration test for SYCL KVarN store/materialize
// Exercises: create tensors → kvarn_store → kvarn_materialize → verify round-trip

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <thread>
#include <cstring>

static constexpr int KVAR_N_DIM = 128;

int main(int argc, char ** argv) {
    ggml_time_init();

    // Find SYCL backend via device
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
    printf("Device count: %d\n", ggml_backend_dev_count());

    // Test parameters
    int bits = 4;
    int sinkhorn_iters = 10;
    int n_heads = 8;
    int n_tokens = 128; // exactly one group
    int n_stream = 1;
    int groups_per_stream = 2;
    int stage_groups = 2;
    bool value = false; // key cache

    int record_bytes = (KVAR_N_DIM * KVAR_N_DIM * bits / 8) + (KVAR_N_DIM * 3 * 2); // payload + 3x F16 axes

    // Create context
    ggml_init_params ctx_params = {
        /* mem_size   = */ 128 * 1024 * 1024,
        /* mem        = */ nullptr,
        /* no_alloc   = */ false
    };
    struct ggml_context * ctx = ggml_init(ctx_params);

    // Create tensors
    ggml_tensor * current = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, KVAR_N_DIM, n_heads, n_tokens);
    ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    ggml_tensor * stage = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, KVAR_N_DIM, n_heads, KVAR_N_DIM * stage_groups * n_stream);
    ggml_tensor * records = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, record_bytes, n_heads, groups_per_stream * n_stream);

    // Create store op
    ggml_tensor * store_op = ggml_kvarn_store(ctx, current, indices, stage, records,
                                               bits, sinkhorn_iters, value, stage_groups);

    // Generate test data
    std::vector<float> input_data(KVAR_N_DIM * n_heads * n_tokens);
    for (int i = 0; i < KVAR_N_DIM * n_heads * n_tokens; ++i) {
        input_data[i] = static_cast<float>(rand() % 1000 - 500) / 1000.0f;
    }
    memcpy(ggml_get_data(current), input_data.data(), input_data.size() * sizeof(float));

    // Set indices (sequential, one group)
    std::vector<int64_t> indices_data(n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        indices_data[i] = i; // group 0, positions 0..127
    }
    memcpy(ggml_get_data(indices), indices_data.data(), indices_data.size() * sizeof(int64_t));

    // Allocate buffers on SYCL
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(sycl_backend);
    ggml_gallocr_t galloc = ggml_gallocr_new(buft);
    if (!galloc) {
        fprintf(stderr, "Failed to create graph allocator\n");
        ggml_free(ctx);
        ggml_backend_free(sycl_backend);
        return 1;
    }

    // Build graph
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, store_op);

    ggml_gallocr_alloc_graph(galloc, graph);

    printf("Computing kvarn_store...\n");
    ggml_backend_graph_compute(sycl_backend, graph);
    printf("Store complete\n");

    // Create materialize op
    ggml_tensor * mat_indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        ((int64_t *)ggml_get_data(mat_indices))[i] = i;
    }

    ggml_tensor * output = ggml_kvarn_materialize(ctx, records, stage, mat_indices,
                                                   n_tokens, 0, n_stream, bits, value, stage_groups);

    ggml_cgraph * graph2 = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph2, output);

    ggml_gallocr_alloc_graph(galloc, graph2);

    printf("Computing kvarn_materialize...\n");
    ggml_backend_graph_compute(sycl_backend, graph2);
    printf("Materialize complete\n");

    ggml_gallocr_free(galloc);

    // Read back output
    int64_t output_size = output->ne[0] * output->ne[1] * output->ne[2];
    std::vector<float> output_data(output_size);
    ggml_backend_tensor_get(output, output_data.data(), 0, output_data.size() * sizeof(float));

    // Compare
    float max_err = 0;
    float rms_err = 0;
    int compare_count = std::min((int64_t)input_data.size(), output_size);
    for (int i = 0; i < compare_count; ++i) {
        float err = std::abs(input_data[i] - output_data[i]);
        if (err > max_err) max_err = err;
        rms_err += err * err;
    }
    rms_err = std::sqrt(rms_err / compare_count);

    printf("Round-trip max error: %.2e\n", max_err);
    printf("Round-trip RMS error: %.2e\n", rms_err);
    printf("Expected quantization error (~4-bit): ~0.03-0.07 max\n");

    bool pass = max_err < 0.15f;
    printf("%s\n", pass ? "PASS" : "FAIL");

    ggml_free(ctx);
    ggml_backend_free(sycl_backend);

    return pass ? 0 : 1;
}
