// KVarN integration test: store → materialize round-trip
// Validates KVarN SYCL kernels through ggml backend
//
// Build: icpx -fsycl -o fattn_kvarn_integration fattn_kvarn_integration.cpp \
//        -L../build/bin -lggml-base -lggml -lggml-sycl -Wl,-rpath,/workspaces/beellama-sycl-kvarn/build/bin

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>

static constexpr int KVAR_N_DIM = 128;

int main() {
    printf("=== KVarN Integration Test (store → materialize) ===\n\n");

    ggml_backend_t backend = nullptr;
    for (int i = 0; i < ggml_backend_dev_count(); ++i) {
        auto dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            backend = ggml_backend_dev_init(dev, nullptr);
            break;
        }
    }
    if (!backend) { fprintf(stderr, "No GPU backend\n"); return 1; }
    printf("Backend: %s\n", ggml_backend_name(backend));

    int dim = KVAR_N_DIM, n_kv = 8, n_kh = 2;
    int bits = 4, sg = 2, ns = 1, gps = 2;
    int rb = dim * dim * bits / 8 + dim * 3 * sizeof(float);

    // Generate test data
    srand(42);
    std::vector<float> k_data(dim * n_kh * n_kv);
    std::vector<float> v_data(dim * n_kh * n_kv);
    for (auto & x : k_data) x = (rand() % 1000 - 500) / 1000.0f;
    for (auto & x : v_data) x = (rand() % 1000 - 500) / 1000.0f;

    // Build graph: store → materialize
    ggml_init_params p = {256 * 1024 * 1024, nullptr, true};
    auto * ctx = ggml_init(p);

    auto * k_cur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, n_kh, n_kv);
    auto * v_cur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, n_kh, n_kv);
    auto * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_kv);
    auto * k_stg = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, dim, n_kh, dim * sg * ns);
    auto * k_rec = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, rb, n_kh, gps * ns);
    auto * v_stg = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, dim, n_kh, dim * sg * ns);
    auto * v_rec = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, rb, n_kh, gps * ns);

    auto * k_store = ggml_kvarn_store(ctx, k_cur, idx, k_stg, k_rec, bits, 10, 0, sg);
    auto * v_store = ggml_kvarn_store(ctx, v_cur, idx, v_stg, v_rec, bits, 10, 0, sg);
    auto * k_mat = ggml_kvarn_materialize(ctx, k_rec, k_stg, idx, n_kv, 0, ns, bits, 0, sg);
    auto * v_mat = ggml_kvarn_materialize(ctx, v_rec, v_stg, idx, n_kv, 0, ns, bits, 0, sg);

    auto * buft = ggml_backend_get_default_buffer_type(backend);
    auto * ga = ggml_gallocr_new(buft);

    auto * gs = ggml_new_graph_custom(ctx, 32, 0);
    ggml_build_forward_expand(gs, k_store);
    ggml_build_forward_expand(gs, v_store);
    ggml_gallocr_alloc_graph(ga, gs);

    auto * gf = ggml_new_graph_custom(ctx, 32, 0);
    ggml_build_forward_expand(gf, k_mat);
    ggml_build_forward_expand(gf, v_mat);
    ggml_gallocr_alloc_graph(ga, gf);

    // Upload
    ggml_backend_tensor_set(k_cur, k_data.data(), 0, k_data.size() * sizeof(float));
    ggml_backend_tensor_set(v_cur, v_data.data(), 0, v_data.size() * sizeof(float));
    std::vector<int64_t> ii(n_kv);
    for (int i = 0; i < n_kv; ++i) ii[i] = i;
    ggml_backend_tensor_set(idx, ii.data(), 0, ii.size() * sizeof(int64_t));

    // Compute
    printf("Computing store + materialize...\n");
    ggml_backend_graph_compute(backend, gs);
    ggml_backend_graph_compute(backend, gf);

    // Read back (materialize outputs F16, read as F16 then convert)
    std::vector<uint16_t> k_out_f16(ggml_nelements(k_mat));
    std::vector<uint16_t> v_out_f16(ggml_nelements(v_mat));
    ggml_backend_tensor_get(k_mat, k_out_f16.data(), 0, k_out_f16.size() * sizeof(uint16_t));
    ggml_backend_tensor_get(v_mat, v_out_f16.data(), 0, v_out_f16.size() * sizeof(uint16_t));
    
    auto f16_to_f32 = [](uint16_t h) -> float {
        uint32_t sign = (h >> 15) & 0x1;
        int32_t exp = (h >> 10) & 0x1F;
        uint32_t frac = h & 0x3FF;
        if (exp == 0) return sign ? -0.0f : 0.0f;
        if (exp == 31) return sign ? -INFINITY : INFINITY;
        exp -= 15 + 127;
        uint32_t i = (sign << 31) | (exp << 23) | (frac << 13);
        union { uint32_t i; float f; } u = {i};
        return u.f;
    };
    
    std::vector<float> k_out(k_out_f16.size());
    std::vector<float> v_out(v_out_f16.size());
    for (size_t i = 0; i < k_out_f16.size(); ++i) k_out[i] = f16_to_f32(k_out_f16[i]);
    for (size_t i = 0; i < v_out_f16.size(); ++i) v_out[i] = f16_to_f32(v_out_f16[i]);

    // Compare
    float k_err = 0, v_err = 0;
    for (size_t i = 0; i < k_data.size(); ++i) {
        float e = std::abs(k_data[i] - k_out[i]);
        if (e > k_err) k_err = e;
    }
    for (size_t i = 0; i < v_data.size(); ++i) {
        float e = std::abs(v_data[i] - v_out[i]);
        if (e > v_err) v_err = e;
    }

    printf("K round-trip error: %.2e %s\n", k_err, k_err < 0.15f ? "PASS" : "FAIL");
    printf("V round-trip error: %.2e %s\n", v_err, v_err < 0.15f ? "PASS" : "FAIL");

    ggml_gallocr_free(ga);
    ggml_free(ctx);
    ggml_backend_free(backend);

    return (k_err < 0.15f && v_err < 0.15f) ? 0 : 1;
}
