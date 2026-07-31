#pragma once

#include "common.hpp"

void ggml_sycl_op_kvarn_store(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
void ggml_sycl_op_kvarn_materialize(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
