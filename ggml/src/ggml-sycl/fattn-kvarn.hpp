#pragma once

#include "common.hpp"
#include "ggml.h"

#include <sycl/sycl.hpp>
#include <cstdint>

// KVarN flash attention — SYCL portable fallback
// Reads KVarN records directly, dequantizes per-element, accumulates Q×K and V
// No XMX/DPAS — scalar F16/F32 only for Phase 2 baseline

// Kernel launcher: KVarN flash attention
void ggml_sycl_flash_attn_ext_kvarn(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
