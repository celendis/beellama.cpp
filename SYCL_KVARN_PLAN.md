# SYCL KVarN — Plan of Action

## Hardware Target

Intel Arc Pro B70 (Xe2 / Battlemage), 32GB VRAM. oneAPI 2026.1.1, Level Zero 1.28.6.

## Goals (in priority order)

1. **Store** — compress F16 KV cache into KVarN quantized records on SYCL
2. **Materialize** — decompress records back to F16 on demand
3. **WHT** — Walsh-Hadamard Transform (used by both store and materialize)
4. **Native Flash Attention** — read KVarN records directly in attention (deferred)

Phase 1 (store + materialize + WHT) gives memory savings immediately. Phase 2 (native flash attention) is a decode-speed optimization that requires XMX matrix extensions and is significantly more work.

## Reference Implementations

| Backend | Location | Lines |
|---------|----------|-------|
| CUDA | `ggml/src/ggml-cuda/kvarn.cu` | ~2165 |
| CUDA FA | `ggml/src/ggml-cuda/fattn-kvarn-dispatch.cu` + 15 headers + 53 templates | ~4800 |
| CUDA WHT | `ggml/src/ggml-cuda/kvarn-wht.cu` | ~233 |
| Vulkan | `ggml/src/ggml-vulkan/vulkan-shaders/kvarn_*.comp` | ~1912 |
| CPU | `ggml/src/ggml-cpu/ggml-cpu.c` (GGML_OP_KVARN_*) | baseline |

## Core Algorithm (backend-agnostic)

The KVarN pipeline per 128×128 tile:

1. **WHT** — 7-stage butterfly transform on 128-element vectors, then scale by `1/√128`
2. **Normalize** — per-column and per-row z-score (subtract mean, divide by std)
3. **Quantize** — uniform quantization to configurable bit-width (2-8 bits, independent K/V)
4. **Pack** — write quantized values + scale/zero-point metadata to compact record

### Store variants

The CUDA backend has 5 store variants based on shared memory availability:

| Variant | Shared Memory | Use Case |
|---------|--------------|----------|
| `headwide` | Full (128×128 F32 tile) | Default, best performance |
| `hishmem` | Full | Per-head, simpler indexing |
| `lowshmem` | Reduced (6×128 + metadata) | When shared mem is constrained |
| `direct` | Minimal, writes to records directly | Ultra-low memory |
| `workspace` | Uses device workspace buffer | Multi-step with validation |

For SYCL Phase 1, start with **headwide** only. Add lowshmem if needed for specific workloads.

## File Layout

New files under `ggml/src/ggml-sycl/`:

```
ggml/src/ggml-sycl/
├── kvarn.hpp              # Declarations (mirror kvarn.cuh)
├── kvarn.cpp              # Store + materialize kernels (SYCL version of kvarn.cu)
├── kvarn-wht.cpp          # WHT kernel (SYCL version of kvarn-wht.cu)
└── template-instances/    # (Phase 2) fattn-kvarn template instances
```

## SYCL Translation Map

| CUDA | SYCL |
|------|------|
| `__global__` kernel | `q.submit([&](handler& h) { h.parallel_for<>(nd_range, kernel); })` |
| `blockIdx.x` | `get_group_id()[0]` |
| `threadIdx.x` | `get_local_id()[0]` |
| `blockDim.x` | `get_local_range()[0]` |
| `__syncthreads()` | `item.get_group().barrier()` |
| `extern __shared__ float[]` | `accessor<float, access::target::local>` |
| `__float2half_rn()` | `static_cast<sycl::half>(f)` |
| `__half2float()` | `static_cast<float>(h)` |
| `cudaStream_t` | `sycl::queue` (implicit via submit) |
| `<<<blocks, threads, shmem, stream>>>` | `nd_range<1>{global, local}` + local accessor |

## Phase 1 — Store + Materialize + WHT

### kvarn-wht.cpp

Walsh-Hadamard Transform on 128-element vectors. Pure arithmetic, no special intrinsics.

```
Input:  F32[128] per thread
Output: F32[128] after 7 butterfly stages + scale
Sync:   barrier after each stage
```

Direct translation of `kvarn_wht_128()` from CUDA. No hardware-specific features needed.

### kvarn.cpp — Store

The store kernel processes one head per work-group (128 threads = 128 work-items). Per token:

1. Load F32 values into local memory (shared)
2. Run WHT in-place
3. Write to F16 stage buffer
4. When a stage group fills, quantize the tile and write the record

Key data structures (from `llama-kvarn.h`):
- `KVAR_N_GROUP = 128` — tile dimension
- Stage buffer: `half[n_stream × stage_groups × 128 × n_heads × 128]`
- Records: `uint8_t[n_records × n_heads × record_bytes]`
- `record_bytes` depends on K/V bit-width pair (e.g., kvarn2 = ~128 bytes/head/record)

### kvarn.cpp — Materialize

Reverse of store: read record → dequantize → inverse WHT → emit F16.

Simpler than store because it's a single-pass decode with no staging logic.

### ggml-sycl.cpp integration

Add case handlers in `ggml_sycl_get_type_op_params()` and `ggml_sycl_init_op()`:

```cpp
case GGML_OP_KVARN_WHT:
case GGML_OP_KVARN_STORE:
case GGML_OP_KVARN_MATERIALIZE:
```

Also register in `ggml_sycl_can_init_cgraph()` for graph support.

### CMakeLists.txt

Add new sources to the glob or explicitly:

```cmake
# kvarn files
list(APPEND GGML_SOURCES_SYCL "kvarn.cpp" "kvarn-wht.cpp")
```

May need a CMake flag like `GGML_SYCL_KVARN=ON` (default ON) to allow disabling.

## Phase 2 — Native Flash Attention

Deferred until Phase 1 is working. This requires:

### XMX Assessment (2026-07-30)

Probed on Arc Pro B70, driver 1.15.38646+6, oneAPI 2026.1.1:

- `ext_intel_matrix` aspect (ID 53): **YES** — advertised by driver
- `joint_matrix` API: **FAILS** on stock oneAPI — `OpJointMatrixLoadINTEL` SPIR-V builtin not linked
  - Works on intel/llvm nightly but requires subprocess bridge (ABI split)
- **ESIMD `xmx::dpas`**: **WORKS** on stock oneAPI — bypasses `joint_matrix` entirely
- **oneDNN**: **WORKS** on stock oneAPI — used by upstream llama.cpp for standard SDPA

See `sycl_probe/NOTES.md` for full findings.

**Critical context from Bryan Vine's B70 investigation** (turboquant-xpu, 2026-04):
- Custom SYCL flash attention with XMX is **30-60× slower than fused Triton** even when DPAS fires
- DPAS is NOT the bottleneck — scalar softmax (55% wall time) and K/V dequant dominate
- Intel's Triton XPU backend already emits DPAS, so custom DPAS isn't the performance lever

**Impact on KVarN:**
- Phase 1 (store + materialize): unaffected, uses scalar F16/F32
- Phase 2 (native flash attention): ESIMD `xmx::dpas` is viable for the matmul portion
  but the dequantize path will dominate. Scalar F16 fallback is the pragmatic starting point.

Action item: compile and run a small XMX capability probe (the initial probe had compilation errors — need to fix the oneAPI matrix API usage).

### Fattn-kvarn translation

The CUDA fattn-kvarn path uses MMA intrinsics for the dequantize-and-multiply step. SYCL equivalent would use either:

- `sycl::ext::oneapi::matrix::read_matrix_async` + `matrix_mul` + `write_matrix` (XMX path)
- Scalar F16 load + dequantize + accumulate (fallback path)

The template matrix covers all ordered K/V bit-width pairs (15 in default build, 36 in full). Each template is a specialized kernel for that pair's dequantization pattern.

For Phase 2, start with a **scalar fallback** that works without XMX, then layer XMX on top for the common pairs (q4/q4, q5/q4, q5/q5, etc.).

## Risks and Unknowns

| Risk | Impact | Mitigation |
|------|--------|------------|
| XMX not exposed in oneAPI 2026.1 for Battlemage | Phase 2 blocked | Scalar fallback works; wait for newer oneAPI |
| SYCL local memory limits smaller than CUDA shared | Must use lowshmem variant | Start with headwide, fall back if needed |
| `sycl::half` performance vs CUDA `half` | Slower store/materialize | Profile early, use `sycl::ext::oneapi::experimental::half` if available |
| Template bloat (53 CUDA templates) | Long compile times | Start with 15 default pairs, gate rest behind flag |
| SWA (sliding window attention) ring semantics | Subtle correctness bugs | Test against CUDA reference output |

## Verification Strategy

For every kernel, compare against CUDA output:

1. **WHT**: known input → compare F32 output bit-exact
2. **Store**: same F16 stage → compare quantized records byte-exact
3. **Materialize**: same records → compare F16 output within quantization error
4. **End-to-end**: run `llama-perplexity` with KVarN cache on SYCL vs CUDA, compare PPL

## Estimated Effort

| Phase | Work | Duration |
|-------|------|----------|
| ~~Phase 1: WHT~~ | ~~kvarn-wht.cpp~~ | ~~✅ Done~~ |
| ~~Phase 1: Store (1a-1d)~~ | ~~hishmem + headwide~~ | ~~✅ Done~~ |
| ~~Phase 1: Materialize~~ | ~~live tracking + decode~~ | ~~✅ Done~~ |
| Phase 1: Backend integration + testing | CMake, ggml-sycl.cpp, tests | 1-2 days |
| Phase 2: XMX probe + scalar fattn fallback | ~1500 lines | 1-2 weeks |
| Phase 2: XMX accelerated path | Template matrix + tuning | 1-2 weeks |
| **Total** | **~6500 lines** | **4-7 weeks** |

## Store Kernel Sub-Phases

The store kernel (~2000 lines in CUDA) breaks into incremental, testable stages:

| Phase | Scope | What It Does | Test |
|-------|-------|--------------|------|
| 1a | Skeleton + Load/Store | Kernel structure, local memory, load F16→F32, write F32→F16 | Data round-trips unchanged |
| 1b | WHT Integration | Inline butterfly stages + scale from kvarn-wht.cpp | Matches standalone WHT output |
| 1c | Normalization | Per-column z-score (mean/variance reduction), per-row z-score | Mean ≈ 0, std ≈ 1 per axis |
| 1d | Quantization + Packing | Convert to unsigned ints at target bit-width, pack bits into records | Records match CUDA byte-exact |
| 2 | Swarms (multi-stream) | Stage groups, stream routing, eager emit | Multi-token correctness |
| 3 | Low shared memory variant | Reduced local memory path for constrained workloads | Matches headwide output |

**Materialize** is simpler — single-pass reverse: unpack → dequantize → inverse WHT → F16.

## First Steps

1. ~~Implement `kvarn-wht.cpp`~~ ✅ **DONE** — committed to `sycl-kvarn` branch
2. Implement `kvarn.cpp` Phase 1a (skeleton + load/store)
3. Phase 1b (WHT integration)
4. Phase 1c (normalization)
5. Phase 1d (quantization + packing)
6. Wire up `GGML_OP_KVARN_STORE` in `ggml-sycl.cpp`
7. Implement materialize
8. Full correctness validation against CUDA
