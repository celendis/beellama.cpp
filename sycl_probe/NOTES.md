# SYCL oneAPI 2026.1 — API Notes

Discovered while building the XMX probe for Intel Arc Pro B70 (Xe2 / Battlemage).

## Device / Aspect Queries

- `dev.get_info<sycl::info::device::aspects>()` returns `std::vector<sycl::aspect>`, **not** a bitmask
  - No `&` operator — use `std::find()` instead
- `sycl::default_selector{}` is deprecated; use `sycl::default_selector_v`
- `sycl::queue q{selector}` with 1.2.1 selector is deprecated; `default_selector_v` is SYCL 2020 so it's fine
- `sycl::aspect::fp32` does **not** exist (FP32 is always assumed)
- `sycl::aspect::ext_intel_matrix` (aspect ID 53) reports XMX support — **YES on B70**
- Intel GPU hardware info (EU count, SIMD width, slices, etc.) moved from `sycl::info::device::ext_intel_gpu_*` to `sycl::ext::intel::info::property::gpu_*` — but the `property` namespace doesn't seem to exist in 2026.1 either. Deprecated path doesn't work. Skip for now.

## Buffer API

- `sycl::buffer<T, 1>(data, range, queue)` — the queue parameter was removed
  - Old: `buffer(ptr, range, queue)`
  - New: `buffer(ptr, range)` — queue is implicit from submission
- `buf.get_access<mode>(h)` needs `template` keyword when `buf` is a dependent name: `buf.template get_access<mode>(h)`
- **Use USM (`sycl::malloc_shared`) instead** — simpler, no accessor boilerplate

## XMX / Cooperative Matrix API

### Headers

```cpp
#include <sycl/ext/oneapi/matrix/matrix.hpp>     // joint_matrix, load/store/mad/fill
#include <sycl/ext/oneapi/bfloat16.hpp>          // sycl::ext::oneapi::bfloat16
```

### Matrix Type

```cpp
joint_matrix<sycl::sub_group, sycl::half, use::a, M, K> matA;
joint_matrix<sycl::sub_group, sycl::half, use::b, K, N> matB;
joint_matrix<sycl::sub_group, float,  use::accumulator, M, N> matC;
```

Note: accumulator must use `layout::dynamic` (default). A/B can use explicit layout.

### Function Signatures (oneAPI 2026.1)

**Load** — 4 args (no layout parameter!):
```cpp
joint_matrix_load(sg, matA, sycl::multi_ptr<const sycl::half, space>(ptr), stride);
```

**Fill** — 3 args:
```cpp
joint_matrix_fill(sg, matC, 0.0f);
```

**MAD** — 5 args (D = A × B + C):
```cpp
joint_matrix_mad(sg, matD, matA, matB, matC);
```
Note: NOT `joint_matrix_multiply_accumulate(sg, matA, matB, matC)` — that doesn't exist.

**Store** — 5 args (needs layout parameter!):
```cpp
joint_matrix_store(sg, matD, sycl::multi_ptr<float, space>(ptr), stride, layout::row_major);
```

### multi_ptr

Must specify both element type and address space explicitly:
```cpp
sycl::multi_ptr<const sycl::half, sycl::access::address_space::global_space>(ptr)
```

### Sub-group vs Group

- Use `sycl::sub_group` (from `nd_item::get_sub_group()`) for matrix ops
- `sycl::group<1>` (from `nd_item::get_group()`) is the work-group, NOT the sub-group
- Matrix tiles are per-sub-group, not per-work-group

### Supported Types (from xmx/common.hpp)

| Type | Enum |
|------|------|
| u2, s2 | 2-bit integer |
| u4, s4 | 4-bit integer |
| u8, s8 | 8-bit integer |
| bf16 | bfloat16 |
| fp16 | half |
| tf32 | tensorfloat32 |

Accumulator is always F32.

### ESIMD-Level XMX

Lower-level API at `sycl::ext::intel::esimd::xmx::dpas()` — direct systolic engine access.
Header: `<sycl/ext/intel/esimd/xmx/dpas.hpp>`

This is the lower-level path that gives more control over repeat counts, systolic depth, etc.
The `joint_matrix` API is the higher-level abstraction.

## XMX on Battlemage — BLOCKED (2026-07-30)

**Finding:** `ext_intel_matrix` aspect (ID 53) is advertised by driver 1.15.38646+6 on Arc Pro B70,
but the `joint_matrix` API fails at device JIT compilation:

```
error: undefined reference to `__builtin_spriv_OpJointMatrixLoadINTEL_PackedA_PackedB_SG16_16x16_...'
```

The SPIR-V builtin `OpJointMatrixLoadINTEL` is not linked by the driver's JIT compiler.
This means XMX cooperative matrix is **not functional** on this driver + oneAPI 2026.1 combination.

**Workarounds to investigate:**
- Update GPU driver (current: 1.15.38646+6)
- Try ESIMD `dpas()` path (`sycl::ext::intel::esimd::xmx::dpas`) — lower-level, might work
- Wait for newer oneAPI / driver that ships with the SPIR-V builtin
- Use scalar F16 math as fallback (what we'll need for KVarN Phase 2 anyway)

**Impact on KVarN plan:** Phase 2 (native flash attention with XMX) is blocked until this is resolved.
Phase 1 (store + materialize) is unaffected — those use scalar F16/F32 math.

## What Still Needs Testing

- ESIMD `dpas()` path — might work where joint_matrix fails
- Driver update — newer driver might ship the SPIR-V builtin

## Additional Findings (2026-07-30)

### Shape support on B70 (stock oneAPI 2026.1)
- 8×8: rejected ("not supported on this device")
- 16×16: passes shape validation, fails at device compile (OpJointMatrixLoadINTEL)
- 16×32, 32×16, 16×64: rejected ("not supported")

### Working XMX paths on Battlemage

**oneDNN** (`fattn-onednn.cpp` in upstream llama.cpp): works on stock oneAPI.
Gated to BMG-G21/G31 only. Handles XMX internally.

**ESIMD `xmx::dpas`**: works on stock oneAPI 2025.3+ — bypasses `joint_matrix`
entirely. See Bryan Vine's investigation:
https://bryanvine.github.io/turboquant-xpu/2026/04/15/sycl-three-attempts-arc-b70/

**`joint_matrix`**: requires intel/llvm nightly (not stock oneAPI). Stock oneAPI's
`get_matrix_combinations()` doesn't include BMG-G31 shapes. Even on nightly,
performance is 30× slower than fused Triton for flash attention.

### Performance Reality (from Vine's B70 PoCs)
- Custom SYCL flash attention with XMX: **30-60× slower than fused Triton**
- DPAS fires but is NOT the bottleneck — scalar softmax (55% of wall time),
  K/V dequant (~47% each) dominate
- Intel's Triton XPU backend already emits DPAS, so custom DPAS isn't the lever
- Gap progression: scalar SYCL 68× → ESIMD 60× → joint_matrix 30×

### For KVarN
oneDNN won't help (does standard SDPA). Custom KVarN attention needs its own kernel.
ESIMD `xmx::dpas` is the viable path for the matmul portion on stock oneAPI.
But the real bottleneck will be the dequantize path, not the matrix multiply.
