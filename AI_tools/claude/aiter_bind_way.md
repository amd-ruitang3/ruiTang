# aiter FFI Binding — De-torch Refactoring Guide

> Goal: Remove `torch` dependency from `csrc/kernels/` C side; use **ctypes** as the sole Python↔C bridge.

---

## 1. Binding Choice: ctypes (not pybind11, not TVM FFI)

| Option | Verdict | Reason |
|--------|---------|--------|
| **ctypes** | **Chosen** | `aiter_tensor_t` is a 160-byte POD struct — ctypes maps it directly. Zero extra dependency (stdlib). No torch headers → faster hipcc compile. |
| pybind11 | Rejected | Existing bindings are all `torch::Tensor` params; once torch is removed pybind adds no value. |
| TVM FFI | Rejected | Too heavy (drags in TVM runtime). We already have our own `AiterTensor`; `DLTensor` is redundant. |

Reference impl: `csrc/py_itfs_cu/` asm kernels already use ctypes + AiterTensor.

---

## 2. Type Naming Convention

| Name | Layer | Style | Role |
|------|-------|-------|------|
| `aiter_tensor_t` | C ABI / ctypes | snake_case + `_t` (like `hipStream_t`) | POD struct, crosses the FFI boundary |
| `AiterTensor` | C++ internal | PascalCase | RAII class, inherits `aiter_tensor_t` |

Case alone distinguishes the two layers.

---

## 3. AiterTensor RAII Class

`AiterTensor` inherits `aiter_tensor_t` and adds:

- **Lifecycle**: constructor/destructor auto-manage GPU memory (`hipMalloc`/`hipFree`)
- **Factory methods**: `AiterTensor::empty(...)`, `AiterTensor::zeros(...)`
- **Move-only**: copy deleted, move allowed

```cpp
auto ws  = AiterTensor::empty({1024},  AITER_DTYPE_u8,   input->device_id, stream);
auto tmp = AiterTensor::zeros({M, N},  AITER_DTYPE_fp32, input->device_id, stream);
launch<<<grid, block, 0, stream>>>(..., ws.data_ptr(), tmp.data_ptr());
// auto-freed when ws/tmp go out of scope
```

Because of inheritance, `AiterTensor*` is implicitly convertible to `aiter_tensor_t*`.

---

## 4. Architecture: Python manages memory, C computes

```
Python (torch.Tensor)
    │  torch_to_aiter()          # zero-copy
    ▼
aiter_tensor_t                   # ctypes.Structure, 160 bytes
    │  lib.func(byref(at), stream)
    ▼
C: extern "C" void func(aiter_tensor_t*, hipStream_t)
    │  ptr->data_ptr(), ptr->size(0), ...
    ▼
HIP kernel launch
```

### Python-side responsibilities

- Allocate **all** input/output tensors via `torch.empty` / `torch.zeros`
- Convert to `aiter_tensor_t` via `torch_to_aiter()`
- Obtain stream: `torch.cuda.current_stream().cuda_stream`
- Obtain RNG seed/offset
- Type dispatch (choose function by dtype)

### C-side responsibilities (zero torch dependency)

- Accept only `aiter_tensor_t*` + scalar args
- `HipDeviceGuard(tensor->device_id)` to set device
- Allocate **temporary** workspace via `AiterTensor::empty/zeros` (RAII auto-free)
- Use `AITER_CHECK` (not `TORCH_CHECK`)
- Launch kernels via `hipLaunchKernelGGL`
- **Never** allocate tensors that return to Python

---

## 5. Torch Dependency Replacement Map

### 5.1 Drop-in replacements (text substitution)

| torch API | Replacement | Notes |
|-----------|-------------|-------|
| `torch::Tensor` param | `aiter_tensor_t*` | AiterTensor covers all accessors |
| `TORCH_CHECK` | `AITER_CHECK` | Already exists |
| `c10::DeviceGuard` | `HipDeviceGuard` | Already exists |
| `at::hip::getCurrentHIPStream` | Pass `hipStream_t` as function arg | Signature change |

### 5.2 Need implementation

| torch API | Replacement | Notes |
|-----------|-------------|-------|
| `AT_DISPATCH` / `VLLM_DISPATCH` | `AITER_DISPATCH_DTYPE` macro | `switch` on `AiterDtype` enum |
| `torch::empty/zeros` (workspace) | `AiterTensor::empty/zeros` | RAII |
| `torch::empty/zeros` (output) | Python pre-allocates, pass in | Convention change |
| `torch::from_blob` + `.to()` | `hipMemcpyAsync` | Host→device copy |
| `.contiguous()` | Python guarantees or `AITER_CHECK(is_contiguous)` | |
| `.to(CPU)` / `.copy_()` | `hipMemcpy` / `hipMemcpyAsync` | |
| `torch::promote_types` | Python decides output dtype, passes in | |
| `at::ScalarType` | `AiterDtype` enum | |

### 5.3 Special cases

| Case | Solution |
|------|----------|
| **Philox RNG** (`sample_kernels.cu` only) | Python passes seed/offset; C side self-implements Philox (~20 LOC) |
| **`vector<Tensor>` return** (mla/metadata) | Python pre-allocates all outputs; C fills data, returns `void` |
| **`vector<Tensor>` input** (cache, all_reduce) | Python packs pointer array: `torch.tensor([t.data_ptr() for t in caches])`; C receives `int64_t*` |

### 5.4 Deferred

- `custom_all_reduce.cu` / `quick_all_reduce.cu` — deeply coupled to `torch.distributed` (IPC handles, buffer registration). Migrate last.

---

## 6. Stream Passing (TBD)

| Option | Description | Status |
|--------|-------------|--------|
| **Current** | `ctypes.c_void_p(stream)` passed per call | In use |
| A | Store in `aiter_tensor_t.stream` field | Rejected — stream is not a tensor property |
| B | Separate `AiterExecCtx` struct (extensible: workspace, events, ...) | Under discussion |
| C | Keep current per-call arg | Default fallback |

Not finalized yet.

### Stream 背景知识

- `hipStream_t` = `typedef struct ihipStream_t*`，定义在 `/opt/rocm/include/hip/hip_runtime_api.h`
- Stream 是 GPU 命令队列（软件抽象），同一 stream 内顺序执行，不同 stream 可并发
- 每个 GPU 可创建任意多个 stream，底层映射到硬件 HW Queue（数量有限，runtime 复用）
- Default stream (NULL) 全局串行，非 default stream 支持异步并发

### PyTorch `getCurrentHIPStream` 原理

PyTorch 的实现（`c10/cuda/CUDAStream.cpp`）核心是：
- `thread_local` 数组存每个 device 的"当前 stream"
- 默认值是 default stream (NULL)
- 额外维护 32 个 stream 的 pool，round-robin 分配

### 纯 HIP 替代方案（已实现 `aiter_stream.h`，暂未启用）

已在 `csrc/include/aiter_stream.h` 实现轻量版，纯 HIP 无 torch 依赖：
```cpp
namespace aiter {
// thread_local per-device stream 数组，默认 NULL (hipStreamDefault)
hipStream_t getCurrentHIPStream(int device_id);
void setCurrentHIPStream(int device_id, hipStream_t stream);
}
```

当前策略：C 端通过函数参数接收 `hipStream_t`，由 Python 端从 torch 获取后传入。
将来完全去 torch 后，Python 可通过 `hip-python` 包或 `aiter::getCurrentHIPStream` 获取。

---

## 7. Migration Difficulty (easy → hard)

| # | Item | Change type |
|---|------|-------------|
| 1 | `TORCH_CHECK` → `AITER_CHECK` | Text replace |
| 2 | `DeviceGuard` → `HipDeviceGuard` | Text replace |
| 3 | Tensor accessors → `aiter_tensor_t` methods | Text replace |
| 4 | `getCurrentStream` → param | Signature change |
| 5 | `AT_DISPATCH` → `switch(dtype)` | Rewrite macro expansion |
| 6 | `torch::empty` → `AiterTensor` / Python | Calling convention change |
| 7 | RNG (Philox) | Independent implementation |
| 8 | Communication (all_reduce) | Keep torch dep (too coupled) |

---

## 8. Code Style Rules

- All modified Python files must be formatted with `python3 -m black <file>` before committing.

---

## 9. 实战案例：`quant_kernels.cu` 去 torch 化

> 以下记录 `csrc/kernels/quant_kernels.cu`（~1970 行，10 个 host 函数，14 个 dispatch 宏）的完整去 torch 化过程。
> 参考实现：`activation_kernels.cu` / `activation.h` / `activation_pybind.cu`（已完成去 torch 化）。

### 9.1 涉及文件

| 文件 | 改动类型 |
|------|----------|
| `csrc/include/quant.h` | 函数声明：类型替换 |
| `csrc/kernels/quant_kernels.cu` | 函数实现：类型、宏、guard、stream、dispatch 全面替换 |
| `csrc/include/rocm_ops.hpp` | QUANT_PYBIND 宏：可选参数默认值 |
| `csrc/pybind/quant_pybind.cu` | 新增 `aiter_stream.h` include 和 `AITER_SET_STREAM_PYBIND` |
| `aiter/ops/quant.py` | `compile_ops` 加 `develop=True` |

### 9.2 `quant.h` 类型映射

```cpp
// Before
#include <torch/extension.h>

// After
#include "aiter_tensor.h"
#include <optional>
```

| torch 类型 | aiter 类型 |
|-----------|-----------|
| `torch::Tensor&` | `aiter_tensor_t&` |
| `torch::Tensor const&` | `const aiter_tensor_t&` |
| `std::optional<torch::Tensor>` | `std::optional<aiter_tensor_t>` (保持 `= std::nullopt` 默认值) |

> **注意**：可选 tensor 保留 `std::optional<aiter_tensor_t>` 而非裸指针 `const aiter_tensor_t*`。
> pybind11 兼容性：`rocm_ops.hpp` 已 include `<pybind11/stl.h>`，`aiter_tensor_t` 通过 `pybind11::class_` 注册，
> `std::optional` 的 Python `None` → `std::nullopt` 转换自动生效。

### 9.3 `quant_kernels.cu` 逐项替换

#### 9.3.1 Includes

```diff
- #include "dispatch_utils.h"
- #include "py_itfs_common.h"
- #include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
+ #include "aiter_dispatch.h"
+ #include "aiter_stream.h"
+ #include "quant.h"
```

保留：`aiter_hip_common.h`, `aiter_opus_plus.h`, `rocprim/rocprim.hpp`, `<hipcub/hipcub.hpp>`

#### 9.3.2 Device Guard & Stream

```diff
- const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(input));
- const hipStream_t stream = at::hip::getCurrentHIPStream();
+ HipDeviceGuard device_guard(input.device_id);
+ const hipStream_t stream = aiter::getCurrentHIPStream();
```

#### 9.3.3 Dispatch 宏

```diff
- AITER_DISPATCH_FLOATING16_TYPES(input.scalar_type(), "kernel_name", [&] {
+ AITER_DISPATCH_FLOATING16_TYPES_rmTorch(input.dtype(), "kernel_name", [&] {

- VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "kernel_name", [&] {
+ VLLM_DISPATCH_FLOATING_TYPES_rmTorch(input.dtype(), "kernel_name", [&] {
```

#### 9.3.4 t2opus → hip2opus（关键坑点）

**问题**：`_rmTorch` dispatch 宏定义 `scalar_t` 为 HIP 包装类型（`__half`, `hip_bfloat16`），
但 opus 内核模板需要 opus 原始标量类型（`opus::fp16_t` = `_Float16`, `opus::bf16_t` = `unsigned short`）。

类型对应关系（`opus/opus.hpp:870-891` 的 `REGISTER_DTYPE` 宏）：

| `_rmTorch` dispatch `scalar_t` | opus 类型 (`opus::fp16_t` 等) | 底层实际类型 |
|-------------------------------|------------------------------|-------------|
| `__half` (HIP struct) | `opus::fp16_t` | `_Float16` (clang < 20) / `__fp16` (clang >= 20) |
| `hip_bfloat16` (HIP struct) | `opus::bf16_t` | `unsigned short` (clang < 20) / `__bf16` (clang >= 20) |
| `float` | `opus::fp32_t` | `float` |

**两者不同！** `__half` 是包装 `_Float16` 的 struct，不能直接用于 `opus::vector_t<T, N>`（ext_vector_type 只接受标量类型）。

**解决方案**：用 `aiter::hip2opus<T>`（定义在 `aiter_opus_plus.h:629-635`）替换原来的 `t2opus<T>`：

```cpp
// aiter_opus_plus.h
template <typename T> struct hip2opus;
template <> struct hip2opus<float>         { using type = opus::fp32_t; };
template <> struct hip2opus<__half>        { using type = opus::fp16_t; };
template <> struct hip2opus<hip_bfloat16>  { using type = opus::bf16_t; };
template <> struct hip2opus<uint8_t>       { using type = opus::fp8_t; };
template <> struct hip2opus<int8_t>        { using type = opus::i8_t; };
```

在每个 dispatch body 中：

```diff
  AITER_DISPATCH_FLOATING16_TYPES_rmTorch(input.dtype(), "kernel", [&] {
-     using input_dtype = typename t2opus<scalar_t>::type;
+     using input_dtype = typename aiter::hip2opus<scalar_t>::type;
      aiter::kernel<input_dtype, ...><<<grid, block, 0, stream>>>(
          ...,
          reinterpret_cast<input_dtype*>(input.data_ptr()),
          ...);
  });
```

**切勿**直接删除 `input_dtype` 把 `scalar_t` 传给 kernel 模板——会导致编译失败或行为错误。

#### 9.3.5 Dtype 比较

| torch | aiter |
|-------|-------|
| `torch_fp8` | `AITER_DTYPE_fp8` |
| `torch::kInt8` | `AITER_DTYPE_i8` |
| `torch_fp4x2` / `torch::kFloat4_e2m1fn_x2` | `AITER_DTYPE_fp4x2` |
| `torch::kUInt8` | `AITER_DTYPE_u8` |

#### 9.3.6 TORCH_CHECK → AITER_CHECK

全文替换，注意错误消息中 `out.dtype()` 需改为 `AiterDtype_to_str(out.dtype())`（因为 `aiter_tensor_t::dtype()` 返回枚举而非可直接打印的对象）。

#### 9.3.7 data_ptr\<T\>() → reinterpret_cast

```diff
- scales.data_ptr<float>()
+ reinterpret_cast<float*>(scales.data_ptr())
```

`aiter_tensor_t::data_ptr()` 返回 `void*`。

#### 9.3.8 可选 tensor 访问

```diff
- scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr
+ scale_ub.has_value() ? reinterpret_cast<float*>(scale_ub->data_ptr()) : nullptr
```

### 9.4 `rocm_ops.hpp` QUANT_PYBIND 宏

可选参数默认值保持 `std::nullopt`：

```cpp
py::arg("scale_ub")        = std::nullopt,
py::arg("num_rows")        = std::nullopt,
py::arg("smooth_scale_map") = std::nullopt,
// ...
```

### 9.5 `quant_pybind.cu`

```cpp
#include "rocm_ops.hpp"
#include "aiter_stream.h"
#include "quant.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND
    QUANT_PYBIND;
}
```

### 9.6 `aiter/ops/quant.py`

所有 `@compile_ops("module_quant")` 改为 `@compile_ops("module_quant", develop=True)`。

`develop=True` 的作用（见 `aiter/jit/core.py:1600-1631`）：
1. 调用时自动执行 `torch_to_aiter_pybind(a)` 将 `torch.Tensor` 转为 `aiter_tensor_t`
2. 自动调用 `module._set_current_hip_stream()` 设置当前 HIP stream

### 9.7 验证方法

```bash
# 1. 确认无 torch 残留
grep -rn "torch\|AT_DISPATCH\|TORCH_CHECK\|t2opus\|scalar_type\|data_ptr<" csrc/kernels/quant_kernels.cu
# 应返回 0 行

# 2. 确认 hip2opus 覆盖
grep -c "input_dtype.*hip2opus" csrc/kernels/quant_kernels.cu
# 应等于 dispatch 位置数

# 3. 编译
python setup.py build_ext

# 4. 测试
python op_tests/test_quant.py
python op_tests/test_smoothquant.py
python op_tests/test_moe_sorting_mxfp4.py
```

### 9.8 踩坑总结

| 坑 | 描述 | 解决 |
|----|------|------|
| **t2opus 不能简单删除** | `_rmTorch` 的 `scalar_t`（`__half`/`hip_bfloat16`）≠ opus 类型（`_Float16`/`unsigned short`），kernel 用了 `opus::vector_t` 等 ext_vector_type，传错类型会编译失败 | 用 `aiter::hip2opus<scalar_t>::type` 替代 `t2opus<scalar_t>::type` |
| **遗漏 `_rmTorch` 后缀** | 个别 dispatch 宏调用漏改为 `_rmTorch` 版本，编译时报 `scalar_type()` 未定义 | 全文搜索 `AITER_DISPATCH_FLOATING16_TYPES(` 确认无遗漏 |
| **develop=True 必须设置** | 不设置 `develop=True` 的 `compile_ops`，Python 端不会做 tensor 转换和 stream 设置 | 参考 `activation.py` 的模式 |
| **可选 tensor 用 `std::optional`** | 裸指针 `const aiter_tensor_t*` 虽然能工作，但 `std::optional<aiter_tensor_t>` 更符合 C++ 惯例，且 pybind11 自动支持 `None` → `std::nullopt` | 保持 `std::optional`，pybind 默认值用 `std::nullopt` |
