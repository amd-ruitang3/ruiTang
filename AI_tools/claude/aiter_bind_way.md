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
