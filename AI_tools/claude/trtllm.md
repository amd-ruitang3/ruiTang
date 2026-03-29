# TensorRT-LLM 架构笔记

## 1. 跨语言调用：Nanobind（非 pybind11）

TensorRT-LLM 使用 **Nanobind** 实现 C++ 与 Python 的跨语言调用。

- Nanobind 是 pybind11 同一作者（Wenzel Jakob）开发的下一代替代品，更轻量、编译更快、生成的二进制更小
- 绑定代码位于 `cpp/tensorrt_llm/nanobind/`：
  - `bindings.cpp` — 顶层绑定入口
  - `batch_manager/` — BatchManager、KV Cache Manager、Scheduler 等核心组件的绑定
  - `common/` — 自定义类型转换器和异常处理

## 2. 多后端架构

TensorRT-LLM 不是传统意义上的"跨框架"（不支持 TensorFlow/JAX），而是支持**多后端执行**：

| 后端 | 状态 | 执行路径 |
|------|------|----------|
| **PyTorch** | 默认 | 直接用 PyTorch 执行模型 |
| **TensorRT** | Legacy | 将模型编译为 TensorRT engine 执行 |
| **AutoDeploy** | Beta | torch.export + 图变换，自动优化 |

### 架构示意

```
HuggingFace Model → LLM API (统一入口)
                        |
              +---------+---------+
          TorchLlmArgs  |    TrtLlmArgs
          (PyTorch)     |    (TensorRT)
              |    AutoDeploy     |
              +---------+---------+
                        |
            共享 C++ 核心 (via Nanobind)
         Scheduler → BatchManager → KV Cache
              Decoder → Sampling
```

- **统一 Python API**：`LLM` 类是统一入口，通过 `TorchLlmArgs` 或 `TrtLlmArgs` 选择后端
- **共享 C++ 核心**：调度、批处理、KV Cache 管理、解码/采样等性能关键组件用 C++ 实现，所有后端共用
- **模型层各自实现**：PyTorch 后端模型在 `tensorrt_llm/_torch/models/`，TensorRT 后端在 `tensorrt_llm/models/`

## 3. 自定义 Tensor 抽象与编译前端

TensorRT-LLM 的 TensorRT 后端维护了一套**自定义 Tensor + 算子体系**（`tensorrt_llm/functional.py`），实现编译层面的"跨框架"。

### 核心原理

`functional.py` 中的 `Tensor` 类是 **TensorRT ITensor 的包装**，本质是计算图节点，不是运行时张量：

```python
class Tensor:
    self.trt_tensor = ...                       # TensorRT 计算图节点
    self._network = weakref.ref(default_net())  # 所属的 TensorRT Network
```

### 类 PyTorch 风格算子

提供了 `matmul`, `softmax`, `gelu`, `layer_norm`, `rms_norm`, `concat`, `gather`, `where`, `embedding` 等算子。这些函数**不执行计算**，而是向 `trt.INetworkDefinition` 插入算子节点，构建计算图。

### 编译流程

```
用类 PyTorch API 写模型代码
        ↓
调用 functional.py 中的 matmul/softmax/...
        ↓
底层调用 trt.INetworkDefinition.add_layer(...)
构建 TensorRT 计算图
        ↓
TensorRT 编译优化（算子融合、量化、内存规划）
        ↓
生成 TensorRT Engine 执行
```

### 小结

通过这套自定义 Tensor 抽象，模型开发者用**类 PyTorch 语法**写代码，但实际**编译到 TensorRT 引擎**执行。这是 TensorRT 后端（Legacy）的做法。当前默认的 PyTorch 后端则直接用原生 PyTorch 执行，不经过此图构建体系。
