# 通过 nano-vllm 学习推理引擎

## 项目概述

nano-vllm 是 vLLM 的精简实现，代码量小但覆盖了推理引擎的核心机制（PagedAttention、continuous batching、KV cache 管理、CUDA Graph 等），非常适合用来学习推理引擎原理。

---

## 模型定义：`nanovllm/models/qwen3.py`

### 整体结构

文件定义了 Qwen3 模型的四层抽象，自底向上组装：

```
Qwen3ForCausalLM              # 最外层入口
  └── Qwen3Model               # Transformer 主干
        ├── VocabParallelEmbedding     # token → hidden_states
        ├── Qwen3DecoderLayer × N      # 重复 N 层
        │     ├── RMSNorm (input_layernorm)
        │     ├── Qwen3Attention        # 自注意力
        │     ├── RMSNorm (post_attention_layernorm)
        │     └── Qwen3MLP              # 前馈网络 (SwiGLU)
        └── RMSNorm (final norm)
```

所有类都是**具体实现**，没有抽象基类或工厂模式。要支持新模型就照着写一个新文件，复用 `layers/` 里的公共组件。

### CausalLM 的含义

CausalLM = Causal Language Model（因果语言模型）：生成第 i 个 token 时，只能看到前面的 token（1 到 i-1），不能看到后面的。通过 causal mask（下三角矩阵）实现。GPT、LLaMA、Qwen 这类自回归生成模型都是 CausalLM。

---

## 各组件详解

### Qwen3Attention（自注意力）

**构造：**
- 张量并行切分：`num_heads` 和 `num_kv_heads` 按 TP size 均分到各 GPU（GQA：K/V 头数少于 Q 头数，多个 Q 头共享同组 K/V）
- `qkv_proj`：Q/K/V 三个投影合并成一个 `QKVParallelLinear`，减少 kernel launch 开销
- `o_proj`：输出投影，`RowParallelLinear`，配合列并行完成一次 all-reduce
- `rotary_emb`：RoPE 旋转位置编码，旋转 Q/K 向量让注意力分数包含相对位置信息
- `q_norm / k_norm`：Qwen3 特点——无 QKV bias 时对 Q/K 做 RMSNorm（QK-Norm），稳定训练
- `scaling = head_dim ** -0.5`：Attention 公式中的 `1/√d_k`，防止点积过大导致 softmax 饱和

**前向：**
```
hidden_states → qkv_proj → split(Q, K, V) → reshape(view, 零开销)
  → [可选] q_norm, k_norm
  → RoPE(positions, Q, K)
  → Attention(Q, K, V)        # 内部封装了 prefill(flash attn) / decode(paged attn)
  → o_proj → output
```

### Qwen3MLP（前馈网络）

SwiGLU 结构：
```
x → gate_up_proj → [gate, up] → SiLU(gate) * up → down_proj → output
```
- `gate_up_proj`：两个矩阵合并成一个 `MergedColumnParallelLinear`
- SwiGLU 比 ReLU/GELU 效果更好，有可学习的门控机制

### Qwen3DecoderLayer（单层 Transformer）

Pre-Norm 架构 + 残差连接。关键优化是 **fused add+norm**：

```python
# 第一层 (residual is None)
hidden_states, residual = self.input_layernorm(hidden_states), hidden_states

# 后续层：RMSNorm 内部做 hidden_states + residual 再 norm，一次 kernel 搞定
hidden_states, residual = self.input_layernorm(hidden_states, residual)
```

正常写法需要两次显存读写（先 add 再 norm），融合后一次搞定，对 memory-bound 的 LLM 推理很关键。

### Qwen3Model（Transformer 主干）

```python
hidden_states = self.embed_tokens(input_ids)   # token ID → 向量
for layer in self.layers:                       # 逐层过
    hidden_states, residual = layer(positions, hidden_states, residual)
hidden_states, _ = self.norm(hidden_states, residual)  # 最后一次 add+norm
```

### Qwen3ForCausalLM（最外层）

**`packed_modules_mapping`：** 告诉权重加载器如何把 HuggingFace 分开的 q/k/v_proj 权重合并到 qkv_proj 里。

**`tie_word_embeddings`：** 输入 embedding 和输出 lm_head 共享同一份权重：
```python
self.lm_head.weight.data = self.model.embed_tokens.weight.data
```
- `.weight` 是 `nn.Parameter`（PyTorch 的可学习参数）
- `.data` 是底层的 `torch.Tensor`
- 用 `.data` 赋值让两者**指向同一块显存**（共享，不是复制），同时绕过 nn.Parameter 的注册机制

**`forward` 与 `compute_logits` 分离：** decode 时只需对最后一个 token 算 logits，不必对所有 token 都算。

---

## 模型如何被启动

`Qwen3ForCausalLM` 在 `engine/model_runner.py` 中被实例化：

```
ModelRunner.__init__()
  ├── 1. dist.init_process_group()        # 初始化分布式环境
  ├── 2. Qwen3ForCausalLM(hf_config)      # 实例化模型（硬编码，非 registry）
  ├── 3. load_model(model, path)          # 加载 safetensors 权重
  ├── 4. Sampler()                        # 创建采样器
  ├── 5. warmup_model()                   # dummy forward，预热
  ├── 6. allocate_kv_cache()              # 分配 KV cache
  └── 7. capture_cudagraph()              # 录制 CUDA Graph
```

### 实例化是立即连锁执行的

`Qwen3ForCausalLM(hf_config)` 调用时，所有 `__init__` 连锁执行，每个 `nn.Parameter` 的显存都立即分配。等 `__init__` 走完，整个网络骨架已存在（参数是随机值），然后 `load_model()` 才填入真实权重。

---

## HuggingFace 的角色

nano-vllm 只使用 HuggingFace 的两样东西：

| 来源 | 作用 |
|------|------|
| `config.json`（通过 AutoConfig） | 提供结构超参数（hidden_size, num_layers 等） |
| `*.safetensors` | 提供训练好的权重数值 |

**模型的推理代码完全自己实现**，替换为推理优化组件：

| HuggingFace 原版 | nano-vllm 替换为 |
|-----------------|-----------------|
| 标准 SDPA | PagedAttention（flash_attn） |
| 普通 nn.Linear | 张量并行 Column/RowParallelLinear |
| 三个独立 Q/K/V Linear | 合并 QKVParallelLinear |
| 两个独立 Gate/Up Linear | 合并 MergedColumnParallelLinear |
| 无 KV cache | 引擎管理的 paged KV cache |

---

## 默认配置

- `max_model_len`：默认 4096（`nanovllm/config.py:11`），运行时取 `min(用户设置, 模型的 max_position_embeddings)`
- `max_num_batched_tokens`：16384
- `max_num_seqs`：512
