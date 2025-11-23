# 真正的 LLaMA/Transformer 推理计算图详解

## 现有实现的问题

我们之前的实现过于简单：

``` cpp
// 过于简单的实现
model.output = ggml_dup(ctx_compute, model.input);  // 直接复制输入
```

## 真正的 Transformer 架构

### 1. **完整的推理流程**

```
输入 Token IDs → Embedding 查找 → 位置编码 →
多层 Transformer Block → 输出归一化 → LM Head → Logits → 采样 → 输出 Token
```

### 2. **Transformer Block 结构**

每个 Transformer Block 包含：

#### 2.1 Multi-Head Self-Attention

``` cpp
// 1. Layer Normalization
x_norm = ggml_rms_norm(ctx, x, model.layers[il].attention_norm);

// 2. 计算 Q, K, V 矩阵
Q = ggml_mul_mat(ctx, model.layers[il].wq, x_norm);
K = ggml_mul_mat(ctx, model.layers[il].wk, x_norm);
V = ggml_mul_mat(ctx, model.layers[il].wv, x_norm);

// 3. 应用旋转位置编码 (RoPE)
Q = ggml_rope_custom(ctx, Q, rope_params);
K = ggml_rope_custom(ctx, K, rope_params);

// 4. 注意力计算
scores = ggml_mul_mat(ctx, K, Q);  // Q @ K^T
scores = ggml_scale(ctx, scores, 1.0f/sqrt(head_dim));  // 缩放
scores = ggml_soft_max(ctx, scores);  // Softmax
attn_out = ggml_mul_mat(ctx, scores, V);  // 加权求和

// 5. 输出投影
attn_out = ggml_mul_mat(ctx, model.layers[il].wo, attn_out);

// 6. 残差连接
x = ggml_add(ctx, x, attn_out);
```

#### 2.2 Feed-Forward Network (SwiGLU)
```cpp
// 1. Layer Normalization
x_norm = ggml_rms_norm(ctx, x, model.layers[il].ffn_norm);

// 2. 门控前馈网络
gate = ggml_mul_mat(ctx, model.layers[il].w1, x_norm);  // gate projection
up   = ggml_mul_mat(ctx, model.layers[il].w3, x_norm);  // up projection
gate = ggml_silu(ctx, gate);                            // SiLU activation
gated = ggml_mul(ctx, gate, up);                        // 门控机制
ffn_out = ggml_mul_mat(ctx, model.layers[il].w2, gated); // down projection

// 3. 残差连接
x = ggml_add(ctx, x, ffn_out);
```

### 3. **关键改进点**

#### 3.1 Token Embedding 查找
```cpp
// 真正的实现：根据 token ID 查找 embedding
struct ggml_tensor * inpL = ggml_get_rows(ctx, model.tok_embeddings, tokens);

// 我们的简化版本（因为输入是float而不是整数token IDs）
cur = ggml_mul_mat(ctx_compute, token_embeddings, cur);
```

#### 3.2 位置编码 (RoPE)
```cpp
// 旋转位置编码 - LLaMA的核心特性
Q = ggml_rope_custom(ctx, ggml_reshape_3d(ctx, Q, n_embd_head, n_head, N),
                     inp_pos, rope_freq_base, rope_freq_scale);
K = ggml_rope_custom(ctx, ggml_reshape_3d(ctx, K, n_embd_head, n_head, N),
                     inp_pos, rope_freq_base, rope_freq_scale);
```

#### 3.3 RMS Layer Normalization
```cpp
// LLaMA 使用 RMS Norm 而不是标准 Layer Norm
x = ggml_rms_norm(ctx, x, model.layers[il].attention_norm);
x = ggml_mul(ctx, x, model.layers[il].attention_norm_weight);
```

#### 3.4 KV Cache (推理优化)
```cpp
// 缓存 Key 和 Value 以提高推理速度
struct ggml_tensor * k_cache = model.kv_cache.k_l[il];
struct ggml_tensor * v_cache = model.kv_cache.v_l[il];

// 更新缓存
k_cache = ggml_set_1d(ctx, k_cache, K, cache_offset);
v_cache = ggml_set_1d(ctx, v_cache, V, cache_offset);
```

## 我们新实现的改进

### 1. **结构化张量识别**
- 智能识别模型中的 Transformer 组件
- 按照真实架构的命名约定查找权重

### 2. **多步骤推理流程**
- Token Embedding → Attention → FFN → Output
- 每个步骤都有适当的命名和日志

### 3. **残差连接**
```cpp
// 注意力的残差连接
cur = ggml_add(ctx_compute, attention_input, attention_out);

// FFN 的残差连接
cur = ggml_add(ctx_compute, ffn_input, ffn_out);
```

### 4. **激活函数**
```cpp
// 使用 SiLU/Swish 激活函数（LLaMA的标准）
gate = ggml_silu(ctx_compute, gate);
```

## 与 llama.cpp 的对比

### llama.cpp 的完整实现特点：
1. **多层处理**: 32-80层 Transformer blocks
2. **KV缓存**: 优化推理速度
3. **量化支持**: 4-bit, 8-bit 量化
4. **批处理**: 同时处理多个序列
5. **Memory mapping**: 高效的模型加载
6. **RoPE**: 旋转位置编码
7. **Group Query Attention**: 优化的注意力机制

### 我们的简化实现：
1. **单层处理**: 演示架构原理
2. **基本操作**: 关键的 Transformer 组件
3. **教育目的**: 理解计算图构建
4. **可扩展**: 可以添加更多层和优化

## 下一步改进方向

1. **多层支持**: 循环处理所有 Transformer 层
2. **正确的数据类型**: 使用整数 token IDs 而不是 float
3. **KV缓存**: 实现推理优化
4. **位置编码**: 添加 RoPE
5. **采样策略**: 实现 top-k, top-p 采样
6. **批处理**: 支持 batch 推理

这个新的实现虽然仍然是简化版本，但更接近真实的 Transformer 架构，展示了现代 LLM 推理的核心组件和计算流程。
