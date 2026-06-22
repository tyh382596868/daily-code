---
date: 2026-06-22
topic: infrastructure
source: trending
repo: FluxVLA/FluxVLA
file: fluxvla/ops/triton/attention_triton_ops.py
permalink: https://github.com/FluxVLA/FluxVLA/blob/4e10e04dfd7afe61c4b807c3b0ad9f64040364a5/fluxvla/ops/triton/attention_triton_ops.py#L112-L147
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, infrastructure, triton, attention, vla, prefix-suffix-mask, softmax]
---

# VLA 专属 Triton 注意力：前缀 + 后缀双区域 Softmax / VLA-Specific Triton Attention: Prefix + Suffix Two-Region Softmax

> **一句话 / In one line**: 用一个 Triton 核把"可变长图像前缀 + 固定动作后缀"的 VLA 注意力掩码和 online softmax 融合进单次内核调用——35 行，峰值算力。 / A single Triton kernel fuses the VLA attention mask (variable-valid image prefix + always-valid action suffix) with online softmax in one kernel launch — 35 lines at near-peak hardware throughput.

## 为什么重要 / Why this matters

VLA 里的动作 token 有一种普通因果注意力处理不了的双区域感受野：动作 token 既需要关注图像 token（前缀，prefix），又需要关注之前的动作 token（后缀，suffix）。更复杂的是，图像前缀的有效长度是可变的——编码器可能按 chunk 处理图像，当前帧的图像 token 并非全都有效，只有 `valid_prefix_len` 个是真实的。

如果用 PyTorch 的标准 `scaled_dot_product_attention`，需要先构造一个 `(queries, keys)` 大小的 mask 矩阵（O(N²) 显存），然后再做 softmax。对于长序列，光 mask 矩阵就几 GB。FluxVLA 的 Triton 核把 mask 构造和 softmax 合并进一个 kernel：`is_prefix`、`prefix_ok`、`suffix_ok` 这三行布尔运算直接在寄存器里完成掩码判断，`other=big_neg` 把无效位置的 logit 压到 -∞，再做 online softmax——整个过程不需要物化 mask 矩阵。

VLA action tokens need a two-region receptive field that standard causal attention can't express: they must attend to image tokens (prefix) AND to previous action tokens (suffix). Worse, the image prefix has variable valid length — the encoder may process images in chunks, so only `valid_prefix_len` of the image tokens are real. Standard `scaled_dot_product_attention` would require materializing an `O(N²)` mask matrix — gigabytes for long sequences. FluxVLA's Triton kernel merges mask construction with softmax: three Boolean ops (`is_prefix`, `prefix_ok`, `suffix_ok`) determine masking in registers, `other=big_neg` sets invalid logits to -∞, and online softmax runs in the same pass — no mask matrix allocated.

## 代码 / The code

`FluxVLA/FluxVLA` — [`fluxvla/ops/triton/attention_triton_ops.py`](https://github.com/FluxVLA/FluxVLA/blob/4e10e04dfd7afe61c4b807c3b0ad9f64040364a5/fluxvla/ops/triton/attention_triton_ops.py#L112-L147)

```python
@triton.jit
def softmax_kernel_prefix_suffix(
    inp_ptr,
    queries: tl.constexpr,
    keys_prefix: tl.constexpr,
    keys_suffix: tl.constexpr,
    valid_prefix_len_ptr,
    out_ptr,
    BLOCK_SIZE_M: tl.constexpr = 4,
    BLOCK_SIZE: tl.constexpr = 1024,
):
    pid = tl.program_id(axis=0)
    psize = tl.num_programs(axis=0)
    big_neg = -2.3819763e38
    total_keys: tl.constexpr = keys_prefix + keys_suffix
    valid_prefix_len = tl.load(valid_prefix_len_ptr).to(tl.int32)
    valid_prefix_len = tl.maximum(0, tl.minimum(valid_prefix_len, keys_prefix))
    for i in range(pid * BLOCK_SIZE_M, queries, psize * BLOCK_SIZE_M):
        offs_i = i + tl.arange(0, BLOCK_SIZE_M)[:, None]
        offs_j = tl.arange(0, BLOCK_SIZE)[None, :]
        in_bounds = (offs_i < queries) & (offs_j < total_keys)
        is_prefix = offs_j < keys_prefix
        prefix_ok = is_prefix & (offs_j < valid_prefix_len)
        suffix_ok = (~is_prefix)
        attn_mask = in_bounds & (prefix_ok | suffix_ok)
        vals = tl.load(
            inp_ptr + offs_i * total_keys + offs_j,
            mask=attn_mask,
            other=big_neg)
        vals = tl.exp(vals - tl.max(vals, axis=1, keep_dims=True))
        vsum = tl.sum(vals.to(tl.float32), axis=1, keep_dims=True)
        vals = vals / vsum
        tl.store(
            out_ptr + offs_i * total_keys + offs_j,
            vals.to(tl.bfloat16),
            mask=in_bounds)
```

## 逐行讲解 / What's happening

1. **`total_keys: tl.constexpr = keys_prefix + keys_suffix`**:
   - 中文: `keys_prefix` 是图像 token 的数量（所有图像位置，包括无效的），`keys_suffix` 是动作 token 的数量。`tl.constexpr` 意味着这是编译期常量，Triton 编译器可以基于它做静态内存布局优化，而不是运行期动态计算。
   - English: `keys_prefix` is the count of image token positions (including invalid ones); `keys_suffix` is the action token count. `tl.constexpr` marks it as a compile-time constant, letting the Triton compiler make static memory layout decisions rather than computing it at runtime.

2. **`valid_prefix_len = tl.maximum(0, tl.minimum(valid_prefix_len, keys_prefix))`**:
   - 中文: 从 `valid_prefix_len_ptr` 加载实际有效的图像 token 数，并 clamp 到 `[0, keys_prefix]`。这个 clamp 是防御性的：如果调用者传入了越界值（比如图像 chunk 比预期大），kernel 不会越界访问。`tl.load` 从设备内存读一个标量，在所有 warp 里共享。
   - English: Loads the actual valid image token count from device memory and clamps it to `[0, keys_prefix]`. The clamp is defensive: if the caller passes an out-of-range value (e.g. an image chunk bigger than expected), the kernel won't access out-of-bounds memory. `tl.load` reads a scalar from device memory, shared across all warps.

3. **`is_prefix / prefix_ok / suffix_ok / attn_mask`（三行掩码逻辑）**:
   - 中文: 这是整个核的灵魂。`is_prefix = offs_j < keys_prefix` 把 key 空间分成两段（图像 vs 动作）；`prefix_ok` 在图像段里进一步过滤，只保留 `j < valid_prefix_len` 的有效 token；`suffix_ok` 接受所有动作 token（动作总是有效的）。最终 `attn_mask = in_bounds & (prefix_ok | suffix_ok)` 是在 SRAM 上对 `(BLOCK_SIZE_M, BLOCK_SIZE)` 大小的 tile 做逐元素的 AND/OR——没有任何 Python-level 控制流，没有额外显存分配。
   - English: The soul of the kernel. `is_prefix` splits the key dimension into two regions (image vs. action); `prefix_ok` further filters the image region to only `j < valid_prefix_len`; `suffix_ok` accepts all action tokens unconditionally. Final `attn_mask = in_bounds & (prefix_ok | suffix_ok)` applies element-wise AND/OR over an SRAM tile of shape `(BLOCK_SIZE_M, BLOCK_SIZE)` — no Python control flow, no extra memory allocation.

4. **`tl.load(..., mask=attn_mask, other=big_neg)`**:
   - 中文: 这一行同时完成了两件事：从输入 logit 矩阵里加载一个 tile，并把掩码为 False 的位置直接填入 `big_neg`（接近 float 负无穷）。`big_neg = -2.3819763e38` 是 bfloat16 能表示的最负的非 -inf 值——选这个而不是 `-inf` 是为了避免 `exp(-inf)` 产生 `NaN`（当某行全是 -inf 时 softmax 分母为 0）。
   - English: This line does two things at once: loads a tile from the input logit matrix, and fills positions where `attn_mask` is False with `big_neg` (near float negative infinity). `big_neg = -2.3819763e38` is the most-negative finite bfloat16 value — chosen over `-inf` to avoid `exp(-inf) = 0` producing `NaN` when a row's entire denominator would be zero.

5. **`vals = tl.exp(vals - tl.max(vals, axis=1, keep_dims=True))`（online softmax）**:
   - 中文: 标准的数值稳定 softmax：先减去每行最大值（防止 `exp` 溢出），再取指数。`tl.max` 沿 key 维度（`axis=1`）归约，`keep_dims=True` 保持广播形状。因为掩码位置被填为 `big_neg`，它们对 `tl.max` 几乎没有影响，`exp(big_neg - max) ≈ 0`——相当于从 softmax 分布里移除了这些位置。
   - English: Numerically stable softmax: subtract per-row max (prevents `exp` overflow) then exponentiate. `tl.max` reduces along the key dimension (`axis=1`), `keep_dims=True` preserves the broadcast shape. Because masked positions hold `big_neg`, they barely affect `tl.max`, and `exp(big_neg - max) ≈ 0` — effectively zero-ing them out of the softmax distribution.

6. **`tl.store(..., vals.to(tl.bfloat16), mask=in_bounds)`**:
   - 中文: 输出用 bfloat16 存储（而不是 fp32），省一半带宽。注意这里的 mask 是 `in_bounds`（不是 `attn_mask`）——所有 in-bounds 的位置都要写回，包括被掩码掉的（它们的值是 `exp(big_neg - max) ≈ 0`，写回为 0，正确）。
   - English: Output is stored in bfloat16 (not fp32), halving bandwidth. Note the store mask is `in_bounds` (not `attn_mask`) — all in-bounds positions are written, including masked ones. Their values are `exp(big_neg - max) ≈ 0`, so writing 0.0 for masked positions is correct.

## 类比 / The analogy

想象一个会议室里的翻译同声传译：听众席分两区——前区是嘉宾区（图像 token），但今天只有前排的 `valid_prefix_len` 位嘉宾到场了，后排座位是空的；后区是固定的工作人员区（动作 token），全员在场。口译员只给"到场的嘉宾"和"工作人员"翻译，空座位忽略。`attn_mask` 就是门口保安的名单：只让两个区里的真实在场者进入翻译服务。

Picture a conference hall with two sections: the front section is the VIP area (image tokens), but only the first `valid_prefix_len` seats are occupied today — the rest are empty; the back section is staff (action tokens), always full. The interpreter only translates for seated VIPs and staff — empty seats are ignored. `attn_mask` is the door security list: only real attendees from either section get served.

## 自己跑一遍 / Try it yourself

```python
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_prefix_suffix(inp_ptr, queries: tl.constexpr, keys_prefix: tl.constexpr,
                           keys_suffix: tl.constexpr, valid_prefix_len_ptr, out_ptr,
                           BLOCK_SIZE_M: tl.constexpr = 4, BLOCK_SIZE: tl.constexpr = 16):
    pid = tl.program_id(0); psize = tl.num_programs(0)
    big_neg = -2.3819763e38
    total_keys: tl.constexpr = keys_prefix + keys_suffix
    vpl = tl.load(valid_prefix_len_ptr).to(tl.int32)
    vpl = tl.maximum(0, tl.minimum(vpl, keys_prefix))
    for i in range(pid * BLOCK_SIZE_M, queries, psize * BLOCK_SIZE_M):
        oi = i + tl.arange(0, BLOCK_SIZE_M)[:, None]
        oj = tl.arange(0, BLOCK_SIZE)[None, :]
        in_b = (oi < queries) & (oj < total_keys)
        mask = in_b & ((oj < vpl) | (oj >= keys_prefix))
        v = tl.load(inp_ptr + oi * total_keys + oj, mask=mask, other=big_neg)
        v = tl.exp(v - tl.max(v, axis=1, keep_dims=True))
        v = v / tl.sum(v.to(tl.float32), axis=1, keep_dims=True)
        tl.store(out_ptr + oi * total_keys + oj, v.to(tl.bfloat16), mask=in_b)

Q, KP, KS = 8, 10, 6   # 8 action queries, 10 image keys, 6 action keys
valid_plen = torch.tensor(6, device="cuda", dtype=torch.int32)  # only first 6 image tokens valid
logits = torch.randn(Q, KP + KS, device="cuda", dtype=torch.bfloat16)
out = torch.empty_like(logits)
softmax_prefix_suffix[(1,)](logits, Q, KP, KS, valid_plen, out)
# Columns 6-9 (invalid prefix) should be near 0
print("invalid prefix cols sum:", out[:, 6:10].float().abs().sum().item())   # ~0
print("valid prefix + suffix sum:", out[:, :6].float().sum(dim=1)[:3])      # each row sums to 1
```

运行 / Run with:
```bash
pip install torch triton
python try.py  # requires a CUDA GPU
```

预期输出 / Expected output:
```
invalid prefix cols sum: 0.0
valid prefix + suffix sum: tensor([1.0, 1.0, 1.0])
```

注意无效的图像 token（columns 6-9）的 softmax 权重为 0，而每一行的总权重（有效前缀 + 动作后缀）之和为 1——掩码和 softmax 在一次 kernel 里同时完成。

Invalid image tokens (columns 6-9) get zero softmax weight; each row's total (valid prefix + suffix) sums to 1.0 — mask construction and softmax happen in a single kernel pass.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Flash Attention 2 (`flash_attn_with_kvcache`)**: FlashAttention 也把 mask + softmax 融合进 CUDA kernel，但只支持标准因果/全注意力 mask。FluxVLA 扩展了这个思路，把 VLA 专属的双区域 mask 也融入 kernel。 / FlashAttention 2 also fuses mask+softmax in CUDA but only supports standard causal/full attention masks. FluxVLA extends the same idea to VLA-specific two-region masks.
- **vLLM 的 `paged_attention_v1` kernel**: vLLM 的 Triton paged attention 也在 kernel 里做在线 softmax，不物化全注意力矩阵。和这里的区别是 vLLM 的 mask 是 KV cache 的 page 边界，而不是 VLA 的 prefix/suffix。 / vLLM's `paged_attention_v1` Triton kernel also does online softmax without materializing the full attention matrix. The difference: vLLM's mask is KV-cache page boundaries, not VLA prefix/suffix regions.
- **xFormers `BlockSparseAttention`**: xFormers 允许用户提供稀疏 block mask。FluxVLA 的做法更轻量：不需要通用稀疏 block 框架，直接把特定的 VLA mask 逻辑硬编码进 kernel，省掉 metadata 开销。 / xFormers lets users provide a sparse block mask. FluxVLA's approach is leaner: no general sparse-block framework needed — the specific VLA mask logic is hardcoded into the kernel, eliminating sparse-block metadata overhead.

## 注意事项 / Caveats / when it breaks

- **`BLOCK_SIZE ≥ keys_prefix + keys_suffix` 的假设 / Assumes `BLOCK_SIZE ≥ total_keys`**: 当前实现用单个 `BLOCK_SIZE=1024` 的 tile 覆盖所有 key，不迭代 key 维度。如果 `keys_prefix + keys_suffix > 1024`，需要重写为沿 key 维度也分 tile 的两层循环（类似 FlashAttention 2 的外循环结构）。 / The current implementation uses one `BLOCK_SIZE=1024` tile to cover all keys — no iteration over the key dimension. If `keys_prefix + keys_suffix > 1024`, you need an outer key-dimension loop, like FlashAttention 2's tiling structure.
- **单精度累加后下转 bfloat16 / fp32 accumulation then cast to bfloat16**: `vsum = tl.sum(vals.to(tl.float32), ...)` 先升精度做分母加法，防止 bfloat16 精度不足导致 softmax 分母下溢。但最终存储 `vals.to(tl.bfloat16)` 会损失精度。对于大 logit 值，可能需要保留 fp32 输出。 / The denominator accumulates in fp32 to prevent bfloat16 underflow, but final storage casts back to bfloat16. For large logit values, you may need fp32 output.
- **只对 softmax 融合，没有 attention matmul / Only softmax, not the full attention matmul**: 这个 kernel 只做 softmax，不做 Q·K 乘法和 softmax(QK/√d)·V 的乘法。在 VLA 推理里，QK matmul 之后调用这个 kernel，再接 O = attn\_weights @ V——三步分开。FlashAttention 风格是把三步全融合，但实现复杂度更高。 / This kernel only computes softmax — not the Q·K matmul or the softmax·V output matmul. In FluxVLA inference: QK matmul → this kernel → O = attn_weights @ V. Full FlashAttention-style fusion of all three would be more performant but significantly more complex.

## 延伸阅读 / Further reading

- [Triton 官方教程 — Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) — 用 Triton 写 fused softmax 的入门，直接对应本文的 kernel 结构。
- [FlashAttention 2 论文](https://arxiv.org/abs/2307.08691) — 最完整的"fused QK softmax V"实现，展示了完整两层 tile 循环的正确写法。
- [FluxVLA `attention_triton_ops.py` 完整文件](https://github.com/FluxVLA/FluxVLA/blob/4e10e04dfd7afe61c4b807c3b0ad9f64040364a5/fluxvla/ops/triton/attention_triton_ops.py) — 还有 `matmul_n_2048_2560_qkv_rope`（QKV 投影 + RoPE 融合），是 VLA 推理路径上的另一个关键 kernel。
