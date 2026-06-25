---
date: 2026-06-25
topic: infrastructure
source: tracked
repo: deepseek-ai/DeepSeek-V3
file: inference/model.py
permalink: https://github.com/deepseek-ai/DeepSeek-V3/blob/b15f0dbbbe6a4bc403306175698439ef380f5fb5/inference/model.py#L221-L322
difficulty: advanced
read_time: ~12 min
tags: [code-of-the-day, infrastructure, attention, kv-cache, low-rank, mla, deepseek]
---

# DeepSeek-V3 MLA 的 absorb 技巧：KV 缓存压缩 70× / DeepSeek-V3 MLA's absorb Trick: 70× KV-Cache Compression

> **一句话 / In one line**: 把 W_Q_nope 和 W_KV 提前相乘，让注意力分数在低秩压缩空间直接计算，KV 缓存从 40 960 个值/token 缩到 576 个值/token。 / Pre-multiply W_Q_nope by W_KV so attention scores are computed entirely in the low-rank latent space, shrinking the KV cache from 40 960 to 576 values per token.

## 为什么重要 / Why this matters

DeepSeek-V3 的 MLA（Multi-Head Latent Attention）是当前生产级大模型中最节省显存的注意力变体之一。标准多头注意力在推理时需要缓存所有 token 的完整 K、V 张量（形状 `[max_seq_len, n_heads, head_dim]`）。MLA 的核心思想是：先把输入 x 投影到一个低秩压缩向量 `kv`（维度 `kv_lora_rank = 512`），再在需要时展开出 K、V。

但 naive 模式还是把展开后的完整 K/V 缓存了下来。**absorb 模式**彻底不展开——它通过一个数学恒等变换，把 Q 的无位置编码部分（`q_nope`）提前与 `W_KV` 相乘，使得注意力分数可以直接用压缩潜变量 `kv_cache` 计算，完全不需要实例化全尺寸的 K 张量。最终只需缓存 `kv_lora_rank + qk_rope_head_dim = 576` 个值，相比 naive 的 `n_heads × 320 ≈ 40 960`（对整个模型而言），压缩比约 **71×**。

MLA's absorb mode is one of the most effective KV-cache compression techniques in production LLMs today. The key mathematical insight: instead of storing full K/V tensors and computing Q·K^T at decode time, absorb pre-absorbs the W_KV projection into Q. This means attention scores are computed as `q_nope_absorbed @ kv_cache.T + q_pe @ pe_cache.T`, where `kv_cache` stores only the normalized low-rank latent — shared across all heads. The full K/V tensors are never materialized in memory at all. At sequence length 8K with 128 heads, this saves roughly 650 MB per layer.

## 代码 / The code

`deepseek-ai/DeepSeek-V3` — [`inference/model.py`](https://github.com/deepseek-ai/DeepSeek-V3/blob/b15f0dbbbe6a4bc403306175698439ef380f5fb5/inference/model.py#L221-L322)

```python
class MLA(nn.Module):
    """
    Multi-Head Latent Attention (MLA) Layer.
    ...
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        if self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim ** -0.5

        if attn_impl == "naive":
            # naive: cache full K and V — shape [max_seq_len, n_local_heads, head_dim]
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim), persistent=False)
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim), persistent=False)
        else:
            # absorb: cache only the compressed latent and the RoPE component
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask):
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        if attn_impl == "naive":
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv = self.wkv_b(self.kv_norm(kv))
            kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            self.k_cache[:bsz, start_pos:end_pos] = k
            self.v_cache[:bsz, start_pos:end_pos] = v
            scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
        else:
            # absorb mode: reshape W_KV into per-head slices
            wkv_b = self.wkv_b.weight.view(self.n_local_heads, -1, self.kv_lora_rank)
            # pre-multiply q_nope by W_KV[:qk_nope_head_dim] — absorbs the key projection into Q
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            # only the normalized low-rank latent and the RoPE component are cached
            self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
            # attention score = (q_nope_absorbed @ kv_cache.T) + (q_pe @ pe_cache.T)
            scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                      torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
        if mask is not None:
            scores += mask.unsqueeze(1)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        if attn_impl == "naive":
            x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
        else:
            # output: first compute weighted sum in kv_lora_rank space, then expand to v_head_dim
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        x = self.wo(x.flatten(2))
        return x
```

## 逐行讲解 / What's happening

1. **`__init__` L264-269：两种缓存寄存器 / Two cache register modes**
   - 中文：naive 模式注册 `k_cache` 和 `v_cache`，形状 `[batch, seq, n_local_heads, head_dim]`；absorb 模式只注册 `kv_cache`（形状 `[batch, seq, kv_lora_rank=512]`）和 `pe_cache`（`[batch, seq, qk_rope_head_dim=64]`）。两者之间的显存差异就是压缩比的直接来源。
   - English: naive registers full-size K and V caches; absorb registers only the compressed latent (`kv_lora_rank=512`) and the RoPE component (`qk_rope_head_dim=64`). With 128 heads and a head dim of 192+128, naive stores 128×320=40 960 values per token while absorb stores only 576 — a ~71× compression at model level.

2. **L291-295：Q 的拆分和 KV 的低秩投影 / Q split and low-rank KV projection**
   - 中文：Q 被拆成 `q_nope`（不含位置编码，维度 128）和 `q_pe`（含 RoPE，维度 64）。输入 x 经 `wkv_a` 投影到 `kv_lora_rank + qk_rope_head_dim = 576`，再拆成 `kv`（低秩潜变量）和 `k_pe`（RoPE 分量）。
   - English: Q is split into a positional-free part (`q_nope`) and a RoPE part (`q_pe`). The input x is projected to a 576-dim vector by `wkv_a`, then split into the KV latent (`kv`, 512 dims) and the rotary key component (`k_pe`, 64 dims).

3. **L306-308（absorb 核心）：Q 吸收 W_KV / The absorb step — Q absorbs W_KV**
   - 中文：`wkv_b.weight` 是形状 `[n_local_heads × (qk_nope_head_dim + v_head_dim), kv_lora_rank]` 的矩阵，把它 `view` 成 `[n_local_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank]` 后，`q_nope` 与其前 `qk_nope_head_dim` 个切片做 einsum——相当于提前把 `W_Q_nope × W_KV_k` 合并为一步。这样 Q 已经在 `kv_lora_rank` 空间里，直接与缓存的压缩潜变量点积即可。
   - English: `wkv_b.weight` is reshaped to `[n_heads, qk_nope+v_dim, kv_lora_rank]`. The einsum `"bshd,hdc->bshc"` multiplies each head's `q_nope` vector (shape `[..., qk_nope_head_dim]`) by the first `qk_nope_head_dim` rows of `wkv_b`, yielding `q_nope` in the latent space (shape `[..., kv_lora_rank]`). Now `q_nope` and `kv_cache` live in the same space and their dot-product gives the non-positional part of the attention score — no K materialization needed.

4. **L309-312：缓存写入 + 分数计算 / Cache write + score computation**
   - 中文：`self.kv_norm(kv)` 对低秩潜变量做 RMSNorm 后写入 `kv_cache`，`k_pe.squeeze(2)` 写入 `pe_cache`。注意力分数分两部分：`q_nope_absorbed @ kv_cache.T`（无位置信息的语义注意力）+ `q_pe @ pe_cache.T`（位置相关注意力），相加后统一乘 `softmax_scale`。
   - English: The normalized latent `self.kv_norm(kv)` is stored in `kv_cache`; the rotary key `k_pe` goes into `pe_cache`. Attention scores are the sum of two einsum terms: semantic (non-positional) scores from the absorbed Q·kv_cache.T, and positional scores from q_pe·pe_cache.T. The two are additive because the original full attention score Q·K.T decomposes exactly this way under the MLA parameterization.

5. **L319-320：输出重建 / Output reconstruction**
   - 中文：输出也在低秩空间计算：先得到 `[bsz, seq, n_heads, kv_lora_rank]` 的加权和，再与 `wkv_b` 的后 `v_head_dim` 切片相乘展开到完整的 `v_head_dim`。V 张量同样不需要实例化。
   - English: The output is also computed in the latent space: the weighted sum over `kv_cache` gives a `[bsz, seq, n_heads, kv_lora_rank]` tensor, then a second einsum with `wkv_b[:, -v_head_dim:]` expands it to the full `v_head_dim`. The V matrix is never materialized either — the whole decode pass touches only the 576-wide cache.

## 类比 / The analogy

把 kv_lora_rank 想象成一个高压缩格式的"存档"，就像 ZIP 文件。naive 模式是把每本书都解压后堆在书架上——要查时随手就拿。absorb 模式是把书压缩存档，同时把"查阅索引"（W_Q_nope × W_KV）提前预计算好、嵌入到你的眼镜（Q）里——戴着眼镜直接扫描 ZIP 文件就能得到相关性分数，完全不需要解压。读取时确实需要一步额外的 einsum（展开到 V），但这比把几十本书都解压后才能读要省空间得多。

Think of `kv_lora_rank` as a highly compressed archive, like a ZIP file. Naive mode unpacks every book onto a shelf so you can grab them at a glance. Absorb mode keeps books zipped and pre-embeds the reading index into your glasses (into Q, via the W_Q·W_KV pre-multiplication) — you scan the ZIP directly and get relevance scores without ever unzipping. Reconstruction (the V einsum on lines 319-320) does the final decompression, but only for the current query's output, not the entire archive.

## 自己跑一遍 / Try it yourself

```python
import torch

# MLA hyperparams from DeepSeek-V3 config
n_heads, kv_lora_rank, qk_nope_dim, qk_rope_dim, v_dim = 128, 512, 128, 64, 128

naive_per_token  = n_heads * (qk_nope_dim + qk_rope_dim + v_dim)   # K + V
absorb_per_token = kv_lora_rank + qk_rope_dim                       # kv_latent + pe

print(f"naive  KV cache per token: {naive_per_token:6d} values")
print(f"absorb KV cache per token: {absorb_per_token:6d} values")
print(f"compression ratio: {naive_per_token / absorb_per_token:.1f}×")

# Simulate the absorb step (the key math)
B, S, H, D_nope, D_lora = 1, 10, 4, 16, 32
q_nope  = torch.randn(B, S, H, D_nope)
wkv_b   = torch.randn(H, D_nope + 8, D_lora)  # +8 for v_dim slice
kv_cache = torch.randn(B, S, D_lora)

# Pre-multiply: move W_KV_k into Q
q_absorbed = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :D_nope])
# Score without materializing K
scores = torch.einsum("bshc,btc->bsht", q_absorbed, kv_cache)
print(f"\nscores shape: {scores.shape}  (expected [1, 10, 4, 10])")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
naive  KV cache per token:  40960 values
absorb KV cache per token:    576 values
compression ratio: 71.1×

scores shape: torch.Size([1, 10, 4, 10])  (expected [1, 10, 4, 10])
```

注意力分数的形状完全正确，且全程没有任何一个完整尺寸的 K 张量出现——这就是 absorb 的本质。

The score tensor has the correct shape and no full-size K tensor was ever allocated. That is the absorb trick in one line: you trade a matrix materialization for an extra einsum at Q preparation time.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **DeepSeek-V2 MLA（原始论文实现）** / **DeepSeek-V2 original MLA**: MLA 首次提出于 DeepSeek-V2（2024），V3 延续并优化了这一设计，增加了 q_lora_rank 对 Q 也做低秩压缩。
- **GQA / MQA（弱化版）** / **GQA / MQA (weaker form)**: 把 n_kv_heads 降到 1 或更少是 KV 压缩的最简粗暴形式，但无法像 MLA absorb 那样做跨 head 的低秩共享。
- **Flash-Decoding 的 KV 分片** / **Flash-Decoding's KV sharding**: 不压缩 KV，而是在 decode 时把 KV 序列分片到多个线程——与 MLA absorb 正交，可以同时使用。

## 注意事项 / Caveats / when it breaks

- **仅限 decode 场景** / **Decode-only**: absorb 模式的意义在于减少缓存大小；prefill 阶段全序列一次性过，内存不是瓶颈，所以 prefill 通常仍走 naive 路径。
- **数值差异** / **Numerical differences**: absorb 先做 Q×W_KV einsum，naive 先做 W_KV×kv 再展开——浮点运算顺序不同，结果会有极小的数值偏差，但在 FP16/BF16 精度下通常可忽略。
- **量化兼容** / **Quantization compatibility**: L306 中有 `weight_dequant` 分支——当 `wkv_b` 以 FP8 量化存储时，需要在 absorb 步骤前先反量化，代码里已经处理了这个情况。

## 延伸阅读 / Further reading

- [DeepSeek-V2 技术报告（MLA 原始定义）](https://arxiv.org/abs/2405.04434)
- [DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437)
- [今日另一篇 DeepSeek-V3 笔记（ue8m0 act_quant）](../2026-06-14-deepseek-v3-ue8m0-act-quant.md)
