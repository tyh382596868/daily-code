---
date: 2026-07-06
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/modules/attention.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/attention.py#L21-L124
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, flash-attention]
build_role: dit-block advanced attention backend
---

# Wan2.1 variable-length FlashAttention：先压平有效 token，再交给 kernel / Wan2.1 Variable-Length FlashAttention: Flatten Valid Tokens Before Calling the Kernel

> **一句话 / In one line**: 这层 wrapper 把 padded batch 转成 varlen attention 需要的扁平 token 和累计长度。 / This wrapper converts a padded batch into flat valid tokens plus cumulative lengths for varlen attention.

## 为什么重要 / Why this matters

视频 latent 的长度经常不一样：不同分辨率、不同帧数、不同裁剪都会让 token 数变化。直接 padding 会浪费大量 attention 计算。Wan2.1 的 wrapper 先按 `q_lens/k_lens` 裁掉无效 token，构造 `cu_seqlens`，再调用 FlashAttention 2 或 3。

Video latents often have different lengths because resolution, frame count, and cropping vary. Padding them directly wastes attention compute. Wan2.1's wrapper slices away invalid tokens using `q_lens/k_lens`, builds `cu_seqlens`, then calls FlashAttention 2 or 3.

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/modules/attention.py`](https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/attention.py#L21-L124)

```python
def flash_attention(
    q, k, v, q_lens=None, k_lens=None, dropout_p=0.,
    softmax_scale=None, q_scale=None, causal=False,
    window_size=(-1, -1), deterministic=False,
    dtype=torch.bfloat16, version=None,
):
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32).to(device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32).to(device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q, k=k, v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq, max_seqlen_k=lk,
            softmax_scale=softmax_scale, causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        x = flash_attn.flash_attn_varlen_func(
            q=q, k=k, v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq, max_seqlen_k=lk,
            dropout_p=dropout_p, softmax_scale=softmax_scale,
            causal=causal, window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    return x.type(out_dtype)
```

## 逐行讲解 / What's happening

1. **第 8-10 行 / Lines 8-10 (asserts)**:
   - 中文: FlashAttention kernel 对 dtype、设备、head_dim 有硬约束，wrapper 提前挡掉不合法输入。
   - English: FlashAttention kernels have strict dtype, device, and head-dim constraints, so the wrapper rejects invalid inputs early.
2. **第 17-28 行 / Lines 17-28 (flatten or trim)**:
   - 中文: 没有长度表时直接 flatten；有长度表时逐样本切到真实长度再拼接。
   - English: without length arrays, the batch is flattened; with them, each sample is sliced to its true length before concatenation.
3. **第 39-43 行 / Lines 39-43 (`cu_seqlens`)**:
   - 中文: varlen kernel 不看 padding mask，而是看每段序列在扁平数组里的起止位置。
   - English: the varlen kernel does not consume a padding mask; it consumes sequence boundaries inside the flat token array.
4. **第 54-60 行 / Lines 54-60 (FA2 fallback)**:
   - 中文: FA3 不可用时落到 FA2，同时保留 window attention、dropout 等参数。
   - English: when FA3 is unavailable, FA2 is used while preserving options such as window attention and dropout.

## 类比 / The analogy

像电影院检票：你不会给空座位也发票。先把真正的观众排成一队，再给检票员一张“每个影厅有多少人”的表。

It is like checking tickets at a cinema: you do not issue tickets to empty seats. Put the real viewers into one line, then hand the usher a table of how many people belong to each room.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

这是 `dit-block` 的高级 attention backend。最小 nanoWAM 可以先用 padded `scaled_dot_product_attention`，但生产视频模型需要 varlen attention 来避免不同视频长度带来的 padding 浪费。它依赖 patchify/position encoding 已经产出 token 序列，并服务于 self-attention 和 cross-attention。

This is an advanced attention backend for the `dit-block`. A minimal nanoWAM can start with padded `scaled_dot_product_attention`, but production video models need varlen attention to avoid padding waste across different video lengths. It depends on patchify/position encoding producing token sequences and serves both self-attention and cross-attention.

## 自己跑一遍 / Try it yourself

```python
import numpy as np

x = np.arange(2 * 4 * 1).reshape(2, 4, 1)
lens = np.array([3, 1])
flat = np.concatenate([row[:n] for row, n in zip(x, lens)], axis=0)
cu = np.r_[0, np.cumsum(lens)]
print(flat[:, 0].tolist())
print(cu.tolist())
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
[0, 1, 2, 4]
[0, 3, 4]
```

padding 的 token 没有进入 flat 数组；`cu` 告诉 kernel 第一段是 `[0:3]`，第二段是 `[3:4]`。

Padded tokens never enter the flat array; `cu` tells the kernel that the first sequence is `[0:3]` and the second is `[3:4]`.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **PyTorch varlen attention** / **PyTorch varlen attention**: paged / varlen attention 也会把形状问题转成索引表问题。 / Paged and varlen attention also turn shape handling into index-table handling.
- **vLLM paged KV cache** / **vLLM paged KV cache**: 同样避免把所有请求 pad 到同一个长度。 / It similarly avoids padding all requests to the same length.

## 注意事项 / Caveats / when it breaks

- **只适合 CUDA kernel 路径** / **This is CUDA-kernel-specific**: CPU 或不支持的 head_dim 要走 fallback。 / CPU or unsupported head dimensions need a fallback.
- **输出 unflatten 假设固定 `lq`** / **The output unflatten assumes fixed `lq`**: 如果查询也是真变长，下游要确认 padding 位不会被误用。 / If queries are truly variable-length, downstream code must avoid using padded positions.

## 延伸阅读 / Further reading

- [Wan2.1 repository](https://github.com/Wan-Video/Wan2.1)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
