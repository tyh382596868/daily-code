---
date: 2026-10-26
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/modules/model.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py#L112-L160
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, patchify-positional, rope, attention]
build_role: patchify-positional advanced variant
---

# Wan2.1 时空 RoPE attention：Q、K 带坐标，V 保持内容 / Wan2.1 Spatiotemporal RoPE Attention: Put Coordinates on Q and K, Keep V as Content

> **一句话 / In one line**: Wan 只旋转 query/key，让注意力感知视频 patch 的时空位置。 / Wan rotates query and key only, making attention aware of video-patch positions.

## 为什么重要 / Why this matters

视频 token 不只是序列编号；它们对应帧、高和宽。把 `grid_sizes` 和频率表送进 `rope_apply`，同一 attention 就能处理不同长度和分辨率。

Video tokens have frame, height, and width—not merely an index. Feeding grid sizes and frequencies into RoPE lets one attention layer handle varying shapes.

## 代码 / The code

```python
q, k, v = qkv_fn(x)
x = flash_attention(
    q=rope_apply(q, grid_sizes, freqs),
    k=rope_apply(k, grid_sizes, freqs),
    v=v, k_lens=seq_lens, window_size=self.window_size)
x = self.o(x.flatten(2))
```

来源 / Source: [`wan/modules/model.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py#L112-L160)。

## 逐行讲解 / What's happening

线性层先产生按 head 切分的 Q/K/V；RoPE 只改变比较用的 Q/K 相位，`v` 留作内容；变长边界由 `k_lens` 交给 attention kernel。

The projections make head-wise Q/K/V; RoPE changes only comparison phases in Q/K, while V stays content; `k_lens` carries ragged boundaries.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

这是已覆盖 `patchify-positional` 的生产级变体，位于 patch embedding 之后、DiT block 内部。输入是 patch tokens、视频网格和有效长度，输出是混合后的 tokens。省略位置编码后，模型会把相同外观的不同时间/位置混淆；生产版还要缓存频率表并正确处理并行切分。

This advanced `patchify-positional` variant lives after patch embedding inside a DiT block. Without it, identical-looking patches at different places or times become ambiguous.

## 类比 / The analogy

像会议座位表：发言内容不变，但座位号让你知道是谁在何时何地说的。

Like a seating chart: the words stay the same, but seat labels provide who-and-where context.

## 自己跑一遍 / Try it yourself

```python
import torch
q = torch.tensor([[1., 0.], [1., 0.]])
theta = torch.tensor([0., 1.])
rot = torch.stack([q[:,0]*theta.cos()-q[:,1]*theta.sin(),
                   q[:,0]*theta.sin()+q[:,1]*theta.cos()], -1)
print(rot.round(decimals=2))
```

运行 / Run with `pip install torch && python try.py`；预期 / Expected: 两个相同向量因位置不同而获得不同相位。

## 注意事项 / Caveats / when it breaks

- **网格一致性 / Grid consistency**：token 排列必须和 `grid_sizes` 完全一致。 / Token ordering must exactly match `grid_sizes`.
