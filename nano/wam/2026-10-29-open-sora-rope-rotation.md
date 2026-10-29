---
date: 2026-10-29
topic: wam
source: wam
repo: hpcaitech/Open-Sora
file: opensora/models/mmdit/math.py
permalink: https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/mmdit/math.py#L39-L67
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, wam, rope]
build_role: positional encoding advanced variant
---

# Open-Sora RoPE 旋转矩阵 / Open-Sora RoPE Rotation Matrices

> **一句话 / In one line**: Open-Sora 将位置变成每对通道共享的二维旋转。 / Open-Sora turns position into a 2D rotation shared by each channel pair.

## 为什么重要 / Why this matters

世界模型必须知道 token 的相对位置；RoPE 将位置信号嵌入 Q/K 的几何关系，不用额外加一个位置向量。

A world model must know relative token positions. RoPE puts position into the geometry of Q and K instead of adding a separate position vector.

## 代码 / The code

`hpcaitech/Open-Sora` — [`math.py`](https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/mmdit/math.py#L39-L67)

```python
scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
omega = 1.0 / (theta**scale)
out = torch.einsum("...n,d->...nd", pos, omega)
out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
```

## 逐行讲解 / What's happening

1. 中文：频率按 channel pair 从快到慢铺开。 English: Frequencies span fast to slow across channel pairs.
2. 中文：外积为每个 token、每种频率给出旋转角。 English: The outer product gives every token and frequency a rotation angle.
3. 中文：sin/cos 被排成 2×2 矩阵，以便同时作用于 Q 和 K。 English: Sine and cosine are arranged as 2×2 matrices for both Q and K.

## 类比 / The analogy

它像在每个时间点把一排小指南针转不同角度，比较两枚指南针时就能读出间距。

It is like rotating a row of tiny compasses by different angles; comparing two compasses reveals their separation.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

课程项是 `patchify-positional` 的高级变体。patchify 将视频变为 token 后，RoPE 在 DiT/MMDiT attention 前生成位置旋转；输入是 token 坐标，输出是 Q/K 旋转频率。没有它，模型仍能看 token，却难以区分“前一帧”和“后一帧”；生产系统还需 3D 时空坐标、长序列外推与精度策略。

This is an advanced `patchify-positional` variant. After patchification turns video into tokens, RoPE generates Q/K rotations before DiT/MMDiT attention; coordinates enter and rotation frequencies leave. Without it, tokens remain visible but temporal order is weak; production systems also need 3D coordinates, long-context extrapolation, and precision policy.

## 自己跑一遍 / Try it yourself

```python
import math
x, y, angle = 1., 0., math.pi / 2
rot = (math.cos(angle)*x-math.sin(angle)*y,
       math.sin(angle)*x+math.cos(angle)*y)
print(tuple(round(v, 1) for v in rot))
```

运行 / Run with: `python try.py`

预期输出 / Expected output: `(0.0, 1.0)`

中文：RoPE 的核心就是对每对特征做这种旋转。 English: This pairwise rotation is the core of RoPE.

## 注意事项 / Caveats / when it breaks

- 中文：坐标尺度和 `theta` 必须与训练设定一致。 / English: Coordinate scale and `theta` must match the training setup.

## 延伸阅读 / Further reading

- [Open-Sora RoPE source](https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/mmdit/math.py#L39-L67)
