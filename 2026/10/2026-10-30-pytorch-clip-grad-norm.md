---
date: 2026-10-30
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/nn/utils/clip_grad.py
permalink: https://github.com/pytorch/pytorch/blob/e9295c692a07e709c9070b7fb55cf76e8f2d79c1/torch/nn/utils/clip_grad.py#L44-L115
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, pytorch, gradient-clipping]
---

# PyTorch 梯度范数裁剪 / PyTorch Gradient-Norm Clipping

> **一句话 / In one line**: 先测所有梯度的整体长度，再用同一个比例缩小它们。 / Measure the whole gradient vector, then shrink every gradient by one shared ratio.

## 为什么重要 / Why this matters

梯度裁剪不是逐 tensor 截断；它是在不改变参数间相对方向的前提下，限制一次更新的总步幅。

Gradient clipping is not per-tensor truncation. It bounds the total update while preserving the relative direction among parameters.

## 代码 / The code

`pytorch/pytorch` — [`clip_grad.py`](https://github.com/pytorch/pytorch/blob/e9295c692a07e709c9070b7fb55cf76e8f2d79c1/torch/nn/utils/clip_grad.py#L104-L115)

```python
total_norm = _get_total_norm(grads, max_norm, norm_type, error_if_nonfinite, foreach)
_clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
return total_norm
```

## 逐行讲解 / What's happening

1. 中文：第一步跨全部参数求一个 p-norm；它是裁剪前的诊断值。
   English: The first step computes one p-norm over all parameters and returns it for diagnostics.
2. 中文：第二步用 `max_norm / total_norm` 得到共享缩放系数。
   English: The second step derives a shared scale from `max_norm / total_norm`.
3. 中文：函数返回原始范数，而不是裁剪后的常数。
   English: The function returns the original norm, not the clipped constant.

## 类比 / The analogy

像把一束不同长度的箭整体缩短，而不是把每支箭削到同样长。

It is like shrinking a bundle of arrows together instead of sawing every arrow to the same length.

## 自己跑一遍 / Try it yourself

```python
import math
grads = [3.0, 4.0]
norm = math.sqrt(sum(g * g for g in grads))
scale = min(1.0, 2.0 / (norm + 1e-6))
print(norm, [round(g * scale, 2) for g in grads])
```

运行 / Run with: `python try.py`

预期输出 / Expected output: `5.0 [1.2, 1.6]`

中文：方向保持为 3:4，只是长度降到 2。 English: The 3:4 direction stays intact while its length becomes 2.

## 注意事项 / Caveats / when it breaks

- 中文：在 `optimizer.step()` 前调用，且不要悄悄忽略 NaN/Inf。 / English: Call it before `optimizer.step()` and do not silently ignore NaN/Inf gradients.

## 延伸阅读 / Further reading

- [PyTorch clip gradients](https://github.com/pytorch/pytorch/blob/e9295c692a07e709c9070b7fb55cf76e8f2d79c1/torch/nn/utils/clip_grad.py#L44-L115)
