---
date: 2026-08-01
topic: wam
source: wam
repo: yuantianyuan01/FastWAM
file: src/fastwam/models/wan22/fastwam.py
permalink: https://github.com/yuantianyuan01/FastWAM/blob/main/src/fastwam/models/wan22/fastwam.py#L379-L413
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, training-loop, mask]
build_role: training-loop advanced variant, per-sample valid-latent loss mask
---

# FastWAM video loss mask：只让有效 latent 参与平均 / FastWAM Video Loss Mask: Average Only Valid Latents

> **一句话 / In one line**: `_compute_video_loss_per_sample` 先把 video mask 下采样到 latent 尺度，再按样本分别做 masked MSE。 / `_compute_video_loss_per_sample` downsamples the video mask to latent scale, then computes masked MSE per sample.

## 为什么重要 / Why this matters

WAM 训练里不同视频可以有不同长度和 padding。如果直接对整个 latent tensor 求均值，padding 帧会参与 loss，模型会学习“预测 padding”。这段代码把 mask 对齐到 latent 尺度，并让每个样本只除以自己的有效元素数。

WAM training often batches videos with different lengths and padding. If you average over the whole latent tensor, padded frames enter the loss and the model learns to predict padding. This code aligns the mask to latent resolution and divides each sample by its own valid element count.

## 代码 / The code

`yuantianyuan01/FastWAM` — [`src/fastwam/models/wan22/fastwam.py`](https://github.com/yuantianyuan01/FastWAM/blob/main/src/fastwam/models/wan22/fastwam.py#L379-L413)

```python
def _compute_video_loss_per_sample(
    self,
    pred: torch.Tensor,
    target: torch.Tensor,
    video_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute per-sample video MSE with a frame-level validity mask."""
    loss = F.mse_loss(pred.float(), target.float(), reduction="none")
    if video_mask is None:
        return loss.flatten(1).mean(dim=1)

    # video_mask: [B, T] at frame rate. Convert to latent time resolution.
    latent_t = loss.shape[2]
    mask = video_mask.float()
    if mask.shape[1] != latent_t:
        mask = F.interpolate(
            mask[:, None, :],
            size=latent_t,
            mode="nearest",
        )[:, 0]

    mask = mask[:, None, :, None, None]
    loss = loss * mask
    denom = mask.expand_as(loss).flatten(1).sum(dim=1).clamp_min(1.0)
    return loss.flatten(1).sum(dim=1) / denom
```

## 逐行讲解 / What's happening

1. **第 386 行 / Line 386 (`reduction="none"`)**:
   - 中文: 先保留每个元素的 MSE，后面才有机会按 mask 精确筛选。
   - English: It keeps elementwise MSE first, so the mask can select valid positions later.
2. **第 387-388 行 / Lines 387-388 (no mask path)**:
   - 中文: 没有 mask 时退化成普通 per-sample 平均。
   - English: Without a mask, it falls back to normal per-sample averaging.
3. **第 391-399 行 / Lines 391-399 (frame to latent)**:
   - 中文: 视频 mask 原本按帧给出；latent 时间长度不同，就用 nearest 下采样/上采样对齐。
   - English: The video mask is frame-level; if latent time differs, nearest interpolation aligns it.
4. **第 401-404 行 / Lines 401-404 (masked mean)**:
   - 中文: mask 扩到 `[B,C,T,H,W]`，分子只加有效 loss，分母只数有效元素。
   - English: The mask expands to `[B,C,T,H,W]`; the numerator sums valid loss and the denominator counts valid elements.

## 类比 / The analogy

这像批改试卷时只算学生实际作答的题。空白附加页不能算错，也不能进平均分分母。

It is like grading only the questions a student actually answered. Blank extra pages should not count as wrong answers or enter the denominator.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

这是 `training-loop` 的 advanced variant。上游是 VAE latent、噪声目标和数据集 padding mask，下游是优化器看到的 video loss；如果省掉它，变长视频 batch 会把 padding 误当训练信号。生产级实现还要同时处理 action mask、camera mask 和跨设备 all-reduce 的有效元素数。

This is an advanced variant of the `training-loop` component. Upstream are VAE latents, noise targets, and dataset padding masks; downstream is the video loss passed to the optimizer. If you skip it, variable-length video batches treat padding as a training signal. A production version also needs action masks, camera masks, and distributed valid-element counts.

## 自己跑一遍 / Try it yourself

```python
loss = [[1, 4, 9, 16], [25, 36, 49, 64]]
mask = [[1, 1, 0, 0], [1, 0, 0, 0]]
for row, m in zip(loss, mask):
    num = sum(v for v, keep in zip(row, m) if keep)
    den = max(sum(m), 1)
    print(num / den)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
2.5
25.0
```

每个样本用自己的有效位置数量做分母，所以短视频不会被 padding 稀释。

Each sample uses its own count of valid positions, so shorter videos are not diluted by padding.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **language-model label mask** / **language-model label mask**: `-100` 标签也是只让有效 token 进入 loss。 / `-100` labels similarly keep only valid tokens in the loss.
- **diffusion action mask** / **diffusion action mask**: 动作 chunk 有 padding 或不可控维度时，也要 mask 后再平均。 / Action chunks with padding or uncontrollable dimensions also need masked averaging.

## 注意事项 / Caveats / when it breaks

- **nearest 对齐是假设** / **Nearest alignment is an assumption**: 如果 VAE 的时间压缩不是简单分段，mask 对齐要跟 encoder stride 一致。 / If VAE temporal compression is not simple binning, mask alignment must match encoder stride.
- **分母不能为零** / **Denominator cannot be zero**: `clamp_min(1.0)` 避免全 padding 样本产生 NaN。 / `clamp_min(1.0)` prevents all-padding samples from producing NaN.

## 延伸阅读 / Further reading

- [FastWAM model source](https://github.com/yuantianyuan01/FastWAM/blob/main/src/fastwam/models/wan22/fastwam.py#L379-L413)

