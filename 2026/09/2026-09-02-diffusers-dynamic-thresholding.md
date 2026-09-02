---
date: 2026-09-02
topic: huggingface
source: huggingface
repo: huggingface/diffusers
file: src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
permalink: https://github.com/huggingface/diffusers/blob/bda3386ddcaa31bc2853567a95e815629c8022b7/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py#L499-L541
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, diffusers, scheduler, thresholding]
---

# Diffusers dynamic thresholding：按分位数收紧 x0 / Diffusers Dynamic Thresholding: Clamp x0 by Percentile

> **一句话 / In one line**: 这段代码不是固定裁剪，而是先看每张图自己的分位数，再把预测的 `x_0` 压回安全区间。 / This code does not clip to a fixed bound; it looks at each sample’s percentile first, then squeezes the predicted `x_0` back into a safe range.

## 为什么重要 / Why this matters

中文：扩散采样最怕一步把像素推爆。固定 `[-1, 1]` 虽然简单，但太粗暴；Diffusers 这里用分位数估计每张样本的“允许幅度”，再按这个幅度缩放，既能抑制离群值，又不会把所有样本一起压扁。

English: Diffusion sampling is easy to blow up in a single step. A fixed `[-1, 1]` clamp is simple but blunt; Diffusers instead estimates a per-sample “safe amplitude” from a percentile, then rescales around that amplitude. The result tames outliers without flattening every image the same way.

## 代码 / The code

`huggingface/diffusers` — [`src/diffusers/schedulers/scheduling_dpmsolver_multistep.py`](https://github.com/huggingface/diffusers/blob/bda3386ddcaa31bc2853567a95e815629c8022b7/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py#L499-L541)

```python
def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
    """
    Apply dynamic thresholding to the predicted sample.
    "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
    prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by s.
    Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
    pixels from saturation at each step.
    """
    dtype = sample.dtype
    batch_size, channels, *remaining_dims = sample.shape
    if dtype not in (torch.float32, torch.float64):
        sample = sample.float()

    # Flatten sample for doing quantile calculation along each image
    sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))

    abs_sample = sample.abs()
    s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
    s = torch.clamp(s, min=1, max=self.config.sample_max_value)
    s = s.unsqueeze(1)
    sample = torch.clamp(sample, -s, s) / s
    sample = sample.reshape(batch_size, channels, *remaining_dims)
    sample = sample.to(dtype)

    return sample
```

## 逐行讲解 / What's happening

1. **第 500-507 行 / Lines 500-507**:
   - 中文: 先记住原始 dtype，再必要时上浮到 float32，避免量化和 `clamp` 的数值问题。
   - English: The code preserves the original dtype, then upcasts to float32 when needed to avoid numerical issues in quantiles and clamping.
2. **第 526-536 行 / Lines 526-536**:
   - 中文: `torch.quantile()` 先找每张图的幅度分位数 `s`，再把整张图压到 `[-s, s]` 后除回去。
   - English: `torch.quantile()` finds a per-image amplitude `s`, then the code clamps the whole sample to `[-s, s]` and divides by `s`.
3. **第 538-541 行 / Lines 538-541**:
   - 中文: 最后把形状和 dtype 还原，调用者拿到的还是原来那个 batch 结构。
   - English: The last step restores the original shape and dtype, so the caller gets back the same batch contract.

## 类比 / The analogy

中文：像摄影师看直方图后再决定曝光补偿。不是机械地把所有照片都拉回同一个亮度，而是每张照片单独判断“哪里开始过曝”。

English: It is like a photographer checking the histogram before dialing exposure compensation. Instead of forcing every photo to the same brightness, you decide where each one starts to clip.

## 自己跑一遍 / Try it yourself

```python
def percentile_clip(xs, p=0.75):
    xs = sorted(abs(x) for x in xs)
    idx = min(len(xs) - 1, int(round((len(xs) - 1) * p)))
    s = max(1, xs[idx])
    return [round(max(-s, min(s, x)) / s, 3) for x in xs]

print(percentile_clip([0.2, -0.7, 2.4, 0.9]))
print(percentile_clip([0.1, 0.2, 0.3, 0.4]))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[0.083, 0.292, 1.0, 0.375]
[0.1, 0.2, 0.3, 0.4]
```

中文：第一组里最大的值会决定缩放上限；第二组没到阈值时，基本不会动。

English: In the first list, the largest value sets the scale limit; in the second, nothing really changes because the values never exceed the threshold.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Diffusers FlowMatch schedule** / **Diffusers FlowMatch schedule**: 中文: 采样器经常把“安全范围”写成显式参数。 / English: Schedulers often make the “safe range” an explicit parameter.
- **PyTorch dynamic gradient clipping** / **PyTorch dynamic gradient clipping**: 中文: 先估计统计量，再决定怎么裁剪。 / English: Estimate a statistic first, then decide how to clip.

## 注意事项 / Caveats / when it breaks

- **分位数不是常数 / The percentile is not a constant**: 中文: 这招依赖 `dynamic_thresholding_ratio` 真的适合你的数据分布。 / English: This depends on `dynamic_thresholding_ratio` matching your data distribution.
- **batch 内每个样本各自算阈值 / Thresholds are per-sample**: 中文: 它不是一个全 batch 共用的硬阈值。 / English: It is not one hard threshold shared across the batch.

## 延伸阅读 / Further reading

- Diffusers source: https://github.com/huggingface/diffusers/blob/bda3386ddcaa31bc2853567a95e815629c8022b7/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
