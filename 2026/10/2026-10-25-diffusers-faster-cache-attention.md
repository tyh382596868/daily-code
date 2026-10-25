---
date: 2026-10-25
topic: huggingface
source: huggingface
repo: huggingface/diffusers
file: src/diffusers/hooks/faster_cache.py
permalink: https://github.com/huggingface/diffusers/blob/aaf57d03a3edfbd9de457d2872417a92f149a91d/src/diffusers/hooks/faster_cache.py#L392-L483
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, huggingface, diffusers, attention-cache]
---

# Diffusers FasterCache：用两次真实结果外推一次 attention / Diffusers FasterCache: Extrapolate an Attention Step from Two Real Ones

> **一句话 / In one line**: FasterCache 在部分去噪步跳过 attention，并用最近两次真实输出做线性外推。 / FasterCache skips attention on selected denoising steps and linearly extrapolates from the two latest real outputs.

## 为什么重要 / Why this matters

去噪相邻步的中间特征通常变化平滑，因此不是每一步都必须完整执行昂贵的 attention。这里没有把缓存当作静态答案，而是保留两帧历史，按模块权重估计下一帧，并特别处理 CFG 拼接 batch。

Intermediate features often change smoothly between denoising steps, so expensive attention need not run every time. This cache is not a static answer: it keeps two snapshots, predicts the next one with a module-specific weight, and handles CFG's concatenated batch carefully.

## 代码 / The code

`huggingface/diffusers` — [`src/diffusers/hooks/faster_cache.py`](https://github.com/huggingface/diffusers/blob/aaf57d03a3edfbd9de457d2872417a92f149a91d/src/diffusers/hooks/faster_cache.py#L392-L483)

```python
def _compute_approximated_attention_output(self, t_2_output, t_output, weight, batch_size):
    if t_2_output.size(0) != batch_size:
        t_2_output = t_2_output[batch_size:]
    if t_output.size(0) != batch_size:
        t_output = t_output[batch_size:]
    return t_output + (t_output - t_2_output) * weight

if should_skip_attention:
    t_2_output, t_output = self.state.cache
    output = self._compute_approximated_attention_output(
        t_2_output, t_output, self.weight_callback(module), batch_size)
else:
    output = self.fn_ref.original_forward(*args, **kwargs)

if self.state.cache is None:
    self.state.cache = [cache_output, cache_output]
else:
    self.state.cache = [self.state.cache[-1], cache_output]
```

## 逐行讲解 / What's happening

1. **第 395-404 行 / Lines 395-404**：若 cache 保存的是 CFG 的无条件+条件双 batch，只保留条件半边，因为近似结果的形状必须匹配当前调用。 / If the cache holds CFG's unconditional-plus-conditional batch, retain the conditional half so the approximation matches the current call.
2. **第 405 行 / Line 405**：`current + weight * (current - previous)` 是一阶外推；`weight=0` 退化成复用当前值。 / This is first-order extrapolation; `weight=0` falls back to reusing the latest value.
3. **第 473-478 行 / Lines 473-478**：缓存是长度为二的滑动窗口，首次用同一值填满以避免“缺少前一帧”的特殊分支。 / The cache is a two-item sliding window; initialization duplicates the first value to avoid a missing-previous special case.

## 类比 / The analogy

像动画师根据最近两张关键帧猜中间姿势；当镜头运动剧烈时，再回到真正画一帧。 / It is an animator inferring an in-between pose from two keyframes, then drawing a real frame again when motion becomes too risky.

## 自己跑一遍 / Try it yourself

```python
previous, current, weight = 10.0, 12.0, 0.5
approx = current + (current - previous) * weight
print(approx)
```

运行 / Run with: `python try.py`

预期输出 / Expected output: `13.0`。

它假设局部变化近似线性；这是加速近似，而非数学恒等变换。 / It assumes locally linear change; this is an acceleration approximation, not a mathematical identity.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **视频帧插值 / Video interpolation**：用相邻帧估计中间状态。 / Estimate an intermediate state from adjacent frames.
- **ODE 求解器 / ODE solvers**：多步方法利用历史斜率预测下一步。 / Multistep methods use historical slopes to predict the next step.

## 注意事项 / Caveats / when it breaks

- **模型输出突变 / Abrupt model changes**：强条件切换或早期高噪声步会让外推失真。 / Strong conditioning changes or early high-noise steps can make extrapolation inaccurate.
- **CFG batch 语义 / CFG batch semantics**：错误切半会把无条件特征误当条件特征。 / Splitting the CFG batch incorrectly mixes unconditional and conditional features.

## 延伸阅读 / Further reading

- [FasterCache hook source](https://github.com/huggingface/diffusers/blob/aaf57d03a3edfbd9de457d2872417a92f149a91d/src/diffusers/hooks/faster_cache.py)
