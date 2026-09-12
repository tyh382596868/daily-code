---
date: 2026-08-19
topic: infrastructure
source: tracked
repo: karpathy/nanoGPT
file: train.py
permalink: https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/train.py#L230-L242
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, infrastructure, learning-rate-schedule]
---

# nanoGPT 学习率调度：先热身，再余弦降温 / nanoGPT LR Schedule: Warm Up, Then Cosine Cool Down

> **一句话 / In one line**: 这段函数把训练步数映射成学习率：warmup 线性升高，中段余弦下降，末段钉在 `min_lr`。 / This function maps an iteration number to a learning rate: linear warmup, cosine decay, then a hard floor at `min_lr`.

## 为什么重要 / Why this matters

训练基础设施里，学习率调度是最小但最影响稳定性的控制器。模型刚开始训练时参数和优化器状态都很乱，直接用最大学习率容易震荡；训练后期还保持大学习率，又会在最优点附近来回跳。nanoGPT 用 12 行把这两个问题都处理掉。

In training infrastructure, the learning-rate schedule is a tiny controller with a large stability impact. Early training needs a gentle ramp because weights and optimizer state are still settling; late training needs a smaller step size because the model is near a useful basin. nanoGPT compresses both ideas into a small pure function.

## 代码 / The code

`karpathy/nanoGPT` — [`train.py`](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/train.py#L230-L242)

```python
# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)
```

## 逐行讲解 / What's happening

1. **第 231-234 行 / Lines 231-234 (`warmup`)**:
   - 中文: `it < warmup_iters` 时学习率按步数线性变大，`it + 1` 避免第一步就是 0。
   - English: During warmup, the learning rate grows linearly. The `it + 1` term avoids starting at exactly zero.
2. **第 235-237 行 / Lines 235-237 (`min_lr`)**:
   - 中文: 超过衰减区间后直接返回下限，不让学习率继续变小到几乎不更新。
   - English: After the decay window, the function clamps to the floor so training keeps taking meaningful, bounded steps.
3. **第 239-242 行 / Lines 239-242 (`cosine`)**:
   - 中文: `decay_ratio` 把当前步归一化到 `[0, 1]`，余弦系数再从 1 平滑滑到 0。
   - English: `decay_ratio` normalizes progress into `[0, 1]`; the cosine coefficient then smoothly moves from 1 to 0.

## 类比 / The analogy

像开车上高速：刚出停车场不能一脚油门到底，所以先平稳加速；进入巡航后慢慢松油门；接近出口时保持一个低速，不是直接熄火滑行。

It is like joining a highway: ease onto the ramp first, cruise while gradually reducing throttle, then keep a low controlled speed near the exit instead of cutting the engine.

## 自己跑一遍 / Try it yourself

```python
import math

learning_rate = 1.0
min_lr = 0.1
warmup_iters = 2
lr_decay_iters = 8

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

print([round(get_lr(i), 3) for i in range(11)])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[0.333, 0.667, 1.0, 0.94, 0.775, 0.55, 0.325, 0.16, 0.1, 0.1, 0.1]
```

中文: 关键现象是曲线没有突然断崖式下降，warmup 和 decay 边界都很平滑。

English: The important behavior is that the schedule has no cliff; both the warmup boundary and decay curve are smooth.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Transformers scheduler** / **Transformers scheduler**: `get_cosine_schedule_with_warmup` 也是 warmup + cosine 的封装。 / It packages the same warmup-plus-cosine pattern for Hugging Face training loops.
- **PyTorch `CosineAnnealingLR`** / **PyTorch `CosineAnnealingLR`**: PyTorch 提供余弦下降，但 warmup 通常要外接 `SequentialLR` 或自定义函数。 / PyTorch provides cosine decay, while warmup is often composed with `SequentialLR` or a custom function.

## 注意事项 / Caveats / when it breaks

- **全局变量隐式依赖** / **Implicit globals**: 这个函数依赖 `warmup_iters`、`learning_rate` 等外部变量，复制到别处时最好改成显式参数。 / The function depends on outer variables; pass them explicitly when extracting it.
- **step 口径要统一** / **Step semantics must match**: `it` 应该对应 optimizer step，不是 micro-batch step，否则梯度累积会让 schedule 走太快。 / `it` should mean optimizer step, not micro-batch step, especially with gradient accumulation.

## 延伸阅读 / Further reading

- [nanoGPT `train.py`](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/train.py#L230-L242)
- [PyTorch `CosineAnnealingLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html)
