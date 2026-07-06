---
date: 2026-07-06
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/optim/lr_scheduler.py
permalink: https://github.com/pytorch/pytorch/blob/main/torch/optim/lr_scheduler.py#L1238-L1308
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, pytorch, lr-scheduler]
---

# PyTorch PolynomialLR：递推和闭式公式要对齐 / PyTorch PolynomialLR: Keep the Recursive and Closed-Form Schedules Aligned

> **一句话 / In one line**: `PolynomialLR` 每步按上一步学习率乘衰减因子，同时保留一个用 `base_lrs` 直接计算的闭式版本。 / `PolynomialLR` updates each step by multiplying the previous learning rate by a decay factor, while also keeping a closed-form path from `base_lrs`.

## 为什么重要 / Why this matters

学习率 scheduler 看似只是公式，但 PyTorch 要同时支持“连续 step”以及“直接跳到某个 epoch”。如果递推公式和闭式公式不一致，resume training 或传入 epoch 时学习率会跳错。这里的代码展示了一个成熟 scheduler 最容易被忽略的双路径设计。

An LR scheduler looks like a formula, but PyTorch has to support both ordinary repeated `step()` calls and jumping directly to a requested epoch. If the recursive update and closed-form update disagree, resumed training can silently use the wrong LR. This code shows the two-path design a production scheduler needs.

## 代码 / The code

`pytorch/pytorch` — [`torch/optim/lr_scheduler.py`](https://github.com/pytorch/pytorch/blob/main/torch/optim/lr_scheduler.py#L1238-L1308)

```python
class PolynomialLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        total_iters: int = 5,
        power: float = 1.0,
        last_epoch: int = -1,
    ) -> None:
        self.total_iters = total_iters
        self.power = power
        super().__init__(optimizer, last_epoch)

    @override
    def get_lr(self) -> list[float | Tensor]:
        _warn_get_lr_called_within_step(self)

        if self._is_initial or self.last_epoch > self.total_iters:
            return _param_groups_val_list(self.optimizer, "lr")

        decay_factor = (
            (1.0 - self.last_epoch / self.total_iters)
            / (1.0 - (self.last_epoch - 1) / self.total_iters)
        ) ** self.power
        return [group["lr"] * decay_factor for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self) -> list[float | Tensor]:
        return [
            (
                base_lr
                * (1.0 - min(self.total_iters, self.last_epoch) / self.total_iters)
                ** self.power
            )
            for base_lr in self.base_lrs
        ]
```

## 逐行讲解 / What's happening

1. **第 3-11 行 / Lines 3-11 (`__init__`)**:
   - 中文: scheduler 只存两个超参：总衰减步数和多项式幂次。
   - English: the scheduler stores only two hyperparameters: total decay steps and the polynomial power.
2. **第 17-18 行 / Lines 17-18 (`_is_initial`)**:
   - 中文: 第一次调用不改学习率；超过 `total_iters` 后保持最后值。
   - English: the first call leaves LR unchanged; after `total_iters`, the current LR is kept.
3. **第 20-23 行 / Lines 20-23 (`decay_factor`)**:
   - 中文: 这里不是直接算 `base_lr * f(t)`，而是算 `f(t) / f(t-1)`，再乘到当前 lr 上。
   - English: this computes `f(t) / f(t-1)` and multiplies the current LR, instead of recomputing from `base_lr`.
4. **第 26-34 行 / Lines 26-34 (`_get_closed_form_lr`)**:
   - 中文: 当用户传 epoch 或恢复状态时，可以从 `base_lrs` 一步算到目标位置。
   - English: when a caller passes an epoch or restores state, PyTorch can compute the target LR directly from `base_lrs`.

## 类比 / The analogy

像爬山时有两种报海拔的方式：平时每走一步说“比刚才低了多少”，但 GPS 也要能直接告诉你“现在海拔是多少”。两套说法必须指向同一个位置。

It is like hiking with two ways to report elevation: during the walk you say how much lower you are than the previous step, but GPS still needs to report the absolute elevation. Both descriptions must land on the same point.

## 自己跑一遍 / Try it yourself

```python
base_lr, total, power = 0.1, 5, 2.0
lr = base_lr
recursive = []
for epoch in range(1, 6):
    factor = ((1 - epoch / total) / (1 - (epoch - 1) / total)) ** power
    lr *= factor
    closed = base_lr * (1 - min(total, epoch) / total) ** power
    recursive.append((round(lr, 6), round(closed, 6)))
print(recursive)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
[(0.064, 0.064), (0.036, 0.036), (0.016, 0.016), (0.004, 0.004), (0.0, 0.0)]
```

递推值和闭式值每一步都一致，这就是 scheduler 能安全 resume 的基础。

The recursive value matches the closed-form value at every step, which is the basis for safe scheduler resume.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **CosineAnnealingLR** / **CosineAnnealingLR**: 同样要处理递推更新和闭式计算。 / It also separates recursive stepping from closed-form evaluation.
- **训练 checkpoint** / **Training checkpoints**: 保存 optimizer 和 scheduler 状态时，`last_epoch` 必须能恢复出同一条曲线。 / Saving optimizer and scheduler state requires `last_epoch` to reconstruct the same curve.

## 注意事项 / Caveats / when it breaks

- **不要手动调 optimizer lr 又继续 scheduler** / **Do not manually edit optimizer LR mid-schedule**: 递推路径会把手动改动当成新的基准。 / The recursive path treats manual edits as the new current baseline.
- **`total_iters` 不能为 0** / **`total_iters` cannot be zero**: 公式里直接除以它。 / The formula divides by it directly.

## 延伸阅读 / Further reading

- [PyTorch LR scheduler docs](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
