---
date: 2026-08-23
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/optim/lr_scheduler.py
permalink: https://github.com/pytorch/pytorch/blob/main/torch/optim/lr_scheduler.py#L914-L979
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, pytorch, lr-scheduler]
---

# PyTorch ExponentialLR：递推乘一次，闭式算全程 / PyTorch ExponentialLR: Multiply Once, or Compute the Whole Curve

> **一句话 / In one line**: `ExponentialLR` 平常只把当前 lr 乘上 `gamma`，但当你指定 epoch 时会用 `base_lr * gamma**epoch` 直接算闭式结果。 / `ExponentialLR` normally multiplies the current lr by `gamma`, but when an epoch is supplied it can compute `base_lr * gamma**epoch` directly.

## 为什么重要 / Why this matters

学习率调度器有两种工作方式：训练循环里一步步走，或者从 checkpoint / 指定 epoch 直接跳到某个位置。PyTorch 同时保留递推公式和闭式公式，就是为了让这两种入口得到一致的学习率。

Learning-rate schedulers have two modes: step forward during training, or jump to a specific epoch after checkpointing or manual control. PyTorch keeps both the recursive update and the closed-form formula so both paths produce the same learning rate.

## 代码 / The code

`pytorch/pytorch` — [`torch/optim/lr_scheduler.py`](https://github.com/pytorch/pytorch/blob/main/torch/optim/lr_scheduler.py#L914-L979)

```python
class ExponentialLR(LRScheduler):
    """Decays the learning rate of each parameter group by gamma every epoch."""

    def __init__(
        self,
        optimizer: Optimizer,
        gamma: float,
        last_epoch: int = -1,
    ) -> None:
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    @override
    def get_lr(self) -> list[float | Tensor]:
        r"""Compute the next learning rate for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups`.
        """
        _warn_get_lr_called_within_step(self)
        # when loading from a checkpoint, we don't want _initial_step (called from the constructor)
        # to update the lr one more step ahead of itself.
        if self._is_initial:
            return _param_groups_val_list(self.optimizer, "lr")
        return [group["lr"] * self.gamma for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        r"""Compute learning rates for each of the optimizer's
        :attr:`~torch.optim.Optimizer.param_groups` at :attr:`last_epoch` using
        a closed-form formula.
        """
        return [base_lr * self.gamma**self.last_epoch for base_lr in self.base_lrs]
```

## 逐行讲解 / What's happening

1. **第 934-941 行 / Lines 934-941**:
   - 中文: 构造函数只保存 `gamma`，其余调度器状态交给基类管理。
   - English: The constructor stores only `gamma`; shared scheduler state is delegated to the base class.
2. **第 961-966 行 / Lines 961-966**:
   - 中文: 普通 `step()` 路径读取当前 optimizer 里的 `lr`，再乘一次 `gamma`。
   - English: The normal `step()` path reads the current optimizer lr and multiplies it by `gamma` once.
3. **第 962-965 行 / Lines 962-965**:
   - 中文: 初始化时不额外前进一步，避免从 checkpoint 加载后 lr 被多衰减一次。
   - English: During initialization it does not advance the lr, avoiding an extra decay after checkpoint loading.
4. **第 967-979 行 / Lines 967-979**:
   - 中文: 闭式路径不看当前 lr，而是从 `base_lrs` 和 `last_epoch` 直接计算。
   - English: The closed-form path ignores current lr and computes directly from `base_lrs` and `last_epoch`.

## 类比 / The analogy

这像每个月把房租打 95 折。你可以从这个月金额继续乘 0.95，也可以拿第一月房租直接乘 `0.95 ** 月数`。只要没有中途人为改价，两种算法应该一样。

It is like applying a 5 percent rent discount every month. You can multiply this month's rent by `0.95`, or compute from the first month with `0.95 ** months`. If nobody manually edits the rent in between, both agree.

## 自己跑一遍 / Try it yourself

```python
import math

base_lr = 0.1
gamma = 0.9

recursive = base_lr
for _ in range(3):
    recursive *= gamma

closed_form = base_lr * gamma ** 3

print(round(recursive, 6))
print(round(closed_form, 6))
print(math.isclose(recursive, closed_form))
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
0.0729
0.0729
True
```

这个例子说明递推和闭式只是同一条指数曲线的两种入口。

This example shows that the recursive update and closed form are two entry points into the same exponential curve.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **`StepLR`** / **`StepLR`**: 递推按 milestone 乘 `gamma`，闭式用整除次数。 / The recursive path multiplies at milestones, while the closed form counts completed intervals.
- **`MultiStepLR`** / **`MultiStepLR`**: 闭式用 `bisect_right` 数过了多少 milestone。 / Its closed form counts passed milestones with `bisect_right`.
- **scheduler checkpointing** / **scheduler checkpointing**: 恢复训练时需要跳到正确 epoch。 / Resumed training needs to land on the correct epoch.

## 注意事项 / Caveats / when it breaks

- **外部修改 lr 会破坏等价性** / **External lr edits break equivalence**: 如果训练中手动改了 optimizer lr，递推路径会继承改动，闭式路径不会。 / If code mutates optimizer lr manually, the recursive path inherits it and the closed form does not.
- **`gamma` 不校验范围** / **`gamma` is not range-checked**: 大于 1 就会指数升高。 / Values above 1 make the lr grow exponentially.
- **不要直接调用 `get_lr()` 看当前值** / **Do not call `get_lr()` just to inspect**: PyTorch 推荐用 `get_last_lr()`。 / PyTorch recommends `get_last_lr()` for inspection.

## 延伸阅读 / Further reading

- PyTorch scheduler source: https://github.com/pytorch/pytorch/blob/main/torch/optim/lr_scheduler.py
- PyTorch learning-rate scheduler docs: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
