---
date: 2026-08-29
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/optim/lr_scheduler.py
permalink: https://github.com/pytorch/pytorch/blob/main/torch/optim/lr_scheduler.py#L648-L698
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, lr-scheduler, warmup]
---

# PyTorch ConstantLR：先压低，再还原 / PyTorch ConstantLR: Lower First, Restore Later

> **一句话 / In one line**: `ConstantLR` 在开头把学习率乘上一个固定因子，到指定步数后再把学习率除回来。 / `ConstantLR` multiplies the learning rate by a fixed factor at the beginning, then divides it back after the configured number of steps.

## 为什么重要 / Why this matters

很多 warmup 不是线性升高，而是先用一个更小的常数学习率稳定几步。PyTorch 这里把“进入常数阶段”和“退出常数阶段”都写成递推更新，所以它能和 optimizer 当前的 lr 状态组合。

Some warmups do not ramp linearly; they hold a smaller constant learning rate for a few steps. PyTorch expresses both entering and leaving that constant phase as recursive updates, so the scheduler composes with the optimizer's current learning-rate state.

## 代码 / The code

`pytorch/pytorch` — [`torch/optim/lr_scheduler.py`](https://github.com/pytorch/pytorch/blob/main/torch/optim/lr_scheduler.py#L648-L698)

```python
class ConstantLR(LRScheduler):
    """Multiply the learning rate of each parameter group by a small constant factor.

    The multiplication is done until the number of epoch reaches a pre-defined milestone:
    total_iters. Notice that such decay can happen simultaneously with other changes to
    the learning rate from outside this scheduler.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        factor: float = 1.0 / 3,
        total_iters: int = 5,
        last_epoch: int = -1,
    ):
        if factor > 1.0 or factor < 0:
            raise ValueError(
                "Constant multiplicative factor expected to be between 0 and 1."
            )

        self.factor = factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        _warn_get_lr_called_within_step(self)

        if self.last_epoch == 0:
            return [group["lr"] * self.factor for group in self.optimizer.param_groups]

        if self.last_epoch != self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]

        return [
            group["lr"] * (1.0 / self.factor) for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        return [
            base_lr
            * (self.factor + (self.last_epoch >= self.total_iters) * (1 - self.factor))
            for base_lr in self.base_lrs
        ]
```

## 逐行讲解 / What's happening

1. **第 665-677 行 / Lines 665-677**:
   - 中文: `factor` 必须在 `[0, 1]`，否则“常数衰减”会变成放大学习率或负学习率。
   - English: `factor` must stay in `[0, 1]`; otherwise a "constant decay" could amplify the learning rate or make it negative.
2. **第 682-684 行 / Lines 682-684**:
   - 中文: 第一次 step 时，把每个 param group 的当前 lr 乘以 `factor`。
   - English: On the first scheduler step, each parameter group's current learning rate is multiplied by `factor`.
3. **第 686-691 行 / Lines 686-691**:
   - 中文: 中间阶段保持不变；到 `total_iters` 那一步，再乘以 `1 / factor` 恢复原比例。
   - English: The middle phase leaves values unchanged; at `total_iters`, multiplying by `1 / factor` restores the previous scale.
4. **第 693-698 行 / Lines 693-698**:
   - 中文: 闭式版本不看当前 lr，而是从 `base_lrs` 直接算某个 epoch 应该是多少。
   - English: The closed form ignores the current lr and computes the epoch's value directly from `base_lrs`.

## 类比 / The analogy

这像开车出停车场：前几米限速很低，出了闸口之后再恢复正常速度。中间不是一直加速，而是先稳住，再切回原速度。

It is like driving out of a parking lot: the first few meters have a low speed limit, and after the gate you return to normal speed. The car does not ramp continuously; it holds steady, then switches back.

## 自己跑一遍 / Try it yourself

```python
base_lr = 0.09
factor = 1 / 3
total_iters = 3
lr = base_lr

for epoch in range(6):
    if epoch == 0:
        lr *= factor
    elif epoch == total_iters:
        lr *= 1 / factor
    print(epoch, round(lr, 3))
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
0 0.03
1 0.03
2 0.03
3 0.09
4 0.09
5 0.09
```

注意第 3 步不是再衰减，而是把前面的缩放撤销。

Notice that step 3 does not decay again; it cancels the earlier scaling.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LinearLR** / **LinearLR**: 用递推系数逐步改变 lr，而不是每次重设绝对值。 / It recursively adjusts lr with a factor instead of resetting an absolute value every time.
- **SequentialLR** / **SequentialLR**: 到 milestone 时切换 scheduler，并修正初始化 step 的影响。 / It switches schedulers at milestones and accounts for initialization effects.
- **optimizer wrappers** / **optimizer wrappers**: 外部可能也改 lr，所以递推写法更容易组合。 / External code may also change lr, making recursive updates easier to compose.

## 注意事项 / Caveats / when it breaks

- **`factor=0` 不能恢复** / **`factor=0` cannot restore**: 递推退出阶段会需要 `1 / factor`。 / The recursive exit needs `1 / factor`.
- **step 顺序重要** / **Step order matters**: 先 `optimizer.step()` 再 `scheduler.step()` 才符合 PyTorch 习惯。 / Calling `optimizer.step()` before `scheduler.step()` matches PyTorch's expected pattern.
- **闭式和递推语义不同** / **Closed form and recursive form differ**: 外部改过 lr 时，两者可能给出不同结果。 / If external code changed lr, the two forms can produce different values.

## 延伸阅读 / Further reading

- PyTorch LR schedulers: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
- PyTorch `lr_scheduler.py`: https://github.com/pytorch/pytorch/blob/main/torch/optim/lr_scheduler.py
