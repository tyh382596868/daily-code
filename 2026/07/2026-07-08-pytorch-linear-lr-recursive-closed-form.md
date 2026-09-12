---
date: 2026-07-08
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/optim/lr_scheduler.py
permalink: https://github.com/pytorch/pytorch/blob/bde5b5af72915d3187d63bfd3f67a1d8afc2f30e/torch/optim/lr_scheduler.py#L876-L1000
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, lr-scheduler]
---

# PyTorch LinearLR：递推更新也要等价闭式公式 / PyTorch LinearLR: Recursive Updates Must Match the Closed Form

> **一句话 / In one line**: `LinearLR` 用递推公式逐步放大学习率，同时保留闭式公式来支持按 epoch 跳转。 / `LinearLR` updates learning rates recursively step by step, while keeping a closed form for explicit epoch jumps.

## 为什么重要 / Why this matters

学习率调度器不只是画一条曲线。PyTorch 的 scheduler 要同时支持 `scheduler.step()` 的递推调用，以及 `scheduler.step(epoch)` 这种直接跳到指定 epoch 的调用。`LinearLR` 展示了一个工程细节：递推比例看起来复杂，是为了让每一步从当前 lr 走到下一点，同时和闭式公式严格对齐。

An LR scheduler is not just a plotted curve. PyTorch schedulers must support normal recursive `scheduler.step()` calls and explicit jumps such as `scheduler.step(epoch)`. `LinearLR` shows the engineering detail: the recursive factor looks complex because it advances from the current lr to the next point while matching the closed-form schedule.

## 代码 / The code

`pytorch/pytorch` — [`torch/optim/lr_scheduler.py`](https://github.com/pytorch/pytorch/blob/bde5b5af72915d3187d63bfd3f67a1d8afc2f30e/torch/optim/lr_scheduler.py#L876-L1000)

```python
class LinearLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        start_factor: float = 1.0 / 3,
        end_factor: float = 1.0,
        total_iters: int = 5,
        last_epoch: int = -1,
    ) -> None:
        if start_factor > 1.0 or start_factor <= 0:
            raise ValueError(
                "Starting multiplicative factor expected to be greater than 0 and less or equal to 1."
            )

        if end_factor > 1.0 or end_factor < 0:
            raise ValueError(
                "Ending multiplicative factor expected to be between 0 and 1."
            )

        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float | Tensor]:
        _warn_get_lr_called_within_step(self)

        if self.last_epoch == 0:
            return [
                group["lr"] * self.start_factor for group in self.optimizer.param_groups
            ]

        if self._is_initial or self.last_epoch > self.total_iters:
            return _param_groups_val_list(self.optimizer, "lr")

        return [
            group["lr"]
            * (
                1.0
                + (self.end_factor - self.start_factor)
                / (
                    self.total_iters * self.start_factor
                    + (self.last_epoch - 1) * (self.end_factor - self.start_factor)
                )
            )
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        return [
            base_lr
            * (
                self.start_factor
                + (self.end_factor - self.start_factor)
                * min(self.total_iters, self.last_epoch)
                / self.total_iters
            )
            for base_lr in self.base_lrs
        ]
```

## 逐行讲解 / What's happening

1. **第 917-927 行 / Lines 917-927 (参数校验)**:
   - 中文: `start_factor` 必须在 `(0, 1]`，`end_factor` 必须在 `[0, 1]`。这保证它是“缩放 base lr”的因子。
   - English: `start_factor` must be in `(0, 1]`, and `end_factor` in `[0, 1]`. They are multipliers of the base lr.
2. **第 934-938 行 / Lines 934-938 (第 0 步)**:
   - 中文: 第一次 step 把当前 lr 乘上 `start_factor`，进入 warmup 起点。
   - English: The first step multiplies the current lr by `start_factor`, entering the warmup starting point.
3. **第 943-955 行 / Lines 943-955 (递推比例)**:
   - 中文: 这里不是直接算 base lr，而是算“从上一点到下一点要乘多少”。这让它能连续修改 optimizer 当前值。
   - English: This does not recompute from the base lr; it computes the multiplier from the previous point to the next one.
4. **第 957-969 行 / Lines 957-969 (闭式公式)**:
   - 中文: 当用户传入显式 epoch 时，PyTorch 可以直接用 base lr 算出任意位置的值。
   - English: When the user passes an explicit epoch, PyTorch can compute the value at that position directly from the base lr.

## 类比 / The analogy

递推公式像每天按昨天的存款加一点利息；闭式公式像直接查银行系统里第 N 天应该有多少钱。两条路径必须对上，否则恢复训练或跳 epoch 会出现学习率漂移。

The recursive formula is like adding interest to yesterday's bank balance. The closed form is like querying what the balance should be on day N. They must agree, or resumed training and epoch jumps drift.

## 自己跑一遍 / Try it yourself

```python
base_lr = 0.1
start, end, total = 0.2, 1.0, 4
lr = base_lr
for epoch in range(5):
    closed = base_lr * (start + (end - start) * min(total, epoch) / total)
    if epoch == 0:
        lr *= start
    elif epoch <= total:
        lr *= 1 + (end - start) / (total * start + (epoch - 1) * (end - start))
    print(epoch, round(lr, 4), round(closed, 4))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
0 0.02 0.02
1 0.04 0.04
2 0.06 0.06
3 0.08 0.08
4 0.1 0.1
```

最值得注意的是递推和闭式两列完全一致。递推代码不是炫技，而是为了保持 optimizer 当前 lr 的连续状态。

The important detail is that the recursive and closed-form columns match exactly. The recursive code is not cleverness; it preserves the optimizer's current lr state.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **`PolynomialLR`** / **`PolynomialLR`**: 也同时实现递推路径和 `_get_closed_form_lr()`。 / It also carries a recursive path and `_get_closed_form_lr()`.
- **Checkpoint resume** / **Checkpoint resume**: 恢复 scheduler state 后继续递推，不能和直接按 epoch 计算的轨迹冲突。 / After restoring scheduler state, recursive stepping must not diverge from epoch-based computation.

## 注意事项 / Caveats / when it breaks

- **不要手动改 optimizer lr** / **Do not mutate optimizer lr manually**: 递推路径依赖当前 `group["lr"]`，外部修改会改变轨迹。 / The recursive path depends on current `group["lr"]`; external mutation changes the curve.
- **超过 `total_iters` 后保持不变** / **It stays fixed after `total_iters`**: 代码在 warmup 完成后返回当前 lr。 / After warmup, the scheduler returns the current lr unchanged.

## 延伸阅读 / Further reading

- [PyTorch `LinearLR`](https://github.com/pytorch/pytorch/blob/bde5b5af72915d3187d63bfd3f67a1d8afc2f30e/torch/optim/lr_scheduler.py#L876-L1000)
