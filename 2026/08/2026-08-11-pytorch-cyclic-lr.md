---
date: 2026-08-11
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/optim/lr_scheduler.py
permalink: https://github.com/pytorch/pytorch/blob/b8e5ddf0324963dc5421832d7b55171f7d4d7a5d/torch/optim/lr_scheduler.py#L2001-L2069
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, lr-scheduler]
---

# PyTorch CyclicLR：学习率按 batch 走三角波 / PyTorch CyclicLR: A Batch-Level Triangular Wave

> **一句话 / In one line**: `CyclicLR.get_lr()` 用当前 batch 计数算出三角波学习率，并可同步反向调 momentum。 / `CyclicLR.get_lr()` maps the current batch count to a triangular learning-rate wave and can inversely cycle momentum.

## 为什么重要 / Why this matters

很多 scheduler 按 epoch 走，`CyclicLR` 按 batch 走。它的核心不是“第几轮乘一个常数”，而是把 `last_epoch` 当作已处理 batch 数，算出当前处在周期上升段还是下降段，再把学习率推到 `base_lr` 和 `max_lr` 之间。

Many schedulers step per epoch; `CyclicLR` steps per batch. Its core is not a simple epoch multiplier. It treats `last_epoch` as a batch index, decides where that batch sits in the up/down cycle, and interpolates between `base_lr` and `max_lr`.

## 代码 / The code

`pytorch/pytorch` — [`torch/optim/lr_scheduler.py`](https://github.com/pytorch/pytorch/blob/b8e5ddf0324963dc5421832d7b55171f7d4d7a5d/torch/optim/lr_scheduler.py#L2001-L2069)

```python
def get_lr(self) -> list[float | Tensor]:
    r"""Compute the next learning rate for each of the optimizer's
    :attr:`~torch.optim.Optimizer.param_groups`.

    Advances each ``group["lr"]`` in the optimizer's
    :attr:`~torch.optim.Optimizer.param_groups` along a cycle between the
    group's ``base_lr`` and ``max_lr`` using :meth:`scale_fn`.
    """
    _warn_get_lr_called_within_step(self)

    cycle = math.floor(1 + self.last_epoch / self.total_size)
    x = 1.0 + self.last_epoch / self.total_size - cycle
    if x <= self.step_ratio:
        scale_factor = x / self.step_ratio
    else:
        scale_factor = (x - 1) / (self.step_ratio - 1)

    lrs = []
    for base_lr, max_lr in zip(self.base_lrs, self.max_lrs, strict=True):
        base_height = (max_lr - base_lr) * scale_factor
        if self.scale_mode == "cycle":
            lr = base_lr + base_height * self.scale_fn(cycle)
        else:
            lr = base_lr + base_height * self.scale_fn(self.last_epoch)
        lrs.append(lr)

    if self.cycle_momentum:
        momentums = []
        for base_momentum, max_momentum in zip(
            self.base_momentums, self.max_momentums, strict=True
        ):
            base_height = (max_momentum - base_momentum) * scale_factor
            if self.scale_mode == "cycle":
                momentum = max_momentum - base_height * self.scale_fn(cycle)
            else:
                momentum = max_momentum - base_height * self.scale_fn(
                    self.last_epoch
                )
            momentums.append(momentum)
        for param_group, momentum in zip(
            self.optimizer.param_groups, momentums, strict=True
        ):
            if self.use_beta1:
                param_group["betas"] = (momentum, *param_group["betas"][1:])
            else:
                param_group["momentum"] = momentum

    return lrs
```

## 逐行讲解 / What's happening

1. **第 2032-2037 行 / Lines 2032-2037 (triangle position)**:
   - 中文: `cycle` 是第几个周期，`x` 是周期内位置；上升段线性从 0 到 1，下降段再从 1 回到 0。
   - English: `cycle` is the cycle number and `x` is the within-cycle position; the up half rises from 0 to 1 and the down half falls back to 0.
2. **第 2039-2046 行 / Lines 2039-2046 (learning rate)**:
   - 中文: `base_height` 是当前三角波高度，再乘 `scale_fn` 实现 `triangular2` 或 `exp_range` 的幅度衰减。
   - English: `base_height` is the current triangular amplitude, then `scale_fn` adds policies like `triangular2` or `exp_range`.
3. **第 2048-2067 行 / Lines 2048-2067 (inverse momentum)**:
   - 中文: momentum 用 `max_momentum - base_height` 的形式反向移动，学习率高时 momentum 低。
   - English: Momentum moves inversely through `max_momentum - base_height`: high learning rate, lower momentum.

## 类比 / The analogy

像跑步机的坡度程序：前半段坡度慢慢升高，后半段慢慢降下来；如果坡度高，扶手阻尼就降低一点，让你能迈开。

It is like a treadmill incline program: the incline rises for the first half and falls for the second half. When incline is high, the handrail resistance drops so you can move.

## 自己跑一遍 / Try it yourself

```python
import math

base_lr, max_lr = 0.001, 0.006
total_size, step_ratio = 6.0, 0.5
for last_epoch in range(7):
    cycle = math.floor(1 + last_epoch / total_size)
    x = 1 + last_epoch / total_size - cycle
    scale = x / step_ratio if x <= step_ratio else (x - 1) / (step_ratio - 1)
    lr = base_lr + (max_lr - base_lr) * scale
    print(last_epoch, round(lr, 4))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
0 0.001
1 0.0027
2 0.0043
3 0.006
4 0.0043
5 0.0027
6 0.001
```

这里 `last_epoch` 实际是 batch 计数；第 3 个 batch 到达峰值，之后开始下降。

Here `last_epoch` is really a batch counter. Batch 3 reaches the peak, then the schedule descends.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **OneCycleLR** / **OneCycleLR**: 也是 batch-level schedule，但把 warmup、anneal 和 momentum 组合得更强。 / Also a batch-level schedule, but with a stronger warmup/anneal/momentum recipe.
- **Cosine warm restarts** / **Cosine warm restarts**: 同样使用周期位置，只是波形从三角波换成余弦。 / It also uses within-cycle position, but swaps the triangle for a cosine.

## 注意事项 / Caveats / when it breaks

- **调用频率要对** / **Call frequency matters**: `scheduler.step()` 应该每个 batch 调一次，不是每个 epoch。 / `scheduler.step()` should run after every batch, not every epoch.
- **optimizer 要支持 momentum** / **The optimizer must support momentum**: `cycle_momentum=True` 时，不支持 `momentum` 或 `betas` 的 optimizer 会报错。 / With `cycle_momentum=True`, optimizers without `momentum` or `betas` fail.

## 延伸阅读 / Further reading

- [PyTorch `CyclicLR.get_lr`](https://github.com/pytorch/pytorch/blob/b8e5ddf0324963dc5421832d7b55171f7d4d7a5d/torch/optim/lr_scheduler.py#L2001-L2069)

