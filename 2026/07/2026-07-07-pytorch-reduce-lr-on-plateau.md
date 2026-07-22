---
date: 2026-07-07
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/optim/lr_scheduler.py
permalink: https://github.com/pytorch/pytorch/blob/main/torch/optim/lr_scheduler.py#L1676-L1772
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, pytorch, lr-scheduler]
---

# PyTorch ReduceLROnPlateau：学习率调度器也会看验证集脸色 / PyTorch ReduceLROnPlateau: An LR Scheduler That Watches Validation Metrics

> **一句话 / In one line**: `ReduceLROnPlateau` 不是按 epoch 公式衰减，而是把验证指标、patience、cooldown 和最小学习率组成一个小状态机。 / `ReduceLROnPlateau` is not formula-per-epoch decay; it is a small state machine over metrics, patience, cooldown, and minimum LR.

## 为什么重要 / Why this matters

很多 scheduler 只关心“现在第几步”，但实际训练中更常见的问题是：模型已经不进步了，还要不要继续用当前学习率硬冲？这段代码展示了 PyTorch 如何把“指标是否显著改善”变成可恢复、可序列化的学习率状态机。

Many schedulers only ask “which step are we on?” In real training, the more practical question is whether the model has stopped improving. This code shows how PyTorch turns “has the metric improved enough?” into a resumable, serializable LR state machine.

## 代码 / The code

`pytorch/pytorch` — [`torch/optim/lr_scheduler.py`](https://github.com/pytorch/pytorch/blob/main/torch/optim/lr_scheduler.py#L1676-L1772)

```python
self.patience = patience
self.cooldown = cooldown
self.eps = eps
self.last_epoch = 0
self._last_lr = _param_groups_val_list(self.optimizer, "lr")
self._init_is_better(mode=mode, threshold=threshold, threshold_mode=threshold_mode)
self._reset()

def _reset(self) -> None:
    self.best = self.mode_worse
    self.cooldown_counter = 0
    self.num_bad_epochs = 0

def step(self, metrics: SupportsFloat, epoch=None) -> None:
    current = float(metrics)
    if epoch is None:
        epoch = self.last_epoch + 1
    self.last_epoch = epoch

    if self._is_better(current, self.best):
        self.best = current
        self.num_bad_epochs = 0
    else:
        self.num_bad_epochs += 1

    if self.in_cooldown:
        self.cooldown_counter -= 1
        self.num_bad_epochs = 0

    if self.num_bad_epochs > self.patience:
        self._reduce_lr(epoch)
        self.cooldown_counter = self.cooldown
        self.num_bad_epochs = 0

def _reduce_lr(self, epoch) -> None:
    for i, param_group in enumerate(self.optimizer.param_groups):
        old_lr = float(param_group["lr"])
        new_lr = max(old_lr * self.factor, self.min_lrs[i])
        if old_lr - new_lr > self.eps:
            _update_param_group_val(param_group, "lr", new_lr)

def _is_better(self, a, best):
    if self.mode == "min" and self.threshold_mode == "rel":
        return a < best * (1.0 - self.threshold)
    elif self.mode == "min" and self.threshold_mode == "abs":
        return a < best - self.threshold
    elif self.mode == "max" and self.threshold_mode == "rel":
        return a > best * (self.threshold + 1.0)
    else:
        return a > best + self.threshold
```

## 逐行讲解 / What's happening

1. **第 1-7 行 / Lines 1-7 (state fields)**:
   - 中文: scheduler 存 `patience`、`cooldown`、`eps`，说明它关心历史状态而不是单个公式。
   - English: the scheduler stores `patience`, `cooldown`, and `eps`, so it depends on history rather than a single formula.
2. **第 14-25 行 / Lines 14-25 (`step`)**:
   - 中文: 每次把指标转成 float，然后判断是否比历史最好值更好。
   - English: each call converts the metric to `float` and checks whether it improves over the best value so far.
3. **第 27-29 行 / Lines 27-29 (cooldown)**:
   - 中文: 刚降过学习率时，坏 epoch 不计数，给新学习率一个观察窗口。
   - English: right after an LR drop, bad epochs are ignored so the new LR gets a window to show effect.
4. **第 31-34 行 / Lines 31-34 (patience trigger)**:
   - 中文: 坏 epoch 数超过 patience 才真正调用 `_reduce_lr`。
   - English: `_reduce_lr` runs only when bad epochs exceed patience.
5. **第 36-41 行 / Lines 36-41 (`min_lrs`, `eps`)**:
   - 中文: 新学习率不能低于下限；变化太小则跳过，避免浮点噪声导致无意义更新。
   - English: the new LR is clamped by a minimum; tiny changes are skipped to avoid floating-point noise updates.

## 类比 / The analogy

像开车上坡：如果速度连续几次都没有改善，司机才降档；刚降档后先等一小段路观察，别马上又降档。

It is like driving uphill: if speed fails to improve several times in a row, the driver shifts down; after shifting, they wait a short stretch before shifting again.

## 自己跑一遍 / Try it yourself

```python
best = float("inf")
bad = 0
lr = 0.1
for loss in [1.0, 0.9, 0.91, 0.92, 0.89, 0.90, 0.91]:
    if loss < best * (1 - 1e-4):
        best, bad = loss, 0
    else:
        bad += 1
    if bad > 1:
        lr *= 0.5
        bad = 0
    print(loss, round(lr, 3), best)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
1.0 0.1 1.0
0.9 0.1 0.9
0.91 0.1 0.9
0.92 0.05 0.9
0.89 0.05 0.89
0.9 0.05 0.89
0.91 0.025 0.89
```

你会看到学习率只在“连续坏消息超过 patience”后下降。

You can see LR drops only after consecutive bad news exceeds patience.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Early stopping** / **Early stopping**: 同样用 `best + patience` 判断何时停训。 / It uses the same `best + patience` pattern to decide when to stop.
- **Checkpoint selection** / **Checkpoint selection**: 保存 best checkpoint 也依赖同一个 `_is_better` 逻辑。 / Best-checkpoint saving depends on the same improvement predicate.

## 注意事项 / Caveats / when it breaks

- **调用顺序不同** / **The call order differs**: 它应该在验证后用 metric 调 `step(metric)`。 / It should be called after validation with `step(metric)`.
- **metric 噪声大时会误触发** / **Noisy metrics can trigger false drops**: 需要调大 `threshold` 或 `patience`。 / Increase `threshold` or `patience` when metrics are noisy.

## 延伸阅读 / Further reading

- [PyTorch `ReduceLROnPlateau`](https://github.com/pytorch/pytorch/blob/main/torch/optim/lr_scheduler.py#L1676-L1772)
