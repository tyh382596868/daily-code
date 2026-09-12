---
date: 2026-07-17
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/optim/lr_scheduler.py
permalink: https://github.com/pytorch/pytorch/blob/main/torch/optim/lr_scheduler.py#L1842-L1918
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, optimizer, scheduler, learning-rate]
---

# PyTorch CosineAnnealingWarmRestarts：余弦下降后重启周期 / PyTorch CosineAnnealingWarmRestarts: Cosine Decay, Then Restart the Cycle

> **一句话 / In one line**: PyTorch 用 `T_cur` 记录当前周期内位置，用 `T_i` 记录周期长度，重启时把学习率拉回 base lr。 / PyTorch tracks the current position with `T_cur`, the cycle length with `T_i`, and resets the LR to the base LR at each restart.

## 为什么重要 / Why this matters

warm restart 不是简单的“每 N 步重置”。当 `T_mult > 1` 时，每个周期会变长；当用户传入显式 epoch 时，还要能直接算出落在哪个周期。这个 scheduler 的价值在于把“余弦曲线 + 周期状态机 + checkpoint 可恢复状态”收进一个小类。

A warm restart is not just "reset every N steps." With `T_mult > 1`, cycles grow longer; with an explicit epoch, the scheduler must jump directly to the right cycle. This class packages the cosine curve, cycle state machine, and checkpointable state.

## 代码 / The code

`pytorch/pytorch` — [`torch/optim/lr_scheduler.py`](https://github.com/pytorch/pytorch/blob/main/torch/optim/lr_scheduler.py)

```python
class CosineAnnealingWarmRestarts(LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0.0, last_epoch=-1):
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            self.eta_min + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        if epoch is None:
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
```

## 逐行讲解 / What's happening

1. **`T_0` 是首周期 / `T_0` is the first cycle**: 中文: 第一次重启前走多少步。 English: how many steps before the first restart.
2. **`T_i` 是当前周期长度 / `T_i` is current cycle length**: 中文: 重启后可按 `T_mult` 扩大。 English: after restart it can grow by `T_mult`.
3. **`T_cur` 是周期内坐标 / `T_cur` is position in the cycle**: 中文: 余弦公式只看当前周期进度。 English: the cosine formula only needs progress inside the current cycle.
4. **谷底后回到峰值 / Trough then peak**: 中文: 当 `T_cur >= T_i`，先减掉旧周期长度，再进入下一周期。 English: once `T_cur >= T_i`, subtract the old length and begin the next cycle.
5. **学习率不直接记忆曲线 / The curve is not stored**: 中文: 只保存状态变量，LR 每次即时计算。 English: it stores state variables and computes LR on demand.

## 类比 / The analogy

像间歇训练跑步：每圈先从快跑慢慢降速，到终点后重新提速；如果 `T_mult=2`，后面每圈距离翻倍。

It is like interval running: start fast, slow down through a lap, then restart fast; with `T_mult=2`, each later lap is twice as long.

## 自己跑一遍 / Try it yourself

```python
import math

base_lr, eta_min, T_i, T_cur = 0.1, 0.0, 4, 0
for step in range(10):
    lr = eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2
    print(step, round(lr, 4), "T_cur", T_cur, "T_i", T_i)
    T_cur += 1
    if T_cur >= T_i:
        T_cur -= T_i
        T_i *= 2
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
0 0.1 T_cur 0 T_i 4
1 0.0854 T_cur 1 T_i 4
2 0.05 T_cur 2 T_i 4
3 0.0146 T_cur 3 T_i 4
4 0.1 T_cur 0 T_i 8
```

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **SGDR**: warm restart 的原始训练配方。 / The original warm-restart schedule.
- **CosineAnnealingLR**: 没有 restart 的同一条余弦曲线。 / The same cosine curve without restarts.

## 注意事项 / Caveats / when it breaks

- **先 `optimizer.step()` 再 `scheduler.step()` / Step optimizer first**: 顺序反了会跳过第一个 LR。 / Wrong order skips the first LR.
- **`T_mult` 必须是整数且 >= 1 / `T_mult` must be an integer >= 1**: 这是状态机假设。 / The state machine assumes it.
- **显式 epoch 路径更复杂 / Explicit epoch path is more complex**: resume 或手动跳步时要依赖闭式周期定位。 / Resume/manual jumps rely on closed-form cycle placement.

## 延伸阅读 / Further reading

- [PyTorch LR schedulers](https://github.com/pytorch/pytorch/blob/main/torch/optim/lr_scheduler.py)
- [SGDR paper](https://arxiv.org/abs/1608.03983)
