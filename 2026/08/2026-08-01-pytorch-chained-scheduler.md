---
date: 2026-08-01
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/optim/lr_scheduler.py
permalink: https://github.com/pytorch/pytorch/blob/main/torch/optim/lr_scheduler.py#L1057-L1128
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, lr-scheduler, composition]
---

# PyTorch ChainedScheduler：多个学习率调度器串起来走 / PyTorch ChainedScheduler: Step Several LR Schedulers in Sequence

> **一句话 / In one line**: `ChainedScheduler` 保存一组 scheduler，每次 `step()` 按顺序调用它们，并把最后一个 scheduler 的 LR 当成结果。 / `ChainedScheduler` stores multiple schedulers, calls each on every `step()`, and reports the last scheduler's LR.

## 为什么重要 / Why this matters

训练配方经常不是单一曲线：你可能同时要 warmup、周期震荡、再乘一个衰减因子。`ChainedScheduler` 的设计很小，但它把“调度器组合”做成了明确接口，而不是让用户在训练循环里手写顺序。

Training recipes are rarely one curve: you may want warmup, cyclic motion, and an extra decay factor. `ChainedScheduler` is small, but it turns scheduler composition into an explicit API instead of hand-written ordering inside the training loop.

## 代码 / The code

`pytorch/pytorch` — [`torch/optim/lr_scheduler.py`](https://github.com/pytorch/pytorch/blob/main/torch/optim/lr_scheduler.py#L1057-L1128)

```python
class ChainedScheduler(LRScheduler):
    def __init__(self, schedulers: Sequence[LRScheduler], optimizer: Optional[Optimizer] = None):
        if len(schedulers) < 1:
            raise ValueError(f"{self.__class__.__name__} expects at least one scheduler, but got no scheduler.")

        optimizer = optimizer or schedulers[0].optimizer
        for scheduler_idx, scheduler in enumerate(schedulers):
            if not hasattr(scheduler, "optimizer"):
                raise TypeError(f"{scheduler_idx} is not an LRScheduler")
            if scheduler.optimizer != optimizer:
                raise ValueError("ChainedScheduler expects all schedulers to belong to the same optimizer")
            if isinstance(scheduler, ReduceLROnPlateau):
                raise ValueError("ChainedScheduler does not support ReduceLROnPlateau scheduler")

        self._schedulers = schedulers
        self.optimizer = optimizer
        self._last_lr = [group["lr"] for group in self._schedulers[-1].optimizer.param_groups]

    def step(self):
        for scheduler in self._schedulers:
            scheduler.step()
        self._last_lr = [group["lr"] for group in self._schedulers[-1].optimizer.param_groups]

    def state_dict(self):
        state = {key: value for key, value in self.__dict__.items() if key not in ("optimizer", "_schedulers")}
        state["_schedulers"] = [None] * len(self._schedulers)
        for idx, s in enumerate(self._schedulers):
            state["_schedulers"][idx] = s.state_dict()
        return state

    def load_state_dict(self, state_dict):
        _schedulers = state_dict.pop("_schedulers")
        self.__dict__.update(state_dict)
        state_dict["_schedulers"] = _schedulers
        for idx, s in enumerate(_schedulers):
            self._schedulers[idx].load_state_dict(s)
```

## 逐行讲解 / What's happening

1. **第 1058-1069 行 / Lines 1058-1069 (validation)**:
   - 中文: 至少要有一个 scheduler，并且所有 scheduler 必须绑定同一个 optimizer。
   - English: There must be at least one scheduler, and every scheduler must point at the same optimizer.
2. **第 1070-1071 行 / Lines 1070-1071 (`ReduceLROnPlateau`)**:
   - 中文: plateau scheduler 依赖指标参数，不适合这个无参数 `step()` 链式接口。
   - English: Plateau scheduling needs a metric argument, so it does not fit this no-argument chained `step()` interface.
3. **第 1074-1080 行 / Lines 1074-1080 (sequential step)**:
   - 中文: 每次训练 step 后，链里的 scheduler 一个接一个改同一组 param group。
   - English: After each training step, each scheduler in the chain mutates the same param groups in order.
4. **第 1082-1128 行 / Lines 1082-1128 (nested state)**:
   - 中文: checkpoint 时不保存 optimizer 本身，只保存每个子 scheduler 的状态。
   - English: Checkpointing skips the optimizer object and stores each child scheduler's state.

## 类比 / The analogy

这像给音响信号串效果器：先过压缩器，再过均衡器，再过混响。每个盒子都改同一条信号，最终听到的是最后一个盒子输出。

It is like chaining audio pedals: compressor, then EQ, then reverb. Each box modifies the same signal, and the final sound is whatever leaves the last box.

## 自己跑一遍 / Try it yourself

```python
lr = 1.0

def warmup(x, step):
    return x * min(1.0, (step + 1) / 3)

def decay(x, step):
    return x * (0.9 ** step)

for step in range(5):
    out = lr
    for scheduler in (warmup, decay):
        out = scheduler(out, step)
    print(step, round(out, 4))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
0 0.3333
1 0.6
2 0.81
3 0.729
4 0.6561
```

顺序很关键：warmup 先决定基础倍率，decay 再在这个结果上继续缩放。

Ordering matters: warmup determines the base multiplier first, then decay scales that result.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **SequentialLR** / **SequentialLR**: 按 milestone 切换 scheduler，而不是每一步全部执行。 / It switches schedulers at milestones instead of running all of them every step.
- **Optimizer wrappers** / **Optimizer wrappers**: 多个小规则包住同一个 optimizer state，是训练系统里常见的组合方式。 / Multiple small rules around one optimizer state are a common composition style in training systems.

## 注意事项 / Caveats / when it breaks

- **顺序是语义的一部分** / **Order is part of the semantics**: 两个 scheduler 交换顺序可能得到不同 LR。 / Swapping two schedulers can produce a different LR.
- **不能混 optimizer** / **Do not mix optimizers**: 链里的 scheduler 必须改同一个 optimizer，否则状态不可解释。 / Every scheduler in the chain must mutate the same optimizer or the state is not meaningful.

## 延伸阅读 / Further reading

- [PyTorch ChainedScheduler source](https://github.com/pytorch/pytorch/blob/main/torch/optim/lr_scheduler.py#L1057-L1128)

