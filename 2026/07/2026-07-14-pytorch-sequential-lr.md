---
date: 2026-07-14
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/optim/lr_scheduler.py
permalink: https://github.com/pytorch/pytorch/blob/da9c79b096d4b842c9fe4ac42298380a78cf69f3/torch/optim/lr_scheduler.py#L1093-L1228
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, lr-scheduler, optimizer]
---

# PyTorch SequentialLR：按里程碑接力学习率调度器 / PyTorch SequentialLR: Relay LR Schedulers at Milestones

> **一句话 / In one line**: `SequentialLR` 把多个 scheduler 接成一条时间线，并在切换点重置新 scheduler 的局部步数。 / `SequentialLR` connects multiple schedulers into one timeline and resets the next scheduler's local step at the switching point.

## 为什么重要 / Why this matters

真实训练常见“先 warmup，再 decay”。如果手写 `if step < warmup`，checkpoint、resume、`last_epoch` 和多 param group 很容易出错。PyTorch 这段把每个阶段仍然交给原 scheduler，只额外维护 milestone 和当前阶段选择。

Real training often uses “warm up, then decay.” Hand-written `if step < warmup` logic can break checkpointing, resume, `last_epoch`, and multi-param-group behavior. This PyTorch code keeps each phase in its own scheduler and only adds milestone-based routing.

## 代码 / The code

`pytorch/pytorch` — [`torch/optim/lr_scheduler.py`](https://github.com/pytorch/pytorch/blob/da9c79b096d4b842c9fe4ac42298380a78cf69f3/torch/optim/lr_scheduler.py#L1093-L1228)

```python
class SequentialLR(LRScheduler):
    def __init__(self, optimizer, schedulers, milestones, last_epoch=-1) -> None:
        if len(schedulers) < 1:
            raise ValueError(...)

        for scheduler_idx, scheduler in enumerate(schedulers):
            if isinstance(scheduler, ReduceLROnPlateau):
                raise ValueError(...)
            if optimizer != scheduler.optimizer:
                raise ValueError(...)

        if len(milestones) != len(schedulers) - 1:
            raise ValueError(...)
        self._schedulers = schedulers
        self._milestones = milestones
        self.last_epoch = last_epoch + 1
        self.optimizer = optimizer

        for group in self.optimizer.param_groups:
            _update_param_group_val(group, "lr", group["initial_lr"])

        self.recursive_undo()

        idx = bisect_right(self._milestones, 0)
        self._schedulers[idx]._initial_step()
        self._last_lr = schedulers[idx].get_last_lr()

    def recursive_undo(self, sched=None) -> None:
        scheds = self if sched is None else sched
        if hasattr(scheds, "_schedulers"):
            for s in scheds._schedulers:
                self.recursive_undo(s)
        elif hasattr(scheds, "last_epoch"):
            scheds.last_epoch -= 1

    def step(self) -> None:
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        scheduler = self._schedulers[idx]
        if idx > 0 and self._milestones[idx - 1] == self.last_epoch:
            scheduler._update_lr(0)
        else:
            scheduler.step()
        self._last_lr = scheduler.get_last_lr()
```

## 逐行讲解 / What's happening

1. **检查 scheduler 属于同一个 optimizer / Check one optimizer**:
   - 中文: 所有子 scheduler 必须包同一个 optimizer，否则不同阶段会改不同参数组。
   - English: Every child scheduler must wrap the same optimizer, otherwise phases would update different param groups.
2. **milestone 数量合约 / Milestone count contract**:
   - 中文: N 个 scheduler 只需要 N-1 个切换点。
   - English: N schedulers need N-1 switch points.
3. **撤销初始化 step / Undo constructor steps**:
   - 中文: 子 scheduler 构造时可能已经前进一步，`recursive_undo` 把它们拉回统一起点。
   - English: Child schedulers may advance during construction; `recursive_undo` pulls them back to a common start.
4. **切换点局部重启 / Local restart at switch**:
   - 中文: 到达 milestone 时调用 `_update_lr(0)`，让新阶段从自己的第 0 步开始。
   - English: At a milestone, `_update_lr(0)` starts the new phase at its local step 0.

## 类比 / The analogy

像接力赛：跑道总时间是连续的，但每个选手都从自己的起跑线开始计步。裁判只负责在交接区把棒交给下一个人。

It is a relay race: race time is continuous, but each runner starts counting from their own start line. The judge only hands off at exchange zones.

## 自己跑一遍 / Try it yourself

```python
milestones = [3]
schedules = [
    lambda local: 0.1 * (local + 1) / 3,
    lambda local: 0.1 * (0.5 ** local),
]
for step in range(7):
    phase = 0 if step < milestones[0] else 1
    local = step if phase == 0 else step - milestones[0]
    print(step, round(schedules[phase](local), 4))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
0 0.0333
1 0.0667
2 0.1
3 0.1
4 0.05
5 0.025
6 0.0125
```

第 3 步切到第二个 scheduler，但第二个 scheduler 的本地步数从 0 开始。

Step 3 switches to the second scheduler, but that scheduler's local step starts at 0.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **warmup + cosine decay** / **warmup + cosine decay**: 训练脚本常把线性 warmup 和余弦下降串起来。 / Training scripts often chain linear warmup and cosine decay.
- **pipeline stage handoff** / **pipeline stage handoff**: 分阶段系统常保留全局时钟，同时让每个阶段维护自己的局部状态。 / Staged systems often keep a global clock while each phase has local state.

## 注意事项 / Caveats / when it breaks

- **不支持 `ReduceLROnPlateau` / No `ReduceLROnPlateau` support**: 它的 `step` 需要 metric，和无参 `step()` 合约不一致。 / Its `step` needs a metric, unlike the no-argument `step()` contract.
- **milestone 排序很关键 / Milestone ordering matters**: `bisect_right` 假设切换点按时间递增。 / `bisect_right` assumes switch points are time-ordered.

## 延伸阅读 / Further reading

- [PyTorch `SequentialLR`](https://github.com/pytorch/pytorch/blob/da9c79b096d4b842c9fe4ac42298380a78cf69f3/torch/optim/lr_scheduler.py#L1093-L1228)
