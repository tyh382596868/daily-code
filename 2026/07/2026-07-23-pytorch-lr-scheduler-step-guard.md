---
date: 2026-07-23
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/optim/lr_scheduler.py
permalink: https://github.com/pytorch/pytorch/blob/main/torch/optim/lr_scheduler.py#L91-L186
difficulty: intermediate
read_time: ~8 min
tags: [code-of-the-day, pytorch, optimizer, lr-scheduler, call-order]
---

# PyTorch LRScheduler：用 wrapper 抓住 step 调用顺序 / PyTorch LRScheduler: Catch Step Order with a Wrapper

> **一句话 / In one line**: `LRScheduler` 在初始化时包住 `optimizer.step`，用一个 `_opt_called` 标记提醒用户先更新参数、再更新学习率。 / `LRScheduler` wraps `optimizer.step` at initialization and uses an `_opt_called` flag to warn when learning rate stepping happens before optimizer stepping.

## 为什么重要 / Why this matters

学习率调度器的数学公式通常很简单，真正容易错的是调用顺序。PyTorch 把这个“隐形约定”写进基类：scheduler 创建时给 optimizer 打一个轻量 wrapper，第一次发现顺序反了就报警。

The scheduler formulas are often simple; the fragile part is call order. PyTorch encodes that hidden contract in the base class: when the scheduler is created, it wraps the optimizer and warns the first time it sees the wrong order.

## 代码 / The code

`pytorch/pytorch` — [`torch/optim/lr_scheduler.py`](https://github.com/pytorch/pytorch/blob/main/torch/optim/lr_scheduler.py#L91-L186)

```python
# Simplified teaching slice.
def patch_track_step_called(opt):
    if hasattr(opt.step, "_wrapped_by_lr_sched"):
        return

    old_step = opt.step

    def wrapper(*args, **kwargs):
        opt._opt_called = True
        return old_step(*args, **kwargs)

    wrapper._wrapped_by_lr_sched = True
    opt.step = wrapper

def scheduler_step(scheduler):
    if scheduler._step_count == 1 and not scheduler.optimizer._opt_called:
        warn("call optimizer.step() before lr_scheduler.step()")
    scheduler._step_count += 1
    scheduler.last_epoch += 1
```

## 逐行讲解 / What's happening

1. **初始化时打补丁 / Patch at construction**: 中文: 调度器不是等出错才检查，而是在创建时就包住 optimizer。 English: the scheduler wraps the optimizer at construction time, not after a failure.
2. **标记真实 optimizer step / Mark real optimizer steps**: 中文: wrapper 只做一件事：设置 `_opt_called=True`。 English: the wrapper's main job is to set `_opt_called=True`.
3. **只在早期报警 / Warn early**: 中文: 第一次 scheduler step 最能发现“顺序写反”的训练循环。 English: the first scheduler step is the best moment to detect an inverted training loop.
4. **防止重复包裹 / Avoid double wrapping**: 中文: `_wrapped_by_lr_sched` 避免多个 scheduler 把同一个方法包成很多层。 English: `_wrapped_by_lr_sched` prevents layers of duplicate wrappers.

## 类比 / The analogy

像门禁先给门把手贴一个感应片。你每次开门都会留下记录；如果系统发现你先打卡“离开”再打卡“进入”，它知道流程写反了。

It is like adding a sensor to a door handle. Every real opening is recorded; if the system sees "exit" before "entry", it knows the procedure is reversed.

## 自己跑一遍 / Try it yourself

```python
class Optim:
    def step(self):
        print("optimizer step")

opt = Optim()
old = opt.step
def wrapped():
    opt.called = True
    old()
opt.step = wrapped

print(getattr(opt, "called", False))
opt.step()
print(getattr(opt, "called", False))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
False
optimizer step
True
```

## 注意事项 / Caveats / when it breaks

- **创建后别重写 optimizer.step / Do not replace optimizer.step later**: 重新赋值会绕过 wrapper，PyTorch 会提示这个风险。 / replacing it later bypasses the wrapper, and PyTorch warns about that.
- **恢复 checkpoint 要注意顺序 / Be careful when restoring checkpoints**: scheduler 初始化会写入 `initial_lr`，恢复时 optimizer 和 scheduler 的加载顺序很重要。 / scheduler initialization touches `initial_lr`, so restore order matters.
- **报警不是调度公式 / The warning is not the schedule**: 它只保护训练循环契约，不改变具体 LR 计算。 / it protects the loop contract, not the specific LR formula.

## 延伸阅读 / Further reading

- [PyTorch lr_scheduler.py](https://github.com/pytorch/pytorch/blob/main/torch/optim/lr_scheduler.py#L91-L186)

