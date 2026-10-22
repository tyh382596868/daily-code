---
date: 2026-10-22
topic: wam
source: wam
repo: yuantianyuan01/FastWAM
file: src/fastwam/trainer.py
permalink: https://github.com/yuantianyuan01/FastWAM/blob/main/src/fastwam/trainer.py#L193-L255
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, training-loop]
build_role: training-loop advanced variant
---

# FastWAM 分布式 step：只在真正更新时推进时间表 / FastWAM Distributed Step: Advance the Schedule Only on Real Updates

> **一句话 / In one line**: 梯度累积下，scheduler、global step 与日志必须跟 optimizer step 同步。 / With gradient accumulation, scheduler, global step, and logging must follow optimizer updates.

## 为什么重要 / Why this matters

世界动作模型通常用大 batch 与累积梯度。如果 micro-batch 也推进 scheduler，学习率会快进，断点续训的 step 语义也会错位。

World-action models often combine large batches with accumulation. Advancing the scheduler on every micro-batch fast-forwards the learning rate and corrupts resume semantics.

## 代码 / The code

`yuantianyuan01/FastWAM` — [`trainer.py`](https://github.com/yuantianyuan01/FastWAM/blob/main/src/fastwam/trainer.py#L193-L255)

```python
with self.accelerator.accumulate(self.model):
    with self.accelerator.autocast():
        loss, loss_dict = train_model.training_loss(sample)
    self.accelerator.backward(loss)
    if self.accelerator.sync_gradients:
        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        if not self.accelerator.optimizer_step_was_skipped:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1
```

## 逐行讲解 / What's happening

1. **累计上下文 / Accumulation context**：Accelerate 决定何时同步梯度。 / Accelerate decides when gradients synchronize.
2. **同步门 / Synchronization gate**：optimizer、scheduler 与计数器只在该门内前进。 / Optimizer, scheduler, and counter advance only behind this gate.
3. **跳步保护 / Skipped-step guard**：溢出导致 optimizer 没更新时，学习率也不该前进。 / If overflow skips an optimizer update, LR must not advance either.

## 类比 / The analogy

像按月还贷款：每天记录消费，但只在结算日真正扣款并翻到下一期。 / Like loan payments: record daily activity, but advance the account only on settlement day.

## 在 nanoWAM 中的位置 / Where this lives in your nanoWAM

这是 `training-loop` 的分布式版本，接在 loss 之后、checkpoint 之前。省掉它不会让单卡 demo 立刻失败，却会在多卡、混精度和恢复训练时破坏时间轴。

This is the distributed `training-loop` component between loss computation and checkpointing. A single-GPU demo may survive without it; mixed precision and resume training will not.

## 自己跑一遍 / Try it yourself

```python
accum = 2; step = 0
for micro in range(4):
    if (micro + 1) % accum == 0:
        step += 1; print("update", step)
```

运行 / Run with: `python try.py`  
预期输出 / Expected output: 两次 update，而不是四次。 / Two updates, not four.
