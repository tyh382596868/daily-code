---
date: 2026-08-21
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/optim/lr_scheduler.py
permalink: https://github.com/pytorch/pytorch/blob/02dbce8ad5294043ecfea9dc5c4299c30b5fb7b0/torch/optim/lr_scheduler.py#L2431-L2606
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, lr-scheduler, one-cycle, momentum]
---

# PyTorch OneCycleLR：学习率和动量反着走 / PyTorch OneCycleLR: Learning Rate and Momentum Move Opposite Ways

> **一句话 / In one line**: `OneCycleLR` 先把训练步数拆成 2 段或 3 段 schedule phase，再按当前 step 在 phase 内插值，同时可反向更新 momentum。 / `OneCycleLR` splits training into two or three schedule phases, interpolates inside the current phase, and can update momentum in the opposite direction.

## 为什么重要 / Why this matters

One-cycle 不是每个 epoch 调一次的粗粒度策略，而是每个 batch 都 step 的细粒度 schedule。它把 learning rate 从小推到大，再降到很小；如果启用 momentum，则在 LR 高时降低 momentum，LR 低时提高 momentum，让探索和稳定性互相配合。

One-cycle is a per-batch schedule, not a coarse per-epoch knob. It raises LR from small to large, then drops it very low. When momentum cycling is enabled, momentum moves oppositely: lower at high LR, higher at low LR.

## 代码 / The code

`pytorch/pytorch` — [`torch/optim/lr_scheduler.py`](https://github.com/pytorch/pytorch/blob/02dbce8ad5294043ecfea9dc5c4299c30b5fb7b0/torch/optim/lr_scheduler.py#L2431-L2606)

```python
self._schedule_phases: list[_SchedulePhase]
if three_phase:
    self._schedule_phases = [
        {"end_step": float(pct_start * self.total_steps) - 1,
         "start_lr": "initial_lr", "end_lr": "max_lr",
         "start_momentum": "max_momentum", "end_momentum": "base_momentum"},
        {"end_step": float(2 * pct_start * self.total_steps) - 2,
         "start_lr": "max_lr", "end_lr": "initial_lr",
         "start_momentum": "base_momentum", "end_momentum": "max_momentum"},
        {"end_step": self.total_steps - 1,
         "start_lr": "initial_lr", "end_lr": "min_lr",
         "start_momentum": "max_momentum", "end_momentum": "max_momentum"},
    ]
else:
    self._schedule_phases = [
        {"end_step": float(pct_start * self.total_steps) - 1,
         "start_lr": "initial_lr", "end_lr": "max_lr",
         "start_momentum": "max_momentum", "end_momentum": "base_momentum"},
        {"end_step": self.total_steps - 1,
         "start_lr": "max_lr", "end_lr": "min_lr",
         "start_momentum": "base_momentum", "end_momentum": "max_momentum"},
    ]

...

for group in self.optimizer.param_groups:
    start_step = 0.0
    for i, phase in enumerate(self._schedule_phases):
        end_step = phase["end_step"]
        if step_num <= end_step or i == len(self._schedule_phases) - 1:
            pct = (step_num - start_step) / (end_step - start_step)
            computed_lr = self._anneal_func(group[phase["start_lr"]], group[phase["end_lr"]], pct)
            if self.cycle_momentum:
                computed_momentum = self._anneal_func(
                    group[phase["start_momentum"]], group[phase["end_momentum"]], pct
                )
            break
        start_step = phase["end_step"]

    lrs.append(computed_lr)
    if self.cycle_momentum:
        if self.use_beta1:
            group["betas"] = (computed_momentum, *group["betas"][1:])
        else:
            group["momentum"] = computed_momentum
```

## 逐行讲解 / What's happening

1. **第 2431-2472 行 / Lines 2431-2472 (phase table)**:
   - 中文: schedule 先变成 phase 表，每段只记录结束 step、LR 起点/终点、momentum 起点/终点。
   - English: The schedule becomes a phase table: each phase stores its end step plus LR and momentum endpoints.
2. **第 2488-2495 行 / Lines 2488-2495 (LR anchors)**:
   - 中文: `initial_lr = max_lr / div_factor`，`min_lr = initial_lr / final_div_factor`，所以最低 LR 可能比初始值低很多。
   - English: `initial_lr = max_lr / div_factor`, and `min_lr = initial_lr / final_div_factor`, so the floor can be far below the start.
3. **第 2581-2597 行 / Lines 2581-2597 (choose phase and interpolate)**:
   - 中文: 当前 step 先找到所属 phase，再用 cosine 或 linear anneal 算出这一 batch 的 LR。
   - English: The current step selects a phase, then cosine or linear annealing computes the LR for this batch.
4. **第 2600-2604 行 / Lines 2600-2604 (momentum side effect)**:
   - 中文: 如果启用 `cycle_momentum`，`get_lr()` 还会直接写 optimizer 的 momentum 或 Adam beta1。
   - English: With `cycle_momentum`, `get_lr()` also writes optimizer momentum or Adam beta1.

## 类比 / The analogy

像开赛车过弯。直道先给油门提高速度，进弯时收油并增加抓地感；one-cycle 把 LR 当油门，把 momentum 当稳定器，两者反向配合。

It is like driving through a turn. You accelerate on the straight, then ease off and regain stability in the turn; one-cycle treats LR as throttle and momentum as the stabilizer.

## 自己跑一遍 / Try it yourself

```python
total_steps = 10
pct_start = 0.3
initial_lr, max_lr, min_lr = 0.01, 0.1, 0.001
max_momentum, base_momentum = 0.95, 0.85

phases = [
    (pct_start * total_steps - 1, initial_lr, max_lr, max_momentum, base_momentum),
    (total_steps - 1, max_lr, min_lr, base_momentum, max_momentum),
]

start = 0.0
for step in [0, 2, 5, 9]:
    start = 0.0
    for end, lr0, lr1, m0, m1 in phases:
        if step <= end:
            pct = (step - start) / (end - start)
            lr = lr0 + (lr1 - lr0) * pct
            mom = m0 + (m1 - m0) * pct
            print(step, round(lr, 4), round(mom, 4))
            break
        start = end
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
0 0.01 0.95
2 0.1 0.85
5 0.0576 0.8929
9 0.001 0.95
```

中文: LR 到峰值时 momentum 到低点，后半段 LR 下降时 momentum 又升回来。

English: Momentum reaches its low point when LR peaks, then rises as LR falls.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Cyclical LR** / **Cyclical LR**: 也按 step 做周期插值，但 one-cycle 通常只跑一个大周期。
- **Warmup + decay** / **Warmup + decay**: 先升后降的核心形状相同，只是 one-cycle 还绑定 momentum。

## 注意事项 / Caveats / when it breaks

- **必须按 batch step** / **Step it per batch**: 如果每个 epoch 才 `scheduler.step()`，`total_steps` 语义会错。
- **resume 要保存 scheduler state** / **Save scheduler state when resuming**: 只恢复 optimizer 不恢复 scheduler，会让 phase 位置漂移。

## 延伸阅读 / Further reading

- [PyTorch `OneCycleLR`](https://github.com/pytorch/pytorch/blob/02dbce8ad5294043ecfea9dc5c4299c30b5fb7b0/torch/optim/lr_scheduler.py#L2431-L2606)
- [PyTorch LR scheduler docs](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
