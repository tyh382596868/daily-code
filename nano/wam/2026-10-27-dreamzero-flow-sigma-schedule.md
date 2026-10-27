---
date: 2026-10-27
topic: wam
source: wam
repo: dreamzero0/dreamzero
file: groot/vla/model/dreamzero/modules/flow_match_scheduler.py
permalink: https://github.com/dreamzero0/dreamzero/blob/ab790c198fbce33503358efbbd4187ce9a89adf3/groot/vla/model/dreamzero/modules/flow_match_scheduler.py#L5-L49
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, wam, flow-matching]
build_role: noise-scheduler
---

# DreamZero 流匹配时间表：先排 sigma，再走一步 / DreamZero Flow Schedule: Place Sigmas, Then Take a Step

> **一句话 / In one line**: scheduler 将线性 sigma 经 shift 重映射，并用相邻 sigma 的差推进样本。 / The scheduler remaps a linear sigma schedule with a shift and advances samples by adjacent-sigma differences.

## 为什么重要 / Why this matters

flow model 预测的是路径上的速度，真正的更新幅度来自时间表而不是网络本身。

A flow model predicts velocity on a path; the schedule, not the network, determines how far each update travels.

## 代码 / The code

`dreamzero0/dreamzero` — [`flow_match_scheduler.py`](https://github.com/dreamzero0/dreamzero/blob/ab790c198fbce33503358efbbd4187ce9a89adf3/groot/vla/model/dreamzero/modules/flow_match_scheduler.py#L18-L49)

```python
self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
self.timesteps = self.sigmas * self.num_train_timesteps
sigma = self.sigmas[timestep_id]
sigma_ = self.sigmas[timestep_id + 1]
prev_sample = sample + model_output * (sigma_ - sigma)
```

## 逐行讲解 / What's happening

shift 把采样密度重新分配给路径的一端；Euler 更新把 velocity 乘相邻 sigma 间隔。

The shift redistributes sampling density along the path; the Euler update multiplies velocity by the next sigma interval.

## 在 nanoWAM 中的位置 / Where this lives in your nanoWAM

这是采样循环的节拍器：DiT 接收当前 latent 与 timestep，scheduler 决定下一 latent。省掉它，模型输出没有可执行的积分尺度。

This is the sampler’s metronome: the DiT consumes a latent and timestep, while the scheduler selects the next latent. Without it, the model output has no integration scale.

## 自己跑一遍 / Try it yourself

```python
import torch
sigmas = torch.linspace(1., 0., 4); x = torch.tensor(1.); v = torch.tensor(2.)
for a, b in zip(sigmas, sigmas[1:]): x = x + v * (b - a)
print(x.item())
```

运行 / Run with `pip install torch && python try.py`。输出 / Expected output: `-1.0`。

## 注意事项 / Caveats / when it breaks

- **方向 / Direction**：反向或 flipped sigma 会改变符号；训练和推理约定必须一致。 / Reversed sigmas change the sign, so training and inference conventions must match.
