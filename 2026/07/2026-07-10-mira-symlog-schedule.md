---
date: 2026-07-10
topic: diffusion
source: trending
repo: mira-wm/mira
file: src/mira/world_model/schedule.py
permalink: https://github.com/mira-wm/mira/blob/737e3d92cadc053c24e2e7733ac55ca4838e49ca/src/mira/world_model/schedule.py#L1-L87
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, diffusion, world-model, sampling-schedule]
---

# MIRA schedule：动作先 symlog，采样步先慢后快 / MIRA Schedule: Symlog Actions, Then Use a Slow-to-Fast Sampling Grid

> **一句话 / In one line**: MIRA 用 symlog 压缩鼠标动作尺度，并提供 linear/linear-quadratic 两种 flow-matching inference schedule。 / MIRA uses symlog to compress mouse-action magnitudes and provides linear and linear-quadratic flow-matching inference schedules.

## 为什么重要 / Why this matters

交互式世界模型既要处理玩家动作，也要按噪声时间积分生成下一帧。这个文件把两件事都压得很清楚：动作值用对称 log 防止大鼠标位移支配模型；采样时间表用简单函数控制 denoising 的步长分布。

An interactive world model must handle player actions and integrate over noise time to generate the next frame. This file makes both concerns explicit: symlog keeps large mouse deltas from dominating, and a small schedule function controls denoising step allocation.

## 代码 / The code

`mira-wm/mira` — [`src/mira/world_model/schedule.py`](https://github.com/mira-wm/mira/blob/737e3d92cadc053c24e2e7733ac55ca4838e49ca/src/mira/world_model/schedule.py#L1-L87)

```python
"""Sampling schedules and the symlog action normalisation used by the world model.

``symlog_normalize`` squashes raw mouse deltas into a bounded range before the action encoder
embeds them; ``build_inference_schedule`` produces the ``tau`` integration grid for the flow-matching
denoiser at inference time.
"""

from __future__ import annotations

import torch


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Signed logarithm ``sign(x) * log(1 + |x|)``."""
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def symlog_normalize(value: torch.Tensor, scale: float, max_value: float) -> torch.Tensor:
    """Normalize a tensor with a symmetric log transform."""
    max_value_t = torch.tensor(max_value, device=value.device)

    value = symlog(scale * value)
    norm_constant = symlog(scale * max_value_t)

    result = value / norm_constant

    return result


def linear_quadratic_schedule(
    n_steps: int,
    device: torch.device,
    threshold_noise: float = 0.1,
    n_linear_steps: int | None = None,
) -> torch.Tensor:
    """Linear-then-quadratic ``tau`` schedule (linear near 0, quadratic toward 1)."""
    if n_linear_steps is None:
        n_linear_steps = n_steps // 2
    if n_steps < 2:
        return torch.tensor([0.0, 1.0], device=device)
    n_quadratic_steps = n_steps - n_linear_steps

    linear_timesteps = torch.linspace(0, threshold_noise, n_linear_steps + 1, device=device)
    start_value = torch.sqrt(linear_timesteps[-1])
    quadratic_timesteps = torch.linspace(start_value, 1.0, n_quadratic_steps + 1, device=device) ** 2

    timesteps = torch.cat([linear_timesteps[:-1], quadratic_timesteps])
    return timesteps


def linear_schedule(n_steps: int, device: torch.device) -> torch.Tensor:
    """Simple uniformly-spaced sampling schedule from 0 to 1 (no threshold / knee)."""
    if n_steps < 1:
        return torch.tensor([0.0, 1.0], device=device)
    return torch.linspace(0, 1.0, n_steps + 1, device=device)


def build_inference_schedule(
    n_steps: int, device: torch.device, schedule_type: str = "linear_quadratic"
) -> torch.Tensor:
    """Dispatch to a sampling schedule by name."""
    if schedule_type == "linear":
        return linear_schedule(n_steps, device)
    elif schedule_type == "linear_quadratic":
        return linear_quadratic_schedule(n_steps, device)
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type!r}")
```

## 逐行讲解 / What's happening

1. **第 13-18 行 / Lines 13-18 (`symlog`)**:
   - 中文: 正负号保留，幅值走 `log(1 + abs(x))`，大动作被压缩但不会丢方向。
   - English: Sign is preserved, magnitude goes through `log(1 + abs(x))`, so large actions are compressed without losing direction.
2. **第 31-38 行 / Lines 31-38 (`symlog_normalize`)**:
   - 中文: 用 `max_value` 的 symlog 结果做分母，让典型最大动作映射到 1 附近。
   - English: The symlog of `max_value` is the denominator, mapping a typical maximum action near 1.
3. **第 41-59 行 / Lines 41-59 (`linear_quadratic_schedule`)**:
   - 中文: 前半段线性走到 `threshold_noise`，后半段在 sqrt 空间均匀再平方，得到靠近 1 的非线性网格。
   - English: The first half linearly reaches `threshold_noise`; the second half is uniform in sqrt space then squared, producing a nonlinear grid toward 1.
4. **第 74-87 行 / Lines 74-87 (`build_inference_schedule`)**:
   - 中文: 外部配置只传字符串，具体 schedule 函数在这里分派。
   - English: External config passes only a string, and this dispatcher selects the schedule function.

## 类比 / The analogy

symlog 像游戏手柄的灵敏度曲线：轻推保持细腻，猛推不会把画面甩飞。schedule 则像剪辑软件的关键帧，决定哪里多放几帧过渡。

Symlog is like a gamepad sensitivity curve: small pushes stay precise, hard pushes do not fling the camera away. The schedule is like timeline keyframes, deciding where to spend more transition steps.

## 自己跑一遍 / Try it yourself

```python
import math

def symlog(x):
    return (1 if x >= 0 else -1) * math.log(1 + abs(x))

def normalize(x, scale=0.5, max_value=100):
    return symlog(scale * x) / symlog(scale * max_value)

print([round(normalize(x), 3) for x in [-100, -10, 0, 10, 100]])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[-1.0, -0.456, 0.0, 0.456, 1.0]
```

中等动作没有被线性压扁到很小，大动作也被限制在稳定范围里。

Medium actions are not crushed to tiny values, while large actions stay in a stable range.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Dreamer-style agents** / **Dreamer-style agents**: 常用 symlog/symexp 处理 reward 或 value 的大动态范围。 / They often use symlog/symexp for large reward or value ranges.
- **Flow-matching samplers** / **Flow-matching samplers**: 也把 inference 看成在 `[0, 1]` 时间轴上积分。 / They also treat inference as integration along a `[0, 1]` time axis.

## 注意事项 / Caveats / when it breaks

- **`max_value` 是数据假设** / **`max_value` is a data assumption**: 设太小会饱和太多动作，设太大又压不住离群值。 / Too small saturates many actions; too large fails to tame outliers.
- **schedule 没有免费午餐** / **Schedules are not free wins**: 非线性步长可能改善某些指标，也可能让别的场景采样不足。 / Nonlinear steps may improve one metric while undersampling another scenario.

## 延伸阅读 / Further reading

- [MIRA `schedule.py`](https://github.com/mira-wm/mira/blob/737e3d92cadc053c24e2e7733ac55ca4838e49ca/src/mira/world_model/schedule.py#L1-L87)
