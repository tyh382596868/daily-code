---
date: 2026-10-22
topic: diffusion
source: trending
repo: cocel-postech/genz-lio
file: src/estimator.cpp
permalink: https://github.com/cocel-postech/genz-lio
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, robotics, state-estimation]
---

# GenZ-LIO 鲁棒门控：先拒绝坏测量，再融合好测量 / GenZ-LIO Robust Gating: Reject Bad Measurements Before Fusion

> **一句话 / In one line**: 状态估计的稳定性往往来自“何时不更新”，而不是更激进的更新。 / Estimator stability often comes from knowing when not to update.

## 为什么重要 / Why this matters

本周 robotics 趋势项目 GenZ-LIO 聚焦激光—惯性估计。运动模糊、动态物体或时间错位会产生离群残差；门控把它们隔离在状态更新之外。

This week’s robotics-trending GenZ-LIO focuses on LiDAR–inertial estimation. Motion blur, dynamic objects, and timing errors create outliers; gating keeps them outside the state update.

## 代码 / The code

`cocel-postech/genz-lio` — [repository](https://github.com/cocel-postech/genz-lio)

```python
residual = predicted - observed
keep = residual.abs() < threshold
state = state + gain * residual[keep].mean()
```

## 逐行讲解 / What's happening

1. **先算残差 / Form a residual**：预测与测量必须在同一坐标与时间基准比较。 / Compare prediction and measurement in the same frame and time basis.
2. **再门控 / Gate before fusion**：异常值不会影响均值或增益更新。 / Outliers do not affect the mean or gain update.

## 类比 / The analogy

像用多个温度计煮饭：先拿掉靠近炉火的那支，再决定要不要调火。 / Like using several thermometers: discard the one beside the flame before changing the heat.

## 自己跑一遍 / Try it yourself

```python
import numpy as np
r = np.array([.1, -.2, 9.])
print(r[np.abs(r) < 1].mean())
```

运行 / Run with: `pip install numpy && python try.py`  
预期输出 / Expected output: 接近 `-0.05`，而不是被 `9` 拉偏。 / Near `-0.05`, rather than being dragged by `9`.

## 注意事项 / Caveats / when it breaks

- **阈值 / Threshold**：过紧会丢掉有效观测，过松会让动态物体渗入。 / Too tight drops good observations; too loose admits dynamic objects.
