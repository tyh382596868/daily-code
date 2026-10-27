---
date: 2026-10-27
topic: vla
source: vla
repo: Physical-Intelligence/openpi
file: src/openpi/transforms.py
permalink: https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/transforms.py#L204-L223
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, action-normalization]
build_role: action-head-continuous
---

# openpi 掩码动作增量：只相对化该相对化的维度 / openpi Masked Delta Actions: Relativize Only the Right Dimensions

> **一句话 / In one line**: `DeltaActions` 用布尔 mask 将部分绝对动作改成相对 state 的偏移。 / `DeltaActions` uses a Boolean mask to turn selected absolute actions into state-relative offsets.

## 为什么重要 / Why this matters

机器人动作里常混有位置、夹爪开度和模式位；把所有维度一概减 state 会破坏不应相对化的字段。

Robot actions often mix positions, gripper widths, and mode bits. Blindly subtracting state corrupts fields that are not relative quantities.

## 代码 / The code

`Physical-Intelligence/openpi` — [`src/openpi/transforms.py`](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/transforms.py#L204-L223)

```python
state, actions = data["state"], data["actions"]
mask = np.asarray(self.mask)
dims = mask.shape[-1]
actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
data["actions"] = actions
```

## 逐行讲解 / What's happening

`np.where` 对被选维度取 state、其余维度取零；新增的时间轴让一个 state 对齐整段 action chunk。

`np.where` supplies state for selected dimensions and zero elsewhere; the inserted time axis broadcasts one state across an action chunk.

## 在 nanoVLA 中的位置 / Where this lives in your nanoVLA

这是连续动作 head 前的数据契约层：输入为当前 state 与 `[chunk, action_dim]` 目标，输出为训练目标增量；推理后用反变换恢复绝对动作。

This is the contract layer before a continuous action head: it maps current state plus chunk targets to deltas, then inference applies the inverse transform.

## 自己跑一遍 / Try it yourself

```python
import numpy as np
state = np.array([10., 5., 1.]); action = np.array([[12., 8., 0.]])
mask = np.array([True, True, False])
print(action - np.where(mask, state, 0))
```

运行 / Run with `pip install numpy && python try.py`。输出 / Expected output: `[[2. 3. 0.]]`。

## 注意事项 / Caveats / when it breaks

- **单位 / Units**：state 与 action 必须同单位、同坐标系。 / State and action must share units and coordinate frame.
