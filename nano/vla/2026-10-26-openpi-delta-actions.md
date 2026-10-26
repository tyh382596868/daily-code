---
date: 2026-10-26
topic: vla
source: vla
repo: Physical-Intelligence/openpi
file: src/openpi/transforms.py
permalink: https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/transforms.py#L204-L223
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, action-tokenizer, action-representation]
build_role: action-tokenizer advanced variant
---

# openpi 动作增量：只把指定自由度改写成相对量 / openpi Delta Actions: Make Only Selected Degrees of Freedom Relative

> **一句话 / In one line**: mask 决定哪些动作维度减去当前 state。 / A mask decides which action dimensions subtract the current state.

## 为什么重要 / Why this matters

同一策略常要面对“位置用相对量、夹爪用绝对量”的混合动作空间。把这个选择做成 mask，比给每种机器人写一条训练路径更可靠。

A policy may need relative arm motion but absolute gripper commands. A mask encodes that choice without creating one training path per robot.

## 代码 / The code

```python
state, actions = data["state"], data["actions"]
mask = np.asarray(self.mask)
dims = mask.shape[-1]
actions[..., :dims] -= np.expand_dims(
    np.where(mask, state[..., :dims], 0), axis=-2)
data["actions"] = actions
```

来源 / Source: [`transforms.py`](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/transforms.py#L204-L223)。

## 逐行讲解 / What's happening

`np.where` 对不应相对化的维度填零；插入时间轴后广播到整个 action chunk。反变换 `AbsoluteActions` 做同一件事的加法版。

`np.where` zeros dimensions that stay absolute; the added time axis broadcasts the state over an action chunk. `AbsoluteActions` is the additive inverse.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

这是已覆盖的 `action-tokenizer` 的连续高级变体：它位于 dataset adapter 与 action head 之间，输入 state 和 `[horizon, dim]` actions，输出模型要拟合的规范化目标。省掉它会把绝对坐标变化误当作任务变化；生产实现还需逐 embodiment 校验 mask、单位和关节限位。

This is an advanced continuous variant of the covered `action-tokenizer` slot. It sits between the dataset adapter and action head. Production code must validate masks, units, and limits per embodiment.

## 类比 / The analogy

像导航里“向东两米”与“经纬度坐标”混用：每个字段必须明确它相对谁。

It is like mixing “two meters east” with GPS coordinates: every field must state its reference frame.

## 自己跑一遍 / Try it yourself

```python
import numpy as np
state = np.array([10., 20., 1.])
actions = np.array([[11., 18., 0.], [12., 21., 1.]])
mask = np.array([True, True, False])
print(actions - np.where(mask, state, 0))
```

运行 / Run with `pip install numpy && python try.py`。预期 / Expected: 前两维成为相对位移，第三维保持原值。

## 注意事项 / Caveats / when it breaks

- **坐标系 / Frames**：state 与 action 必须同一坐标系和单位。 / State and actions must share frame and units.
