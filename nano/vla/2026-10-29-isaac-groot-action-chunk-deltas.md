---
date: 2026-10-29
topic: vla
source: vla
repo: NVIDIA/Isaac-GR00T
file: gr00t/data/state_action/action_chunking.py
permalink: https://github.com/NVIDIA/Isaac-GR00T/blob/c56f54128d92f743889b7e4cecfc1ceafd88739e/gr00t/data/state_action/action_chunking.py#L29-L142
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, action-chunking]
build_role: action-chunking advanced variant
---

# Isaac GR00T 动作 chunk 增量 / Isaac GR00T Action-Chunk Deltas

> **一句话 / In one line**: 将 pose 序列相邻相减，保留一个 chunk 的时间轴。 / Subtract adjacent poses while retaining a chunk’s time axis.

## 为什么重要 / Why this matters

动作 chunk 让 policy 一次预测多步，但控制器和数据集必须同意“绝对目标、相对目标、还是增量”到底是什么意思。

Action chunks let a policy predict several steps at once, but the controller and dataset must agree whether a target is absolute, relative, or a delta.

## 代码 / The code

`NVIDIA/Isaac-GR00T` — [`action_chunking.py`](https://github.com/NVIDIA/Isaac-GR00T/blob/c56f54128d92f743889b7e4cecfc1ceafd88739e/gr00t/data/state_action/action_chunking.py#L111-L142)

```python
prev_pose = reference_frame if reference_frame is not None else self._poses[0]
for current_pose in self._poses:
    delta = current_pose - prev_pose
    delta_poses.append(delta)
    prev_pose = current_pose
return self.__class__(delta_poses, times=self.times.tolist())
```

## 逐行讲解 / What's happening

1. 中文：没给 reference 时，首项减自己，得到 identity/zero。 English: Without a reference, the first pose subtracts itself and yields identity/zero.
2. 中文：循环随后把上一帧作为锚点，形成局部控制增量。 English: The loop then uses the previous frame as the anchor, producing local control increments.
3. 中文：时间戳原样带走，动作值和调度语义不混在一起。 English: Timestamps are carried through unchanged, keeping action values separate from scheduling semantics.

## 类比 / The analogy

绝对 pose 是城市地址；delta 是“从这里向东走两步”的路口指令。

An absolute pose is a street address; a delta is the instruction “walk two blocks east from here.”

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

课程项是 `action-chunking`（基础课程已覆盖；本篇是 Isaac GR00T 的高级变体）。它位于 action head 之后、执行器之前：输入为 `[B, H, D]` 的预测姿态和当前 reference，输出为可逐拍积分的 `[B, H, D]` 增量。缺掉它时，模型和机器人会在坐标系语义上失配；生产版本还要处理旋转群、限幅、时钟抖动和动作中断。

This is the `action-chunking` curriculum item (a post-coverage Isaac GR00T variant). It sits after the action head and before the actuator: predicted `[B, H, D]` poses plus a reference become deltas that can be integrated tick by tick. Omitting it creates coordinate-system mismatch; production code also needs rotation-group math, limits, clock jitter, and interruption handling.

## 自己跑一遍 / Try it yourself

```python
poses = [10, 13, 12, 18]
prev = poses[0]
deltas = []
for pose in poses:
    deltas.append(pose - prev)
    prev = pose
print(deltas)
```

运行 / Run with: `python try.py`

预期输出 / Expected output: `[0, 3, -1, 6]`

中文：首项为零是“没有前一步”的明确约定。 English: The leading zero is an explicit convention for “no previous step.”

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- 中文：LeRobot 的 action queue 将 horizon 拆成控制 tick。 / English: LeRobot action queues split a horizon into control ticks.
- 中文：openpi 的 delta transform 只转换指定 action 维度。 / English: openpi delta transforms convert selected action dimensions.

## 注意事项 / Caveats / when it breaks

- 中文：欧拉角不能像线性标量一样直接相减。 / English: Euler angles cannot safely be subtracted like linear scalars.

## 延伸阅读 / Further reading

- [Isaac GR00T action chunking](https://github.com/NVIDIA/Isaac-GR00T/blob/c56f54128d92f743889b7e4cecfc1ceafd88739e/gr00t/data/state_action/action_chunking.py#L111-L142)
