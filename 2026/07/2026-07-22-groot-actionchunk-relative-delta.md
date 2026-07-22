---
date: 2026-07-22
topic: robotics
source: tracked
repo: NVIDIA/Isaac-GR00T
file: gr00t/data/state_action/action_chunking.py
permalink: https://github.com/NVIDIA/Isaac-GR00T/blob/9c7e746b2cd37a810070a98ef41d290a07e806c2/gr00t/data/state_action/action_chunking.py#L25-L128
difficulty: beginner
read_time: ~8 min
tags: [code-of-the-day, robotics, action-chunking, relative-actions]
---

# Isaac-GR00T ActionChunk：同一段动作可以变成相对量或逐步增量 / Isaac-GR00T ActionChunk: One Action Chunk Can Become Relative Poses or Step Deltas

> **一句话 / In one line**: `ActionChunk` 把未来多步 pose 包成一个对象，再用同一个 `pose - reference` 原语生成相对动作或逐步 delta。 / `ActionChunk` wraps future poses in one object, then uses the same `pose - reference` primitive to produce either relative actions or step-by-step deltas.

## 为什么重要 / Why this matters

机器人策略通常不应该直接背绝对坐标。相对当前末端、相对第一帧、或相对上一帧的动作，更容易跨初始位置、跨场景泛化。GR00T 的这段代码把这个选择做成一个明确接口，而不是散落在数据集和模型里。

Robot policies usually should not memorize absolute coordinates. Actions relative to the current end effector, the first frame, or the previous frame generalize better across starts and scenes. This code makes that choice an explicit interface instead of hiding it in dataset glue or model code.

## 代码 / The code

`NVIDIA/Isaac-GR00T` — [`gr00t/data/state_action/action_chunking.py`](https://github.com/NVIDIA/Isaac-GR00T/blob/9c7e746b2cd37a810070a98ef41d290a07e806c2/gr00t/data/state_action/action_chunking.py#L25-L128)

```python
# Simplified teaching slice, preserving the control flow.
class ActionChunk:
    def __init__(self, poses, times=None):
        if not poses:
            raise ValueError("ActionChunk must contain at least one pose")
        self._poses = list(poses)
        self._times = list(range(len(poses))) if times is None else list(times)
        if len(self._times) != len(self._poses):
            raise ValueError("Number of times must match number of poses")

    @property
    def times(self):
        return self._times.copy()

    def relative_chunking(self, reference_frame=None):
        ref_pose = reference_frame if reference_frame is not None else self._poses[0]
        relative_poses = [pose - ref_pose for pose in self._poses]
        return self.__class__(relative_poses, times=self.times)

    def delta_chunking(self, reference_frame=None):
        prev_pose = reference_frame if reference_frame is not None else self._poses[0]
        delta_poses = []
        for current_pose in self._poses:
            delta_poses.append(current_pose - prev_pose)
            prev_pose = current_pose
        return self.__class__(delta_poses, times=self.times)
```

## 逐行讲解 / What's happening

1. **构造函数先保护输入 / The constructor protects the contract**: 中文: 空 chunk 没有参考帧，时间长度不匹配也无法插值，所以两种情况直接拒绝。 English: an empty chunk has no reference frame, and mismatched timestamps cannot be interpolated safely, so both are rejected.
2. **`times` 返回 copy / `times` returns a copy**: 中文: 调用者不能在外面偷偷改掉对象内部时间轴。 English: callers cannot mutate the object's internal timeline from the outside.
3. **`relative_chunking` 固定一个参考 / `relative_chunking` fixes one reference**: 中文: 所有未来 pose 都减同一个 `ref_pose`，得到“从当前姿态出发要到哪里”。 English: every future pose subtracts the same `ref_pose`, producing "where to go from this reference."
4. **`delta_chunking` 每步换参考 / `delta_chunking` advances the reference**: 中文: 每个输出只描述从上一帧到当前帧的局部运动。 English: each output describes only the local motion from the previous frame to the current one.
5. **`self.__class__` 保留子类型 / `self.__class__` preserves the subtype**: 中文: joint pose 和末端 pose 可以共用逻辑，但返回自己的 chunk 类型。 English: joint-pose and end-effector chunks can share logic while returning their own subtype.

## 类比 / The analogy

像导航指令。相对 chunk 是“从酒店出发，第一站在东边 100 米，第二站在东边 200 米”；delta chunk 是“先走 100 米，再走 100 米”。两者都能到同一个地方，但适合不同控制器。

It is like navigation instructions. A relative chunk says "from the hotel, stop one is 100 meters east, stop two is 200 meters east." A delta chunk says "walk 100 meters, then walk another 100 meters." Both reach the same place, but they fit different controllers.

## 自己跑一遍 / Try it yourself

```python
class Pose:
    def __init__(self, x): self.x = x
    def __sub__(self, other): return Pose(self.x - other.x)
    def __repr__(self): return f"Pose({self.x})"

class Chunk:
    def __init__(self, poses): self.poses = poses
    def relative(self): return [p - self.poses[0] for p in self.poses]
    def delta(self):
        prev, out = self.poses[0], []
        for p in self.poses:
            out.append(p - prev); prev = p
        return out

c = Chunk([Pose(10), Pose(13), Pose(18)])
print(c.relative())
print(c.delta())
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[Pose(0), Pose(3), Pose(8)]
[Pose(0), Pose(3), Pose(5)]
```

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LeRobot RTC** / **LeRobot RTC**: 中文: 重叠 action chunk 会把旧动作变成约束，再混入新预测。 / English: overlapping chunks turn old actions into constraints before blending new predictions.
- **openpi DeltaActions** / **openpi DeltaActions**: 中文: 数据 transform 阶段就把绝对动作改写成相对控制。 / English: the data transform rewrites absolute actions into relative controls before the model sees them.
- **Diffusion Policy** / **Diffusion Policy**: 中文: 预测的是一个未来 horizon，执行时逐步弹出。 / English: the model predicts a future horizon and execution pops one step at a time.

## 注意事项 / Caveats / when it breaks

- **减法语义必须一致 / Subtraction semantics must be consistent**: pose 的 `__sub__` 要明确是在关节空间、SE(3) 空间还是末端空间里求差。 / The pose `__sub__` must define whether the difference lives in joint space, SE(3), or end-effector space.
- **delta 会累积误差 / Deltas accumulate error**: 执行器如果每步有偏差，后续 delta 会叠在偏差上。 / If execution drifts at each step, later deltas build on that drift.
- **时间轴不能忽略 / The timeline matters**: 非均匀采样时，delta 大小和速度含义不同。 / With non-uniform sampling, a delta's magnitude and velocity meaning are different.

## 延伸阅读 / Further reading

- [Isaac-GR00T action_chunking.py](https://github.com/NVIDIA/Isaac-GR00T/blob/9c7e746b2cd37a810070a98ef41d290a07e806c2/gr00t/data/state_action/action_chunking.py#L25-L128)

