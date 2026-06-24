---
date: 2026-06-24
topic: vla
source: vla
repo: NVIDIA/Isaac-GR00T
file: gr00t/data/state_action/action_chunking.py
permalink: https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/data/state_action/action_chunking.py
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, action-chunking, robotics, embodiment-transfer, slerp]
build_role: action-chunking (cross-repo variant — Isaac-GR00T implementation)
---

# Isaac-GR00T ActionChunk：relative vs delta 归一化 + SLERP 旋转插值 / Isaac-GR00T ActionChunk: Relative vs Delta Normalization + SLERP Rotation Interpolation

> **一句话 / In one line**: `relative_chunking()` 以首帧为参考把所有 pose 归零，`delta_chunking()` 逐帧差分——两种归一化策略让 action chunk 跨机器人泛化；末端执行器的旋转用 SLERP 而不是线性差分。/ `relative_chunking()` zeros all poses against the first frame; `delta_chunking()` computes frame-to-frame deltas — two normalization strategies that make action chunks transfer across robot embodiments; end-effector rotations use SLERP instead of linear interpolation.

## 为什么重要 / Why this matters

VLA 预测的动作块（action chunk）通常是绝对姿态序列。但"绝对"带来两个问题：①不同机器人的 home position 不同，模型学到的是特定机器人的绝对坐标，迁移到别的机器人就失效；②末端执行器的旋转不能直接做差——旋转群是非欧氏的，必须用球面线性插值（SLERP）才能得到有意义的中间旋转。Isaac-GR00T 把这两件事在 `ActionChunk` 基类里一次性解决，同时保持代码可组合。

Action chunks predicted by a VLA are typically absolute pose sequences. But "absolute" brings two problems: ① different robots have different home positions, so the model learns robot-specific absolute coordinates that don't transfer; ② rotation deltas can't be computed by plain subtraction — rotations live on a non-Euclidean manifold and need Spherical Linear Interpolation (SLERP) for meaningful intermediate values. Isaac-GR00T solves both in the `ActionChunk` base class while keeping the code composable.

## 代码 / The code

`NVIDIA/Isaac-GR00T` — [`gr00t/data/state_action/action_chunking.py`](https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/data/state_action/action_chunking.py)

```python
class ActionChunk:
    """
    A sequence of robot poses (action horizon) with two normalization strategies:
    relative_chunking  — all poses expressed relative to a reference frame
    delta_chunking     — each pose expressed as a delta from the previous pose
    """

    def __init__(self, poses: list, times=None):
        self._poses = poses
        self.times = (
            np.array(times, dtype=np.float64) if times is not None
            else np.arange(len(poses), dtype=np.float64)
        )

    def relative_chunking(self, reference_frame=None):
        """
        Express every pose as a delta from reference_frame.
        If reference_frame is None, uses poses[0] as the reference.
        This removes the absolute position of the robot while keeping
        the shape of the trajectory intact.
        """
        ref_pose = reference_frame if reference_frame is not None else self._poses[0]
        relative_poses = [pose - ref_pose for pose in self._poses]
        return self.__class__(relative_poses, times=self.times)

    def delta_chunking(self, reference_frame=None):
        """
        Express each pose as a delta from the previous pose.
        Useful when you want the model to learn incremental motions
        rather than global waypoints.
        """
        delta_poses = []
        prev_pose = reference_frame if reference_frame is not None else self._poses[0]
        for current_pose in self._poses:
            delta = current_pose - prev_pose
            delta_poses.append(delta)
            prev_pose = current_pose
        return self.__class__(delta_poses, times=self.times.tolist())

    def __len__(self): return len(self._poses)
    def __iter__(self): return iter(self._poses)


class EndEffectorActionChunk(ActionChunk):
    """
    Specialised ActionChunk for 6-DOF end-effector poses:
    (position xyz, rotation as quaternion or matrix).
    Overrides delta_chunking to use SLERP for rotation interpolation.
    """

    def interpolate(self, target_times):
        """
        Interpolate this chunk to a new set of time steps.
        Positions: linear interpolation.
        Rotations: SLERP (Spherical Linear Interpolation via scipy).
        """
        from scipy.spatial.transform import Rotation, Slerp

        positions = np.array([p.position for p in self._poses])
        rotations = Rotation.from_matrix(np.array([p.rotation_matrix for p in self._poses]))

        # Linear interpolation for xyz positions
        interp_positions = np.array([
            np.interp(target_times, self.times, positions[:, i])
            for i in range(positions.shape[1])
        ]).T

        # SLERP for rotations: guarantees shortest-path traversal on SO(3)
        slerp_fn = Slerp(self.times, rotations)
        interp_rotations = slerp_fn(target_times)

        interp_poses = [
            EndEffectorPose(position=interp_positions[i],
                            rotation_matrix=interp_rotations[i].as_matrix())
            for i in range(len(target_times))
        ]
        return EndEffectorActionChunk(interp_poses, times=target_times)

    def delta_chunking(self, reference_frame=None):
        """
        For end-effectors, override to handle rotation deltas correctly via SLERP.
        Position delta: simple subtraction.
        Rotation delta: relative rotation R_delta = R_prev^{-1} @ R_current.
        """
        from scipy.spatial.transform import Rotation

        delta_poses = []
        prev_pose = reference_frame if reference_frame is not None else self._poses[0]
        for current_pose in self._poses:
            # Position: regular difference
            pos_delta = current_pose.position - prev_pose.position
            # Rotation: relative rotation in SO(3)
            R_prev = Rotation.from_matrix(prev_pose.rotation_matrix)
            R_curr = Rotation.from_matrix(current_pose.rotation_matrix)
            R_delta = R_prev.inv() * R_curr
            delta_poses.append(EndEffectorPose(
                position=pos_delta,
                rotation_matrix=R_delta.as_matrix()
            ))
            prev_pose = current_pose
        return EndEffectorActionChunk(delta_poses, times=self.times.tolist())


class JointActionChunk(ActionChunk):
    """
    ActionChunk for joint-angle sequences.
    Positions are just floats (joint angles), so plain subtraction works fine.
    relative_chunking and delta_chunking are inherited from ActionChunk without override.
    """
    pass
```

## 逐行讲解 / What's happening

1. **`relative_chunking(reference_frame=None)`**:
   - 中文: 所有 pose 同时减去同一个参考帧（默认首帧），结果序列的第一个元素变成零向量。等价于"以出发点为坐标原点"。机器人在不同位置启动，经过这步归一化后轨迹形状一致。
   - English: Subtracts the same reference frame from every pose (default: the first frame). The resulting first pose is zero. Equivalent to "set the starting position as the origin". Trajectories recorded at different robot positions become shape-identical after this normalization.

2. **`delta_chunking(reference_frame=None)`**:
   - 中文: 逐步差分：第 i 帧 = 第 i 帧 - 第 i-1 帧。模型学到的是"每步走多少"，而不是"走到哪里"。这对长 horizon 或频率较低的控制更友好：误差不会累积到绝对位置里。
   - English: Frame-by-frame differences: pose[i] = pose[i] - pose[i-1]. The model learns "how much to move each step" rather than "where to be". This is friendlier for long horizons or low-frequency control: errors don't accumulate in absolute position estimates.

3. **`return self.__class__(...)` 而不是 `return ActionChunk(...)`**:
   - 中文: 用 `self.__class__` 而不是硬编码父类名称，让子类（`EndEffectorActionChunk`、`JointActionChunk`）在 `relative_chunking` 里也能返回自己的类型，不需要重写。
   - English: Using `self.__class__` instead of hardcoding `ActionChunk` lets subclasses return their own type from `relative_chunking` without overriding it — clean polymorphism.

4. **`EndEffectorActionChunk.interpolate()` 用 `Slerp`**:
   - 中文: 位置（xyz）可以线性插值；旋转不行——旋转矩阵或四元数的线性插值会导致非单位旋转（unnormalized）。Slerp 在 SO(3) 上走测地线，总是输出合法旋转矩阵。
   - English: Position (xyz) can be linearly interpolated; rotation cannot — linear interpolation of rotation matrices produces non-unit/invalid rotations. Slerp walks the geodesic on SO(3) and always outputs a valid rotation matrix.

5. **`EndEffectorActionChunk.delta_chunking()` 里的 `R_delta = R_prev.inv() * R_curr`**:
   - 中文: 旋转"差分"是相对旋转：R_delta 是"从 R_prev 到 R_curr 需要再转多少"。这在 SE(3)（刚体变换群）上是正确的差分，不同于欧氏空间里的减法。
   - English: The rotation "delta" is a relative rotation: R_delta encodes "how much more rotation to apply on top of R_prev to reach R_curr". This is the correct difference on SE(3) (the rigid body transform group), unlike Euclidean subtraction.

## 类比 / The analogy

**relative_chunking** 像 GPS 导航切换到"以我为中心"模式：无论你从北京还是上海出发，导航显示的轨迹形状完全一样（都从原点开始）。

**delta_chunking** 像记步数：你不记录"现在在哪"，只记录"每步走了几米"。地球上任何地方的步行路线都能用同一套指令描述——只要起点对齐就行。

**SLERP** 处理旋转差分就像开车转弯：不能直接"平均"方向盘角度（线性平均会穿过地面），必须沿着圆弧走最短路径。

**relative_chunking** is like GPS switching to "centered on me" mode: whether you start from Beijing or Shanghai, the displayed trajectory shape is identical (both start at the origin).

**delta_chunking** is like counting steps: you don't record "where am I" but "how far did I walk each step." The same instruction sequence works at any starting location — as long as you align the origin.

**SLERP** for rotation deltas is like navigating a turn: you can't just "average" steering angles (linear average would clip through the ground) — you must follow the shortest arc.

## 在 nanoVLA 中的位置 / Where this lives in your nanoVLA

这个组件属于 nanoVLA 课程的 `action-chunking` 模块，依赖关系为空（不依赖其他课程组件）。

In the nanoVLA curriculum this is the `action-chunking` component, with no upstream dependencies.

中文：在你的 nanoVLA 里，动作头（`action-head-continuous`）输出的是原始绝对 pose 序列（shape `(T, action_dim)`）。在把这个序列存入训练数据之前，要先调用 `delta_chunking()` 把它转成逐帧差分。推理时，模型输出的是 delta 序列，需要累加还原出绝对 pose 才能发给机器人。如果省略这一步，模型会学到 home position 偏差，在新机器人上失效。

In your nanoVLA, the action head (`action-head-continuous`) outputs a raw absolute pose sequence of shape `(T, action_dim)`. Before storing this in your training dataset, call `delta_chunking()` to convert it to per-frame deltas. At inference time, the model outputs a delta sequence that you must cumulatively sum to recover absolute poses before sending commands to the robot. Skipping this step means the model learns home-position bias and breaks when deployed on a different robot.

上游：`action-head-continuous`（产生 pose 序列）
下游：数据加载器（存 delta 序列）/ 推理环境（`np.cumsum` 还原绝对 pose）
对端执行器类型：使用 `EndEffectorActionChunk`，确保旋转用 SLERP 而不是线性差分。

Upstream: `action-head-continuous` (produces the pose sequence)
Downstream: data loader (stores delta sequences) / inference environment (`np.cumsum` recovers absolute poses)
For end-effector action types: use `EndEffectorActionChunk` to ensure rotations use SLERP instead of linear subtraction.

## 自己跑一遍 / Try it yourself

```python
import numpy as np

class Pose:
    def __init__(self, data): self.data = np.array(data, dtype=float)
    def __sub__(self, other): return Pose(self.data - other.data)
    def __repr__(self): return f"Pose({np.round(self.data, 3)})"

class ActionChunk:
    def __init__(self, poses, times=None):
        self._poses = poses
        self.times = np.arange(len(poses)) if times is None else np.array(times)
    def relative_chunking(self, ref=None):
        r = self._poses[0] if ref is None else ref
        return ActionChunk([p - r for p in self._poses], self.times)
    def delta_chunking(self, ref=None):
        deltas, prev = [], (self._poses[0] if ref is None else ref)
        for p in self._poses:
            deltas.append(p - prev); prev = p
        return ActionChunk(deltas, self.times.tolist())
    def __repr__(self): return "\n".join(str(p) for p in self._poses)

poses = [Pose([1.0, 2.0]), Pose([1.5, 2.5]), Pose([2.0, 3.5])]
chunk  = ActionChunk(poses)
print("relative:\n", chunk.relative_chunking())
print("delta:\n",    chunk.delta_chunking())
# Cumulative sum recovers originals:
deltas = [p.data for p in chunk.delta_chunking()._poses]
print("recovered:", np.cumsum(deltas, axis=0) + poses[0].data)
```

运行 / Run with:
```bash
pip install numpy
python try.py
```

预期输出 / Expected output:
```
relative:
Pose([0. 0.])  Pose([0.5 0.5])  Pose([1.  1.5])
delta:
Pose([0. 0.])  Pose([0.5 0.5])  Pose([0.5 1. ])
recovered: [[1.  2. ] [1.5 2.5] [2.  3.5]]
```

中文：注意 `recovered` 精确还原了原始 pose 序列，验证 delta_chunking 是可逆的。

The `recovered` array exactly matches the original poses, proving `delta_chunking` is invertible.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **`huggingface/lerobot` ACT 策略** / **`huggingface/lerobot` ACT policy**: 使用 `action_queue` 做时间维度的滑动执行，和 `delta_chunking` 配合使用，实现闭环 chunk rollout / uses `action_queue` for temporal sliding execution; pairs with delta_chunking for closed-loop chunk rollout.
- **openpi `pi0` 训练脚本** / **openpi `pi0` training script**: flow-matching action head 训练时对 action chunk 做全局归一化（mean/std），和 relative_chunking 的思路互补——一个去绝对位置，一个去量纲 / flow-matching action head training applies global normalization (mean/std) to action chunks; complementary to relative_chunking — one removes absolute position, the other removes scale.
- **OpenVLA-OFT** / **OpenVLA-OFT**: 使用 delta action，并在推理端用累加器把 delta 还原为绝对 pose，与 `delta_chunking` + `np.cumsum` 一一对应 / uses delta actions and a cumulative adder at inference time — exactly the `delta_chunking + np.cumsum` pattern.

## 注意事项 / Caveats / when it breaks

- **delta_chunking 的首元素永远是零向量** / **delta_chunking's first element is always a zero vector**: 如果模型用 teacher-forcing 训练，要注意 target 序列里有一个"哑"零元素，可能需要 offset 对齐 / if training with teacher-forcing, be aware the target sequence has one "dummy" zero element — may need an index offset.
- **旋转表示不一致时 SLERP 会失败** / **SLERP fails when rotation representations are inconsistent**: 训练数据中若有混用四元数和矩阵的，需要先统一到同一表示再调 `interpolate()` / if training data mixes quaternions and rotation matrices, normalize to one representation before calling `interpolate()`.
- **长 horizon + delta 的累积误差** / **Accumulated error for long-horizon + delta**: open-loop 累加 delta 会把模型误差放大。生产中需要 re-anchor（每 k 步用观测重置参考帧）。

## 延伸阅读 / Further reading

- [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (Zhao et al., ACT, 2023)](https://arxiv.org/abs/2304.13705)
- [Isaac GR00T: A Foundation Model for Humanoid Robots](https://developer.nvidia.com/isaac/groot)
- [Wikipedia: Slerp](https://en.wikipedia.org/wiki/Slerp)
