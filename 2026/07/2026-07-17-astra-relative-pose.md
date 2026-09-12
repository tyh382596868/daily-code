---
date: 2026-07-17
topic: diffusion
source: trending
repo: EternalEvan/Astra
file: train_single.py
permalink: https://github.com/EternalEvan/Astra/blob/main/train_single.py
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, world-model, camera, pose]
---

# Astra relative pose：把相机轨迹变成局部动作 / Astra Relative Pose: Turn Camera Trajectory into Local Actions

> **一句话 / In one line**: Astra 训练前把相邻相机 pose 转成相对变换，让视频世界模型学习“下一步怎么动”而不是记全局坐标。 / Before training, Astra converts neighboring camera poses into relative transforms so the video world model learns "how the camera moves next" instead of memorizing global coordinates.

## 为什么重要 / Why this matters

action-conditioned world model 的动作不一定是机器人关节，也可以是相机运动。关键是动作要在局部坐标系里表达：同样的“前进一米”不应因为场景全局坐标不同而变成两个标签。Astra 的相对 pose helper 把全局 pose 差值变成可泛化的局部控制信号。

An action-conditioned world model does not need robot joints; camera motion can be the action. The action should live in a local coordinate frame: the same "move forward one meter" should not become two labels because the scene's global coordinates differ. Astra's relative-pose helper turns global pose differences into generalizable local controls.

## 代码 / The code

`EternalEvan/Astra` — [`train_single.py`](https://github.com/EternalEvan/Astra/blob/main/train_single.py)

```python
def compute_relative_pose(pose_a, pose_b, use_torch=False):
    if use_torch:
        pose_a_inv = torch.inverse(pose_a.float())
        relative_pose = torch.matmul(pose_b.float(), pose_a_inv)
    else:
        pose_a_inv = np.linalg.inv(pose_a)
        relative_pose = np.matmul(pose_b, pose_a_inv)
    return relative_pose

def compute_relative_pose_matrix(pose1, pose2):
    t1, q1 = pose1[:3], pose1[3:]
    t2, q2 = pose2[:3], pose2[3:]

    rot1 = R.from_quat(q1)
    rot2 = R.from_quat(q2)
    rot_rel = rot2 * rot1.inv()
    R_rel = rot_rel.as_matrix()
    t_rel = rot1.as_matrix().T @ (t2 - t1)
    return np.concatenate([R_rel, t_rel[:, None]], axis=1)
```

## 逐行讲解 / What's happening

1. **矩阵版直接相乘 / Matrix form multiplies transforms**: 中文: `pose_b @ inv(pose_a)` 得到 B 相对 A 的位姿。 English: `pose_b @ inv(pose_a)` gives B relative to A.
2. **四元数版分开算旋转和平移 / Quaternion form splits rotation and translation**: 中文: 旋转用 `rot2 * rot1.inv()`。 English: rotation uses `rot2 * rot1.inv()`.
3. **平移进入 A 的局部坐标 / Translation enters A's local frame**: 中文: `R1^T @ (t2 - t1)` 把世界坐标差转回当前相机坐标。 English: `R1^T @ (t2 - t1)` maps world displacement into the current camera frame.
4. **输出 3x4 控制矩阵 / Output is a 3x4 control matrix**: 中文: `[R_rel | t_rel]` 可直接作为条件。 English: `[R_rel | t_rel]` can be used as conditioning.
5. **NumPy/Torch 双路径 / NumPy and Torch paths**: 中文: 预处理和训练图内计算都能复用。 English: preprocessing and graph-side computation can share the helper.

## 类比 / The analogy

这像导航指令不用“走到经纬度 X”，而说“向前一步，左转十度”。世界模型更容易学局部动作对画面的影响。

It is like giving directions as "move forward, turn left ten degrees" instead of "go to latitude X." A world model learns local motion effects more easily.

## 自己跑一遍 / Try it yourself

```python
import numpy as np

R1 = np.eye(3)
t1 = np.array([10.0, 0.0, 0.0])
R2 = np.eye(3)
t2 = np.array([11.0, 2.0, 0.0])

t_rel = R1.T @ (t2 - t1)
rel = np.concatenate([R2 @ R1.T, t_rel[:, None]], axis=1)
print(rel.tolist())
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 1.0, 0.0]]
```

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Cosmos / driving world models**: ego-motion 常用相对 pose 作为条件。 / Ego-motion is often represented as relative pose.
- **robot delta actions**: 末端执行器动作通常用相对位移而不是绝对坐标。 / End-effector actions often use relative displacement.

## 注意事项 / Caveats / when it breaks

- **四元数顺序必须一致 / Quaternion order must be consistent**: SciPy 使用 `[x, y, z, w]`。 / SciPy uses `[x, y, z, w]`.
- **坐标系约定要写死 / Coordinate conventions must be fixed**: camera/world 方向错会让动作反号。 / Wrong camera/world conventions flip controls.
- **相对动作会累积漂移 / Relative actions can drift**: 长 rollout 仍需要历史记忆或校正。 / Long rollouts still need memory or correction.

## 延伸阅读 / Further reading

- [Astra repository](https://github.com/EternalEvan/Astra)
- [Astra training entry](https://github.com/EternalEvan/Astra/blob/main/train_single.py)
