---
date: 2026-07-07
topic: diffusion
source: trending
repo: nvidia-cosmos/cosmos-predict2.5
file: cosmos_predict2/action_conditioned.py
permalink: https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/cosmos_predict2/action_conditioned.py#L35-L123
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, action-conditioned-video]
---

# Cosmos action-conditioned video：把机器人状态改成相对动作 / Cosmos Action-Conditioned Video: Convert Robot States into Relative Actions

> **一句话 / In one line**: Cosmos-Predict2.5 从连续机器人状态中计算相邻帧的局部坐标相对位移和相对旋转，作为 action-conditioned world model 的动作输入。 / Cosmos-Predict2.5 converts consecutive robot states into local-frame relative translation and rotation actions for action-conditioned world modeling.

## 为什么重要 / Why this matters

动作条件视频模型不应该只看“下一帧是什么”，还要知道机器人做了什么。绝对位姿会强依赖场景坐标系；相对动作更接近控制命令，也更容易跨初始位置泛化。Cosmos 这段代码把状态序列变成 action 序列，是 robot Video2World 的入口之一。

An action-conditioned video model should not only ask “what is the next frame?” It also needs to know what the robot did. Absolute poses depend heavily on the scene frame; relative actions are closer to control commands and generalize better across starting poses. This code turns state sequences into action sequences for robot Video2World.

## 代码 / The code

`nvidia-cosmos/cosmos-predict2.5` — [`cosmos_predict2/action_conditioned.py`](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/cosmos_predict2/action_conditioned.py#L35-L123)

```python
def _get_robot_states(label, state_key="state", gripper_key="continuous_gripper_state"):
    all_states = np.array(label[state_key])
    all_cont_gripper_states = np.array(label[gripper_key])
    return all_states, all_cont_gripper_states


def _get_actions(arm_states, gripper_states, sequence_length, use_quat=False):
    if use_quat:
        action = np.zeros((sequence_length - 1, 8))
    else:
        action = np.zeros((sequence_length - 1, 7))

    for k in range(1, sequence_length):
        prev_xyz = arm_states[k - 1, 0:3]
        prev_rpy = arm_states[k - 1, 3:6]
        prev_rotm = euler2rotm(prev_rpy)
        curr_xyz = arm_states[k, 0:3]
        curr_rpy = arm_states[k, 3:6]
        curr_gripper = gripper_states[k]
        curr_rotm = euler2rotm(curr_rpy)
        rel_xyz = np.dot(prev_rotm.T, curr_xyz - prev_xyz)
        rel_rotm = prev_rotm.T @ curr_rotm

        if use_quat:
            rel_rot = rotm2quat(rel_rotm)
            action[k - 1, 0:3] = rel_xyz
            action[k - 1, 3:7] = rel_rot
            action[k - 1, 7] = curr_gripper
        else:
            rel_rot = rotm2euler(rel_rotm)
            action[k - 1, 0:3] = rel_xyz
            action[k - 1, 3:6] = rel_rot
            action[k - 1, 6] = curr_gripper
    return action


def get_action_sequence_from_states(data, fps_downsample_ratio=1, use_quat=False,
                                    state_key="state", gripper_scale=1.0,
                                    gripper_key="continuous_gripper_state", action_scaler=20.0):
    arm_states, cont_gripper_states = _get_robot_states(data, state_key, gripper_key)
    actions = _get_actions(
        arm_states[::fps_downsample_ratio],
        cont_gripper_states[::fps_downsample_ratio],
        len(data[state_key][::fps_downsample_ratio]),
        use_quat=use_quat,
    )
    actions *= np.array([action_scaler, action_scaler, action_scaler,
                         action_scaler, action_scaler, action_scaler, gripper_scale])
    return actions
```

## 逐行讲解 / What's happening

1. **第 1-4 行 / Lines 1-4 (`_get_robot_states`)**:
   - 中文: 从 label 字典中取出机械臂状态和连续夹爪状态。
   - English: robot arm states and continuous gripper states are extracted from the label dictionary.
2. **第 8-11 行 / Lines 8-11 (action shape)**:
   - 中文: Euler 动作是 7 维；quat 动作是 8 维。
   - English: Euler actions are 7D; quaternion actions are 8D.
3. **第 14-22 行 / Lines 14-22 (relative transform)**:
   - 中文: 当前位移先减去上一帧位置，再乘 `prev_rotm.T` 转到上一帧末端坐标系。
   - English: current translation subtracts the previous position, then multiplies by `prev_rotm.T` to move into the previous end-effector frame.
4. **第 24-35 行 / Lines 24-35 (rotation + gripper)**:
   - 中文: 旋转也用 `prev_rotm.T @ curr_rotm` 表示“从上一帧转到当前帧”。
   - English: rotation uses `prev_rotm.T @ curr_rotm` to represent “rotate from previous frame to current frame.”
5. **第 39-51 行 / Lines 39-51 (downsample + scale)**:
   - 中文: 可以按帧率下采样，再统一缩放动作量级。
   - English: the sequence can be downsampled by FPS ratio, then action magnitudes are scaled consistently.

## 类比 / The analogy

绝对位姿像地图上的经纬度；相对动作像导航提示“向前 10 米，右转 15 度”。机器人策略通常更需要后者，因为它描述的是从当前姿态出发该怎么动。

Absolute pose is like latitude and longitude on a map; relative action is like “move forward 10 meters, turn right 15 degrees.” Robot policies usually need the latter because it describes what to do from the current pose.

## 自己跑一遍 / Try it yourself

```python
import numpy as np
prev_xyz = np.array([1., 2., 0.])
curr_xyz = np.array([2., 2., 0.])
prev_rotm = np.eye(3)
rel_xyz = prev_rotm.T @ (curr_xyz - prev_xyz)
action_scaler = 20.0
print((rel_xyz * action_scaler).tolist())
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
[20.0, 0.0, 0.0]
```

这个例子里上一帧坐标系和世界坐标系一致，所以相对位移就是 x 方向前进一步，再乘训练用缩放系数。

Here the previous frame matches the world frame, so the relative motion is one step along x, then multiplied by the training scale.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **openpi DeltaActions** / **openpi DeltaActions**: 也把绝对动作改写成相对控制。 / It also rewrites absolute actions into relative control.
- **Diffusion Policy action normalizers** / **Diffusion Policy action normalizers**: 相对动作后通常还会做 scale/offset 归一化。 / Relative actions are often followed by scale/offset normalization.

## 注意事项 / Caveats / when it breaks

- **旋转表示要一致** / **Rotation conventions must match**: Euler 角顺序或坐标系错了，相对旋转会错。 / If Euler order or coordinate frames differ, relative rotations are wrong.
- **缩放系数会影响训练稳定性** / **Scaling affects training stability**: `action_scaler` 太大或太小都会改变 loss 尺度。 / Too large or too small `action_scaler` changes the loss scale.

## 延伸阅读 / Further reading

- [Cosmos-Predict2.5 action-conditioned inference](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/cosmos_predict2/action_conditioned.py)
