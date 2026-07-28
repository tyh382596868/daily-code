---
date: 2026-07-28
topic: robotics
source: tracked
repo: huggingface/lerobot
file: src/lerobot/processor/delta_action_processor.py
permalink: https://github.com/huggingface/lerobot/blob/95211b98f1cd6b638bda84a8d28f9e41323229dd/src/lerobot/processor/delta_action_processor.py#L25-L143
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, robotics, action-processing, teleoperation]
---

# LeRobot delta action：把手柄增量变成机器人目标 / LeRobot Delta Action: Turn Teleop Deltas into Robot Targets

> **一句话 / In one line**: 这段 processor 把策略或遥操作器输出的 `delta_x/y/z` 转成机器人控制器能执行的目标字段，并用阈值过滤微小噪声。 / This processor converts `delta_x/y/z` from a policy or teleoperator into robot-controller target fields, while filtering tiny noise with a threshold.

## 为什么重要 / Why this matters

机器人策略常输出紧凑的连续向量，但底层控制器更常消费带名字的字段，比如 `target_x`、`target_y`、`enabled`。这段代码把两种世界隔开：模型只管预测小向量，processor 负责命名、缩放、去噪和 feature schema 更新。

Robot policies often emit compact continuous vectors, while low-level controllers usually consume named fields such as `target_x`, `target_y`, and `enabled`. This code separates the two worlds: the model predicts a small vector, and the processor handles naming, scaling, denoising, and feature-schema updates.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/processor/delta_action_processor.py`](https://github.com/huggingface/lerobot/blob/95211b98f1cd6b638bda84a8d28f9e41323229dd/src/lerobot/processor/delta_action_processor.py#L25-L143)

```python
@ProcessorStepRegistry.register("map_tensor_to_delta_action_dict")
@dataclass
class MapTensorToDeltaActionDictStep(ActionProcessorStep):
    use_gripper: bool = True

    def action(self, action: PolicyAction) -> RobotAction:
        if not isinstance(action, PolicyAction):
            raise ValueError("Only PolicyAction is supported for this processor")

        if action.dim() > 1:
            action = action.squeeze(0)

        # TODO (maractingi): add rotation
        delta_action = {
            "delta_x": action[0].item(),
            "delta_y": action[1].item(),
            "delta_z": action[2].item(),
        }
        if self.use_gripper:
            delta_action["gripper"] = action[3].item()
        return delta_action

    def transform_features(self, features):
        for axis in ["x", "y", "z"]:
            features[PipelineFeatureType.ACTION][f"delta_{axis}"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )
        if self.use_gripper:
            features[PipelineFeatureType.ACTION]["gripper"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )
        return features


@ProcessorStepRegistry.register("map_delta_action_to_robot_action")
@dataclass
class MapDeltaActionToRobotActionStep(RobotActionProcessorStep):
    position_scale: float = 1.0
    noise_threshold: float = 1e-3

    def action(self, action: RobotAction) -> RobotAction:
        delta_x = action.pop("delta_x")
        delta_y = action.pop("delta_y")
        delta_z = action.pop("delta_z")
        gripper = action.pop("gripper")

        position_magnitude = (delta_x**2 + delta_y**2 + delta_z**2) ** 0.5
        enabled = position_magnitude > self.noise_threshold

        scaled_delta_x = delta_x * self.position_scale
        scaled_delta_y = delta_y * self.position_scale
        scaled_delta_z = delta_z * self.position_scale

        action = {
            "enabled": enabled,
            "target_x": scaled_delta_x,
            "target_y": scaled_delta_y,
            "target_z": scaled_delta_z,
            "target_wx": 0.0,
            "target_wy": 0.0,
            "target_wz": 0.0,
            "gripper_vel": float(gripper),
        }

        return action
```

## 逐行讲解 / What's happening

1. **第 25-27 行 / Lines 25-27 (`register`)**:
   - 中文: 每个转换步骤用字符串注册，pipeline 可以从配置里恢复这一层。
   - English: Each conversion step is registered by string, so a pipeline can restore it from config.
2. **第 42-57 行 / Lines 42-57 (`tensor -> dict`)**:
   - 中文: 策略输出向量被拆成 `delta_x/y/z`，可选第 4 维作为 gripper。
   - English: The policy vector is split into `delta_x/y/z`, with an optional fourth gripper dimension.
3. **第 94-105 行 / Lines 94-105 (`noise gate`)**:
   - 中文: 控制器先算位移模长，小于 `noise_threshold` 就不启用目标动作。
   - English: The controller computes movement magnitude first; below `noise_threshold`, the target is not enabled.
4. **第 107-130 行 / Lines 107-130 (`robot target`)**:
   - 中文: 增量被缩放并重命名成机器人控制接口需要的目标字段。
   - English: Deltas are scaled and renamed into the target fields expected by the robot control interface.

## 类比 / The analogy

这像把游戏手柄摇杆翻译成叉车指令：摇杆只有前后左右的小数，叉车系统要的是“是否启用、目标位置、夹爪速度”这些明确命令。

It is like translating a gamepad stick into forklift commands: the stick gives small numeric deltas, while the vehicle system needs explicit commands such as enabled state, target position, and gripper speed.

## 自己跑一遍 / Try it yourself

```python
def delta_to_robot(delta, scale=2.0, threshold=1e-3):
    dx, dy, dz, gripper = delta
    mag = (dx * dx + dy * dy + dz * dz) ** 0.5
    return {
        "enabled": mag > threshold,
        "target_x": dx * scale,
        "target_y": dy * scale,
        "target_z": dz * scale,
        "target_wx": 0.0,
        "target_wy": 0.0,
        "target_wz": 0.0,
        "gripper_vel": float(gripper),
    }

print(delta_to_robot([0.01, 0.0, -0.02, 0.5]))
print(delta_to_robot([0.0001, 0.0, 0.0, 0.0]))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
{'enabled': True, 'target_x': 0.02, 'target_y': 0.0, 'target_z': -0.04, 'target_wx': 0.0, 'target_wy': 0.0, 'target_wz': 0.0, 'gripper_vel': 0.5}
{'enabled': False, 'target_x': 0.0002, 'target_y': 0.0, 'target_z': 0.0, 'target_wx': 0.0, 'target_wy': 0.0, 'target_wz': 0.0, 'gripper_vel': 0.0}
```

关键点是第二个输入仍会保留目标字段，但 `enabled=False` 让下游知道这只是噪声。

The key detail is that the second input still keeps target fields, but `enabled=False` tells downstream control that it is only noise.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **openpi transforms** / **openpi transforms**: 中文: openpi 也把策略张量和机器人环境字段分层处理。 / English: openpi also separates policy tensors from robot-environment fields.
- **Diffusion Policy normalizer** / **Diffusion Policy normalizer**: 中文: 动作在进入控制器前通常还会经过尺度恢复。 / English: Actions are often unnormalized before they reach the controller.

## 注意事项 / Caveats / when it breaks

- **固定维度** / **Fixed dimensions**: 中文: 这里假设至少 3 维位置和可选 gripper，旋转还没接入。 / English: This assumes at least 3 position dimensions plus optional gripper; rotation is not wired yet.
- **原地 `pop`** / **In-place `pop`**: 中文: 第二个 step 会消费输入 dict，复用同一个对象时要小心。 / English: The second step consumes the input dict, so reuse of the same object needs care.

## 延伸阅读 / Further reading

- [huggingface/lerobot source](https://github.com/huggingface/lerobot/blob/95211b98f1cd6b638bda84a8d28f9e41323229dd/src/lerobot/processor/delta_action_processor.py#L25-L143)
