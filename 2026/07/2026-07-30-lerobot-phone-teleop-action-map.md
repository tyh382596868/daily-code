---
date: 2026-07-30
topic: robotics
source: tracked
repo: huggingface/lerobot
file: src/lerobot/teleoperators/phone/phone_processor.py
permalink: https://github.com/huggingface/lerobot/blob/36b8face988669509272b00f4abe6592d0b17aa0/src/lerobot/teleoperators/phone/phone_processor.py#L26-L112
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, teleoperation, processor]
---

# LeRobot phone teleop：手机坐标要翻译成机器人动作 / LeRobot Phone Teleop: Translate Phone Pose into Robot Actions

> **一句话 / In one line**: `MapPhoneActionToRobotAction` 把手机的 6-DoF 姿态和按键输入改写成机器人控制器能直接消费的 target pose。 / `MapPhoneActionToRobotAction` rewrites phone 6-DoF pose and buttons into target-pose fields a robot controller can consume.

## 为什么重要 / Why this matters

手机遥操作不是把传感器原样丢给机器人。手机坐标系、平台按键、enable 状态和夹爪速度都要先变成统一动作 schema，否则同一个 policy processor pipeline 后面没法复用。

Phone teleoperation is not raw sensor forwarding. Phone coordinate axes, platform buttons, enable state, and gripper velocity must become one action schema before the rest of the policy processor pipeline can reuse them.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/teleoperators/phone/phone_processor.py`](https://github.com/huggingface/lerobot/blob/36b8face988669509272b00f4abe6592d0b17aa0/src/lerobot/teleoperators/phone/phone_processor.py#L26-L112)

```python
@ProcessorStepRegistry.register("map_phone_action_to_robot_action")
@dataclass
class MapPhoneActionToRobotAction(RobotActionProcessorStep):
    """
    Maps calibrated phone pose actions to standardized robot action inputs.

    This processor step acts as a bridge between the phone teleoperator's output
    and the robot's expected action format. It remaps the phone's 6-DoF pose
    (position and rotation) to the robot's target end-effector pose, applying
    necessary axis inversions and swaps. It also interprets platform-specific
    button presses to generate a gripper command.

    Attributes:
        platform: The operating system of the phone (iOS or Android), used
            to determine the correct button mappings for the gripper.
    """

    # TODO(Steven): Gripper vel could be output of phone_teleop directly
    platform: PhoneOS
    _enabled_prev: bool = field(default=False, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        """
        Processes the phone action dictionary to create a robot action dictionary.

        Args:
            act: The input action dictionary from the phone teleoperator.

        Returns:
            A new action dictionary formatted for the robot controller.

        Raises:
            ValueError: If 'pos' or 'rot' keys are missing from the input action.
        """
        # Pop them from the action
        enabled = bool(action.pop("phone.enabled"))
        pos = action.pop("phone.pos")
        rot = action.pop("phone.rot")
        inputs = action.pop("phone.raw_inputs")

        if pos is None or rot is None:
            raise ValueError("pos and rot must be present in action")

        rotvec = rot.as_rotvec()  # Absolute orientation as rotvec

        # Map certain inputs to certain actions
        if self.platform == PhoneOS.IOS:
            gripper_vel = float(inputs.get("a3", 0.0))
        else:
            a = float(inputs.get("reservedButtonA", 0.0))
            b = float(inputs.get("reservedButtonB", 0.0))
            gripper_vel = (
                a - b
            )  # Positive if a is pressed, negative if b is pressed, 0 if both or neither are pressed

        # For some actions we need to invert the axis
        action["enabled"] = enabled
        action["target_x"] = -pos[1] if enabled else 0.0
        action["target_y"] = pos[0] if enabled else 0.0
        action["target_z"] = pos[2] if enabled else 0.0
        action["target_wx"] = rotvec[1] if enabled else 0.0
        action["target_wy"] = rotvec[0] if enabled else 0.0
        action["target_wz"] = -rotvec[2] if enabled else 0.0
        action["gripper_vel"] = gripper_vel  # Still send gripper action when disabled
        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in ["enabled", "pos", "rot", "raw_inputs"]:
            features[PipelineFeatureType.ACTION].pop(f"phone.{feat}", None)

        for feat in [
            "enabled",
            "target_x",
            "target_y",
            "target_z",
            "target_wx",
            "target_wy",
            "target_wz",
            "gripper_vel",
        ]:
            features[PipelineFeatureType.ACTION][f"{feat}"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features
```

## 逐行讲解 / What's happening

1. **第 26-45 行 / Lines 26-45 (`ProcessorStepRegistry`)**:
   - 中文: 这个 processor 被注册成字符串名字，配置和 checkpoint 只需要引用 `map_phone_action_to_robot_action`。
   - English: The processor is registered under a string name, so configs and checkpoints can refer to `map_phone_action_to_robot_action`.
2. **第 61-64 行 / Lines 61-64 (pop phone fields)**:
   - 中文: 输入里的 `phone.*` 字段被消费掉，避免后面 controller 同时看到原始字段和标准字段。
   - English: The `phone.*` fields are consumed so downstream controllers do not see both raw and standardized fields.
3. **第 72-89 行 / Lines 72-89 (axis and gripper mapping)**:
   - 中文: iOS 和 Android 的按键语义不同；位置和旋转也做了轴交换/符号翻转。
   - English: iOS and Android buttons have different semantics; position and rotation also get axis swaps and sign flips.
4. **第 92-112 行 / Lines 92-112 (`transform_features`)**:
   - 中文: feature schema 同步改写：删掉 phone action，声明机器人目标字段。
   - English: The feature schema is rewritten too: phone actions are removed and robot target fields are declared.

## 类比 / The analogy

这像把游戏手柄接到挖掘机上：手柄摇杆的左右，不一定等于铲斗的左右，中间必须有一张接线表。

It is like wiring a game controller to an excavator: stick-left does not automatically mean bucket-left; you need a mapping table in between.

## 自己跑一遍 / Try it yourself

```python
def map_phone(action, platform):
    enabled = bool(action.pop('phone.enabled'))
    pos = action.pop('phone.pos')
    rotvec = action.pop('phone.rotvec')
    inputs = action.pop('phone.raw_inputs')
    if platform == 'ios':
        gripper = float(inputs.get('a3', 0.0))
    else:
        gripper = float(inputs.get('reservedButtonA', 0.0)) - float(inputs.get('reservedButtonB', 0.0))
    action.update({
        'enabled': enabled, 'target_x': -pos[1] if enabled else 0.0,
        'target_y': pos[0] if enabled else 0.0, 'target_z': pos[2] if enabled else 0.0,
        'target_wx': rotvec[1] if enabled else 0.0, 'target_wy': rotvec[0] if enabled else 0.0,
        'target_wz': -rotvec[2] if enabled else 0.0, 'gripper_vel': gripper,
    })
    return action

print(map_phone({'phone.enabled': 1, 'phone.pos': [1,2,3], 'phone.rotvec': [0.1,0.2,0.3], 'phone.raw_inputs': {'reservedButtonA': 1}}, 'android'))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
{'enabled': True, 'target_x': -2, 'target_y': 1, 'target_z': 3, 'target_wx': 0.2, 'target_wy': 0.1, 'target_wz': -0.3, 'gripper_vel': 1.0}
```

最值得看的是 schema 边界：原始 phone 字段消失，机器人目标字段出现。

The important boundary is the schema change: raw phone fields disappear and robot target fields appear.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LeRobot delta action** / **LeRobot delta action**: 同样把用户/策略输出转换为机器人目标动作。 / It also converts user or policy outputs into robot target actions.
- **OpenPI transforms** / **OpenPI transforms**: 用小 transform 串联数据集字段、模型字段和执行字段。 / Small transforms connect dataset fields, model fields, and execution fields.

## 注意事项 / Caveats / when it breaks

- **坐标系约定** / **Coordinate conventions**: 符号翻转错一维，末端执行器就会朝反方向走。 / Flip one sign incorrectly and the end effector moves the wrong way.
- **禁用状态** / **Disabled state**: 位置/旋转在 disabled 时清零，但夹爪速度仍会发送，调用方要知道这个语义。 / Position and rotation are zeroed while disabled, but gripper velocity still passes through; callers must know that contract.

## 延伸阅读 / Further reading

- [LeRobot repository](https://github.com/huggingface/lerobot)
- [Source permalink](https://github.com/huggingface/lerobot/blob/36b8face988669509272b00f4abe6592d0b17aa0/src/lerobot/teleoperators/phone/phone_processor.py#L26-L112)
