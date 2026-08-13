---
date: 2026-08-13
topic: robotics
source: trending
repo: NVIDIA/IsaacTeleop
file: src/python/isaacteleop/retargeters/joint_space/joint_state_retargeter.py
permalink: https://github.com/NVIDIA/IsaacTeleop/blob/0f3916754f348722f0dc344e6a57892718e6ba08/src/python/isaacteleop/retargeters/joint_space/joint_state_retargeter.py#L204-L289
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, retargeting]
---

# IsaacTeleop JointStateRetargeter：把设备关节翻译成机器人动作 / IsaacTeleop JointStateRetargeter: Translate Device Joints into Robot Actions

> **一句话 / In one line**: `JointStateRetargeter` 支持 joint 直通和 EE pose 两种模式，用同一份 name-keyed joint input 生成 Isaac Lab 需要的动作输出。 / `JointStateRetargeter` supports joint pass-through and EE-pose modes, turning the same name-keyed joint input into Isaac Lab action outputs.

## 为什么重要 / Why this matters

遥操作的难点不是读到一个关节角，而是把不同设备、不同机器人、不同 action layout 接成稳定接口。这段代码把模式分发、缺失输入保持、关节名映射和 scale/sign/offset 放在一个 retargeter 里。

The hard part of teleoperation is not reading a joint angle; it is connecting different devices, robots, and action layouts through a stable interface. This code keeps mode dispatch, missing-input hold behavior, joint-name mapping, and scale/sign/offset in one retargeter.

## 代码 / The code

`NVIDIA/IsaacTeleop` — [`src/python/isaacteleop/retargeters/joint_space/joint_state_retargeter.py`](https://github.com/NVIDIA/IsaacTeleop/blob/0f3916754f348722f0dc344e6a57892718e6ba08/src/python/isaacteleop/retargeters/joint_space/joint_state_retargeter.py#L204-L289)

```python
def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
    if self._mode == "joint":
        self._compute_joint(inputs, outputs, context)
    else:
        self._compute_ee(inputs, outputs, context)

def _read_positions(self, joints_group) -> dict[str, float]:
    """Read the name-keyed joint group into a ``{device_name: position}`` dict."""
    return {
        name: float(joints_group[i])
        for i, name in enumerate(self._cfg.device_joints)
    }

def _compute_joint(self, inputs, outputs, context) -> None:
    if context.execution_events.reset:
        self._last_targets = np.zeros(len(self._target_joints), dtype=np.float32)

    out = outputs[JOINT_TARGETS_KEY]
    jin = inputs[self.JOINTS]
    if jin.is_none:
        for i in range(len(self._target_joints)):
            out[i] = float(self._last_targets[i])
        return

    positions = self._read_positions(jin)
    for i, tgt in enumerate(self._target_joints):
        raw = positions.get(self._device_for_target[tgt], 0.0)
        value = (
            self._cfg.offset.get(tgt, 0.0)
            + self._cfg.sign.get(tgt, 1.0) * self._cfg.scale.get(tgt, 1.0) * raw
        )
        self._last_targets[i] = value
        out[i] = float(value)

def _compute_gripper(self, positions: dict[str, float]) -> float:
    raw = positions.get(self._cfg.gripper_joint, 0.0)
    lo, hi = self._cfg.gripper_open, self._cfg.gripper_close
    if lo is None or hi is None or hi == lo:
        return float(raw)
    c = (raw - lo) / (hi - lo)
    return float(min(1.0, max(0.0, c)))
```

## 逐行讲解 / What's happening

1. **第 204-208 行 / Lines 204-208: 一个 graph 节点内部按 `joint` 或 `ee_pose` 分发。 / One graph node dispatches internally into `joint` or `ee_pose` mode.**
2. **第 210-215 行 / Lines 210-215: 输入按 `device_joints` 顺序读出，再变成按名字索引的 dict。 / Input is read in `device_joints` order and converted into a name-keyed dict.**
3. **第 217-236 行 / Lines 217-236: 输入缺失时保持上次输出；输入存在时按 target joint 查 device joint 并应用标定。 / Missing input holds the last output; available input maps target joints to device joints and applies calibration.**
4. **第 283-289 行 / Lines 283-289: gripper 可以透传，也可以从 open/close 标定归一化到 `[0,1]`。 / The gripper can pass through raw values or normalize open/close calibration into `[0,1]`.**

## 类比 / The analogy

像给不同国家插头配转换器：墙上的孔、设备插头、电压方向都可能不同，转换器负责把它们变成电器能稳定使用的接口。

It is like using a power adapter across countries: sockets, plug shapes, and voltage orientation may differ, and the adapter makes them usable by the device.

## 自己跑一遍 / Try it yourself

```python
def retarget(positions, targets, device_for_target, scale, sign, offset):
    out = []
    for tgt in targets:
        raw = positions.get(device_for_target.get(tgt, tgt), 0.0)
        out.append(offset.get(tgt, 0.0) + sign.get(tgt, 1.0) * scale.get(tgt, 1.0) * raw)
    return out

positions = {"leader_elbow": 0.4, "leader_wrist": -0.2}
targets = ["robot_elbow", "robot_wrist"]
map_ = {"robot_elbow": "leader_elbow", "robot_wrist": "leader_wrist"}
print(retarget(positions, targets, map_, {"robot_wrist": 2.0}, {"robot_elbow": -1.0}, {"robot_elbow": 0.1}))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[-0.30000000000000004, -0.4]
```

一个 target joint 的输出由名字映射和三个标定表共同决定，缺一项就用默认值。

A target joint is determined by name mapping and three calibration tables; missing entries fall back to defaults.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LeRobot policy/robot bridge** / **LeRobot policy/robot bridge**: 也把 policy tensor 和 robot action dict 分开。 / It also separates policy tensors from robot action dicts.
- **Unitree exo IK retargeting** / **Unitree exo IK retargeting**: 端到端遥操作常把 leader pose 转成 EE target 再解 robot joints。 / End-to-end teleoperation often converts leader pose into EE target before solving robot joints.

## 注意事项 / Caveats / when it breaks

- **名字顺序是硬合同** / **Name order is a hard contract**: upstream source 和 `device_joints` 顺序不一致会错位。 / If upstream source and `device_joints` disagree, values shift to wrong joints.
- **hold-last 不是安全控制器** / **Hold-last is not a safety controller**: 真实机器人还需要超时和急停策略。 / Real robots still need timeout and emergency-stop policies.

## 延伸阅读 / Further reading

- [NVIDIA/IsaacTeleop source](https://github.com/NVIDIA/IsaacTeleop/blob/0f3916754f348722f0dc344e6a57892718e6ba08/src/python/isaacteleop/retargeters/joint_space/joint_state_retargeter.py#L204-L289)
