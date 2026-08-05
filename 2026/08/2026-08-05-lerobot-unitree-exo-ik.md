---
date: 2026-08-05
topic: robotics
source: tracked
repo: huggingface/lerobot
file: src/lerobot/teleoperators/unitree_g1/exo_ik.py
permalink: https://github.com/huggingface/lerobot/blob/64b23178d5348609c266250d3e1f511eba4c33ff/src/lerobot/teleoperators/unitree_g1/exo_ik.py#L239-L353
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, robotics, teleoperation, inverse-kinematics]
---

# LeRobot Unitree 外骨骼 IK：先算腕部目标，再解机器人关节 / LeRobot Unitree Exo IK: Compute Wrist Targets, Then Solve Robot Joints

> **一句话 / In one line**: 外骨骼关节先通过 FK 变成左右腕的世界坐标目标，再交给 G1 的双臂 IK 求出机器人电机角。 / Exoskeleton joints are first turned into left/right wrist targets with FK, then G1 arm IK converts those targets into robot motor angles.

## 为什么重要 / Why this matters

遥操作不是把人的每个关节一一拷贝到机器人上。人的外骨骼和 Unitree G1 的连杆长度、零位、关节定义都不同，直接拷贝角度会错。这里的中间表示是末端执行器位姿：外骨骼负责告诉系统“手腕应该在世界里哪里”，机器人 IK 再负责“用自己的关节到达那里”。

Teleoperation is not a joint-by-joint copy. The exoskeleton and Unitree G1 have different link lengths, zero poses, and joint definitions. Copying angles directly would be wrong. This code uses end-effector pose as the bridge: the exoskeleton says where the wrist should be in world space, and the robot IK decides how its own joints should reach that target.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/teleoperators/unitree_g1/exo_ik.py`](https://github.com/huggingface/lerobot/blob/64b23178d5348609c266250d3e1f511eba4c33ff/src/lerobot/teleoperators/unitree_g1/exo_ik.py#L239-L353)

```python
def _fk_target_world(self, side: str, angles: dict[str, float]) -> np.ndarray | None:
    """returns wrist frame target to be used for G1 IK in 4x4 homogeneous transform. Takes offset into account."""
    if side not in self.exo or not angles:
        return None

    pin = self.pin
    q = self.q_exo[side]
    qmap = self.qmap[side]

    for name, ang in angles.items():
        idx = qmap.get(name)
        if idx is not None:
            q[idx] = float(ang)

    r = self.exo[side]
    pin.forwardKinematics(r.model, r.data, q)
    pin.updateFramePlacements(r.model, r.data)

    ee = r.data.oMf[self.ee_id_exo[side]]
    target = np.eye(4)
    target[:3, :3] = ee.rotation
    # offset gets applied in world space
    cfg = next(a for a in self.arms if a.side == side)
    target[:3, 3] = cfg.offset + ee.translation
    return target

def compute_g1_joints_from_exo(
    self,
    left_angles: dict[str, float],
    right_angles: dict[str, float],
) -> dict[str, float]:
    """
    Performs FK on exoskeleton to get end-effector poses in world frame,
    after which it solves IK on G1 to return joint angles matching those poses in G1 motor order.
    """
    pin = self.pin

    targets = {
        "left": self._fk_target_world("left", left_angles),
        "right": self._fk_target_world("right", right_angles),
    }

    # fallback to current g1 ee pose if missing target
    pin.forwardKinematics(self.robot_g1.model, self.robot_g1.data, self.q_g1)
    pin.updateFramePlacements(self.robot_g1.model, self.robot_g1.data)

    for a in self.arms:
        if targets[a.side] is not None:
            continue
        fid = self.ee_id_g1.get(a.side)
        if fid is not None:
            targets[a.side] = self.robot_g1.data.oMf[fid].homogeneous

    if targets["left"] is None or targets["right"] is None:
        logger.warning("missing ik targets, returning current pose")
        return {}

    frozen_vals = {n: self.q_g1[i] for n, i in self.frozen_idx.items()}

    self.q_g1, _ = self.g1_ik.solve_ik(
        targets["left"], targets["right"], current_lr_arm_motor_q=self.q_g1
    )

    for n, i in self.frozen_idx.items():
        self.q_g1[i] = frozen_vals[n]

    return {
        f"{j.name}.q": float(self.q_g1[i])
        for i, j in enumerate(G1_29_JointArmIndex)
        if i < len(self.q_g1)
    }
```

## 逐行讲解 / What's happening

1. **第 241-252 行 / Lines 241-252 (update exo joint vector)**:
   - 中文: 先检查这一侧外骨骼是否存在，再把传入的关节名映射到 Pinocchio 的 `q` 下标。未知关节名会被跳过，避免遥操作端多发字段时直接炸掉。
   - English: The code first checks that this exoskeleton side exists, then maps incoming joint names into Pinocchio `q` indices. Unknown joints are ignored, so extra teleop fields do not crash the path.
2. **第 253-263 行 / Lines 253-263 (FK target)**:
   - 中文: 外骨骼跑 forward kinematics，取腕部 frame 的旋转和平移，再加上世界坐标里的左右臂 offset，得到给 G1 IK 用的 4x4 目标矩阵。
   - English: The exoskeleton runs forward kinematics, reads the wrist frame rotation and translation, adds the arm offset in world coordinates, and produces a 4x4 target matrix for G1 IK.
3. **第 320-338 行 / Lines 320-338 (missing-side fallback)**:
   - 中文: 如果某侧外骨骼没有目标，就用 G1 当前末端位姿补位。这样左手掉线时，右手还能继续动，而不是让双臂 IK 收到半个目标。
   - English: If one exoskeleton side has no target, the current G1 end-effector pose fills the slot. If the left side drops out, the right side can still move instead of giving the two-arm IK half a problem.
4. **第 340-353 行 / Lines 340-353 (freeze, solve, restore)**:
   - 中文: 被标记冻结的关节先保存，IK 解完后再写回，最后按 G1 电机枚举返回 `{joint.q: angle}`。
   - English: Frozen joints are saved before IK, restored afterward, and the result is returned in G1 motor-name order as `{joint.q: angle}`.

## 类比 / The analogy

这像让一个人戴着手套操控机械臂。你不会要求机械臂每根“手指骨”弯同样角度，而是看手套掌心最终在哪里，再让机械臂用自己的骨架摆到同一个位置。

It is like controlling a robot arm with a glove. You do not ask each robot bone to bend by the same angle as the glove. You track where the glove palm ends up, then let the robot arrange its own skeleton to reach that pose.

## 自己跑一遍 / Try it yourself

```python
def fk_target(side, angles, offset):
    if not angles:
        return None
    x = sum(angles.values())
    return {"side": side, "pos": [offset[0] + x, offset[1], offset[2]]}

def solve_ik(left, right, frozen):
    q = {"left_shoulder.q": left["pos"][0], "right_shoulder.q": right["pos"][0], "torso.q": 0.4}
    q.update(frozen)
    return q

targets = {
    "left": fk_target("left", {"elbow": 0.2, "wrist": 0.1}, [0.6, 0.3, 0.0]),
    "right": fk_target("right", {"elbow": -0.1}, [0.6, -0.3, 0.0]),
}
print(solve_ik(targets["left"], targets["right"], {"torso.q": 0.0}))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
{'left_shoulder.q': 0.9, 'right_shoulder.q': 0.5, 'torso.q': 0.0}
```

中文: 关键不是角度相同，而是先把输入统一成目标位姿，再让机器人自己的 IK 求关节。

English: The point is not equal angles. The input is first normalized into target poses, and the robot's own IK computes the joints.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **VR 手柄遥操作 / VR-controller teleop**: 手柄姿态通常先变成 wrist pose，再进机器人 IK。 / Controller poses are usually converted into wrist poses before robot IK.
- **运动捕捉到人形机器人 / Motion capture to humanoids**: mocap 骨架也常通过末端/关键点目标，而不是逐关节硬拷贝。 / Mocap retargeting often goes through endpoint or keypoint targets instead of copying joint angles.

## 注意事项 / Caveats / when it breaks

- **坐标系 offset 必须校准 / Offsets must be calibrated**: offset 错了，IK 会认真追一个错误目标。 / If the offset is wrong, IK faithfully tracks the wrong target.
- **冻结关节会改变可达性 / Frozen joints change reachability**: 冻得太多，末端目标可能无解或抖动。 / Freezing too many joints can make targets unreachable or unstable.

## 延伸阅读 / Further reading

- LeRobot `exo_ik.py`: https://github.com/huggingface/lerobot/blob/64b23178d5348609c266250d3e1f511eba4c33ff/src/lerobot/teleoperators/unitree_g1/exo_ik.py#L239-L353
