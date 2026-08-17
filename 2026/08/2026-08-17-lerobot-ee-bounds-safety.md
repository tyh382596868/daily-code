---
date: 2026-08-17
topic: robotics
source: tracked
repo: huggingface/lerobot
file: src/lerobot/robots/so_follower/robot_kinematic_processor.py
permalink: https://github.com/huggingface/lerobot/blob/6adf51511b7625090eade8d82d9f61a1846ebe56/src/lerobot/robots/so_follower/robot_kinematic_processor.py#L190-L262
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, safety]
---

# LeRobot EE safety：末端目标先限幅再限速 / LeRobot EE Safety: Clip the Target, Then Rate-Limit the Step

> **一句话 / In one line**: `EEBoundsAndSafety` 先把末端执行器目标裁进工作空间，再检查相邻控制帧之间的跳变是否过大。 / `EEBoundsAndSafety` clips the end-effector target into the workspace, then checks whether the per-frame motion jump is too large.

## 为什么重要 / Why this matters

真实机器人不是仿真里的无限速度点。策略或遥操作偶尔会给出离谱目标，这段代码把“目标位置是否合法”和“这一步是否跳太远”放在机器人动作管线里，能在命令进入 IK 或电机前拦住风险。

Real robots are not infinitely fast points in simulation. A policy or teleop stream can emit a bad target, and this step puts workspace clipping plus jump detection directly in the robot action pipeline before IK or motors see the command.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/robots/so_follower/robot_kinematic_processor.py`](https://github.com/huggingface/lerobot/blob/6adf51511b7625090eade8d82d9f61a1846ebe56/src/lerobot/robots/so_follower/robot_kinematic_processor.py#L190-L262)

```python
@ProcessorStepRegistry.register("ee_bounds_and_safety")
@dataclass
class EEBoundsAndSafety(RobotActionProcessorStep):
    """
    Clips the end-effector pose to predefined bounds and checks for unsafe jumps.

    This step ensures that the target end-effector pose remains within a safe operational workspace.
    It also moderates the command to prevent large, sudden movements between consecutive steps.
    """

    end_effector_bounds: dict
    max_ee_step_m: float = 0.05
    raise_on_jump: bool = True
    _last_pos: np.ndarray | None = field(default=None, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        x = action["ee.x"]
        y = action["ee.y"]
        z = action["ee.z"]
        wx = action["ee.wx"]
        wy = action["ee.wy"]
        wz = action["ee.wz"]

        if None in (x, y, z, wx, wy, wz):
            raise ValueError(
                "Missing required end-effector pose components: x, y, z, wx, wy, wz must all be present in action"
            )

        pos = np.array([x, y, z], dtype=float)
        twist = np.array([wx, wy, wz], dtype=float)

        pos = np.clip(pos, self.end_effector_bounds["min"], self.end_effector_bounds["max"])

        if self._last_pos is not None:
            dpos = pos - self._last_pos
            n = float(np.linalg.norm(dpos))
            if n > self.max_ee_step_m and n > 0:
                pos = self._last_pos + dpos * (self.max_ee_step_m / n)
                if self.raise_on_jump:
                    raise ValueError(f"EE jump {n:.3f}m > {self.max_ee_step_m}m")
                logger.warning(
                    "EE jump %.3fm > %.3fm; rate-limited to the per-frame step "
                    "(likely a transient tracking glitch; if it recurs every frame "
                    "the commanded target is systematically out of workspace).",
                    n,
                    self.max_ee_step_m,
                )

        self._last_pos = pos
        action["ee.x"] = float(pos[0])
        action["ee.y"] = float(pos[1])
        action["ee.z"] = float(pos[2])
        action["ee.wx"] = float(twist[0])
        action["ee.wy"] = float(twist[1])
        action["ee.wz"] = float(twist[2])
        return action
```

## 逐行讲解 / What's happening

1. **第 210-213 行 / Lines 210-213**: 中文: `end_effector_bounds` 是硬工作空间，`max_ee_step_m` 是单帧最大位移，`_last_pos` 让这个 processor 带有短期状态。 / English: `end_effector_bounds` defines the hard workspace, `max_ee_step_m` limits each frame, and `_last_pos` gives the processor short-term state.
2. **第 224-233 行 / Lines 224-233**: 中文: 缺少任一姿态分量就直接报错，然后只对位置 `x/y/z` 做边界裁剪。 / English: Missing pose components fail fast, then only the position part is clipped to the configured bounds.
3. **第 236-245 行 / Lines 236-245**: 中文: 如果本帧和上一帧距离过大，代码沿原方向缩短向量；`raise_on_jump` 决定是中断还是限速继续。 / English: If the step is too large, the vector is shortened in the same direction; `raise_on_jump` chooses aborting or rate-limiting.
4. **第 254-262 行 / Lines 254-262**: 中文: 最终写回同一个 action dict，后面的 IK processor 只会看到已经处理过的安全目标。 / English: The sanitized values are written back into the same action dict, so downstream IK only sees the safe target.

## 类比 / The analogy

像开车进地下车库：入口有限高杆先挡住太高的车，减速带再限制进入速度。两个检查一个管范围，一个管变化率。

It is like entering an underground parking garage: the height bar rejects vehicles outside the space, and the speed bump limits how quickly you enter. One guard checks range, the other checks rate of change.

## 自己跑一遍 / Try it yourself

```python
import math

def clamp_step(last, target, max_step):
    clipped = [max(-1.0, min(1.0, v)) for v in target]
    delta = [c - l for c, l in zip(clipped, last)]
    norm = math.sqrt(sum(d * d for d in delta))
    if norm > max_step:
        clipped = [l + d * max_step / norm for l, d in zip(last, delta)]
    return [round(v, 3) for v in clipped]

print(clamp_step([0.0, 0.0, 0.0], [2.0, 2.0, 0.0], 0.05))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[0.035, 0.035, 0.0]
```

目标先被裁到 `[1, 1, 0]`，再被缩成 5 cm 的一步；这就是范围限制和限速的叠加。

The target is first clipped to `[1, 1, 0]`, then shortened to a 5 cm step. That is workspace clipping plus rate limiting.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **遥操作平滑器** / **Teleop smoothers**: 通常也保存上一帧命令，用低通或限速器避免突变。 / They also keep the previous command and use smoothing or rate limits to avoid sudden jumps.
- **安全控制屏障** / **Control barrier filters**: 更严格的机器人系统会把这种检查升级成可证明的约束投影。 / Stricter robot stacks turn this idea into a provable constrained projection.

## 注意事项 / Caveats / when it breaks

- **只限制位置** / **Only position is bounded**: `wx/wy/wz` 原样写回，姿态安全还需要单独的限幅或限速。 / `wx/wy/wz` are passed through, so orientation safety needs its own bounds or rate limiter.
- **状态要 reset** / **State must be reset**: 换任务或重新连接机器人时，应清掉 `_last_pos`，否则第一步可能被上一段轨迹影响。 / Reset `_last_pos` across tasks or reconnects, otherwise the first step can be constrained by an old trajectory.

## 延伸阅读 / Further reading

- [LeRobot source](https://github.com/huggingface/lerobot/blob/6adf51511b7625090eade8d82d9f61a1846ebe56/src/lerobot/robots/so_follower/robot_kinematic_processor.py#L190-L262)
