---
date: 2026-08-29
topic: robotics
source: trending
repo: isaac-sim/IsaacLab
file: source/isaaclab/isaaclab/managers/action_manager.py
permalink: https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/managers/action_manager.py
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, robotics, simulation, action-manager]
---

# IsaacLab ActionManager：动作先分发，再落到执行器 / IsaacLab ActionManager: Dispatch Actions Before Applying Them

> **一句话 / In one line**: IsaacLab 把一整个 action tensor 切给多个 action term，每个 term 负责自己的预处理和执行。 / IsaacLab slices one full action tensor across action terms, and each term owns its own preprocessing and application.

## 为什么重要 / Why this matters

强化学习环境里的 action 往往不是一个简单向量：一部分控制关节位置，一部分控制夹爪，另一部分可能是基座速度。集中管理切片能让 policy 输出保持一个 tensor，同时让各执行器模块独立演化。

Actions in an RL robotics environment are rarely a single simple vector: some entries may control joints, others a gripper, and others base velocity. Central slicing keeps the policy output as one tensor while actuator modules evolve independently.

## 代码 / The code

`isaac-sim/IsaacLab` — [`source/isaaclab/isaaclab/managers/action_manager.py`](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/managers/action_manager.py)

```python
class ActionManager(ManagerBase):
    """Manager for processing and applying actions for articulated assets."""

    def __init__(self, cfg: object, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._action = torch.zeros((self.num_envs, self.total_action_dim), device=self.device)
        self._prev_action = torch.zeros_like(self._action)

    def process_action(self, action: torch.Tensor):
        """Process actions for all terms."""
        if action.shape != (self.num_envs, self.total_action_dim):
            raise ValueError(
                f"Invalid action shape, expected: {(self.num_envs, self.total_action_dim)}, received: {action.shape}."
            )
        self._prev_action[:] = self._action
        self._action[:] = action.to(self.device)

        idx = 0
        for term in self._terms.values():
            term_actions = action[:, idx : idx + term.action_dim]
            term.process_actions(term_actions)
            idx += term.action_dim

    def apply_action(self) -> None:
        """Apply actions to the simulation."""
        for term in self._terms.values():
            term.apply_actions()
```

## 逐行讲解 / What's happening

1. **初始化 / Initialization**:
   - 中文: manager 维护当前 action 和上一次 action，方便限速、平滑或日志对比。
   - English: The manager keeps both current and previous actions, which supports rate limiting, smoothing, or logging.
2. **shape 检查 / Shape check**:
   - 中文: policy 必须输出 `(num_envs, total_action_dim)`，否则在进入执行器之前就失败。
   - English: The policy must output `(num_envs, total_action_dim)`; invalid shapes fail before actuator code runs.
3. **切片循环 / Slicing loop**:
   - 中文: `idx` 是游标，每个 term 拿走自己那段 action 并做预处理。
   - English: `idx` is a cursor; each term takes its slice and preprocesses it.
4. **apply 阶段 / Apply phase**:
   - 中文: 处理和应用分开，便于先统一解析 action，再在仿真 tick 内一起写入。
   - English: Processing and application are separated, so actions can be parsed first and written into simulation together.

## 类比 / The analogy

这像剧场舞台监督拿到一张总 cue 表：灯光、音响、升降台各取自己的几行，准备完成后再统一执行。

It is like a stage manager reading a master cue sheet: lighting, sound, and lifts each take their own rows, prepare, then execute together.

## 自己跑一遍 / Try it yourself

```python
class Term:
    def __init__(self, name, dim):
        self.name, self.action_dim, self.last = name, dim, None
    def process_actions(self, x):
        self.last = x
    def apply_actions(self):
        print(self.name, self.last)

terms = [Term("arm", 3), Term("gripper", 1), Term("base", 2)]
action = [[0.1, 0.2, 0.3, 1.0, -0.5, 0.0]]

idx = 0
for term in terms:
    term.process_actions([row[idx:idx + term.action_dim] for row in action])
    idx += term.action_dim
for term in terms:
    term.apply_actions()
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
arm [[0.1, 0.2, 0.3]]
gripper [[1.0]]
base [[-0.5, 0.0]]
```

一个 policy 输出被稳定拆成多个执行器输入；这就是 manager 的价值。

One policy output is reliably split into multiple actuator inputs; that is the manager's value.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LeRobot action processors** / **LeRobot action processors**: policy action 会先过 processor，再发给 robot。 / Policy actions pass through processors before reaching the robot.
- **Isaac-GR00T action chunking** / **Isaac-GR00T action chunking**: 一段 action 被拆成不同坐标语义。 / One action chunk is decomposed by coordinate semantics.
- **Gym wrappers** / **Gym wrappers**: wrapper 常把扁平 action 变成环境内部结构。 / Wrappers often convert flat actions into environment-specific structures.

## 注意事项 / Caveats / when it breaks

- **term 顺序是契约** / **Term order is a contract**: 顺序变了，切片含义也会变。 / If term order changes, slice meaning changes.
- **维度总和要一致** / **Dimensions must sum correctly**: `total_action_dim` 必须等于各 term 维度之和。 / `total_action_dim` must equal the sum of term dimensions.
- **处理和执行不要混用** / **Do not mix process and apply**: 预处理阶段不该直接写仿真状态。 / Preprocessing should not directly write simulation state.

## 延伸阅读 / Further reading

- IsaacLab repository: https://github.com/isaac-sim/IsaacLab
- IsaacLab managers: https://isaac-sim.github.io/IsaacLab/
