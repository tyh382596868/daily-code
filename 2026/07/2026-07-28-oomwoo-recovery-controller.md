---
date: 2026-07-28
topic: robotics
source: trending
repo: makerspet/oomwoo
file: contributions/recovery-safety/xbattlax/oomwoo_recovery_safety/oomwoo_recovery_safety/core.py
permalink: https://github.com/makerspet/oomwoo/blob/b3ca46c3207a9ac11687feb5414e97d3db248147/contributions/recovery-safety/xbattlax/oomwoo_recovery_safety/oomwoo_recovery_safety/core.py#L122-L225
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, safety, recovery-state-machine]
---

# oomwoo RecoveryController：安全恢复要有阶梯 / oomwoo RecoveryController: Safety Recovery Needs a Ladder

> **一句话 / In one line**: `RecoveryController` 把碰撞、卡住、定位丢失等事件映射成一串可升级 recovery steps，同时把悬崖、急停等安全事件直接暂停。 / `RecoveryController` maps bumpers, wedging, and localization loss to escalating recovery steps, while cliff, pickup, and e-stop events pause immediately.

## 为什么重要 / Why this matters

真实机器人不能把所有异常都交给策略模型自由发挥。oomwoo 这个开源扫地机器人项目把恢复逻辑写成小状态机：可恢复问题走 ladder，一步失败就升级；不可恢复安全事件进入 paused，等待人工 reset。

Real robots should not let a learned policy improvise every failure mode. This open-source vacuum robot project writes recovery as a small state machine: recoverable problems follow a ladder and escalate on failure; unsafe events enter paused state and wait for manual reset.

## 代码 / The code

`makerspet/oomwoo` — [`contributions/recovery-safety/xbattlax/oomwoo_recovery_safety/oomwoo_recovery_safety/core.py`](https://github.com/makerspet/oomwoo/blob/b3ca46c3207a9ac11687feb5414e97d3db248147/contributions/recovery-safety/xbattlax/oomwoo_recovery_safety/oomwoo_recovery_safety/core.py#L122-L225)

```python
class RecoveryController:
    def __init__(self, ladders: Mapping[Situation, tuple[RecoveryStep, ...]] | None = None):
        self._ladders = dict(ladders or DEFAULT_LADDERS)
        self._state = ControllerState.IDLE
        self._situation: Situation | None = None
        self._step_index = 0
        self._current_step: RecoveryStep | None = None
        self._last_status = self._make_status("READY", "Recovery controller ready", True)

    def trigger(self, situation: Situation | str) -> Decision:
        parsed = self._parse_situation(situation)

        if parsed in SAFETY_SITUATIONS:
            return self._pause(
                parsed,
                reason_code=self._safety_reason(parsed),
                message=f"Safety event {parsed.value} paused the robot",
                recoverable=False,
            )

        if self._state == ControllerState.RECOVERING:
            status = self._make_status(
                "RECOVERY_ALREADY_ACTIVE",
                f"Ignoring {parsed.value}; already recovering from {self._situation.value}",
                True,
            )
            self._last_status = status
            return Decision(DecisionKind.IGNORED, status)

        ladder = self._ladders.get(parsed)
        if not ladder:
            return self._pause(
                parsed,
                reason_code="NO_RECOVERY_LADDER",
                message=f"No recovery ladder configured for {parsed.value}",
                recoverable=True,
            )

        self._state = ControllerState.RECOVERING
        self._situation = parsed
        self._step_index = 0
        self._current_step = ladder[0]
        status = self._make_status(
            "RECOVERY_STARTED",
            f"Starting recovery step {self._current_step.name}",
            True,
        )
        self._last_status = status
        return Decision(DecisionKind.START_STEP, status, self._current_step)

    def step_failed(self, detail: str = "step failed") -> Decision:
        if self._state != ControllerState.RECOVERING or self._situation is None:
            status = self._make_status("NO_ACTIVE_RECOVERY", "No active recovery to fail", True)
            self._last_status = status
            return Decision(DecisionKind.IGNORED, status)

        ladder = self._ladders[self._situation]
        next_index = self._step_index + 1
        if next_index >= len(ladder):
            return self._pause(
                self._situation,
                reason_code="RECOVERY_EXHAUSTED",
                message=f"Recovery ladder exhausted after {detail}",
                recoverable=True,
            )

        self._step_index = next_index
        self._current_step = ladder[next_index]
        status = self._make_status(
            "RECOVERY_ESCALATED",
            f"Escalating after {detail}; starting {self._current_step.name}",
            True,
        )
        self._last_status = status
        return Decision(DecisionKind.START_STEP, status, self._current_step)
```

## 逐行讲解 / What's happening

1. **第 122-129 行 / Lines 122-129 (`state`)**:
   - 中文: controller 保存当前状态、触发原因、step index 和最后一次状态消息。
   - English: The controller stores current state, triggering situation, step index, and last status.
2. **第 139-148 行 / Lines 139-148 (`safety first`)**:
   - 中文: 安全事件不进入自动恢复，直接 `_pause(... recoverable=False)`。
   - English: Safety events do not enter automatic recovery; they immediately `_pause(... recoverable=False)`.
3. **第 150-167 行 / Lines 150-167 (`ignore reentry`)**:
   - 中文: 已经在恢复中或 paused 时，新事件会被忽略，避免多个恢复流程互相打架。
   - English: While recovering or paused, new events are ignored so multiple recovery flows do not conflict.
4. **第 168-187 行 / Lines 168-187 (`start ladder`)**:
   - 中文: 可恢复事件查 ladder，从第一步开始并返回 `START_STEP` 决策。
   - English: Recoverable events look up a ladder, start at step zero, and return a `START_STEP` decision.
5. **第 201-225 行 / Lines 201-225 (`escalate`)**:
   - 中文: 失败后进入下一阶；用完 ladder 后暂停并上报 exhausted。
   - English: Failure advances to the next rung; after the ladder is exhausted, the controller pauses and reports it.

## 类比 / The analogy

这像电梯救援手册：门卡住先重试开门，再切换备用动作，还是失败就停梯等维修；火警则不能自动处理，必须立即停。

It is like an elevator rescue manual: if the door sticks, retry opening, then try a backup action, and if that fails stop for maintenance; a fire alarm stops everything immediately.

## 自己跑一遍 / Try it yourself

```python
class Recovery:
    def __init__(self):
        self.ladders = {"bumper": ["back_up", "turn", "clear_map"]}
        self.state = "idle"
        self.i = 0

    def trigger(self, event):
        if event in {"e_stop", "cliff"}:
            self.state = "paused"
            return ("paused", None)
        self.state = "recovering"
        self.i = 0
        return ("start", self.ladders[event][self.i])

    def failed(self):
        self.i += 1
        if self.i >= len(self.ladders["bumper"]):
            self.state = "paused"
            return ("exhausted", None)
        return ("escalate", self.ladders["bumper"][self.i])

r = Recovery()
print(r.trigger("bumper"))
print(r.failed())
print(r.failed())
print(r.trigger("e_stop"))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
('start', 'back_up')
('escalate', 'turn')
('escalate', 'clear_map')
('paused', None)
```

这个例子把“恢复动作”和“安全停机”分成两条不同控制路径。

The example separates recovery actions from safety shutdown into two different control paths.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **ROS Nav2 recovery behaviors** / **ROS Nav2 recovery behaviors**: 中文: 导航栈也常用清 costmap、后退、旋转等恢复序列。 / English: Navigation stacks also use recovery sequences such as clearing costmaps, backing up, and rotating.
- **Robot policy fallback** / **Robot policy fallback**: 中文: 学习策略外层通常还要有规则式 safety wrapper。 / English: Learned policies usually need a rule-based safety wrapper around them.

## 注意事项 / Caveats / when it breaks

- **ladder 需要实机标定** / **Ladders need real-robot tuning**: 中文: 后退速度、旋转时间和超时都要按底盘调参。 / English: Reverse speed, rotation duration, and timeouts must be tuned for the platform.
- **状态机不是规划器** / **A state machine is not a planner**: 中文: 它处理局部恢复，不替代全局路径规划。 / English: It handles local recovery; it does not replace global planning.

## 延伸阅读 / Further reading

- [makerspet/oomwoo source](https://github.com/makerspet/oomwoo/blob/b3ca46c3207a9ac11687feb5414e97d3db248147/contributions/recovery-safety/xbattlax/oomwoo_recovery_safety/oomwoo_recovery_safety/core.py#L122-L225)
