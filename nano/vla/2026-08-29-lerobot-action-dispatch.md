---
date: 2026-08-29
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/rollout/strategies/core.py
permalink: https://github.com/huggingface/lerobot/blob/4aaff99be4a1d81568c08c8f0296b41b40c99ec4/src/lerobot/rollout/strategies/core.py#L318-L331
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, vla, inference-loop, action-dispatch]
build_role: inference-loop advanced variant
---

# LeRobot action dispatch：动作队列最后一公里 / LeRobot Action Dispatch: The Last Mile of the Action Queue

> **一句话 / In one line**: LeRobot 的 rollout 策略从 action queue 取出下一拍动作，经过插值和 processor 后再发给机器人。 / LeRobot's rollout strategy takes the next action from the action queue, passes it through interpolation and processors, then sends it to the robot.

## 为什么重要 / Why this matters

VLA 推理循环通常一次预测一段 action chunk，但真实机器人每个控制 tick 只吃下一拍。这个分发层把“大块预测”变成“小步执行”，也是插值、限幅和坐标转换的最后入口。

A VLA inference loop often predicts an action chunk, while the real robot consumes one control tick at a time. This dispatch layer turns a large prediction into small executable steps and is the final hook for interpolation, clipping, and coordinate transforms.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/rollout/strategies/core.py`](https://github.com/huggingface/lerobot/blob/4aaff99be4a1d81568c08c8f0296b41b40c99ec4/src/lerobot/rollout/strategies/core.py#L318-L331)

```python
def send_next_action(self, env: Robot, policy: PreTrainedPolicy, action_features: dict[str, PolicyFeature]):
    """Get and send next action using the action queue."""
    if len(self.action_queue) == 0:
        return

    action_to_send = self.action_queue.popleft()
    if self.action_interpolator:
        action_to_send = self.action_interpolator(action_to_send)

    if self.action_processor is not None:
        action_to_send = self.action_processor(
            observation=None,
            action=action_to_send,
            robot=env,
            policy=policy,
            action_features=action_features,
            is_safe=True,
        )

    env.send_action(action_to_send)
```

## 逐行讲解 / What's happening

1. **队列为空直接返回 / Empty queue returns**:
   - 中文: 如果 policy 还没产出动作，控制层不会发送旧动作假装有效。
   - English: If the policy has not produced an action, the control layer does not pretend an old action is valid.
2. **`popleft()` / `popleft()`**:
   - 中文: 每个 tick 只取 action chunk 的下一步，保持执行顺序先进先出。
   - English: Each tick consumes only the next step of the action chunk, preserving FIFO execution order.
3. **插值器 / Interpolator**:
   - 中文: 稀疏 policy 动作可以在真正发出前被补成更平滑的控制命令。
   - English: Sparse policy actions can be smoothed into denser commands before being sent.
4. **processor / Processor**:
   - 中文: 最后执行安全、坐标或特征相关的 action 转换，再调用 `env.send_action`。
   - English: The last stage applies safety, coordinate, or feature-aware transforms before `env.send_action`.

## 类比 / The analogy

这像地铁调度中心：计划表一次给出一串车次，但站台屏幕每分钟只放下一班；必要时还能临时调整发车间隔。

It is like a subway control room: a schedule contains many departures, but the platform display releases one next train at a time and can adjust intervals if needed.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

在 nanoVLA 里，这属于 `inference-loop` 的执行侧高级变体。上游是 policy/action head 生成的 action chunk，下游是机器人 driver。没有这层，你只能把整段动作一次性丢给机器人，难以做安全检查、实时取消、插值或与观测闭环同步。

In a nanoVLA, this is an execution-side advanced variant of `inference-loop`. Upstream is the action chunk produced by the policy or action head; downstream is the robot driver. Without this layer, you would dump a whole action sequence into the robot and lose clean hooks for safety checks, cancellation, interpolation, and observation-synchronized control.

## 自己跑一遍 / Try it yourself

```python
from collections import deque

def interpolate(a):
    return round(a * 0.5, 2)

def process(a):
    return max(-1.0, min(1.0, a))

queue = deque([3.0, 0.6])
sent = []
while queue:
    action = queue.popleft()
    action = interpolate(action)
    action = process(action)
    sent.append(action)

print(sent)
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
[1.0, 0.3]
```

第一步被插值后仍超限，所以 processor 把它夹到安全范围。

The first step is still out of range after interpolation, so the processor clips it into the safe interval.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LeRobot ACT temporal ensemble** / **LeRobot ACT temporal ensemble**: 多个 chunk 的同一 tick 可以先融合再执行。 / Multiple chunks for the same tick can be fused before execution.
- **openpi runtime loop** / **openpi runtime loop**: 每拍做 observe、infer、act、log。 / Each tick observes, infers, acts, and logs.
- **IsaacLab ActionManager** / **IsaacLab ActionManager**: 扁平 action 先分发给各执行器 term。 / A flat action is dispatched to actuator-specific terms.

## 注意事项 / Caveats / when it breaks

- **空队列不是成功** / **Empty queue is not success**: 需要上游及时补 action，否则控制会停顿。 / Upstream must refill actions on time or control stalls.
- **插值不能破坏语义** / **Interpolation must preserve semantics**: 离散 gripper 命令不一定能线性插值。 / Discrete gripper commands may not support linear interpolation.
- **processor 顺序重要** / **Processor order matters**: 归一化、限幅和相对坐标转换换顺序会改变结果。 / Normalization, clipping, and relative transforms can change results if reordered.

## 延伸阅读 / Further reading

- LeRobot rollout strategy core: https://github.com/huggingface/lerobot/blob/main/src/lerobot/rollout/strategies/core.py
- LeRobot policies: https://huggingface.co/docs/lerobot/
