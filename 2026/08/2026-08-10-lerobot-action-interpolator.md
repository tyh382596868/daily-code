---
date: 2026-08-10
topic: robotics
source: tracked
repo: huggingface/lerobot
file: src/lerobot/utils/action_interpolator.py
permalink: https://github.com/huggingface/lerobot/blob/22bd7a2f489b367d8df42de803b1e8c4ca63a3f9/src/lerobot/utils/action_interpolator.py#L24-L116
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, robotics, action-smoothing]
---

# LeRobot action interpolation：把低频动作补成平滑控制流 / LeRobot Action Interpolation: Turn Sparse Actions into Smooth Control

> **一句话 / In one line**: policy 只产一帧动作时，`ActionInterpolator` 用上一帧和当前帧线性插值，给机器人更高频、更平滑的命令。 / When the policy emits sparse actions, `ActionInterpolator` linearly fills the gap between the previous and current action for smoother high-rate control.

## 为什么重要 / Why this matters

很多机器人策略按相机帧率或 action chunk 输出动作，但电机控制环更喜欢稳定的小步更新。直接把同一个动作保持几十毫秒会产生顿挫；插值层把策略频率和控制频率解耦，让模型不用承担所有低层平滑细节。

Many robot policies act at camera or chunk rate, while motor loops prefer small steady updates. Holding one action for too long can make motion jerky; an interpolation layer decouples policy frequency from control frequency.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/utils/action_interpolator.py`](https://github.com/huggingface/lerobot/blob/22bd7a2f489b367d8df42de803b1e8c4ca63a3f9/src/lerobot/utils/action_interpolator.py#L24-L116)

```python
class ActionInterpolator:
    """Interpolates between consecutive actions for smoother control."""

    def __init__(self, multiplier: int = 1):
        if multiplier < 1:
            raise ValueError(f"multiplier must be >= 1, got {multiplier}")
        self.multiplier = multiplier
        self._prev: Tensor | None = None
        self._buffer: list[Tensor] = []
        self._idx = 0

    @property
    def enabled(self) -> bool:
        return self.multiplier > 1

    def reset(self):
        self._prev = None
        self._buffer = []
        self._idx = 0

    def needs_new_action(self) -> bool:
        return self._idx >= len(self._buffer)

    def add(self, action: Tensor) -> None:
        if self.multiplier > 1 and self._prev is not None:
            self._buffer = []
            for i in range(1, self.multiplier + 1):
                t = i / self.multiplier
                interp = self._prev + t * (action - self._prev)
                self._buffer.append(interp)
        else:
            self._buffer = [action.clone()]
        self._prev = action.clone()
        self._idx = 0

    def get(self) -> Tensor | None:
        if self._idx >= len(self._buffer):
            return None
        action = self._buffer[self._idx]
        self._idx += 1
        return action

    def get_control_interval(self, fps: float) -> float:
        return 1.0 / (fps * self.multiplier)
```

## 逐行讲解 / What's happening

1. **第 49-60 行 / Lines 49-60 (state)**:
   - 中文: `multiplier` 决定每个 policy 动作被展开成几个控制动作，`_prev` 记住上一帧，`_buffer` 存放插值后的短序列。
   - English: `multiplier` controls how many motor commands one policy action becomes; `_prev` stores the previous action and `_buffer` stores the interpolated sequence.
2. **第 73-75 行 / Lines 73-75 (back-pressure)**:
   - 中文: `needs_new_action()` 让控制环只在 buffer 消费完时去队列里取新动作。
   - English: `needs_new_action()` makes the control loop request a new policy action only after the buffered commands are consumed.
3. **第 83-88 行 / Lines 83-88 (linear blend)**:
   - 中文: `t = i / multiplier` 从上一动作平滑走到当前动作，最后一步正好等于当前动作。
   - English: `t = i / multiplier` walks from the previous action to the current one, ending exactly at the current action.
4. **第 95-105 行 / Lines 95-105 (cursor)**:
   - 中文: `get()` 像读队列一样推进 `_idx`，buffer 空时返回 `None`。
   - English: `get()` advances `_idx` like a small queue reader and returns `None` when exhausted.

## 类比 / The analogy

像电梯从 1 楼到 10 楼，不是瞬移到 10 楼，而是按楼层逐步经过；乘客感受到的是连续运动。

It is like an elevator going from floor 1 to floor 10: it does not teleport, it passes intermediate floors so the ride feels continuous.

## 自己跑一遍 / Try it yourself

```python
prev = 0.0
new = 9.0
multiplier = 3
buffer = []
for i in range(1, multiplier + 1):
    t = i / multiplier
    buffer.append(prev + t * (new - prev))
print(buffer)
print("interval", 1.0 / (30 * multiplier))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[3.0, 6.0, 9.0]
interval 0.011111111111111112
```

最关键的是最后一个插值点就是新动作本身，因此插值不会改变 policy 的最终目标，只改变到达方式。

The important part is that the last interpolated command equals the new policy action, so interpolation changes the path, not the target.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Trajectory resampling** / **Trajectory resampling**: 规划器给稀疏 waypoint，控制器按固定频率重采样。 / A planner emits sparse waypoints and the controller resamples them at a fixed rate.
- **Audio crossfade** / **Audio crossfade**: 两段信号之间用线性或曲线混合，避免突然跳变。 / Two signals are blended linearly or with a curve to avoid a sudden jump.

## 注意事项 / Caveats / when it breaks

- **线性插值不懂动力学** / **Linear interpolation is not dynamics-aware**: 关节限位、速度限位和碰撞仍然要由低层控制器兜住。 / Joint limits, velocity limits, and collisions still need lower-level checks.
- **第一帧无法插值** / **The first frame has no history**: 没有 `_prev` 时只能直接输出当前动作。 / Without `_prev`, the first action is emitted directly.

## 延伸阅读 / Further reading

- [LeRobot `ActionInterpolator`](https://github.com/huggingface/lerobot/blob/22bd7a2f489b367d8df42de803b1e8c4ca63a3f9/src/lerobot/utils/action_interpolator.py#L24-L116)
