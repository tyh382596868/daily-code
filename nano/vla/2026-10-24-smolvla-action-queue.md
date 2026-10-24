---
date: 2026-10-24
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/policies/smolvla/modeling_smolvla.py
permalink: https://github.com/huggingface/lerobot/blob/d3793b7e36522321c1c455f0dc4826db3793dceb/src/lerobot/policies/smolvla/modeling_smolvla.py#L255-L291
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, action-chunking]
build_role: action-chunking
---

# SmolVLA 动作队列：一次预测，逐拍消费 / SmolVLA Action Queue: Predict Once, Consume One Control Tick at a Time

> **一句话 / In one line**: SmolVLA 只在队列为空时推理一个动作 chunk，再每个控制周期弹出一条动作。 / SmolVLA infers an action chunk only when its queue is empty, then pops one action per control tick.

## 为什么重要 / Why this matters

VLA 推理常比机器人控制慢。动作 chunk 把昂贵的模型调用和高频执行解耦，但它也要求明确的队列形状和补货条件。

VLA inference is often slower than robot control. Action chunks decouple costly model calls from high-rate execution, but require a precise queue shape and refill condition.

## 代码 / The code

`huggingface/lerobot` — [`modeling_smolvla.py`](https://github.com/huggingface/lerobot/blob/d3793b7e36522321c1c455f0dc4826db3793dceb/src/lerobot/policies/smolvla/modeling_smolvla.py#L255-L291)

```python
if self._check_get_actions_condition():
    actions = self._get_action_chunk(batch, noise=noise)
    # (batch, steps, dim) -> queue of (batch, dim)
    self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

return self._queues[ACTION].popleft()

def _check_get_actions_condition(self) -> bool:
    return len(self._queues[ACTION]) == 0
```

## 逐行讲解 / What's happening

模型输出的时间维原本在第二轴；`transpose(0, 1)` 把它移到前面，令 deque 的每个元素恰好是一拍、包含整个 batch。条件只看队列是否为空，因此一次预测会连续供应 `n_action_steps` 次控制。

The model’s time dimension starts in axis one; `transpose(0, 1)` moves it first so every deque entry is exactly one tick for the whole batch. The empty-queue condition means one prediction supplies `n_action_steps` control calls.

## 类比 / The analogy

像餐厅一次烤一盘面包，服务员每分钟取一片；烤箱只在盘子空了时再开，不会为每一片面包重新预热。

It is a restaurant baking one tray of bread and serving one slice per minute; the oven runs again only when the tray is empty.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

这是 `action-chunking` 的 cross-repo 变体。上游 action head 输出 `(B, horizon, action_dim)`，broker 将其转为逐拍动作；下游是 safety clamp 和机器人驱动。省掉它会让每个控制 tick 都重新跑大模型；生产版还需时间戳、过期 chunk 丢弃和观测变化时的重规划。

This is a cross-repo `action-chunking` variant. An action head produces `(B, horizon, action_dim)`; a broker turns it into tick-wise actions for safety clamping and robot drivers. Without it, every tick reruns the large model; production needs timestamps, stale-chunk rejection, and replanning on observation changes.

## 自己跑一遍 / Try it yourself

```python
from collections import deque
chunk, q = [[1, 2], [3, 4], [5, 6]], deque()
for _ in range(5):
    if not q: q.extend(chunk)
    print(q.popleft())
```

运行 / Run with: `python try.py`

预期输出 / Expected output: 前三拍消费一个 chunk，第四拍重新补货。 / The first three ticks consume one chunk; tick four refills it.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Isaac GR00T action chunks** / **Isaac GR00T action chunks**: 也把预测 horizon 与控制频率分离。 / It likewise separates prediction horizon from control frequency.

## 注意事项 / Caveats / when it breaks

- **不能只检查“有无动作”** / **Do not only check availability**: 真实机器人还要检查 chunk 是否因延迟或新观测而过期。 / Real robots must also reject chunks made stale by delay or new observations.

## 延伸阅读 / Further reading

- [SmolVLA action selection](https://github.com/huggingface/lerobot/blob/d3793b7e36522321c1c455f0dc4826db3793dceb/src/lerobot/policies/smolvla/modeling_smolvla.py#L255-L291)
