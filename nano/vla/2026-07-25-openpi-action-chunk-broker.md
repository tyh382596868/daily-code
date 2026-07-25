---
date: 2026-07-25
topic: vla
source: vla
repo: Physical-Intelligence/openpi
file: packages/openpi-client/src/openpi_client/action_chunk_broker.py
permalink: https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/packages/openpi-client/src/openpi_client/action_chunk_broker.py#L10-L50
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, vla, action-chunking]
build_role: action-chunking advanced variant, client-side streaming consumption
---

# openpi ActionChunkBroker：一次推理，多步消费 / openpi ActionChunkBroker: Infer Once, Consume Many Steps

> **一句话 / In one line**: 策略一次吐出 action chunk, broker 每次只交出当前 step, 用完才重新推理。 / The policy emits an action chunk once, and the broker returns one step at a time until the chunk is exhausted.

## 为什么重要 / Why this matters

很多 VLA 模型的输出不是单个动作, 而是一段 horizon。控制循环却通常一帧只需要一个动作。`ActionChunkBroker` 把这两个节奏隔开: 模型慢慢推理一整段, 机器人实时消费其中一步。

Many VLA models output a whole horizon, while a control loop usually needs one action per tick. `ActionChunkBroker` separates those rhythms: the model predicts a chunk, the robot consumes one step at a time.

## 代码 / The code

`Physical-Intelligence/openpi` -- [`packages/openpi-client/src/openpi_client/action_chunk_broker.py`](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/packages/openpi-client/src/openpi_client/action_chunk_broker.py#L10-L50)

```python
class ActionChunkBroker(_base_policy.BasePolicy):
    """Wraps a policy to return action chunks one-at-a-time.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is only made when the current
    list of chunks is exhausted.
    """

    def __init__(self, policy: _base_policy.BasePolicy, action_horizon: int):
        self._policy = policy
        self._action_horizon = action_horizon
        self._cur_step: int = 0

        self._last_results: Dict[str, np.ndarray] | None = None

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        if self._last_results is None:
            self._last_results = self._policy.infer(obs)
            self._cur_step = 0

        def slicer(x):
            if isinstance(x, np.ndarray):
                return x[self._cur_step, ...]
            else:
                return x

        results = tree.map_structure(slicer, self._last_results)
        self._cur_step += 1

        if self._cur_step >= self._action_horizon:
            self._last_results = None

        return results

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0
```

## 逐行讲解 / What's happening

1. **第 19-24 行 / Lines 19-24 (`缓存状态`)**:
   - 中文: 保存内部 policy、horizon、当前 step 和最近一次推理结果。
   - English: It stores the wrapped policy, horizon, current step, and latest inference result.
2. **第 28-30 行 / Lines 28-30 (`懒推理`)**:
   - 中文: 只有 `_last_results` 为空时才调用内部 policy。
   - English: It calls the inner policy only when there is no cached chunk.
3. **第 32-38 行 / Lines 32-38 (`tree map 切片`)**:
   - 中文: 对结果树里的每个 numpy 数组取当前 step, 非数组元数据原样保留。
   - English: It slices every NumPy array in the result tree at the current step and preserves metadata.
4. **第 39-43 行 / Lines 39-43 (`chunk 用完清空`)**:
   - 中文: 步数达到 horizon 后清空缓存, 下次 infer 才会重新跑模型。
   - English: When the horizon is exhausted, it clears the cache so the next call triggers a new inference.
5. **第 47-50 行 / Lines 47-50 (`reset`)**:
   - 中文: 外部 reset 要同时清内部 policy 和 broker 缓存。
   - English: External reset clears both the inner policy and the broker cache.

## 类比 / The analogy

这像电饭煲一次煮一锅饭, 但你每顿只盛一碗。锅里还有饭时不重新煮, 吃完一锅再开始下一锅。

It is like a rice cooker making a whole pot while you serve one bowl per meal. You do not cook again while rice remains; you start a new pot only after it is gone.

## 在 nanoVLA 中的位置 / Where this lives in your nanoVLA

在 nanoVLA 里, 这个组件放在 `PolicyClient` 或 `Controller` 层, 介于模型 forward 和机器人 low-level actuator 之间。上游是 `action-head-continuous` 或 `action-tokenizer` 产生的 horizon, 下游是每帧控制命令。省掉它, 你要么每帧都做昂贵推理, 要么一次把多步动作错误地发给机器人。生产级实现还要处理剩余 chunk 融合、latency compensation 和 emergency reset。

In a nanoVLA, this belongs in the `PolicyClient` or `Controller` layer between model forward and low-level actuators. Upstream is a horizon from the continuous action head or action tokenizer; downstream is one command per control tick. Without it, you either run expensive inference every tick or send a whole chunk incorrectly. Production versions add chunk blending, latency compensation, and emergency reset handling.

## 自己跑一遍 / Try it yourself

```python
class Policy:
    def __init__(self): self.calls = 0
    def infer(self, obs):
        self.calls += 1
        return {"action": [[1, 0], [2, 0], [3, 0]]}
    def reset(self): self.calls = 0

class Broker:
    def __init__(self, policy, horizon):
        self.policy, self.horizon, self.i, self.cache = policy, horizon, 0, None
    def infer(self, obs):
        if self.cache is None:
            self.cache, self.i = self.policy.infer(obs), 0
        out = {k: (v[self.i] if isinstance(v, list) else v) for k, v in self.cache.items()}
        self.i += 1
        if self.i >= self.horizon: self.cache = None
        return out

p = Policy(); b = Broker(p, 3)
print([b.infer({})["action"] for _ in range(4)], p.calls)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
[[1, 0], [2, 0], [3, 0], [1, 0]] 2
```

第四次调用重新触发一次 policy, 因为前三步已经耗尽了缓存 chunk。

The fourth call triggers the policy again because the first three steps exhausted the cached chunk.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LeRobot action queue**: 队列式 policy 也把 chunk 展开成逐步动作。 / Queue-based policies also unroll chunks into stepwise actions.
- **Real-Time Chunking**: RTC 在 chunk 之间额外融合重叠区, 解决接缝问题。 / RTC additionally blends overlapping chunks to smooth boundaries.

## 注意事项 / Caveats / when it breaks

- **horizon 必须匹配输出**: 代码假设数组第 0 维就是 chunk size。 / The code assumes axis 0 is the chunk dimension.
- **异常时要 reset**: 机器人环境 reset 后旧 chunk 不能继续执行。 / After environment reset, stale chunks must not continue executing.

## 延伸阅读 / Further reading

- openpi client package: https://github.com/Physical-Intelligence/openpi
- LeRobot policies: https://github.com/huggingface/lerobot
