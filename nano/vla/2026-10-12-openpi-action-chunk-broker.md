---
date: 2026-10-12
topic: vla
source: vla
repo: Physical-Intelligence/openpi
file: packages/openpi-client/src/openpi_client/action_chunk_broker.py
permalink: https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/packages/openpi-client/src/openpi_client/action_chunk_broker.py#L10-L50
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, action-chunking, inference-loop]
build_role: action-chunking advanced variant, client-side chunk broker
---

# openpi ActionChunkBroker：一次推理，多步消费 / openpi ActionChunkBroker: Infer Once, Consume Many Steps

> **一句话 / In one line**: 它把一次 policy 输出的 action horizon 缓存起来，每次 `infer()` 只吐当前步，直到 chunk 用完才重新调用模型。 / It caches one policy output horizon and returns only the current step on each `infer()`, calling the model again only after the chunk is exhausted.

## 为什么重要 / Why this matters

VLA policy 通常一次预测未来多步动作，但机器人控制环是一拍一拍执行的。如果每一拍都重跑大模型，延迟会很高；如果一次性把整段动作交给执行器，又难以及时插入新观测。`ActionChunkBroker` 是中间层：模型按 chunk 推理，控制器按单步消费。

A VLA policy often predicts a future action horizon, while the robot control loop executes one tick at a time. Re-running the large model every tick is expensive; handing the whole chunk directly to the actuator makes feedback harder. `ActionChunkBroker` sits between them: the model infers by chunk, the controller consumes by step.

## 代码 / The code

`Physical-Intelligence/openpi` - [`packages/openpi-client/src/openpi_client/action_chunk_broker.py`](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/packages/openpi-client/src/openpi_client/action_chunk_broker.py#L10-L50)

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

1. **第 10-17 行 / Lines 10-17 (class contract)**:
   - 中文: docstring 明确假设所有 action 字段的第一维都是 chunk size。这个约定让 broker 能用同一个索引切所有数组。
   - English: The docstring states that the first dimension of every action field is the chunk size. That contract lets the broker slice all arrays with the same index.

2. **第 19-24 行 / Lines 19-24 (state)**:
   - 中文: broker 保存内部 policy、horizon、当前步计数和最近一次推理结果。它自己不预测动作，只管理缓存生命周期。
   - English: The broker stores the inner policy, horizon, current step, and last inference result. It does not predict actions; it manages cache lifetime.

3. **第 27-30 行 / Lines 27-30 (lazy policy call)**:
   - 中文: 只有 `_last_results` 为空时才调用内层 policy。第一拍触发推理，后面几拍直接切缓存。
   - English: The inner policy is called only when `_last_results` is empty. The first tick triggers inference; following ticks slice the cache.

4. **第 32-38 行 / Lines 32-38 (`tree.map_structure`)**:
   - 中文: 结果可能是嵌套 dict/list。`tree.map_structure` 对每个叶子做同样切片，非数组字段保持不变。
   - English: Results may be nested dicts or lists. `tree.map_structure` applies the same slicing to every leaf, while non-array fields pass through unchanged.

5. **第 41-43 行 / Lines 41-43 (cache expiry)**:
   - 中文: 消费到 horizon 后清空缓存。下一次 `infer()` 会带着最新 observation 重新跑模型。
   - English: Once the horizon is consumed, the cache is cleared. The next `infer()` will run the model again with the latest observation.

## 类比 / The analogy

这像一次煮好一锅米饭，但每顿只盛一碗。锅里还有饭时不用重新开火；吃完了再煮下一锅。broker 就是那只饭勺，按步把 chunk 盛出来。

It is like cooking a pot of rice and serving one bowl per meal. You do not restart the stove while rice remains; you cook again after the pot is empty. The broker is the serving spoon for action chunks.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

在 nanoVLA 里，这是 `action-chunking` 组件的推理侧适配器。上游是 VLA policy 的 `predict_action_chunk(obs) -> {"actions": [H, A]}`，下游是机器人控制循环的 `step(action)`。如果省掉它，你要么每拍重跑模型，要么把整段动作一次性塞给执行端，都会让延迟或反馈变差。生产级版本还要加入时间戳、紧急 reset、观测变化时提前刷新 chunk、以及多机器人字段的 shape 校验。

In a nanoVLA, this is the inference-side adapter for the `action-chunking` component. Upstream is the VLA policy, such as `predict_action_chunk(obs) -> {"actions": [H, A]}`. Downstream is the robot loop, such as `step(action)`. Without it, you either re-run the model every tick or dump the full chunk into the actuator, hurting latency or feedback. A production version should add timestamps, emergency reset, early refresh when observations drift, and shape validation for multi-robot fields.

## 自己跑一遍 / Try it yourself

```python
class ToyPolicy:
    def __init__(self):
        self.calls = 0
    def infer(self, obs):
        self.calls += 1
        return {"actions": [10, 11, 12], "calls": self.calls}
    def reset(self):
        self.calls = 0

class Broker:
    def __init__(self, policy, horizon):
        self.policy, self.horizon = policy, horizon
        self.cur, self.last = 0, None
    def infer(self, obs):
        if self.last is None:
            self.last, self.cur = self.policy.infer(obs), 0
        out = {"actions": self.last["actions"][self.cur], "calls": self.last["calls"]}
        self.cur += 1
        if self.cur >= self.horizon:
            self.last = None
        return out

b = Broker(ToyPolicy(), 3)
print([b.infer({}) for _ in range(5)])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[{'actions': 10, 'calls': 1}, {'actions': 11, 'calls': 1}, {'actions': 12, 'calls': 1}, {'actions': 10, 'calls': 2}, {'actions': 11, 'calls': 2}]
```

前三步来自同一次模型调用，第四步才刷新 chunk。你可以从 `calls` 看出推理频率被降下来了。

The first three steps come from one model call; the fourth step refreshes the chunk. The `calls` field makes the lower inference frequency visible.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LeRobot action queues** / **LeRobot action queues**: LeRobot 的在线控制也常把预测 chunk 放进队列，再按控制频率弹出。 / LeRobot online control often puts predicted chunks into a queue and pops them at control frequency.
- **ACT temporal ensembling** / **ACT temporal ensembling**: ACT 不只是缓存一个 chunk，还会把重叠 chunk 做加权平均。 / ACT goes further by averaging overlapping chunks, not just caching one chunk.

## 注意事项 / Caveats / when it breaks

- **horizon 必须匹配数组第一维** / **Horizon must match the first array dimension**: 如果 policy 返回的 chunk 长度比 `action_horizon` 短，切片会越界。 / If the returned chunk is shorter than `action_horizon`, slicing will go out of range.
- **观测变化不会自动打断缓存** / **Observation changes do not interrupt the cache**: 当前实现只在 chunk 用完后刷新。真实机器人可能需要根据误差提前 reset。 / This implementation refreshes only after the chunk is consumed. Real robots may need early reset based on tracking error.

## 延伸阅读 / Further reading

- [Source permalink](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/packages/openpi-client/src/openpi_client/action_chunk_broker.py#L10-L50)
- [openpi repository](https://github.com/Physical-Intelligence/openpi)

