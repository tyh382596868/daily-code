---
date: 2026-09-14
topic: vla
source: vla
repo: Physical-Intelligence/openpi
file: packages/openpi-client/src/openpi_client/action_chunk_broker.py
permalink: https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/packages/openpi-client/src/openpi_client/action_chunk_broker.py#L10-L50
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, action-chunking, control-loop, inference-cache]
build_role: action-chunking (advanced variant, horizon buffering at the client boundary)
---

# openpi ActionChunkBroker：一次预测，逐拍执行 / openpi ActionChunkBroker: Predict Once, Execute One Tick at a Time

> **一句话 / In one line**: `ActionChunkBroker` 把 policy 输出的 action horizon 暂存在客户端，每个控制 tick 只切出一行，耗尽后才重新推理。 / `ActionChunkBroker` buffers a policy's action horizon at the client, slices one row per control tick, and infers again only when the chunk is exhausted.

## 为什么重要 / Why this matters

中文：策略推理通常比机器人控制 tick 慢得多。若每个 tick 都重新调用 VLA，算力和延迟都会把控制环拖住；若一次预测一整段动作，却没有一个清楚的消费层，动作 horizon 又会散落在各个调用方。`ActionChunkBroker` 把这件事收成一个小而明确的适配器：上游 policy 仍然返回 chunk，下游控制器仍然每次拿一个 action。

English: Policy inference is usually much slower than the robot control tick. Re-running a VLA on every tick wastes compute and adds latency, while exposing a full horizon to every caller spreads buffering logic across the system. `ActionChunkBroker` makes the boundary explicit: the policy returns a chunk, and the controller receives one action per call.

## 代码 / The code

`Physical-Intelligence/openpi` — [`packages/openpi-client/src/openpi_client/action_chunk_broker.py`](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/packages/openpi-client/src/openpi_client/action_chunk_broker.py#L10-L50)

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

1. **第 10-17 行 / Lines 10-17 (the wrapper contract)**:
   - 中文：broker 假定所有 action field 的第一维都是 chunk 维度，因此可以统一切片，而不用知道 action 是关节位置、末端位姿还是夹爪值。
   - English: The broker assumes the first dimension of every action field is the chunk dimension, so it can slice uniformly without knowing whether a field is joint position, end-effector pose, or gripper state.
2. **第 19-24 行 / Lines 19-24 (state)**:
   - 中文：`_last_results` 是缓存的整段结果，`_cur_step` 是当前消费位置；没有缓存时才需要调用内层 policy。
   - English: `_last_results` stores the full prediction and `_cur_step` marks the next row to consume. The inner policy is called only when the cache is empty.
3. **第 27-30 行 / Lines 27-30 (lazy inference)**:
   - 中文：第一次 `infer` 或 chunk 耗尽后，broker 把新的 observation 交给 policy，并把游标归零。
   - English: On the first call, or after exhaustion, the broker sends the new observation to the policy and resets the cursor.
4. **第 32-38 行 / Lines 32-38 (`slicer`)**:
   - 中文：numpy array 沿第一维切一行；非 array 的辅助输出原样保留。`tree.map_structure` 让嵌套字典也共享这条规则。
   - English: Numpy arrays are sliced along dimension zero, while non-array metadata is preserved. `tree.map_structure` applies the rule through nested structures.
5. **第 39-44 行 / Lines 39-44 (retire the chunk)**:
   - 中文：返回当前 action 后游标加一；到达 horizon 就清缓存，下一次调用自然触发新一轮推理。
   - English: After returning the current action, the cursor advances. Once the horizon is reached, the cache is cleared so the next call starts a new inference.
6. **第 46-50 行 / Lines 46-50 (`reset`)**:
   - 中文：reset 同时清理内层 policy 和 broker 自己的状态，避免换 episode 后继续执行上一段动作。
   - English: Reset clears both the wrapped policy and the broker state, preventing actions from the previous episode from leaking into the next one.

## 在 nanoVLA 中的位置 / Where this lives in your nanoVLA

中文：这是 `nanoVLA` 的 action-chunking 边界层，位于连续动作 head 和真实控制循环之间。动作 head 输入视觉、语言和状态，输出 `[H, action_dim]`；broker 把它缓存起来，控制器每拍消费 `[action_dim]`。如果省掉它，你要么每拍重新运行昂贵的 VLA，要么把 horizon 管理逻辑复制到每个机器人后端。生产实现还需要处理 observation 变化、chunk 不足、异步推理、时间戳、动作平滑，以及 policy 推理失败时的安全降级。

English: In a `nanoVLA`, this is the action-chunking boundary between the continuous action head and the real control loop. The head consumes vision, language, and state and returns `[H, action_dim]`; the broker stores that horizon while the controller consumes `[action_dim]` per tick. Without it, every robot backend must either rerun the expensive VLA or duplicate horizon management. Production code also needs observation freshness, partial chunks, asynchronous inference, timestamps, smoothing, and safe fallback on policy failure.

## 自己跑一遍 / Try it yourself

```python
class Broker:
    def __init__(self, policy, horizon):
        self.policy, self.horizon = policy, horizon
        self.cache, self.step = None, 0

    def infer(self, obs):
        if self.cache is None:
            self.cache, self.step = self.policy(obs), 0
        action = self.cache[self.step]
        self.step += 1
        if self.step == self.horizon:
            self.cache = None
        return action


calls = []
broker = Broker(lambda obs: calls.append(obs) or [[1], [2], [3]], 3)
print([broker.infer("obs") for _ in range(3)])
print("policy calls:", len(calls))
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
[[1], [2], [3]]
policy calls: 1
```

中文：三次控制调用只产生一次 policy call；第四次调用才会开始下一段 chunk。这是“昂贵推理”和“便宜执行”解耦的全部收益。

English: Three control calls produce only one policy call; the fourth call would start the next chunk. That is the whole payoff of separating expensive inference from cheap execution.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LeRobot action queue** / **LeRobot action queue**: queues consume predicted actions while inference runs at a slower cadence. / Queues consume predicted actions while inference runs at a slower cadence.
- **ACT temporal ensembling** / **ACT temporal ensembling**: overlapping chunks can be averaged before a control tick. / Overlapping chunks can be averaged before a control tick.
- **FastWAM action-only inference** / **FastWAM action-only inference**: cached video context supports repeated action denoising without recomputing the full video path. / Cached video context supports repeated action denoising without recomputing the full video path.

## 注意事项 / Caveats / when it breaks

- **horizon 必须和返回形状一致** / **The horizon must match the output shape**: 配错会越界或提前清缓存。 / A mismatch causes indexing errors or premature cache eviction.
- **新 observation 不会中途打断 chunk** / **New observations do not interrupt a chunk**: 需要显式设计 preemption，否则缓存动作可能已经过时。 / Add explicit preemption if cached actions can become stale.
- **reset 是安全边界** / **Reset is a safety boundary**: 新 episode、急停或机器人重连都应清理缓存。 / New episodes, emergency stops, and reconnects should clear the cache.

## 延伸阅读 / Further reading

- [openpi ActionChunkBroker](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/packages/openpi-client/src/openpi_client/action_chunk_broker.py)
- [openpi policy client](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479/packages/openpi-client/src/openpi_client)
