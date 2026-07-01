---
date: 2026-07-01
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/async_inference/robot_client.py
permalink: https://github.com/huggingface/lerobot/blob/8414188db0b178b947985a7a9a91314708837315/src/lerobot/async_inference/robot_client.py#L224-L267
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, inference-loop, action-chunking]
build_role: inference-loop action buffer for streaming robot control
---

# LeRobot action queue：把重叠 action chunk 合成连续控制流 / LeRobot Action Queue: Merge Overlapping Action Chunks into a Continuous Control Stream

> **一句话 / In one line**: 新的 action chunk 不是简单追加，而是按 timestep 跳过过期动作、补新增动作、合并重叠动作。 / A new action chunk is not blindly appended; it skips stale actions, adds new ones, and merges overlapping timesteps.

## 为什么重要 / Why this matters

流式 VLA 推理会不断收到未来几十步动作。新 chunk 可能和队列里的旧 chunk 重叠，也可能已经落后于机器人执行进度。LeRobot 用 timestep 做主键：过期的丢掉，缺失的加入，重叠的交给 `aggregate_fn`，最后一次性替换队列。

Streaming VLA inference receives future actions repeatedly. A new chunk may overlap the old queue or arrive behind the robot's executed timestep. LeRobot uses timestep as the key: stale actions are dropped, missing ones are inserted, overlapping ones go through `aggregate_fn`, then the queue is replaced in one step.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/async_inference/robot_client.py`](https://github.com/huggingface/lerobot/blob/8414188db0b178b947985a7a9a91314708837315/src/lerobot/async_inference/robot_client.py#L224-L267)

```python
    def _aggregate_action_queues(
        self,
        incoming_actions: list[TimedAction],
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        """Finds the same timestep actions in the queue and aggregates them using the aggregate_fn"""
        if aggregate_fn is None:
            # default aggregate function: take the latest action
            def aggregate_fn(x1, x2):
                return x2

        future_action_queue = Queue()
        with self.action_queue_lock:
            internal_queue = self.action_queue.queue

        current_action_queue = {action.get_timestep(): action.get_action() for action in internal_queue}

        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action

            # New action is older than the latest action in the queue, skip it
            if new_action.get_timestep() <= latest_action:
                continue

            # If the new action's timestep is not in the current action queue, add it directly
            elif new_action.get_timestep() not in current_action_queue:
                future_action_queue.put(new_action)
                continue

            # If the new action's timestep is in the current action queue, aggregate it
            # TODO: There is probably a way to do this with broadcasting of the two action tensors
            future_action_queue.put(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=new_action.get_timestep(),
                    action=aggregate_fn(
                        current_action_queue[new_action.get_timestep()], new_action.get_action()
                    ),
                )
            )

        with self.action_queue_lock:
            self.action_queue = future_action_queue
```

## 逐行讲解 / What's happening

1. **第 230-234 行 / Lines 230-234**: 中文: 默认聚合函数选择最新动作，所以没有配置时系统偏向新预测。 / English: The default aggregation function chooses the newest action, so the system favors fresh predictions when no custom policy is configured.
2. **第 235-240 行 / Lines 235-240**: 中文: 先快照当前队列，把 timestep 映射到 action，后面可以 O(1) 判断是否重叠。 / English: It snapshots the current queue into a timestep-to-action map so overlap checks are O(1).
3. **第 241-264 行 / Lines 241-264**: 中文: 每个新动作先跟 `latest_action` 比较，过期则跳过；不重叠则加入；重叠则构造新的 `TimedAction`。 / English: Each incoming action is compared with `latest_action`; stale ones are skipped, non-overlapping ones are added, and overlapping ones create a new `TimedAction`.
4. **第 266-267 行 / Lines 266-267**: 中文: 最后在锁内替换整个队列，避免控制线程看到半更新状态。 / English: Finally the whole queue is swapped under the lock so the control thread never sees a half-updated state.

## 类比 / The analogy

像地铁时刻表滚动更新：已经发车的班次不能改；新班次直接贴上；同一时间的预测班次要按规则选择新版或平均版。

It is like a live subway timetable: departed trains cannot be edited; new departures are added; predictions for the same time slot are replaced or blended by policy.


## 在 nanoVLA 中的位置 / Where this lives in your nanoVLA

中文: 这属于 `inference-loop` 的 client 侧 action buffer。上游是 policy server 返回的 action chunk，下游是机器人 `send_action`。如果没有这层队列，网络抖动或推理延迟会直接让控制环断粮；生产级实现还要补时间同步、超时策略和安全停止。

English: This is the client-side action buffer in the `inference-loop` component. Upstream is the action chunk returned by the policy server; downstream is the robot `send_action`. Without this queue, network jitter or inference latency starves the control loop. A production version also needs clock sync, timeout policy, and safe stopping.


## 自己跑一遍 / Try it yourself

```python
from queue import Queue
old = {3: 0.3, 4: 0.4}
incoming = [(2, 9.9), (4, 0.8), (5, 0.5)]
latest = 2
q = Queue()
for t, a in incoming:
    if t <= latest:
        continue
    q.put((t, a if t not in old else 0.5 * old[t] + 0.5 * a))
while not q.empty():
    print(q.get())
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
(4, 0.6000000000000001)
(5, 0.5)
```

中文: 这个小例子保留了源码里的关键控制流，但把依赖压到最低，便于你直接观察形状、索引或状态变化。

English: The miniature keeps the original control-flow idea while stripping dependencies down so the shape, index, or state change is visible.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Real-Time Chunking** / **Real-Time Chunking**: 中文: RTC 也把不同时间预测的 chunk 对齐后融合。 / English: RTC also aligns and fuses chunks predicted at different times.
- **MPC warm-start buffers** / **MPC warm-start buffers**: 中文: MPC 常保留未来控制序列，并用新规划滚动覆盖。 / English: MPC often keeps a future control sequence and rolls it forward with new plans.

## 注意事项 / Caveats / when it breaks

- **时钟漂移 / Clock drift**: 中文: server 和 robot 对 timestep 理解不一致会让队列错位。 / English: If server and robot disagree on timestep, the queue misaligns.
- **聚合函数要守物理约束 / Aggregation must respect physics**: 中文: 平均两个夹爪命令可能产生非法中间动作。 / English: Averaging two gripper commands may create an invalid intermediate action.

## 延伸阅读 / Further reading

- Source permalink above.
- Project repository linked from the frontmatter.
