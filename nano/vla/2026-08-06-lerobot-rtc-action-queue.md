---
date: 2026-08-06
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/policies/rtc/action_queue.py
permalink: https://github.com/huggingface/lerobot/blob/64b23178d5348609c266250d3e1f511eba4c33ff/src/lerobot/policies/rtc/action_queue.py#L35-L101
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, action-chunking]
build_role: action-chunking advanced variant, thread-safe real-time action buffer
---

# LeRobot RTC ActionQueue：动作 chunk 要线程安全地消费 / LeRobot RTC ActionQueue: Consume Action Chunks Thread-Safely

> **一句话 / In one line**: 实时 VLA 推理慢于控制频率时，action queue 是把“整段预测”拆成“每 tick 一个动作”的边界。 / When VLA inference is slower than the control loop, an action queue turns a predicted chunk into one action per tick.

## 为什么重要 / Why this matters

VLA policy 往往一次预测多个未来动作，但机器人控制线程每几十毫秒只需要一个动作。队列必须能在推理线程写入、控制线程读取时保持一致，并且不能把内部 tensor 暴露给外部修改。

A VLA policy often predicts several future actions at once, while the robot control loop consumes one action every few milliseconds. The queue must stay consistent across inference writes and control reads, and it must not expose mutable internal tensors.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/rtc/action_queue.py`](https://github.com/huggingface/lerobot/blob/64b23178d5348609c266250d3e1f511eba4c33ff/src/lerobot/policies/rtc/action_queue.py#L35-L101)

```python
class ActionQueue:
    """Thread-safe queue for managing action chunks in real-time control.

    This queue handles two types of action sequences:
    - Original actions: Used for RTC to compute leftovers from previous chunks
    - Processed actions: Post-processed actions ready for robot execution

    The queue operates in two modes:
    1. RTC-enabled: Replaces the entire queue with new actions, accounting for inference delay
    2. RTC-disabled: Appends new actions to the queue, maintaining continuity
    """

    def __init__(self, cfg: RTCConfig):
        self.queue = None  # Processed actions for robot rollout
        self.original_queue = None  # Original actions for RTC
        self.lock = Lock()
        self.last_index = 0
        self.cfg = cfg

    def get(self) -> Tensor | None:
        """Get the next action from the queue."""
        with self.lock:
            if self.queue is None or self.last_index >= len(self.queue):
                return None

            action = self.queue[self.last_index]
            self.last_index += 1
            return action.clone()

    def clear(self) -> None:
        """Clear queued actions and reset consumption index."""
        with self.lock:
            self.queue = None
            self.original_queue = None
            self.last_index = 0

    def qsize(self) -> int:
        """Get the number of remaining actions in the queue."""
        with self.lock:
            if self.queue is None:
                return 0
            return len(self.queue) - self.last_index

    def empty(self) -> bool:
        """Check if the queue is empty."""
```

## 逐行讲解 / What's happening

1. **第 38-45 行 / Lines 38-45 (two queues, two modes)**:
   - 中文: processed actions 给机器人执行，original actions 保留给 RTC 计算旧 chunk 的剩余约束。
   - English: Processed actions are executed by the robot, while original actions are kept so RTC can reason about leftovers from the previous chunk.
2. **第 61-65 行 / Lines 61-65 (state)**:
   - 中文: `last_index` 是消费游标；`Lock` 明确说明这里会被多个线程碰到。
   - English: `last_index` is the consumption cursor; `Lock` makes the cross-thread boundary explicit.
3. **第 74-80 行 / Lines 74-80 (get)**:
   - 中文: 取动作时先锁住，读当前动作，游标前进，最后 `clone()` 防止调用方改坏队列内部状态。
   - English: Reading locks the queue, fetches the current action, advances the cursor, and returns `clone()` so callers cannot mutate internal state.
4. **第 82-98 行 / Lines 82-98 (clear and size)**:
   - 中文: 清空和查询剩余长度也在锁里完成，避免控制线程看到半更新状态。
   - English: Clearing and sizing also happen under the lock so the control thread never sees a half-updated state.

## 类比 / The analogy

它像寿司传送带：厨房一次放上一盘拼好的寿司，顾客每次只拿一块；如果厨师换了一整盘，传送带上的位置指针也要同步归零。

It is like a sushi conveyor belt: the kitchen places a full plate, while the customer takes one piece at a time. If the chef replaces the whole plate, the position cursor must reset consistently.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

在 nanoVLA 里，它属于 `action-chunking` 和 `inference-loop` 之间的边界模块。上游 action head 产出 `[horizon, action_dim]`，队列缓存后，下游 robot loop 每个控制 tick 调一次 `get()`。如果省掉它，policy 的预测频率就必须等于机器人控制频率，延迟抖动会直接打到执行动作上。

In a nanoVLA, this sits between `action-chunking` and the `inference-loop`. The upstream action head emits `[horizon, action_dim]`; the queue stores it; the downstream robot loop calls `get()` once per control tick. Without this layer, policy inference frequency must match robot control frequency, and latency jitter leaks straight into execution.

## 自己跑一遍 / Try it yourself

```python
class Queue:
    def __init__(self):
        self.q, self.i = [], 0
    def put(self, actions):
        self.q, self.i = list(actions), 0
    def get(self):
        if self.i >= len(self.q):
            return None
        action = self.q[self.i]
        self.i += 1
        return action

q = Queue()
q.put(["open", "move", "close"])
print(q.get(), q.get(), q.get(), q.get())
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
open move close None
```

这个最小版本没有线程锁，但保留了 chunk 缓存和逐步消费。

This minimal version omits locks, but keeps the chunk-buffer and one-step-at-a-time consumption behavior.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **ACT temporal ensemble** / **ACT temporal ensemble**: 多个 chunk 重叠时不是简单替换，而是加权融合。 / Overlapping chunks are weighted and blended rather than simply replaced.
- **openpi action broker** / **openpi action broker**: 一次推理生成多步动作，控制端按 tick 消费。 / One inference call produces multiple actions, and the controller consumes them tick by tick.

## 注意事项 / Caveats / when it breaks

- **延迟估计错误** / **Wrong latency estimate**: RTC 模式需要知道推理期间已经消耗了多少旧动作。 / RTC mode needs to know how many old actions were consumed during inference.
- **clone 有成本** / **Clone has a cost**: 返回 clone 更安全，但大动作张量会多一次拷贝。 / Returning a clone is safer, but large action tensors pay an extra copy.

## 延伸阅读 / Further reading

- [LeRobot RTC action queue](https://github.com/huggingface/lerobot/blob/64b23178d5348609c266250d3e1f511eba4c33ff/src/lerobot/policies/rtc/action_queue.py)
- Diffusion Policy, action chunking, and real-time control loops
