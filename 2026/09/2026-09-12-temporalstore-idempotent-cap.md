---
date: 2026-09-12
topic: diffusion
source: trending
repo: matrixarkai/TemporalStore
file: sdk/python/temporalstore/control_state.py
permalink: https://github.com/matrixarkai/TemporalStore/blob/53a1aff488a117e19e990a21b56514428640b095/sdk/python/temporalstore/control_state.py#L393-L463
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, trending, serving-state, idempotency, retry-safety]
---

# TemporalStore 决策：把计数和准入放进一次原子操作 / TemporalStore Decisions: Make Counting and Admission One Atomic Operation

> **一句话 / In one line**: TemporalStore 用服务端原子计数加上可选 `event_id` 去重，把“是否允许”从易竞争的读改写成一次安全决策。 / TemporalStore combines an atomic server-side counter with optional `event_id` deduplication so admission becomes one race-safe decision.

## 为什么重要 / Why this matters

中文：在广告频控、API quota、扩散任务排队或机器人 serving 中，最危险的代码通常是“先读计数，再判断，再写回”。并发请求会同时读到旧值，最后突破上限。TemporalStore 把递增和读取放进一个命令；如果调用方能提供稳定的 `event_id`，重试和消息重放也不会重复计数。

English: In ad frequency caps, API quotas, diffusion-job admission, or robot serving, the dangerous pattern is “read, decide, write.” Concurrent requests can read the same old value and exceed the cap. TemporalStore moves increment and read into one command, and a stable `event_id` makes retries and queue replays idempotent.

## 代码 / The code

`matrixarkai/TemporalStore` — [`sdk/python/temporalstore/control_state.py`](https://github.com/matrixarkai/TemporalStore/blob/53a1aff488a117e19e990a21b56514428640b095/sdk/python/temporalstore/control_state.py#L393-L463)

```python
    def _backoff(self, attempt: int) -> None:
        delay = min(self._cfg.backoff_max_s, self._cfg.backoff_base_s * (2 ** (attempt - 1)))
        time.sleep(delay * (0.5 + random.random() / 2))  # full jitter

    def _emit(self, name: str, value: float, tags: dict) -> None:
        if self._cfg.metrics is not None:
            try:
                self._cfg.metrics(name, value, tags)
            except Exception:  # metrics must never break serving
                logger.debug("metrics hook raised", exc_info=True)

    def _incr_and_read(self, key: str, amount: int, start_ms: int, end_ms: int,
                       occur_ms: int, aggregator: str = "sum") -> int:
        """Atomic increment-then-read (``HSETANDGET``, a.k.a. ``COUNTERSETANDGET``).

        At-most-once on retry.
        """
        reply = self._call(
            ["HSETANDGET", key, occur_ms, amount, start_ms, end_ms, aggregator],
            idempotent=False,
        )
        return int(reply)

    def _incr_and_read_idempotent(self, key: str, amount: int, start_ms: int, end_ms: int,
                                  occur_ms: int, precision_ms: int, ttl_ms: int,
                                  event_id: str, aggregator: str = "sum") -> int:
        """Atomic, idempotent increment-then-read (``HSETANDGETOPT``, a.k.a. ``COUNTERSETANDGETOPT``).

        A repeated ``event_id`` inside the dedup window is a no-op write that
        still returns the current total, so at-least-once queue replays and
        client retries do not double-count.
        """
        reply = self._call(
            ["HSETANDGETOPT", key, occur_ms, amount, start_ms, end_ms, aggregator,
             precision_ms, ttl_ms, event_id],
            idempotent=True,
        )
        return int(reply)

    def frequency_cap(self, *, user_id: str, campaign_id: str, limit: int,
                      event: str = "impression", amount: int = 1,
                      tenant: Optional[str] = None, now_ms: Optional[int] = None,
                      event_id: Optional[str] = None) -> Decision:
        now_ms = now_ms if now_ms is not None else _now_ms()
        start_ms, end_ms, day = _utc_day_bounds_ms(now_ms)
        key = self.key("user", user_id, "campaign", campaign_id, event, day, tenant=tenant)
        if event_id is not None:
            count = self._incr_and_read_idempotent(
                key, amount, start_ms, end_ms, now_ms,
                precision_ms=_DAY_MS, ttl_ms=_DAY_MS, event_id=event_id)
        else:
            count = self._incr_and_read(key, amount, start_ms, end_ms, now_ms)
        allowed = count <= limit
        return Decision(allowed=allowed, count=count, limit=limit,
                        reason="" if allowed else "frequency_cap_exceeded")
```

## 逐行讲解 / What's happening

1. **第 393-395 行 / Lines 393-395 (`_backoff`)**:
   - 中文：重试等待采用指数退避加 full jitter，避免一批客户端同时在同一个时间点再次撞向服务端。
   - English: Exponential backoff with full jitter prevents a group of clients from retrying in lockstep.
2. **第 397-402 行 / Lines 397-402 (`_emit`)**:
   - 中文：metrics hook 被包在异常保护里；监控系统坏了不能反过来阻断 serving。
   - English: The metrics hook is isolated so an observability failure cannot break serving.
3. **第 405-415 行 / Lines 405-415 (`HSETANDGET`)**:
   - 中文：普通写入明确标记 `idempotent=False`。如果请求已经发出后超时，客户端不能盲目重试，否则可能双写。
   - English: The ordinary write is explicitly non-idempotent. After a post-send timeout, blind retry could double-apply the increment.
4. **第 417-431 行 / Lines 417-431 (`HSETANDGETOPT`)**:
   - 中文：带 `event_id` 的命令让服务端在去重窗口内把重复事件变成 no-op，但仍返回当前总数。
   - English: The event-aware command turns duplicates into a no-op inside the dedup window while still returning the current total.
5. **第 439-463 行 / Lines 439-463 (`frequency_cap`)**:
   - 中文：先把用户、campaign、事件类型和 UTC day 组成稳定 key，再选择普通或幂等原子命令，最后只做一次本地 `count <= limit` 判断。
   - English: The method builds a stable key from user, campaign, event, and UTC day, selects the normal or idempotent atomic command, then performs one local `count <= limit` decision.

## 类比 / The analogy

中文：像售票窗口里的盖章员。不能让两个窗口都先看“还剩一张票”再各自卖出；正确做法是盖章和减库存由同一个柜台动作完成，重复出示同一订单号也只算一次。

English: Think of a ticket office. Two windows must not both observe “one ticket left” and sell it. The stamp and inventory decrement happen at one counter action, and showing the same order ID again counts once.

## 自己跑一遍 / Try it yourself

```python
class Cap:
    def __init__(self, limit):
        self.limit, self.count, self.seen = limit, 0, set()

    def accept(self, event_id):
        if event_id in self.seen:
            return self.count <= self.limit
        self.seen.add(event_id)
        self.count += 1
        return self.count <= self.limit

cap = Cap(2)
print(cap.accept("a"), cap.accept("a"), cap.accept("b"), cap.accept("c"))
print(cap.count)
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
True True True False
3
```

中文：重复的 `"a"` 没有再次增加计数，但第三个新事件仍然会被记录，并因此被拒绝；这正是“计数”和“准入”必须一起设计的地方。

English: Repeating `"a"` does not increment again, but the third new event is still recorded and rejected. That is why counting and admission must be designed together.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Payment idempotency keys** / **Payment idempotency keys**: 用订单号保护网络重试。 / Use an order ID to make network retries safe.
- **Distributed rate limiters** / **Distributed rate limiters**: 在服务端原子更新 token bucket 或 counter。 / Atomically update a token bucket or counter on the server.
- **Diffusion job admission** / **Diffusion job admission**: 给一次生成请求分配唯一 job ID，防止 worker 重放重复扣 quota。 / Assign a unique job ID so worker replay does not consume quota twice.

## 注意事项 / Caveats / when it breaks

- **event ID 必须稳定且有范围** / **Event IDs must be stable and scoped**: 随机生成新 ID 会让重试失去去重意义。
- **去重窗口不是永久记忆** / **The dedup window is not permanent memory**: TTL 过期后再次发送同一 ID 可能重新计数。
- **“已发送但未收到回复”最难** / **The sent-but-no-reply case is the hard one**: 普通非幂等写入不能无脑重试。

## 延伸阅读 / Further reading

- [TemporalStore Control State](https://github.com/matrixarkai/TemporalStore/blob/53a1aff488a117e19e990a21b56514428640b095/sdk/python/temporalstore/control_state.py)
- [TemporalStore README](https://github.com/matrixarkai/TemporalStore)
