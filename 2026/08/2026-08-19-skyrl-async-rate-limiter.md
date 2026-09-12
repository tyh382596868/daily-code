---
date: 2026-08-19
topic: infrastructure
source: trending
repo: NovaSky-AI/SkyRL
file: skyrl/train/utils/rate_limiter.py
permalink: https://github.com/NovaSky-AI/SkyRL/blob/9719b4f74ae9cbb6ec022a8d67c1e8a835b52c7d/skyrl/train/utils/rate_limiter.py#L82-L177
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, infrastructure, async, rate-limiter]
---

# SkyRL async limiter：速率和并发是两道闸 / SkyRL Async Limiter: Rate and Concurrency Are Two Gates

> **一句话 / In one line**: SkyRL 用 token bucket 控制“启动速度”，再用 semaphore 控制“同时在跑多少个 rollout”。 / SkyRL uses a token bucket to control start rate, then a semaphore to control how many rollouts run concurrently.

## 为什么重要 / Why this matters

LLM RL 训练不只是反向传播，rollout 生成、环境交互、打分服务也会成为瓶颈。如果只限制并发，任务可能瞬间挤爆外部服务；如果只限制速率，慢任务又可能无限堆积。SkyRL 把这两个控制面拆开，适合真实异步训练流水线。

LLM RL training is not only backpropagation; rollout generation, environment interaction, and reward services can all bottleneck. Limiting only concurrency can still burst external services; limiting only rate can still allow slow tasks to pile up. SkyRL separates the two controls, which is a practical shape for asynchronous training pipelines.

## 代码 / The code

`NovaSky-AI/SkyRL` — [`skyrl/train/utils/rate_limiter.py`](https://github.com/NovaSky-AI/SkyRL/blob/9719b4f74ae9cbb6ec022a8d67c1e8a835b52c7d/skyrl/train/utils/rate_limiter.py#L82-L177)

```python
class AsyncRateLimiter(RateLimiterInterface):
    """Combined rate limiter and concurrency limiter for async code.

    Rate limiting (token bucket algorithm):
    - Bucket holds tokens, max capacity = rate (trajectories_per_second)
    - Tokens refill at rate N per second
    - Each acquire() call consumes 1 token
    - If no tokens available, caller waits until refill

    Concurrency limiting (semaphore):
    - Limits how many operations can run simultaneously
    - acquire() waits if max_concurrency operations are already running
    - release() must be called when operation completes

    Note: Fractional rates >= 1.0 are supported (e.g., 1.5 means 1.5 ops/second).
    Rates < 1.0 are not supported because the bucket capacity equals the rate,
    so the bucket could never hold a full token to allow an acquire().
    """

    def __init__(
        self,
        rate: Optional[float] = None,
        max_concurrency: Optional[int] = None,
    ):
        """Initialize the rate limiter.

        Args:
            rate: Maximum operations per second (tokens per second).
                  Must be >= 1.0 if provided. None disables rate limiting.
            max_concurrency: Maximum concurrent operations allowed.
                  Must be >= 1 if provided. None disables concurrency limiting.
        """
        if rate is not None and rate < 1.0:
            raise ValueError(
                f"Rate must be >= 1.0, got {rate}. "
                "Rates < 1.0 are not supported due to the token bucket implementation."
            )
        if max_concurrency is not None and max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")

        # Rate limiting state (token bucket)
        self._rate = rate
        if rate is not None:
            self._max_tokens = rate  # bucket capacity = rate
            self._tokens = rate  # start with a full bucket
            self._last_refill = time.monotonic()
            self._rate_lock = asyncio.Lock()

        # Concurrency limiting state (semaphore)
        self._max_concurrency = max_concurrency
        if max_concurrency is not None:
            self._semaphore = asyncio.Semaphore(max_concurrency)

    async def acquire(self) -> None:
        """Acquire permission to proceed, waiting if necessary.

        First applies rate limiting (controls how fast operations start),
        then acquires concurrency slot (controls how many run simultaneously).
        """
        # Rate limit first (controls start rate)
        if self._rate is not None:
            await self._acquire_rate_token()

        # Then concurrency limit (controls concurrent execution)
        if self._max_concurrency is not None:
            await self._semaphore.acquire()

    def release(self) -> None:
        """Release a concurrency slot.

        Must be called after the operation completes to allow other
        operations to proceed. Safe to call even if concurrency limiting
        is disabled.
        """
        if self._max_concurrency is not None:
            self._semaphore.release()

    async def _acquire_rate_token(self) -> None:
        """Acquire a rate limit token, waiting if necessary."""
        while True:
            async with self._rate_lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                # Calculate wait time for next token
                wait_time = (1.0 - self._tokens) / self._rate
            # Sleep outside lock so other coroutines can check/update state
            await asyncio.sleep(wait_time)

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._max_tokens, self._tokens + elapsed * self._rate)
        self._last_refill = now
```

## 逐行讲解 / What's happening

1. **第 114-120 行 / Lines 114-120 (validation)**:
   - 中文: token bucket 容量等于 rate，所以 rate 小于 1 时桶永远装不下完整 1 个 token，直接拒绝。
   - English: The bucket capacity equals the rate; if rate is below 1, it can never hold a full token, so the config is rejected.
2. **第 122-133 行 / Lines 122-133 (two states)**:
   - 中文: rate state 用 `_tokens` 和 `_last_refill`，并发 state 用 `asyncio.Semaphore`，两个概念不混在一起。
   - English: Rate state uses `_tokens` and `_last_refill`; concurrency state uses an `asyncio.Semaphore`. The two concepts stay separate.
3. **第 135-147 行 / Lines 135-147 (`acquire`)**:
   - 中文: 先拿速率 token，再拿并发槽位，表示“允许启动”早于“允许占用运行资源”。
   - English: The limiter takes a rate token first, then a concurrency slot, meaning "allowed to start" is checked before "allowed to occupy runtime capacity".
4. **第 159-170 行 / Lines 159-170 (sleep outside lock)**:
   - 中文: 没 token 时先算等待时间，释放 lock 后再 sleep，避免一个等待者堵住所有协程。
   - English: When no token is available, the code computes wait time, releases the lock, then sleeps so one waiter does not block every coroutine.

## 类比 / The analogy

像游乐园入口有两道门：第一道每秒只放几个人进队伍，第二道限制同一时间坐上设备的人数。只靠其中一道门都管不好拥堵。

It is like an amusement ride with two gates: one controls how quickly people enter the line, and the other limits how many people are on the ride at once. Either gate alone is insufficient.

## 自己跑一遍 / Try it yourself

```python
import asyncio, time

async def worker(i, sem, starts):
    async with sem:
        starts.append((i, round(time.monotonic() - starts[0][1], 2)))
        await asyncio.sleep(0.05)

async def main():
    sem = asyncio.Semaphore(2)
    starts = [("t0", time.monotonic())]
    tasks = [worker(i, sem, starts) for i in range(5)]
    await asyncio.gather(*tasks)
    print(starts[1:])

asyncio.run(main())
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[(0, 0.0), (1, 0.0), (2, 0.05), (3, 0.05), (4, 0.1)]
```

中文: 这个小例子只演示并发闸；SkyRL 再加 token bucket 来控制启动速率。

English: This small example shows only the concurrency gate; SkyRL adds a token bucket to control start rate as well.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **API gateway throttling** / **API gateway throttling**: 常见组合是 QPS limiter 加 inflight request limiter。 / A common setup is a QPS limiter plus an in-flight request limiter.
- **rollout workers** / **rollout workers**: RL 系统经常需要同时保护环境服务、reward model 和推理服务。 / RL systems often need to protect environment services, reward models, and inference servers at the same time.

## 注意事项 / Caveats / when it breaks

- **必须 release** / **You must release**: 使用者忘记 `release()` 会泄漏 semaphore 槽位；最好用 async context manager 包住。 / Forgetting `release()` leaks semaphore capacity; prefer using an async context manager.
- **rate 小于 1 不支持** / **Rates below 1 are unsupported**: 这个实现的桶容量等于 rate，所以不能表达“每 5 秒 1 个任务”。 / This implementation cannot express "one task every five seconds" because bucket capacity equals rate.

## 延伸阅读 / Further reading

- [SkyRL `AsyncRateLimiter`](https://github.com/NovaSky-AI/SkyRL/blob/9719b4f74ae9cbb6ec022a8d67c1e8a835b52c7d/skyrl/train/utils/rate_limiter.py#L82-L177)
- [SkyRL repository](https://github.com/NovaSky-AI/SkyRL)
