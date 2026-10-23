---
date: 2026-10-23
topic: infrastructure
source: trending
repo: rokbenko/quackd
file: adapters/lerobot/src/quackd_lerobot/policy/loop.py
permalink: https://github.com/rokbenko/quackd/blob/80b5e85cb57574424946e6acecb7ed223117b295/adapters/lerobot/src/quackd_lerobot/policy/loop.py#L217-L260
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, action-chunking]
---

# quackd 动作 chunk 补给：延迟决定何时请求下一包 / quackd Action-Chunk Refill: Latency Sets the Next Request Point

> **一句话 / In one line**: quackd 把推理延迟换算成 tick，并在队列到达「半 chunk 或两倍延迟」时提前补给。 / quackd converts inference latency to ticks and refills when the queue reaches half a chunk or twice the latency.

## 为什么重要 / Why this matters

机器人推理不是只要模型快：动作 chunk 太短会在下一次结果抵达前耗尽。这个阈值把时延预算写进控制循环。

Robotic inference is not only about a fast model: a short action chunk can run out before the next result arrives. This threshold puts the latency budget into the control loop.

## 代码 / The code

`rokbenko/quackd` — [`loop.py`](https://github.com/rokbenko/quackd/blob/80b5e85cb57574424946e6acecb7ed223117b295/adapters/lerobot/src/quackd_lerobot/policy/loop.py#L217-L260)

```python
def latency_ticks(latency_s, rate_hz): return max(0, math.ceil(latency_s * rate_hz - EXACT))
def refill_at(latency, chunk): return max(REFILL_LATENCIES * latency, chunk // 2)
def refill_due(left, latency, chunk): return left <= refill_at(latency, chunk)
def starved_each_chunk(latency, chunk): return max(0, 2 * latency - chunk)
```

## 逐行讲解 / What's happening

`ceil` 不低估到达时间；补给阈值取两倍 latency 与半 chunk 的较大值。chunk 小于两倍 latency 时，每个 chunk 都会饿死若干 tick。

`ceil` never underestimates arrival time; the threshold is the larger of two latencies and half a chunk. When a chunk is shorter than twice latency, every chunk starves for some ticks.

## 类比 / The analogy

像输液泵换药袋：在旧袋仍有安全余量时请求新袋。 / Like an infusion pump: request the next bag while a safe reserve remains.

## 自己跑一遍 / Try it yourself

```python
import math
ticks, chunk = math.ceil(.12 * 10), 4
print(ticks, max(2*ticks, chunk//2), max(0, 2*ticks-chunk))
```

运行 / Run with: `python try.py`

预期输出 / Expected output: `2 4 0`。

## 注意事项 / Caveats / when it breaks

- 使用偏保守的端到端延迟，而不是平均 GPU forward 时间。 / Use conservative end-to-end latency, not average GPU forward time.

## 延伸阅读 / Further reading

- [quackd policy loop source](https://github.com/rokbenko/quackd/blob/80b5e85cb57574424946e6acecb7ed223117b295/adapters/lerobot/src/quackd_lerobot/policy/loop.py)
