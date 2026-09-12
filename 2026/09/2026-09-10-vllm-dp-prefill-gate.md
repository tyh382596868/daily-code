---
date: 2026-09-10
topic: infrastructure
source: tracked
repo: vllm-project/vllm
file: vllm/v1/core/sched/scheduler.py
permalink: https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py#L601-L605
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, infrastructure, scheduling, prefill, data-parallel]
---

# vLLM prefill gate：解码不断，新 prefill 等节拍 / vLLM Prefill Gate: Keep Decoding, Admit Prefill on Cadence

> **一句话 / In one line**: vLLM 的 DP prefill balancing 会在非对齐 step 暂缓 prefill chunk，但继续调度 decode 请求填满本轮计算。 / vLLM's DP prefill balancing can defer prefill chunks on off-cadence steps while still scheduling decode work for the current step.

## 为什么重要 / Why this matters

中文：多路 data-parallel 推理里，某个 rank 突然接纳长 prefill 会把整个 step 拉长，其他 rank 只能等。vLLM 用 `defer_prefills` 把新 prefill 限制到对齐节拍上，让 decode 请求继续跑；这不是简单少干活，而是把高方差的 prompt 计算集中到更可控的 step。

English: In data-parallel serving, one rank admitting a long prefill can stretch the whole synchronized step while other ranks wait. vLLM uses `defer_prefills` to admit prefill work only on aligned cadence steps, while decode requests keep running. The goal is lower step-time variance, not idling the engine.

## 代码 / The code

`vllm-project/vllm` — [`vllm/v1/core/sched/scheduler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py#L601-L605)

```python
if defer_prefills and request.is_prefill_chunk:
    # DP prefill balancing: defer this in-progress prefill chunk to a
    # cadence-aligned step; decodes still run to fill this step.
    req_index += 1
    continue
```

Related config lives in [`vllm/config/scheduler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/config/scheduler.py#L187-L195):

```python
def get_scheduler_cls(self) -> type["SchedulerInterface"]:
    if self.scheduler_cls is None:
        if self.async_scheduling:
            from vllm.v1.core.sched.async_scheduler import AsyncScheduler
            return AsyncScheduler
        from vllm.v1.core.sched.scheduler import Scheduler
        return Scheduler
```

## 逐行讲解 / What's happening

1. **`defer_prefills` / `defer_prefills`**:
   - 中文: 这是外层调度 step 算出的门控条件，表示当前不是适合接纳 prefill 的节拍。
   - English: This is the per-step gate computed by the scheduler, meaning the current step is not the cadence for prefill admission.
2. **`request.is_prefill_chunk` / `request.is_prefill_chunk`**:
   - 中文: 只有 prompt/prefill chunk 被跳过；已经进入 decode 的请求不受这条分支影响。
   - English: Only prompt/prefill chunks are skipped. Requests already decoding are not blocked by this branch.
3. **`req_index += 1` + `continue` / `req_index += 1` + `continue`**:
   - 中文: 调度器向后看其他 running request，而不是终止整个 step。
   - English: The scheduler advances to the next running request instead of ending the whole step.
4. **scheduler class selection / scheduler class selection**:
   - 中文: 配置层还会在 async scheduling 打开时选择 `AsyncScheduler`，所以同一个策略可以落到不同调度实现里。
   - English: The config layer selects `AsyncScheduler` when async scheduling is enabled, so the same policy surface can map to different scheduler implementations.

## 类比 / The analogy

中文：像高速收费站给大货车设定固定放行窗口。小车继续过闸，大货车等下一次窗口一起走，这样整个车队速度更稳定。

English: It is like a toll plaza admitting trucks only at fixed windows. Cars keep moving through the lanes, while trucks wait for the next coordinated release so the convoy speed stays steadier.

## 自己跑一遍 / Try it yourself

```python
def schedule(requests, defer_prefills):
    picked = []
    for req in requests:
        if defer_prefills and req["phase"] == "prefill":
            continue
        picked.append(req["id"])
    return picked

requests = [
    {"id": "decode-a", "phase": "decode"},
    {"id": "prefill-b", "phase": "prefill"},
    {"id": "decode-c", "phase": "decode"},
]

print(schedule(requests, defer_prefills=True))
print(schedule(requests, defer_prefills=False))
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
['decode-a', 'decode-c']
['decode-a', 'prefill-b', 'decode-c']
```

中文：关键是跳过 prefill 时仍继续扫描队列；否则一个长 prompt 会挡住后面的 decode 工作。

English: The key is continuing the queue scan after skipping prefill; otherwise one long prompt would block decode work behind it.

## 注意事项 / Caveats / when it breaks

- **prefill 不能永远推迟** / **Prefill cannot be deferred forever**: cadence 配置必须保证等待请求会在后续 step 被接纳。
- **decode 仍受 token budget 限制** / **Decode still respects token budget**: 跳过 prefill 不代表本轮无限接纳 decode。
- **多 rank 才有收益** / **The benefit is strongest across ranks**: 单实例 serving 里，过度门控 prefill 可能只会增加首 token 延迟。

## 延伸阅读 / Further reading

- [vLLM scheduler.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)
- [vLLM scheduler config](https://github.com/vllm-project/vllm/blob/main/vllm/config/scheduler.py)
