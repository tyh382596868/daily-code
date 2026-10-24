---
date: 2026-10-24
topic: huggingface
source: huggingface
repo: huggingface/transformers
file: src/transformers/generation/continuous_batching/requests.py
permalink: https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/generation/continuous_batching/requests.py#L184-L259
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, generation, continuous-batching]
---

# Transformers 连续批处理状态：状态转换就是计时边界 / Transformers Continuous Batching State: State Transitions Define Timing Boundaries

> **一句话 / In one line**: 连续批处理把每个请求建模为显式状态机，开始和结束时间由状态转换自动记录。 / Continuous batching models each request as an explicit state machine whose transitions automatically record start and end times.

## 为什么重要 / Why this matters

流式生成既要维护 token，又要回答“何时真正开始”和“为何结束”。把时序逻辑塞进 scheduler 容易遗漏异常路径；这里让状态 setter 成为唯一记账入口。

Streaming generation must track tokens and answer when a request truly began and why it ended. Hiding timing in a scheduler misses failure paths; here the status setter becomes the single accounting point.

## 代码 / The code

`huggingface/transformers` — [`requests.py`](https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/generation/continuous_batching/requests.py#L184-L259)

```python
def __post_init__(self):
    self._new_tokens_limit = 2147483647 if self.max_new_tokens is None else self.max_new_tokens
    self.remaining_prefill_tokens = self.initial_tokens[:]
    if isinstance(self.eos_token_id, int) and self.eos_token_id >= 0:
        self._eos_token_ids.add(self.eos_token_id)
    elif self.eos_token_id is not None:
        self._eos_token_ids.update(i for i in self.eos_token_id if i >= 0)

@status.setter
def status(self, value):
    if self._status == RequestStatus.PENDING:
        self.lifespan = (time.perf_counter(), -1)
    if value >= RequestStatus.FINISHED:
        self.lifespan = (self.lifespan[0], time.perf_counter())
    self._status = value
```

## 逐行讲解 / What's happening

初始化把可选长度和 EOS 输入转成调度器可直接比较的内部表示。之后任何离开 `PENDING` 的转换盖上开始时间，任何终止状态盖上结束时间；成功和失败因此共享完整的延迟记录。

Initialization converts optional length and EOS inputs into scheduler-friendly internal values. Every transition out of `PENDING` stamps a start, and every terminal state stamps an end, so success and failure share complete latency records.

## 类比 / The analogy

像图书馆借阅卡：借出时盖第一个章，归还或遗失时盖最后一个章；不用让每位馆员各自记时间。

It is a library card: stamp once when the book leaves and once when it returns or is declared lost; do not ask every librarian to keep separate clocks.

## 自己跑一遍 / Try it yourself

```python
from enum import IntEnum
class S(IntEnum): PENDING=0; DECODING=1; FINISHED=2
class R:
    def __init__(self): self.s=S.PENDING; self.events=[]
    def set(self, s):
        if self.s == S.PENDING: self.events.append('start')
        if s >= S.FINISHED: self.events.append('end')
        self.s=s
r=R(); r.set(S.DECODING); r.set(S.FINISHED); print(r.events)
```

运行 / Run with: `python try.py`

预期输出 / Expected output: `['start', 'end']`。 / `['start', 'end']`.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **serving queues** / **serving queues**: 请求生命周期也是限流、取消和指标采集的共同边界。 / Request lifecycle is also the shared boundary for rate limits, cancellation, and metrics.

## 注意事项 / Caveats / when it breaks

- **状态必须单向收敛** / **States must converge**: 终态后再回到 decode 会让 latency 和输出语义失真。 / Returning to decode after a terminal state corrupts latency and output semantics.

## 延伸阅读 / Further reading

- [Transformers continuous-batching requests](https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/generation/continuous_batching/requests.py#L184-L259)
