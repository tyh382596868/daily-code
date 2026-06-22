---
date: 2026-06-22
topic: huggingface
source: huggingface
repo: huggingface/transformers
file: src/transformers/generation/continuous_batching/scheduler.py
permalink: https://github.com/huggingface/transformers/blob/123f5dd72644728722a970b3e3ad1445cf620470/src/transformers/generation/continuous_batching/scheduler.py#L122-L215
difficulty: intermediate
read_time: ~12 min
tags: [code-of-the-day, huggingface, continuous-batching, paged-attention, inference, serving]
---

# Transformers 自己的连续批处理调度器 / Transformers' Native Continuous Batching Scheduler

> **一句话 / In one line**: Transformers 现在内置了连续批处理引擎——这 90 行是它的核心：前缀缓存命中判断 + token 预算切割 + 分页 KV 块分配，三个操作合力把 GPU 吞吐量提升 3-10 倍。 / Transformers now ships its own continuous batching engine — these 90 lines are its heart: prefix-cache hit detection + token-budget splitting + paged KV-block allocation, the three moves that deliver 3-10× throughput over naïve batching.

## 为什么重要 / Why this matters

传统 LLM serving 把一个 batch 里所有请求 pad 到最长序列的长度，等最慢的那条请求跑完才开始下一个 batch——GPU 大量时间在等待和处理 padding token。连续批处理（Continuous Batching，CB）让每次 forward 只处理"真正需要处理的 token"，一条请求 decode 完就立刻填进新请求，GPU 利用率接近满载。vLLM 和 TGI 早就实现了这个，现在 transformers 也内置了 CB 引擎。

这三个方法是调度器的核心：`_allocate_blocks_if_needed` 负责给每条请求申请分页 KV 缓存块（不够了就记录"饥饿"等待逐出）；`_infer_request_tokens` 检查前缀缓存，如果新请求的 prompt 前缀已经被算过了，直接跳过那些 token，position_offset 往后移；`_schedule_request` 根据剩余 token 预算决定是把整个 prompt 排进当前 batch，还是先切一段（PENDING → PREFILLING → DECODING 三态机）。

Classic LLM serving pads every request in a batch to the length of the longest one, stalling until the slowest finishes. Continuous batching processes only the tokens that actually need computation this step, swapping in new requests the instant one finishes — GPU utilization near 100%. vLLM and TGI have had this for years; transformers just shipped its own CB engine.

These three methods are the scheduler's core: `_allocate_blocks_if_needed` claims paged KV-cache blocks (records "starvation" if blocks are exhausted); `_infer_request_tokens` checks the prefix cache and skips already-computed tokens by advancing `position_offset`; `_schedule_request` uses a three-state machine (PENDING → PREFILLING → DECODING) to decide whether the full prompt fits the token budget or needs splitting across multiple forward passes.

## 代码 / The code

`huggingface/transformers` — [`src/transformers/generation/continuous_batching/scheduler.py`](https://github.com/huggingface/transformers/blob/123f5dd72644728722a970b3e3ad1445cf620470/src/transformers/generation/continuous_batching/scheduler.py#L122-L215)

```python
def _allocate_blocks_if_needed(self, state: RequestState, len_next_tokens: int) -> bool:
    """Allocate additional cache blocks for a request if the currently allocated
    blocks are insufficient to accommodate the next tokens."""
    current_len = state.current_len()
    occupancy = state.allocated_blocks * self.cache.block_size - current_len
    if occupancy < len_next_tokens or state.allocated_blocks == 0:
        blocks_needed = ((len_next_tokens - occupancy + 1) // self.cache.block_size) + 1
        allocated = self.cache.allocate_blocks(blocks_needed, state.request_id, state.allocated_blocks)
        if allocated is None:
            # Starved active requests are tracked so the offloading manager can size bulk evictions.
            if state.request_id in self.active_requests:
                physical_blocks = self.cache.blocks_needed(blocks_needed, state.allocated_blocks)
                self.starved_requests.append((state, physical_blocks))
            return False
        state.allocated_blocks += allocated
    return True

def _infer_request_tokens(self, state: RequestState, request_ids_to_remove_from_waiting: set[str]) -> list[int]:
    """Prepares a request for processing in the current batch. If prefix sharing is enabled,
    this is where we look for a prefix match and split the request if found."""
    if self.cache.use_prefix_sharing and state.status == RequestStatus.PENDING and not state.is_cpu_offloaded:
        prefill_length = self.cache.search_prefix_match(state.request_id, state.remaining_prefill_tokens)
        if prefill_length > 0:
            self.active_requests[state.request_id] = state
            request_ids_to_remove_from_waiting.add(state.request_id)
            state.status = RequestStatus.PREFILLING
            # We keep track of the number of allocated blocks to avoid double allocation
            state.allocated_blocks += prefill_length // self.cache.block_size
            # Even if we match the whole request, we keep at least 1 token to start decoding
            prefill_length = min(prefill_length, len(state.remaining_prefill_tokens) - 1)
            state.remaining_prefill_tokens = state.remaining_prefill_tokens[prefill_length:]
            state.position_offset += prefill_length

    if state.status == RequestStatus.DECODING:
        request_tokens = state.tokens_to_process
    else:
        request_tokens = state.remaining_prefill_tokens
    return request_tokens

def _schedule_request(
    self,
    state: RequestState,
    request_tokens: list[int],
    token_budget: int,
    request_ids_to_remove_from_waiting: set[str],
) -> None:
    """Schedules a request for the current batch, updating the request's status according
    to the token budget left."""
    # If the request has one or more children we make sure not to prefill it entirely
    if state.num_children > 0 and token_budget >= len(request_tokens) - 1:
        token_budget = len(request_tokens) - 1
        self._requests_to_fork.append(state)

    # Case: we can process the entire prompt/remainder
    if len(request_tokens) <= token_budget:
        if state.status == RequestStatus.PENDING:
            self.active_requests[state.request_id] = state
            request_ids_to_remove_from_waiting.add(state.request_id)
        if state.status <= RequestStatus.PREFILLING:
            state.tokens_to_process = state.remaining_prefill_tokens
            state.remaining_prefill_tokens = []
            state.status = RequestStatus.DECODING

    # Otherwise: we need to split the request
    else:
        if state.status == RequestStatus.PENDING:
            self.active_requests[state.request_id] = state
            state.status = RequestStatus.PREFILLING
            request_ids_to_remove_from_waiting.add(state.request_id)
        state.remaining_prefill_tokens = request_tokens[token_budget:]
        state.tokens_to_process = request_tokens[:token_budget]
```

## 逐行讲解 / What's happening

1. **`occupancy = state.allocated_blocks * self.cache.block_size - current_len`（`_allocate_blocks_if_needed`）**:
   - 中文: 算出当前已分配的 KV 块还剩多少空位（"空余容量"）。如果空余容量小于即将处理的 token 数，就要再申请块。
   - English: Computes how many token slots remain in the already-allocated KV blocks. If fewer free slots than needed tokens, request more blocks.

2. **`blocks_needed = ((len_next_tokens - occupancy + 1) // self.cache.block_size) + 1`**:
   - 中文: 向上取整后再加 1，确保不会因为整除边界少申请一块。这里的 `+1` 是防御性分配——KV 缓存块申请失败（OOM）比多一块更糟糕。
   - English: Ceiling-divide with an extra block as safety margin. Failing to allocate (triggering eviction or stall) is worse than over-allocating by one block.

3. **`self.starved_requests.append((state, physical_blocks))`**:
   - 中文: 如果申请失败（GPU KV 缓存满了），把这条请求记录为"饥饿"。外层的 offloading manager 会查这个列表，批量把最老的请求从 GPU KV 缓存逐出到 CPU，腾出空间。
   - English: If allocation fails (GPU KV cache full), record the request as "starved." The offloading manager reads this list to bulk-evict the oldest requests to CPU, freeing blocks.

4. **`prefill_length = self.cache.search_prefix_match(...)`（`_infer_request_tokens`）**:
   - 中文: 前缀缓存命中检查。如果这条新请求的 prompt 前 N 个 token 已经在其他请求的 KV 缓存里算过了（例如同一个 system prompt），直接"借用"那些 KV blocks，跳过重新计算。`position_offset += prefill_length` 让模型知道这些 token 其实已经在哪里了。
   - English: Prefix cache hit check. If the first N tokens of this new request's prompt have already been computed (e.g., shared system prompt with another request), reuse those KV blocks directly. `position_offset += prefill_length` tells the model where to start without recomputing.

5. **`prefill_length = min(prefill_length, len(state.remaining_prefill_tokens) - 1)`**:
   - 中文: 即使整个 prompt 都命中前缀缓存，也保留最后一个 token 交给 decode 阶段处理，否则没有"启动 token"来开始自回归 decode。
   - English: Even if the entire prompt hits the prefix cache, keep the last token for the decode phase — the autoregressive decode loop needs at least one token to start generating from.

6. **`state.status = RequestStatus.DECODING`（`_schedule_request`，整段 prompt 能放下时）**:
   - 中文: 状态机的关键跳转：PENDING → DECODING，跳过了 PREFILLING。注释解释了原因：prefill 其实在下一个 batch 的 forward 里才真正执行，但提前把状态设成 DECODING 让异步调度保持一致（避免多个 batch 对同一请求产生歧义）。
   - English: Key state machine jump: PENDING → DECODING, skipping PREFILLING. The comment explains: the actual prefill happens in the next forward pass, but pre-setting to DECODING keeps async scheduling coherent (prevents multiple batches from double-scheduling the same request).

7. **`state.remaining_prefill_tokens = request_tokens[token_budget:]`（prompt 太长时）**:
   - 中文: 把超出 token 预算的那部分留下来，等下一个调度轮次处理。这是"split prefill"：长 prompt 分多次 forward 处理，避免一次 forward 消耗过多 token 预算导致其他请求等待。
   - English: Saves the tail of the prompt for the next scheduling round. This is "split prefill": a long prompt is processed across multiple forward passes, preventing one large prompt from monopolizing the token budget and stalling shorter requests.

## 类比 / The analogy

把 LLM 服务器想象成一个繁忙的餐厅（GPU）。旧方法：每桌客人（请求）必须等所有菜全部上齐才能结账，哪怕其他桌已经吃完在干等。CB 调度器就是一位高效的服务员：新客来了先问"你点的菜有没有现成的"（前缀缓存），有的话直接上；菜单太长就先送一部分（split prefill），腾出厨师（token 预算）给其他桌；一桌吃完立刻清桌迎新客（DECODING 结束 → 新请求顶上）。厨房永远不闲着。

Think of the LLM server as a busy restaurant (GPU). Old approach: each table (request) must wait until all its dishes are ready before the kitchen starts cooking for the next table — other tables sit idle waiting. The CB scheduler is an efficient headwaiter: when a new table arrives, first check if any of their dishes are pre-made (prefix cache); if the menu is too long, serve partial courses first (split prefill) to free the kitchen (token budget) for other tables; the moment a table finishes, immediately seat new guests (DECODING ends → new request slots in). The kitchen never idles.

## 自己跑一遍 / Try it yourself

```python
# Conceptual demo — no real GPU needed
class FakeState:
    def __init__(self, tokens):
        self.remaining_prefill_tokens = tokens
        self.tokens_to_process = []
        self.status = "PENDING"
        self.allocated_blocks = 0
        self.num_children = 0
        self.request_id = id(self)

def schedule_request(state, token_budget, waiting_set):
    tokens = state.remaining_prefill_tokens
    if len(tokens) <= token_budget:
        state.tokens_to_process = tokens
        state.remaining_prefill_tokens = []
        state.status = "DECODING"
        waiting_set.discard(state.request_id)
    else:
        state.tokens_to_process = tokens[:token_budget]
        state.remaining_prefill_tokens = tokens[token_budget:]
        state.status = "PREFILLING"
        waiting_set.discard(state.request_id)

req = FakeState(list(range(20)))
waiting = {req.request_id}
schedule_request(req, 8, waiting)
print(f"status={req.status}, to_process={req.tokens_to_process}, remaining={req.remaining_prefill_tokens}")
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
status=PREFILLING, to_process=[0, 1, 2, 3, 4, 5, 6, 7], remaining=[8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
```

一条 20 个 token 的 prompt 在 token_budget=8 的情况下被切成两段：前 8 个进入当前 batch，剩余 12 个等下一轮。

A 20-token prompt with `token_budget=8` is split: the first 8 go into the current batch, the remaining 12 wait for the next scheduling round.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **vLLM `core/scheduler.py`**: vLLM 的调度器是这个的先驱，逻辑相同但更复杂（抢占式逐出、beam search fork、prefix caching with radix tree）。 / vLLM's scheduler pioneered this pattern; its logic is equivalent but more complex (preemptive eviction, beam search forking, radix-tree prefix cache).
- **TGI（Text Generation Inference）**: Hugging Face 自家的独立 serving 框架，和这里的 CB engine 现在形成了"轻量内置 vs 生产级专用"的分层。 / HF's own standalone serving framework; now forms a "lightweight built-in vs. production-grade" tier alongside this CB engine.
- **`FIFOScheduler` + `PrefillFirstScheduler`（同文件 L331+）**: 这三个私有方法被两个公开的调度策略共用。FIFO 优先解码请求（更低延迟），PrefillFirst 优先完成 prefill 切片（更高吞吐）。 / These three private methods are shared by two public scheduler strategies in the same file. FIFO prioritizes decoding (lower latency); PrefillFirst prioritizes completing prefill splits (higher throughput).

## 注意事项 / Caveats / when it breaks

- **前缀缓存命中需要严格 token id 匹配 / Prefix cache requires exact token-id match**: `search_prefix_match` 比较 token id，不比较文字内容。同一段文字如果被不同 tokenizer 处理，结果不同，前缀不共享。 / Prefix matching compares token IDs, not text. The same text tokenized differently won't share a prefix — use a single tokenizer per service.
- **split prefill 增加首 token 延迟 / Split prefill increases time-to-first-token**: 长 prompt 被切成多段，每段占用一个 forward，首 token 延迟 ≈ ceil(prompt_len / token_budget) 个 forward pass 时间。生产环境需要调大 `token_budget` 或给长 prompt 请求独占调度。 / A long prompt split over multiple forwards means TTFT ≈ ceil(prompt_len / token_budget) forward passes. Tune `token_budget` up or give long prompts dedicated scheduling.
- **`block_size` 对碎片率有影响 / `block_size` affects fragmentation**: `block_size` 越大，内部碎片越多（最坏 `block_size - 1` tokens 浪费）；越小，block table 越大，overhead 上升。vLLM 默认 16，这是实践中的平衡点。 / Larger `block_size` increases internal fragmentation (up to `block_size - 1` wasted slots); smaller `block_size` grows the block table. vLLM's default of 16 is a practical sweet spot.

## 延伸阅读 / Further reading

- [Orca 论文 (Yu et al. 2022)](https://www.usenix.org/conference/osdi22/presentation/yu) — Continuous Batching 的原始论文，"iteration-level scheduling" 这个概念就来自这里。
- [vLLM 论文 (Kwon et al. 2023)](https://arxiv.org/abs/2309.06180) — PagedAttention + 连续批处理的完整系统设计。
- [transformers `generation/continuous_batching/`](https://github.com/huggingface/transformers/tree/main/src/transformers/generation/continuous_batching) — 完整的 CB 引擎：scheduler + paged cache + model runner + offloading manager。
