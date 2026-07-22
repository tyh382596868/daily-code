---
date: 2026-07-21
topic: infrastructure
source: trending
repo: vllm-project/vllm
file: vllm/v1/core/kv_cache_manager.py
permalink: https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py#L2236-L2524
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, infrastructure, vllm, kv-cache, scheduler, trending]
---

# vLLM allocate_slots：一次分配同时考虑命中、外部 KV 和 speculative lookahead / vLLM allocate_slots: Allocate for Hits, External KV, and Speculative Lookahead Together

> **一句话 / In one line**: `allocate_slots` 把请求序列拆成已算、prefix 命中、外部 connector 命中、新 token 和 lookahead token，再决定 KV block 是否足够。 / `allocate_slots` splits a request into computed tokens, prefix-cache hits, external connector hits, new tokens, and lookahead tokens, then decides whether KV blocks are available.

## 为什么重要 / Why this matters

现代 LLM serving 不是“来一个 token 分一个 block”这么简单。prefix cache、P/D 分离、异步 KV connector、滑窗 attention、speculative decoding 和 admission control 会同时影响一个请求能不能进 batch。vLLM 把这些约束集中在 slot allocation 入口。

Modern LLM serving is not as simple as "allocate one block for each new token." Prefix cache, P/D separation, async KV connectors, sliding-window attention, speculative decoding, and admission control all affect whether a request can enter the batch. vLLM concentrates these constraints at slot allocation.

## 代码 / The code

`vllm-project/vllm` — [`vllm/v1/core/kv_cache_manager.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py#L2236-L2524)

```python
# Simplified teaching slice, not a verbatim copy.
def allocate_slots(request, new, new_comp=0, ext_comp=0, lookahead=0):
    if new == 0 and ext_comp == 0:
        raise ValueError("nothing to allocate")

    local_computed = request.computed + new_comp
    total_computed = min(local_computed + ext_comp, max_model_len)

    watermark = watermark_blocks if request.is_waiting_or_preempted else 0

    if full_sequence_must_fit:
        required = blocks_for(request.total_tokens, total_computed) + watermark
        if required > free_blocks:
            return None

    need_slot = min(total_computed + new + lookahead, max_model_len)
    remove_skipped_blocks(request, total_computed - request.in_flight)
    return allocate_blocks_until(need_slot)
```

## 逐行讲解 / What's happening

1. **空分配直接拒绝 / Reject empty allocation**: 中文: 如果没有新 token，也没有外部 KV 要落位，就没有工作可做。 English: if there are no new tokens and no external KV to place, there is no work.
2. **本地命中和外部命中分开 / Local and external hits are separate**: 中文: vLLM 自己缓存的 prefix 和 connector 提供的 KV 不是同一种所有权。 English: prefix cached by vLLM and KV supplied by a connector have different ownership.
3. **watermark 保护在飞请求 / Watermark protects in-flight work**: 中文: 等待/抢占请求不能把最后的空闲 block 全吃掉。 English: waiting or preempted requests should not consume the last free blocks needed by in-flight work.
4. **full sequence gate 防止过度准入 / Full-sequence gate prevents over-admission**: 中文: chunked prefill 不能只看第一段能不能放下。 English: chunked prefill cannot admit a request just because its first chunk fits.
5. **先释放跳过窗口外的 block / Free skipped blocks before allocating**: 中文: 滑窗外或不再读的 KV 先归还，减少不必要驱逐。 English: KV outside the attention window is returned first to reduce unnecessary eviction.

## 类比 / The analogy

像机场登机口调度：有些乘客已经安检，有些从贵宾通道转入，有些是候补同行人。登机口不能只数新来的乘客，还要给已经在廊桥上的人留路。

It is like airport gate scheduling: some passengers already cleared security, some arrive through a partner lane, and some are standby companions. The gate cannot count only new arrivals; it must leave room for people already on the jet bridge.

## 自己跑一遍 / Try it yourself

```python
def can_allocate(free_blocks, computed, new_hit, external_hit, new, lookahead, block_size=4, reserve=1):
    total_tokens = computed + new_hit + external_hit + new + lookahead
    needed_blocks = (total_tokens + block_size - 1) // block_size
    return needed_blocks + reserve <= free_blocks

print(can_allocate(6, computed=8, new_hit=4, external_hit=0, new=4, lookahead=2))
print(can_allocate(4, computed=8, new_hit=4, external_hit=4, new=4, lookahead=2))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
True
False
```

## 注意事项 / Caveats / when it breaks

- **block 对齐会重算边界 token / Block alignment can recompute boundary tokens**: cache hit 到最后一个 token 时，logits 需求可能迫使重算。 / a full cache hit may still recompute a boundary token to produce logits.
- **connector KV 有不同生命周期 / Connector KV has a different lifecycle**: 外部 KV 可能已经算好，但本地 block 引用计数还没增加。 / external KV may be computed while local block refcounts are not yet increased.
- **speculative lookahead 不是免费 / Speculative lookahead is not free**: 草稿 token 也要预留 KV 空间。 / draft tokens also need reserved KV space.

## 延伸阅读 / Further reading

- [vLLM KV cache manager allocate_slots](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py#L2236-L2524)
- [vLLM repository](https://github.com/vllm-project/vllm)
