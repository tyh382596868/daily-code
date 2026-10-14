---
date: 2026-10-14
topic: huggingface
source: huggingface
repo: huggingface/transformers
file: src/transformers/generation/continuous_batching/cache.py
permalink: https://github.com/huggingface/transformers/blob/28034545c35acac05793695bd5c878f66e6b544e/src/transformers/generation/continuous_batching/cache.py#L308-L369
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, continuous-batching, prefix-cache]
---

# Transformers PagedAttentionCache：前缀命中要所有 allocator 同意 / Transformers PagedAttentionCache: Prefix Hits Need Every Allocator to Agree

> **一句话 / In one line**: continuous batching 的 prefix sharing 先在每个 allocator 上找最长命中，再按最短公共前缀共享。 / Continuous batching prefix sharing finds matches in every allocator, then shares only the shortest common prefix.

## 为什么重要 / Why this matters

生成服务里的 KV cache 不一定只有一种 attention 层；如果 full attention 和 sliding attention 对前缀长度的理解不同，就必须以共同可恢复的那段为准。

KV cache in a generation server may span multiple attention types. If full and sliding attention disagree on recoverable prefix length, the shared prefix must be the common safe portion.

## 代码 / The code

`huggingface/transformers` - [`src/transformers/generation/continuous_batching/cache.py`](https://github.com/huggingface/transformers/blob/28034545c35acac05793695bd5c878f66e6b544e/src/transformers/generation/continuous_batching/cache.py#L308-L369)

```python
def search_prefix_match(self, request_id: str, prompt_ids: list[int]) -> int:
    """Searches the prompt for a prefix whose blocks are already in the cache. Matched blocks are shared with the
    request instead of being recomputed, and the number of matched tokens is returned."""
    if not self.use_prefix_sharing:
        return 0

    # Loop over all allocators to find the longest prefix match by all
    prefix_len = 2**32 - 1  # ~inf
    all_matched_blocks = {}
    for name, allocator in self.cache_allocators.items():
        matched_blocks = allocator.match_prefix_blocks(prompt_ids)
        # Stop if there was no prefill match
        if not matched_blocks:
            prefix_len = 0
            break
        # Otherwise, update accumulators
        else:
            prefix_len = min(prefix_len, len(matched_blocks) * allocator.tokens_per_page)
            all_matched_blocks[name] = matched_blocks

    # If there was a prefix match, acquire the blocks and return the number of matched tokens
    if prefix_len > 0:
        logger.debug(f"Found a prefix match of {prefix_len} tokens for request {request_id}")
        for name, matched_blocks in all_matched_blocks.items():
            self.cache_allocators[name].acquire_prefix_blocks(request_id, prefix_len, matched_blocks)
    return prefix_len

def count_new_complete_blocks(self, state: RequestState, request_len: int) -> dict[str, int]:
    """Counts the blocks that the forward pass over request_len new tokens will complete for the request, so they
    can be marked for de-duplication after the forward."""
    if not self.use_prefix_sharing:
        return {}
    complete_blocks = {}
    for name, allocator in self.cache_allocators.items():
        tokens_in_last_block = state.current_len() % allocator.tokens_per_page
        new_complete_blocks = (tokens_in_last_block + request_len) // allocator.tokens_per_page
        if new_complete_blocks:
            complete_blocks[name] = new_complete_blocks
    return complete_blocks

def mark_complete_blocks(self, state: RequestState, complete_blocks: dict[str, int]) -> None:
    """Registers the content hashes of the blocks the last forward pass completed for the request, making them
    available for de-duplication."""
    # The status can be FINISHED in async mode, because batch N+1 offloaded the request before batch N was over:
    # the block table no longer exists in that case
    if not self.allow_block_sharing or not complete_blocks or state.status == RequestStatus.FINISHED:
        return
    token_ids = state.initial_tokens + state.generated_tokens
    for name, new_complete_blocks in complete_blocks.items():
        self.cache_allocators[name].mark_complete_blocks(state.request_id, token_ids, new_complete_blocks)

def evict_cached_blocks(self) -> bool:
    """Evicts all the blocks kept cached for de-duplication, returning them to the pool. Returns True if any block
    was evicted. Called under memory pressure: cached blocks are a bonus, never worth starving live requests."""
    evicted_any = False
    for allocator in self.cache_allocators.values():
        evicted_blocks = allocator.ledger.evict_cached_blocks()
        if evicted_blocks:
            logger.debug(f"Evicting {len(evicted_blocks)} cached blocks from allocator {allocator.index}")
            self.pool.free_blocks(allocator.index, evicted_blocks)
            evicted_any = True
    return evicted_any
```

## 逐行讲解 / What's happening

1. **第 308-333 行 / Lines 308-333 (`search_prefix_match`)**:
   - 中文: 每个 allocator 都要匹配 prompt；任意一个没命中，整个前缀共享就归零。
   - English: Every allocator must match the prompt; if any one misses, prefix sharing falls back to zero.
2. **第 325-332 行 / Lines 325-332 (shortest common prefix)**:
   - 中文: 命中长度用 `min` 收缩，因为不同 allocator 的 page 大小可能不同。
   - English: The hit length shrinks with `min` because allocators can have different page sizes.
3. **第 335-357 行 / Lines 335-357 (new complete blocks)**:
   - 中文: forward 后只登记刚完成的整块，半块不能作为后续请求的可复用前缀。
   - English: After a forward pass, only newly completed whole blocks are indexed; partial blocks are not reusable prefixes.
4. **第 359-369 行 / Lines 359-369 (eviction)**:
   - 中文: cache hit 是 bonus；内存紧张时 cached blocks 先让路给活请求。
   - English: Cache hits are a bonus; under memory pressure, cached blocks yield to live requests.

## 类比 / The analogy

这像多人拼车走高速：只有每辆车都能走同一段路，这段路才算真正可共享；有一辆车上不了高速，就得从共同起点重新规划。

It is like carpooling across a highway segment: the segment is shareable only if every car can use it. If one car cannot enter, the group must fall back to the common route.

## 自己跑一遍 / Try it yourself

```python
def prefix_hit(alloc_hits):
    best = 2**32 - 1
    for blocks, tokens_per_page in alloc_hits:
        if not blocks:
            return 0
        best = min(best, len(blocks) * tokens_per_page)
    return best

print(prefix_hit([([1, 2, 3], 16), ([7], 32)]))
print(prefix_hit([([1, 2], 16), ([], 32)]))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
32
0
```

中文: 多 allocator cache 的安全共享长度，是所有 allocator 的交集，不是最长的那一个。

English: In a multi-allocator cache, the safe shared length is the intersection, not the longest individual hit.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **vLLM prefix cache** / **vLLM prefix cache**: block hash 命中也必须保持从头连续。 / Block-hash hits also need to stay contiguous from the start.
- **SGLang radix cache** / **SGLang radix cache**: 前缀树返回最长公共前缀，再让调度器决定是否复用。 / The radix tree returns a longest common prefix, then the scheduler decides whether to reuse it.

## 注意事项 / Caveats / when it breaks

- **page 大小不同** / **Page sizes may differ**: 用 block 数乘 page 大小后再取最小值。 / Convert block counts into token lengths before taking the minimum.
- **finished 请求没有 block table** / **Finished requests may have no block table**: async 模式下完成状态要直接跳过登记。 / In async mode, a finished request can already have lost its block table.

## 延伸阅读 / Further reading

- [Transformers continuous batching cache](https://github.com/huggingface/transformers/blob/28034545c35acac05793695bd5c878f66e6b544e/src/transformers/generation/continuous_batching/cache.py#L308-L369)
