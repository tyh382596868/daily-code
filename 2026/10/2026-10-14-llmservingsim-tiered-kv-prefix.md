---
date: 2026-10-14
topic: infrastructure
source: trending
repo: casys-kaist/LLMServingSim
file: serving/core/kv_cache_manager.py
permalink: https://github.com/casys-kaist/LLMServingSim/blob/b09c7aaab4b072d73136641841cc2688616c49eb/serving/core/kv_cache_manager.py#L41-L176
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, infrastructure, kv-cache, prefix-caching]
---

# LLMServingSim tiered KV：一条 hash 链穿过 NPU、CPU 和 CXL / LLMServingSim Tiered KV: One Hash Chain Across NPU, CPU, and CXL

> **一句话 / In one line**: LLMServingSim 用同一条累积 block-hash 链同时查 NPU cache 和下层存储。 / LLMServingSim uses one cumulative block-hash chain to query both NPU cache and lower storage tiers.

## 为什么重要 / Why this matters

长上下文推理的瓶颈不只是算力，也包括 KV cache 迁移和恢复；模拟器如果把不同层级各自建一套 key，就很难忠实估算可恢复前缀和召回流量。

Long-context inference is limited not only by compute but also by KV-cache movement and restoration. If a simulator gives each tier its own key space, it becomes hard to faithfully estimate recoverable prefixes and recall traffic.

## 代码 / The code

`casys-kaist/LLMServingSim` - [`serving/core/kv_cache_manager.py`](https://github.com/casys-kaist/LLMServingSim/blob/b09c7aaab4b072d73136641841cc2688616c49eb/serving/core/kv_cache_manager.py#L41-L176)

```python
def request_block_hashes(req, block_size):
    """Chained block hashes over ``input_hash_ids + output_hash_ids``.

    Computed once and cached on the request. The simulator knows the whole token
    sequence up front, so unlike vLLM -- which extends the chain each time an
    output token is appended -- the entire chain can be built in one pass. That
    is not future information leaking into scheduling: indexing into the chain
    is gated by ``num_computed_tokens`` at every call site, so a block only
    becomes insertable after its tokens have actually been computed.

    Returns an empty list for workloads that carry no token ids, which disables
    prefix caching for those requests without disabling allocation.
    """
    cached = req.block_hashes
    if cached is not None:
        return cached

    tokens = req.input_hash_ids
    if not tokens:
        req.block_hashes = []
        return req.block_hashes
    if req.output_hash_ids:
        tokens = tokens + req.output_hash_ids

    hashes = []
    parent = NONE_HASH
    for start in range(0, len(tokens) - block_size + 1, block_size):
        parent = hash((parent, tuple(tokens[start:start + block_size])))
        hashes.append(parent)
    req.block_hashes = hashes
    return hashes


class TieredKVCacheManager:
    def _factor(self, pool):
        return pool.block_size // self.block_size

    def _coarse_hash(self, hashes, pool, coarse_idx):
        """Key for coarse block ``coarse_idx`` of ``pool``: the last fine hash
        it covers. ``offloading/scheduler.py::_get_block_hashes``."""
        factor = self._factor(pool)
        fine_idx = factor * coarse_idx + factor - 1
        if fine_idx >= len(hashes):
            return None
        return hashes[fine_idx]

    # -------------------- lookup --------------------

    def get_computed_blocks(self, req):
        """Longest recoverable prefix for ``req``, split by tier.

        Returns ``(npu_blocks, num_npu_hit_tokens, num_lower_hit_tokens)``.
        ``num_lower_hit_tokens`` are tokens whose KV has to be recalled into
        fresh NPU blocks; the recall bytes are charged in
        :meth:`allocate_slots`, once the allocation is known to succeed.
        """
        if not self.enable_caching:
            return [], 0, 0

        hashes = request_block_hashes(req, self.block_size)
        if not hashes:
            return [], 0, 0

        # vLLM caps the hit at num_tokens - 1: at least one token must be
        # recomputed to produce logits. Combined with block alignment this is
        # why a fully resident resume still costs up to one block.
        max_hit_len = req.num_tokens_reached - 1
        max_blocks = max_hit_len // self.block_size

        npu_blocks = []
        for block_hash in hashes[:max_blocks]:
            block = self.npu_pool.get_cached_block(block_hash)
            if block is None:
                # The chain is cumulative, so a miss means everything after it
                # is uncomputed or gone. Stop.
                break
            npu_blocks.append(block)
        num_npu_hit = len(npu_blocks) * self.block_size

        num_lower_hit = self._lookup_lower(req, hashes, num_npu_hit)
        return npu_blocks, num_npu_hit, num_lower_hit
```

## 逐行讲解 / What's happening

1. **第 41-71 行 / Lines 41-71 (`request_block_hashes`)**:
   - 中文: 每个 block 的 key 都包含上一个 block 的 hash，所以一处 miss 之后的后缀也不能当成可恢复前缀。
   - English: Each block key includes the previous block hash, so once a miss happens, the suffix after it is not a recoverable prefix.
2. **第 130-140 行 / Lines 130-140 (`_coarse_hash`)**:
   - 中文: 下层存储 block 更粗时，不重新 hash；直接用覆盖范围内最后一个 fine hash 当 coarse key。
   - English: When lower-tier blocks are coarser, the code does not rehash; it uses the last fine hash covered by the coarse block.
3. **第 144-163 行 / Lines 144-163 (lookup budget)**:
   - 中文: 命中上限是 `num_tokens_reached - 1`，因为至少要重算一个 token 来产出 logits。
   - English: The hit cap is `num_tokens_reached - 1` because at least one token must be recomputed to produce logits.
4. **第 165-176 行 / Lines 165-176 (tier split)**:
   - 中文: 先查 NPU 连续命中，再把已命中 token 数交给下层 lookup，得到还可召回多少。
   - English: It first walks contiguous NPU hits, then passes that token count to lower-tier lookup to compute additional recallable tokens.

## 类比 / The analogy

这像仓库用同一串递增箱号管理货架和地下库：地下库箱子更大，但仍按最后一个小箱号登记，所以两个系统不会对不上账。

It is like using one increasing box-number chain for both shelves and basement storage. Basement boxes are larger, but they are still registered by the last small-box number they contain, keeping both ledgers aligned.

## 自己跑一遍 / Try it yourself

```python
NONE = 0

def hashes(tokens, block):
    out, parent = [], NONE
    for i in range(0, len(tokens) - block + 1, block):
        parent = hash((parent, tuple(tokens[i:i+block])))
        out.append(parent)
    return out

h = hashes([1, 2, 3, 4, 5, 6, 7, 8], 2)
coarse_factor = 2
coarse_key = h[coarse_factor * 0 + coarse_factor - 1]
print(len(h), coarse_key == h[1])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
4 True
```

中文: coarse cache 没有单独 invent key，而是复用 fine 链上的边界 hash。

English: The coarse cache does not invent a separate key; it reuses the boundary hash from the fine chain.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **vLLM prefix cache** / **vLLM prefix cache**: vLLM 同样用连续 block hash 表达可复用前缀。 / vLLM also uses contiguous block hashes to represent reusable prefixes.
- **Storage hierarchy simulators** / **Storage hierarchy simulators**: 多层 cache 模拟通常需要一个共享地址空间。 / Multi-tier cache simulators usually need one shared address space.

## 注意事项 / Caveats / when it breaks

- **Python `hash` 非跨进程稳定** / **Python `hash` is not stable across processes**: 这里适合单次模拟；持久化 key 需要稳定 hash。 / This is fine for one simulation run; persistent keys need a stable hash.
- **只能恢复前缀** / **Only prefixes are recoverable**: 中间命中不能跳过前面 miss，因为注意力 KV 是顺序依赖的。 / A middle hit cannot skip an earlier miss because attention KV is order-dependent.

## 延伸阅读 / Further reading

- [LLMServingSim KV cache manager](https://github.com/casys-kaist/LLMServingSim/blob/b09c7aaab4b072d73136641841cc2688616c49eb/serving/core/kv_cache_manager.py#L41-L176)
