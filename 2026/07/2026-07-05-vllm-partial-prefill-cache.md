---
date: 2026-07-05
topic: infrastructure
source: tracked
repo: vllm-project/vllm
file: vllm/v1/core/block_pool.py
permalink: https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, infrastructure, kv-cache, prefix-cache]
---

# vLLM partial prefill cache：没填满的 block 也值得记住 / vLLM Partial Prefill Cache: Even an Unfilled Block Can Be Worth Remembering

> **一句话 / In one line**: prefix cache 不只缓存完整 KV block，也可以给正在增长的 partial block 建索引，减少长 prompt prefill 的重复劳动。 / Prefix caching can index a growing partial block, not only fully sealed KV blocks, so long-prompt prefill does less repeated work.

## 为什么重要 / Why this matters

LLM serving 里，prompt 共享经常不是整齐地落在 block 边界上。vLLM 的 block pool 把 hash、block id、LRU 状态放在一起管理，让 partial block 也能进入缓存表；下次相同前缀再来时，可以从更细的边界开始复用。

In LLM serving, shared prompts rarely stop exactly at a block boundary. vLLM's block pool keeps the hash, block id, and LRU state together so a partial block can also be indexed; the next request with the same prefix can reuse work from a finer boundary.

## 代码 / The code

`vllm-project/vllm` — [`vllm/v1/core/block_pool.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py)

```python
    def cache_partial_block(
        self,
        block: KVCacheBlock,
        block_size: int,
        kv_cache_group_id: int,
        block_hash: BlockHashType,
    ) -> Optional[KVCacheBlock]:
        if block.is_null:
            return None

        assert block_size > self.hash_block_size
        if block_hash in self.cached_block_hash_to_block:
            return self.cached_block_hash_to_block[block_hash]

        block.block_hash = block_hash
        block.kv_cache_group_id = kv_cache_group_id
        self.cached_block_hash_to_block[block_hash] = block
        return block
```

## 逐行讲解 / What's happening

1. **空 block 直接跳过 / Skip null blocks**: 中文: `is_null` 表示这个 block 没有可复用的 KV 内容，给它建索引只会污染缓存表。 / English: `is_null` means the block has no reusable KV content; indexing it would only pollute the cache table.
2. **partial block 有自己的粒度 / Partial blocks have their own granularity**: 中文: `block_size > hash_block_size` 表示这里处理的是“比完整块更细”的前缀边界。 / English: `block_size > hash_block_size` marks this as a prefix boundary finer than the regular full-block hash.
3. **先查重再登记 / Deduplicate before registering**: 中文: 同一个 `block_hash` 已经存在时复用旧 block，避免两个 block id 指向同一段语义前缀。 / English: If the same `block_hash` already exists, the pool returns the existing block so two block ids do not represent the same semantic prefix.
4. **hash 和 group 一起存 / Store hash with group metadata**: 中文: `kv_cache_group_id` 把缓存归到具体 KV cache group，后续调度和释放才能找到正确池子。 / English: `kv_cache_group_id` ties the cached entry to the right KV-cache group so scheduling and eviction hit the correct pool.

## 类比 / The analogy

像图书馆给书架贴标签：整层书架当然要贴，但如果一个读者经常只借到某一格，给那一格也贴小标签，下次就不用从整层开始找。

It is like labeling shelves in a library. A whole shelf deserves a label, but if readers often stop at one cubby, labeling that cubby saves the next search from starting at the shelf level.

## 自己跑一遍 / Try it yourself

```python
class Block:
    def __init__(self, is_null=False):
        self.is_null = is_null
        self.block_hash = None

cache = {}
def cache_partial(block, h):
    if block.is_null:
        return None
    if h in cache:
        return cache[h]
    block.block_hash = h
    cache[h] = block
    return block

a, b = Block(), Block()
print(cache_partial(a, "prefix-128") is a)
print(cache_partial(b, "prefix-128") is a)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
True
True
```

中文: 第二次传入的是新对象，但返回的是已缓存的旧 block，这就是 prefix cache 的去重核心。

English: The second call passes a new object, but the function returns the already cached block. That is the deduplication core of a prefix cache.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **SGLang radix cache** / **SGLang radix cache**: 中文: 用树结构按 token 前缀复用 KV。 / English: It reuses KV by walking a token-prefix tree.
- **Transformers StaticCache** / **Transformers StaticCache**: 中文: 预分配 KV 空间，减少 decode 时重新分配。 / English: It preallocates KV space to avoid decode-time reallocations.

## 注意事项 / Caveats / when it breaks

- **hash 必须稳定 / Hashes must be stable**: 中文: hash 输入要包含 token、cache group 和相关配置，否则会把不兼容 KV 混在一起。 / English: The hash input must include tokens, cache group, and relevant configuration, or incompatible KV can collide.
- **释放策略更复杂 / Eviction becomes more complex**: 中文: partial block 数量多，LRU 和引用计数必须跟上。 / English: Partial blocks increase entry count, so LRU and reference counts must stay correct.

## 延伸阅读 / Further reading

- vLLM block pool source linked above.
- vLLM prefix caching documentation.
