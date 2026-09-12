---
date: 2026-08-23
topic: infrastructure
source: tracked
repo: vllm-project/vllm
file: vllm/v1/core/block_pool.py
permalink: https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py#L411-L527
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, infrastructure, prefix-cache, kv-cache]
---

# vLLM partial block cache：半块前缀也能命中 / vLLM Partial Block Cache: Even Half a Block Can Hit

> **一句话 / In one line**: vLLM 给一个已有 KV block 增加更细粒度的 hash 入口，让尚未填满整个 block 的前缀也能被复用。 / vLLM adds a finer-grained hash entry to an existing KV block so prefixes inside a larger block can still be reused.

## 为什么重要 / Why this matters

KV cache 通常按固定 block 管理。问题是有些模型的物理 cache block 比 hash block 更大：请求可能只共享了一个大 block 中间的前半段。如果只能等整块写满再缓存，就会错过很多 prefix-cache 命中。

KV caches are usually managed in fixed blocks. The tricky case is when the physical cache block is larger than the hash block: two requests may share only the early part of a larger block. If caching waits for the whole block to fill, many prefix hits are lost.

## 代码 / The code

`vllm-project/vllm` — [`vllm/v1/core/block_pool.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py#L411-L527)

```python
def cache_partial_block(
    self,
    request: Request,
    block: KVCacheBlock,
    num_tokens: int,
    kv_cache_group_id: int,
    block_size: int,
) -> BlockHashWithGroupId | None:
    """Register a partial prefix-cache entry for an existing block."""
    if block.is_null:
        return None
    assert block_size > self.hash_block_size
    assert block_size % self.hash_block_size == 0
    assert num_tokens % block_size != 0
    block_hash = self._get_partial_block_hash(request, num_tokens)
    num_hash_blocks = num_tokens // self.hash_block_size
    block_hash_with_group_id = make_block_hash_with_group_id(
        block_hash, kv_cache_group_id
    )
    already_cached = block.block_hash == block_hash_with_group_id or (
        self.cached_block_hash_to_block.contain(
            block_hash_with_group_id, block.block_id
        )
    )
    if (
        not already_cached
        and block.block_hash is not None
        and block.block_hash_num_tokens is not None
        and block.block_hash_num_tokens < num_hash_blocks * self.hash_block_size
    ):
        removed_hashes = self._remove_cached_block_hashes(block)
        self._emit_block_removed_events(removed_hashes)
    self._insert_block_hash(
        block_hash_with_group_id,
        block,
        num_tokens=num_hash_blocks * self.hash_block_size,
    )
    if self.enable_kv_cache_events and not already_cached:
        parent_hash, block_start = self._get_partial_block_parent_hash_and_start(
            request, num_tokens
        )
        parent_block_hash = (
            maybe_convert_block_hash(parent_hash)
            if parent_hash is not None
            else None
        )
        block_end = num_tokens
        curr_mm_idx = -1 if block_start > 0 else 0
        extra_keys, _ = generate_block_hash_extra_keys(
            request, block_start, block_end, curr_mm_idx
        )
        self.kv_event_queue.append(
            BlockStored(
                block_hashes=[maybe_convert_block_hash(block_hash)],
                parent_block_hash=parent_block_hash,
                token_ids=request.all_token_ids[block_start:block_end],
                block_size=block_end - block_start,
                lora_id=request.lora_request.adapter_id
                if request.lora_request
                else None,
                medium=MEDIUM_GPU,
                lora_name=request.lora_request.name
                if request.lora_request
                else None,
                extra_keys=[extra_keys],
                group_idx=kv_cache_group_id,
            )
        )
    return block_hash_with_group_id
```

## 逐行讲解 / What's happening

1. **第 446-450 行 / Lines 446-450**:
   - 中文: 先排除 null block，再断言这是“大物理 block、小 hash block”的场景，并且 `num_tokens` 不是整块边界。
   - English: It rejects the null block, then asserts the intended shape: a larger physical block, smaller hash blocks, and a prefix boundary that is not a full block boundary.
2. **第 451-460 行 / Lines 451-460**:
   - 中文: 从请求的 prefix hash 链里取出 `num_tokens` 对应的 hash，再加上 KV group id，形成真正的 cache key。
   - English: It reads the hash at the requested prefix boundary, combines it with the KV group id, and gets the real cache key.
3. **第 461-473 行 / Lines 461-473**:
   - 中文: 如果旧 hash 表示的 token 更少，就先移除旧入口；随后把这个 block 挂到新的 partial hash 上。
   - English: If the block already had a shorter hash entry, the old entries are removed before the block is attached to the new partial hash.
4. **第 474-504 行 / Lines 474-504**:
   - 中文: 如果启用了 KV cache event，vLLM 还会发一个 `BlockStored`，告诉外部消费者这个 partial prefix 的 token 范围和 parent hash。
   - English: With KV cache events enabled, vLLM emits a `BlockStored` event that records the token span and parent hash for this partial prefix.

## 类比 / The analogy

这像图书馆给一本厚书加书签。整本书已经在书架上，但读者常常只需要第 1 到第 3 章。与其复印半本书，图书馆只给第 3 章末尾贴一个索引卡，下次有人查同样前缀时直接找到原书。

Think of a library adding bookmarks to a thick book. The whole book is already on the shelf, but readers often need only chapters 1-3. Instead of copying half the book, the library adds an index card at the chapter boundary and points future readers to the same physical book.

## 自己跑一遍 / Try it yourself

```python
hash_block_size = 4
request_hashes = ["h4", "h8", "h12"]
cache = {}
block = {"id": 7, "primary": None, "extra": set()}

def partial_hash(num_tokens):
    assert num_tokens % hash_block_size == 0
    return request_hashes[num_tokens // hash_block_size - 1]

def insert(num_tokens, group):
    key = (partial_hash(num_tokens), group)
    if block["primary"] is None:
        block["primary"] = key
    else:
        block["extra"].add(key)
    cache[key] = block["id"]

insert(8, group=0)
insert(12, group=0)
print(cache)
print(block)
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
{('h8', 0): 7, ('h12', 0): 7}
{'id': 7, 'primary': ('h8', 0), 'extra': {('h12', 0)}}
```

一个物理 block 可以有多个 hash 入口；这就是 partial cache 的核心。

One physical block can be reachable through multiple hash keys. That is the core idea behind partial caching.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **SGLang radix cache** / **SGLang radix cache**: 用前缀树表达可复用 token 前缀。 / It represents reusable token prefixes as a radix tree.
- **LMCache chunk keys** / **LMCache chunk keys**: 用更细粒度的 chunk address 复用 KV。 / It uses finer-grained chunk addresses for KV reuse.
- **浏览器 HTTP cache range** / **Browser HTTP cache ranges**: 同一个文件可以按 byte range 命中。 / A single file can be served through byte-range cache hits.

## 注意事项 / Caveats / when it breaks

- **hash 粒度要一致** / **Hash granularity must be consistent**: `num_tokens` 必须落在 `hash_block_size` 边界上。 / `num_tokens` must align to `hash_block_size`.
- **事件只是通知** / **Events are only notifications**: cache 正确性仍然来自 hash map 和 block 元数据。 / Cache correctness still comes from the hash map and block metadata.
- **eviction 要清理所有入口** / **Eviction must clean every entry**: 一个 block 多个 hash，删除时必须全部移除。 / A block may have multiple hashes, so eviction must remove all of them.

## 延伸阅读 / Further reading

- vLLM `BlockPool` source: https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py
- vLLM prefix caching docs: https://docs.vllm.ai/
