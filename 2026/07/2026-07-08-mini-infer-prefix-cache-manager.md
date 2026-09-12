---
date: 2026-07-08
topic: infrastructure
source: trending
repo: psmarter/mini-infer
file: mini_infer/cache/kv_cache.py
permalink: https://github.com/psmarter/mini-infer/blob/85a4bd2a2b40cf593bad325b8ea634f3e734c4a2/mini_infer/cache/kv_cache.py#L55-L143
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, infrastructure, kv-cache, prefix-cache]
---

# mini-infer PrefixCacheManager：链式 hash 让前缀缓存不串台 / mini-infer PrefixCacheManager: Chained Hashes Keep Prefix Cache Honest

> **一句话 / In one line**: `PrefixCacheManager` 用“前一块 hash + 当前 token block”计算链式 hash，避免相同 token block 在不同上下文里误命中。 / `PrefixCacheManager` hashes each token block together with the previous block hash, preventing identical blocks in different contexts from colliding semantically.

## 为什么重要 / Why this matters

LLM serving 里，前缀缓存能复用 prompt 的 KV cache。但只按当前 block 的 token 算 hash 有风险：同一段 token 出现在不同前缀后面，语义上下文并不相同。mini-infer 这段实现虽然小，但抓住了核心：block hash 要包含前缀历史，并且 LRU 淘汰时要看 ref count。

In LLM serving, prefix caching reuses KV cache for prompt prefixes. Hashing only the current block is risky: the same token block can appear after different prefixes and mean different things. This mini-infer implementation is small, but it captures the key idea: block hashes must include prefix history, and LRU eviction must respect ref counts.

## 代码 / The code

`psmarter/mini-infer` — [`mini_infer/cache/kv_cache.py`](https://github.com/psmarter/mini-infer/blob/85a4bd2a2b40cf593bad325b8ea634f3e734c4a2/mini_infer/cache/kv_cache.py#L55-L143)

```python
class PrefixCacheManager:
    def __init__(self, block_size: int) -> None:
        self.block_size = block_size
        # hash -> phys_block_id
        self._cache: dict[int, int] = {}
        # LRU order, most recent at the end
        self._lru: OrderedDict[int, None] = OrderedDict()

    def compute_hashes(self, token_ids: list[int]) -> list[int]:
        num_full_blocks = len(token_ids) // self.block_size
        max_cacheable = (num_full_blocks - 1) if (
            len(token_ids) > 0 and len(token_ids) % self.block_size == 0
        ) else num_full_blocks

        hashes: list[int] = []
        prev_hash = 0
        for i in range(max_cacheable):
            start = i * self.block_size
            end = start + self.block_size
            buf = struct.pack(
                f">Q{end - start}i",
                prev_hash & 0xFFFFFFFFFFFFFFFF,
                *token_ids[start:end],
            )
            block_hash = int.from_bytes(hashlib.sha256(buf).digest()[:8], "big")
            hashes.append(block_hash)
            prev_hash = block_hash
        return hashes

    def find(self, token_ids: list[int]) -> tuple[int, list[int]]:
        hashes = self.compute_hashes(token_ids)
        cached_blocks: list[int] = []
        for block_hash in hashes:
            if block_hash not in self._cache:
                break
            cached_blocks.append(self._cache[block_hash])
            self._lru.move_to_end(block_hash)
        return len(cached_blocks) * self.block_size, cached_blocks

    def register(self, token_ids: list[int], block_table: list[int]) -> list[int]:
        hashes = self.compute_hashes(token_ids)
        new_blocks: list[int] = []
        for block_hash, phys_block in zip(hashes, block_table):
            if block_hash in self._cache:
                self._lru.move_to_end(block_hash)
                continue
            self._cache[block_hash] = phys_block
            self._lru[block_hash] = None
            new_blocks.append(phys_block)
        return new_blocks
```

## 逐行讲解 / What's happening

1. **第 68-72 行 / Lines 68-72 (`max_cacheable`)**:
   - 中文: 如果 prompt 长度刚好整除 block size，就故意少缓存最后一块，保证后续 suffix 不是空的。
   - English: If prompt length is exactly block-aligned, it intentionally skips the last block so the suffix is non-empty.
2. **第 76-86 行 / Lines 76-86 (链式 hash)**:
   - 中文: `prev_hash` 被打包进当前 block 的 hash 输入，所以第 N 块的 key 隐含了前 N-1 块。
   - English: `prev_hash` is packed into the current block hash, so block N's key includes blocks 0 through N-1.
3. **第 89-97 行 / Lines 89-97 (`find`)**:
   - 中文: 查找最长连续命中，一旦某个 block miss，后面的 block 也不能复用。
   - English: It finds the longest contiguous prefix hit; after one miss, later blocks cannot be reused.
4. **第 99-109 行 / Lines 99-109 (`register`)**:
   - 中文: 新 block 写入 cache，已有 block 只刷新 LRU，不重复注册。
   - English: New blocks enter the cache; existing blocks only refresh their LRU position.

## 类比 / The analogy

这像给书的每一页盖章：章上不仅有本页内容，还有上一页章的编号。即使两页文字相同，只要前文不同，章也不同。

It is like stamping each page of a book with both the page text and the previous page's stamp. Two identical pages get different stamps if their histories differ.

## 自己跑一遍 / Try it yourself

```python
import hashlib, struct

def hashes(tokens, block=2):
    out, prev = [], 0
    for i in range(len(tokens) // block):
        chunk = tokens[i * block:(i + 1) * block]
        buf = struct.pack(f">Q{block}i", prev, *chunk)
        prev = int.from_bytes(hashlib.sha256(buf).digest()[:8], "big")
        out.append(prev)
    return out

print(hashes([1, 2, 3, 4]))
print(hashes([9, 9, 3, 4]))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[294502332756174867, 16260699825610302236]
[16331119929143643461, 6944347769374929427]
```

第二个 block token 都是 `[3, 4]`，但前缀不同，所以第二个 hash 也不同。

The second block tokens are `[3, 4]` in both cases, but the prefixes differ, so the second hash differs too.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **vLLM prefix cache** / **vLLM prefix cache**: 同样需要区分逻辑 token 前缀和物理 KV block。 / It also separates logical token prefixes from physical KV blocks.
- **Merkle tree** / **Merkle tree**: 父节点 hash 包含子节点 hash，历史被压进一个固定长度摘要。 / Parent hashes include child hashes, compressing history into fixed-size digests.

## 注意事项 / Caveats / when it breaks

- **hash 不是权限系统** / **A hash is not an authorization system**: 它降低碰撞概率，但真正生产系统还要考虑隔离、租户和 cache policy。 / It reduces collision probability, but production systems still need isolation, tenancy, and cache policy.
- **LRU 要配合 ref count** / **LRU must respect ref counts**: 正在被请求使用的 block 不能因为“最老”就被释放。 / A block in use by a request cannot be freed just because it is old.

## 延伸阅读 / Further reading

- [mini-infer `PrefixCacheManager`](https://github.com/psmarter/mini-infer/blob/85a4bd2a2b40cf593bad325b8ea634f3e734c4a2/mini_infer/cache/kv_cache.py#L55-L143)
