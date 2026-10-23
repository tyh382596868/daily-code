---
date: 2026-10-23
topic: infrastructure
source: tracked
repo: vllm-project/vllm
file: vllm/v1/core/block_pool.py
permalink: https://github.com/vllm-project/vllm/blob/58cf73223e3ef8f69cff7c8e5ffe44e731d4658b/vllm/v1/core/block_pool.py#L28-L107
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, infrastructure, kv-cache]
---

# vLLM 前缀块索引：一个 hash 不一定只指向一块 / vLLM Prefix Block Index: One Hash Can Name Several Blocks

> **一句话 / In one line**: vLLM 用「单对象或小字典」表示 hash 到物理 KV block 的一对多关系，常见路径不付字典成本。 / vLLM represents a hash-to-physical-block one-to-many relation as either one object or a tiny dictionary, keeping the common path cheap.

## 为什么重要 / Why this matters

前缀缓存按内容 hash 查 KV block；同一段 token 可以在不同物理 block 中出现。这个小容器既保留 append-only block ID 的约束，也避免每个 hash 都分配字典。

Prefix caching looks up KV blocks by content hash, while equal token prefixes can live in multiple physical blocks. This container preserves append-only IDs without allocating a dictionary for every key.

## 代码 / The code

`vllm-project/vllm` — [`block_pool.py`](https://github.com/vllm-project/vllm/blob/58cf73223e3ef8f69cff7c8e5ffe44e731d4658b/vllm/v1/core/block_pool.py#L45-L106)

```python
def insert(self, key, block):
    blocks = self._cache.get(key)
    if blocks is None:
        self._cache[key] = block
    elif isinstance(blocks, KVCacheBlock):
        self._cache[key] = {blocks.block_id: blocks, block.block_id: block}
    elif isinstance(blocks, dict):
        blocks[block.block_id] = block

def pop(self, key, block_id):
    blocks = self._cache.pop(key, None)
    if blocks is None:
        return None
    if isinstance(blocks, KVCacheBlock):
        if blocks.block_id == block_id:
            return blocks
        self._cache[key] = blocks
        return None
    if isinstance(blocks, dict):
        block = blocks.pop(block_id, None)
        if len(blocks) > 0:
            self._cache[key] = blocks
        return block
```

## 逐行讲解 / What's happening

`insert` 先走单 block 快路径；第二次冲突才升级成以 `block_id` 为键的字典。`pop` 先移出整个槽位，再在未命中或仍有成员时放回，保证 map 不留下空字典。

`insert` takes the single-block fast path and promotes only on a second collision. `pop` removes the slot first, then restores it on a miss or when members remain, so the map never retains empty dictionaries.

## 类比 / The analogy

像图书馆索引卡：多数书名只有一册，只放一张卡；有复本时才换成写着馆藏编号的小卡盒。

Think of a library catalogue: most titles need one card; only duplicate copies warrant a small box keyed by copy ID.

## 自己跑一遍 / Try it yourself

```python
cache = {}
def put(k, block):
    old = cache.get(k)
    cache[k] = block if old is None else ({old[0]: old, block[0]: block} if isinstance(old, tuple) else old | {block[0]: block})
put("abc", (3, "A")); put("abc", (8, "B"))
print(cache["abc"])
```

运行 / Run with: `python try.py`

预期输出 / Expected output: `{3: (3, 'A'), 8: (8, 'B')}`。

这个升级策略把少数复杂情况隔离起来。 / The promotion isolates the rare complex case.

## 注意事项 / Caveats / when it breaks

- hash 相等不是内容绝对相等；生产实现仍要采用正确的 hash key 和生命周期管理。 / Hash equality is not absolute content equality; production still needs correct keys and lifecycle management.

## 延伸阅读 / Further reading

- [vLLM block pool source](https://github.com/vllm-project/vllm/blob/58cf73223e3ef8f69cff7c8e5ffe44e731d4658b/vllm/v1/core/block_pool.py)
