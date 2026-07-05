---
date: 2026-07-05
topic: infrastructure
source: trending
repo: sgl-project/sglang
file: python/sglang/srt/mem_cache/radix_cache.py
permalink: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, infrastructure, radix-tree, kv-cache]
---

# SGLang radix cache：用前缀树找最长可复用 KV / SGLang Radix Cache: Use a Prefix Tree to Find the Longest Reusable KV

> **一句话 / In one line**: KV cache 命中不是“全有或全无”，radix tree 可以返回最长共享 token 前缀。 / A KV-cache hit is not all-or-nothing; a radix tree can return the longest shared token prefix.

## 为什么重要 / Why this matters

在线推理里，很多请求共享 system prompt、工具描述或 few-shot 示例。SGLang 的 radix cache 把 token 序列按前缀组织成树，查询时走到最长匹配节点，并对命中节点加引用，避免正在使用的 KV 被释放。

In online inference, many requests share system prompts, tool descriptions, or few-shot examples. SGLang's radix cache organizes token sequences as a prefix tree, walks to the longest matching node, and increments references on the hit so active KV is not evicted.

## 代码 / The code

`sgl-project/sglang` — [`python/sglang/srt/mem_cache/radix_cache.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py)

```python
    def match_prefix(self, key, **kwargs):
        if self.disable or len(key) == 0:
            return [], self.root_node

        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            self._inc_lock_ref(last_node)
        return value, last_node

    def insert(self, key, value=None):
        if self.disable:
            return 0
        if value is None:
            value = [x for x in key]
        return self._insert_helper(self.root_node, key, value)
```

## 逐行讲解 / What's happening

1. **禁用或空 key 走根节点 / Disabled or empty keys return the root**: 中文: 没有前缀可查时，不制造假命中。 / English: With no searchable prefix, the cache avoids fake hits.
2. **helper 负责树遍历 / The helper walks the tree**: 中文: `_match_prefix_helper` 从 root 沿 token 边往下走，停在最长匹配节点。 / English: `_match_prefix_helper` starts at the root and follows token edges until the longest match stops.
3. **命中后加锁引用 / Hits increment a lock reference**: 中文: `value` 非空说明找到了可复用 KV，引用计数保护它不被 eviction 抢走。 / English: A non-empty `value` means reusable KV was found, and the reference count protects it from eviction.
4. **插入时默认 value 等于 key / Insert defaults value to key**: 中文: 如果调用方没有传 KV 位置列表，cache 至少能保存 token 序列本身的结构。 / English: If the caller does not pass KV indices, the cache can still store the token sequence structure.

## 类比 / The analogy

像通讯录自动补全：你输入“张三的公司电话”，系统不一定要完整匹配整句；只要“张三”这段前缀命中，就能先把相关条目拿出来。

It is like contact autocomplete. When you type "Alice office phone", the system does not need the whole phrase to match; the shared "Alice" prefix is already useful.

## 自己跑一遍 / Try it yourself

```python
class Node(dict):
    pass

root = Node()
def insert(tokens):
    n = root
    for t in tokens:
        n = n.setdefault(t, Node())
    n["$"] = tokens

def match(tokens):
    n, best = root, []
    for i, t in enumerate(tokens):
        if t not in n:
            break
        n = n[t]
        best = n.get("$", best)
    return best

insert([1, 2, 3])
print(match([1, 2, 3, 5]))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[1, 2, 3]
```

中文: 真实 radix cache 会处理边分裂和 KV 索引；这个 toy 版只展示“按前缀走树”的核心。

English: A real radix cache handles edge splitting and KV indices; this toy version only shows the core idea of walking a prefix tree.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **vLLM prefix cache** / **vLLM prefix cache**: 中文: 用 block hash 查共享前缀，粒度更偏 KV block。 / English: It finds shared prefixes with block hashes, at a KV-block granularity.
- **Browser URL trie** / **Browser URL trie**: 中文: 输入一部分路径时，前缀树很适合找候选。 / English: A prefix tree is natural for candidates from partial URL paths.

## 注意事项 / Caveats / when it breaks

- **共享越少收益越小 / Less sharing means less benefit**: 中文: 如果请求 prompt 完全不同，radix tree 只是额外索引开销。 / English: If prompts are unrelated, the radix tree is mostly indexing overhead.
- **引用计数必须严格 / Reference counts must be strict**: 中文: 命中后没加锁，或释放时没减锁，都会导致泄漏或误删。 / English: Missing increments or decrements after hits can cause leaks or accidental eviction.

## 延伸阅读 / Further reading

- SGLang radix cache source linked above.
- SGLang runtime documentation.
