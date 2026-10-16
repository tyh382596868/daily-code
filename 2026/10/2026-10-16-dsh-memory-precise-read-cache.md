---
date: 2026-10-16
topic: robotics
source: trending
repo: FuRongJun-1999/dsh-memory
file: md_cg/readcache.py
permalink: https://github.com/FuRongJun-1999/dsh-memory/blob/4cc069b3c608ca68fd9efb249343f1c7c46d185d/md_cg/readcache.py#L123-L190
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, robotics, cache-invalidation]
---

# dsh-memory 读缓存：只让被写的路径失效 / dsh-memory Read Cache: Invalidate Only the Path That Changed

> **一句话 / In one line**: 缓存用每路径代际判断新鲜度，避免单节点写入把整个记忆池都赶出缓存。 / The cache uses per-path generations for freshness, preventing one write from evicting the whole memory pool.

## 为什么重要 / Why this matters

知识或机器人记忆系统常常读多写少。全局版本号很安全，却会让一次局部更新制造全量 cache miss；精确失效保住了正确性和热读性能。

Knowledge and robot-memory systems are commonly read-heavy. A global version is safe but turns a local update into a full cache miss; precise invalidation preserves both correctness and hot-read performance.

## 代码 / The code

`FuRongJun-1999/dsh-memory` — [`readcache.py`](https://github.com/FuRongJun-1999/dsh-memory/blob/4cc069b3c608ca68fd9efb249343f1c7c46d185d/md_cg/readcache.py#L123-L190)

```python
def _fresh(p, hit):
    dirty = cg._dirty
    pg = getattr(dirty, "path_gen", None)
    if pg is None:
        return hit[0] == getattr(dirty, "write_gen", len(dirty))
    return (pg.get(p, 0) <= hit[0]
            and dirty.broad_gen <= hit[0])
```

## 逐行讲解 / What's happening

1. 中文: cache hit 带着写入时的 generation；同一路径之后被标脏才会失效。
   English: A hit carries the generation at which it was stored; only a later dirty mark for that path invalidates it.
2. 中文: `broad_gen` 是无法归因到单一路径时的保守兜底，例如整体重建。
   English: `broad_gen` is the conservative fallback for changes that cannot be attributed to one path, such as a rebuild.
3. 中文: 没有新接口时退回全局 generation，兼容旧的 dirty-set 实现。
   English: When the new interface is absent, it falls back to a global generation for compatibility with older dirty sets.

## 类比 / The analogy

像图书馆给每本书贴归还日期：一本书被借走，不需要把所有书架上的“可借”牌都翻掉。

It is a library tracking the return date per book: one checkout does not require flipping every shelf’s availability sign.

## 自己跑一遍 / Try it yourself

```python
path_gen, broad_gen = {"a": 3}, 0
def fresh(path, cached_gen):
    return path_gen.get(path, 0) <= cached_gen and broad_gen <= cached_gen

print(fresh("a", 2), fresh("b", 2))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
False True
```

中文: 写 a 只淘汰 a，b 仍可复用。  
English: Writing `a` evicts only `a`; `b` remains reusable.

## 注意事项 / Caveats / when it breaks

- 中文: 所有写路径都必须正确标脏，否则会读到陈旧值。 / English: Every write path must mark dirty correctly or stale values escape.
- 中文: 不可归因的批量更新应推进 `broad_gen`。 / English: Unattributable batch updates must advance `broad_gen`.

## 延伸阅读 / Further reading

- [dsh-memory read cache](https://github.com/FuRongJun-1999/dsh-memory/blob/4cc069b3c608ca68fd9efb249343f1c7c46d185d/md_cg/readcache.py#L123-L190)
