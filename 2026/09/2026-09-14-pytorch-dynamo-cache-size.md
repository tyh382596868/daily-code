---
date: 2026-09-14
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/_dynamo/cache_size.py
permalink: https://github.com/pytorch/pytorch/blob/0bed4e802fa3454f7609a409f0c5f828060726f8/torch/_dynamo/cache_size.py#L72-L175
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, pytorch, dynamo, recompilation, cache, weakref]
---

# PyTorch Dynamo 重编译缓存：局部限额和全局刹车 / PyTorch Dynamo Cache Size: Local Limits and a Global Brake

> **一句话 / In one line**: Dynamo 同时统计同一组 `ID_MATCH` 对象的缓存项和整个 code object 的缓存项，既允许合理变化，也防止重编译失控。 / Dynamo tracks cache entries for the same `ID_MATCH` objects and across the whole code object, allowing useful variation while stopping runaway recompilation.

## 为什么重要 / Why this matters

中文：`torch.compile` 遇到新输入或新模块实例时可能需要重新编译。只设一个全局上限会太粗：16 个合法的 module 实例可能各自需要一份缓存，但一个真正动态的函数也可能用同样的次数把缓存撑爆。`cache_size.py` 的核心想法是把“同一对象的重复编译压力”和“所有区域累计的安全上限”拆开计数。

English: `torch.compile` may recompile when inputs or module instances change. A single global limit is too blunt: sixteen legitimate module instances may each need a cache entry, while one highly dynamic function could exhaust the same budget by itself. This module separates per-object recompilation pressure from a global safety cap.

## 代码 / The code

`pytorch/pytorch` — [`torch/_dynamo/cache_size.py`](https://github.com/pytorch/pytorch/blob/0bed4e802fa3454f7609a409f0c5f828060726f8/torch/_dynamo/cache_size.py#L72-L175)

```python
@dataclass
class CacheSizeRelevantForFrame:
    """
    We track the number of cache entries that have same id_match objects as the
    given frame.

    TODO(janimesh) - Consider adding a map from tuple_of_match_ids to count -
    https://github.com/pytorch/pytorch/pull/107496#discussion_r1304564682 - this
    could be useful for debugging as well.
    """

    # Number of CacheEntry objects in this region's cache list
    num_cache_entries: int = 0

    # Number of CacheEntry objects having same ID_MATCH'd objects as given frame.
    num_cache_entries_with_same_id_matched_objs: int = 0

    # Total cache entries across ALL regions on this code object.
    # Used for accumulated_recompile_limit which is a global safety cap.
    total_cache_entries_all_regions: int = 0

    def will_compilation_exceed(self, limit: int) -> bool:
        # Checks if a compilation will exceed the given limit (that's why >=).
        return (
            self.will_compilation_exceed_accumulated_limit()
            or self.will_compilation_exceed_specific_limit(limit)
        )

    def will_compilation_exceed_accumulated_limit(self) -> bool:
        # accumulated_recompile_limit is a global safety cap across all regions.
        return (
            self.total_cache_entries_all_regions >= config.accumulated_recompile_limit
        )

    def will_compilation_exceed_specific_limit(self, limit: int) -> bool:
        return self.num_cache_entries_with_same_id_matched_objs >= limit


def _get_weakref_from_f_locals(
    frame: DynamoFrameType, local_name: str
) -> weakref.ref[Any] | None:
    obj = frame.f_locals.get(local_name, None)
    weak_id = None
    try:
        weak_id = weakref.ref(obj)
    except TypeError:
        pass  # cannot weakref bool object
    return weak_id


def _has_same_id_matched_objs(frame: DynamoFrameType, cache_entry: Any) -> bool:
    """
    Checks if the ID_MATCH'd objects saved on cache_entry are same as the ones
    in frame.f_locals.
    """
    if not cache_entry:
        return False

    for (
        local_name,
        weakref_from_cache_entry,
    ) in cache_entry.guard_manager.id_matched_objs.items():
        if weakref_from_cache_entry() is not None:
            weakref_from_frame = _get_weakref_from_f_locals(frame, local_name)
            if weakref_from_frame is not weakref_from_cache_entry:
                return False

    # Also covers the case where no ID_MATCH objects are saved in frame.f_locals
    return True


def compute_cache_size(
    frame: DynamoFrameType,
    cache_entries: list[Any],
    total_cache_entries_all_regions: int = 0,
) -> CacheSizeRelevantForFrame:
    # cache_entries is already scoped to a single isolate_recompiles region.
    # recompile_limit is checked per-region. accumulated_recompile_limit uses
    # total_cache_entries_all_regions as a global safety cap.
    num_cache_entries = 0
    num_cache_entries_with_same_id_matched_objs = 0

    for cache_entry in cache_entries:
        num_cache_entries += 1
        if _has_same_id_matched_objs(frame, cache_entry):
            num_cache_entries_with_same_id_matched_objs += 1

    return CacheSizeRelevantForFrame(
        num_cache_entries,
        num_cache_entries_with_same_id_matched_objs,
        max(total_cache_entries_all_regions, num_cache_entries),
    )


def is_recompilation(cache_size: CacheSizeRelevantForFrame) -> bool:
    """
    If the frame (earlier parsed by compute_cache_size) has more than 1 cache
    entry with same ID_MATCH'd objects, then its a recompilation.
    """
    # Note that you can have multiple entries in the cache but still not a
    # recompile, e.g., you can have 64 nn module instances, each one having an
    # ID_MATCH guard, and each one having just 1 cache entry in the cache.  In
    # this case, we can have 64 entries in the cache, but no recompilation
    # because there is only one entry for each id_matched_obj.
    return cache_size.will_compilation_exceed(1)
```

## 逐行讲解 / What's happening

1. **第 72-90 行 / Lines 72-90 (`CacheSizeRelevantForFrame`)**:
   - 中文：三个计数分别回答三个问题：当前 region 有多少缓存、和当前对象匹配的有多少、整个 code object 总共有多少。
   - English: The three counters answer three questions: how many entries are in this region, how many match the current objects, and how many exist across the whole code object.
2. **第 92-106 行 / Lines 92-106 (`will_compilation_exceed`)**:
   - 中文：局部限制和累计限制用 `or` 合并；`>=` 表示本次新编译一旦会触碰上限，就先刹车。
   - English: Local and accumulated limits are combined with `or`; `>=` means a new compilation is rejected as soon as it would reach the limit.
3. **第 109-118 行 / Lines 109-118 (`_get_weakref_from_f_locals`)**:
   - 中文：Dynamo 不把局部对象强引用进缓存，而是尝试用 weak reference 保存身份。像 `bool` 这样的对象不可 weakref 时，函数返回 `None`。
   - English: Dynamo avoids keeping locals alive through the cache and tries to represent identity with weak references. Objects such as `bool` may not support weak references, so the helper returns `None`.
4. **第 121-139 行 / Lines 121-139 (`_has_same_id_matched_objs`)**:
   - 中文：代码比较的是 weakref 对象身份，而不只是底层对象相等。这是在判断“是不是同一个 module instance”。
   - English: The comparison is between weak-reference identities, not merely value equality. The question is whether this is the same module instance.
5. **第 142-162 行 / Lines 142-162 (`compute_cache_size`)**:
   - 中文：函数遍历当前 region 的 cache entries，同时累计匹配数量；总量用 `max` 合并 region 局部观察值和外部传入的累计值。
   - English: The function walks the current region's entries, counts matching identities, and carries the global total with `max` across the local observation and the caller's accumulated value.
6. **第 165-175 行 / Lines 165-175 (`is_recompilation`)**:
   - 中文：一次匹配对象的第一次缓存不是重编译；同一组对象出现第二份匹配缓存时，才算进入 recompilation pressure。
   - English: The first cache entry for an object is not a recompilation. A second matching entry is what creates recompilation pressure.

## 类比 / The analogy

中文：像酒店的房卡系统。不同客人可以各自拿到一张房卡，这是合法的局部变化；但同一位客人在同一间房不断申请新卡，说明系统可能出了问题。前台还要设置整栋酒店的总发卡上限，防止任何一层把库存耗尽。

English: Imagine a hotel key-card system. Different guests can each receive a card, which is legitimate variation. But the same guest repeatedly requesting new cards for the same room signals trouble. The front desk also needs a building-wide cap so one floor cannot exhaust the inventory.

## 自己跑一遍 / Try it yourself

```python
def cache_pressure(total, same_object, specific_limit=2, global_limit=5):
    hit_specific = same_object >= specific_limit
    hit_global = total >= global_limit
    return hit_specific or hit_global


print(cache_pressure(total=3, same_object=1))
print(cache_pressure(total=3, same_object=2))
print(cache_pressure(total=5, same_object=1))
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
False
True
True
```

中文：第二行展示局部 `ID_MATCH` 限制，第三行展示全局累计限制；两条规则服务的是不同故障模式。

English: The second line shows the per-identity limit, while the third shows the accumulated limit. The two rules protect against different failure modes.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **TorchDynamo guard cache** / **TorchDynamo guard cache**: guards decide whether an existing compiled entry can be reused. / Guards decide whether an existing compiled entry can be reused.
- **HTTP cache keys** / **HTTP cache keys**: per-key churn and total cache size need separate policies. / Per-key churn and total cache size need separate policies.
- **GPU compilation caches** / **GPU compilation caches**: many shapes may be valid, but a runaway shape space still needs a global brake. / Many shapes may be valid, but a runaway shape space still needs a global brake.

## 注意事项 / Caveats / when it breaks

- **对象身份不是值相等** / **Identity is not value equality**: 两个结构相同的 module 仍可能需要两份 guarded cache。 / Two structurally identical modules may still need separate guarded cache entries.
- **weakref 生命周期会改变观测** / **Weakref lifetime affects observations**: 被回收的对象不应继续制造“同一对象”的假匹配。 / Collected objects should not continue to look like the same matched object.
- **累计值必须跨 region 传递** / **Carry the accumulated value across regions**: 只统计当前 region 会让全局安全阀失效。 / Counting only the current region disables the global safety brake.

## 延伸阅读 / Further reading

- [PyTorch cache_size.py](https://github.com/pytorch/pytorch/blob/0bed4e802fa3454f7609a409f0c5f828060726f8/torch/_dynamo/cache_size.py)
- [TorchDynamo recompilation limits](https://github.com/pytorch/pytorch/blob/0bed4e802fa3454f7609a409f0c5f828060726f8/torch/_dynamo/config.py)
