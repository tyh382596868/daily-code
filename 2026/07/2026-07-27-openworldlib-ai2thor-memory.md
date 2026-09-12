---
date: 2026-07-27
topic: infrastructure
source: trending
repo: OpenDCAI/OpenWorldLib
file: src/openworldlib/memories/simulation_environment/thor/ai2thor_memory.py
permalink: https://github.com/OpenDCAI/OpenWorldLib/blob/3d9034dd9c6cb3a1abca4e5ca8fd273d3d3d2153/src/openworldlib/memories/simulation_environment/thor/ai2thor_memory.py#L21-L88
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, infrastructure, world-model, memory]
---

# OpenWorldLib Memory：把交互流收进统一记录表 / OpenWorldLib Memory: Store Interaction Streams as Uniform Records

> **一句话 / In one line**: `Ai2ThorMemory` 把图像、动作和其他事件都保存成同一类记录，再用类型、时间和自定义 filter 检索。 / `Ai2ThorMemory` stores images, actions, and other events as one record shape, then retrieves them by type, time, or custom filters.

## 为什么重要 / Why this matters

world model 框架不只是模型本体，还需要记录“刚刚看到了什么、做了什么、下一步要拿哪些上下文”。OpenWorldLib 这段 memory 代码展示了一个朴素但清晰的 runtime state 层：统一入库、容量驱逐、按 query 取回。

A world-model framework is not only the model; it also needs to remember what it just saw, what it did, and which context the next step should use. This OpenWorldLib memory code shows a simple runtime state layer: uniform ingestion, capacity eviction, and query-based retrieval.

## 代码 / The code

`OpenDCAI/OpenWorldLib` — [`src/openworldlib/memories/simulation_environment/thor/ai2thor_memory.py`](https://github.com/OpenDCAI/OpenWorldLib/blob/3d9034dd9c6cb3a1abca4e5ca8fd273d3d3d2153/src/openworldlib/memories/simulation_environment/thor/ai2thor_memory.py#L21-L88)

```python
    def record(self, data, metadata: Optional[Dict[str, Any]] = None, **kwargs):
        if metadata is None:
            metadata = {}

        t = str(metadata.get("type", "other"))
        if t not in self.TYPE_LIST:
            metadata = dict(metadata)
            metadata["type_original"] = t
            metadata["type"] = "other"
            t = "other"

        item = {
            "content": data,                 # 关键：完整保存 content（dict/ndarray/any）
            "type": t,
            "timestamp": time.time(),
            "metadata": metadata,
        }
        self.storage.append(item)

        # capacity FIFO eviction
        if self.capacity is not None and len(self.storage) > int(self.capacity):
            overflow = len(self.storage) - int(self.capacity)
            if overflow > 0:
                self.storage = self.storage[overflow:]

    # ---------------- 2. select (retrieval) ----------------
    def select(self, context_query: ContextQuery = None, **kwargs) -> List[Dict[str, Any]]:
        if len(self.storage) == 0:
            return []

        if context_query is None:
            return list(self.storage)

        if isinstance(context_query, str):
            q = context_query.lower().strip()
            if q == "all":
                return list(self.storage)
            if q == "last_image":
                return self.select({"type": "image", "last_n": 1})
            if q == "last_action":
                return self.select({"type": "action", "last_n": 1})
            return list(self.storage)

        if not isinstance(context_query, dict):
            return list(self.storage)

        items = list(self.storage)

        q_type = context_query.get("type", None)
        if isinstance(q_type, str):
            items = [it for it in items if it.get("type") == q_type.strip()]

        since_time = context_query.get("since_time", None)
        if isinstance(since_time, (int, float)):
            st = float(since_time)
            items = [it for it in items if float(it.get("timestamp", 0.0)) >= st]

        flt = context_query.get("filter", None)
        if callable(flt):
            items = [it for it in items if bool(flt(it))]

        last_n = context_query.get("last_n", None)
        if isinstance(last_n, (int, float)):
            n = max(0, int(last_n))
            if n > 0:
                items = items[-n:]

        return items
```

## 逐行讲解 / What's happening

1. **第 21-31 行 / Lines 21-31 (`type normalization`)**:
   - 中文: 未知 type 不直接报错，而是保留原 type 并降级成 `other`。
   - English: Unknown types do not fail; the original type is preserved and the record is downgraded to `other`.
2. **第 32-38 行 / Lines 32-38 (`record shape`)**:
   - 中文: 每条记录都有 `content/type/timestamp/metadata`，后续模块只需要认识这一种表结构。
   - English: Every item has `content/type/timestamp/metadata`, so later modules only need one table shape.
3. **第 40-44 行 / Lines 40-44 (`FIFO capacity`)**:
   - 中文: 超出容量时从头部切掉旧记录，保持 memory 有界。
   - English: When capacity is exceeded, old records are sliced off from the front to keep memory bounded.
4. **第 47-62 行 / Lines 47-62 (`string shortcuts`)**:
   - 中文: `last_image`、`last_action` 这种字符串 query 会转成结构化查询。
   - English: String queries such as `last_image` and `last_action` become structured queries.
5. **第 67-88 行 / Lines 67-88 (`structured query`)**:
   - 中文: dict query 可以按 type、since_time、callable filter 和 last_n 层层过滤。
   - English: Dict queries filter by type, `since_time`, callable filters, and `last_n` in sequence.

## 类比 / The analogy

这像实验室记录本：显微镜照片、按钮动作和备注都按同一页格式登记，之后可以按日期、类别或关键词翻回去。

It is like a lab notebook: microscope images, button presses, and notes are logged with one page format, then retrieved by date, category, or keyword later.

## 自己跑一遍 / Try it yourself

```python
import time

store = []
def record(content, typ):
    store.append({"content": content, "type": typ, "timestamp": time.time()})
def select(typ=None, last_n=None):
    items = [x for x in store if typ is None or x["type"] == typ]
    return items[-last_n:] if last_n else items

record("rgb0", "image")
record({"move": "left"}, "action")
record("rgb1", "image")
print(select("image", 1))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
[{'content': 'rgb1', 'type': 'image', 'timestamp': ...}]
```

统一记录格式让 `last image` 和 `last action` 都只是同一个 select 函数的不同参数。

A uniform record shape makes `last image` and `last action` different parameters to the same `select` function.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **robot policy server memory** / **Robot policy server memory**: 中文: 异步 policy server 也需要缓存最近观测和动作。 / English: Async policy servers also need recent observation/action buffers.
- **KV cache** / **KV cache**: 中文: LLM KV cache 是另一种 memory，只是 value 是 transformer 中间状态。 / English: LLM KV cache is another memory form whose values are transformer hidden state.

## 注意事项 / Caveats / when it breaks

- **无持久化** / **No persistence**: 中文: 这里是内存列表，进程退出后记录消失。 / English: This is an in-memory list, so records vanish when the process exits.
- **filter 安全** / **Filter safety**: 中文: callable filter 很灵活，但生产环境要限制副作用。 / English: Callable filters are flexible, but production systems should constrain side effects.

## 延伸阅读 / Further reading

- [OpenDCAI/OpenWorldLib source](https://github.com/OpenDCAI/OpenWorldLib/blob/3d9034dd9c6cb3a1abca4e5ca8fd273d3d3d2153/src/openworldlib/memories/simulation_environment/thor/ai2thor_memory.py#L21-L88)
