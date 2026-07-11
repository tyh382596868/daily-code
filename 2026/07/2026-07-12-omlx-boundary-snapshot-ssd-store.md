---
date: 2026-07-12
topic: infrastructure
source: trending
repo: jundot/omlx
file: omlx/cache/boundary_snapshot_store.py
permalink: https://github.com/jundot/omlx/blob/d5fcb22a87c3b46ab6dd91016fbbbdb1e624f374/omlx/cache/boundary_snapshot_store.py#L122-L190
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, infrastructure, ssd-cache]
---

# oMLX BoundarySnapshotSSDStore：先写内存队列，再让后台线程落盘 / oMLX BoundarySnapshotSSDStore: Buffer First, Flush on a Writer Thread

> **一句话 / In one line**: oMLX 把 prefill 边界 cache 快照先序列化到内存，再放入有界队列交给后台线程写 SSD。 / oMLX serializes boundary cache snapshots into memory first, then places them on a bounded queue for a background thread to write to SSD.

## 为什么重要 / Why this matters

本地 MLX 推理想支持长上下文和连续 batching，就不能把所有中间 KV 状态都压在 GPU 内存里。这个 trending 项目里的 `BoundarySnapshotSSDStore.save` 展示了一个实用策略：推理线程只做 Metal-safe 的提取和内存缓冲，慢速磁盘写入异步完成。

Local MLX inference with long contexts and continuous batching cannot keep every intermediate KV state in GPU memory. `BoundarySnapshotSSDStore.save` shows a practical design: the inference thread performs Metal-safe extraction and memory buffering, while slower disk writes happen asynchronously.

## 代码 / The code

`jundot/omlx` — [`omlx/cache/boundary_snapshot_store.py`](https://github.com/jundot/omlx/blob/d5fcb22a87c3b46ab6dd91016fbbbdb1e624f374/omlx/cache/boundary_snapshot_store.py#L122-L190)

```python
def save(
    self,
    request_id: str,
    token_count: int,
    snapshot_cache: list[Any],
    extract_cache_states_fn: Callable,
) -> bool:
    if not HAS_MLX:
        return False

    try:
        extracted, model_cache_config = extract_cache_states_fn(snapshot_cache)
        if not extracted:
            return False

        tensors_raw, metadata = self._serialize_extracted(
            extracted, request_id, token_count
        )

        pw_key = (request_id, token_count)
        with self._pending_lock:
            self._pending_writes[pw_key] = {
                "tensors_raw": tensors_raw,
                "metadata": metadata,
                "extracted": extracted,
            }

        file_path = self._file_path(request_id, token_count)
        with self._registry_lock:
            self._file_registry.setdefault(request_id, {})[token_count] = file_path

        try:
            self._write_queue.put_nowait((pw_key, tensors_raw, metadata, file_path))
        except queue.Full:
            logger.warning("Boundary snapshot write queue full, dropping snapshot %s/%d", request_id, token_count)
            with self._pending_lock:
                self._pending_writes.pop(pw_key, None)
            with self._registry_lock:
                req_files = self._file_registry.get(request_id)
                if req_files is not None:
                    req_files.pop(token_count, None)
                    if not req_files:
                        self._file_registry.pop(request_id, None)
            return False

        return True

    except Exception as e:
        logger.debug("Failed to save boundary snapshot: %s", e)
        return False
```

## 逐行讲解 / What's happening

1. **先提取 cache 状态 / Extract cache state first**:
   - 中文: `extract_cache_states_fn` 把模型 cache 对象转成可序列化结构。
   - English: `extract_cache_states_fn` converts model cache objects into a serializable structure.
2. **内存 pending 区 / In-memory pending area**:
   - 中文: `_pending_writes` 让刚保存的快照可以立即读回，不必等后台线程写完。
   - English: `_pending_writes` lets a newly saved snapshot be read back immediately without waiting for disk flush.
3. **注册 request 到文件路径 / Register request to file path**:
   - 中文: `_file_registry` 是清理和加载时的目录索引。
   - English: `_file_registry` is the directory index used for load and cleanup.
4. **有界队列与回滚 / Bounded queue and rollback**:
   - 中文: 如果队列满了，代码会撤销 pending 和 registry，避免留下永远不会写盘的假记录。
   - English: If the queue is full, the code rolls back pending and registry state so it does not leave records that will never hit disk.

## 类比 / The analogy

像咖啡店高峰期接单：收银员先把订单贴在“待做”板上，让顾客能查到状态；如果后厨传送带满了，就把订单从看板上撤掉并明确失败，而不是假装已经开始制作。

It is like a busy cafe: the cashier first puts the order on the pending board so status is visible; if the kitchen belt is full, the order is removed and rejected instead of pretending it is in progress.

## 自己跑一遍 / Try it yourself

```python
import queue

pending, registry = {}, {}
q = queue.Queue(maxsize=1)

def save(rid, tok):
    key = (rid, tok)
    pending[key] = "bytes"
    registry.setdefault(rid, {})[tok] = f"{rid}/{tok}.bin"
    try:
        q.put_nowait(key)
        return True
    except queue.Full:
        pending.pop(key, None)
        registry[rid].pop(tok, None)
        return False

print(save("a", 128), save("a", 256))
print(pending, registry)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
True False
{('a', 128): 'bytes'} {'a': {128: 'a/128.bin'}}
```

第二次保存失败后，内存索引被回滚，不会留下不可读的快照记录。

After the second save fails, the in-memory indexes are rolled back, so no unreadable snapshot record remains.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **vLLM paged KV cache** / **vLLM paged KV cache**: 也把 cache 生命周期拆成元数据、块存储和请求引用。 / It also separates cache lifetime into metadata, block storage, and request references.
- **数据库 WAL** / **database WAL**: 先记录可恢复状态，再异步完成较慢的持久化工作。 / Durable systems often record recoverable state before slower persistence completes.

## 注意事项 / Caveats / when it breaks

- **队列满是背压信号** / **a full queue is backpressure**: 这里选择丢弃快照，而不是阻塞推理线程。 / This code drops the snapshot instead of blocking the inference thread.
- **清理路径必须加锁** / **cleanup paths need locks**: pending、registry、cancelled request 都可能被后台线程同时访问。 / Pending writes, registries, and cancelled requests can all be touched by the writer thread concurrently.

## 延伸阅读 / Further reading

- oMLX boundary snapshot source — https://github.com/jundot/omlx/blob/d5fcb22a87c3b46ab6dd91016fbbbdb1e624f374/omlx/cache/boundary_snapshot_store.py
