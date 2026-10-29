---
date: 2026-10-29
topic: huggingface
source: huggingface
repo: huggingface/trl
file: trl/models/activation_offloading.py
permalink: https://github.com/huggingface/trl/blob/e29597afcf973b91b412662988cf7dc88dde4cd1/trl/models/activation_offloading.py#L196-L285
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, huggingface, activation-offloading]
---

# TRL activation storage 去重 / TRL Activation Storage Deduplication

> **一句话 / In one line**: TRL 用 storage identity 防止同一块 activation 被卸载两次。 / TRL uses storage identity to avoid offloading the same activation twice.

## 为什么重要 / Why this matters

view、共享 storage 和参数别名会让“每个 tensor 都复制”既浪费带宽又可能破坏布局；先辨认底层 storage 才能安全节省显存。

Views, shared storage, and parameter aliases make a copy-per-tensor approach wasteful and sometimes layout-breaking. Recognizing the backing storage first makes memory savings safe.

## 代码 / The code

`huggingface/trl` — [`activation_offloading.py`](https://github.com/huggingface/trl/blob/e29597afcf973b91b412662988cf7dc88dde4cd1/trl/models/activation_offloading.py#L196-L285)

```python
storage_key = _get_unique_tensor_key(activation)
if self.use_streams and storage_key in self.storage_to_tensor_id:
    self.tracker[tensor_id] = (activation, False, None, None, None)
    track_storage_key(storage_key, tensor_id)
    return tensor_id
if num_bytes < self.min_tensor_size_bytes:
    self.tracker[tensor_id] = (activation, False, None, None, None)
    return tensor_id
```

## 逐行讲解 / What's happening

1. 中文：key 针对 storage 而不是 Python 对象，因此 view 也会命中。 English: The key targets storage rather than a Python object, so views can hit it too.
2. 中文：已追踪的 storage 保留引用但不再传到 CPU。 English: An already tracked storage keeps a reference but is not transferred to CPU again.
3. 中文：小 tensor 直接留下，避免传输成本反超节省。 English: Tiny tensors stay put when transfer cost exceeds the saving.

## 类比 / The analogy

同一叠文件被两个文件夹引用时，仓库只搬一次纸，不会搬两车副本。

When two folders reference one stack of papers, a warehouse moves the papers once, not two duplicate trucks.

## 自己跑一遍 / Try it yourself

```python
seen = set()
for storage in ["A", "A", "B"]:
    if storage in seen:
        print("reuse", storage)
    else:
        seen.add(storage); print("offload", storage)
```

运行 / Run with: `python try.py`

预期输出 / Expected output: `offload A`, `reuse A`, `offload B`

中文：关键不是 tensor 名字，而是它背后的 storage。 English: The important identity is backing storage, not the tensor name.

## 注意事项 / Caveats / when it breaks

- 中文：异步 stream 需要事件同步；去重本身不能保证跨 stream 正确性。 / English: Asynchronous streams need event synchronization; deduplication alone does not ensure cross-stream correctness.

## 延伸阅读 / Further reading

- [TRL activation offloading](https://github.com/huggingface/trl/blob/e29597afcf973b91b412662988cf7dc88dde4cd1/trl/models/activation_offloading.py#L196-L285)
