---
date: 2026-08-10
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/utils/data/_utils/fetch.py
permalink: https://github.com/pytorch/pytorch/blob/2becd4799c88cc7774b4138e2fb34386f0a8a6c5/torch/utils/data/_utils/fetch.py#L21-L45
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, pytorch, dataloader]
---

# PyTorch IterableDataset fetcher：流式数据也要凑 batch / PyTorch IterableDataset Fetcher: Streaming Data Still Needs Batches

> **一句话 / In one line**: `_IterableDatasetFetcher` 把一个只能 `next()` 的数据流，包装成 DataLoader 能 collate 的单样本或 batch。 / `_IterableDatasetFetcher` turns a `next()`-only stream into either a single sample or a collatable batch for DataLoader.

## 为什么重要 / Why this matters

`IterableDataset` 没有随机索引，DataLoader 不能靠 `dataset[idx]` 取样。这个 fetcher 的价值在于：它把“往前读几个样本”“遇到结尾怎么办”“最后半个 batch 要不要丢掉”这些边界规则集中到一个小类里。

`IterableDataset` has no random indexing, so DataLoader cannot call `dataset[idx]`. This fetcher centralizes the boundary rules: how many samples to pull, what end-of-stream means, and whether an incomplete final batch is allowed.

## 代码 / The code

`pytorch/pytorch` — [`torch/utils/data/_utils/fetch.py`](https://github.com/pytorch/pytorch/blob/2becd4799c88cc7774b4138e2fb34386f0a8a6c5/torch/utils/data/_utils/fetch.py#L21-L45)

```python
class _IterableDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last) -> None:
        super().__init__(dataset, auto_collation, collate_fn, drop_last)
        self.dataset_iter = iter(dataset)
        self.ended = False

    def fetch(self, possibly_batched_index):
        if self.ended:
            raise StopIteration

        if self.auto_collation:
            data = []
            for _ in possibly_batched_index:
                try:
                    data.append(next(self.dataset_iter))
                except StopIteration:
                    self.ended = True
                    break
            if len(data) == 0 or (
                self.drop_last and len(data) < len(possibly_batched_index)
            ):
                raise StopIteration
        else:
            data = next(self.dataset_iter)
        return self.collate_fn(data)
```

## 逐行讲解 / What's happening

1. **第 22-25 行 / Lines 22-25 (iterator ownership)**:
   - 中文: fetcher 在初始化时调用 `iter(dataset)`，之后所有 worker 都从这个 iterator 往前读。
   - English: The fetcher owns `iter(dataset)` and reads forward from that iterator.
2. **第 27-29 行 / Lines 27-29 (sticky end)**:
   - 中文: 一旦数据流结束，`ended` 会让后续 fetch 立即 `StopIteration`，避免反复读空流。
   - English: Once the stream ends, `ended` makes later fetches immediately raise `StopIteration`.
3. **第 31-38 行 / Lines 31-38 (batch fill)**:
   - 中文: `possibly_batched_index` 在这里不是随机索引，而是 batch 大小的占位迭代器。
   - English: Here `possibly_batched_index` is not random indexing; it acts as a placeholder for how many samples to pull.
4. **第 39-45 行 / Lines 39-45 (drop and collate)**:
   - 中文: 空 batch 或 `drop_last=True` 的短 batch 都直接结束，否则交给 `collate_fn` 拼装。
   - English: Empty batches and short final batches under `drop_last=True` stop iteration; otherwise `collate_fn` assembles the result.

## 类比 / The analogy

像从传送带上装盒：map-style 数据像仓库编号取货，iterable 数据像传送带，只能按顺序拿。盒子没装满时要不要发货，就是 `drop_last`。

It is like packing items from a conveyor belt: map-style data is warehouse lookup by ID, while iterable data can only be consumed in order. Whether a half-filled box ships is `drop_last`.

## 自己跑一遍 / Try it yourself

```python
stream = iter([10, 20, 30, 40, 50])
batch_slots = range(3)
drop_last = True
data = []
for _ in batch_slots:
    try:
        data.append(next(stream))
    except StopIteration:
        break
print(data if not (drop_last and len(data) < len(batch_slots)) else "stop")
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[10, 20, 30]
```

这个例子里 `range(3)` 只是“要凑 3 个”的信号，不代表数据集真的按这些数字索引。

Here `range(3)` only means “try to collect three samples”; it does not index the dataset by those numbers.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Kafka consumers** / **Kafka consumers**: 流式消息只能向前消费，也需要批量交付。 / Streaming messages are consumed forward-only but still delivered in batches.
- **Python generators** / **Python generators**: 生成器没有长度和索引，终止状态必须显式处理。 / Generators have no length or indexing, so termination must be handled explicitly.

## 注意事项 / Caveats / when it breaks

- **不能倒带** / **No rewind**: iterator 读过就没了，重复 epoch 需要重新创建 DataLoader/iterator。 / Once consumed, samples are gone; a new epoch needs a fresh iterator.
- **短 batch 语义要清楚** / **Short-batch semantics matter**: 训练代码如果假设固定 batch size，就要配合 `drop_last=True`。 / Training code that assumes fixed batch size should use `drop_last=True`.

## 延伸阅读 / Further reading

- [PyTorch DataLoader fetch internals](https://github.com/pytorch/pytorch/blob/2becd4799c88cc7774b4138e2fb34386f0a8a6c5/torch/utils/data/_utils/fetch.py#L21-L45)
