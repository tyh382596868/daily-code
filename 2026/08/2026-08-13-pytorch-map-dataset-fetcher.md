---
date: 2026-08-13
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/utils/data/_utils/fetch.py
permalink: https://github.com/pytorch/pytorch/blob/e77e5d57b455deece295bb28d79f4da480f68450/torch/utils/data/_utils/fetch.py#L10-L57
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, pytorch, dataloader]
---

# PyTorch MapDatasetFetcher：batch 索引可以走快速通道 / PyTorch MapDatasetFetcher: Batched Indices Can Take a Fast Path

> **一句话 / In one line**: `_MapDatasetFetcher` 在 auto-collation 时优先调用 dataset 的 `__getitems__`，否则才逐个 `__getitem__`。 / With auto-collation enabled, `_MapDatasetFetcher` first tries dataset-level `__getitems__`, then falls back to per-index `__getitem__`.

## 为什么重要 / Why this matters

很多人以为 DataLoader 只能一个 index 一个 index 地拿样本。这个小类说明 map-style dataset 可以声明批量读取接口，把磁盘 IO、远端读取或列式存储访问合并成一次操作。

Many users assume a DataLoader must fetch map-style samples one index at a time. This class shows the batched-fetch escape hatch, allowing disk IO, remote reads, or columnar storage access to be fused into one operation.

## 代码 / The code

`pytorch/pytorch` — [`torch/utils/data/_utils/fetch.py`](https://github.com/pytorch/pytorch/blob/e77e5d57b455deece295bb28d79f4da480f68450/torch/utils/data/_utils/fetch.py#L10-L57)

```python
class _BaseDatasetFetcher:
    def __init__(self, dataset, auto_collation, collate_fn, drop_last) -> None:
        self.dataset = dataset
        self.auto_collation = auto_collation
        self.collate_fn = collate_fn
        self.drop_last = drop_last

    def fetch(self, possibly_batched_index) -> NoReturn:
        raise NotImplementedError


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


class _MapDatasetFetcher(_BaseDatasetFetcher):
    def fetch(self, possibly_batched_index):
        if self.auto_collation:
            if hasattr(self.dataset, "__getitems__") and self.dataset.__getitems__:
                data = self.dataset.__getitems__(possibly_batched_index)
            else:
                data = [self.dataset[idx] for idx in possibly_batched_index]
        else:
            data = self.dataset[possibly_batched_index]
        return self.collate_fn(data)
```

## 逐行讲解 / What's happening

1. **第 10-18 行 / Lines 10-18: base fetcher 只保存 dataset、collate 和模式标志，真正策略留给子类。 / The base fetcher stores dataset, collate function, and mode flags; strategy belongs to subclasses.**
2. **第 49-54 行 / Lines 49-54: 一组 index 到来时先走 `__getitems__` 批量接口，没有才逐个读。 / When a batch of indices arrives, `__getitems__` gets first chance; otherwise it reads one index at a time.**
3. **第 55-57 行 / Lines 55-57: 非 auto-collation 时直接传单个 index，最后仍统一 `collate_fn`。 / Without auto-collation, the single index is passed through, and `collate_fn` still owns the output contract.**

## 类比 / The analogy

像去仓库取货：如果仓库支持“一张清单一次取完”，就不要让搬运工一件一件跑。

It is like warehouse picking: if the warehouse can fulfill a whole list in one trip, do that instead of walking item by item.

## 自己跑一遍 / Try it yourself

```python
class Dataset:
    def __getitem__(self, i):
        print("slow", i)
        return i * 10
    def __getitems__(self, indices):
        print("fast", indices)
        return [i * 10 for i in indices]

def fetch(dataset, index, auto, collate):
    if auto:
        if hasattr(dataset, "__getitems__") and dataset.__getitems__:
            data = dataset.__getitems__(index)
        else:
            data = [dataset[i] for i in index]
    else:
        data = dataset[index]
    return collate(data)

print(fetch(Dataset(), [2, 4, 5], True, tuple))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
fast [2, 4, 5]
(20, 40, 50)
```

`__getitem__` 没有被调用；批量读取接口接管了整个 index 列表。

`__getitem__` is not called; the batched fetch method consumes the full index list.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **HF Datasets columnar reads** / **HF Datasets columnar reads**: 列式后端适合一次取多个 row。 / Columnar backends are good at fetching many rows at once.
- **IterableDatasetFetcher** / **IterableDatasetFetcher**: iterable-style 数据没有随机 index，只能从 iterator 连续凑 batch。 / Iterable-style data has no random index and must batch by advancing an iterator.

## 注意事项 / Caveats / when it breaks

- **不要返回已 collate 的 batch** / **Do not return an already-collated batch**: `collate_fn` 还会再跑一次。 / `collate_fn` will still run afterward.
- **顺序要保持** / **Preserve order**: 返回样本顺序必须匹配输入 index 顺序。 / Returned samples must match input index order.

## 延伸阅读 / Further reading

- [pytorch/pytorch source](https://github.com/pytorch/pytorch/blob/e77e5d57b455deece295bb28d79f4da480f68450/torch/utils/data/_utils/fetch.py#L10-L57)
