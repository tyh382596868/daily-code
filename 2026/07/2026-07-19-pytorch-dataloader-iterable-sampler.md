---
date: 2026-07-19
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/utils/data/dataloader.py
permalink: https://github.com/pytorch/pytorch/blob/main/torch/utils/data/dataloader.py
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, dataloader, iterable-dataset, sampler]
---

# PyTorch DataLoader：IterableDataset 不能自定义 sampler / PyTorch DataLoader: IterableDataset Cannot Use a Custom Sampler

> **一句话 / In one line**: `DataLoader` 遇到 `IterableDataset` 时会禁用 `sampler` 和 `batch_sampler`，因为 key 对流式数据没有稳定意义。 / When `DataLoader` sees an `IterableDataset`, it rejects `sampler` and `batch_sampler` because keys have no stable meaning for streaming data.

## 为什么重要 / Why this matters

map-style dataset 可以按 index 抽样，streaming dataset 只能沿着 iterator 往前读。如果还允许用户塞一个 sampler，用户会以为自己控制了样本顺序，实际多 worker 下可能重复、丢样本或批次错位。PyTorch 在构造阶段直接报错，避免把错留到训练半小时之后。

A map-style dataset can sample by index; a streaming dataset can only move through an iterator. If a custom sampler were allowed, users might think they control sample order, while multi-worker loading could duplicate or drop samples. PyTorch rejects it during construction instead of letting training fail later.

## 代码 / The code

`pytorch/pytorch` — [`torch/utils/data/dataloader.py`](https://github.com/pytorch/pytorch/blob/main/torch/utils/data/dataloader.py)

```python
if isinstance(dataset, IterableDataset):
    self._dataset_kind = _DatasetKind.Iterable

    if isinstance(dataset, IterDataPipe):
        if shuffle is not None:
            dataset = torch.utils.data.graph_settings.apply_shuffle_settings(dataset, shuffle=shuffle)
    elif shuffle not in {False, None}:
        raise ValueError(
            f"DataLoader with IterableDataset: expected unspecified shuffle option, but got shuffle={shuffle}"
        )

    if sampler is not None:
        raise ValueError(
            f"DataLoader with IterableDataset: expected unspecified sampler option, but got sampler={sampler}"
        )
    elif batch_sampler is not None:
        raise ValueError(
            "DataLoader with IterableDataset: expected unspecified batch_sampler option, "
            f"but got batch_sampler={batch_sampler}"
        )
```

## 逐行讲解 / What's happening

1. **先判 dataset 类型 / Detect the dataset kind first**: 中文: 一旦是 `IterableDataset`，后面就走流式逻辑。 English: once the dataset is iterable-style, the loader switches to streaming logic.
2. **DataPipe 是特例 / DataPipe is a special case**: 中文: `IterDataPipe` 可以把 shuffle 写进 graph setting。 English: `IterDataPipe` can encode shuffle as graph settings.
3. **普通 IterableDataset 不接受 shuffle / Plain IterableDataset rejects shuffle**: 中文: 除了 `False` 或 `None`，其他值都会报错。 English: anything except `False` or `None` is rejected.
4. **sampler 被禁止 / sampler is forbidden**: 中文: sampler 产出 index，但 iterable dataset 没有稳定 index。 English: a sampler emits indices, but an iterable dataset has no stable indices.
5. **batch_sampler 也被禁止 / batch_sampler is forbidden too**: 中文: 多 worker 流式读取时，批次归属是 loader 内部细节。 English: in multi-worker streaming, batch assignment is an internal loader detail.

## 类比 / The analogy

map-style dataset 像图书馆书架，你可以按编号拿第 42 本。IterableDataset 像传送带，箱子来了就拿，下一个箱子是谁只能等它出现。

A map-style dataset is like a library shelf where you can fetch book 42. An `IterableDataset` is like a conveyor belt: you take the next box when it arrives.

## 自己跑一遍 / Try it yourself

```python
class IterableDataset:
    pass

def validate(dataset, shuffle=None, sampler=None, batch_sampler=None):
    if isinstance(dataset, IterableDataset):
        if shuffle not in (False, None):
            raise ValueError("IterableDataset cannot use shuffle")
        if sampler is not None:
            raise ValueError("IterableDataset cannot use sampler")
        if batch_sampler is not None:
            raise ValueError("IterableDataset cannot use batch_sampler")
    return "ok"

print(validate(IterableDataset()))
try:
    validate(IterableDataset(), sampler=[2, 0, 1])
except ValueError as e:
    print(e)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
ok
IterableDataset cannot use sampler
```

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **HF streaming datasets** / **HF streaming datasets**: 流式数据集更常把随机性写进 iterator 自身。 / Streaming datasets usually encode randomness in the iterator itself.
- **PyTorch DataPipes** / **PyTorch DataPipes**: graph setting 比外部 index sampler 更适合流式管道。 / Graph settings fit streaming pipelines better than external index samplers.

## 注意事项 / Caveats / when it breaks

- **不要把 iterable 伪装成 map-style / Do not fake map-style behavior**: 加一个假的 `__len__` 不会让流式读取安全。 / A fake `__len__` does not make streaming reads index-safe.
- **多 worker 会放大问题 / Multi-worker loading amplifies the issue**: 重复和丢样本通常只在并行读取时暴露。 / Duplication and missing samples often appear only under parallel loading.
- **shuffle 要在源头做 / Shuffle at the source**: 对无限流，buffer shuffle 或 shard shuffle 更靠谱。 / For infinite streams, buffer or shard shuffle is the right level.

## 延伸阅读 / Further reading

- [PyTorch DataLoader source](https://github.com/pytorch/pytorch/blob/main/torch/utils/data/dataloader.py)
- [PyTorch data loading docs](https://pytorch.org/docs/stable/data.html)
