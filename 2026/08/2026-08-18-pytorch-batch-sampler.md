---
date: 2026-08-18
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/utils/data/sampler.py
permalink: https://github.com/pytorch/pytorch/blob/9ff255cb2d385bd7306ac815b0ba9def13213eb2/torch/utils/data/sampler.py#L286-L354
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, pytorch, dataloader]
---

# PyTorch BatchSampler：把索引流切成小批次 / PyTorch BatchSampler: Slice an Index Stream into Mini-Batches

> **一句话 / In one line**: `BatchSampler` 包住任意 sampler，把单个 index 流变成 `list[int]` batch，并用 `drop_last` 决定尾巴要不要保留。 / `BatchSampler` wraps any sampler, turns an index stream into `list[int]` batches, and uses `drop_last` to decide whether the tail survives.

## 为什么重要 / Why this matters

DataLoader 的 batch 不是从数据集里凭空出现的。底层先有一个 index sampler，再由 `BatchSampler` 把 index 按批次打包；理解这层后，自定义采样、分布式采样和可复现 shuffle 都更清楚。

A DataLoader batch does not appear directly from the dataset. There is first an index sampler, then `BatchSampler` groups indices into mini-batches. Once you see this layer, custom sampling, distributed sampling, and reproducible shuffling become easier to reason about.

## 代码 / The code

`pytorch/pytorch` — [`torch/utils/data/sampler.py`](https://github.com/pytorch/pytorch/blob/9ff255cb2d385bd7306ac815b0ba9def13213eb2/torch/utils/data/sampler.py#L333-L354)

```python
def __iter__(self) -> Iterator[list[int]]:
    sampler_iter = iter(self.sampler)
    if self.drop_last:
        args = [sampler_iter] * self.batch_size
        for batch_droplast in zip(*args, strict=False):
            yield [*batch_droplast]
    else:
        batch = [*itertools.islice(sampler_iter, self.batch_size)]
        while batch:
            yield batch
            batch = [*itertools.islice(sampler_iter, self.batch_size)]

def __len__(self) -> int:
    if self.drop_last:
        return len(self.sampler) // self.batch_size
    return (len(self.sampler) + self.batch_size - 1) // self.batch_size
```

## 逐行讲解 / What's happening

1. **第 333-334 行 / Lines 333-334**: 中文: 先拿到底层 sampler 的 iterator，之后只消费 index，不碰 dataset 本身。 / English: It first obtains the base sampler iterator and only consumes indices, never dataset items.
2. **第 335-339 行 / Lines 335-339**: 中文: `drop_last=True` 时，`[sampler_iter] * batch_size` 创建的是同一个 iterator 的多个引用，`zip` 每轮拉满一个 batch；不满就自然丢掉。 / English: With `drop_last=True`, repeated references to the same iterator let `zip` pull one full batch per round; an incomplete tail is naturally dropped.
3. **第 340-344 行 / Lines 340-344**: 中文: `drop_last=False` 时，`islice` 每次最多切 `batch_size` 个，最后一个短 batch 也会 yield。 / English: With `drop_last=False`, `islice` takes up to `batch_size` items, so the last short batch is yielded too.
4. **第 346-354 行 / Lines 346-354**: 中文: 长度计算正好对应上面两种策略：整除是丢尾，向上取整是留尾。 / English: The length formula mirrors the two policies: floor division drops the tail, ceiling division keeps it.

## 类比 / The analogy

像把排队号码按三张一组发给窗口。`drop_last=True` 是“最后不足三个人就等下一轮”，`drop_last=False` 是“最后一个窗口两个人也办”。

It is like handing queue tickets to a service counter in groups of three. `drop_last=True` waits if the last group is short; `drop_last=False` serves the short final group.

## 自己跑一遍 / Try it yourself

```python
import itertools

def batch_indices(indices, batch_size, drop_last):
    it = iter(indices)
    if drop_last:
        for batch in zip(*([it] * batch_size)):
            yield list(batch)
    else:
        batch = list(itertools.islice(it, batch_size))
        while batch:
            yield batch
            batch = list(itertools.islice(it, batch_size))

print(list(batch_indices(range(10), 3, False)))
print(list(batch_indices(range(10), 3, True)))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
[[0, 1, 2], [3, 4, 5], [6, 7, 8]]
```

同一条 index 流，只是尾部策略不同。

The index stream is the same; only the tail policy changes.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **distributed sampler** / **distributed samplers**: 先决定每个 rank 的 index 流，再让 batch sampler 打包。 / They first decide each rank's index stream, then batch it.
- **bucketing** / **length bucketing**: NLP/语音常先按长度排序或分桶，再用 batch sampler 切 batch。 / NLP and speech pipelines often bucket by length before batching.

## 注意事项 / Caveats / when it breaks

- **同一个 iterator 引用不是复制数据** / **Repeated iterator references do not copy data**: `zip(*args)` 的技巧依赖所有引用都指向同一个 iterator。 / The `zip(*args)` trick works because every reference points to the same iterator.
- **`__len__` 要求 sampler 有长度** / **`__len__` requires a sized sampler**: 如果底层 sampler 没有 `len`，这个长度查询会失败。 / If the base sampler has no `len`, querying batch-sampler length fails.

## 延伸阅读 / Further reading

- [PyTorch source](https://github.com/pytorch/pytorch/blob/9ff255cb2d385bd7306ac815b0ba9def13213eb2/torch/utils/data/sampler.py#L286-L354)
