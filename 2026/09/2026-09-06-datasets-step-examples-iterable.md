---
date: 2026-09-06
topic: huggingface
source: huggingface
repo: huggingface/datasets
file: src/datasets/iterable_dataset.py
permalink: https://github.com/huggingface/datasets/blob/19f69de53015525126e6d3f96859acb90498e364/src/datasets/iterable_dataset.py#L3288-L3400
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, datasets, streaming, sharding]
---

# Datasets StepExamplesIterable：按步长抽样，同时保住断点 / Datasets StepExamplesIterable: Step Through Streaming Data Without Losing Resume State

> **一句话 / In one line**: 这个 iterable 让 streaming 数据按 step 和 offset 取样，还能把 shard、shuffle、state_dict 一起传下去。 / This iterable samples streaming data by step and offset while carrying shard, shuffle, and state_dict through the pipeline.

## 为什么重要 / Why this matters

中文：流式数据最难的不是“能不能遍历”，而是“分布式切分后还能不能恢复到同一个位置”。`StepExamplesIterable` 把步长抽样、offset、恢复状态和 Arrow 读取的分片语义都串到一起，避免训练重启后样本顺序漂掉。

English: With streaming data, the hard part is not iteration; it is whether you can resume from the same position after distributed slicing. `StepExamplesIterable` ties step sampling, offsets, resume state, and Arrow shard semantics together so restarts do not drift the sample order.

## 代码 / The code

`huggingface/datasets` — [`src/datasets/iterable_dataset.py`](https://github.com/huggingface/datasets/blob/19f69de53015525126e6d3f96859acb90498e364/src/datasets/iterable_dataset.py#L3288-L3400)

```python
class StepExamplesIterable:
    def __init__(self, ex_iterable, step, offset=0):
        self.ex_iterable = ex_iterable
        self.step = step
        self.offset = offset

    def _init_state_dict(self):
        self.state_dict = self.ex_iterable._init_state_dict()

    def __iter__(self):
        return islice(self.ex_iterable, self.offset, None, self.step)

    def _iter_arrow(self, *args, **kwargs):
        for idx, item in enumerate(self.ex_iterable._iter_arrow(*args, **kwargs)):
            if idx % self.step == self.offset:
                yield item

    def shard(self, num_shards, index):
        return StepExamplesIterable(
            self.ex_iterable.shard(num_shards, index), self.step, self.offset
        )
```

## 逐行讲解 / What's happening

1. **第 3288-3297 行 / Lines 3288-3297**:
   - 中文: 构造器只保存 `step` 和 `offset`，它本质上是一个轻量 wrapper。
   - English: The constructor only stores `step` and `offset`; this is a lightweight wrapper by design.
2. **第 3317-3344 行 / Lines 3317-3344**:
   - 中文: `__iter__()` 和 `_iter_arrow()` 都在做“每隔 N 个取一个”，只是一个走 Python 迭代器，一个走 Arrow 流。
   - English: `__iter__()` and `_iter_arrow()` both implement “take every Nth item,” but one walks a Python iterator and the other walks an Arrow stream.
3. **第 3365-3400 行 / Lines 3365-3400**:
   - 中文: `shuffle()`、`shard()` 和 `reshard()` 都保持这个 step 语义，不会把 stream 重新洗乱。
   - English: `shuffle()`, `shard()`, and `reshard()` all preserve the step semantics instead of scrambling the stream.

## 类比 / The analogy

中文：像翻一本长目录，只看每隔三页的页签，但书签位置必须能在下次继续翻的时候原样恢复。

English: It is like scanning a long index and only reading every third tab, while keeping the bookmark position exactly recoverable on the next pass.

## 自己跑一遍 / Try it yourself

```python
def step(xs, step, offset=0):
    return xs[offset::step]

print(step(list(range(10)), 3))
print(step(list(range(10)), 3, 1))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[0, 3, 6, 9]
[1, 4, 7]
```

中文：这类 iterable 的关键是“取样规则”和“恢复状态”同样重要。

English: For this kind of iterable, the sampling rule and the resume state matter equally.
