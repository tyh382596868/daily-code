---
date: 2026-07-19
topic: huggingface
source: huggingface
repo: huggingface/datasets
file: src/datasets/iterable_dataset.py
permalink: https://github.com/huggingface/datasets/blob/main/src/datasets/iterable_dataset.py
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, datasets, iterable, map]
---

# Datasets map：流式样本也能攒成 batch 再变换 / Datasets map: Streamed Examples Can Still Be Batched Before Transform

> **一句话 / In one line**: `MappedExamplesIterable` 在 `batched=True` 时先从流里取一小段样本，拼成 batch，应用函数，再拆回逐样本输出。 / With `batched=True`, `MappedExamplesIterable` pulls a short slice from the stream, builds a batch, applies the function, then yields examples again.

## 为什么重要 / Why this matters

流式 dataset 不能随机访问整张表，但很多预处理函数天然是 batch 形态：tokenizer、特征抽取、列变换都更适合一次处理多条。Datasets 的做法是只攒当前窗口，不把整个流 materialize 到内存里。

A streaming dataset cannot randomly access the whole table, but many preprocessing functions are naturally batched: tokenizers, feature extractors, and column transforms work better on multiple rows. Datasets batches only the current window, without materializing the full stream.

## 代码 / The code

`huggingface/datasets` — [`src/datasets/iterable_dataset.py`](https://github.com/huggingface/datasets/blob/main/src/datasets/iterable_dataset.py)

```python
class MappedExamplesIterable(_BaseExamplesIterable):
    def __init__(self, ex_iterable, function, batched=False, batch_size=1000):
        self.ex_iterable = ex_iterable
        self.function = function
        self.batched = batched
        self.batch_size = batch_size

    def __iter__(self):
        iterator = iter(self.ex_iterable)
        for key, example in iterator:
            if self.batched:
                key_examples_list = [(key, example)] + [
                    (key, example) for key, example in islice(iterator, self.batch_size - 1)
                ]
                keys, examples = zip(*key_examples_list)
                batch = _examples_to_batch(examples)
                transformed_batch = self.function(batch)
                yield from zip(keys, _batch_to_examples(transformed_batch))
            else:
                yield key, self.function(example)
```

## 逐行讲解 / What's happening

1. **包装底层 iterable / Wrap the base iterable**: 中文: 这个类不拥有数据，只包住一个 examples iterator。 English: this class does not own data; it wraps an examples iterator.
2. **batched 决定路径 / `batched` chooses the path**: 中文: 非 batch 模式就是一条进一条出。 English: non-batched mode is one example in, one example out.
3. **用 `islice` 攒窗口 / Use `islice` to collect a window**: 中文: 它只向前多取 `batch_size - 1` 条。 English: it only reads up to `batch_size - 1` additional rows.
4. **样本转 batch / Examples become a batch**: 中文: `_examples_to_batch` 把 list of dict 变成 dict of lists。 English: `_examples_to_batch` turns a list of dicts into a dict of lists.
5. **batch 再拆回样本 / The batch is split back**: 中文: 输出仍保持流式逐样本协议。 English: output keeps the streaming per-example protocol.

## 类比 / The analogy

像面包店的烤箱。顾客是一个一个排队来的，但师傅会攒满一盘再烤；出炉后仍然一只一只卖给顾客。

It is like a bakery oven. Customers arrive one by one, but the baker fills a tray before baking; after baking, buns are still handed out individually.

## 自己跑一遍 / Try it yourself

```python
from itertools import islice

def to_batch(rows):
    return {k: [r[k] for r in rows] for k in rows[0]}

def to_rows(batch):
    keys = list(batch)
    for i in range(len(batch[keys[0]])):
        yield {k: batch[k][i] for k in keys}

def stream_map(rows, fn, batch_size):
    it = iter(rows)
    for first in it:
        window = [first] + list(islice(it, batch_size - 1))
        yield from to_rows(fn(to_batch(window)))

rows = [{"x": 1}, {"x": 2}, {"x": 3}]
print(list(stream_map(rows, lambda b: {"x": [v * 10 for v in b["x"]]}, 2)))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[{'x': 10}, {'x': 20}, {'x': 30}]
```

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Transformers tokenizer map** / **Transformers tokenizer map**: tokenizer 通常吃 dict of lists，再吐 dict of lists。 / Tokenizers usually consume and return dicts of lists.
- **PyTorch DataPipe map** / **PyTorch DataPipe map**: 流式 map 也会尽量保持 iterator 协议。 / Streaming maps preserve the iterator contract.

## 注意事项 / Caveats / when it breaks

- **函数必须返回等长列 / The function must return aligned columns**: 每列长度不一致会让拆 batch 失败。 / Unequal output column lengths break row reconstruction.
- **batch_size 是内存旋钮 / batch_size controls memory**: 太大会失去流式优势。 / Too large a batch erodes the streaming advantage.
- **key 仍然来自原样本 / Keys still come from original samples**: transform 改内容，不应乱改样本定位。 / The transform changes content, not sample identity.

## 延伸阅读 / Further reading

- [Datasets iterable source](https://github.com/huggingface/datasets/blob/main/src/datasets/iterable_dataset.py)
- [Datasets streaming docs](https://huggingface.co/docs/datasets/stream)
