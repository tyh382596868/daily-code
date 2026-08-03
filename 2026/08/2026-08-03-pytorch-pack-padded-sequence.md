---
date: 2026-08-03
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/nn/utils/rnn.py
permalink: https://github.com/pytorch/pytorch/blob/3d99896d477c69400c9206d00fa8bf7c4a139681/torch/nn/utils/rnn.py#L258-L324
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, rnn, variable-length]
---

# PyTorch pack_padded_sequence：变长序列先排序再压紧 / PyTorch pack_padded_sequence: Sort Variable-Length Sequences, Then Pack Them

> **一句话 / In one line**: `pack_padded_sequence` 把 padded batch 转成紧凑的 `PackedSequence`，并在需要时按长度降序重排。 / `pack_padded_sequence` turns a padded batch into a compact `PackedSequence`, sorting by descending length when needed.

## 为什么重要 / Why this matters

RNN 遇到变长序列时，padding 位置不该浪费计算，也不该影响状态转移。这个函数把 Python 侧的长度、排序和 batch 维度处理完，再交给底层 packing op。

For variable-length RNN inputs, padded positions should not waste compute or affect state transitions. This function handles lengths, sorting, and batch dimension conventions in Python before calling the lower-level packing op.

## 代码 / The code

`pytorch/pytorch` — [`torch/nn/utils/rnn.py`](https://github.com/pytorch/pytorch/blob/3d99896d477c69400c9206d00fa8bf7c4a139681/torch/nn/utils/rnn.py#L258-L324)

```python
def pack_padded_sequence(
    input: Tensor,
    lengths: Tensor | list[int],
    batch_first: bool = False,
    enforce_sorted: bool = True,
) -> PackedSequence:
    r"""Packs a Tensor containing padded sequences of variable length.
    """
    if not isinstance(lengths, torch.Tensor):
        if torch._C._get_tracing_state():
            warnings.warn(
                "pack_padded_sequence has been called with a Python list of "
                "sequence lengths. The tracer cannot track the data flow of Python "
                "values, and it will treat them as constants, likely rendering "
                "the trace incorrect for any other combination of lengths.",
                stacklevel=2,
            )
        lengths = torch.as_tensor(lengths, dtype=torch.int64, device="cpu")
    else:
        lengths = lengths.to(dtype=torch.int64)

    if enforce_sorted:
        sorted_indices = None
    else:
        lengths, sorted_indices = torch.sort(lengths, descending=True)
        sorted_indices = sorted_indices.to(input.device)
        batch_dim = 0 if batch_first else 1
        input = input.index_select(batch_dim, sorted_indices)

    data, batch_sizes = _VF._pack_padded_sequence(input, lengths, batch_first)
    return _packed_sequence_init(data, batch_sizes, sorted_indices, None)
```

## 逐行讲解 / What's happening

1. **第 302-314 行 / Lines 302-314 (length normalization)**:
   - 中文: Python list 会被转成 CPU `int64` tensor；trace 模式会额外警告，因为 list 会被当常量。
   - English: A Python list is converted into a CPU `int64` tensor; tracing warns because the list becomes a constant.
2. **第 315-322 行 / Lines 315-322 (optional sort)**:
   - 中文: 如果 `enforce_sorted=False`，函数自己按长度降序排序，并同步重排 input 的 batch 维。
   - English: With `enforce_sorted=False`, the function sorts lengths descending and reorders the batch dimension.
3. **第 323-324 行 / Lines 323-324 (packed output)**:
   - 中文: 真正压紧序列的是 `_VF._pack_padded_sequence`，Python wrapper 负责把结果包装成 `PackedSequence`。
   - English: The actual packing is delegated to `_VF._pack_padded_sequence`; the wrapper returns a `PackedSequence`.

## 类比 / The analogy

这像把几摞高低不同的书先按高度排好，再按层装箱：第一层放所有书，第二层只放够高的书，空位不会进箱子。

It is like sorting stacks of books by height, then boxing them layer by layer: all stacks contribute to the first layer, only tall stacks to later layers, and empty padding never goes into the box.

## 自己跑一遍 / Try it yourself

```python
seqs = [[1, 2, 3], [4, 5, 0], [6, 0, 0]]
lengths = [3, 2, 1]
packed = []
for t in range(max(lengths)):
    for row, n in zip(seqs, lengths):
        if t < n:
            packed.append(row[t])
print(packed)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[1, 4, 6, 2, 5, 3]
```

中文: packed 顺序按时间层展开，而不是保留矩形 padding。  
English: The packed order walks time layer by time layer instead of keeping a padded rectangle.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **`pad_packed_sequence`** / **`pad_packed_sequence`**: 反向把 packed 数据恢复成 padded tensor。 / It reverses packed data back into a padded tensor.
- **NLP bucket batching** / **NLP bucket batching**: 训练前也常按长度排序或分桶来减少 padding。 / Training pipelines often sort or bucket by length to reduce padding.

## 注意事项 / Caveats / when it breaks

- **ONNX 排序要求** / **ONNX sorting requirement**: `enforce_sorted=True` 主要给导出场景用；普通训练常设为 `False` 更省心。 / `enforce_sorted=True` mostly matters for export; regular training often uses `False`.
- **长度不能在 CUDA list 里漂移** / **Lengths need clear device semantics**: list 长度会落到 CPU，tensor 长度会转 dtype。 / List lengths become CPU tensors, while tensor lengths keep device but convert dtype.

## 延伸阅读 / Further reading

- PyTorch `rnn.py`: https://github.com/pytorch/pytorch/blob/3d99896d477c69400c9206d00fa8bf7c4a139681/torch/nn/utils/rnn.py#L258-L324
