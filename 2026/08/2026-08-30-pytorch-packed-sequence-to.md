---
date: 2026-08-30
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/nn/utils/rnn.py
permalink: https://github.com/pytorch/pytorch/blob/460948b96a67002b7257ac4f3d6a192f70d61d27/torch/nn/utils/rnn.py#L104-L136
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, packed-sequence, device-transfer]
---

# PyTorch PackedSequence.to：只搬该搬的张量 / PyTorch PackedSequence.to: Move Only the Tensors That Should Move

> **一句话 / In one line**: `PackedSequence.to` 会转换 packed data 和索引张量，但刻意保留 `batch_sizes` 不跟着搬设备。 / `PackedSequence.to` converts the packed data and index tensors, but deliberately keeps `batch_sizes` from following device moves.

## 为什么重要 / Why this matters

RNN 的 packed sequence 不是一个普通 tensor；它同时保存压扁后的数据、每个时间步的 batch 大小、排序索引和反排序索引。PyTorch 在 `.to(...)` 里维护这些字段的不同设备语义，避免用户一调用 `.cuda()` 就破坏底层 RNN 约定。

An RNN packed sequence is not just a tensor; it stores flattened data, per-timestep batch sizes, sorted indices, and unsorted indices. PyTorch preserves the different device semantics of those fields inside `.to(...)`, so a user calling `.cuda()` does not break the low-level RNN contract.

## 代码 / The code

`pytorch/pytorch` — [`torch/nn/utils/rnn.py`](https://github.com/pytorch/pytorch/blob/460948b96a67002b7257ac4f3d6a192f70d61d27/torch/nn/utils/rnn.py#L104-L136)

```python
def to(self, *args: Any, **kwargs: Any) -> Self:
    r"""Perform dtype and/or device conversion on `self.data`.

    It has similar signature as :meth:`torch.Tensor.to`

    .. note::

        If the ``self.data`` Tensor already has the correct :class:`torch.dtype`
        and :class:`torch.device`, then ``self`` is returned.
        Otherwise, returns a copy with the desired configuration.
    """

    # Why not convert `batch_sizes`?
    # See NOTE [ device and dtype of a PackedSequence ]
    data = self.data.to(*args, **kwargs)
    if data is self.data:
        return self
    else:
        _device, _dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
            *args, **kwargs
        )

        # Does not forward device or dtype arg/kwargs, device is set from data.device
        def call_to(t: torch.Tensor) -> torch.Tensor:
            return t.to(
                data.device,
                non_blocking=non_blocking,
                memory_format=convert_to_format,
            )

        sorted_indices = bind(self.sorted_indices, call_to)
        unsorted_indices = bind(self.unsorted_indices, call_to)
        return type(self)(data, self.batch_sizes, sorted_indices, unsorted_indices)
```

## 逐行讲解 / What's happening

1. **第 104-114 行 / Lines 104-114**:
   - 中文: 这个方法承诺像 `Tensor.to` 一样接收 dtype/device 参数，但转换目标主要是 `self.data`。
   - English: The method accepts dtype/device arguments like `Tensor.to`, but the main conversion target is `self.data`.
2. **第 116-120 行 / Lines 116-120**:
   - 中文: `batch_sizes` 不转换；如果 `data.to(...)` 没产生新 tensor，整个 `PackedSequence` 直接原样返回。
   - English: `batch_sizes` is not converted; if `data.to(...)` returns the same tensor, the whole `PackedSequence` is returned unchanged.
3. **第 122-132 行 / Lines 122-132**:
   - 中文: `_parse_to` 拿出 `non_blocking` 和 `memory_format`，但索引张量只跟随 `data.device`，不跟随用户传入的 dtype。
   - English: `_parse_to` extracts `non_blocking` and `memory_format`, while index tensors follow `data.device` without inheriting the requested dtype.
4. **第 134-136 行 / Lines 134-136**:
   - 中文: 新对象复用旧的 `batch_sizes`，同时把排序索引搬到和数据相同的设备。
   - English: The new object reuses the old `batch_sizes` while moving sort indices to the same device as the data.

## 类比 / The analogy

这像搬乐队设备：乐器可以搬到新舞台，乐谱编号牌也要跟过去，但后台的演出时间表必须留在调度台上，不应该塞进乐器箱。

It is like moving a band's gear: instruments go to the new stage, and numbered tags follow them, but the backstage schedule stays at the control desk instead of being packed into an instrument case.

## 自己跑一遍 / Try it yourself

```python
class Packed:
    def __init__(self, data, batch_sizes, sorted_indices=None):
        self.data = data
        self.batch_sizes = batch_sizes
        self.sorted_indices = sorted_indices

    def to(self, device):
        data = (self.data[0], device)
        if data == self.data:
            return self
        idx = None if self.sorted_indices is None else (self.sorted_indices[0], device)
        return Packed(data, self.batch_sizes, idx)

p = Packed(("values", "cpu"), ("batch_sizes", "cpu"), ("order", "cpu"))
q = p.to("cuda")
print(q.data, q.batch_sizes, q.sorted_indices)
print(q.batch_sizes is p.batch_sizes)
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
('values', 'cuda') ('batch_sizes', 'cpu') ('order', 'cuda')
True
```

这个 toy 没有真正搬 tensor，但展示了 packed data 和 metadata 的设备策略并不相同。

This toy does not move real tensors, but it shows that packed data and metadata do not share the same device policy.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **`NestedTensor` helpers** / **`NestedTensor` helpers**: 容器里不同字段可能有不同 layout 或 device 约定。 / Different fields inside a container can carry different layout or device contracts.
- **DataLoader pin memory** / **DataLoader pin memory**: 递归处理 batch 时要保住原来的容器结构。 / Recursive batch handling preserves the original container structure.
- **Distributed metadata** / **Distributed metadata**: rank、shape、offset 这类控制信息常常不能和 payload 一起随便转换。 / Control metadata such as rank, shape, and offset often cannot be converted exactly like the payload.

## 注意事项 / Caveats / when it breaks

- **不要手工重组字段** / **Do not rebuild fields casually**: 错把 `batch_sizes` 放到 GPU 上可能触发底层 RNN 路径的隐含假设。 / Moving `batch_sizes` to GPU can violate assumptions in lower-level RNN paths.
- **索引 dtype 不跟随 data dtype** / **Index dtype does not follow data dtype**: 索引仍然应该保持索引语义，而不是变成 float。 / Indices should keep index semantics instead of becoming floats.
- **同对象返回是优化也是语义** / **Returning `self` is both optimization and semantics**: 无需转换时不制造新对象，调用方可以少一次分配。 / When no conversion is needed, no new object is allocated.

## 延伸阅读 / Further reading

- PyTorch RNN utilities: https://github.com/pytorch/pytorch/blob/460948b96a67002b7257ac4f3d6a192f70d61d27/torch/nn/utils/rnn.py
- Packed sequences docs: https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.PackedSequence.html
