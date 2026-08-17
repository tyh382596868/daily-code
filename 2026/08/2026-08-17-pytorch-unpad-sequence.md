---
date: 2026-08-17
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/nn/utils/rnn.py
permalink: https://github.com/pytorch/pytorch/blob/e1f1b393a1b21a2989ed88ac10385aa45badee8f/torch/nn/utils/rnn.py#L473-L519
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, pytorch, rnn]
---

# PyTorch unpad_sequence：用长度 mask 把 padding 拆回列表 / PyTorch unpad_sequence: Use Length Masks to Split Padding Back Out

> **一句话 / In one line**: `unpad_sequence` 根据每个样本的真实长度创建布尔 mask，把 padded batch 还原成变长 tensor 列表。 / `unpad_sequence` builds a boolean mask from each sequence length and restores a padded batch into a list of variable-length tensors.

## 为什么重要 / Why this matters

训练 RNN 或序列模型时，batch 里常用 padding 对齐长度；但评估、可视化、loss 后处理经常要回到原始变长序列。这个函数的价值在于它不猜 padding 值，只相信显式的 `lengths`。

Batches often use padding to align sequence lengths, but evaluation, visualization, and loss post-processing often need the original variable-length items back. The key is that this function does not infer padding values; it trusts explicit `lengths`.

## 代码 / The code

`pytorch/pytorch` — [`torch/nn/utils/rnn.py`](https://github.com/pytorch/pytorch/blob/e1f1b393a1b21a2989ed88ac10385aa45badee8f/torch/nn/utils/rnn.py#L473-L519)

```python
def unpad_sequence(
    padded_sequences: Tensor,
    lengths: Tensor,
    batch_first: bool = False,
) -> list[Tensor]:
    r"""Unpad padded Tensor into a list of variable length Tensors."""
    unpadded_sequences = []

    if not batch_first:
        padded_sequences.transpose_(0, 1)

    max_length = padded_sequences.shape[1]
    idx = torch.arange(max_length, device=lengths.device)

    for seq, length in zip(padded_sequences, lengths, strict=True):
        mask = idx < length
        unpacked_seq = seq[mask]
        unpadded_sequences.append(unpacked_seq)

    return unpadded_sequences
```

## 逐行讲解 / What's happening

1. **第 506 行 / Line 506**: 中文: 结果不是再拼一个 tensor，而是 list，因为每个序列长度可能不同。 / English: The result is a list rather than another tensor because each item can have a different length.
2. **第 508-509 行 / Lines 508-509**: 中文: 如果输入是 `T x B x *`，先原地转成 `B x T x *`，让后面的循环按 batch 遍历。 / English: If the input is `T x B x *`, it is transposed in place to `B x T x *` so the loop walks batch items.
3. **第 511-516 行 / Lines 511-516**: 中文: `idx < length` 生成一条长度 mask，短序列后面的 padding 位置自然被丢掉。 / English: `idx < length` creates a length mask, naturally dropping padded positions at the tail.
4. **第 514 行 / Line 514**: 中文: `strict=True` 要求 `padded_sequences` 和 `lengths` 数量完全一致。 / English: `strict=True` requires the number of padded rows and lengths to match exactly.

## 类比 / The analogy

像把三张不同长度的收据都贴在同一张 A4 纸上，`lengths` 是每张收据真实结束的位置。拆回去时看剪裁线，而不是看纸上的空白。

It is like taping receipts of different lengths onto the same sheet of paper. `lengths` marks where each receipt actually ends, so you cut by the mark rather than guessing from blank space.

## 自己跑一遍 / Try it yourself

```python
def unpad(padded, lengths):
    width = len(padded[0])
    idx = list(range(width))
    out = []
    if len(padded) != len(lengths):
        raise ValueError("padded and lengths must have the same batch size")
    for seq, length in zip(padded, lengths):
        out.append([value for value, i in zip(seq, idx) if i < length])
    return out

print(unpad([[1, 2, 0], [3, 0, 0], [4, 5, 6]], [2, 1, 3]))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[[1, 2], [3], [4, 5, 6]]
```

注意第三个序列里没有 padding，但它仍然走同一条 mask 逻辑。

The third sequence has no padding, but it still goes through the same mask logic.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **attention mask** / **attention masks**: 语言模型用同样的长度 mask 阻止 attention 看见 padding。 / Language models use the same length-mask idea to keep attention away from padding.
- **ragged batch 后处理** / **Ragged batch post-processing**: 检测框、轨迹点和 token span 都常用显式长度拆回列表。 / Boxes, trajectories, and token spans often use explicit lengths to unpack ragged batches.

## 注意事项 / Caveats / when it breaks

- **会原地 transpose** / **It transposes in place**: `batch_first=False` 时输入 tensor 的视图会被原地转置，调用方不要假设原对象布局不变。 / With `batch_first=False`, the input is transposed in place, so callers should not assume the original layout is untouched.
- **长度才是真相** / **Lengths are the source of truth**: 如果 `lengths` 错了，函数不会从 padding 值自动修正。 / If `lengths` are wrong, the function will not infer a correction from padding values.

## 延伸阅读 / Further reading

- [PyTorch source](https://github.com/pytorch/pytorch/blob/e1f1b393a1b21a2989ed88ac10385aa45badee8f/torch/nn/utils/rnn.py#L473-L519)
