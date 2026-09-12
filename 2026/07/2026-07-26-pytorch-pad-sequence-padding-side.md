---
date: 2026-07-26
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/nn/utils/rnn.py
permalink: https://github.com/pytorch/pytorch/blob/main/torch/nn/utils/rnn.py
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, pytorch, rnn, padding]
---

# PyTorch pad_sequence：padding 也有左右方向 / PyTorch pad_sequence: Padding Has a Direction Too

> **一句话 / In one line**: `pad_sequence` 不只补齐长度，还把 batch 维度和左/右 padding 策略固定成统一张量。 / `pad_sequence` does more than equalize lengths; it also fixes batch layout and left/right padding policy.

## 为什么重要 / Why this matters

序列模型里，padding 位置会影响 mask、相对位置、最后 token 取法和 KV cache 对齐。PyTorch 把 `batch_first` 和 `padding_side` 都放到 `pad_sequence` 的入口，等于把“怎么补齐”变成显式数据契约。

In sequence models, padding position affects masks, relative positions, last-token selection, and KV-cache alignment. By exposing both `batch_first` and `padding_side`, PyTorch turns padding into an explicit data contract.

## 代码 / The code

`pytorch/pytorch` — [`torch/nn/utils/rnn.py`](https://github.com/pytorch/pytorch/blob/main/torch/nn/utils/rnn.py)

```python
def pad_sequence(
    sequences,
    batch_first: bool = False,
    padding_value: float = 0.0,
    padding_side: str = "right",
):
    if not (torch.jit.is_tracing() or torch.jit.is_scripting()):
        if not isinstance(sequences, Iterable):
            raise RuntimeError("pad_sequence: Expected iterable for input sequences")
        sequences = tuple(sequences)
    else:
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.unbind(0)

    return torch._C._nn.pad_sequence(
        sequences,
        batch_first,
        padding_value,
        padding_side,
    )
```

## 逐行讲解 / What's happening

1. **参数列表 / Signature**
   - 中文: `batch_first` 决定输出是 `B x T x *` 还是 `T x B x *`，`padding_side` 决定短序列补在左边还是右边。
   - English: `batch_first` chooses `B x T x *` versus `T x B x *`, while `padding_side` chooses where short sequences are padded.
2. **`Iterable` 检查 / Iterable check**
   - 中文: eager 模式下先把输入转成 tuple，避免迭代器只能消费一次。
   - English: In eager mode, the function materializes a tuple so one-shot iterators are not consumed unpredictably.
3. **JIT 分支 / JIT branch**
   - 中文: tracing/scripting 时如果传进来的是 tensor，就沿第 0 维拆成多个序列，方便图捕获。
   - English: Under tracing or scripting, a tensor input is unbound along dimension 0 so graph capture can represent the sequence list.
4. **C++ kernel / C++ kernel**
   - 中文: 真正填充交给底层实现，Python 层只负责输入规范化。
   - English: The actual filling happens in the backend; Python owns input normalization.

## 类比 / The analogy

这像把不同长度的票据装进文件夹：你可以让所有票据左边对齐，也可以右边对齐，但必须提前告诉归档员，否则后面查日期的人会看错列。

It is like filing receipts of different lengths: you can align them on the left or right edge, but the clerk must know the rule before anyone reads dates from a column.

## 自己跑一遍 / Try it yourself

```python
def pad(rows, value=0, side="right"):
    width = max(map(len, rows))
    out = []
    for row in rows:
        gap = [value] * (width - len(row))
        out.append(row + gap if side == "right" else gap + row)
    return out

print(pad([[1, 2, 3], [4]], side="right"))
print(pad([[1, 2, 3], [4]], side="left"))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
[[1, 2, 3], [4, 0, 0]]
[[1, 2, 3], [0, 0, 4]]
```

中文: 左 padding 常用于 decoder-only LM，因为最后一个真实 token 会落在右侧。
English: Left padding is common for decoder-only LMs because the last real token stays on the right edge.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **tokenizer padding** / **Tokenizer padding**: 中文: HF tokenizer 也显式区分 `padding_side`。 / English: HF tokenizers also expose `padding_side` explicitly.
- **attention masks** / **Attention masks**: 中文: mask 必须和 padding 方向一致。 / English: Masks must match the padding direction.

## 注意事项 / Caveats / when it breaks

- **尾部读取** / **Reading the tail**: 中文: 右 padding 时不能简单取最后一列当最后 token。 / English: With right padding, the last column may be padding, not the last token.
- **形状一致** / **Shape consistency**: 中文: 除时间维以外的维度必须一致。 / English: All non-time dimensions must match.

## 延伸阅读 / Further reading

- [PyTorch `pad_sequence` docs](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html)
- [PyTorch source file](https://github.com/pytorch/pytorch/blob/main/torch/nn/utils/rnn.py)
