---
date: 2026-08-04
topic: infrastructure
source: tracked
repo: Dao-AILab/flash-attention
file: flash_attn/bert_padding.py
permalink: https://github.com/Dao-AILab/flash-attention/blob/df61ab6c4a0fb1f94f1f43b2a23479a0ab92b8ab/flash_attn/bert_padding.py#L98-L128
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, infrastructure, flash-attention, padding]
---

# FlashAttention unpad_input：先把有效 token 压紧 / FlashAttention unpad_input: Compact Valid Tokens First

> **一句话 / In one line**: `unpad_input` 把 padded batch 展平成有效 token 列表，并生成 varlen attention kernel 需要的索引和 cumulative lengths。 / `unpad_input` flattens a padded batch into valid tokens and produces the indices plus cumulative lengths needed by varlen attention kernels.

## 为什么重要 / Why this matters

FlashAttention 的高吞吐来自“只算真正存在的 token”。在变长 batch 里，如果继续保留矩形 padding，kernel 会在大量 0 token 上浪费带宽和 SRAM。`unpad_input` 是进入 varlen kernel 前的收纳步骤：压紧数据，同时保留足够的索引信息，方便之后恢复原 batch 形状。

FlashAttention gets much of its throughput from computing only real tokens. In variable-length batches, keeping the padded rectangle wastes memory traffic and SRAM on zeros. `unpad_input` is the packing step before the varlen kernel: compact the data, but keep enough index metadata to restore the original batch later.

## 代码 / The code

`Dao-AILab/flash-attention` — [`flash_attn/bert_padding.py`](https://github.com/Dao-AILab/flash-attention/blob/df61ab6c4a0fb1f94f1f43b2a23479a0ab92b8ab/flash_attn/bert_padding.py#L98-L128)

```python
def unpad_input(hidden_states, attention_mask, unused_mask=None):
    """
    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
        unused_mask: (batch, seqlen), bool / int, 1 means the element is allocated but unused.
    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens selected in attention_mask + unused_mask.
        indices: (total_nnz), the indices of masked tokens from the flattened input sequence.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
        seqused: (batch), returns the number of tokens selected in attention_mask + unused_mask.
    """
    all_masks = (attention_mask + unused_mask) if unused_mask is not None else attention_mask
    seqlens_in_batch = all_masks.sum(dim=-1, dtype=torch.int32)
    used_seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    # TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
    # bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
    # times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
    # index with integer indices. Moreover, torch's index is a bit slower than it needs to be,
    # so we write custom forward and backward to make it a bit faster.
    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
        used_seqlens_in_batch,
    )
```

## 逐行讲解 / What's happening

1. **第 111 行 / Line 111 (`all_masks`)**:
   - 中文: `unused_mask` 允许“分配了但不用”的位置也进入索引统计，这对某些 packed/concatenated 训练布局很有用。
   - English: `unused_mask` lets allocated-but-unused positions participate in indexing, which helps some packed or concatenated training layouts.
2. **第 112-116 行 / Lines 112-116 (length metadata)**:
   - 中文: 每个样本的有效长度、最大长度和 `cu_seqlens` 都在这里生成，后者就是 varlen kernel 切分 flattened token 的边界表。
   - English: Per-sample lengths, max length, and `cu_seqlens` are built here; `cu_seqlens` is the boundary table used by varlen kernels to slice flattened tokens.
3. **第 114、123 行 / Lines 114 and 123 (integer indexing)**:
   - 中文: 代码先把 mask 变成整数下标，再对 `(batch * seqlen)` 维做 gather，避免 bool mask 展开带来的额外内存。
   - English: The code turns the mask into integer indices, then gathers along the flattened `(batch * seqlen)` axis to avoid the extra memory expansion of boolean indexing.

## 类比 / The analogy

这像登机前把每排空座位去掉，只让乘客排成一条队，同时给每个航班留一张“第几个乘客开始属于下一排”的表。安检只处理真人，登机时还能按表回到原座位。

It is like removing empty seats before boarding: passengers form one compact line, while a boundary sheet records where each row starts. Security handles only people, and boarding can still reconstruct the original seating.

## 自己跑一遍 / Try it yourself

```python
tokens = [["a", "b", "_", "_"], ["c", "d", "e", "_"]]
mask = [[1, 1, 0, 0], [1, 1, 1, 0]]

flat, indices, lengths = [], [], []
for row_i, row in enumerate(tokens):
    count = 0
    for col_i, tok in enumerate(row):
        if mask[row_i][col_i]:
            flat.append(tok)
            indices.append(row_i * len(row) + col_i)
            count += 1
    lengths.append(count)

cu = [0]
for n in lengths:
    cu.append(cu[-1] + n)
print(flat)
print(indices)
print(cu)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
['a', 'b', 'c', 'd', 'e']
[0, 1, 4, 5, 6]
[0, 2, 5]
```

中文: `flat` 是 kernel 真正要处理的 token，`cu` 告诉它每个样本在哪里切开。
English: `flat` is what the kernel really processes, while `cu` tells it where each sample starts and ends.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **PyTorch `PackedSequence`** / **PyTorch `PackedSequence`**: RNN 也会把 padded 序列压成紧凑表示。 / RNN utilities also compact padded sequences before computation.
- **vLLM paged attention** / **vLLM paged attention**: serving 系统同样用索引表管理非矩形 token 布局。 / Serving systems also use index tables to manage non-rectangular token layouts.

## 注意事项 / Caveats / when it breaks

- **恢复形状依赖 `indices`** / **Restoration depends on `indices`**: 如果后续丢了 indices，就只能得到一条 flattened token 流。 / If later code drops the indices, it only has a flat token stream.
- **mask dtype 要清楚** / **Mask dtype must be clear**: bool/int mask 混用时要确认加法后的语义仍是“选中”。 / When mixing bool and integer masks, make sure addition still means “selected.”

## 延伸阅读 / Further reading

- FlashAttention `bert_padding.py`: https://github.com/Dao-AILab/flash-attention/blob/df61ab6c4a0fb1f94f1f43b2a23479a0ab92b8ab/flash_attn/bert_padding.py#L98-L128
