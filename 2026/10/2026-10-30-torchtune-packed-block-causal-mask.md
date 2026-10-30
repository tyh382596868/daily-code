---
date: 2026-10-30
topic: infrastructure
source: tracked
repo: pytorch/torchtune
file: torchtune/modules/attention_utils.py
permalink: https://github.com/pytorch/torchtune/blob/bd2a0fc7c31430972728494fa01aaeeb0ebf1ba1/torchtune/modules/attention_utils.py#L95-L174
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, infrastructure, packed-attention]
---

# torchtune 打包序列的块因果 mask / torchtune Packed Block-Causal Mask

> **一句话 / In one line**: 把同一行里的多个文档隔离开，同时保留每个文档内部的因果 attention。 / Isolate documents packed into one row while keeping causal attention inside each document.

## 为什么重要 / Why this matters

样本打包能消灭 padding，但如果没有边界 mask，后一个文档会偷看前一个文档的 token。这个函数把长度表编译成 attention 的许可规则。

Packing removes padding, but without a boundary mask a later document can read an earlier one. This function compiles lengths into an attention permission rule.

## 代码 / The code

`pytorch/torchtune` — [`attention_utils.py`](https://github.com/pytorch/torchtune/blob/bd2a0fc7c31430972728494fa01aaeeb0ebf1ba1/torchtune/modules/attention_utils.py#L133-L174)

```python
if _SUPPORTS_FLEX_ATTENTION:
    document_ids = _get_document_ids_from_seq_lens(seq_lens)
    batch_size, max_seq_len = document_ids.shape
    document_ids = document_ids.to("cuda")

    def mask_mod(b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        document_mask = document_ids[b, q_idx] == document_ids[b, kv_idx]
        return causal_mask & document_mask

    return create_block_causal_mask_flex(mask_mod, batch_size, None, max_seq_len, max_seq_len)
```

## 逐行讲解 / What's happening

1. 中文：`document_ids` 给 pack 中每个位置贴上原始文档编号。
   English: `document_ids` labels every packed position with its source document.
2. 中文：`q_idx >= kv_idx` 保住自回归的“只看过去”。
   English: `q_idx >= kv_idx` preserves the autoregressive past-only rule.
3. 中文：相等的 document id 再加一道墙；两条规则取交集。
   English: Equal document ids add the wall between examples; the two rules are intersected.

## 类比 / The analogy

像把几封信塞进同一个信封袋：省了包装空间，但每封信仍要有自己的封口。

It is like putting several letters in one mailer: you save packaging, but each letter still needs its own seal.

## 自己跑一遍 / Try it yourself

```python
import torch
doc = torch.tensor([0, 0, 0, 1, 1])
i = torch.arange(len(doc))
mask = (i[:, None] >= i[None, :]) & (doc[:, None] == doc[None, :])
print(mask.int())
```

运行 / Run with: `python try.py`

预期输出 / Expected output: 两个互不相连的下三角块。 / Two disconnected lower-triangular blocks.

中文：只用下三角会造成样本泄漏；只用块对角又会泄漏未来。 English: A triangle alone leaks examples; blocks alone leak the future.

## 注意事项 / Caveats / when it breaks

- 中文：`document_ids` 必须与真实 pack 布局一致。 / English: `document_ids` must exactly match the actual packed layout.

## 延伸阅读 / Further reading

- [torchtune attention utilities](https://github.com/pytorch/torchtune/blob/bd2a0fc7c31430972728494fa01aaeeb0ebf1ba1/torchtune/modules/attention_utils.py#L95-L174)
