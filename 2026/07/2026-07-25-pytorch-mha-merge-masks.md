---
date: 2026-07-25
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/nn/modules/activation.py
permalink: https://github.com/pytorch/pytorch/blob/a4116fb2da1229203a0b8049fa306dd15cfb596b/torch/nn/modules/activation.py#L1525-L1573
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, attention-mask]
---

# PyTorch MultiheadAttention：把两种 mask 展成同一张四维表 / PyTorch MultiheadAttention: Expand Two Masks into One 4D Table

> **一句话 / In one line**: MHA fast path 要的是统一 mask, 所以 PyTorch 把 attention mask 和 padding mask 都广播到 `(B,H,L,L)`。 / The MHA fast path wants one mask, so PyTorch broadcasts attention and padding masks to `(B,H,L,L)`.

## 为什么重要 / Why this matters

多头注意力里有两种常见遮罩: token 之间能不能互相看, 以及 batch 里哪些位置只是 padding。`merge_masks` 的价值在于把这两种语义统一成一个形状, 后面的 kernel 不再关心 mask 来自哪里。

Multi-head attention has two common masks: which token pairs may attend, and which batch positions are padding. `merge_masks` turns those semantics into one shape, letting later kernels ignore where the mask came from.

## 代码 / The code

`pytorch/pytorch` -- [`torch/nn/modules/activation.py`](https://github.com/pytorch/pytorch/blob/a4116fb2da1229203a0b8049fa306dd15cfb596b/torch/nn/modules/activation.py#L1525-L1573)

```python
    def merge_masks(
        self,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        query: Tensor,
    ) -> tuple[Tensor | None, int | None]:
        r"""Determine mask type and combine masks if necessary.

        If only one mask is provided, that mask
        and the corresponding mask type will be returned. If both masks are provided, they will be both
        expanded to shape ``(batch_size, num_heads, seq_len, seq_len)``, combined with logical ``or``
        and mask type 2 will be returned
        Args:
            attn_mask: attention mask of shape ``(seq_len, seq_len)``, mask type 0
            key_padding_mask: padding mask of shape ``(batch_size, seq_len)``, mask type 1
            query: query embeddings of shape ``(batch_size, seq_len, embed_dim)``
        Returns:
            merged_mask: merged mask
            mask_type: merged mask type (0, 1, or 2)
        """
        mask_type: int | None = None
        merged_mask: Tensor | None = None

        if key_padding_mask is not None:
            mask_type = 1
            merged_mask = key_padding_mask

        if attn_mask is not None:
            # In this branch query can't be a nested tensor, so it has a shape
            batch_size, seq_len, _ = query.shape
            mask_type = 2

            # Always expands attn_mask to 4D
            if attn_mask.dim() == 3:
                attn_mask_expanded = attn_mask.view(batch_size, -1, seq_len, seq_len)
            else:  # attn_mask.dim() == 2:
                attn_mask_expanded = attn_mask.view(1, 1, seq_len, seq_len).expand(
                    batch_size, self.num_heads, -1, -1
                )
            merged_mask = attn_mask_expanded

            if key_padding_mask is not None:
                key_padding_mask_expanded = key_padding_mask.view(
                    batch_size, 1, 1, seq_len
                ).expand(-1, self.num_heads, -1, -1)
                merged_mask = attn_mask_expanded + key_padding_mask_expanded

        # no attn_mask and no key_padding_mask, returns None, None
        return merged_mask, mask_type
```

## 逐行讲解 / What's happening

1. **第 1548-1550 行 / Lines 1548-1550 (`只有 padding mask`)**:
   - 中文: 直接返回 `(B,L)` 并标成 type 1。
   - English: With only padding, it returns the `(B,L)` mask and marks it type 1.
2. **第 1552-1564 行 / Lines 1552-1564 (`有 attn_mask`)**:
   - 中文: 先取 batch 和 seq_len, 再把 2D/3D attn mask 展成四维。
   - English: With an attention mask, it reads batch and sequence length, then expands 2D or 3D masks to 4D.
3. **第 1566-1570 行 / Lines 1566-1570 (`两种 mask 同时存在`)**:
   - 中文: padding mask 变成 `(B,H,1,L)`, 加到 attention mask 上。
   - English: When both masks exist, padding becomes `(B,H,1,L)` and is added to the attention mask.
4. **第 1572-1573 行 / Lines 1572-1573 (`空 mask`)**:
   - 中文: 没有任何 mask 时返回 `(None, None)`, fast path 可以少做事。
   - English: With no masks, it returns `(None, None)` so the fast path can avoid extra work.

## 类比 / The analogy

这像电影院检票。一个规则说哪些座位被封, 另一个规则说哪些票是空座占位。入口处把两张表合成一张座位图, 放映厅工作人员只看最终图。

It is like movie-theater seating. One rule blocks seats, another marks placeholder tickets. The entrance merges both into one seating chart, and the staff inside only reads that final chart.

## 自己跑一遍 / Try it yourself

```python
B, H, L = 2, 3, 4
attn = [[1 if j > i else 0 for j in range(L)] for i in range(L)]
pad = [[0, 0, 1, 1], [0, 1, 0, 1]]
merged = []
for b in range(B):
    heads = []
    for _ in range(H):
        heads.append([[attn[i][j] + pad[b][j] for j in range(L)] for i in range(L)])
    merged.append(heads)
print((len(merged), len(merged[0]), len(merged[0][0]), len(merged[0][0][0])))
for row in merged[0][0]:
    print(row)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
(2, 3, 4, 4)
[0, 1, 2, 2]
[0, 0, 2, 2]
[0, 0, 1, 2]
[0, 0, 1, 1]
```

padding 只沿 key 维度广播, 所以它会影响每一行 query 对这些 key 的可见性。

Padding broadcasts along the key dimension, so every query row sees those key positions as blocked.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **FlashAttention 前处理**: 高性能 attention 通常先把 ragged/mask 信息变成 kernel 喜欢的统一布局。 / High-performance attention often converts ragged or mask metadata into a kernel-friendly layout first.
- **VLA prefix/suffix mask**: prefix-LM 和 action suffix 也常用广播 mask 表达可见性规则。 / Prefix-LM plus action suffix setups also encode visibility through broadcast masks.

## 注意事项 / Caveats / when it breaks

- **加法依赖 dtype 语义**: bool mask 和 float mask 的上游规范化会影响 `+` 的含义。 / The addition depends on dtype semantics. Upstream canonicalization matters.
- **四维 mask 成本更高**: 长序列下 `(B,H,L,L)` 很大, 所以 fast path 会尽量避免不必要 mask。 / A 4D mask is expensive for long sequences, so fast paths avoid it when possible.

## 延伸阅读 / Further reading

- PyTorch MultiheadAttention docs: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
- Source permalink in this note
