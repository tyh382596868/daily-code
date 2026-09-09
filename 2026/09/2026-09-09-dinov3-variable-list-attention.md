---
date: 2026-09-09
topic: diffusion
source: tracked
repo: facebookresearch/dinov3
file: dinov3/layers/attention.py
permalink: https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/layers/attention.py#L43-L118
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, diffusion, self-attention, variable-length, rope]
---

# DINOv3 变长 attention：先合并投影，再保住每段边界 / DINOv3 Variable-Length Attention: Fuse the Projection, Keep Each Segment

> **一句话 / In one line**: `forward_list` 把不同长度的 token 序列拼起来只做一次 QKV 投影，算完 attention 后再按原形状拆回去。 / `forward_list` concatenates variable-length token sequences for one fused QKV projection, then restores the original shapes after attention.

## 为什么重要 / Why this matters

中文：视觉模型经常同时处理不同分辨率、不同裁剪尺寸的图像。为了凑成规则 batch 而强行 padding，会浪费显存和 attention 计算。DINOv3 的 `forward_list` 选择把序列暂时拼平，用一次 `self.qkv`，再按每段的 shape 拆回去；这是一种很实用的“保持变长语义、共享矩阵乘法”的折中。

English: Vision systems often process crops or views with different token counts. Padding every sample to the same length makes the tensor convenient but wastes memory and attention work. DINOv3’s `forward_list` keeps each sequence variable-length, concatenates only for the shared QKV projection, and reconstructs the list afterward.

## 代码 / The code

`facebookresearch/dinov3` — [`dinov3/layers/attention.py`](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/layers/attention.py#L43-L118)

```python
class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_k_bias: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        linear_class = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear_class(dim, dim * 3, bias=qkv_bias, device=device)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

    def apply_rope(self, q: Tensor, k: Tensor, rope: Tensor | Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        # All operations will use the dtype of rope, the output is cast back to the dtype of q and k
        q_dtype = q.dtype
        k_dtype = k.dtype
        sin, cos = rope
        rope_dtype = sin.dtype
        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)
        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        assert prefix >= 0
        q_prefix = q[:, :, :prefix, :]
        q = rope_apply(q[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        q = torch.cat((q_prefix, q), dim=-2)  # [B, head, N, D//head]
        k_prefix = k[:, :, :prefix, :]
        k = rope_apply(k[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        k = torch.cat((k_prefix, k), dim=-2)  # [B, head, N, D//head]
        q = q.to(dtype=q_dtype)
        k = k.to(dtype=k_dtype)
        return q, k

    def forward(self, x: Tensor, attn_bias=None, rope: Tensor = None) -> Tensor:
        qkv = self.qkv(x)
        attn_v = self.compute_attention(qkv=qkv, attn_bias=attn_bias, rope=rope)
        x = self.proj(attn_v)
        x = self.proj_drop(x)
        return x

    def forward_list(self, x_list, attn_bias=None, rope_list=None) -> List[Tensor]:
        assert len(x_list) == len(rope_list)  # should be enforced by the Block
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        qkv_flat = self.qkv(x_flat)
        qkv_list = uncat_with_shapes(qkv_flat, shapes, num_tokens)
        att_out = []
        for _, (qkv, _, rope) in enumerate(zip(qkv_list, shapes, rope_list)):
            att_out.append(self.compute_attention(qkv, attn_bias=attn_bias, rope=rope))
        x_flat, shapes, num_tokens = cat_keep_shapes(att_out)
        x_flat = self.proj(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)

    def compute_attention(self, qkv: Tensor, attn_bias=None, rope=None) -> Tensor:
        assert attn_bias is None
        B, N, _ = qkv.shape
        C = self.qkv.in_features

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2)
        return x.reshape([B, N, C])
```

## 逐行讲解 / What's happening

1. **第 60-63 行 / Lines 60-63 (`self.qkv` and `self.proj`)**:
   - 中文: Q、K、V 先由一个输出宽度为 `3 * dim` 的 Linear 一次性产生，attention 结果再过一个输出投影。
   - English: One Linear emits all three of Q, K, and V with width `3 * dim`; the attended result then goes through one output projection.
2. **第 66-85 行 / Lines 66-85 (`apply_rope`)**:
   - 中文: 代码保留 prefix 不旋转，只对带有空间位置的 suffix 应用 RoPE；计算时暂时转成 RoPE 的 dtype，最后转回原 dtype。
   - English: The prefix stays untouched while RoPE is applied only to the spatially positioned suffix. Computation uses the RoPE dtype and casts back afterward.
3. **第 94-104 行 / Lines 94-104 (`forward_list`)**:
   - 中文: `cat_keep_shapes` 记录每段边界，拼平后只做一次 QKV 投影。attention 仍逐段执行，避免不同长度序列互相看到，最后再拼平做一次输出投影并拆回列表。
   - English: `cat_keep_shapes` records boundaries, allowing one fused QKV projection. Attention still runs per segment so unrelated sequences cannot interact; the outputs are flattened for one projection and then restored.
4. **第 106-118 行 / Lines 106-118 (`compute_attention`)**:
   - 中文: `3 * dim` 被 reshape 成 `[B, N, 3, heads, head_dim]`，再交换维度给 SDPA。
   - English: The `3 * dim` projection is reshaped to `[B, N, 3, heads, head_dim]`, transposed into the layout expected by SDPA, and reshaped back.

## 类比 / The analogy

中文：像把三种尺寸的快递先贴上区间标签，统一送进同一台分拣机，分拣完成后再按标签把包裹放回各自的传送带。机器共享了，但不同订单没有混在一起。

English: Imagine sending parcels from three differently sized orders through one sorting machine. Each parcel carries its order boundaries, so the machine is shared without mixing one customer’s package with another’s.

## 自己跑一遍 / Try it yourself

```python
def flatten_with_shapes(sequences):
    flat = [token for seq in sequences for token in seq]
    sizes = [len(seq) for seq in sequences]
    return flat, sizes

def restore(flat, sizes):
    out, start = [], 0
    for size in sizes:
        out.append(flat[start:start + size])
        start += size
    return out

parts = [["a", "b"], ["c"], ["d", "e", "f"]]
flat, sizes = flatten_with_shapes(parts)
print(flat, sizes)
print(restore(flat, sizes))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
['a', 'b', 'c', 'd', 'e', 'f'] [2, 1, 3]
[['a', 'b'], ['c'], ['d', 'e', 'f']]
```

中文：真正的优化点不是“把 list 变成 tensor”，而是只在可以共享的 Linear 上合并，序列之间的 attention 语义仍然隔离。

English: The optimization is not simply turning a list into one tensor. It merges only the shareable Linear work while keeping attention semantics isolated between sequences.

## 注意事项 / Caveats / when it breaks

- **不要把不同样本直接拼进同一个 attention 序列** / **Do not concatenate unrelated samples into one attention sequence**: `forward_list` 必须逐段调用 `compute_attention`，否则会发生跨样本注意力。
- **RoPE 的长度必须匹配 suffix** / **RoPE length must match the suffix**: `prefix = N - sin.shape[-2]` 依赖位置编码只覆盖最后那段 token。
- **`attn_bias` 在这里被拒绝** / **`attn_bias` is rejected here**: `compute_attention` 明确要求它为 `None`，扩展这条路径需要同时设计 bias 的分段语义。

## 延伸阅读 / Further reading

- [DINOv3 attention.py](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/layers/attention.py)
- [PyTorch scaled_dot_product_attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
