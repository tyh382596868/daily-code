---
date: 2026-07-20
topic: diffusion
source: tracked
repo: facebookresearch/dinov3
file: dinov3/layers/attention.py
permalink: https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/layers/attention.py#L14-L130
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, dinov3, rope, attention]
---

# DINOv3 RoPE attention：只旋转图像 token，保留 prefix / DINOv3 RoPE Attention: Rotate Image Tokens, Keep the Prefix

> **一句话 / In one line**: DINOv3 的 attention 在 Q/K 上应用 RoPE 时先切出 prefix token，只把后面的 patch token 放进二维/多维位置编码。 / DINOv3 applies RoPE to Q/K by slicing off prefix tokens first, so only patch tokens receive positional rotation.

## 为什么重要 / Why this matters

视觉 transformer 常常不只有图像 patch，还有 class/register/prefix token。位置编码如果不分青红皂白地套到所有 token 上，prefix token 会被当成图像格点的一部分。DINOv3 的实现把 prefix 保留下来，只旋转真正带空间位置的 token。

Vision transformers often carry more than image patches: class, register, or prefix tokens may sit in front. If positional encoding is applied blindly, those prefix tokens are treated as grid locations. DINOv3 keeps the prefix untouched and rotates only tokens that actually live on the visual grid.

## 代码 / The code

`facebookresearch/dinov3` — [`dinov3/layers/attention.py`](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/layers/attention.py#L14-L130)

```python
def rope_rotate_half(x: Tensor) -> Tensor:
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    return (x * cos) + (rope_rotate_half(x) * sin)

class SelfAttention(nn.Module):
    def apply_rope(self, q: Tensor, k: Tensor, rope: Tensor | Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
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
        q = rope_apply(q[:, :, prefix:, :], sin, cos)
        q = torch.cat((q_prefix, q), dim=-2)
        k_prefix = k[:, :, :prefix, :]
        k = rope_apply(k[:, :, prefix:, :], sin, cos)
        k = torch.cat((k_prefix, k), dim=-2)
        q = q.to(dtype=q_dtype)
        k = k.to(dtype=k_dtype)
        return q, k

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

1. **半维旋转 / Half-dimension rotation**: 中文: `rope_rotate_half` 把向量拆成两半，第二半取负后放到前面，这是复数旋转的实数写法。 English: `rope_rotate_half` splits the vector in half and moves the negated second half forward, the real-valued form of complex rotation.
2. **sin/cos 混合 / Sin-cos mixing**: 中文: `rope_apply` 用 `x*cos + rotate(x)*sin` 把位置相位注入 Q/K。 English: `rope_apply` injects positional phase into Q/K with `x*cos + rotate(x)*sin`.
3. **prefix 长度来自差值 / Prefix length comes from the gap**: 中文: `N - sin.shape[-2]` 表示总 token 比 RoPE 网格多出的 token 数。 English: `N - sin.shape[-2]` is the number of tokens that do not belong to the RoPE grid.
4. **只旋转 patch token / Rotate only patch tokens**: 中文: `q[:, :, prefix:, :]` 和 `k[:, :, prefix:, :]` 才进 RoPE。 English: only `q[:, :, prefix:, :]` and `k[:, :, prefix:, :]` go through RoPE.
5. **恢复原 dtype / Restore original dtype**: 中文: 计算用 RoPE dtype，输出再转回 Q/K 原来的 dtype，避免混精训练里类型漂移。 English: computation uses the RoPE dtype, then Q/K are cast back to their original dtype for mixed precision stability.

## 类比 / The analogy

像给电影院座位贴行列号：观众席里的座位需要编号，门口检票员不该被贴成“第 0 排第 0 座”。prefix token 就是检票员，patch token 才是座位。

It is like labeling seats in a theater. Seats need row and column labels; the ticket checker at the door should not be labeled as seat `(0, 0)`. Prefix tokens are the ticket checker, while patch tokens are the seats.

## 自己跑一遍 / Try it yourself

```python
def rotate_half(x):
    n = len(x) // 2
    return [-v for v in x[n:]] + x[:n]

def rope_apply(x, sin, cos):
    r = rotate_half(x)
    return [round(a * c + b * s, 3) for a, b, s, c in zip(x, r, sin, cos)]

prefix = [[99, 99, 99, 99]]
patches = [[1, 2, 3, 4], [2, 0, 1, 0]]
sin = [0.0, 0.5, 0.0, 0.5]
cos = [1.0, 0.866, 1.0, 0.866]
print(prefix + [rope_apply(p, sin, cos) for p in patches])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[[99, 99, 99, 99], [1.0, -0.268, 3.0, 4.464], [2.0, 0.0, 1.0, 0.0]]
```

这个例子里第一行完全没动，说明 prefix token 没有被位置旋转污染。

The first row is unchanged, showing that prefix tokens are not polluted by positional rotation.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Wan2.1 3D RoPE** / **Wan2.1 3D RoPE**: 把时间、高度、宽度的频率拆开后再拼回每个 head。 / It splits temporal, height, and width frequencies before recombining them per head.
- **V-JEPA register tokens** / **V-JEPA register tokens**: register/prefix token 通常要和 patch token 分开处理。 / Register or prefix tokens usually need handling separate from patch tokens.

## 注意事项 / Caveats / when it breaks

- **RoPE 长度必须匹配 patch 数 / RoPE length must match patch count**: `prefix` 为负说明位置表比 token 还长。 / A negative `prefix` means the position table is longer than the token sequence.
- **prefix 不一定只有一个 / Prefix may be more than one token**: class token、register token、prompt token 都可能在前面。 / Class, register, and prompt tokens may all sit before patches.
- **只旋转 Q/K / Rotate Q/K only**: V 不带相位，否则 attention 权重和内容会混在一起。 / Values should not carry phase, or attention weights and content get entangled.

## 延伸阅读 / Further reading

- [DINOv3 attention.py](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/layers/attention.py#L14-L130)
- [DINOv3 repository](https://github.com/facebookresearch/dinov3)
