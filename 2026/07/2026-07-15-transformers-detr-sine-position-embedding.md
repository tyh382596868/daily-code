---
date: 2026-07-15
topic: huggingface
source: huggingface
repo: huggingface/transformers
file: src/transformers/models/detr/modeling_detr.py
permalink: https://github.com/huggingface/transformers/blob/b7f0101522ddf1fb7b49aef9aff85fa22ceff36b/src/transformers/models/detr/modeling_detr.py#L299-L368
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, transformers, position-embedding]
---

# Transformers DETR：无 mask 时用 arange 保住 dtype / Transformers DETR: Use arange Without a Mask to Preserve dtype

> **一句话 / In one line**: DETR 的 sine position embedding 在无 mask 路径不用 `cumsum(ones)`，而是直接 `arange`，避免 `torch.compile` 半精度下 dtype 被改写。 / DETR's sine position embedding uses `arange` instead of `cumsum(ones)` on the no-mask path, avoiding a dtype rewrite under half-precision `torch.compile`.

## 为什么重要 / Why this matters

位置编码看起来是“纯数学小函数”，但在编译器里也可能踩 dtype 坑。无 mask 时，`cumsum(ones)` 数学上等价于 `1..H` / `1..W`；可是编译器可能把这个模式重写成用 ones 的 dtype，丢掉调用者要求的 `float16` / `bfloat16`。Transformers 在这里显式写成 `arange(..., dtype=dtype)`，把数学等价换成编译稳定。

Position embeddings look like small pure math helpers, but compiler rewrites can still change dtype behavior. Without a mask, `cumsum(ones)` is mathematically just `1..H` and `1..W`; a compiler rewrite can accidentally use the dtype of the ones tensor instead of the caller's `float16` or `bfloat16`. Transformers writes the no-mask path as `arange(..., dtype=dtype)`, trading an equivalent formula for a more compiler-stable one.

## 代码 / The code

`huggingface/transformers` — [`src/transformers/models/detr/modeling_detr.py`](https://github.com/huggingface/transformers/blob/b7f0101522ddf1fb7b49aef9aff85fa22ceff36b/src/transformers/models/detr/modeling_detr.py#L299-L368)

```python
@staticmethod
@compile_compatible_method_lru_cache(maxsize=1)
def build_sine_position_embedding(
    shape: torch.Size,
    device: torch.device | str,
    dtype: torch.dtype,
    num_position_features: int,
    normalize: bool = False,
    scale: float | None = None,
    temperature: int = 10000,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_size, _, height, width = shape
    if mask is None:
        # Without a mask this is just a cumsum over ones, written out as arange
        # instead: inductor's cumsum(ones) rewrite drops the requested dtype
        # (https://github.com/pytorch/pytorch/issues/189518), which breaks
        # float16/bfloat16 under torch.compile — don't revert to cumsum here
        # until that fix is widely released.
        y_embed = torch.arange(1, height + 1, dtype=dtype, device=device)[None, :, None].expand(
            batch_size, height, width
        )
        x_embed = torch.arange(1, width + 1, dtype=dtype, device=device)[None, None, :].expand(
            batch_size, height, width
        )
    else:
        embed_mask = mask.to(dtype)
        y_embed = embed_mask.cumsum(1)
        x_embed = embed_mask.cumsum(2)
    if normalize:
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    dim_t = torch.arange(num_position_features, dtype=torch.int64, device=device).to(dtype)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_position_features)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    return pos
```

## 逐行讲解 / What's happening

1. **`mask is None`**:
   - 中文: 没有 padding mask 时，每个像素都是有效像素，坐标就是规则网格。
   - English: With no padding mask, every pixel is valid and coordinates are a regular grid.
2. **`torch.arange(..., dtype=dtype)`**:
   - 中文: 直接生成目标 dtype 的坐标，避免先造 ones 再 `cumsum` 时被编译器重写。
   - English: It creates coordinates directly in the requested dtype, avoiding a compiler rewrite of `ones + cumsum`.
3. **`mask.to(dtype).cumsum`**:
   - 中文: 有 mask 时必须累计有效区域，所以仍然走 cumsum，但先把 mask 转到目标 dtype。
   - English: With a mask, cumulative valid-region coordinates are required, so cumsum remains, after casting the mask to the target dtype.
4. **sin/cos interleave**:
   - 中文: 偶数维走 sin，奇数维走 cos，再把 x/y 拼起来，得到 `(B, C, H, W)` 的图像位置编码。
   - English: Even channels use sine, odd channels use cosine; x and y are concatenated into `(B, C, H, W)` image position embeddings.

## 类比 / The analogy

这像给电影院座位贴编号。没有坏座位时，直接按行列打印 1、2、3 就行；有坏座位时，才需要一边走一边数“到这里为止有几个可用座位”。前者用 `arange`，后者用 `cumsum`。

It is like numbering seats in a theater. If every seat is usable, you print row and column numbers directly. If some seats are blocked, you walk through and count how many usable seats have appeared so far. The first case is `arange`; the second is `cumsum`.

## 自己跑一遍 / Try it yourself

```python
import numpy as np

batch, height, width = 1, 3, 4
y = np.arange(1, height + 1, dtype=np.float32)[None, :, None]
x = np.arange(1, width + 1, dtype=np.float32)[None, None, :]
y = np.broadcast_to(y, (batch, height, width))
x = np.broadcast_to(x, (batch, height, width))

mask = np.array([[[1, 1, 0, 0],
                  [1, 1, 1, 0],
                  [1, 1, 1, 1]]], dtype=np.float32)
y_masked = np.cumsum(mask, axis=1)
x_masked = np.cumsum(mask, axis=2)
print(x[0])
print(x_masked[0])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[[1. 2. 3. 4.]
 [1. 2. 3. 4.]
 [1. 2. 3. 4.]]
[[1. 2. 2. 2.]
 [1. 2. 3. 3.]
 [1. 2. 3. 4.]]
```

中文: 无 mask 是规则坐标；有 mask 时，坐标代表“累计有效像素数”。

English: Without a mask, coordinates are regular; with a mask, they count valid pixels cumulatively.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **ViT absolute position interpolation** / **ViT absolute-position interpolation**: 简单网格也会因为 dtype/device 和插值细节影响训练稳定性。 / Even simple grids depend on dtype, device, and interpolation details.
- **diffusion scheduler timesteps** / **Diffusion scheduler timesteps**: 采样步看似只是数组，但 dtype 和 device 必须跟模型路径一致。 / Timesteps look like arrays, but dtype and device must match the model path.

## 注意事项 / Caveats / when it breaks

- **mask 语义不同 / Mask semantics differ**: 这里的 mask 代表有效区域累计；如果你的 mask 是 invalid=True，需要先取反。
- **normalize 依赖最后一行/列 / Normalization uses the final row/column**: 全零 mask 会让归一化失去意义，只靠 `eps` 避免除零。
- **缓存 key 要包含 dtype / Cache keys must include dtype**: 这个函数被 compile-compatible LRU cache 包住，shape/device/dtype 变化都不能混淆。

## 延伸阅读 / Further reading

- [Transformers `modeling_detr.py`](https://github.com/huggingface/transformers/blob/b7f0101522ddf1fb7b49aef9aff85fa22ceff36b/src/transformers/models/detr/modeling_detr.py)
- [PyTorch issue 189518](https://github.com/pytorch/pytorch/issues/189518)
