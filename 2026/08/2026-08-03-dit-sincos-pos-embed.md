---
date: 2026-08-03
topic: diffusion
source: tracked
repo: facebookresearch/DiT
file: models.py
permalink: https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L274-L321
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, positional-embedding, dit]
---

# DiT sin/cos 位置编码：把二维网格切成两把频率尺 / DiT Sin/Cos Position Embedding: Split a 2D Grid into Two Frequency Rulers

> **一句话 / In one line**: DiT 用固定 sin/cos 表示 patch 的二维坐标，一半通道写横坐标，一半通道写纵坐标。 / DiT encodes patch coordinates with a fixed sin/cos table, spending half the channels on width and half on height.

## 为什么重要 / Why this matters

扩散 Transformer 看到的是一串 patch token，但图像本身是二维网格。如果位置编码不把行列关系带进去，模型就很难知道相邻 patch、同行 patch、不同区域之间的空间结构。

A diffusion transformer receives a sequence of patch tokens, while the image is a 2D grid. Without a coordinate signal, the model has to infer spatial relations such as neighboring patches, rows, and regions from content alone.

## 代码 / The code

`facebookresearch/DiT` — [`models.py`](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L274-L321)

```python
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
```

## 逐行讲解 / What's happening

1. **第 280-285 行 / Lines 280-285 (grid build)**:
   - 中文: 先构造 `grid_w` 和 `grid_h`，再把二维坐标整理成 `[2, 1, H, W]`。
   - English: It builds width and height coordinate arrays, then reshapes them into `[2, 1, H, W]`.
2. **第 292-300 行 / Lines 292-300 (split channels)**:
   - 中文: `embed_dim` 一分为二，分别编码两个坐标轴，最后沿通道拼回完整 embedding。
   - English: `embed_dim` is split in half, one half per axis, then concatenated back into a full embedding.
3. **第 310-320 行 / Lines 310-320 (frequency bank)**:
   - 中文: `omega` 是一排频率，位置和频率做外积后再走 sin/cos。
   - English: `omega` is a bank of frequencies; positions take an outer product with it before sin/cos.

## 类比 / The analogy

这像给棋盘上的每个格子贴两张标签：一张写第几列，一张写第几行。标签不是可训练贴纸，而是一套固定刻度尺。

It is like labeling every square on a board twice: one label for the column and one for the row. The labels are fixed rulers, not learned stickers.

## 自己跑一遍 / Try it yourself

```python
import math

def one_axis(pos, dim):
    omega = [1 / (10000 ** (i / (dim / 2))) for i in range(dim // 2)]
    out = [pos * w for w in omega]
    return [round(math.sin(x), 3) for x in out] + [round(math.cos(x), 3) for x in out]

def pos2d(row, col, dim=8):
    return one_axis(col, dim // 2) + one_axis(row, dim // 2)

print(pos2d(row=1, col=2))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[0.909, 0.02, -0.416, 1.0, 0.841, 0.01, 0.54, 1.0]
```

中文: 前四个数来自列坐标，后四个数来自行坐标。  
English: The first four values come from the column, and the last four come from the row.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **MAE / ViT 固定位置表** / **MAE / ViT fixed position tables**: 同样用 sin/cos 表示 patch 网格。 / The same sin/cos table pattern encodes patch grids.
- **RoPE** / **RoPE**: 也用频率尺度表达位置，但在 attention 的 Q/K 上旋转。 / It also uses frequency scales for position, but rotates attention Q/K.

## 注意事项 / Caveats / when it breaks

- **固定网格大小** / **Fixed grid size**: 这里默认方形 `grid_size x grid_size`；非方形或多尺度输入要改接口。 / This assumes a square grid; rectangular or multi-scale inputs need a different interface.
- **不是可学习坐标** / **Not learned coordinates**: 固定表稳定，但不能自己适配数据集偏置。 / The fixed table is stable, but cannot adapt to dataset-specific biases.

## 延伸阅读 / Further reading

- DiT `models.py` positional embedding source: https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L274-L321
- MAE positional embedding utility: https://github.com/facebookresearch/mae
