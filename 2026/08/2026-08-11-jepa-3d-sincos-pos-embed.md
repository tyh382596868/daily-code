---
date: 2026-08-11
topic: diffusion
source: tracked
repo: facebookresearch/jepa
file: src/models/utils/pos_embs.py
permalink: https://github.com/facebookresearch/jepa/blob/51c59d518fc63c08464af6de585f78ac0c7ed4d5/src/models/utils/pos_embs.py#L11-L44
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, diffusion, video-position-embedding]
---

# V-JEPA 3D sin/cos 位置编码：时间轴分到更多维度 / V-JEPA 3D Sin/Cos Position Embedding: Give Time More Room

> **一句话 / In one line**: `get_3d_sincos_pos_embed()` 把视频 patch 的时间、高度、宽度分别编码，再拼成一个固定位置表。 / `get_3d_sincos_pos_embed()` encodes video-patch depth, height, and width separately, then concatenates them into one fixed table.

## 为什么重要 / Why this matters

图像 ViT 只需要二维坐标，视频世界模型还要知道“第几帧”。V-JEPA 这里没有让一个大 embedding 自己学三维结构，而是显式把 `d/h/w` 三个轴拆开；默认还把一半维度留给时间轴，因为视频预测里时间顺序通常比单个空间轴更脆弱。

Image ViTs need 2D coordinates; video world models also need frame order. V-JEPA does not ask one learned table to discover 3D structure from scratch. It builds separate `d/h/w` sin/cos codes and, by default, gives half of the embedding budget to time.

## 代码 / The code

`facebookresearch/jepa` — [`src/models/utils/pos_embs.py`](https://github.com/facebookresearch/jepa/blob/51c59d518fc63c08464af6de585f78ac0c7ed4d5/src/models/utils/pos_embs.py#L11-L44)

```python
def get_3d_sincos_pos_embed(
    embed_dim,
    grid_size,
    grid_depth,
    cls_token=False,
    uniform_power=False
):
    """
    grid_size: int of the grid height and width
    grid_depth: int of the grid depth
    returns:
        pos_embed: [grid_depth*grid_size*grid_size, embed_dim] (w/o cls_token)
                or [1+grid_depth*grid_size*grid_size, embed_dim] (w/ cls_token)
    """
    grid_d = np.arange(grid_depth, dtype=float)
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid_h, grid_d, grid_w = np.meshgrid(grid_h, grid_d, grid_w)  # order of meshgrid is very important for indexing as [d,h,w]

    if not uniform_power:
        h_embed_dim = embed_dim // 4
        w_embed_dim = embed_dim // 4
        d_embed_dim = embed_dim // 2
    else:
        h_embed_dim = w_embed_dim = d_embed_dim = int(np.ceil(embed_dim/6)*2)

    emb_h = get_1d_sincos_pos_embed_from_grid(h_embed_dim, grid_h)  # (T*H*W, D1)
    emb_w = get_1d_sincos_pos_embed_from_grid(w_embed_dim, grid_w)  # (T*H*W, D2)
    emb_d = get_1d_sincos_pos_embed_from_grid(d_embed_dim, grid_d)  # (T*H*W, D3)
    pos_embed = np.concatenate([emb_d, emb_h, emb_w], axis=1)
    pos_embed = pos_embed[:, :embed_dim]
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed
```

## 逐行讲解 / What's happening

1. **第 25-28 行 / Lines 25-28 (3D grid)**:
   - 中文: 先生成 depth、height、width 三个坐标表，`meshgrid` 顺序被固定成 `[d,h,w]` 的 flatten 语义。
   - English: The code builds depth, height, and width coordinate grids, with `meshgrid` ordered to match `[d,h,w]` flattening.
2. **第 30-35 行 / Lines 30-35 (dimension budget)**:
   - 中文: 默认时间轴拿 `embed_dim // 2`，两个空间轴各拿四分之一；`uniform_power` 则让三轴用同样的频率容量。
   - English: By default, time gets half the width and each spatial axis gets a quarter; `uniform_power` gives the axes equal frequency capacity.
3. **第 37-40 行 / Lines 37-40 (axis-wise encoding)**:
   - 中文: 每个轴都调用同一个 1D sin/cos 编码函数，最后按 `[time, height, width]` 拼接。
   - English: Each axis uses the same 1D sin/cos encoder, then the results are concatenated as `[time, height, width]`.
4. **第 41-44 行 / Lines 41-44 (trim and cls token)**:
   - 中文: 如果均分维度时多算了一点，`[:, :embed_dim]` 裁回目标宽度；需要分类 token 时，在最前面补一行零。
   - English: If equal allocation overshoots, `[:, :embed_dim]` trims to the target width; a class token adds one zero row at the front.

## 类比 / The analogy

像给视频胶片贴三张尺：一张量第几帧，一张量行号，一张量列号。模型看到的是三张尺拼起来的坐标标签。

Think of labeling a film strip with three rulers: frame number, row number, and column number. The model receives the three labels concatenated into one coordinate tag.

## 自己跑一遍 / Try it yourself

```python
import math

def dims(embed_dim, uniform=False):
    if uniform:
        d = h = w = math.ceil(embed_dim / 6) * 2
    else:
        h, w, d = embed_dim // 4, embed_dim // 4, embed_dim // 2
    return d, h, w, sum((d, h, w))

print(dims(64))
print(dims(65, uniform=True))
print("tokens", 3 * 2 * 2)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
(32, 16, 16, 64)
(22, 22, 22, 66)
tokens 12
```

这个例子显示默认模式偏向时间轴，而 `uniform_power=True` 可能先生成更宽的表，再裁回 `embed_dim`。

The example shows the default time-heavy allocation. With `uniform_power=True`, the table can be slightly wider first and then trimmed back to `embed_dim`.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **DiT 2D sin/cos** / **DiT 2D sin/cos**: 图像 diffusion 模型也把 `h/w` 拆开编码，只是没有时间轴。 / Image diffusion models also split `h/w`, but without the time axis.
- **Wan2.1 3D RoPE** / **Wan2.1 3D RoPE**: 旋转位置编码也会显式管理 `t/h/w` 三轴，只是作用在 attention 的 q/k 上。 / Rotary video embeddings also manage `t/h/w`, but apply them to attention q/k.

## 注意事项 / Caveats / when it breaks

- **flatten 顺序不能乱** / **Flatten order must match**: patchify 如果不是 `[d,h,w]` 顺序，位置表会贴错 token。 / If patchify uses a different `[d,h,w]` order, positions attach to the wrong tokens.
- **固定表不适应任意分辨率** / **Fixed tables are not arbitrary-resolution magic**: 改 `grid_depth` 或 `grid_size` 时要重建位置表。 / Changing `grid_depth` or `grid_size` requires rebuilding the table.

## 延伸阅读 / Further reading

- [V-JEPA 3D position embedding](https://github.com/facebookresearch/jepa/blob/51c59d518fc63c08464af6de585f78ac0c7ed4d5/src/models/utils/pos_embs.py#L11-L44)

