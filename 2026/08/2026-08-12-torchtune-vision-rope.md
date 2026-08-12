---
date: 2026-08-12
topic: infrastructure
source: tracked
repo: pytorch/torchtune
file: torchtune/modules/position_embeddings.py
permalink: https://github.com/pytorch/torchtune/blob/main/torchtune/modules/position_embeddings.py#L112-L233
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, infrastructure, vision-rope]
---

# torchtune Vision RoPE：把图像 patch 拆成 X/Y 两把尺 / torchtune Vision RoPE: Split Image Patches into X/Y Rulers

> **一句话 / In one line**: `VisionRotaryPositionalEmbeddings` 把每个 patch 的二维坐标变成 RoPE 旋转缓存，并让 CLS token 的频率归零。 / `VisionRotaryPositionalEmbeddings` turns each patch's 2D coordinate into a RoPE cache while zeroing the CLS token frequency.

## 为什么重要 / Why this matters

文本 RoPE 只有一条序列轴；视觉 transformer 的 patch 来自二维网格。如果直接把图像拍平成一维，模型仍能学，但位置结构被折成一条线。torchtune 的做法是把 head 维度的一半给 x 坐标、一半给 y 坐标，再用同一套复数旋转公式作用到 attention head 上。

Text RoPE has one sequence axis; image patches come from a 2D grid. Flattening the grid into one line works, but it hides the geometry. torchtune splits the rotary frequency budget between x and y coordinates, then applies the same complex rotation pattern to attention heads.

## 代码 / The code

`pytorch/torchtune` — [`torchtune/modules/position_embeddings.py`](https://github.com/pytorch/torchtune/blob/main/torchtune/modules/position_embeddings.py#L112-L233)

```python
class VisionRotaryPositionalEmbeddings(nn.Module):
    def __init__(
        self,
        patch_size: int,
        tile_size: int,
        dim: int,
        base: int = 10_000,
        append_cls_token: bool = True,
    ) -> None:
        super().__init__()
        self.patch_grid_size = tile_size // patch_size
        self.seq_len = self.patch_grid_size**2 + 1
        self.dim = dim
        self.base = base
        self.append_cls_token = append_cls_token
        self.rope_init()

    def rope_init(self):
        dim = self.dim // 2
        theta = 1.0 / (
            self.base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache()

    def build_rope_cache(self) -> None:
        patches_per_tile = self.patch_grid_size**2
        patch_idx = torch.arange(
            patches_per_tile, dtype=self.theta.dtype, device=self.theta.device
        )
        if self.append_cls_token:
            patch_idx = torch.cat(
                [patch_idx, -1 * torch.ones(1, dtype=patch_idx.dtype, device=patch_idx.device)]
            )
        else:
            patch_idx = torch.cat(
                [-1 * torch.ones(1, dtype=patch_idx.dtype, device=patch_idx.device), patch_idx]
            )

        patch_x_pos = patch_idx % self.patch_grid_size
        patch_y_pos = patch_idx // self.patch_grid_size
        x_theta = torch.einsum("i, j -> ij", patch_x_pos + 1, self.theta).float()
        y_theta = torch.einsum("i, j -> ij", patch_y_pos + 1, self.theta).float()

        freqs = torch.cat([x_theta, y_theta], dim=-1)
        freqs = freqs.masked_fill(patch_idx.unsqueeze(-1) < 0, 0)
        cache = torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        bsz, _, n_h, h_d = x.shape
        xshaped = x.float().reshape(bsz, -1, self.seq_len, n_h, h_d // 2, 2)
        rope_cache = self.cache.view(1, 1, self.seq_len, 1, h_d // 2, 2)
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )
        x_out = x_out.reshape(bsz, -1, n_h, h_d)
        return x_out.type_as(x)
```

## 逐行讲解 / What's happening

1. **第 143-148 行 / Lines 143-148 (grid contract)**:
   - 中文: `tile_size // patch_size` 得到每条边有多少 patch，`seq_len` 额外加 1 给 CLS token。
   - English: `tile_size // patch_size` gives patches per side, and `seq_len` adds one slot for the CLS token.
2. **第 149-155 行 / Lines 149-155 (frequency ladder)**:
   - 中文: `theta` 是几何级数频率表，注册成非持久 buffer，进入设备迁移但不写进 checkpoint。
   - English: `theta` is the geometric frequency table, registered as a non-persistent buffer so it moves with the module but stays out of checkpoints.
3. **第 156-192 行 / Lines 156-192 (2D cache)**:
   - 中文: patch index 先拆成 `x` 和 `y`，再各自和 `theta` 做外积；CLS token 用 `-1` 标出来，最后整行频率清零。
   - English: Patch indices are split into `x` and `y`, each takes an outer product with `theta`; the CLS token is marked by `-1` and its frequency row is zeroed.
4. **第 212-233 行 / Lines 212-233 (complex rotation)**:
   - 中文: 最后一维两两成对，按 `(a, b) -> (a cos - b sin, b cos + a sin)` 旋转，然后还原原形状和 dtype。
   - English: The last dimension is paired and rotated as `(a, b) -> (a cos - b sin, b cos + a sin)`, then reshaped and cast back.

## 类比 / The analogy

像给瓷砖贴坐标贴纸：每块砖既有横向编号也有纵向编号。CLS token 是前台接待员，它不属于任何瓷砖，所以贴一张空白标签。

It is like labeling floor tiles with both row and column stickers. The CLS token is the receptionist, not a tile, so it gets a blank label.

## 自己跑一遍 / Try it yourself

```python
import math

grid = 2
theta = [1.0, 0.1]
patch_idx = [0, 1, 2, 3, -1]
for idx in patch_idx:
    x = 0 if idx < 0 else idx % grid
    y = -1 if idx < 0 else idx // grid
    freqs = [0.0, 0.0] if idx < 0 else [(x + 1) * theta[0], (y + 1) * theta[1]]
    cache = [(round(math.cos(f), 3), round(math.sin(f), 3)) for f in freqs]
    print(idx, cache)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
0 [(0.54, 0.841), (0.995, 0.1)]
1 [(-0.416, 0.909), (0.995, 0.1)]
2 [(0.54, 0.841), (0.98, 0.199)]
3 [(-0.416, 0.909), (0.98, 0.199)]
-1 [(1.0, 0.0), (1.0, 0.0)]
```

最值得看的是 `-1`：频率清零后，cos/sin 变成恒等旋转，CLS token 不被强行塞进图像坐标。

The important part is `-1`: zero frequencies become an identity rotation, so the CLS token is not forced into image coordinates.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Wan2.1 3D RoPE** / **Wan2.1 3D RoPE**: 视频 token 把频率拆到时间、高、宽三条轴。 / Video tokens split frequencies across time, height, and width.
- **DiT 2D sin/cos embedding** / **DiT 2D sin/cos embedding**: 也是把二维网格拆轴，只是用加法位置编码而不是旋转。 / It also splits a 2D grid by axis, but uses additive positional embeddings instead of rotation.

## 注意事项 / Caveats / when it breaks

- **维度要偶数再偶数** / **Dimensions need pairing**: RoPE 两两成对，2D 版本还要给 x/y 分账，head dim 设计不能随便取。
- **tile 假设要一致** / **Tile assumptions must match**: `seq_len` 假设每个 tile 的 patch 数固定；动态裁剪必须在外层保证长度一致。

## 延伸阅读 / Further reading

- [torchtune `VisionRotaryPositionalEmbeddings`](https://github.com/pytorch/torchtune/blob/main/torchtune/modules/position_embeddings.py#L112-L233)
