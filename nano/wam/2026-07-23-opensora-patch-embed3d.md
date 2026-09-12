---
date: 2026-07-23
topic: wam
source: wam
repo: PKU-YuanGroup/Open-Sora
file: opensora/models/layers/blocks.py
permalink: https://github.com/PKU-YuanGroup/Open-Sora/blob/main/opensora/models/layers/blocks.py
difficulty: intermediate
read_time: ~8 min
tags: [code-of-the-day, wam, open-sora, patchify-positional, video-tokens]
---

# Open-Sora PatchEmbed3D：一次卷积把视频切成时空 token / Open-Sora PatchEmbed3D: One Convolution Turns Video into Spacetime Tokens

> **一句话 / In one line**: 3D patch embedding 用 `(T, H, W)` 三个步长同时切视频，把一段 latent video 变成 DiT 可以处理的 token 序列。 / 3D patch embedding slices video with `(T, H, W)` strides at once, turning a latent video clip into a token sequence for a DiT.

## 为什么重要 / Why this matters

图像 DiT 只需要二维 patch。视频 / world-action model 还多一个时间轴：你可以逐帧 patchify，也可以用 3D 卷积直接生成 tubelet token。后者让每个 token 一开始就带有局部时间上下文。

Image DiTs only need 2D patches. Video and world-action models add a time axis: you can patchify frame by frame, or use a 3D convolution to produce tubelet tokens directly. The latter gives each token local temporal context from the start.

## 代码 / The code

`PKU-YuanGroup/Open-Sora` — [`opensora/models/layers/blocks.py`](https://github.com/PKU-YuanGroup/Open-Sora/blob/main/opensora/models/layers/blocks.py)

```python
# Simplified teaching slice.
class PatchEmbed3D:
    def __init__(self, patch_size=(1, 2, 2), in_chans=4, embed_dim=1152):
        self.patch_size = patch_size
        self.proj = Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, video):
        # video: [B, C, T, H, W]
        tokens = self.proj(video)          # [B, D, T', H', W']
        tokens = flatten_spacetime(tokens) # [B, T' * H' * W', D]
        return tokens
```

## 逐行讲解 / What's happening

1. **kernel 等于 patch / Kernel equals patch**: 中文: 卷积核覆盖一个 tubelet。 English: the convolution kernel covers one video tubelet.
2. **stride 也等于 patch / Stride also equals patch**: 中文: 相邻 token 不重叠，输出网格就是 token 网格。 English: non-overlapping strides make the output grid the token grid.
3. **通道变成 embed_dim / Channels become embed_dim**: 中文: 每个 tubelet 被线性投到 transformer hidden size。 English: each tubelet is linearly projected into transformer width.
4. **展平成序列 / Flatten into a sequence**: 中文: DiT 后续只关心 token 序列，时间和空间位置交给位置编码保留。 English: the DiT consumes a token sequence; positional encoding preserves spacetime coordinates.

## 在 nanoWAM 中的位置 / Where this fits in nanoWAM

这是 `patchify-positional` 的 advanced variant。上游是 VAE latent `[B, C, T, H, W]`，下游是 DiT / MMDiT block。生产 WAM 还需要保存 `(T', H', W')` 网格，供 3D RoPE、attention mask、unpatchify 输出头使用。

This is an advanced variant of `patchify-positional`. Upstream is a VAE latent `[B, C, T, H, W]`; downstream is the DiT or MMDiT block. A production WAM also keeps the `(T', H', W')` grid for 3D RoPE, attention masks, and the unpatchify output head.

## 自己跑一遍 / Try it yourself

```python
def patch_grid(shape, patch):
    _, _, t, h, w = shape
    pt, ph, pw = patch
    return (t // pt, h // ph, w // pw)

grid = patch_grid((2, 4, 16, 32, 32), (2, 4, 4))
print(grid, "tokens =", grid[0] * grid[1] * grid[2])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
(8, 8, 8) tokens = 512
```

## 注意事项 / Caveats / when it breaks

- **尺寸必须可整除 / Sizes must divide cleanly**: 不能整除时要先 pad 或裁剪。 / if dimensions do not divide, pad or crop first.
- **时间 patch 是延迟预算 / Temporal patch size is latency budget**: `pt` 越大，token 越少，但每个 token 跨越更长时间。 / larger `pt` means fewer tokens but coarser temporal granularity.
- **展平顺序要固定 / Flatten order must be fixed**: RoPE 和 unpatchify 必须使用同一套 `(t, h, w)` 顺序。 / RoPE and unpatchify must agree on the same `(t, h, w)` order.

## 延伸阅读 / Further reading

- [Open-Sora blocks.py](https://github.com/PKU-YuanGroup/Open-Sora/blob/main/opensora/models/layers/blocks.py)

