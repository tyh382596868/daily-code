---
date: 2026-09-12
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/modules/vae.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vae.py#L57-L160
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, vae-encoder-decoder, temporal-compression, streaming-cache, bfloat16]
build_role: vae-encoder-decoder advanced variant
---

# Wan2.1 Resample：空间缩放和时间 cache 分开处理 / Wan2.1 Resample: Separate Spatial Resizing from Temporal Cache State

> **一句话 / In one line**: Wan2.1 把视频的空间 resample 展平到 2D Conv2d，同时用 causal 3D conv 和 `feat_cache` 保住跨 chunk 的时间连续性。 / Wan2.1 flattens spatial resampling into 2D Conv2d while causal 3D convolution plus `feat_cache` preserves temporal continuity across chunks.

## 为什么重要 / Why this matters

中文：视频 VAE 同时要处理空间尺寸和时间尺寸。直接对 `[B, C, T, H, W]` 做一个巨大的 3D resample，显存和 kernel 复杂度都会上升。Wan2.1 的 `Resample` 先把每一帧视作一张 2D 图，复用成熟的 2D upsample/downsample；只有时间方向需要 causal conv 和 cache。这个分工让 chunked video encode/decode 更容易落地。

English: A video VAE must change both spatial and temporal resolution. Applying one giant 3D resampler to `[B, C, T, H, W]` is expensive and awkward. Wan2.1 treats each frame as a 2D image for spatial resize, while temporal causal convolution and cache handle only the time axis. That split makes chunked video encode/decode practical.

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/modules/vae.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vae.py#L57-L160)

```python
class Upsample(nn.Upsample):

    def forward(self, x):
        """
        Fix bfloat16 support for nearest neighbor interpolation.
        """
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):

    def __init__(self, dim, mode):
        assert mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d',
                        'downsample3d')
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
            self.time_conv = CausalConv3d(
                dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode == 'downsample2d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == 'downsample3d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == 'upsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] != 'Rep':
                        cache_x = torch.cat([
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                cache_x.device), cache_x
                        ], dim=2)
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] == 'Rep':
                        cache_x = torch.cat([
                            torch.zeros_like(cache_x).to(cache_x.device),
                            cache_x
                        ], dim=2)
                    if feat_cache[idx] == 'Rep':
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        if self.mode == 'downsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x
```

## 逐行讲解 / What's happening

1. **第 57-63 行 / Lines 57-63 (`Upsample`)**:
   - 中文：插值前转成 float，结束后 cast 回原 dtype，专门绕开 nearest interpolation 对 bfloat16 的限制。
   - English: Interpolation runs in float and then casts back, working around nearest-neighbor limitations for bfloat16.
2. **第 75-99 行 / Lines 75-99 (mode construction)**:
   - 中文：2D 模式只创建空间层；3D 模式额外创建 `time_conv`，因此时间计算不会污染纯图片路径。
   - English: 2D modes build only spatial layers; 3D modes add `time_conv`, keeping temporal work out of the image-only path.
3. **第 101-132 行 / Lines 101-132 (upsample cache)**:
   - 中文：第一块用 `'Rep'` 作为“还没有真实历史”的哨兵；后续 chunk 从上一块取最后几帧作为 causal context。
   - English: The first chunk uses `'Rep'` as a sentinel for “no real history yet.” Later chunks reuse the last frames from the previous chunk as causal context.
4. **第 134-141 行 / Lines 134-141 (spatial path)**:
   - 中文：时间维暂时折叠到 batch，空间 resample 用标准 2D 模块完成，再折回视频布局。
   - English: Time is temporarily folded into the batch dimension, spatial resampling uses standard 2D modules, and the video layout is restored.
5. **第 143-160 行 / Lines 143-160 (downsample cache)**:
   - 中文：下采样先保存当前块的最后一帧，下一块把它拼到输入前面，保证 causal kernel 看到连续时间。
   - English: Downsampling stores the current chunk’s last frame, then prepends it to the next chunk so the causal kernel sees continuous time.

## 类比 / The analogy

中文：像把一卷电影分成几段剪辑。每段内部可以把每一帧当照片处理，但剪辑点必须把上一段最后几帧递给下一段，否则镜头会在接缝处突然跳动。

English: It is like editing a film in chunks. Each frame can use a photo-editing tool internally, but the last few frames of one clip must be handed to the next clip or the cut will jump.

## 在 nanoWAM 中的位置 / Where this lives in your nanoWAM

中文：这是 `vae-encoder-decoder` 的 advanced variant，位于视频帧输入和 DiT latent 之间。输入是 `[B, C, T, H, W]`，输出是空间或时间缩放后的 latent feature；上游是视频预处理，下游是 patchify 和 DiT。省掉 temporal cache，长视频分块时会在 chunk 边界产生时序断裂。生产版还要处理 cache 生命周期、不同 VAE 层的 cache index、显存回收、编码和解码对称性，以及不同 chunk 长度下的边界测试。

English: This is an advanced `vae-encoder-decoder` component between video frames and DiT latents. It consumes `[B, C, T, H, W]` and returns spatially or temporally resampled features; patchify and the DiT consume the result downstream. Without temporal cache, chunked long-video processing produces discontinuities at boundaries. Production code also needs cache lifecycle management, per-layer cache indices, memory release, encode/decode symmetry, and boundary tests for uneven chunk lengths.

## 自己跑一遍 / Try it yourself

```python
def stream_resample(chunks):
    previous = None
    output = []
    for chunk in chunks:
        prefix = [previous] if previous is not None else []
        merged = prefix + chunk
        output.extend([frame * 2 for frame in merged])
        previous = chunk[-1]
    return output

print(stream_resample([[1, 2], [3, 4]]))
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
[2, 4, 4, 6, 8]
```

中文：第二块先看第一块的最后一帧 `2`，再处理自己的 `3, 4`；真实实现把“看一帧”换成 causal convolution 的上下文。

English: The second chunk sees the first chunk’s final frame `2` before processing `3, 4`; the real implementation replaces this toy context with causal-convolution state.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Open-Sora temporal-tiled VAE** / **Open-Sora temporal-tiled VAE**: 用 tile overlap 处理任意长视频。 / Use temporal tiles and overlap to handle arbitrarily long videos.
- **Wan2.1 CausalConv3d** / **Wan2.1 CausalConv3d**: 只给过去方向补 padding。 / Pad only toward the past.
- **streaming video encoders** / **streaming video encoders**: 在采集或推理期间携带短期 state。 / Carry short-term state during collection or inference.

## 注意事项 / Caveats / when it breaks

- **cache 的 index 必须和模块顺序稳定** / **Cache indices must match module order**: 递归网络重排层后，旧 cache 不能复用。
- **第一块和后续块不是同一条路径** / **The first and later chunks differ**: sentinel 初始化要有单独测试。
- **空间和时间的 layout 要反复检查** / **Check spatial and temporal layouts repeatedly**: `(b*t, c, h, w)` 与 `(b, c, t, h, w)` 混用时最容易出错。

## 延伸阅读 / Further reading

- [Wan2.1 VAE resampling](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vae.py)
- [Open-Sora causal VAE](https://github.com/hpcaitech/Open-Sora)
