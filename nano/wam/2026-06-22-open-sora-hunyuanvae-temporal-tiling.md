---
date: 2026-06-22
topic: wam
source: wam
repo: hpcaitech/Open-Sora
file: opensora/models/hunyuan_vae/autoencoder_kl_causal_3d.py
permalink: https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/hunyuan_vae/autoencoder_kl_causal_3d.py#L376-L554
difficulty: intermediate
read_time: ~12 min
tags: [code-of-the-day, wam, vae, video-generation, temporal-tiling, memory-optimization]
build_role: vae-encoder-decoder — temporal tiling extension that makes the VAE handle arbitrarily long videos in O(constant) memory
---

# 时序 Tile VAE：无限长视频的 O(1) 显存编码 / Temporal-Tiled VAE: O(1) Memory Encoding for Arbitrarily Long Videos

> **一句话 / In one line**: 沿时间轴把视频切成重叠的小窗口，每个窗口单独编码，相邻窗口交界处用线性插值渐变融合——从而把任意长度视频的显存峰值压缩到单窗口大小。 / Slice a video along the time axis into overlapping windows, encode each independently, then linearly blend the boundaries — compressing peak VAE memory from O(video length) to O(one tile).

## 为什么重要 / Why this matters

视频 VAE（Wan2.1、HunyuanVideo、Open-Sora）用因果 3D 卷积把视频压缩到 latent 空间，再由 DiT/U-Net 处理 latent。问题在于：一次 forward 的显存需求随视频时间长度线性增长。编码 33 帧没问题，编码 200 帧就 OOM。

HunyuanVideo 的 VAE（被 Open-Sora 引入）用"时序 tile"解决了这个问题：`temporal_tiled_encode` 把 T 帧视频沿时间轴切成 `tile_sample_min_tsize` 帧的小块（有重叠），每块单独编码；相邻块交界处用 `blend_t` 做线性插值渐变，消除接缝；最后把各块的 latent 拼起来。解码时 `temporal_tiled_decode` 做镜像操作。整个过程显存峰值只取决于单块大小，与视频总帧数无关。

Video VAEs (Wan2.1, HunyuanVideo, Open-Sora) use causal 3D convolutions to compress video into latent space for a DiT/U-Net. The problem: one forward pass's memory grows linearly with video length. Encoding 33 frames is fine; 200 frames OOM.

HunyuanVideo's VAE (imported into Open-Sora) solves this with temporal tiling: `temporal_tiled_encode` splits a T-frame video into `tile_sample_min_tsize`-frame windows with overlap, encodes each independently, blends adjacent window boundaries with linear interpolation in `blend_t`, then concatenates the latent outputs. `temporal_tiled_decode` mirrors the process. Peak memory depends only on window size — independent of total video length.

## 代码 / The code

`hpcaitech/Open-Sora` — [`opensora/models/hunyuan_vae/autoencoder_kl_causal_3d.py`](https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/hunyuan_vae/autoencoder_kl_causal_3d.py#L376-L554)

```python
def blend_t(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
    for x in range(blend_extent):
        b[:, :, x, :, :] = a[:, :, -blend_extent + x, :, :] * (1 - x / blend_extent) + b[:, :, x, :, :] * (
            x / blend_extent
        )
    return b

def temporal_tiled_encode(self, x: torch.FloatTensor) -> DiagonalGaussianDistribution:
    B, C, T, H, W = x.shape
    overlap_size = int(self.tile_sample_min_tsize * (1 - self.tile_overlap_factor))
    blend_extent = int(self.tile_latent_min_tsize * self.tile_overlap_factor)
    t_limit = self.tile_latent_min_tsize - blend_extent

    row = []
    for i in range(0, T, overlap_size):
        tile = x[:, :, i : i + self.tile_sample_min_tsize + 1, :, :]
        if self.use_spatial_tiling and (
            tile.shape[-1] > self.tile_sample_min_size or tile.shape[-2] > self.tile_sample_min_size
        ):
            tile = self.spatial_tiled_encode(tile, return_moments=True)
        else:
            tile = self.encoder(tile)
            tile = self.quant_conv(tile)
        if i > 0:
            tile = tile[:, :, 1:, :, :]  # strip first frame to remove boundary overlap
        row.append(tile)
    result_row = []
    for i, tile in enumerate(row):
        if i > 0:
            tile = self.blend_t(row[i - 1], tile, blend_extent)
            result_row.append(tile[:, :, :t_limit, :, :])
        else:
            result_row.append(tile[:, :, : t_limit + 1, :, :])
    moments = torch.cat(result_row, dim=2)
    posterior = DiagonalGaussianDistribution(moments)
    return posterior

def temporal_tiled_decode(self, z: torch.FloatTensor, return_dict: bool = True):
    B, C, T, H, W = z.shape
    overlap_size = int(self.tile_latent_min_tsize * (1 - self.tile_overlap_factor))
    blend_extent = int(self.tile_sample_min_tsize * self.tile_overlap_factor)
    t_limit = self.tile_sample_min_tsize - blend_extent

    row = []
    for i in range(0, T, overlap_size):
        tile = z[:, :, i : i + self.tile_latent_min_tsize + 1, :, :]
        if self.use_spatial_tiling and (
            tile.shape[-1] > self.tile_latent_min_size or tile.shape[-2] > self.tile_latent_min_size
        ):
            decoded = self.spatial_tiled_decode(tile, return_dict=True).sample
        else:
            tile = self.post_quant_conv(tile)
            decoded = self.decoder(tile)
        if i > 0:
            decoded = decoded[:, :, 1:, :, :]
        row.append(decoded)
    result_row = []
    for i, tile in enumerate(row):
        if i > 0:
            tile = self.blend_t(row[i - 1], tile, blend_extent)
            result_row.append(tile[:, :, :t_limit, :, :])
        else:
            result_row.append(tile[:, :, : t_limit + 1, :, :])

    dec = torch.cat(result_row, dim=2)
    if not return_dict:
        return (dec,)
    return DecoderOutput(sample=dec)
```

## 逐行讲解 / What's happening

1. **`blend_t(a, b, blend_extent)`**:
   - 中文: 这是整个 tile 方案里最关键的一个函数，只有 5 行。`a` 是前一个 tile 的末尾帧段，`b` 是当前 tile 的开头帧段，`blend_extent` 是重叠帧数。循环对重叠区域内的每一帧做线性插值：`b[:,:,x] = a[:,:,-blend_extent+x] * (1 - x/blend_extent) + b[:,:,x] * (x/blend_extent)`。`x=0` 时完全是前一 tile，`x=blend_extent-1` 时几乎全是当前 tile——渐变消除了接缝。
   - English: The linchpin of the whole tiling scheme — just 5 lines. `a` is the tail of the previous tile, `b` is the head of the current tile, `blend_extent` is the overlap frame count. For each overlapping frame index `x`, it linear-interpolates: `b[:,:,x] = a[:,:,-blend_extent+x] * (1-x/blend_extent) + b[:,:,x] * (x/blend_extent)`. At `x=0` the result is entirely the previous tile; at `x=blend_extent-1` it's almost entirely the current tile — the gradient removes the seam.

2. **`overlap_size` vs `blend_extent` vs `t_limit`**:
   - 中文: 三个关键常数的关系：`overlap_size = tile_sample_min_tsize * (1 - tile_overlap_factor)` 是相邻 tile 起始帧的间距；`blend_extent = tile_latent_min_tsize * tile_overlap_factor` 是 latent 空间里的融合宽度（比像素空间小，因为有时间压缩比）；`t_limit = tile_latent_min_tsize - blend_extent` 是每个 tile 实际贡献到输出的帧数（去掉被下一 tile 覆盖的融合区）。三者加起来确保相邻 tile 拼接后总帧数正确。
   - English: The three key constants: `overlap_size` is the stride between tile start frames in pixel space; `blend_extent` is the blend width in latent space (smaller than pixel space by the temporal compression ratio); `t_limit` is how many latent frames each tile actually contributes to the output (all but the blend region that gets overwritten by the next tile). Together they ensure the concatenated output has the correct total frame count.

3. **`tile = x[:, :, i : i + tile_sample_min_tsize + 1, :, :]`（+1 是因果上下文）**:
   - 中文: 每个 tile 多取一帧（+1），是为了给因果卷积提供前一帧的上下文。因果 3D 卷积不能用未来帧，但需要"上一帧"来产生正确的第一帧输出。多取的这一帧在编码后被 `tile[:, :, 1:, :, :]` 剥掉（只对 `i > 0` 的非首块），避免它进入输出。
   - English: Each tile takes one extra frame (+1) to give the causal convolution its prior-frame context. Causal 3D conv can't use future frames but needs "the previous frame" to produce a correct first-frame output. This extra frame is stripped off after encoding with `tile[:, :, 1:, :, :]` (for non-first tiles where `i > 0`), preventing it from entering the output.

4. **两遍循环（先编码 + 后融合）**:
   - 中文: 代码分两步走：第一个 `for` 循环编码所有 tile（每次只需一个 tile 的显存），结果存入 `row`；第二个 `for` 循环对相邻 tile 做 `blend_t` 融合，再截取 `t_limit` 帧存入 `result_row`。首块和后续块处理逻辑不同（首块保留 `t_limit+1` 帧因为没有前驱要覆盖）。
   - English: The code runs two passes: the first `for` loop encodes all tiles (holding only one tile in GPU memory at a time) into `row`; the second `for` loop blends adjacent tiles with `blend_t` and trims each to `t_limit` frames for `result_row`. The first tile is special — it contributes `t_limit+1` frames since there is no predecessor to blend into it.

5. **`temporal_tiled_decode` 镜像结构**:
   - 中文: 解码和编码完全对称：也有同样的 `overlap_size`（但在 latent 空间计算）、`blend_extent`（在像素空间计算）和 `t_limit`，同样的两遍循环，同样的 `decoded[:, :, 1:, :, :]` strip。唯一区别是 latent T 和 pixel T 之间存在时间压缩比，所以各常数的绝对值不同。
   - English: Decode is symmetric to encode: the same two-pass structure, the same strip of the first frame, the same `blend_t`. The only difference is that constants are computed in latent space (for decode strides) rather than pixel space, reflecting the temporal compression ratio between latent T and pixel T.

## 类比 / The analogy

想象你要给一部 3 小时电影配字幕，但翻译团队的显示器只能同时显示 10 分钟的视频。解决办法：把电影切成 12 分钟的段（有 2 分钟重叠），逐段翻译；在每段交界的 2 分钟里，把两段的字幕做"淡入淡出"——开头 2 分钟从上一段渐渐淡出，下一段渐渐淡入。观众看到的是连贯字幕，翻译团队的显示器始终只加载 12 分钟。

Imagine subtitling a 3-hour film when your translation workstation can only display 10 minutes of video at once. Solution: cut the film into 12-minute segments with 2-minute overlaps; translate each segment independently; in the 2-minute overlap zone, crossfade the two segments' subtitles — the previous segment fades out, the next one fades in. The audience sees seamless subtitles; the workstation loads at most 12 minutes at a time.

## 在 nanoWAM 中的位置 / Where this lives in your nanoWAM

这是 nanoWAM 构建计划中 **`vae-encoder-decoder`** 组件的"长视频扩展"版本。前两次覆盖这个组件时，Wan2.1 教了因果 Conv3D 的基本编解码结构，Open-Sora 教了空间像素 shuffle 上采样；这次教的是**时序 tile 技术**——让同一个 VAE 架构能处理任意长度的视频。

在 nanoWAM 系统里，VAE 是最前端的组件：真实视频（高分辨率像素帧）→ **`temporal_tiled_encode`** → latent（低维时空表示）→ DiT/U-Net 世界模型推理 → latent → **`temporal_tiled_decode`** → 生成视频。上游没有依赖（VAE 直接读原始帧），下游是 DiT/U-Net 的 conditioning 和去噪主干。如果省掉时序 tile，你的 WAM 只能处理短于 `tile_sample_min_tsize` 帧的片段，无法做长时域的世界模型预训练或推理。生产级实现还需要补：tile 大小和重叠率的 grid-search 调参（影响质量/速度 tradeoff）、与空间 tile 的嵌套（已有 `use_spatial_tiling` 开关）、以及梯度检查点（训练时 `temporal_tiled_encode` 还需要 backward）。

This is the **"long-video extension"** of the `vae-encoder-decoder` component in the nanoWAM build plan. The first two times this component was covered: Wan2.1 taught the basic causal Conv3D encoder-decoder structure; Open-Sora's pixel-shuffle taught spatial upsampling. This note teaches **temporal tiling** — making the same VAE architecture handle videos of arbitrary length.

In nanoWAM, the VAE is the frontmost component: raw video (pixel frames) → **`temporal_tiled_encode`** → latent (low-dimensional spatiotemporal representation) → DiT/U-Net world model inference → latent → **`temporal_tiled_decode`** → generated video. No upstream dependencies — the VAE reads raw frames directly. Downstream is the DiT/U-Net's conditioning and denoising. Without temporal tiling, your WAM can only process clips shorter than `tile_sample_min_tsize` frames — impossible for long-horizon world-model pretraining. Production needs: grid-search for tile size and overlap rate (quality/speed tradeoff), nesting with spatial tiling (`use_spatial_tiling` switch already present), and gradient checkpointing for backward through `temporal_tiled_encode`.

## 自己跑一遍 / Try it yourself

```python
import torch

def blend_t(a, b, blend_extent):
    blend_extent = min(a.shape[2], b.shape[2], blend_extent)
    for x in range(blend_extent):
        b[:, :, x] = a[:, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, x] * (x / blend_extent)
    return b

# Simulate two encoded tiles, each (B=1, C=4, T=8, H=4, W=4)
tile_a = torch.ones(1, 4, 8, 4, 4)       # first tile (all ones)
tile_b = torch.zeros(1, 4, 8, 4, 4)      # second tile (all zeros)
blend_extent = 4

blended = blend_t(tile_a, tile_b, blend_extent)

# Inspect the temporal blend at spatial position [0,0,0,0]
print("blend region (should linearly fade 1.0→0.0):")
for t in range(blend_extent):
    val = blended[0, 0, t, 0, 0].item()
    print(f"  t={t}: {val:.3f}")
print("after blend (should be 0.0):")
print(f"  t={blend_extent}: {blended[0, 0, blend_extent, 0, 0].item():.3f}")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
blend region (should linearly fade 1.0→0.0):
  t=0: 1.000
  t=1: 0.750
  t=2: 0.500
  t=3: 0.250
after blend (should be 0.0):
  t=4: 0.000
```

注意 `x=0` 处完全是 tile_a 的值（1.0），`x=blend_extent-1` 处几乎全是 tile_b（0.25 = 1.0 × 0.25 + 0.0 × 0.75）——这正是线性渐变消除接缝的核心机制。

At `x=0` the result is entirely from `tile_a` (1.0); by `x=blend_extent-1` it's 75% `tile_b`. This gradual crossfade is what removes the seam between encoded tiles.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Stable Video Diffusion (`tiled_vae_encode`)**: SVD 的 VAE 也有 spatial tile，但没有时序 tile——因为 SVD 每次只处理固定 25 帧。HunyuanVideo 是把这个思路扩展到了时间维度。 / SVD's VAE has spatial tiling but not temporal — SVD always processes exactly 25 frames. HunyuanVideo extends the same idea to the time dimension.
- **Open-Sora 的 `spatial_tiled_encode` / `spatial_tiled_decode`**: 同一文件里的空间 tile 方法（处理超大分辨率帧），结构与 `temporal_tiled_encode` 完全对称，只是方向从 T 轴变成 H/W 轴。 / `spatial_tiled_encode` in the same file tiles along H/W instead of T — structurally identical, just a different axis.
- **Wan2.1 `vae.py` 的流式编码**: Wan2.1 的 VAE 用滑动窗口逐帧处理（类因果流式推理），不做 blend 而是靠因果卷积的历史感受野自然对齐——trade-off 是延迟比 tile 方案低但训练时显存没有 tile 方案省。 / Wan2.1's VAE uses a sliding-window causal approach rather than explicit blending — lower latency but higher training memory than tile-based VAE.

## 注意事项 / Caveats / when it breaks

- **`tile_overlap_factor` 过小会导致接缝 / Too-small `tile_overlap_factor` causes seams**: 重叠太少，`blend_t` 融合宽度不够，接缝会变成画面里明显的"闪帧"。通常 `tile_overlap_factor ≥ 0.25` 才能让接缝不可见。 / Too little overlap means `blend_t` can't smooth the boundary, producing visible "flash frames" at tile seams. Typically `tile_overlap_factor ≥ 0.25` is needed for invisible seams.
- **首帧 +1 的因果上下文只在因果卷积模型里有意义 / The +1 causal context frame only matters for causal conv models**: 如果你换成非因果的 3D 卷积（双向时序依赖），+1 frame 和 `tile[:, :, 1:]` strip 的逻辑就是多余的甚至有害的。要按模型架构判断是否保留。 / If you switch to a non-causal 3D conv (bidirectional temporal attention), the +1 frame and `tile[:, :, 1:]` strip is redundant or harmful. Match this logic to your encoder architecture.
- **梯度穿过 blend_t 的 for 循环很慢 / Gradient through the `blend_t` for-loop is slow**: 训练时 `blend_t` 里的 Python for 循环会产生 `blend_extent` 个独立 autograd 节点，backward 很慢。可以用 `torch.linspace` + einops 把 blend 写成一个向量化操作来提速。 / During training the Python for-loop in `blend_t` produces `blend_extent` separate autograd nodes, making backward slow. Rewrite as a vectorized operation with `torch.linspace` + einops for a significant speedup.

## 延伸阅读 / Further reading

- [HunyuanVideo 论文](https://arxiv.org/abs/2412.03603) — 描述了因果 VAE 的设计，以及时序 tile 在长视频生成中的作用。
- [Open-Sora `autoencoder_kl_causal_3d.py` 完整文件](https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/hunyuan_vae/autoencoder_kl_causal_3d.py) — 还包含 `blend_v`、`blend_h`、`spatial_tiled_encode`，以及它们与时序 tile 的嵌套组合。
- [Stable Diffusion Tiled VAE 实现](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/blob/main/tiledvae.py) — 空间 tile 的经典实现，比对这个可以看清时序 tile 如何把同样思路拓展到 T 轴。
