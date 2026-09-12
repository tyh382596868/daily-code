---
date: 2026-09-06
topic: wam
source: wam
repo: dreamzero0/dreamzero
file: groot/vla/model/dreamzero/modules/wan_video_vae.py
permalink: https://github.com/dreamzero0/dreamzero/blob/ab790c198fbce33503358efbbd4187ce9a89adf3/groot/vla/model/dreamzero/modules/wan_video_vae.py#L3736-L3833
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, vae-encoder-decoder, causal-conv3d, cache]
build_role: vae-encoder-decoder advanced variant
---

# DreamZero VideoVAE：4 帧一组编码，缓存把时间切开 / DreamZero VideoVAE: Encode in Four-Frame Chunks, Carry the Cache Across Time

> **一句话 / In one line**: 这个 VAE 用 causal 3D 卷积和特征缓存处理长视频，编码时按 4 帧分块，解码时沿用同一套时间缓存。 / This VAE uses causal 3D convs and feature caches for long videos, encoding in 4-frame chunks and decoding with the same temporal cache logic.

## 为什么重要 / Why this matters

中文：WAM 的 VAE 不是纯粹的压缩器，它还决定时间信息能不能平滑跨 chunk 传递。DreamZero 这里把 encoder/decoder、causal conv、feature cache 和 chunk 边界全部绑在一起，长视频就能被稳定地拆开处理。

English: A WAM VAE is not just a compressor; it also decides whether time information can flow smoothly across chunks. DreamZero ties encoder, decoder, causal convs, feature caches, and chunk boundaries together so long videos can be processed in pieces without losing temporal continuity.

## 代码 / The code

`dreamzero0/dreamzero` — [`groot/vla/model/dreamzero/modules/wan_video_vae.py`](https://github.com/dreamzero0/dreamzero/blob/ab790c198fbce33503358efbbd4187ce9a89adf3/groot/vla/model/dreamzero/modules/wan_video_vae.py#L3736-L3833)

```python
class VideoVAE_(nn.Module):
    def __init__(self, z_dim, *args, **kwargs):
        super().__init__()
        self.encoder = Encoder3d(...)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(...)
        self.num_causal_convs = 2

    def encode(self, x, scale):
        feat_map = {}
        iter_ = 1 + (t - 1) // 4
        out = self.encoder(x[:, :, :1], feat_map)
        for i in range(iter_):
            out = self.encoder(x[:, :, 1 + i * 4 : 1 + (i + 1) * 4], feat_map)
        mu, _ = self.conv1(out).chunk(2, dim=1)
        mu = mu * scale
        return mu
```

## 逐行讲解 / What's happening

1. **第 3736-3785 行 / Lines 3736-3785**:
   - 中文: 初始化里先搭好 3D encoder/decoder，再显式记录 causal conv 的数量，方便缓存管理。
   - English: The constructor wires up the 3D encoder/decoder first, then records how many causal convs exist so cache management stays explicit.
2. **第 3787-3815 行 / Lines 3787-3815**:
   - 中文: `encode()` 先处理首帧，再按 4 帧分块继续，`feat_map` 让后续 chunk 复用历史状态。
   - English: `encode()` handles the first frame, then continues in 4-frame chunks, with `feat_map` carrying reusable history across chunks.
3. **第 3816-3833 行 / Lines 3816-3833**:
   - 中文: `conv1` 输出被拆成均值和方差通道，最后再按 scale 归一到 latent 空间。
   - English: `conv1` splits its output into mean and variance channels, then the result is scaled back into latent space.

## 类比 / The analogy

中文：像把长胶片分段扫描进档案馆，每段都带上前一段的胶片编号，后面才能无缝接着读。

English: It is like scanning a long film reel into an archive in chunks, each chunk carrying the previous reel index so the next chunk can continue seamlessly.

## 在 nanoWAM 中的位置 / Where this lives in your nanoWAM

中文：这是 `vae-encoder-decoder` 的 advanced variant。它在 nanoWAM 里负责把原始视频压成可供 DiT 处理的 latent，同时在时间上保住 chunk 间的连续性。上游是视频输入和缓存状态，下游是 DiT/denoiser 看到的 latent 序列。少了这层，长视频会被迫一次性塞进显存，或者 chunk 边界处出现明显的时间断层。生产版通常还会加更细的缓存回收、不同分辨率的对齐和解码端的反向缓存。

English: This is an advanced variant of `vae-encoder-decoder`. In nanoWAM it compresses raw video into latents the DiT can consume while preserving continuity across chunk boundaries. Upstream are video inputs and cache state; downstream is the latent sequence the DiT/denoiser sees. Without this layer, long clips either have to fit in memory at once or show visible temporal seams at chunk edges. A production version usually adds finer cache eviction, alignment across resolutions, and reverse caches on the decode side.

## 自己跑一遍 / Try it yourself

```python
frames = list(range(10))
chunks = [frames[:1]] + [frames[1 + i * 4 : 1 + (i + 1) * 4] for i in range(3)]
print(chunks)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[[0], [1, 2, 3, 4], [5, 6, 7, 8], [9]]
```

中文：这个 VAE 的重点不是“能不能压缩”，而是“压缩时别把时间关系压碎”。

English: The point of this VAE is not merely compression; it is compression without crushing time relationships.
