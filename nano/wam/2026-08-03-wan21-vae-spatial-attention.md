---
date: 2026-08-03
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/modules/vae.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vae.py#L223-L262
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, vae, attention]
build_role: vae-encoder-decoder advanced variant, framewise spatial attention inside the video VAE
---

# Wan2.1 VAE attention：每一帧内部做空间自注意力 / Wan2.1 VAE Attention: Run Spatial Self-Attention Inside Each Frame

> **一句话 / In one line**: `AttentionBlock` 把 `[B,C,T,H,W]` 摊成 `(B*T)` 张图，在每帧内部做单头空间 attention，再残差加回视频。 / `AttentionBlock` flattens `[B,C,T,H,W]` into `(B*T)` images, applies single-head spatial attention per frame, then adds the result back residually.

## 为什么重要 / Why this matters

VAE 不是只能靠卷积压缩视频。卷积擅长局部纹理，attention 可以让同一帧内远距离像素互相通信，帮助 latent 记住物体整体形状和空间关系。

A VAE does not have to rely only on convolutions. Convolutions capture local texture; attention lets distant positions inside the same frame communicate, helping the latent preserve object shape and spatial relations.

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/modules/vae.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vae.py#L223-L262)

```python
class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.norm(x)
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3,
                                         -1).permute(0, 1, 3,
                                                     2).contiguous().chunk(
                                                         3, dim=-1)

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
        )
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

        x = self.proj(x)
        x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
        return x + identity
```

## 逐行讲解 / What's happening

1. **第 232-239 行 / Lines 232-239 (small attention block)**:
   - 中文: `1x1 Conv2d` 同时产生 Q/K/V，输出投影权重初始化为 0，让残差分支一开始不扰动 VAE。
   - English: A `1x1 Conv2d` produces Q/K/V, and the output projection starts at zero so the residual branch is initially harmless.
2. **第 241-244 行 / Lines 241-244 (frame flattening)**:
   - 中文: 时间维不参与 attention；每个 frame 被当成一张独立图片。
   - English: Time is not included in attention; every frame becomes an independent image.
3. **第 246-262 行 / Lines 246-262 (spatial attention and restore)**:
   - 中文: 空间像素展平成序列，attention 后再 reshape 回 `[B,C,T,H,W]`。
   - English: Spatial positions become a sequence, then the result is reshaped back to `[B,C,T,H,W]`.

## 类比 / The analogy

这像逐帧开会：每一帧里的所有像素都能讨论同一张图里的关系，但会议室不跨帧，时间连续性留给别的模块处理。

It is like holding one meeting per frame: all pixels in that frame can discuss spatial relationships, but meetings do not cross frames. Temporal continuity is handled elsewhere.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

这是 `vae-encoder-decoder` 的 advanced variant。上游是卷积 VAE 的 feature map，下游是更有全局空间感的 latent。省掉它，nanoWAM 仍能跑，但对大物体、遮挡和远距离依赖的表达会更弱。生产级实现要平衡 attention 的显存和分辨率。

This is an advanced variant of `vae-encoder-decoder`. Upstream is the convolutional VAE feature map; downstream is a latent with stronger global spatial context. A nanoWAM can run without it, but large objects, occlusion, and long-distance dependencies are harder to encode. A production version must balance attention memory against resolution.

## 自己跑一遍 / Try it yourself

```python
def framewise_mean_attention(video):
    out = []
    for frames in video:
        new_frames = []
        for frame in frames:
            avg = sum(frame) / len(frame)
            new_frames.append([x + avg for x in frame])
        out.append(new_frames)
    return out

video = [[[1, 2, 3], [10, 20, 30]]]
print(framewise_mean_attention(video))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[[[3.0, 4.0, 5.0], [30.0, 40.0, 50.0]]]
```

中文: 示例用均值代替 attention，展示“每帧内部混合，帧与帧不混合”的结构。  
English: The toy uses a mean instead of attention to show the structure: mix inside each frame, not across frames.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Latent diffusion VAE attention** / **Latent diffusion VAE attention**: 图像 VAE 常在中低分辨率层插入 self-attention。 / Image VAEs often insert self-attention at mid or low resolutions.
- **Open-Sora VAE blocks** / **Open-Sora VAE blocks**: 也会在压缩链路里混合卷积和注意力。 / They also mix convolution and attention in the compression path.

## 注意事项 / Caveats / when it breaks

- **高分辨率成本** / **High-resolution cost**: attention 序列长度是 `H*W`，太早使用会很贵。 / The attention length is `H*W`, so using it too early is expensive.
- **只做帧内关系** / **Frame-local only**: 这段不会建模跨帧运动。 / This block does not model motion across frames.

## 延伸阅读 / Further reading

- Wan2.1 VAE attention source: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vae.py#L223-L262
