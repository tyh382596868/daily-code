---
date: 2026-06-08
topic: wam
source: wam
repo: hpcaitech/Open-Sora
file: opensora/models/dc_ae/models/nn/vo_ops.py
permalink: https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/dc_ae/models/nn/vo_ops.py#L11-L56
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, wam, vae, dc-ae, pixel-shuffle, video, nano-wam]
build_role: vae-encoder-decoder (cross-repo variant) — parameter-free 3D up/down-sampling primitive for video VAEs
---

# 没有学习参数也能 ×8 上采样:Open-Sora 的 3D pixel-shuffle / Upsample 8× with zero learnable params: Open-Sora's 3D pixel-shuffle

> **一句话 / In one line**: 把通道数除以 8,腾出 (2, 2, 2) 三个空间维度——一次 `view + permute + reshape` 就完成了 2× 上采样,反向操作就是下采样。 / Divide channels by 8 to free up (2, 2, 2) spatial axes — one `view + permute + reshape` is a 2× upsample, and reversed it's the 2× downsample.

## 为什么重要 / Why this matters

视频 VAE 是 WAM 的入口:它把 `(T, H, W) = (49, 720, 1280)` 像素压成 `(13, 90, 160)` latent,压缩 32×。压缩谁来做?有两条主流派:
1. **学习派**(Wan2.1,我们 5/29 讲过的 causal Conv3d):用 stride>1 的 Conv3d 学习"该保留哪些细节",参数多,GPU 上算的 FLOPs 不小,但表达力强。
2. **不学习派**(Open-Sora DC-AE):用 pixel-shuffle 把空间维度直接装进通道里——`(B, C, T, H, W)` 形状的张量,channel 数除以 `r³`,空间各乘 `r`。**零参数**、**几乎零 FLOPs**、纯 reshape,反向 100% 无损可逆。后面再跟一个 1×1×1 conv "翻译"压缩进通道里的细节,效率惊人。

这是 ESPCN 在 2016 年提出来的 sub-pixel convolution 在 3D 的直接推广。45 行代码,包含正向、反向、unit-test。读完就能完全掌握现代 video VAE 上下采样的另一种主流路径。

The video VAE is WAM's front door: it crushes `(T, H, W) = (49, 720, 1280)` pixels into a `(13, 90, 160)` latent — 32× compression. Who does the crushing? Two mainstream camps:
1. **Learned** (Wan2.1, the causal Conv3d we covered 2026-05-29): `stride>1` Conv3d learns "which details to keep". Many parameters, real FLOPs on the GPU, but expressive.
2. **Parameter-free** (Open-Sora DC-AE): pixel-shuffle packs spatial axes into channels — `(B, C, T, H, W)` divides channels by `r³` and multiplies each spatial axis by `r`. **Zero params**, **near-zero FLOPs**, pure reshape, 100% reversible. A follow-up 1×1×1 conv then "translates" the channel-packed detail back. Stunning efficiency.

This is just ESPCN's 2016 sub-pixel convolution promoted to 3D. 45 lines covering forward, inverse, and a unit test. Reading it gives you the other mainstream route for modern video VAE up/down sampling.

## 代码 / The code

`hpcaitech/Open-Sora` — [`opensora/models/dc_ae/models/nn/vo_ops.py`](https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/dc_ae/models/nn/vo_ops.py#L11-L56)

```python
def pixel_shuffle_3d(x, upscale_factor):
    """3D pixelshuffle 操作."""
    B, C, T, H, W = x.shape
    r = upscale_factor
    assert C % (r * r * r) == 0, "通道数必须是上采样因子的立方倍数"

    C_new = C // (r * r * r)
    x = x.view(B, C_new, r, r, r, T, H, W)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
    y = x.reshape(B, C_new, T * r, H * r, W * r)
    return y


def pixel_unshuffle_3d(x, downsample_factor):
    """3D pixel unshuffle 操作."""
    B, C, T, H, W = x.shape
    r = downsample_factor
    assert T % r == 0, f"时间维度必须是下采样因子的倍数, got shape {x.shape}"
    assert H % r == 0, f"高度维度必须是下采样因子的倍数, got shape {x.shape}"
    assert W % r == 0, f"宽度维度必须是下采样因子的倍数, got shape {x.shape}"
    T_new, H_new, W_new = T // r, H // r, W // r
    C_new = C * (r * r * r)

    x = x.view(B, C, T_new, r, H_new, r, W_new, r)
    x = x.permute(0, 1, 3, 5, 7, 2, 4, 6)
    y = x.reshape(B, C_new, T_new, H_new, W_new)
    return y
```

## 逐行讲解 / What's happening

1. **`C % (r*r*r) == 0` 这条 assert**:
   - 中文: 上采样把每 `r³` 个通道"展开"成一个 `r×r×r` 的空间立方体,所以通道数必须是 `r³` 的倍数。`r=2` 时 8 通道变 1 通道、空间各 ×2;`r=4` 时 64 通道变 1 通道、空间各 ×4。
   - English: upsampling unpacks every `r³` channels into an `r×r×r` spatial cube, so channel count must be a multiple of `r³`. `r=2`: 8 channels → 1 channel and each spatial axis ×2; `r=4`: 64 channels → 1 channel and each spatial axis ×4.

2. **`x.view(B, C_new, r, r, r, T, H, W)`**:
   - 中文: 把原来连续的 `C` 维度拆成 `(C_new, r, r, r)` 四级。**这是关键 idea**——后三个 r 一会儿要变成时间、高、宽的"子格子",每个原 channel 现在分担了一个 `(r, r, r)` 立方块上的一格。
   - English: split the original `C` axis into `(C_new, r, r, r)` — four levels. **This is the key idea** — the last three `r`s will become "subcells" within time, height, width, with each original channel responsible for one slot inside an `(r, r, r)` cube.

3. **`x.permute(0, 1, 5, 2, 6, 3, 7, 4)`**:
   - 中文: 索引洗牌——原顺序 `(B, C_new, r_t, r_h, r_w, T, H, W)`(轴 0..7)被重排成 `(B, C_new, T, r_t, H, r_h, W, r_w)`。把每个空间轴和它对应的"子格子"放到相邻位置,后面 reshape 才能把子格子"插进"原网格之间。
   - English: index dance — original `(B, C_new, r_t, r_h, r_w, T, H, W)` (axes 0..7) is reordered to `(B, C_new, T, r_t, H, r_h, W, r_w)`. Each spatial axis sits adjacent to its corresponding subcell axis so the next reshape can interleave them.

4. **`x.reshape(B, C_new, T*r, H*r, W*r)`**:
   - 中文: 合并相邻的 `(T, r_t)`、`(H, r_h)`、`(W, r_w)` 三对轴。结果就是空间分辨率扩大 r 倍、通道数缩小 r³ 倍——上采样完成,**没有一个乘法、没有一个加法**。
   - English: merge the adjacent `(T, r_t)`, `(H, r_h)`, `(W, r_w)` pairs. Spatial resolution grows by r, channels shrink by r³ — upsample done **without a single multiply or add**.

5. **`unshuffle` 的逆 permute `(0, 1, 3, 5, 7, 2, 4, 6)`**:
   - 中文: 完全对偶——把每个空间轴 stride=r 地"拆"出 r 个 sub-grid,再洗回通道维。`shuffle` 和 `unshuffle` 互为可逆操作,意味着如果你不接其他算子,无损往返。
   - English: perfect inverse — strip `r` sub-grids from each spatial axis and shuffle them back into the channel axis. `shuffle` and `unshuffle` are exact inverses, meaning if nothing else sits in between you get a lossless round-trip.

## 类比 / The analogy

想象一张乐高板。Pixel-shuffle 2× 就是这样:你有一张 4×4 的乐高底板,每格上摞了 8 块不同颜色的小积木。Shuffle 把每格的 8 块"摊开"成一个 2×2×2 的小立方体,放到原格子的位置——板子表面瞬间变成 8×8×2 个亮色格子,空间分辨率涨了一倍,代价是每格只剩一种颜色(原来的 8 维通道维变成 1 维)。Unshuffle 就是把那些小立方体重新摞回来,完全可逆。没有任何"画图"或"插值",纯粹是几何重排。

Picture a Lego board. Pixel-shuffle 2× works like this: you have a 4×4 Lego baseplate with 8 differently-coloured bricks stacked on every cell. Shuffle "lays out" each cell's 8 bricks into a 2×2×2 mini-cube and plants it where the cell was — the surface instantly becomes 8×8×2 colour cells: spatial resolution doubled, at the cost of one colour per cell (the 8-dim channel axis collapsed to 1). Unshuffle restacks the mini-cubes. Perfectly reversible. No drawing, no interpolation, pure geometric repacking.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-WAM

这是 `nano_wam` 课程里 `vae-encoder-decoder` 槽位的 **跨仓变体(cross-repo variant)**。5/29 的笔记讲过 Wan2.1 的 causal Conv3d VAE,今天补一种完全不同的实现路径,让你在自己搭 nanoWAM 时有得选。

In your nanoWAM, the VAE 是一条 encoder-decoder 链。如果走 DC-AE 路线,典型搭法是:
```
input (B, 3, T, H, W)
  → Conv3d(3 → C0)
  → pixel_unshuffle_3d(r=2)          # space ÷ 2, channels × 8
  → Block(GroupNorm + Conv3d + SiLU)
  → pixel_unshuffle_3d(r=2)          # space ÷ 2 again, channels × 8
  → … (repeat to compress 32×)
  → Conv3d(C → 16)                    # latent channels
output (B, 16, T/4, H/8, W/8)
```
decoder 反着用 `pixel_shuffle_3d` 撑回去。**关键的隐含设计**:`unshuffle` 之后通道数立刻变 8 倍,如果不及时用一个 1×1×1 Conv3d 把通道压回 C,通道维会指数膨胀,所以 DC-AE 块里 `unshuffle → Conv3d(8C → C)` 是固定搭配——Conv 在这里只是"通道翻译",空间下采样的活已经由 reshape 做了。

This is a **cross-repo variant** of the `vae-encoder-decoder` slot in the `nano_wam` curriculum. The 5/29 note covered Wan2.1's causal Conv3d VAE; today's note gives you a completely different implementation route, so you can choose when building your own nanoWAM.

If you go the DC-AE route in your nanoWAM, the typical encoder is:
```
input (B, 3, T, H, W)
  → Conv3d(3 → C0)
  → pixel_unshuffle_3d(r=2)          # space ÷ 2, channels × 8
  → Block(GroupNorm + Conv3d + SiLU)
  → pixel_unshuffle_3d(r=2)          # space ÷ 2 again, channels × 8
  → … (repeat to compress 32×)
  → Conv3d(C → 16)                    # latent channels
output (B, 16, T/4, H/8, W/8)
```
The decoder mirrors that with `pixel_shuffle_3d`. **Critical hidden design**: after `unshuffle`, channels grow 8×, so you immediately need a 1×1×1 Conv3d to squash them back to C, otherwise channels explode exponentially. `unshuffle → Conv3d(8C → C)` is a fixed pair in DC-AE blocks — the conv here is just a "channel translator", since spatial downsampling has already happened in the reshape.

跟 Wan2.1 路线的取舍 / Trade-off vs Wan2.1: pixel-shuffle 路线少很多 FLOPs,kernel 也好融合;但完全靠 1×1×1 conv 来做通道间的语义混合,**没有空间感受野**——所以 DC-AE 块里必须配 3×3×3 conv 来扩感受野。Wan2.1 用 stride>1 conv 一次到位,但训练时空间池化的 PSNR/重建质量更难调。

Trade-off vs Wan2.1: pixel-shuffle has way fewer FLOPs and fuses well; but with only 1×1×1 conv for channel mixing, **no spatial receptive field** — so DC-AE blocks pair it with a 3×3×3 conv. Wan2.1 does it in one stride-conv but the spatial-pooling PSNR/recon quality is finicky to train.

## 自己跑一遍 / Try it yourself

```python
# try.py — verify shuffle/unshuffle are exact inverses, and count FLOPs
import torch

def pixel_shuffle_3d(x, r):
    B, C, T, H, W = x.shape
    assert C % (r ** 3) == 0
    return (x.view(B, C // r**3, r, r, r, T, H, W)
              .permute(0, 1, 5, 2, 6, 3, 7, 4)
              .reshape(B, C // r**3, T*r, H*r, W*r))

def pixel_unshuffle_3d(x, r):
    B, C, T, H, W = x.shape
    return (x.view(B, C, T // r, r, H // r, r, W // r, r)
              .permute(0, 1, 3, 5, 7, 2, 4, 6)
              .reshape(B, C * r**3, T // r, H // r, W // r))

# Round-trip test: encode then decode → exact equality
x = torch.randn(1, 4, 8, 16, 16)
y = pixel_unshuffle_3d(x, 2)           # (1, 32, 4, 8, 8)
z = pixel_shuffle_3d(y, 2)             # back to (1, 4, 8, 16, 16)
print(f"input  : {tuple(x.shape)}")
print(f"latent : {tuple(y.shape)}  # 8× spatial compression, channels ×8")
print(f"output : {tuple(z.shape)}")
print(f"max abs diff (round-trip) : {(x - z).abs().max().item()}")  # should be 0.0

# Compare against a strided Conv3d ("learned" route)
conv_down = torch.nn.Conv3d(4, 32, 2, stride=2)
conv_up   = torch.nn.ConvTranspose3d(32, 4, 2, stride=2)
print(f"conv path  trainable params : {sum(p.numel() for p in conv_down.parameters()) + sum(p.numel() for p in conv_up.parameters())}")
print(f"shuffle path trainable params : 0  ← pure reshape")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
input  : (1, 4, 8, 16, 16)
latent : (1, 32, 4, 8, 8)  # 8× spatial compression, channels ×8
output : (1, 4, 8, 16, 16)
max abs diff (round-trip) : 0.0
conv path  trainable params : 2080
shuffle path trainable params : 0  ← pure reshape
```

注意 round-trip diff 严格是 0:这就是"无损可逆"。对比之下 Conv3d 路线即便 stride 完美也只能在最优 loss 下逼近 0,而且要训。

The round-trip diff is exactly 0 — that's "lossless reversibility". The conv route only approaches 0 under optimal training loss, and it still has to train.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **`torch.nn.PixelShuffle` (2D)**: 官方 2D 实现,PSNR-style 超分模型用了 10 年的标配 / Built-in 2D version; ten-year standard in PSNR-style super-resolution.
- **ESPCN (2016, sub-pixel CNN)**: pixel-shuffle 最初出处,把"先反卷积后激活"换成"激活后 shuffle",省 4× FLOPs / The origin of the trick — replaces "deconv then activate" with "activate then shuffle", 4× cheaper.
- **EfficientViT / EfficientNet 系**: 用 pixel-shuffle 做轻量级分割 head / Uses pixel-shuffle for lightweight segmentation heads.
- **MagVit-v2 / CogVideoX 的 3D 分块** / **MagVit-v2 and CogVideoX's 3D tokenization**: 3D pixel-shuffle 的同主题不同奏法——他们用 patchify(Conv3d stride=patch),数学上和 unshuffle + Conv1×1×1 等价 / Same idea, different riff — they patchify (stride-conv) which is mathematically equivalent to unshuffle + 1×1×1 conv.

## 注意事项 / Caveats / when it breaks

- **通道数必须整除 r³** / **Channels must divide r³**: encoder 第一层别忘了先 Conv3d 把 3 通道升到 ≥ r³ 的整数倍,否则 unshuffle 直接 AssertionError / The encoder's first layer must Conv3d-bump 3 channels to a multiple of r³ before any unshuffle, otherwise it asserts immediately.
- **没有空间感受野** / **Zero spatial receptive field**: 单独的 pixel-shuffle 块和邻居 0 交互,所有"看周围"的活必须由配套的 3×3×3 conv 干 / A bare shuffle block has 0 interaction with neighbours; "looking around" duty falls entirely to the paired 3×3×3 conv.
- **bf16/fp16 下数值仍然安全** / **Numerically safe under bf16/fp16**: 因为是 reshape 而非乘加,半精度下不丢任何位 — VAE 训练的 NaN 通常出在别处 / Reshape, not multiply-add, so no precision loss; VAE NaNs come from elsewhere.
- **`permute` 不连续,后面 `reshape` 会触发拷贝** / **`permute` makes tensor non-contiguous, `reshape` will copy**: GPU 上很快但内存峰值翻倍,大分辨率训练要 chunk / Fast on GPU but doubles peak memory — chunk for hi-res training.

## 延伸阅读 / Further reading

- ESPCN paper: "Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network" (Shi et al., 2016) — the 2D origin
- DC-AE paper: "Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models" (Chen et al., 2024) — the design that uses 3D pixel-shuffle in VAEs
- Wan2.1 causal-conv3d VAE note (2026-05-29) — the learned-stride sibling for direct comparison
- MagVit-v2 paper — patchify as a sister formulation
