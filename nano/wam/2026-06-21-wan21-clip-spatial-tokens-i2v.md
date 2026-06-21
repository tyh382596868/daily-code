---
date: 2026-06-21
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/modules/clip.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/clip.py#L209-L301
difficulty: intermediate
read_time: ~12 min
tags: [code-of-the-day, wam, clip, i2v, spatial-tokens, vit, conditioning, curriculum]
build_role: text-conditioning (advanced I2V variant — CLIP spatial patch features alongside T5 text)
---

# Wan2.1 的 I2V 秘密:CLIP 的第 31 层特征,而不是 CLS pooling / Wan2.1's I2V secret: CLIP's block-31 features, not CLS pooling

> **一句话 / In one line**: Wan2.1 的图像到视频(I2V)模式不用 CLIP 的 pooled CLS embedding — 它用 `use_31_block=True` 跳过最后一个 Transformer block,返回 `[B, H*W/P², D]` 的空间 patch 特征网格,让 DiT 的 cross-attention 能对齐到参考图像的具体区域。/ Wan2.1's image-to-video (I2V) mode doesn't use CLIP's pooled CLS embedding — it uses `use_31_block=True` to skip the final Transformer block and return a `[B, H*W/P², D]` spatial patch feature grid, letting the DiT's cross-attention align to specific regions of the reference image.

## 为什么重要 / Why this matters

在你已经掌握的 T5 文本条件路径之后,I2V 引入了第二条条件流:参考图像。最简单的做法是把 CLIP 的 CLS token (pooled global embedding) 喂给 DiT——但这损失了所有空间信息。Wan2.1 多走了一步:用 CLIP ViT 第 31 块(共 32 块)的输出而不是最后一块,拿到的是 `[B, 196, 1024]` 的 patch 特征矩阵。DiT 的 cross-attention 对这 196 个空间位置的特征 attend——生成的每一帧不仅"全局语义"对齐参考图,连"哪个位置有什么"也得到保留。这就是为什么 Wan2.1 I2V 视频开头的构图和参考图高度一致。

After mastering the T5 text-conditioning path, I2V introduces a second conditioning stream: the reference image. The simplest approach is feeding CLIP's CLS token (pooled global embedding) to the DiT — but that discards all spatial information. Wan2.1 goes one step further: using CLIP ViT's output at block 31 of 32 (not the final block) returns a `[B, 196, 1024]` patch feature matrix. DiT's cross-attention attends over these 196 spatial positions, so generated frames align with the reference image not just globally ("a kitchen") but locally ("the red kettle is in the top-left corner"). That's why Wan2.1 I2V videos start with compositions nearly identical to the reference.

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/modules/clip.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/clip.py#L209-L301)

```python
class VisionTransformer(nn.Module):

    def __init__(self, image_size=224, patch_size=16, dim=768,
                 mlp_ratio=4, out_dim=512, num_heads=12, num_layers=12,
                 pool_type='token', pre_norm=True, ...):
        # ...
        self.patch_embedding = nn.Conv2d(3, dim,
                                         kernel_size=patch_size, stride=patch_size,
                                         bias=not pre_norm)
        self.cls_embedding  = nn.Parameter(gain * torch.randn(1, 1, dim))
        self.pos_embedding  = nn.Parameter(gain * torch.randn(
            1, self.num_patches + 1, dim))   # +1 for CLS
        self.transformer = nn.Sequential(*[
            AttentionBlock(...) for _ in range(num_layers)
        ])
        # head: CLS token → project to out_dim
        self.head = nn.Parameter(gain * torch.randn(dim, out_dim))

    def forward(self, x, interpolation=False, use_31_block=False):
        b = x.size(0)

        # ── Patch embedding ──
        x = self.patch_embedding(x).flatten(2).permute(0, 2, 1)
        # x: (B, H*W/P², dim)

        # ── Prepend CLS token ──
        x = torch.cat([self.cls_embedding.expand(b, -1, -1), x], dim=1)
        # x: (B, 1 + H*W/P², dim)

        # ── Positional embedding ──
        e = pos_interpolate(self.pos_embedding, x.size(1)) if interpolation else self.pos_embedding
        x = self.dropout(x + e)
        if self.pre_norm is not None:
            x = self.pre_norm(x)

        # ── Transformer forward — the I2V branch ──
        if use_31_block:
            x = self.transformer[:-1](x)   # skip the LAST block → return intermediate features
            return x                        # (B, 1 + num_patches, dim) — CLS included
        else:
            x = self.transformer(x)        # full 32-block pass
            return x                        # then head projects CLS to out_dim
```

以及 `CLIPModel.visual()` 的预处理 (同文件,更上方):

```python
# CLIPModel.visual() — 被 I2V pipeline 调用
def visual(self, videos):
    """
    videos: list of PIL frames 或 (B, C, H, W) float tensor [0,1]
    returns: (B, num_patches+1, dim)  当 use_31_block=True
    """
    # 缩放到 (224, 224),再 CLIP 归一化
    x = F.interpolate(videos, size=(224, 224), mode='bicubic', align_corners=False)
    x = (x - self.image_mean) / self.image_std
    with torch.cuda.amp.autocast(dtype=self.dtype):
        out = self.model.visual(x, use_31_block=True)
    return out   # spatial patch features, NOT pooled CLS
```

## 逐行讲解 / What's happening

1. **`self.patch_embedding(x).flatten(2).permute(0, 2, 1)`**
   - 中文: `Conv2d(kernel=patch_size, stride=patch_size)` 是 ViT 标准的 patch 化操作——把图像切成 `H/P × W/P` 个 `P×P` 小块,每块 embed 成一个 `dim` 维向量。`flatten(2)` 把 `(B, dim, H/P, W/P)` 展平为 `(B, dim, num_patches)`,`permute` 换轴成 `(B, num_patches, dim)`,符合 Transformer 的 `(batch, seq, dim)` 约定。
   - English: `Conv2d(kernel=patch_size, stride=patch_size)` is ViT's standard patchification — the image is cut into `H/P × W/P` tiles of size `P×P`, each embedded into a `dim`-dimensional vector. `flatten(2)` reshapes `(B, dim, H/P, W/P)` to `(B, dim, num_patches)`, and `permute` gives `(B, num_patches, dim)` — the Transformer's `(batch, seq, dim)` convention.

2. **`cls_embedding.expand(b, -1, -1)` → `torch.cat`**
   - 中文: 把一个可学习的 CLS token 前置到序列。这个 CLS token 在完整前向 (`use_31_block=False`) 结束后经 `self.head` 投影得到全局 embedding。但在 `use_31_block=True` 时,CLS token 的位置也保留在返回值里——虽然 I2V pipeline 主要用的是后面的 patch 位置。
   - English: Prepends a learnable CLS token to the sequence. In the full forward pass, this CLS is projected by `self.head` to produce the global embedding. When `use_31_block=True`, the CLS position is still included in the returned tensor — though the I2V pipeline primarily uses the spatial patch positions that follow it.

3. **`self.transformer[:-1](x)` — 跳过最后一块的关键**
   - 中文: `nn.Sequential.__getitem__` 支持切片——`self.transformer[:-1]` 是"除最后一块之外的所有 block"。为什么跳过最后一块?最后一块负责把空间信息汇聚到 CLS token,处理后的 patch 特征更"全局化"、空间信息被"稀释"。第 31 块的 patch 特征保留了更多局部/区域性细节,更适合 DiT 的空间 cross-attention。
   - English: `nn.Sequential.__getitem__` supports slicing — `self.transformer[:-1]` is "all blocks except the last one". Why skip the last block? The last block's job is to aggregate spatial information into the CLS token, after which patch features become more "global" and spatial detail is diluted. Block-31 patch features retain more local/regional detail, making them better suited for the DiT's spatial cross-attention.

4. **`F.interpolate(..., size=(224, 224))` in `visual()`**
   - 中文: CLIP 用固定分辨率 `224×224` 训练。`bicubic` 插值把参考帧缩放到这个尺寸,同时保持色彩信息。后续 CLIP 归一化 `(x - mean) / std` 把 `[0,1]` 范围的 float 图像调整到 CLIP 的训练分布。
   - English: CLIP was trained at a fixed `224×224` resolution. `bicubic` interpolation resizes the reference frame while preserving color fidelity. The CLIP normalization `(x - mean) / std` then shifts the `[0,1]` float image into CLIP's training distribution.

5. **输出形状 `(B, 197, 1024)` 的意义**
   - 中文: 224×224 图像,patch_size=16 → 14×14=196 个 patch。加上 CLS 共 197 个 token,每个 dim=1024(ViT-L 规格)。DiT 的 cross-attention query 来自视频 latent token,key/value 来自这 197 个 CLIP token——模型可以让每个视频位置 attend 到参考图的对应空间区域。
   - English: 224×224 image with patch_size=16 → 14×14=196 patches. Plus CLS = 197 tokens total, each dim=1024 (ViT-L spec). The DiT's cross-attention queries come from video latent tokens, with key/values from these 197 CLIP tokens — the model can attend each video position to the spatially corresponding region of the reference image.

## 在 nanoWAM 中的位置 / Where this lives in your nanoWAM

这是 `text-conditioning` 组件的**进阶 I2V 变体**。你的 nanoWAM 已经掌握了 T5 文本编码路径 (Open-Sora 版本, 2026-06-14)。I2V 在这条 T5 路径**旁边**增加了第二条条件流:

**输入/输出:**
- 输入: 参考视频帧 (PIL 图像 list 或 `[B, C, H, W]` tensor)
- 输出: `[B, 197, 1024]` 空间 patch 特征 — 直接作为 DiT cross-attention 的 key/value 拼接到 T5 特征旁边

**上游依赖:** `text-conditioning` (T5 编码) + `vae-encoder-decoder` (把参考帧编码到 latent space 前需要先有 VAE)

**下游:** DiT 的 cross-attention 层同时接收 T5 特征 (语义) 和 CLIP 特征 (视觉空间)

**省掉 CLIP I2V 流会发生什么?** 退化成纯文本到视频 (T2V) 模式。模型不知道参考图像长什么样,只能根据文字生成——开头帧和参考图的构图通常不一致。

**生产级还需要加什么?** (1) 如何把 T5 text features 和 CLIP image features 合并传给 DiT (Wan2.1 是 concat 后一起 cross-attend);(2) 在 I2V 模式下参考帧的第一帧通常用 VAE encode 后直接 inpaint 到 latent 序列 position 0,确保生成视频从参考帧精确开始;(3) `interpolation=True` 处理非 224×224 输入的位置 embedding 插值。

This is the **advanced I2V variant** of the `text-conditioning` curriculum component. Your nanoWAM already covers the T5 text encoding path (Open-Sora version, 2026-06-14). I2V adds a second conditioning stream *alongside* T5:

**I/O:** Input: reference video frames (PIL images or `[B, C, H, W]` tensor). Output: `[B, 197, 1024]` spatial patch features — concatenated with T5 features as key/values for the DiT's cross-attention.

**Upstream dependencies:** `text-conditioning` (T5 encoder) + `vae-encoder-decoder` (need VAE before encoding the reference frame into latent space).

**Downstream:** DiT cross-attention layers receive both T5 features (semantic) and CLIP features (visual spatial), typically concatenated.

**What if you skip the CLIP I2V stream?** You fall back to pure text-to-video (T2V). The model has no knowledge of the reference image's appearance and generates based on text alone — opening frame composition will often diverge from the reference.

**What production needs on top?** (1) How to merge T5 text features and CLIP image features for the DiT (Wan2.1 concatenates them and cross-attends jointly); (2) in I2V mode the reference frame is typically VAE-encoded and inpainted into latent position 0 to ensure the generated video starts from exactly the reference frame; (3) `interpolation=True` for handling non-224×224 inputs via positional embedding interpolation.

## 自己跑一遍 / Try it yourself

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniViT(nn.Module):
    """Stripped-down ViT with use_31_block support."""
    def __init__(self, img=32, patch=8, dim=64, layers=4):
        super().__init__()
        n = (img // patch) ** 2
        self.patch_emb = nn.Conv2d(3, dim, patch, patch)
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, n + 1, dim) * 0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, 4, dim * 4, batch_first=True)
            for _ in range(layers)
        ])
        self.head = nn.Linear(dim, 32)  # project CLS for T2V

    def forward(self, x, use_31_block=False):
        b = x.size(0)
        x = self.patch_emb(x).flatten(2).permute(0, 2, 1)  # (B, patches, dim)
        x = torch.cat([self.cls.expand(b, -1, -1), x], dim=1)
        x = x + self.pos
        blocks = self.blocks[:-1] if use_31_block else self.blocks
        for blk in blocks:
            x = blk(x)
        return x if use_31_block else self.head(x[:, 0])

vit = MiniViT()
img = torch.randn(2, 3, 32, 32)

t2v_emb = vit(img, use_31_block=False)  # global CLS → (B, 32)
i2v_emb = vit(img, use_31_block=True)   # spatial patches → (B, 1+16, 64)
print("T2V global:", t2v_emb.shape)      # torch.Size([2, 32])
print("I2V spatial:", i2v_emb.shape)     # torch.Size([2, 17, 64])
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
T2V global: torch.Size([2, 32])
I2V spatial: torch.Size([2, 17, 64])
```

中文: `use_31_block=True` 多出来的 seq 维度 (17 vs 1) 就是 DiT 可以 cross-attend 的空间网格——不是一个全局向量,而是 4×4=16 个 patch 特征加一个 CLS。

English: The extra sequence dimension from `use_31_block=True` (17 vs 1) is the spatial grid the DiT can cross-attend to — not a single global vector but 16 patch features (4×4) plus the CLS token.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **DALL-E 3 / Stable Diffusion XL refiner** / **SD-XL refiner**: 用 CLIP image embedding 引导 refiner 阶段,但通常只用 pooled CLS — 没有空间信息。Wan2.1 多保留了 patch 特征,这是其 I2V 质量更好的原因之一。/ Uses pooled CLS for image-to-image guidance but no spatial features. Wan2.1's retention of patch features is one reason its I2V composition is tighter.
- **IP-Adapter** / **IP-Adapter**: 同样用 CLIP image features 做 conditioning,区别是 IP-Adapter 通过一个独立的 cross-attention 路径注入,Wan2.1 直接把 CLIP 和 T5 的特征拼在一起传给同一套 cross-attention。/ Also uses CLIP image features for conditioning, but injects them via a separate cross-attention path rather than concatenating with T5 features.
- **CogVideoX** / **CogVideoX**: 用 VAE-encoded reference frame 直接拼接在 latent sequence 前面,而不是独立的 CLIP 特征流——更简单但 CLIP 的语义对齐损失了。/ Concatenates a VAE-encoded reference frame directly into the latent sequence rather than maintaining a separate CLIP feature stream — simpler but loses CLIP's semantic alignment.

## 注意事项 / Caveats / when it breaks

- **`use_31_block` 是 32 层 ViT 的写法** / **`use_31_block` assumes 32 ViT layers**: 如果你用的 CLIP 是 ViT-B(12 层)或 ViT-H(32+ 层),需要相应调整 `self.transformer[:-1]` 里的切片——"倒数第二层"才是通用写法。/ If your CLIP backbone is ViT-B (12 layers) or ViT-H (32+ layers), adjust the slice accordingly — "all but last" is the universal pattern.
- **CLS token 在 `use_31_block=True` 时仍在返回值里** / **CLS is still in the returned tensor with `use_31_block=True`**: 返回的序列第 0 位是 CLS,第 1-196 位是 patch。如果 DiT cross-attention 只需要 patch 特征,要 slice `out[:, 1:]`。
  / The returned sequence has CLS at position 0, patches at positions 1-196. If the DiT only needs patch features, slice `out[:, 1:]`.
- **分辨率不是 224×224 时** / **Non-224×224 inputs**: `F.interpolate` 会先缩放,但 ViT 的 pos_embedding 是为 224×224 设计的。`interpolation=True` 分支调用 `pos_interpolate` 做 2D 插值,否则位置编码会错位。/ `F.interpolate` handles resizing, but ViT's `pos_embedding` is sized for 224×224. The `interpolation=True` branch calls `pos_interpolate` to bilinearly resize positional embeddings, otherwise they'd misalign.

## 延伸阅读 / Further reading

- Wan2.1 技术报告: [arXiv (Wan-Video)](https://github.com/Wan-Video/Wan2.1)
- CLIP 原论文: [arXiv 2103.00020](https://arxiv.org/abs/2103.00020) — Learning Transferable Visual Models From Natural Language Supervision
- IP-Adapter (独立 CLIP cross-attn 路径): [arXiv 2308.06721](https://arxiv.org/abs/2308.06721)
- 对应 T5 文本条件基础版本 (2026-06-14): `nano/wam/2026-06-14-open-sora-hf-embedder-text-conditioning.md`
