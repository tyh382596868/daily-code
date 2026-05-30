---
date: 2026-05-29
topic: vla
source: vla
repo: huggingface/nanoVLM
file: models/vision_transformer.py
permalink: https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/models/vision_transformer.py#L7-L44
difficulty: beginner
read_time: ~8 min
tags: [code-of-the-day, vla, vision-encoder, patch-embed, vit]
build_role: Vision encoder — turns raw camera images into the patch tokens a VLA backbone consumes
---

# 一个 Conv2d 就是整个 patch embedding / One Conv2d is the entire patch embedding

> **一句话 / In one line**: nanoVLM 把图像切 patch 这件事用**一个 stride=patch_size 的 Conv2d** 一步搞定 —— 卷积核扫过不重叠的 16×16 块,每块直接变成一个 token,再 flatten + 加位置编码就得到 VLA backbone 要吃的视觉 token 序列。 / nanoVLM turns an image into patch tokens with **a single stride=patch_size Conv2d** — the kernel sweeps non-overlapping 16×16 blocks, each block becomes one token, then flatten + add position embedding yields the visual token sequence a VLA backbone consumes.

## 为什么重要 / Why this matters

VLA 的第一步永远是"把相机图像变成 token"。很多人以为 patch embedding 需要复杂的切块逻辑(unfold、reshape、循环……),其实 ViT 的原始洞察就是:**一个 kernel_size = stride = patch_size 的卷积,等价于"把每个不重叠 patch 投影成一个向量"**。nanoVLM 这 38 行把整个视觉编码入口讲透 —— 它是你 nanoVLA 里夹在"相机"和"modality projector"之间的那块,也是后面所有 attention 的输入来源。看懂这一段,你就知道为什么 ViT/SigLIP/DINO 的第一层都是一个大 stride 卷积。

The first thing any VLA does is turn camera pixels into tokens. People often assume patch embedding needs fiddly slicing (unfold, reshape, loops). The original ViT insight is simpler: **a conv with kernel_size = stride = patch_size is exactly "project each non-overlapping patch into a vector"**. nanoVLM's 38 lines lay the whole visual entry point bare — it's the block between your camera and the modality projector in nanoVLA, and the source of every downstream attention input. Understand this and you understand why ViT/SigLIP/DINO all start with one big-stride conv.

## 代码 / The code

`huggingface/nanoVLM` — [`models/vision_transformer.py`](https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/models/vision_transformer.py#L7-L44)

```python
class ViTPatchEmbeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.img_size = cfg.vit_img_size
        self.patch_size = cfg.vit_patch_size
        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.cls_flag = cfg.vit_cls_flag
        self.embd_dim = cfg.vit_hidden_dim

        # Conv layer to extract the patches
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=self.embd_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        if self.cls_flag:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embd_dim))
            self.position_embedding = nn.Parameter(torch.rand(1, self.num_patches + 1, self.embd_dim))
        else:
            self.position_embedding = nn.Parameter(torch.rand(1, self.num_patches, self.embd_dim))

    def forward(self, x):
        x = self.conv(x)  # extract patches
        x = x.flatten(2)  # flatten the patches into a single dimension
        x = x.transpose(1, 2)  # transpose to (batch_size, num_patches, hidden_dim)

        # Add CLS token (according to original ViT Paper) and position embeddings
        if self.cls_flag:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        x = x + self.position_embedding
        return x
```

## 逐行讲解 / What's happening

1. **`num_patches = (img_size // patch_size) ** 2` / Patch count**:
   - 中文:224×224 图、patch 16,就是 `(224//16)² = 14² = 196` 个 patch。这个数决定了视觉 token 序列长度 —— 也是为什么昨天的 modality projector 要用 pixel shuffle 压缩它。
   - English: a 224×224 image with patch 16 gives `(224//16)² = 196` patches. This count *is* the visual token sequence length — which is exactly why yesterday's modality projector pixel-shuffles it down.

2. **`nn.Conv2d(3, embd_dim, kernel=patch, stride=patch)` 是核心 / The conv is the whole trick**:
   - 中文:`kernel_size == stride == patch_size` 意味着卷积核每次跳一整个 patch、彼此不重叠。每个 16×16×3 的 patch 被卷积核内积成一个 `embd_dim` 维向量 —— 数学上等价于"把 patch 展平成 768 维,再乘一个 `[768, embd_dim]` 的线性层",但用 conv 写一行就完事,还能用 cuDNN 加速。
   - English: `kernel_size == stride == patch_size` makes the kernel hop exactly one patch at a time, no overlap. Each 16×16×3 patch is inner-product'd into one `embd_dim` vector — mathematically identical to "flatten the patch to 768 dims and apply a `[768, embd_dim]` Linear", but one conv line, cuDNN-accelerated.

3. **`padding="valid"` / No padding**:
   - 中文:不补零,要求 `img_size` 能被 `patch_size` 整除,否则边缘 patch 丢失。VLA 里相机分辨率通常预 resize 到整除尺寸。
   - English: no padding, so `img_size` must be divisible by `patch_size` or edge patches are dropped. VLA pipelines pre-resize camera frames to a divisible size.

4. **`flatten(2) + transpose(1, 2)` / Reshape to a token sequence**:
   - 中文:conv 输出是 `[B, embd_dim, H/p, W/p]`(图像格式);`flatten(2)` 把空间两维合并成 `[B, embd_dim, num_patches]`;`transpose(1,2)` 变成 transformer 要的 `[B, num_patches, embd_dim]`。从"通道在前"翻成"序列在前"。
   - English: the conv outputs `[B, embd_dim, H/p, W/p]` (image layout); `flatten(2)` merges the spatial dims to `[B, embd_dim, num_patches]`; `transpose(1,2)` gives the transformer layout `[B, num_patches, embd_dim]`. Channels-first becomes sequence-first.

5. **`cls_token` 可选 / Optional CLS token**:
   - 中文:原版 ViT 在序列前面拼一个可学习的 `cls_token` 用来做分类聚合。VLA 里通常**不需要** CLS(我们要所有 patch token 喂给 LM,不是做单标签分类),所以 `cls_flag` 经常是 False。SigLIP 就没有 CLS。
   - English: vanilla ViT prepends a learnable `cls_token` for classification pooling. VLAs usually **don't** need it (all patch tokens feed the LM, we're not doing single-label classification), so `cls_flag` is often False. SigLIP drops the CLS entirely.

6. **`position_embedding` 是 learnable 的 / Learnable position embedding**:
   - 中文:`nn.Parameter(torch.rand(1, num_patches, embd_dim))` —— 每个 patch 位置一个可学习向量,直接加到 token 上。注意这是**绝对位置编码**,不像 video DiT 那边用 RoPE。ViT 用 learnable 绝对位置就够了,因为图像分辨率固定。
   - English: `nn.Parameter(torch.rand(1, num_patches, embd_dim))` — one learnable vector per patch position, added directly. This is **absolute** positional encoding, unlike the video DiT's RoPE. ViT gets away with learnable absolute positions because image resolution is fixed.

## 类比 / The analogy

像用一台"方格印章机"盖图章:把一张照片铺在桌上,印章是 16×16 的网格大小,每次盖一格(stride=16,不重叠),每盖一下就把那一格的内容压缩成一串数字记在卡片上。盖完整张照片,你手里就有 196 张卡片(token),再给每张卡片标个座位号(position embedding),整副卡片就能交给后面的"阅读者"(transformer)按顺序读了。

Picture a grid-stamp machine. Lay a photo on the table; the stamp is a 16×16 grid, and you press once per cell (stride 16, no overlap), each press compressing that cell into a string of numbers on a card. After stamping the whole photo you hold 196 cards (tokens); label each with a seat number (position embedding) and hand the deck to the reader (transformer).

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-VLA

中文:在 nanoVLA 里这是 `nano/vla/blocks/patch_embed.py` —— 视觉栈的**最底层入口**。上游:相机图像 `[B, 3, H, W]`(已经 resize + normalize);下游:堆 N 个 ViTBlock(self-attention + MLP)做视觉编码,然后是昨天讲的 modality projector(把 196 个 token pixel-shuffle 成 ~49 个再投到 LM 空间)。如果省掉这一层,你就没有视觉 token,VLA 直接瞎了。它的依赖关系最简单 —— 不依赖任何其他组件,是 nanoVLA 构建图的根节点之一。生产实现要补:(1) **预训练权重**(自己从头训 ViT 几乎不可能,通常加载 SigLIP / DINOv2 的 patch embed);(2) **多相机融合**(机器人有多个相机,每个相机各跑一次 patch embed,然后沿序列维拼接,或者用相机 id embedding 区分);(3) **分辨率自适应**(learnable 绝对位置编码不能外推,换分辨率要插值位置编码,所以 SOTA 越来越多用 2-D RoPE 替代)。

English: in nanoVLA this is `nano/vla/blocks/patch_embed.py` — the very bottom of the vision stack. Upstream: a camera image `[B, 3, H, W]` (already resized + normalized). Downstream: N stacked ViTBlocks (self-attention + MLP) for visual encoding, then yesterday's modality projector (pixel-shuffle 196 tokens to ~49 and project into LM space). Omit this and the VLA is blind. Its dependency graph is the simplest possible — it depends on nothing, a root node of the nanoVLA build. Production additions: (1) **pretrained weights** — training a ViT from scratch is impractical, so you load SigLIP/DINOv2 patch-embed weights; (2) **multi-camera fusion** — robots have several cameras; run patch embed per camera and concatenate along sequence, or add a camera-id embedding; (3) **resolution adaptivity** — learnable absolute position embeddings don't extrapolate, so changing resolution needs interpolation, which is why SOTA increasingly swaps in 2-D RoPE.

## 自己跑一遍 / Try it yourself

```python
# pip install torch
import torch, torch.nn as nn

class ViTPatchEmbeddings(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embd_dim=384):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.conv = nn.Conv2d(3, embd_dim, kernel_size=patch_size, stride=patch_size)
        self.pos = nn.Parameter(torch.rand(1, self.num_patches, embd_dim))
    def forward(self, x):
        x = self.conv(x)            # [B, embd, 14, 14]
        x = x.flatten(2)            # [B, embd, 196]
        x = x.transpose(1, 2)       # [B, 196, embd]
        return x + self.pos

patch = ViTPatchEmbeddings()
img = torch.randn(2, 3, 224, 224)
tokens = patch(img)
print("image :", img.shape)         # [2, 3, 224, 224]
print("tokens:", tokens.shape)      # [2, 196, 384]
print("num_patches:", patch.num_patches)

# Prove conv == per-patch Linear:
# a stride=patch conv is one Linear applied to each flattened patch
w = patch.conv.weight.flatten(1)    # [embd, 3*16*16]
print("Linear-equivalent weight shape:", w.shape)
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
image : torch.Size([2, 3, 224, 224])
tokens: torch.Size([2, 196, 384])
num_patches: 196
Linear-equivalent weight shape: torch.Size([384, 768])
```

中文:`[B, 3, 224, 224]` 的图变成 `[B, 196, 384]` 的 token 序列;最后一行证明 conv 权重 reshape 后就是一个 `[384, 768]` 的线性层 —— patch embedding 本质是线性投影。

English: a `[B, 3, 224, 224]` image becomes a `[B, 196, 384]` token sequence; the last line proves the conv weight reshapes into a `[384, 768]` Linear — patch embedding is fundamentally a linear projection.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **SigLIP / DINOv2 / CLIP ViT** / **SigLIP / DINOv2 / CLIP ViT**: 中文 — 全部用 stride=patch 的 conv 做 patch embed;VLA 通常直接加载它们的预训练权重。 / English — all use a stride=patch conv for patch embed; VLAs typically load their pretrained weights directly.
- **昨天的 Wan2.1 patchify (WAM)** / **Yesterday's Wan2.1 patchify (WAM)**: 中文 — WAM 那边用 `nn.Conv3d` 做 3D patchify,完全是这个 2D 版本的视频推广。 / English — the WAM side uses `nn.Conv3d` for 3-D patchify, a direct video generalisation of this 2-D version.
- **nanoVLM modality projector(前天讲过)** / **nanoVLM modality projector**: 中文 — patch embed 的下游邻居,把这里产出的 196 token 压成更少的 fat token。 / English — the downstream neighbour that pixel-shuffles these 196 tokens into fewer fat tokens.

## 注意事项 / Caveats / when it breaks

- **`img_size` 必须整除 `patch_size`** / **`img_size` must divide `patch_size`**: 中文 — `padding="valid"` 不补零,余数 patch 直接丢掉。机器人相机分辨率千奇百怪,务必预 resize。 / English — `padding="valid"` drops the remainder. Robot cameras come in odd resolutions; always pre-resize.
- **learnable 位置编码不能换分辨率** / **Learnable position embeddings don't transfer resolution**: 中文 — `num_patches` 写死在 `position_embedding` 形状里,换图像尺寸就要插值位置编码或重训。 / English — `num_patches` is baked into the position embedding shape; a new resolution needs interpolation or retraining.
- **从头训 ViT 几乎学不动** / **Training ViT from scratch barely converges**: 中文 — patch embed + ViT 在小数据上极难训。VLA 几乎总是加载 SigLIP/DINO 预训练权重,只 fine-tune。 / English — patch embed + ViT is extremely data-hungry. VLAs almost always load SigLIP/DINO pretrained weights and only fine-tune.

## 延伸阅读 / Further reading

- [An Image is Worth 16x16 Words (ViT, Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929)
- [SigLIP (Zhai et al., 2023)](https://arxiv.org/abs/2303.15343)
- [nanoVLM repo](https://github.com/huggingface/nanoVLM)
- [Today's VLA action summary doc](./README-action-survey.md)
