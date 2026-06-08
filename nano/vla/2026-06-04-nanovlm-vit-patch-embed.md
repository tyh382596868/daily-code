---
date: 2026-06-04
topic: vla
source: vla
repo: huggingface/nanoVLM
file: models/vision_transformer.py
permalink: https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/models/vision_transformer.py#L7-L43
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, vla, vision-encoder, vit, patch-embedding, siglip]
build_role: vision encoder front-end (image -> patch tokens) for nanoVLA
---

# 一个 Conv2d 就是整个 patch embedding / One Conv2d is the entire patch embedding

> **一句话 / In one line**: `nn.Conv2d(in=3, out=embd_dim, kernel=patch, stride=patch)` 同时完成"切 patch"和"线性投影"两件事,加上一个可学习的位置嵌入,一个完整 ViT 的输入层就 37 行结束。 / `nn.Conv2d(in=3, out=embd_dim, kernel=patch, stride=patch)` simultaneously crops every 16×16 patch and projects each to `embd_dim`. Add a learnable positional embedding and the entire ViT input layer is 37 lines.

## 为什么重要 / Why this matters

每个 VLA 的图像支路前几行长得都一样:把 RGB 像素变成一串 patch token 喂进 transformer。问题是这一步看起来需要 unfold + reshape + Linear 三步,实际上 PyTorch 里一行 Conv2d 就够了。学会这个"用 Conv2d 当 patch embedding"的小技巧后,你看 SigLIP、CLIP、DINO、SAM、SmolVLM 的视觉塔会发现它们前几行一模一样 —— 这就是 ViT 论文里的标准做法,直接成了行业默认。nanoVLM 把它写得特别干净,没有 abstract base class、没有 config 继承,适合直接抄进你的 nanoVLA 里。

Every VLA's image pathway starts with the same step: turn RGB pixels into a stream of patch tokens for the transformer. The procedure *looks* like it needs `unfold + reshape + Linear`, but in PyTorch a single `Conv2d` does both jobs. Once you internalize "Conv2d as patch embedding," you'll spot the same three lines at the top of SigLIP, CLIP, DINO, SAM, and SmolVLM vision towers — it's the standard recipe from the ViT paper and the industry default. nanoVLM writes it with no abstract base classes and no nested configs, copy-pasteable straight into your own nanoVLA.

## 代码 / The code

`huggingface/nanoVLM` — [`models/vision_transformer.py`](https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/models/vision_transformer.py#L7-L43)

```python
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py#L245
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

1. **`nn.Conv2d(in=3, out=embd_dim, kernel=patch_size, stride=patch_size)`**:
   - 中文: 整段代码的关键。这个 Conv2d 的 kernel 和 stride 都等于 patch_size,意味着两件事同时发生:(1) 在空间上把图像切成 `patch_size × patch_size` 的不重叠小块;(2) 每个 patch 的 `3 * patch_size^2` 个像素被一个线性层投影到 `embd_dim`。和"先 unfold 再 Linear"完全等价,但只用一次 cuDNN kernel。
   - English: The pivot line. With kernel = stride = patch_size, the Conv2d does two things at once: (1) spatially partition the image into non-overlapping `patch_size × patch_size` tiles, and (2) linearly project each tile's `3 * patch_size^2` pixels to `embd_dim`. Mathematically identical to `unfold + Linear`, but a single cuDNN kernel.
2. **`x = self.conv(x)`** 后形状是 `(B, embd_dim, H/p, W/p)`:
   - 中文: 注意 channel 维度变成了 `embd_dim`,空间维度从 `H × W` 收缩到 `H/patch × W/patch`。224 / 14 = 16,所以 224 输入 + patch=14 出 `16 × 16 = 256` 个 patch。
   - English: After the conv the tensor is `(B, embd_dim, H/p, W/p)`. The channel axis is now `embd_dim`; the spatial axes have shrunk from `H × W` to `H/p × W/p`. With 224 / 14 = 16 you get a `16 × 16 = 256`-patch grid.
3. **`x = x.flatten(2)` + `x.transpose(1, 2)`**:
   - 中文: flatten 把 H/p、W/p 两维合成一个 N=num_patches,然后 transpose 把 channel 和序列调换 —— 得到 transformer 标准输入 `(B, N, embd_dim)`。两步零参数,几乎不耗时。
   - English: `flatten(2)` collapses H/p × W/p into one length-N axis; `transpose(1, 2)` swaps channel and sequence to produce the standard transformer input `(B, N, embd_dim)`. Both zero-parameter and essentially free.
4. **`cls_token` + `position_embedding`**:
   - 中文: cls_token 是一个可学习的 `(1, 1, D)` 张量,broadcast 到 batch,prepend 到 patch 序列前面 —— 后面的 attention 让它"汇聚"图像全局信息,常被用作分类/池化输出。位置嵌入是直接加到 token 上的可学习 `(1, N+1, D)` 张量(SigLIP 没有 cls,只有 N 个)。
   - English: `cls_token` is a learnable `(1, 1, D)` tensor, broadcast to batch and prepended to the patch sequence. Subsequent attention layers let it "absorb" a global summary of the image, and downstream it's the classification / pool output. The positional embedding is a learnable `(1, N+1, D)` tensor added to the tokens (SigLIP omits the CLS, so it's `(1, N, D)`).
5. **`x = x + self.position_embedding`**:
   - 中文: 加法。位置编码是绝对位置(每个 patch 有自己的 D 维向量),不是 RoPE 这种相对位置 —— SigLIP/CLIP 都用绝对位置,后期一些 ViT 改用 RoPE 见 dinov3。
   - English: Plain addition. This is an *absolute* positional embedding — each patch has its own learnable D-dim vector — not RoPE. SigLIP and CLIP both use absolute positions; some newer ViTs (e.g. DINOv3) switched to RoPE.

## 类比 / The analogy

想象你拿一台老式胶片切割机把一张 224×224 的照片切成 16×16 的小方格,每个小方格再通过一台"颜色分析仪"输出一个 768 维的特征向量。然后你把所有方格按行优先顺序排成一条长队,长度 256。再给每个位置贴个号码牌(位置嵌入),整张照片就变成了一条 256 元素的"特征带"。这条带就是 ViT 的输入。Conv2d 是那台兼具"切割 + 颜色分析"两功能的奇迹机器。

Picture an old-fashioned film cutter that turns a 224×224 photo into a grid of 16×16 squares. Each square then passes through a "color analyzer" that outputs a 768-dim feature vector. You lay the squares out in row-major order into a queue of length 256, slap a numbered tag on each (positional embedding), and the photo is now a 256-element feature strip. That strip is the ViT input. The `Conv2d` is the miracle box that does both the cutting and the color analysis in one pass.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-VLA

> 这是 nanoVLA 视觉塔的**第一层**,curriculum 项 `vision-encoder` 的起点。

**中文**: 在你从零搭的 nanoVLA 里,数据流是 `image (B, 3, 224, 224) -> ViTPatchEmbeddings -> ViTBlock × N -> LayerNorm -> ModalityProjector -> VLM`。这个文件里的 `ViT` 类(forward 在 156-168 行)就是把今天教的 `ViTPatchEmbeddings` 串上 N 个 `ViTBlock`(标准 pre-LN transformer block)再过一次 LayerNorm,得到 `(B, num_patches, embd_dim)` 的 patch token。它的下游是 **modality projector**(5-29 那篇 nanoVLM 的 pixel shuffle 把 256 patch token 压成 64 个 fat token),然后接进 **VLM backbone**(5-29 的 SmolVLM with expert)。如果你少了这一步,大模型只能吃文字,看不到图像。生产级实现要再补:(1) 多分辨率 + AnyRes 支持(不同输入分辨率的 patch 数动态变化,位置嵌入需要做 2D 插值);(2) 多相机融合(VLA 通常 3-4 个相机,需要 camera-id embedding);(3) 从 SigLIP / DINO / SAM 加载预训练权重 —— 见这个文件最后的 `from_pretrained` 方法(170+ 行)。

**English**: In a from-scratch nanoVLA, the data flow is `image (B, 3, 224, 224) -> ViTPatchEmbeddings -> ViTBlock × N -> LayerNorm -> ModalityProjector -> VLM`. The `ViT` class right below in this file (forward at lines 156-168) chains today's `ViTPatchEmbeddings` through N standard pre-LN `ViTBlock`s plus a final LayerNorm to produce `(B, num_patches, embd_dim)` patch tokens. Downstream it feeds the **modality projector** (the 5-29 nanoVLM pixel-shuffle note that squeezes 256 patch tokens into 64 fat tokens) and then the **VLM backbone** (the 5-29 SmolVLM-with-expert wiring). Skip this stage and your big model only ever sees text. Production-grade implementations add: (1) multi-resolution / AnyRes support (variable patch count, requires 2D interpolation of the positional embedding); (2) multi-camera fusion (VLAs typically use 3-4 cameras, needing a camera-id embedding); (3) loading pretrained weights from SigLIP / DINO / SAM — see the `from_pretrained` method later in this same file.

**依赖关系 / Dependencies**: 无前置 curriculum 依赖。下游连接 `modality-projector`(已覆盖 5-29)、`vlm-backbone-wiring`(已覆盖 5-29)。 / No prerequisites. Downstream connects to `modality-projector` (covered 5-29) and `vlm-backbone-wiring` (covered 5-29).

## 自己跑一遍 / Try it yourself

```python
import torch, torch.nn as nn

class ViTPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=14, embd_dim=384):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.conv = nn.Conv2d(3, embd_dim, kernel_size=patch_size, stride=patch_size)
        self.pos = nn.Parameter(torch.randn(1, self.num_patches, embd_dim) * 0.02)

    def forward(self, x):
        x = self.conv(x)                  # (B, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x + self.pos

img = torch.randn(2, 3, 224, 224)
ve = ViTPatchEmbed(224, 14, 384)
tokens = ve(img)
print("input :", img.shape)
print("tokens:", tokens.shape)
print("num patches expected:", (224 // 14) ** 2)
print("conv weight shape:", ve.conv.weight.shape, "<- (out=D, in=3, p, p)")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
input : torch.Size([2, 3, 224, 224])
tokens: torch.Size([2, 256, 384])
num patches expected: 256
conv weight shape: torch.Size([384, 3, 14, 14]) <- (out=D, in=3, p, p)
```

中文重点:conv 权重 `(384, 3, 14, 14)` 总共 `384 * 3 * 14 * 14 = 225,792` 个参数,这就是 patch embedding 的所有可学习权重(再加上 bias 和 pos)。一个 ViT 的"输入投影"加起来不到 1M。

The patch embedding has only `384 * 3 * 14 * 14 = 225,792` learnable weights (plus bias and pos). The whole input projection of a ViT is under 1M params — tiny compared to the transformer body downstream.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **SigLIP `SiglipVisionEmbeddings`** / **SigLIP `SiglipVisionEmbeddings`**: 完全相同的 Conv2d patch embed,只是没 CLS token、位置嵌入是 `nn.Embedding(num_patches, D)`(同价于这里的 `nn.Parameter`)。 / Identical Conv2d patch embed, just without CLS token and using `nn.Embedding(num_patches, D)` (functionally equivalent to a `nn.Parameter`).
- **DINOv2 / DINOv3 `PatchEmbed`** / **DINOv2 / DINOv3 `PatchEmbed`**: 同样的套路,但 DINOv3 把位置编码换成了 RoPE (你刚看过 dinov3 的 `rope_position_encoding.py`)。 / Same trick, but DINOv3 switched the positional encoding to RoPE (see today's other dinov3 file).
- **SAM `ImageEncoderViT.PatchEmbed`** / **SAM `ImageEncoderViT.PatchEmbed`**: 一模一样,只是处理 1024 分辨率,所以 `num_patches = 4096`。 / Identical implementation, just for 1024-res inputs so `num_patches = 4096`.
- **lerobot SmolVLA 的 `embed_image`** / **lerobot SmolVLA's `embed_image`**: 不是自己实现 ViT,而是调用 SmolVLM2 的 vision tower —— 但内部第一层完全一样。 / It doesn't reimplement the ViT — it calls SmolVLM2's pretrained vision tower — but inside that tower the first layer is identical to this.

## 注意事项 / Caveats / when it breaks

- **`img_size` 必须能被 `patch_size` 整除** / **`img_size` must divide `patch_size`**: 否则 conv 的输出空间维度不是整数,后续 `flatten` 会丢掉边缘像素。一般在 dataloader 里做 resize 到固定尺寸。 / Otherwise the conv output spatial dim isn't an integer and `flatten` silently drops edge pixels. Usually resize in the dataloader to a fixed size.
- **`position_embedding` 是绝对位置 + 固定长度** / **absolute positional embedding is fixed-length**: 切换分辨率(比如训练 224、推理 384)时 num_patches 变了,位置嵌入数量对不上。SigLIP 解决办法是 2D 双线性插值预训练的 pos embedding;DINOv3 用 RoPE 直接绕开这个问题。 / Switching resolution (e.g. train 224, infer 384) changes `num_patches` and the positional embedding length mismatches. SigLIP fixes it with 2D bilinear interpolation of the pretrained pos embed; DINOv3 sidesteps it by using RoPE.
- **`padding="valid"`** / **`padding="valid"`**: 不补零,所以确实是非重叠的 patch。如果误改成 `"same"` 会出 stride mismatch 报错。 / No padding, so patches are truly non-overlapping. If you flip it to `"same"` you'll hit a stride-mismatch error.
- **CLS token 的位置嵌入要 +1** / **the CLS token needs +1 in pos embed**: `(1, num_patches + 1, D)` 多出来的那一行是给 CLS 的位置嵌入。漏掉 +1 会导致前后 shape 对不上,而且 CLS 会拿到第一个 patch 的位置编码,后续 attention 出问题。 / The `+1` row in `(1, num_patches + 1, D)` is the CLS positional slot. Omitting it causes a shape mismatch and CLS ends up reusing patch 0's position, polluting downstream attention.

## 延伸阅读 / Further reading

- ViT paper "An Image is Worth 16x16 Words": <https://arxiv.org/abs/2010.11929>
- SigLIP (the basis for nanoVLM's vision tower): <https://arxiv.org/abs/2303.15343>
- karpathy/nanoGPT (the comments in this file reference nanoGPT's attention block): <https://github.com/karpathy/nanoGPT>
