---
date: 2026-06-02
topic: vla
source: vla
repo: huggingface/nanoVLM
file: models/vision_transformer.py
permalink: https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/models/vision_transformer.py#L7-L43
difficulty: beginner
read_time: ~9 min
tags: [code-of-the-day, vla, vision-encoder, patch-embedding, nano-vla]
build_role: nanoVLA / vision-encoder — turns raw camera pixels into the patch tokens that the rest of the system (modality projector → LM backbone → action head) consumes.
---

# 一个 Conv2d 等于整个 ViT 的入口 / One Conv2d *is* the entire ViT entry point

> **一句话 / In one line**: 一张 (3, 224, 224) 的图,经过一个 stride 等于 kernel 的 Conv2d,瞬间变成 196 个 patch token —— 加上 CLS 和绝对位置 embedding,这就是 nanoVLM/SigLIP/ViT 共用的 37 行入口。 / A (3, 224, 224) image goes through one Conv2d with stride == kernel and instantly becomes 196 patch tokens — add a CLS token and absolute position embeddings, and that's the 37-line entry point shared by nanoVLM / SigLIP / vanilla ViT.

## 为什么重要 / Why this matters

每个 VLA(Vision-Language-Action 模型)的第一步都是「pixels → tokens」。这一步看起来玄妙,实际只有一个 Conv2d 加上一组可学的位置 embedding。理解它有两个直接的工程价值:(1)以后你看任何 VLA paper 提到「multi-cam fusion」「dual-resolution input」「相机 ID 注入」,本质都是在这层下手 —— 改 Conv2d 就行;(2)在自己搭 nanoVLA 时,这是少数几个你完全可以从零写、不依赖任何预训练权重就能跑通的组件。

Every VLA (Vision-Language-Action) model starts with "pixels → tokens." It sounds mystical but it's literally one Conv2d plus learnable position embeddings. Two practical reasons to grok it: (1) future VLA papers talking about "multi-cam fusion," "dual-resolution input," or "camera-ID injection" all hack on this layer — you just modify the Conv2d; (2) when building your own nanoVLA, this is one of the very few components you can write from scratch without any pretrained weights and have it actually work.

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

1. **`num_patches = (img_size // patch_size) ** 2`**:
   - 中文: 比如 224 / 16 = 14,所以 196 patch。注意是整除 —— img_size 必须是 patch_size 的整数倍,不是的话直接错位。SigLIP 强制 224, ViT 强制 224/384,nanoVLM 把这做成 config。
   - English: e.g. 224 / 16 = 14, so 196 patches. Note integer division — img_size *must* be a multiple of patch_size, otherwise misalignment. SigLIP locks 224, ViT locks 224/384, nanoVLM makes it a config.

2. **`nn.Conv2d(3, embd_dim, kernel_size=patch_size, stride=patch_size, padding="valid")`**:
   - 中文: 整段最关键。kernel 和 stride 相同,意味着滑窗没有重叠 —— 每个 16×16 patch 被独立地映射成一个 `embd_dim` 维向量。数学上等价于「先把图切成 14×14 个 patch,每个 flatten 成 768 维,再过一个 Linear(768, embd_dim)」,但 Conv2d 在 cuDNN 里是单次融合 kernel,且省了 Python 切片开销。
   - English: The crux of the file. Same kernel and stride means no overlap — each 16×16 patch is independently mapped to one `embd_dim` vector. Mathematically equivalent to "slice into 14×14 patches, flatten each to 768, run a Linear(768, embd_dim)," but Conv2d compiles to a single fused cuDNN kernel and skips the Python slicing overhead.

3. **`cls_token = nn.Parameter(torch.zeros(1, 1, embd_dim))` (zero init!)**:
   - 中文: CLS 是可学的「总结 token」,初始化为 0。这是原始 ViT 的传统;SigLIP 不用 CLS 改用 pooled mean,所以 `cls_flag` 是 config 开关。VLA 通常会把 CLS 留着,后面给 LM 用作「整图摘要」。
   - English: CLS is the learnable "summary token," zero-initialized. Original ViT convention; SigLIP skips CLS and uses mean pooling, so `cls_flag` is a config switch. VLAs usually keep CLS — the LM uses it as a coarse "whole-image gist."

4. **`position_embedding = nn.Parameter(torch.rand(1, num_patches + 1, embd_dim))`**:
   - 中文: 可学的**绝对**位置 embedding。注意是 `torch.rand`(均匀 [0,1])不是 zeros —— 因为如果初始化为 0 加上去等于没加,网络梯度也分不开 patch 顺序。这种「learned absolute」是 ViT/SigLIP 的传统;现代视频/3D 模型更多用 RoPE 替代。
   - English: Learnable **absolute** position embedding. Note `torch.rand` (uniform [0,1]), not zeros — zero-initialized would add nothing and the network couldn't disambiguate patch order from gradients. "Learned absolute" is the ViT/SigLIP tradition; modern video/3D models prefer RoPE.

5. **`x = self.conv(x)`** → 形状 `(B, embd_dim, H/p, W/p)`:
   - 中文: 输入 `(B, 3, 224, 224)`,输出 `(B, 768, 14, 14)`。这一步把 spatial 维度从「像素网格」变成「patch 网格」。
   - English: Input `(B, 3, 224, 224)` → output `(B, 768, 14, 14)`. The spatial dim transitions from "pixel grid" to "patch grid."

6. **`x.flatten(2).transpose(1, 2)`** → 形状 `(B, num_patches, embd_dim)`:
   - 中文: 两步标准变形。`flatten(2)` 把 `(H/p, W/p)` 压成 `num_patches`,得到 `(B, 768, 196)`;`transpose(1, 2)` 调到 transformer 期望的 `(B, T, D)` 布局。**这两个操作不复制内存**(只是改 stride),所以零开销。
   - English: Two-step standard reshape. `flatten(2)` collapses `(H/p, W/p)` into `num_patches` giving `(B, 768, 196)`; `transpose(1, 2)` brings it to the `(B, T, D)` layout transformers want. **Neither copies memory** (just stride changes), so it's free.

7. **`cls_token.expand(x.shape[0], -1, -1)` + `torch.cat`**:
   - 中文: `expand` 不复制内存,只把 batch 维「广播」到 B。然后 cat 到 patch 序列前面。最终 token 序列长度是 `num_patches + 1`(如果开了 CLS)。
   - English: `expand` doesn't copy memory, just broadcasts the batch dim to B. Then cat to the front of the patch sequence. Final sequence length is `num_patches + 1` (with CLS enabled).

8. **`x = x + self.position_embedding`**:
   - 中文: 直接相加。注意 position embedding 的 shape 是 `(1, T, D)`,会自动广播到 batch。这一步是网络**唯一**知道 patch 顺序的来源 —— 删掉它,网络就把所有 patch 当成无序集合。
   - English: Direct add. Position embedding shape is `(1, T, D)`, broadcast over batch. This is the **only** source of patch ordering — delete it and the network treats all patches as an unordered set.

## 类比 / The analogy

想象你在做一份巨型马赛克拼图。每张小拼图(patch)是 16×16 像素,你把它们排成 14×14 的网格。然后每张拼图背面贴一张小卡片(`embd_dim` 维向量)记录它的纹理摘要 —— 这就是 Conv2d 干的事。然后你给每张卡片在边缘画一个编号(position embedding),否则你打乱了堆在桌上,完全不知道哪张该贴在拼图的哪个位置。CLS token 是你额外多印一张「整体描述卡」,放在堆顶,后面的人(LM)看一眼就知道这是一张猫还是狗。

Picture assembling a giant mosaic puzzle. Each small piece (patch) is 16×16 pixels and you lay them out in a 14×14 grid. Then you stick a tiny index card (`embd_dim` vector) on the back of each piece recording its texture summary — that's what Conv2d does. Then you write a corner number (position embedding) on every card, otherwise you shuffle them and have no idea where each goes back. The CLS token is an extra "overall description" card you place on top of the stack, so anyone reading (the LM) immediately knows "this is a cat" without flipping through all 196 pieces.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-VLA

**Curriculum slot**: `vision-encoder` (无依赖 / no deps — this is one of the very first components you write). Downstream consumer: `modality-projector` (covered 2026-05-29, nanoVLM pixel-shuffle), which compresses these 196 tokens down to ~64 "fat" tokens for the LM. Upstream provider: the camera / data loader.

在你自己的 nanoVLA 里,这就是视觉塔的入口模块,输入 `(B, num_cameras, 3, H, W)`,输出 `(B, num_cameras * num_patches, D)`。最小可工作版本 = 一个 ViTPatchEmbeddings + 一个相机 ID embedding(给每帧加一个标识它来自第几个相机的可学向量,通常直接加到 position embedding 上)+ 一个 `for camera in cameras: tokens = patch_embed(image); tokens += camera_emb` 的循环或 vmap。生产级要再补:(a)动态分辨率(SigLIP-NaViT 的做法,处理 224×336 这种非方形);(b)2D RoPE 或 ALiBi 替代绝对位置,让你的模型能 zero-shot 处理训练时没见过的分辨率;(c)对深度/触觉这类非 RGB 模态,改 `in_channels=N` 即可。

In your nanoVLA this is the vision tower's entry module: input `(B, num_cameras, 3, H, W)`, output `(B, num_cameras * num_patches, D)`. Minimum working version = one ViTPatchEmbeddings + one camera-ID embedding (a learned vector tagging each frame's source camera, usually added directly to the position embedding) + a `for camera in cameras: tokens = patch_embed(image); tokens += camera_emb` loop or vmap. Production needs: (a) dynamic resolution (SigLIP-NaViT style, handling 224×336 non-square inputs); (b) 2D RoPE or ALiBi instead of absolute positions so the model zero-shot generalizes to unseen resolutions; (c) for depth/tactile/non-RGB modalities, set `in_channels=N`. If you skip this module entirely, the LM never sees pixels — your VLA degenerates into a text-only language model with no visual grounding.

## 自己跑一遍 / Try it yourself

```python
# pip install torch
import torch
import torch.nn as nn

class ViTPatchEmbeddings(nn.Module):
    def __init__(self, img_size=224, patch_size=16, dim=768, use_cls=True):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.use_cls = use_cls
        self.conv = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        if use_cls:
            self.cls = nn.Parameter(torch.zeros(1, 1, dim))
            self.pos = nn.Parameter(torch.rand(1, self.num_patches + 1, dim))
        else:
            self.pos = nn.Parameter(torch.rand(1, self.num_patches, dim))

    def forward(self, x):
        x = self.conv(x).flatten(2).transpose(1, 2)
        if self.use_cls:
            x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], dim=1)
        return x + self.pos

img = torch.randn(2, 3, 224, 224)  # batch of 2 RGB images
emb = ViTPatchEmbeddings()(img)
print(f"shape: {emb.shape}")  # (2, 197, 768) — 196 patches + 1 CLS
print(f"params: {sum(p.numel() for p in ViTPatchEmbeddings().parameters()):,}")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
shape: torch.Size([2, 197, 768])
params: 742,656     # ~590K for Conv2d, ~150K for position table, ~768 for CLS
```

注意一个细节:整个 patch embedding 只有约 74 万参数 —— 不到 1MB。一个 ViT-B/16 的总参数量 86M 里,patch embed 不到 1%,几乎所有参数都在后面的 transformer block 里。**这就是为什么你可以放心从零训这一层,但 transformer 主干必须靠预训练。**

A subtle but important number: the whole patch embedding has ~742K parameters — under 1 MB. In a ViT-B/16 (86M total), patch embed is well under 1%; nearly all parameters live in the transformer blocks that follow. **That's exactly why you can comfortably train this layer from scratch, while the transformer backbone has to come from pretraining.**

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **OpenVLA 的多相机融合** / **OpenVLA's multi-cam fusion**: 用同一个 DINOv2 + SigLIP 双塔 patch embed,然后把 3 个相机的 token 在序列维 cat 起来扔给 LLaMA。代码在 `prismatic/models/backbones/vision/`。 / Uses one DINOv2 + SigLIP dual-tower patch embed, then concatenates tokens from 3 cameras along the sequence dim and hands them to LLaMA. Code at `prismatic/models/backbones/vision/`.
- **NVIDIA Isaac-GR00T 的 EagleVision encoder** / **Isaac-GR00T's EagleVision encoder**: 同样的 Conv2d patchifier,但加了「state token」—— 机器人本体感知(joint angles)经过一个 linear 后被插入 patch token 序列前。这就是上面提到的「在 patchify 这层做模态融合」的实操。 / Same Conv2d patchifier, but adds a "state token" — proprioception (joint angles) goes through a Linear and gets inserted before the patch tokens. Exact instance of the "fuse modalities at the patchify layer" pattern.
- **LeRobot SmolVLA 的 SigLIP 包装** / **LeRobot SmolVLA's SigLIP wrapper**: 直接用 HF transformers 的 `SiglipVisionModel`,patch embed 就是这段代码的精确翻版(连 weight key 都对得上,看 nanoVLM 的 `from_pretrained` mapping 就懂)。 / Wraps HF transformers' `SiglipVisionModel` directly — its patch embed is precisely this code (weight keys even line up; see nanoVLM's `from_pretrained` mapping).
- **NaViT (Pix2Struct/SigLIP-NaViT)** / **NaViT (Pix2Struct/SigLIP-NaViT)**: 同样的 Conv2d,但用变长序列+masking 支持任意宽高比 —— 你的 nanoVLA 想处理非方形相机帧时直接照抄这个模式。 / Same Conv2d, but with variable-length sequences + masking for arbitrary aspect ratios — copy this pattern when your nanoVLA needs non-square frames.

## 注意事项 / Caveats / when it breaks

- **绝对位置不能 zero-shot 换分辨率** / **Absolute positions don't zero-shot to new resolutions**: 224 训的模型,推理时给 336 直接挂(position 表只有 196 项)。常规救济:`bicubic_interpolate` 把 14×14 的 position 表插值到 21×21;更好的方案是换 2D RoPE。 / Train at 224, infer at 336 → crash (position table has only 196 entries). Workaround: bicubic-interpolate the 14×14 position grid to 21×21; better: switch to 2D RoPE.
- **Conv2d 的 `padding="valid"`** / **`padding="valid"` on Conv2d**: 边缘的不完整 patch 被直接丢弃。如果你的相机帧是 225×225,会丢掉最右一列和最底一行。要么改成 224×224,要么用 `padding="same"`(但会引入边界 artifact)。 / Edge incomplete patches are silently dropped. A 225×225 frame loses the rightmost column and bottom row. Either crop to 224 or use `padding="same"` (introduces boundary artifacts).
- **CLS 加在最前还是最后**: 这里是 `cat([cls, patches])`(原始 ViT 风格)。HF transformers 里有些模型是后置 CLS,加载权重时 `position_embedding` 切片对不上。`from_pretrained` 的 mapping 必须手写。 / Some HF models put CLS at the end. When loading weights, position embedding slicing will mismatch. The `from_pretrained` mapping has to be hand-written (see nanoVLM lines 191-247 for the full SigLIP example).
- **CLS 的零初始化** / **Zero-init of CLS**: 是合理的,因为后续 LayerNorm + attention 会立刻把它推离 0。但**位置 embedding 不能也零初始化**,否则前几个 step 网络看到的所有 patch 都是同一个 embedding 加同一个 0,梯度无法区分 patch。 / Zero-init for CLS is fine because LayerNorm + attention move it off zero immediately. But **never zero-init the position embedding** — for the first few steps every patch would receive the same zero offset, and gradients can't tell patches apart.

## 延伸阅读 / Further reading

- [Original ViT paper (Dosovitskiy et al. 2020)](https://arxiv.org/abs/2010.11929)
- [SigLIP paper (Zhai et al. 2023)](https://arxiv.org/abs/2303.15343)
- [HF transformers `SiglipVisionModel` source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py#L245)
- [NaViT — variable-resolution ViT](https://arxiv.org/abs/2307.06304)
- [nanoVLM README — full from-scratch VLA recipe](https://github.com/huggingface/nanoVLM)
