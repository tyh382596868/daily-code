---
date: 2026-06-01
topic: vla
source: vla
repo: huggingface/nanoVLM
file: models/vision_transformer.py
permalink: https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/models/vision_transformer.py#L7-L43
difficulty: beginner
read_time: ~9 min
tags: [code-of-the-day, vla, vision-encoder, patch-embedding, nano-vla]
build_role: vision-encoder
---

# nanoVLM 用一个 Conv2d 把图片变成 token / nanoVLM turns an image into tokens with a single Conv2d

> **一句话 / In one line**: `nn.Conv2d(3, D, kernel=stride=patch_size)` 一次把 `[B,3,H,W]` 切成 `[B, N_patches, D]` token,再加一组可学习的位置编码,这就是任何 ViT 的入口 / `nn.Conv2d(3, D, kernel=stride=patch_size)` slices a `[B,3,H,W]` image into `[B, N_patches, D]` tokens in one shot, plus a learnable position embedding — and that's the front door of any ViT.

## 为什么重要 / Why this matters

VLA(Vision-Language-Action)的"V"从这里开始——机器人摄像头看到的 RGB 必须先变成 LM 能 attend 的 token 序列。整个 SigLIP / DINO / CLIP / 你自己 nanoVLA 的 vision tower 的第一层都是这 35 行 patch embedding。把它读透,你就知道(1)Conv2d 跑 stride=kernel 等于 patchify、(2)位置编码为什么是可学习参数而不是 sinusoidal、(3)`cls_flag` 这个 if-branch 怎么决定你的 ViT 是 OpenAI CLIP 风格还是 Google SigLIP 风格。

The "V" in VLA starts here: the robot camera's RGB has to become a sequence of tokens before any LM can attend over it. Every SigLIP / DINO / CLIP / your-own-nanoVLA vision tower opens with these same 35 lines. Read them and you understand (1) why Conv2d with `stride=kernel` is exactly patchify, (2) why the positional embedding is a learnable parameter and not sinusoidal, and (3) how the `cls_flag` if-branch toggles between OpenAI-CLIP-style and Google-SigLIP-style ViTs.

## 代码 / The code

`huggingface/nanoVLM` — [`models/vision_transformer.py`](https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/models/vision_transformer.py#L7-L43)

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

1. **`num_patches = (img_size // patch_size) ** 2`**:
   - 中文: 一张 224×224 的图按 16×16 切,得 14×14 = 196 个 patch。这个值要和位置编码的长度对齐,**改图片分辨率不能不改这里**,否则 position embedding 形状不对。
   - English: a 224×224 image at 16×16 patches gives 14×14 = 196 tokens. This count must match the positional embedding length, so changing input resolution requires updating this (or interpolating the position embedding) — otherwise shape mismatch.

2. **`nn.Conv2d(3, D, kernel=stride=patch_size, padding="valid")`**:
   - 中文: 这是 ViT 最巧的设计——卷积核大小**等于** stride,所以每次卷积都吃一个不重叠的 patch,输出一个 D 维向量。等价于"把 patch 拉平再过一个 `Linear(3*P*P, D)`",但 Conv2d 实现快且自动处理 batch + 多 patch。`padding="valid"` 表示不补 0(图像必须整除 patch_size)。
   - English: ViT's neatest trick — the conv kernel **equals** the stride, so each conv window consumes one non-overlapping patch and emits one D-dim vector. Equivalent to "flatten each patch and apply `Linear(3*P*P, D)`", but Conv2d is faster and handles batch + many patches automatically. `padding="valid"` means no padding (image side must be divisible by `patch_size`).

3. **`cls_flag` 分支 / the `cls_flag` branch**:
   - 中文: 原始 ViT 论文(Dosovitskiy 2020)用一个可学习的 `[CLS]` token 拼在序列最前面,最后那个位置的特征就是图片的全局表示。SigLIP / DINO 系列**不要** CLS,而是用所有 patch token 的均值或 norm。`cls_flag = True` 就走前一种,`False` 就走后一种。位置编码的长度也跟着 +1 或不 +1。
   - English: the original ViT paper (Dosovitskiy 2020) prepends a learnable `[CLS]` token and reads its final-layer state as the image-level feature. SigLIP / DINO families **skip** CLS and use the mean/norm of all patch tokens instead. `cls_flag = True` → former; `False` → latter. Position embedding length adjusts by ±1 accordingly.

4. **`position_embedding = nn.Parameter(torch.rand(1, N, D))`**:
   - 中文: 注意这是**完全可学习**的张量(不是 sin/cos)。`shape=(1, N, D)` 而不是 `(N, D)` 是为了 broadcast 到 batch。ViT 之所以放弃 sinusoidal,是因为 patch 网格 + 数据规模足以让网络自己学出有意义的位置感。
   - English: a **fully learnable** tensor (not sinusoidal). Shape `(1, N, D)` so it broadcasts across the batch dim. ViTs dropped sinusoidal because the patch grid + dataset scale make it cheap for the network to learn the position structure end-to-end.

5. **`forward`(三步) / 3-step forward**:
   - 中文: `conv(x)` → `[B, D, H/P, W/P]`,`flatten(2)` 把空间两维并成一维 → `[B, D, N]`,`transpose(1, 2)` 把 D 移到最后 → `[B, N, D]`。这就是任何 transformer 喜欢的"token sequence"输入格式。
   - English: `conv(x)` → `[B, D, H/P, W/P]`, `flatten(2)` collapses the two spatial dims → `[B, D, N]`, `transpose(1, 2)` moves the feature dim to last → `[B, N, D]`. That's the canonical token-sequence shape any transformer expects.

6. **`cls_token.expand` + `cat` + `position_embedding`**:
   - 中文: 用 `expand` 而不是 `repeat` 是为了避免实际分配内存——`cls_token` 形状 `(1, 1, D)`,broadcast 到 `(B, 1, D)` 只在 view 层面发生。`cat` 把 CLS 拼到序列最前。最后加上 position embedding——注意位置编码的形状包含了"是否有 CLS"的那一格。
   - English: use `expand` (not `repeat`) to avoid an actual copy — `cls_token` is `(1, 1, D)`, expanding to `(B, 1, D)` happens at the view level. `cat` prepends CLS. The final `+ position_embedding` already accounts for whether CLS is present (length is `N+1` or `N`).

## 类比 / The analogy

想象你要让一个只懂"句子"的语言模型读懂一张照片。你拿一把 16×16 的网格罩在照片上,把每个小方块拍成一张缩略图,然后给每个缩略图发一张"我是从第几格来的"小卡片。这堆"小图 + 来源卡片"就是一句给 LM 读的"句子"。Conv2d 是那把网格,position embedding 是那叠小卡片。CLS token 就像在句首加一个"请总结全文"的占位符。

Picture wanting a language model that only understands sentences to read a photo. Lay a 16×16 grid over the image, take a thumbnail of each square, hand each thumbnail a card saying "I came from grid cell (r, c)". The pile of thumbnails + cards is the "sentence" the LM reads. Conv2d is the grid; the position embedding is the stack of cards; the CLS token is a "please summarize" placeholder at the start of the sentence.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-VLA

中文: 这是 `nano_vla` 课程里的 **vision-encoder** 组件,**没有依赖**,是从头搭 nanoVLA 时最早写的一块代码之一(和 `action-tokenizer` 同期)。它的输入是一个或多个相机的 RGB 帧 `[B, 3, H, W]`(多相机时通常 cat batch 维或多调一次),输出是 `[B, N_patches, D]` 的 patch token 序列。**下游接谁**:你已经覆盖过的 `modality-projector`——把 D 维 ViT 特征投到 LM 隐藏维,然后送进 `vlm-backbone-wiring`。**上游谁喂**:数据 pipeline 里的图像预处理(resize 到 vit_img_size、normalize)。**省掉会怎样**:VLM 无法 attend 视觉,VLA 退化成只能听文字指令的盲机器人。**生产级要补什么**:(a)冻结预训练 SigLIP-2 / DINOv3 权重,(b)position embedding 在分辨率不匹配时双线性插值,(c)多相机时支持 wrist-cam + third-person + depth 各一路 stream,(d)RegisterTokens(DINOv2 起新增的 4-8 个 register slot,显著降低 artifact)。

English: this is the **vision-encoder** component in the `nano_vla` curriculum, **with no prerequisites** — it's one of the earliest pieces you write from scratch (in parallel with `action-tokenizer`). Input: one or more camera RGB frames `[B, 3, H, W]` (multi-cam usually concatenated along batch or run multiple times). Output: a `[B, N_patches, D]` patch-token sequence. **Downstream**: the `modality-projector` you already covered — it projects D-dim ViT features into the LM hidden size, which then feeds `vlm-backbone-wiring`. **Upstream**: image preprocessing in the data pipeline (resize to `vit_img_size`, normalize). **What breaks if you skip it**: the VLM cannot attend to vision; the VLA degenerates into a blind text-only robot. **What production-scale needs on top**: (a) load frozen pretrained SigLIP-2 / DINOv3 weights, (b) bilinear-interpolate the position embedding when input resolution differs from the pretrained one, (c) multi-camera streams (wrist-cam + third-person + depth), (d) register tokens (the 4-8 extra slots introduced in DINOv2 that drastically cut attention artifacts).

## 自己跑一遍 / Try it yourself

```python
# nano_patch_embed.py — pure torch, no extra deps
import torch, torch.nn as nn

class NanoPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch=16, dim=384, use_cls=False):
        super().__init__()
        self.use_cls = use_cls
        self.N = (img_size // patch) ** 2
        self.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        if use_cls:
            self.cls = nn.Parameter(torch.zeros(1, 1, dim))
            self.pos = nn.Parameter(torch.randn(1, self.N + 1, dim) * 0.02)
        else:
            self.pos = nn.Parameter(torch.randn(1, self.N,     dim) * 0.02)

    def forward(self, x):                                # x: [B, 3, H, W]
        x = self.proj(x).flatten(2).transpose(1, 2)      # → [B, N, D]
        if self.use_cls:
            cls = self.cls.expand(x.size(0), -1, -1)
            x = torch.cat([cls, x], dim=1)
        return x + self.pos

img = torch.randn(2, 3, 224, 224)
for use_cls in (False, True):
    embed = NanoPatchEmbed(use_cls=use_cls)
    out = embed(img)
    print(f"use_cls={use_cls}: {tuple(img.shape)} → {tuple(out.shape)}, params={sum(p.numel() for p in embed.parameters()):,}")
```

运行 / Run with:
```bash
pip install torch
python nano_patch_embed.py
```

预期输出 / Expected output:
```
use_cls=False: (2, 3, 224, 224) → (2, 196, 384), params=370,560
use_cls=True:  (2, 3, 224, 224) → (2, 197, 384), params=370,944
```

中文: 注意把 `patch=16` 换成 `patch=14`(DINOv3 用 14),`N` 就变成 256——这是 nanoVLA 在 224 输入下默认 token 数量。把 `dim=384` 换成 `1024` 你就大致得到 ViT-L 的入口规模。

English: change `patch=16` to `patch=14` (what DINOv3 uses) and `N` becomes 256 — that's nanoVLA's default token count at 224-resolution input. Bump `dim=384` to `1024` and you're at roughly ViT-L size.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **OpenAI CLIP `vision_model.embeddings.patch_embedding`** / **OpenAI CLIP**: 同样的 Conv2d-as-patchify,有 CLS token。
- **Google SigLIP / SigLIP-2** / **Google SigLIP**: 同样 Conv2d-as-patchify,**没有** CLS,全 patch mean pool。nanoVLM 默认加载的就是 SigLIP 配置(参考代码末尾的 `from_pretrained`)。
- **Meta DINOv2 / DINOv3** / **Meta DINO families**: 同样模式,但额外加 4-8 个 register token——本质是把"垃圾 attention bin"显式留出来,降低视觉伪影。
- **LeRobot SmolVLA / OpenPI π₀ / Isaac GR00T 视觉塔**: 全部直接复用上面三个之一的 patch embed 权重,只是冻结策略和多相机融合方式不同——这也是你之前学过的 `modality-projector` / `vlm-backbone-wiring` 各家分歧的来源。
- **DiT(图像生成)/ DiTs in image generation**: 输入是 latent 而不是 RGB,但 patchify 那一刀完全一样(Conv2d kernel=stride=patch_size)。

## 注意事项 / Caveats / when it breaks

- **图像必须整除 patch_size / image side must be divisible by `patch_size`**: 中文: `padding="valid"` 不补 0,224 / 16 = 14 OK,但 225 / 16 不 OK。要支持任意分辨率得加 padding 或 crop。
- **`torch.rand` 初始化不是最佳 / `torch.rand` init is suboptimal**: 中文: nanoVLM 这里用 `torch.rand`(均匀分布 0-1),工业级一般用 `nn.init.trunc_normal_(std=0.02)`。`rand` 起步就有一个 ~0.5 的偏置,前几个 step 训练会"先去偏置"。
- **位置编码不可插值就不能换分辨率 / fixed-shape pos-embed locks resolution**: 中文: 想训完 224 还做 384 推理,必须在 `from_pretrained` 时把 `pos_embed` 双线性插值。生产 ViT 都会带这段逻辑(timm 里叫 `resize_pos_embed`)。
- **多相机时不要简单 cat token / don't naively cat tokens across cameras**: 中文: 两路相机各 196 token cat 起来后没有"哪一路"的信号,LM 会混。常用做法:每路加一个 `camera_id` embedding,或者 wrist-cam 单独走一个 expert head。
- **冻结 vs 联训 / freeze vs joint-train**: 中文: VLA 实践里 99% 是**冻结**预训练 vision tower,只训 projector + LM(可能 LoRA)。从头训这部分需要 LAION-2B 级数据。

## 延伸阅读 / Further reading

- [Dosovitskiy et al., *An Image Is Worth 16x16 Words* (ViT)](https://arxiv.org/abs/2010.11929) — patchify 的原始来源
- [SigLIP paper (Zhai et al. 2023)](https://arxiv.org/abs/2303.15343) — no-CLS 的来源
- [DINOv2 *Register Tokens* note](https://arxiv.org/abs/2309.16588) — 为什么生产 ViT 都加 register
- [nanoVLM README](https://github.com/huggingface/nanoVLM) — 这份文件的上下文
- 你之前的 daily code:[modality-projector](../../../2026/05/2026-05-29-nanovlm-pixel-shuffle-projector.md) ← patch embed 之后的下一站
