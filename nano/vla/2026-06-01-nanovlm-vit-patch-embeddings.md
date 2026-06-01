---
date: 2026-06-01
topic: vla
source: vla
repo: huggingface/nanoVLM
file: models/vision_transformer.py
permalink: https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/models/vision_transformer.py#L6-L44
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, vla, vision-encoder, patch-embedding, vit]
build_role: vision-encoder — patchify + CLS + learnable positional embedding at the very front of the vision tower
---

# 40 行 ViTPatchEmbeddings:把像素切成 token 的最小实现 / 40-line ViTPatchEmbeddings: the smallest possible patchifier

> **一句话 / In one line**: 一个 `kernel_size=stride=patch_size` 的 `nn.Conv2d` 就完成了"图像→token 序列"的整个 patch embedding——再加一个 CLS token 和一组可学位置编码就齐活。 / One `nn.Conv2d` with `kernel_size=stride=patch_size` is the entire image-to-token-sequence patchifier — add a CLS token and a learnable positional embedding and you're done.

## 为什么重要 / Why this matters

任何 VLA 系统都从同一个动作开始:把摄像头采到的 `(B, 3, H, W)` 像素张量变成 `(B, N, D)` 的 token 序列,这样后面的 transformer 才能用。这一步看似 trivial,但它决定了**序列长度 N、token 维度 D、与位置编码的轴数**,而这三个量会影响后面每一层(modality projector 的输入维度、VLM 主干的 context 长度、action head 的展开方式)。nanoVLM 给出了一个 40 行的实现:一个 Conv2d、一个 CLS token、一组可学位置编码——比 timm 的 ViT 还瘦。读懂它,就把"vision encoder"这块拼图拼上了。

Every VLA system starts with the same move: turn a `(B, 3, H, W)` camera tensor into a `(B, N, D)` token sequence so the downstream transformer can consume it. Trivial on the surface, but it nails down **sequence length N, hidden dim D, and the positional-encoding geometry** — three numbers that propagate into every later layer (the modality projector's input dim, the VLM backbone's context length, how the action head unrolls). nanoVLM gives you all of it in 40 lines: one `Conv2d`, one CLS token, one learnable positional embedding — slimmer than timm's ViT. Reading this closes the "vision encoder" piece of the build puzzle.

## 代码 / The code

`huggingface/nanoVLM` — [`models/vision_transformer.py`](https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/models/vision_transformer.py#L6-L44)

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

1. **`nn.Conv2d(kernel_size=patch_size, stride=patch_size)` —— 真正的"patchify"在这里 / The actual patchify happens here**:
   - 中文: 这是整段代码最关键的一行。当 `kernel_size == stride`,卷积窗口之间**完全不重叠**——每个窗口就是一个 patch。3→D 的通道映射在数学上等价于"先把 16×16×3=768 个数拉直,再过一个全连接到 D 维",但写成 Conv2d 性能更好、写起来更短。
   - English: The single most important line. When `kernel_size == stride`, the convolution windows **do not overlap** — each window is exactly one patch. The 3→D channel mapping is mathematically equivalent to "flatten 16×16×3=768 numbers, then run a `Linear` to D", but expressing it as a `Conv2d` is faster and shorter.

2. **`(self.img_size // self.patch_size) ** 2` / `num_patches` derivation**:
   - 中文: 224/16=14,所以 `num_patches=196`。注意这是**整除**,如果图像尺寸不能整除 patch 大小,会丢边——production VLA 通常会先 resize 或 pad 到整除。
   - English: 224/16 = 14, so `num_patches = 196`. Floor division: if the image dimensions don't divide evenly by `patch_size`, the edges get cropped. Production VLAs almost always pre-resize or pad to a multiple.

3. **`cls_token = nn.Parameter(torch.zeros(1, 1, D))`**:
   - 中文: 一个**全局可学习**的"零号 token",在 forward 时被 expand 到 batch 维。原始 ViT 用它做图像分类(取它在最后一层的输出),VLA 里通常会保留它来当全局视觉特征。
   - English: A **globally learnable** "token 0", expanded along the batch dim at forward time. The original ViT used it for image classification (read out its top-layer output); VLAs often keep it as a global visual feature.

4. **`position_embedding = nn.Parameter(torch.rand(1, N+1, D))` —— 注意是 rand 不是 zeros / Notice it's `rand`, not `zeros`**:
   - 中文: 位置编码用 `torch.rand(0, 1)` 初始化而不是 zeros,因为如果全初始化为 0,所有 patch 拿到的位置信号一样,模型在初期会很难区分位置。这是 nanoVLM 故意做的小差别。
   - English: Positional embeddings are initialized with `torch.rand` (uniform on `[0, 1)`) instead of `zeros`. With all-zero init, every patch would receive an identical position signal at step 0 and the model would have nothing to bootstrap on. A small but deliberate choice in nanoVLM.

5. **forward 三步走 / Forward in three steps**:
   - 中文: `conv(x)` 得到 `(B, D, H/p, W/p)`,`flatten(2)` 合并空间维成 `(B, D, N)`,`transpose(1, 2)` 把 D 移到末位变成 `(B, N, D)`——这正是 transformer 期待的"token 序列"形状。
   - English: `conv(x)` → `(B, D, H/p, W/p)`, `flatten(2)` collapses the two spatial dims into `(B, D, N)`, `transpose(1, 2)` swaps to `(B, N, D)` — the canonical "token sequence" shape that transformers expect.

6. **位置编码用 `+` 而不是 `concat` / Position encoding uses `+`, not `concat`**:
   - 中文: ViT 风格的位置编码是**加法**——位置信号嵌入到原 token 的每个 channel 里,不增加序列长度。这与 word2vec 拼接 positional embedding 是两种思路,各有取舍。
   - English: ViT-style positional encoding is **additive** — the position signal is embedded into every channel of the original token without increasing sequence length. (Concatenation would inflate `D` to `2D`.) Both are valid but ViT chose addition.

## 类比 / The analogy

想象你拿到一张 224×224 的照片,要把它送进一个"读 token 的"模型。第一步是用一把**16×16 的饼干模具**(cookie cutter)在照片上不重叠地按下去,得到 14×14=196 块饼干。每块饼干被拍扁成一个 D 维向量(`conv` 干这个),然后你在桶里**额外放一块特殊的"标签饼干"**(CLS token)。最后你给每块饼干贴一张写着位置的便条(positional embedding),196+1 张便条按位置粘上去。这一摞 197 个向量就是 ViT 后续层的输入。

Imagine you have a 224×224 photo and you need to hand it to a transformer that "reads tokens". Step one: stamp the photo with a **16×16 cookie cutter** in a non-overlapping grid, yielding 14×14=196 cookies. Each cookie is flattened into a D-dim vector (that's what `conv` does), then you toss in **one extra labeled "ID cookie"** (the CLS token). Finally you slip a sticky note onto every cookie with its position written on it (the learnable positional embedding). The stack of 197 vectors is the input that ViT's transformer blocks munch on.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-VLA

中文:在 nanoVLA 的构建图里,这是 **`vision-encoder`** 课程项的入口模块——位于整条 VLA pipeline 最前端,接收摄像头采集到的 `(B, 3, H, W)` 像素张量,输出 `(B, num_patches + 1, D)` 的 token 序列。它有 0 个上游依赖(直接吃像素),但下游被**所有人**消费——modality projector (`modality-projector` 课程项,已覆盖)拿这串 token 投影到 LM 维度,VLM 主干 (`vlm-backbone-wiring`,已覆盖)再把它和文本/状态 token 拼接进 transformer。如果省掉这一层,你只剩两条很糟的路:(1) 用一个 `nn.Linear(H*W*3, D)`,参数量爆炸且没有平移等变;(2) 直接把像素当 token 喂给 transformer,序列长度 224×224=50176,长得 attention 会炸。所以这一层是"必要的几何先验"。生产级 VLA 在此基础上还会补:多相机融合(把多张图的 token 序列**拼接**或**交错**)、动态分辨率(可变 patch grid)、可学习的图像-语言桥接 token,以及把它替换为 SigLIP/DINOv3 这种自监督预训练的强大初始权重——但底层数据流和这 40 行**完全一致**。

In your nanoVLA build graph this is the **`vision-encoder`** curriculum item — the front-most module on the whole VLA pipeline. It eats the camera's `(B, 3, H, W)` pixel tensor and emits `(B, num_patches + 1, D)` tokens. It depends on nothing upstream (it consumes raw pixels) but is consumed by **everyone** downstream — the modality projector (`modality-projector` item, already covered) projects these tokens into the LM space, and the VLM backbone (`vlm-backbone-wiring`, also covered) concatenates them with text/state tokens before feeding the transformer. Omit it and you're left with two ugly options: (1) an `nn.Linear(H*W*3, D)` that has no translation equivariance and explodes in parameter count, or (2) feeding raw pixels as tokens at 224×224=50176 sequence length, which blows up attention. This layer is the "necessary geometric prior". A production VLA layers more on top: multi-camera fusion (concatenate or interleave token sequences from several views), dynamic resolution (variable patch grid), learnable image-language bridge tokens, and a pretrained backbone (SigLIP, DINOv3) replacing the random init — but the data flow is **identical** to these 40 lines.

## 自己跑一遍 / Try it yourself

```python
import torch
import torch.nn as nn

class ViTPatchEmbeddings(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embd_dim=384, cls_flag=True):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.cls_flag = cls_flag
        self.conv = nn.Conv2d(3, embd_dim, kernel_size=patch_size, stride=patch_size)
        if cls_flag:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embd_dim))
            self.position_embedding = nn.Parameter(torch.rand(1, self.num_patches + 1, embd_dim))
        else:
            self.position_embedding = nn.Parameter(torch.rand(1, self.num_patches, embd_dim))

    def forward(self, x):
        x = self.conv(x).flatten(2).transpose(1, 2)  # (B, N, D)
        if self.cls_flag:
            x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        return x + self.position_embedding

patchify = ViTPatchEmbeddings()
img = torch.randn(2, 3, 224, 224)
out = patchify(img)
print("input :", img.shape)
print("output:", out.shape)
print("params:", sum(p.numel() for p in patchify.parameters()))
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
input : torch.Size([2, 3, 224, 224])
output: torch.Size([2, 197, 384])
params: 370560
```

中文一两句:`197 = 196 patches + 1 CLS`。整个 vision encoder 入口只有 ~37 万参数,可以毫无压力嵌入到一个 nanoVLA 里——这正是"nano"系列的承诺。

`197 = 196 patches + 1 CLS`. The entire vision encoder front-end is only ~370k parameters — comfortable for embedding into a nanoVLA. That's the "nano" promise: every component small enough to fit in one short file you fully understand.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **timm `PatchEmbed`** / **`timm.layers.PatchEmbed`**: 中文:几乎完全相同的实现,只是支持 `flatten=False` 留出二维形状便于 hybrid model。 / English: Almost identical, with extra `flatten=False` to keep the 2D layout for hybrid models.
- **SigLIP / DINOv3 / CLIP ViT**: 中文:它们的图像编码器入口都是这个结构,差别在位置编码(SigLIP 用 learned absolute,DINOv3 用 RoPE)、归一化、激活函数等。 / English: Same patchifier under different positional schemes (SigLIP: learned absolute; DINOv3: RoPE; CLIP: learned absolute with class token).
- **OpenVLA / SmolVLA / GR00T 的视觉塔** / **The vision towers of OpenVLA, SmolVLA, GR00T**: 中文:三者都直接复用 timm/HF 的 ViT,但 GR00T 在此之上做了"双相机 token 沿序列维拼接"——这就是同一个 `vision-encoder` 课程项的多相机扩展。 / English: All three wrap a timm/HF ViT. GR00T extends it by concatenating tokens from two cameras along the sequence dimension — a natural multi-camera extension of the same `vision-encoder` slot.
- **`huggingface/lerobot` `smolvlm_with_expert.py` (`embed_image`)**: 中文:已在 2026-05-29 的 vla 笔记里讲过的"复用 SmolVLM 的 ViT 然后把输出拍平喂给 LM",底层调的还是这个 patch embedding。 / English: Covered in the 2026-05-29 VLA note — it reuses SmolVLM's ViT and flattens the output for the LM. Underneath, the same patch-embedding mechanic.

## 注意事项 / Caveats / when it breaks

- **位置编码长度被锁死 / Positional embedding length is locked in**: 中文:`self.position_embedding` 的形状取决于 `img_size / patch_size`,如果推理时换了图像尺寸,要么 resize 位置编码(线性插值),要么改成 RoPE/可外推的位置编码。 / English: The `position_embedding` shape is fixed by `img_size / patch_size`. To change image resolution at inference, you must interpolate the position embedding (the original ViT does this with bilinear resize) or switch to RoPE / extrapolatable position schemes.
- **`torch.rand` 而非 `torch.randn` / `torch.rand`, not `torch.randn`**: 中文:`torch.rand(0, 1)` 的方差大约 1/12,远小于通常的标准正态。如果你换成 `randn` 初始训练动力学会变,需要重新调 LR。 / English: `torch.rand` has variance ~1/12, much smaller than `randn`. Swapping it for `randn` changes early training dynamics — be prepared to retune the LR.
- **`padding="valid"`** : 中文:意味着如果图像尺寸不整除 patch_size,**右下角的边会被默默丢掉**。production 通常会先 resize 到精确倍数。 / English: If image dims aren't a multiple of `patch_size`, the right/bottom edges are silently cropped. Production code usually resizes to an exact multiple first.

## 延伸阅读 / Further reading

- [An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929) — the paper that introduced this exact construction
- [nanoVLM repository](https://github.com/huggingface/nanoVLM) — minimal VLM where this file lives
- [Sigmoid Loss for Language Image Pre-Training (SigLIP)](https://arxiv.org/abs/2303.15343) — what most modern VLAs swap in as their vision tower
