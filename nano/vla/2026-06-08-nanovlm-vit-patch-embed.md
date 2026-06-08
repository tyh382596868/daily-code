---
date: 2026-06-08
topic: vla
source: vla
repo: huggingface/nanoVLM
file: models/vision_transformer.py
permalink: https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/models/vision_transformer.py#L7-L128
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, vla, vision-encoder, vit, siglip, nano-vla]
build_role: vision-encoder — the first stage of a from-scratch VLA observation pipeline
---

# 用一层 Conv2d 把图片切成 token:nanoVLM 的视觉编码器 / One Conv2d turns pixels into tokens: nanoVLM's vision encoder

> **一句话 / In one line**: 视觉编码器的精华就两件事——`Conv2d(stride=patch_size)` 把图片拍成 patch 序列,加位置编码;剩下的 ViT 块就是标准 Transformer。 / The whole vision encoder boils down to two ideas — one `Conv2d(stride=patch_size)` slams the image into a patch sequence with positional embeddings, the rest is a plain Transformer stack.

## 为什么重要 / Why this matters

VLA 的第一公里永远是同一个问题:相机给你一张 `(3, H, W)` 的 RGB 图,LM backbone 想要 `(N_patches, D)` 的 token 序列。生产级的 OpenVLA / π₀ / GR00T 直接拿一个冻结的 SigLIP 或 DINOv2 的预训练权重接上去,看上去很黑盒。nanoVLM 把这个"黑盒"拆开重写了 120 行可读 PyTorch:`Conv2d(in=3, out=embd, kernel=patch, stride=patch)` 一刀切到底,展平就是 patch 序列,再走标准 LayerNorm-Attn-MLP 残差块。这是从零搭 nanoVLA 第一个该实现的模块,它的接口就是后面所有阶段的输入契约。

The first mile of any VLA is the same problem: the camera hands you a `(3, H, W)` RGB image and the LM backbone wants `(N_patches, D)` tokens. Production OpenVLA / π₀ / GR00T just snap on a frozen pretrained SigLIP or DINOv2 and call it a day — feels like a black box. nanoVLM cracks that box open in 120 lines of readable PyTorch: `Conv2d(in=3, out=embd, kernel=patch, stride=patch)` does the chopping in one stroke, flatten gives you the patch sequence, then a vanilla LayerNorm-Attn-MLP residual stack does the rest. This is the very first module you should implement when building nanoVLA from scratch — its output shape is the input contract for every later stage.

## 代码 / The code

`huggingface/nanoVLM` — [`models/vision_transformer.py`](https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/models/vision_transformer.py#L7-L128)

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
        x = self.conv(x)                # (B, D, H/p, W/p)
        x = x.flatten(2)                # (B, D, N)
        x = x.transpose(1, 2)           # (B, N, D)
        if self.cls_flag:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        x = x + self.position_embedding
        return x


class ViTMultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.vit_n_heads
        self.embd_dim = cfg.vit_hidden_dim
        self.head_dim = self.embd_dim // self.n_heads
        self.dropout = cfg.vit_dropout
        # Combined projections for all heads
        self.qkv_proj = nn.Linear(self.embd_dim, 3 * self.embd_dim, bias=True)
        self.out_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=True)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,   # ViT attention is bidirectional
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(y))


class ViTBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.vit_hidden_dim, eps=cfg.vit_ln_eps)
        self.attn = ViTMultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.vit_hidden_dim, eps=cfg.vit_ln_eps)
        self.mlp = ViTMLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
```

## 逐行讲解 / What's happening

1. **`nn.Conv2d(3, D, kernel=p, stride=p)` 切 patch**:
   - 中文: 这是 ViT 整篇论文里最像"魔法"的一行。一个 stride=kernel 的 Conv2d,数学上等价于"把图片切成不重叠的 p×p 块,每块拉平后过一个全连接 `(3 p² → D)`"——但用 Conv2d 实现是最高效的,而且天然带 batch / 并行。
   - English: the most "magic" line in the whole ViT paper. A `stride == kernel` Conv2d is mathematically equivalent to "carve the image into non-overlapping p×p patches and run each through a `(3*p² → D)` linear" — but written as a Conv2d, it's the fastest path and trivially batched.

2. **`x.flatten(2).transpose(1, 2)` reshape 成序列**:
   - 中文: `(B, D, H/p, W/p) → (B, D, N) → (B, N, D)`。从这一行起,图像彻底变成"长度 N 的 token 序列",和 BERT / GPT 的输入一模一样。
   - English: `(B, D, H/p, W/p) → (B, D, N) → (B, N, D)`. From here on the image is a "length-N token sequence" — pin-for-pin compatible with BERT / GPT input.

3. **可选的 `cls_token`**:
   - 中文: ViT 原版从 BERT 抄来的"汇总 token"。SigLIP 这种为图像-文本对比训练的模型用 mean pool,所以 nanoVLM 默认 `cls_flag=False`。VLA 里通常也用 mean / 用所有 patch token,几乎不用 CLS。
   - English: the ViT-original "summary token", borrowed from BERT. SigLIP, trained for image-text contrast, uses mean pooling instead — so nanoVLM defaults `cls_flag=False`. VLAs also tend to use mean / use all patch tokens; CLS is rarely needed.

4. **学习的 `position_embedding`**:
   - 中文: 一个 `(1, N, D)` 的可学习张量直接 broadcast 加到 patch 上。简单粗暴,但和 SigLIP / CLIP 的做法一致。如果想支持任意分辨率,这里要换成 RoPE 或 ALiBi——这是后面跨分辨率 fine-tune 的痛点之一。
   - English: a learned `(1, N, D)` tensor broadcast-added to the patches. Crude but matches SigLIP / CLIP. If you want resolution-flexible inference you'll have to swap to RoPE or ALiBi — a known pain point in cross-resolution fine-tuning.

5. **`ViTMultiHeadAttention` 的 `qkv_proj`**:
   - 中文: 一次 `nn.Linear(D, 3D)` 同时算 Q/K/V,然后 split 成三份。比三个独立的 `nn.Linear` 省 kernel-launch 开销;权重加载时需要把 HF 的 q/k/v 三个 weight 拼起来。
   - English: a single `nn.Linear(D, 3D)` produces Q/K/V at once, then split. Saves kernel-launch overhead vs. three separate Linears; loading HF weights requires concatenating their separate q/k/v tensors.

6. **`F.scaled_dot_product_attention(..., is_causal=False)`**:
   - 中文: 视觉编码器是双向 attention——图片没有时间序。这一行直接走 PyTorch 内置 SDPA(自动选 FlashAttention),不用自己写 softmax,免去数值稳定性的坑。
   - English: the vision encoder uses bidirectional attention — there's no temporal order in an image. This line dispatches to PyTorch's built-in SDPA (which picks FlashAttention when available), so you don't write the softmax yourself and skip the numerical-stability footguns.

7. **pre-norm 残差**:
   - 中文: `x = x + attn(ln1(x))`,这是现代 ViT/LLM 的事实标准——norm 在残差前。比 post-norm 更稳定,大模型必须用。
   - English: `x = x + attn(ln1(x))`, the de facto standard for modern ViT/LLMs — norm before residual. More stable than post-norm, and required at scale.

## 类比 / The analogy

想象给一张照片做拼图。`Conv2d(stride=patch)` 就是"剪刀+尺子",咔嚓咔嚓把 224×224 的照片裁成 14×14 = 196 小块。每块 16×16 像素本来是 768 个数(16×16×3),通过 Conv 的卷积核投影成一个 384 维的"token"。位置编码就是拼图盒里印的网格坐标——告诉你这块本来该放在第 (3, 7) 格。最后 Transformer 不是在拼回原图,而是让每块在脑子里和其他所有块"商量信息"——边角的小块知道中间有只猫,中间的小块知道边角是墙。

Picture doing a jigsaw of a photo. `Conv2d(stride=patch)` is the scissors + ruler that chops a 224×224 image into 14×14 = 196 tiles. Each 16×16 tile (originally 768 raw numbers, 16×16×3) gets projected through the conv kernel into a 384-dim "token". The positional embedding is the grid label printed on each piece's back — "this one belongs at (3, 7)". The Transformer doesn't reassemble the photo; instead each piece negotiates with every other piece in token-space — corner tiles learn "there's a cat in the middle", middle tiles learn "edges are wall".

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-VLA

这是 `nano_vla` 课程里的 `vision-encoder` 组件,**无前置依赖**——可以最早开始实现。在你自己搭的 nanoVLA 里,这模块的接口契约是:
- 输入: `(B, 3, H, W)` 单相机 RGB(H=W=224 是 SigLIP 标准)
- 输出: `(B, N, D)`,N = `(H/p)²` = 196,D 通常 768

下游消费者是 `modality-projector`(2026-05-29 已经用 nanoVLM 的 pixel shuffle 实现讲过了):它把 196 个 patch token 压成 64 个"胖 token",喂给 LM backbone。再下游就是 `vlm-backbone-wiring`(2026-05-29 已经讲过 SmolVLA 的接法),然后接 `action-head` 输出动作。

多相机融合(GR00T、π₀ 的常见做法)在这里如何加?最简单是给每个相机一份独立的 `ViT` 实例,各自出一个 token 序列,在通道维 concat 后送进 projector;或共享 ViT,先把多张图沿 batch 拼,出 token 后再切回来按相机 concat。从零写 nanoVLA 时建议先单相机跑通,再加多相机。生产级实现要补的:Resolution-flexible positional embedding (RoPE 2D)、frozen + LoRA pattern、bf16/fp8 推理、image augmentation。

This is the `vision-encoder` slot of the `nano_vla` curriculum, **with no upstream dependencies** — so it's the first module to write. The contract in your own nanoVLA is:
- Input: `(B, 3, H, W)` single-camera RGB (H=W=224 is SigLIP standard)
- Output: `(B, N, D)`, N = `(H/p)²` = 196, D typically 768

Downstream consumer is `modality-projector` (covered 2026-05-29 via nanoVLM's pixel shuffle): it compresses 196 patch tokens into 64 fat tokens for the LM backbone. After that is `vlm-backbone-wiring` (covered 2026-05-29 via SmolVLA), then the action head emits actions.

How do you add multi-camera fusion (GR00T / π₀ style)? Simplest: instantiate one `ViT` per camera, get one token sequence each, concat along the token axis before the projector. Or share the ViT, batch-stack the camera images, run them, then split back and concat per camera. When writing nanoVLA from scratch, get single-camera working first then add multi-camera. Production must add: resolution-flexible positional embedding (RoPE-2D), the frozen+LoRA pattern, bf16/fp8 inference, image augmentation.

## 自己跑一遍 / Try it yourself

```python
# try.py — a 50-line nanoVLA vision encoder, no nanoVLM dependency
import torch, torch.nn as nn, torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, img=224, patch=16, dim=384):
        super().__init__()
        self.conv = nn.Conv2d(3, dim, patch, stride=patch)
        n = (img // patch) ** 2
        self.pos = nn.Parameter(torch.randn(1, n, dim) * 0.02)
    def forward(self, x):
        x = self.conv(x).flatten(2).transpose(1, 2)   # (B, N, D)
        return x + self.pos

class Block(nn.Module):
    def __init__(self, dim=384, heads=6, mlp_ratio=4):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, 3*dim), nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio*dim), nn.GELU(),
                                 nn.Linear(mlp_ratio*dim, dim))
        self.heads = heads
    def forward(self, x):
        h = self.ln1(x)
        B, N, D = h.shape
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q, k, v = [t.view(B, N, self.heads, D//self.heads).transpose(1, 2) for t in (q, k, v)]
        y = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D)
        x = x + self.proj(y)
        x = x + self.mlp(self.ln2(x))
        return x

class NanoViT(nn.Module):
    def __init__(self, depth=6):
        super().__init__()
        self.embed = PatchEmbed()
        self.blocks = nn.ModuleList(Block() for _ in range(depth))
        self.norm = nn.LayerNorm(384)
    def forward(self, x):
        x = self.embed(x)
        for blk in self.blocks: x = blk(x)
        return self.norm(x)

img = torch.randn(2, 3, 224, 224)
tokens = NanoViT()(img)
print(f"image  : {tuple(img.shape)}")
print(f"tokens : {tuple(tokens.shape)}  # (B, 196, 384), ready for the projector")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
image  : (2, 3, 224, 224)
tokens : (2, 196, 384)  # (B, 196, 384), ready for the projector
```

注意 `(196, 384)` 这两个数:196 是 `(224/16)² = 14²`,384 是 head_dim 64 × heads 6。从这一刻起,你的图片就是一个序列,后面所有 LM 工具都能直接用。

Note the two numbers `(196, 384)`: 196 = `(224/16)² = 14²`, 384 = head_dim 64 × heads 6. From this moment on the image *is* a sequence, and every downstream LM tool just works on it.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **SigLIP / SigLIP-2 (Google)**: 生产级 VLA 真正会用的预训练权重;同一个 PatchEmbed + ViT 架构,差别在训练目标(sigmoid 对比) / The pretrained weights production VLAs actually load; same PatchEmbed + ViT skeleton, only the training objective (sigmoid contrastive) differs.
- **OpenVLA 的 fused encoder**: 把 SigLIP + DINOv2 两个 patch encoder 并联,token 维 concat 给 LM —— DINOv2 的纹理 + SigLIP 的语义双重视角 / OpenVLA parallels SigLIP + DINOv2 PatchEmbeds and concats along the channel axis — DINOv2's textures + SigLIP's semantics, two views.
- **DINOv3 (今天的 tracked)**: 一样的 ViT 骨架,仅在训练 loss 上加 Gram loss;参考今天的 tracked 笔记 / Same ViT skeleton, only adds Gram loss during training; see today's tracked note.
- **GR00T 的 Eagle vision tower / GR00T's Eagle vision tower**: 在 SigLIP 基础上换成 Eagle 多分辨率特征金字塔 / Swaps SigLIP for Eagle's multi-resolution feature pyramid on top of the same base.

## 注意事项 / Caveats / when it breaks

- **位置编码不是分辨率无关的** / **Positional embedding is resolution-locked**: `position_embedding` 的形状是 `(1, N, D)`,N 由训练分辨率写死。换分辨率推理要双线性插值或重训 PE / `position_embedding`'s shape locks N to the training resolution. To infer at a different size you must bilinearly interpolate or retrain the PE.
- **patch_size 选得太大丢细节** / **Big patches lose detail**: 32 像素一个 patch 在大屏画面里粗糙得不能识别小物体;机器人场景常用 14 或 16 / 32-px patches are too coarse to spot small objects on wide screens; robotics usually picks 14 or 16.
- **HF 权重的 QKV 是分开的** / **HF weights store Q/K/V separately**: 想加载 SigLIP 预训练权重,记得把 q_proj/k_proj/v_proj 三个 tensor concat 成单个 qkv_proj——nanoVLM 的 `from_pretrained` 已经为你做了 / To load pretrained SigLIP you must concat the three q/k/v tensors into one qkv tensor — nanoVLM's `from_pretrained` already handles this.
- **Conv2d 的 padding="valid"** / **`padding="valid"` is required**: 改成 `"same"` 就不是 patch 切分而是滑窗了,token 数不对 / Switch to `"same"` and it becomes a sliding-window conv with N off by one or more — broken.

## 延伸阅读 / Further reading

- ViT paper: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020) — the original PatchEmbed idea
- SigLIP paper: "Sigmoid Loss for Language Image Pre-Training" (Zhai et al., 2023)
- nanoVLM repo README — the curriculum-friendly companion to this code
- OpenVLA paper § 3.2 — fused SigLIP + DINOv2 vision tower
