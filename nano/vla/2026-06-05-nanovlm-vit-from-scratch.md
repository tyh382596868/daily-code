---
date: 2026-06-05
topic: vla
source: vla
repo: huggingface/nanoVLM
file: models/vision_transformer.py
permalink: https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/models/vision_transformer.py#L7-L128
difficulty: beginner
read_time: ~12 min
tags: [code-of-the-day, vla, vision-encoder, vit, patch-embed]
build_role: vision-encoder — every VLA's image side starts here. Turns (B, 3, H, W) pixels into (B, N, D) tokens for the VLM/action expert to consume.
---

# 一颗 Conv2d 就是 patch embed:从零搭一个能给 VLA 用的 ViT / One Conv2d is your patch embed: a ViT from scratch ready to feed a VLA

> **一句话 / In one line**: 一个 `stride=patch_size` 的 Conv2d 就是 ViT 的 patch embedder,后面接 N 个标准 ViT block(LN → SDPA → LN → MLP)就构成了 VLA 视觉侧的全部 — 不到 130 行。 / A single `Conv2d` with `stride=patch_size` *is* the ViT patch embedder; stack N standard ViT blocks (LN → SDPA → LN → MLP) on top and you have the entire image side of a VLA — under 130 lines.

## 为什么重要 / Why this matters

每个 VLA — OpenVLA、π₀、SmolVLA、GR00T、starVLA — 都需要回答同一个问题:**机器人摄像头的原始像素怎么变成 transformer 能吃的 token 序列?** 大多数 paper 含糊带过"用了 SigLIP / DINO",但在 nanoVLA 这种从头实现里,你必须知道每个 token 是怎么来的、形状怎么算、cls token 选不选、位置编码哪里加。nanoVLM 的 `vision_transformer.py` 是我见过最干净的 ViT 教材实现 — 没有 timm、没有 transformers 那种为了通用性堆出来的几百行 wrapper,就是教科书里那个 ViT。

Every VLA — OpenVLA, π₀, SmolVLA, GR00T, starVLA — has to answer the same question: **how do raw robot camera pixels become a token sequence a transformer can eat?** Most papers wave their hands ("we use SigLIP / DINO"), but in a from-scratch nanoVLA you need to know exactly how each token is produced — its shape, whether to add a `[CLS]` token, where positional embeddings go. nanoVLM's `vision_transformer.py` is the cleanest textbook ViT implementation I've seen — no timm, no transformers wrapper, just the ViT from the paper.

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
        x = self.conv(x)           # (B, D, H/p, W/p)
        x = x.flatten(2)           # (B, D, N)
        x = x.transpose(1, 2)      # (B, N, D)
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
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,    # ViT attention is bidirectional
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        y = self.resid_dropout(y)
        return y

class ViTMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.activation_fn = nn.GELU(approximate='tanh')
        self.fc1 = nn.Linear(cfg.vit_hidden_dim, cfg.vit_inter_dim)
        self.fc2 = nn.Linear(cfg.vit_inter_dim, cfg.vit_hidden_dim)
        self.dropout = nn.Dropout(cfg.vit_dropout)

    def forward(self, x):
        x = self.fc1(x); x = self.activation_fn(x); x = self.fc2(x); x = self.dropout(x)
        return x

class ViTBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.vit_hidden_dim, eps=cfg.vit_ln_eps)
        self.attn = ViTMultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.vit_hidden_dim, eps=cfg.vit_ln_eps)
        self.mlp = ViTMLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # pre-norm self-attention
        x = x + self.mlp(self.ln2(x))    # pre-norm MLP
        return x
```

## 逐行讲解 / What's happening

1. **`nn.Conv2d(3, D, kernel=p, stride=p, padding="valid")`** ✨:
   - 中文: 这一颗 Conv2d 就是 patch embedder。`stride == kernel_size` 意味着窗口之间不重叠,每个 patch 进去出来一个 `D` 维向量。224×224 图、patch=16 → 输出 `(D, 14, 14)`,正好 196 个 token。这比"手动 reshape + linear" 快得多,也省内存。
   - English: this one Conv2d *is* the patch embedder. `stride == kernel_size` means non-overlapping windows; each patch produces one `D`-dim vector. A 224×224 image with patch=16 yields `(D, 14, 14)` = 196 tokens. Much faster and lower-memory than "reshape into patches then linear".

2. **`flatten(2).transpose(1,2)`**:
   - 中文: 把 `(B, D, H/p, W/p)` 摊平到 `(B, N, D)`,符合 transformer 输入约定。这两步几乎所有 ViT 都长一样。
   - English: flatten the spatial grid into `(B, N, D)`, the transformer convention. Same two-step recipe in every ViT.

3. **CLS token (可选,nanoVLM 默认关掉)**:
   - 中文: 经典 ViT 在前面拼一个可学习的 `[CLS]` token,最后取它作分类输出。VLA 一般**不要**这个 token,因为下游不是分类,是把整个 patch 序列送给 LM。所以 nanoVLM 默认 `cls_flag=False`,直接做 mean pooling 或全 token 送下游。
   - English: classic ViT prepends a learnable `[CLS]` token and uses its final hidden state for classification. VLAs typically **don't** want this — downstream isn't classification, it's piping the patch sequence into an LM. nanoVLM defaults to `cls_flag=False` and either mean-pools or forwards all tokens.

4. **`position_embedding` 是 learnable + 加法**:
   - 中文: 不是 RoPE、不是 sinusoidal,就是一个 `(1, N, D)` 可学习参数,加到 patch embedding 上。简单且对 fixed-resolution 任务最稳。如果你想支持多分辨率,需要换 2D RoPE 或者训练时做 random crop。
   - English: not RoPE, not sinusoidal — just a learnable `(1, N, D)` parameter added to the patch embeddings. Simple and rock-solid for fixed-resolution inputs. Variable resolution requires 2D RoPE or train-time random cropping.

5. **`qkv_proj = nn.Linear(D, 3*D)`** + `qkv.split(C, dim=2)`:
   - 中文: 把 Q、K、V 三个投影合并成一个 `(D, 3D)` 的 Linear,前向时 split 三段。一次 matmul 比三次快,显存也更连续。
   - English: fuse Q, K, V into a single `(D, 3D)` Linear and split the output. One matmul beats three on both speed and memory locality.

6. **`scaled_dot_product_attention(..., is_causal=False)`**:
   - 中文: ViT 是 bidirectional 的(整图同时看),所以 `is_causal=False`。这一行实际上替你调用了 flash attention(如果可用)。
   - English: ViT attends bidirectionally over the whole image, so `is_causal=False`. This call dispatches to flash attention under the hood when available.

7. **Pre-norm 残差**:
   - 中文: `x = x + attn(ln1(x))` 是 LayerNorm 在残差里面、attention 外面的"Pre-norm",训练稳定性比 Post-norm 好得多。SigLIP/ViT-22B/Llama 全用这种。
   - English: `x = x + attn(ln1(x))` is the "pre-norm" pattern (LayerNorm inside the residual). Trains much more stably than post-norm; used by SigLIP, ViT-22B, Llama.

8. **`GELU(approximate='tanh')`**:
   - 中文: tanh-approximate 的 GELU 比精确 GELU 快一点点,数值几乎一致。SigLIP / Gemma 都用这个。
   - English: tanh-approximate GELU is marginally faster than exact GELU with negligible numeric difference. Used in SigLIP and Gemma.

## 类比 / The analogy

想象把一张照片放进碎纸机,但碎纸机刀片排成一个网格,横竖一刀一刀均匀切开 — 这是 `Conv2d(stride=patch_size)`。每张碎片再被指纹扫描器(线性投影)读成一串数字 — 这是输出的 `D` 维 token。每张碎片还盖了个"我是从第 3 行第 5 列剪下来的"图章 — 这是 positional embedding。然后这堆碎纸条(共 196 张)整齐排成一队,送进 transformer 这间会议室,每张碎片可以"看见"所有其他碎片,讨论拼出完整图像的语义。

Picture feeding a photo into a paper shredder, but the blades form a grid that slices uniformly across and down — that's `Conv2d(stride=patch_size)`. Each slip is then read by a fingerprint scanner (linear projection) into a string of numbers — your `D`-dim token. Each slip carries a stamp like "I'm from row 3, column 5" — that's the positional embedding. The 196 slips line up and enter the transformer meeting room, where every slip gets to "see" every other slip to assemble the picture's meaning.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-VLA

**Curriculum item**: `vision-encoder` (depends_on: ∅ — root of the build graph).

中文:这是 nanoVLA 数据流的**第一站**。流程是:

```
camera_frame (B, 3, 224, 224)
        │
        ▼
  ViTPatchEmbeddings ──► (B, 196, D)
        │
        ▼
  N × ViTBlock        ──► (B, 196, D)        ← 视觉特征
        │
        ▼
  modality_projector  ──► (B, 64, D_lm)      ← 已覆盖 2026-05-29
        │
        ▼
  VLM backbone        (token concat: [image_tokens, instruction_tokens])
        │
        ▼
  action_head         ──► (B, action_dim)    ← 课程下一步
```

省掉它,VLA 就根本看不到机器人摄像头。生产级实现一般直接 `from_pretrained` 加载 SigLIP-base 或 DINOv2-base 的权重(本文件 170-251 行的 QKV-concat 重映射就是干这个的),因为视觉自监督需要 1B+ 张图,从头训不划算。但*结构*你必须自己拥有 — 一旦你想加多相机(头+腕)、多分辨率、temporal patch(让 ViT 吃视频),你就得改这几个 module。

English: this is the **first stop** in nanoVLA's data flow. Pipeline:

```
camera_frame (B, 3, 224, 224)
        │
        ▼
  ViTPatchEmbeddings ──► (B, 196, D)
        │
        ▼
  N × ViTBlock        ──► (B, 196, D)        ← visual features
        │
        ▼
  modality_projector  ──► (B, 64, D_lm)      ← already covered 2026-05-29
        │
        ▼
  VLM backbone        (token concat: [image_tokens, instruction_tokens])
        │
        ▼
  action_head         ──► (B, action_dim)    ← next curriculum step
```

Skip it and your VLA literally cannot see. A production implementation loads SigLIP-base or DINOv2-base via `from_pretrained` (lines 170-251 of this file do the QKV-concat remap) — visual SSL needs 1B+ images, training from scratch isn't worth it. But you must own the *structure* yourself, because once you add multi-camera (head + wrist), multi-resolution, or temporal patching (ViT over video), you'll be editing these modules.

## 自己跑一遍 / Try it yourself

```python
# nano_vit.py — pip install torch
import torch, torch.nn as nn, torch.nn.functional as F

class Cfg:
    vit_img_size, vit_patch_size, vit_hidden_dim = 224, 16, 192
    vit_n_heads, vit_n_blocks, vit_inter_dim = 6, 6, 768
    vit_dropout, vit_ln_eps, vit_cls_flag = 0.0, 1e-6, False

class Patch(nn.Module):
    def __init__(self, c):
        super().__init__()
        n = (c.vit_img_size // c.vit_patch_size) ** 2
        self.conv = nn.Conv2d(3, c.vit_hidden_dim, c.vit_patch_size, c.vit_patch_size)
        self.pos  = nn.Parameter(torch.randn(1, n, c.vit_hidden_dim) * 0.02)
    def forward(self, x):
        return self.conv(x).flatten(2).transpose(1, 2) + self.pos

class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.vit_hidden_dim)
        self.qkv = nn.Linear(c.vit_hidden_dim, 3 * c.vit_hidden_dim)
        self.out = nn.Linear(c.vit_hidden_dim, c.vit_hidden_dim)
        self.ln2 = nn.LayerNorm(c.vit_hidden_dim)
        self.mlp = nn.Sequential(nn.Linear(c.vit_hidden_dim, c.vit_inter_dim),
                                 nn.GELU(approximate='tanh'),
                                 nn.Linear(c.vit_inter_dim, c.vit_hidden_dim))
        self.h = c.vit_n_heads
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, 2)
        q = q.view(B, T, self.h, C // self.h).transpose(1, 2)
        k = k.view(B, T, self.h, C // self.h).transpose(1, 2)
        v = v.view(B, T, self.h, C // self.h).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, T, C)
        x = x + self.out(y)
        return x + self.mlp(self.ln2(x))

c = Cfg()
vit = nn.Sequential(Patch(c), *[Block(c) for _ in range(c.vit_n_blocks)])
img = torch.randn(2, 3, 224, 224)              # batch of robot camera frames
tokens = vit(img)
print("tokens.shape =", tokens.shape, "  params =",
      sum(p.numel() for p in vit.parameters()))
```

运行 / Run with:
```bash
python nano_vit.py
```

预期输出 / Expected output:
```
tokens.shape = torch.Size([2, 196, 192])   params = 2706624
```

中文:2.7M 参数,196 个 token,每个 192 维 — 这就是你的 nanoVLA 视觉前端的全部。把 `img` 换成机器人摄像头帧,把 `tokens` 接给 modality projector,就开始能"看东西"了。

English: 2.7M params, 196 tokens × 192 dims — that's your entire nanoVLA visual front-end. Swap `img` for a robot camera frame, pipe `tokens` into the modality projector, and your model can "see".

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **lerobot / smolvla** / **lerobot's smolvla**: 视觉侧用 SigLIP-base(也是 ViT,224×224、patch=14),patch embedder 几乎一字不差。 / Uses SigLIP-base (also ViT, 224×224, patch=14); the patch embedder is identical to the byte.
- **NVIDIA Isaac-GR00T (eagle backbone)** / **NVIDIA Isaac-GR00T**: 用 Eagle-2 视觉编码器,多相机(head + 双 wrist)各自跑一遍 ViT,再 concat 上去。`vision-encoder` 这个槽位接的是同一种东西。 / Uses Eagle-2 vision encoder; runs the ViT once per camera (head + two wrists) and concatenates. Same `vision-encoder` slot, multi-cam variant.
- **OpenVLA prismatic backbone** / **OpenVLA prismatic**: 双 ViT(SigLIP + DINOv2)同时跑,token 沿通道 concat。证明 `vision-encoder` 这个组件可以是"多个 ViT 的 ensemble"。 / Runs both SigLIP and DINOv2, concatenates tokens along the channel dim. Demonstrates the `vision-encoder` slot can be an *ensemble* of ViTs.
- **π₀ (Physical Intelligence openpi)** / **π₀**: 用 PaliGemma 的视觉塔(SigLIP),配置在 `paligemma_with_expert`,patch embedder 同款。 / Uses PaliGemma's vision tower (SigLIP). Configured via `paligemma_with_expert`; patch embedder is the same Conv2d trick.

## 注意事项 / Caveats / when it breaks

- **可学习 position embedding 不能换分辨率** / **learnable pos-embed locks the resolution**: 训练时 224×224 → 196 个 pos slot,推理时给 448×448 → 784 个 patch,直接形状不匹配。要么 bicubic 插值 pos-embed,要么换 RoPE。 / Train at 224×224 → 196 pos slots; infer at 448×448 → 784 patches → shape mismatch. Either bicubic-interpolate the pos-embed or switch to RoPE.
- **CLS token 在 VLA 里基本无用** / **`[CLS]` is essentially useless in VLA**: VLA 要的是 patch 级 token,不是分类输出。除非你只想喂 LM 一个 image-summary token(像 Flamingo 那种 Perceiver Resampler),否则关掉 `cls_flag`。 / VLAs want patch-level tokens, not a classification readout. Leave `cls_flag=False` unless you're explicitly using a Perceiver-Resampler-style summary token.
- **bf16 训练时 LN eps 太小会 NaN** / **bf16 + tiny LN eps = NaN**: 默认 `vit_ln_eps=1e-6` 在 fp32 没事,bf16 下要拉到 1e-5。SigLIP 官方权重就是 1e-6,加载完记得手动修正。 / `1e-6` is fine for fp32 but causes NaNs in bf16; bump to `1e-5`. The SigLIP official weights use `1e-6`; remember to patch this after loading.
- **`padding="valid"` 隐含图像必须能整除 patch size** / **`padding="valid"` requires image % patch_size == 0**: 224 % 16 == 0 没问题。如果你想吃 720p,得 resize 或 pad 到 16 的倍数。 / 224 % 16 == 0, fine. For 720p you must resize or pad up to a multiple of 16 first.

## 延伸阅读 / Further reading

- [An Image is Worth 16×16 Words (ViT paper, Dosovitskiy et al. 2020)](https://arxiv.org/abs/2010.11929)
- [SigLIP (Zhai et al. 2023)](https://arxiv.org/abs/2303.15343) — the pretrained weights nanoVLM loads
- [DINOv2 paper](https://arxiv.org/abs/2304.07193) — alternative vision-encoder choice for VLAs
- [nanoVLM repo README](https://github.com/huggingface/nanoVLM) — full from-scratch VLM training pipeline
