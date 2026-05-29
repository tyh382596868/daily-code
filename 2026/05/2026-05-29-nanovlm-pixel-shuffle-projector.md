---
date: 2026-05-29
topic: huggingface
source: huggingface
repo: huggingface/nanoVLM
file: models/modality_projector.py
permalink: https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/models/modality_projector.py#L1-L44
difficulty: beginner
read_time: ~8 min
tags: [code-of-the-day, huggingface, vlm, pixel-shuffle, modality-projector]
---

# nanoVLM 用 reshape 把 256 个图像 token 压成 64 个 / nanoVLM trades 256 image tokens for 64 fat tokens with one reshape

> **一句话 / In one line**: nanoVLM 在视觉塔和语言模型之间塞了一个 44 行的 ModalityProjector：先用 pixel shuffle 把 16×16 patch token 重排成 8×8 的"胖"token，再过一个 Linear 投到 LM 的 hidden_size。 / nanoVLM's whole image-to-LM bridge is a 44-line module: pixel-shuffle reshapes 16×16 vision tokens into 8×8 fatter tokens, then a single Linear projects into the LM's hidden space.

## 为什么重要 / Why this matters

把视觉 backbone 输出的 patch tokens 直接拼到 LLM 输入序列里，最大的痛点是序列长度爆炸 —— SigLIP 一张图就 256 个 token，5 张图就 1280 个 token，KV cache 直接撑爆。SmolVLM、Idefics 系列都用了一个 trick：在投影前先做"pixel shuffle"，把 `4` 个相邻 token 沿 channel 拼成 `1` 个 fat token，序列长度直接除以 4，参数量只多一个 Linear。这段 44 行代码是这个 trick 的最干净实现，恰好是你做 nanoVLA 时夹在 ViT 和 LM 之间的那一块。

The hardest part of stitching a vision tower into an LLM is sequence-length blow-up: SigLIP returns 256 patch tokens per image, five images give you 1280 tokens, and the KV cache explodes. SmolVLM/Idefics fixed this by doing a pixel shuffle before projection — pack four adjacent tokens into one fatter token along the channel axis, cutting sequence length by 4× at no learnable cost. nanoVLM ships the canonical 44-line implementation. If you are building a nanoVLA, this is exactly the block you need between a frozen ViT and a language backbone.

## 代码 / The code

`huggingface/nanoVLM` — [`models/modality_projector.py`](https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/models/modality_projector.py#L1-L44)

```python
# Modality Projection from Vision to Language
import torch.nn as nn

class ModalityProjector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_dim = cfg.vit_hidden_dim * (cfg.mp_pixel_shuffle_factor**2)
        self.output_dim = cfg.lm_hidden_dim
        self.scale_factor = cfg.mp_pixel_shuffle_factor

        self.proj = nn.Linear(self.input_dim, self.output_dim, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # https://github.com/huggingface/smollm/blob/main/vision/m4/models/vllama3/modeling_vllama3.py#L1281
    def pixel_shuffle(self, x):
        bsz, seq, embed_dim = x.size()
        seq_root = int(seq**0.5)
        assert seq_root**2 == seq # Sequence length must be a perfect square for pixel shuffle
        assert seq_root % self.scale_factor == 0 # Sequence root must be divisible by scale factor

        height = width = seq_root
        x = x.view(bsz, height, width, embed_dim)
        h_out = height // self.scale_factor
        w_out = width // self.scale_factor

        x = x.reshape(bsz, h_out, self.scale_factor, w_out, self.scale_factor, embed_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(bsz, h_out * w_out, embed_dim * self.scale_factor**2)

        return x

    def forward(self, x):
        x = self.pixel_shuffle(x)
        x = self.proj(x)

        return x
```

## 逐行讲解 / What's happening

1. **构造函数 / `__init__` (lines 5-14)**:
   - 中文：`input_dim = vit_hidden_dim * scale**2`。这是关键一行 —— 输入维度被预先放大 `scale²` 倍，因为 pixel shuffle 会把 `scale²` 个 token 沿 channel 拼起来。`proj` 就是把这个胖 token 压回到 `lm_hidden_dim`。
   - English: the projector's `input_dim` is the ViT hidden size multiplied by `scale**2`, because pixel shuffle concatenates `scale*scale` neighbouring tokens along the channel axis. `proj` then squeezes that fat token down to the LM's hidden size.

2. **`pixel_shuffle` 的形状变换 / Shape choreography in `pixel_shuffle`**:
   - 中文：输入 `[B, T, C]`，先假定 `T = H*W`（一张方形图），reshape 成 `[B, H, W, C]`。
   - English: input is `[B, T, C]` with the assumption that `T = H*W` (a square image), reshaped to `[B, H, W, C]`.
3. **核心 reshape / The core 6-D reshape (lines 35-37)**:
   - 中文：
     ```
     [B, H, W, C]
       → [B, H/s, s, W/s, s, C]      # 拆出空间块
       → [B, H/s, W/s, s, s, C]      # permute 把块内坐标 (s, s) 移到一起
       → [B, (H/s)*(W/s), s*s*C]     # 块内坐标和 C 合并 → 胖 token
     ```
   - English:
     ```
     [B, H, W, C]
       → [B, H/s, s, W/s, s, C]      # split into spatial blocks
       → [B, H/s, W/s, s, s, C]      # permute so block-local coords sit next to channels
       → [B, (H/s)*(W/s), s*s*C]     # merge block-local coords into channels → fat token
     ```
4. **最后的 Linear / Final Linear projection**:
   - 中文：`proj` 是一个 `bias=False` 的 Linear，从 `vit_hidden_dim * scale²` 投到 `lm_hidden_dim`。整个模块没有激活，没有 LayerNorm，没有别的非线性 —— 真的就是一个矩阵乘法。
   - English: `proj` is a bias-free Linear from `vit_hidden_dim * scale**2` to `lm_hidden_dim`. No activation, no LayerNorm, no residual — literally a single matmul, and that is all SmolVLM-style VLMs need.

## 类比 / The analogy

像折毛巾。你有 256 块小毛巾码在桌上（视觉 token 序列），桌子不够大塞进 LLM 的输入区。于是把毛巾 2×2 折一起变成更厚但只剩 64 块，桌面省了 4 倍，毛巾里的纤维（信息）一根都没丢，最后用一台压平机（Linear）把每块折好的毛巾压成 LLM 想要的厚度。

Think of folding towels. You have 256 small towels (vision tokens) laid out on a table, but the LLM's table only fits a fraction of that. You fold each 2×2 group of towels into one thicker towel — now there are only 64 of them, and not a single fibre (bit of information) was lost. A flatten-iron (the Linear) presses each folded towel down to the exact thickness the LLM expects.

## 自己跑一遍 / Try it yourself

```python
# pip install torch
import torch
import torch.nn as nn

class ModalityProjector(nn.Module):
    def __init__(self, vit_hidden=384, lm_hidden=512, scale=2):
        super().__init__()
        self.scale = scale
        self.proj = nn.Linear(vit_hidden * scale * scale, lm_hidden, bias=False)
    def pixel_shuffle(self, x):
        B, T, C = x.shape
        H = W = int(T ** 0.5)
        s = self.scale
        x = x.view(B, H, W, C)
        x = x.reshape(B, H // s, s, W // s, s, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.reshape(B, (H // s) * (W // s), s * s * C)
    def forward(self, x):
        return self.proj(self.pixel_shuffle(x))

proj = ModalityProjector(vit_hidden=384, lm_hidden=512, scale=2)
patches = torch.randn(2, 256, 384)   # B=2, 16x16 patch tokens, ViT hidden=384
tokens  = proj(patches)
print("input :", patches.shape)      # [2, 256, 384]
print("output:", tokens.shape)       # [2,  64, 512]
print("ratio :", patches.numel() / tokens.numel())  # ~ 1.5
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
input : torch.Size([2, 256, 384])
output: torch.Size([2, 64, 512])
ratio : 1.5
```

中文：序列长度从 256 砍到 64，但总信息量（参数数量）只下降了 1.5×，因为通道维度同时被撑胖到了 `4 * 384 → 512`。

English: sequence length drops from 256 to 64 tokens but the total information content drops by only ~1.5× — the channel dim got fatter (`4 * 384 → 512`) to soak up what would otherwise have been thrown away.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **SmolVLM / Idefics3** / **SmolVLM / Idefics3**: 中文 — 同一个 trick 的工业版本，pixel shuffle factor 通常是 3（256 patch → 28 token）。 / English — the industrial version, where `scale=3` is common and 256 patches collapse to 28 tokens.
- **PixelShuffle in super-resolution** / **PixelShuffle in super-resolution**: 中文 — 这个 reshape 本来是 ESPCN 的"sub-pixel convolution"，在 VLM 里方向反过来用：那边是放大，这里是压缩。 / English — the same reshape originated as ESPCN's sub-pixel convolution for super-resolution; VLMs run it in reverse (downscaling instead of upscaling).
- **Patch merging in Swin Transformer** / **Patch merging in Swin**: 中文 — Swin 在每个 stage 之间也做了 2×2 patch merge + Linear，思想完全一致。 / English — Swin's between-stage patch merging is exactly this `reshape + permute + Linear` recipe.

## 注意事项 / Caveats / when it breaks

- **序列长度必须是完全平方** / **Sequence length must be a perfect square**: 中文 — 代码里直接 `int(seq**0.5)` 然后 assert，CLS token 必须提前剥掉。 / English — the `assert seq_root**2 == seq` will fail if a CLS token is left in; strip it first.
- **scale 必须整除 sqrt(seq)** / **`scale` must divide `sqrt(seq)`**: 中文 — 16×16 → scale=2 或 4 OK，scale=3 直接 assert 挂。 / English — `scale=3` will not work on 16×16 patch grids; you need a tower whose grid is divisible by `scale`.
- **没有位置编码** / **No position info added**: 中文 — pixel shuffle 保留了空间相邻关系，但语言模型那一侧需要自己加 RoPE 或者位置 embedding，这里不负责。 / English — pixel shuffle preserves spatial neighbourhood, but positional encoding for the LM side is added downstream — this module is purely shape-mangling.

## 延伸阅读 / Further reading

- [nanoVLM repo (the whole training stack is < 1.5 k LoC)](https://github.com/huggingface/nanoVLM)
- [SmolVLM blog post — pixel-shuffle factor 3](https://huggingface.co/blog/smolvlm)
- [ESPCN — sub-pixel convolution origin](https://arxiv.org/abs/1609.05158)
