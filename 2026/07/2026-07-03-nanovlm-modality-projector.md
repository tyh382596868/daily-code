---
date: 2026-07-03
topic: huggingface
source: huggingface
repo: huggingface/nanoVLM
file: models/modality_projector.py
permalink: https://github.com/huggingface/nanoVLM/blob/main/models/modality_projector.py#L4-L44
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, huggingface, nanoVLM, modality-projector]
---

# nanoVLM projector：用 pixel shuffle 少传视觉 token / nanoVLM Projector: Use Pixel Shuffle to Send Fewer Vision Tokens

> **一句话 / In one line**: `ModalityProjector` 先把相邻 patch 的通道拼厚，再用一个 Linear 投到语言模型维度。 / `ModalityProjector` first packs neighboring patches into wider channels, then uses one Linear layer to project into the language-model width.

## 为什么重要 / Why this matters

VLM 的瓶颈常常不是视觉编码器，而是“把多少视觉 token 塞进 LLM”。nanoVLM 这段代码用 pixel shuffle 把空间 token 数减少，同时把每个 token 的特征维度变厚，再交给投影层。

The bottleneck in a VLM is often not the vision encoder, but how many vision tokens are injected into the LLM. nanoVLM uses pixel shuffle to reduce the number of spatial tokens while widening each token before projection.

## 代码 / The code

`huggingface/nanoVLM` — [`models/modality_projector.py`](https://github.com/huggingface/nanoVLM/blob/main/models/modality_projector.py#L4-L44)

```python
class ModalityProjector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_dim = cfg.vit_hidden_dim * (cfg.mp_pixel_shuffle_factor**2)
        self.output_dim = cfg.lm_hidden_dim
        self.scale_factor = cfg.mp_pixel_shuffle_factor

        self.proj = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.apply(self._init_weights)

    def pixel_shuffle(self, x):
        bsz, seq, embed_dim = x.size()
        seq_root = int(seq**0.5)
        assert seq_root**2 == seq
        assert seq_root % self.scale_factor == 0

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

1. **输入维度变厚 / Input width gets thicker**: 中文: `vit_hidden_dim * factor**2` 表示每个输出 token 合并一个 `factor x factor` patch 小块。 / English: `vit_hidden_dim * factor**2` means each output token merges a `factor x factor` patch neighborhood.
2. **要求正方形 token 网格 / Requires a square token grid**: 中文: `seq_root**2 == seq` 保证 token 能还原成 H x W。 / English: `seq_root**2 == seq` ensures the token sequence can be viewed as an H x W grid.
3. **先 reshape 再 permute / Reshape then permute**: 中文: 代码把局部空间维搬到通道维，token 数下降，通道数上升。 / English: The code moves local spatial axes into the channel axis, reducing token count and increasing channel width.

## 类比 / The analogy

像把四张小便利贴叠成一张厚卡片：桌面上卡片数量少了，但每张卡片带的信息更多。

It is like stacking four small sticky notes into one thicker card. There are fewer cards on the desk, but each card carries more information.

## 自己跑一遍 / Try it yourself

```python
import torch

B, H, W, D, f = 1, 4, 4, 3, 2
x = torch.arange(B * H * W * D).view(B, H * W, D)
x = x.view(B, H, W, D)
x = x.reshape(B, H // f, f, W // f, f, D)
x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
x = x.reshape(B, (H // f) * (W // f), D * f * f)
print(x.shape)
```

运行 / Run with:

```bash
pip install torch
python try.py
```

预期输出 / Expected output:

```text
torch.Size([1, 4, 12])
```

16 个视觉 token 被压成 4 个 token；每个 token 从 3 维变成 12 维。

Sixteen vision tokens become four tokens; each token widens from 3 dimensions to 12.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **SmolVLM / SmolVLA projectors** / **SmolVLM / SmolVLA projectors**: 常用类似的 token 压缩来控制 LLM 上下文成本。
- **图像超分的 pixel shuffle** / **Image super-resolution pixel shuffle**: 方向相反，但核心都是在空间维和通道维之间搬信息。

## 注意事项 / Caveats / when it breaks

- **必须是正方形序列** / **The sequence must be square**: 带 CLS token 或非方形 crop 时要先处理 token 布局。
- **压缩会损失细粒度定位** / **Compression can lose fine localization**: token 少了，精细空间关系更依赖通道表达。

## 延伸阅读 / Further reading

- [nanoVLM repository](https://github.com/huggingface/nanoVLM)
- [SmolVLM paper and model family](https://huggingface.co/blog/smolvlm)
