---
date: 2026-10-23
topic: huggingface
source: huggingface
repo: huggingface/nanoVLM
file: models/modality_projector.py
permalink: https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/models/modality_projector.py#L4-L44
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, multimodal]
---

# nanoVLM pixel shuffle：四个视觉 token 合成一个语言 token / nanoVLM Pixel Shuffle: Four Vision Tokens Become One Language Token

> **一句话 / In one line**: 先按空间邻域合并视觉 token，再线性投影到语言模型宽度。 / Spatially merge neighbouring vision tokens first, then linearly project them to the language-model width.

## 为什么重要 / Why this matters

视觉 encoder 的 token 往往比文本多；pixel shuffle 将 token 数缩小 `scale_factor²`，同时把局部特征拼到通道维。

Vision encoders often produce many more tokens than text. Pixel shuffle reduces token count by `scale_factor²` while moving local detail into channels.

## 代码 / The code

`huggingface/nanoVLM` — [`modality_projector.py`](https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/models/modality_projector.py#L23-L44)

```python
def pixel_shuffle(self, x):
    bsz, seq, embed_dim = x.size()
    seq_root = int(seq**0.5)
    assert seq_root**2 == seq
    x = x.view(bsz, seq_root, seq_root, embed_dim)
    x = x.reshape(bsz, seq_root // self.scale_factor, self.scale_factor,
                  seq_root // self.scale_factor, self.scale_factor, embed_dim)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.reshape(bsz, -1, embed_dim * self.scale_factor**2)
```

## 逐行讲解 / What's happening

断言确保序列仍能还原成方格。`reshape → permute → reshape` 先把每个局部方格相邻摆放，再压到一个更宽的 token。

The assertion ensures the sequence can become a square grid. `reshape → permute → reshape` places each local tile together, then packs it into a wider token.

## 类比 / The analogy

像把 2×2 马赛克砖粘成一块大砖：砖少了，但每块砖携带四格的颜色。

It is like gluing a 2×2 mosaic into one tile: fewer tiles, each carrying four colours.

## 自己跑一遍 / Try it yourself

```python
import torch
x = torch.arange(16.).view(1, 4, 4)
y = x.view(1,2,2,4).reshape(1,1,2,1,2,4).permute(0,1,3,2,4,5).reshape(1,1,16)
print(y.shape)
```

运行 / Run with: `pip install torch && python try.py`

预期输出 / Expected output: `torch.Size([1, 1, 16])`。

## 注意事项 / Caveats / when it breaks

- 这个实现要求 token 网格为正方形且边长可整除。 / This implementation requires a square token grid whose side divides evenly.

## 延伸阅读 / Further reading

- [nanoVLM projector source](https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/models/modality_projector.py)
