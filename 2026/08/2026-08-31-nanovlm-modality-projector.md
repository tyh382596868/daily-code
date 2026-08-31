---
date: 2026-08-31
topic: huggingface
source: huggingface
repo: huggingface/nanoVLM
file: models/modality_projector.py
permalink: https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/models/modality_projector.py#L4-L44
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, nanovlm, modality-projector, pixel-shuffle]
---

# nanoVLM projector：先降 token，再升通道 / nanoVLM Projector: Reduce Tokens, Expand Channels

> **一句话 / In one line**: `pixel_shuffle` 把相邻视觉 token 合并到通道维，再用线性层投到语言模型 hidden size。 / `pixel_shuffle` merges neighboring vision tokens into channels, then a linear layer projects them to the language-model hidden size.

## 为什么重要 / Why this matters

VLM 不能把 ViT 输出原样塞进 LLM：token 太多，维度也不一定对。nanoVLM 的 projector 用一个很小的模块同时解决这两个问题，先把 2x2 视觉网格压成 1 个 token，再把通道数投成 `lm_hidden_dim`。

A VLM cannot usually feed ViT outputs directly into an LLM: there are too many tokens and the hidden size may not match. nanoVLM solves both with a tiny projector: compress a 2x2 visual grid into one token, then project channels to `lm_hidden_dim`.

## 代码 / The code

`huggingface/nanoVLM` — [`models/modality_projector.py`](https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/models/modality_projector.py#L4-L44)

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

1. **第 8 行 / Line 8 (`input_dim`)**:
   - 中文: token 数会按 `factor^2` 降低，所以每个新 token 的通道数要乘同样倍数。
   - English: Token count shrinks by `factor^2`, so each new token's channel width grows by the same factor.
2. **第 24-27 行 / Lines 24-27 (square grid assumptions)**:
   - 中文: 这里假设视觉 token 能还原成正方形网格，并且边长可以被 shuffle factor 整除。
   - English: The code assumes vision tokens can be reshaped into a square grid whose side is divisible by the shuffle factor.
3. **第 34-36 行 / Lines 34-36 (reshape + permute)**:
   - 中文: 这三步把局部 2D 邻域从空间维搬到通道维，没有学习参数。
   - English: These three steps move local 2D neighborhoods from spatial axes into channels without learned parameters.
4. **第 40-42 行 / Lines 40-42 (project to LLM)**:
   - 中文: 压缩后的视觉 token 最后通过 `Linear` 进入语言模型的 embedding 空间。
   - English: The compressed visual tokens then pass through `Linear` into the language model's embedding space.

## 类比 / The analogy

像把四张小便签叠成一张厚便签。桌面上的便签数量少了，但每张便签包含的信息更多，最后再换成同一种纸张大小方便归档。

It is like stacking four small sticky notes into one thicker note. There are fewer notes on the desk, each carries more information, and a final cut makes every note fit the same filing size.

## 自己跑一遍 / Try it yourself

```python
import torch

def pixel_shuffle_tokens(x, factor=2):
    b, seq, d = x.shape
    side = int(seq**0.5)
    x = x.view(b, side, side, d)
    x = x.reshape(b, side // factor, factor, side // factor, factor, d)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.reshape(b, (side // factor) ** 2, d * factor * factor)

x = torch.arange(1 * 16 * 3).float().view(1, 16, 3)
print(pixel_shuffle_tokens(x).shape)
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

中文: 16 个 token 变成 4 个 token，但通道从 3 变成 12，信息量没有凭空消失。

English: Sixteen tokens become four, but channels grow from 3 to 12, so the local information is preserved structurally.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **SmolVLM-style projectors** / **SmolVLM-style projectors**: 中文: 很多轻量 VLM 都会先减少视觉 token，再接 LLM。 / English: Many compact VLMs reduce visual tokens before feeding the LLM.
- **GR00T Eagle projector** / **GR00T Eagle projector**: 中文: 机器人 VLA 也常用视觉压缩投影器把图像特征送进动作模型。 / English: Robotic VLAs also use visual compression projectors before action modeling.

## 注意事项 / Caveats / when it breaks

- **必须是规则网格 / It needs a regular grid**: 中文: 非方形或动态裁剪后的 token 序列要先记录真实 `H, W`。 / English: Non-square or dynamically cropped token sequences need real `H, W` metadata.
- **压缩会丢空间精度 / Compression can lose spatial precision**: 中文: 小物体控制任务可能需要更小 factor 或多尺度特征。 / English: Small-object control tasks may need a smaller factor or multiscale features.

## 延伸阅读 / Further reading

- nanoVLM source: https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/models/modality_projector.py
- Pixel shuffle idea in PyTorch: https://pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html
