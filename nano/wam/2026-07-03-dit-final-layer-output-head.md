---
date: 2026-07-03
topic: wam
source: wam
repo: facebookresearch/DiT
file: models.py
permalink: https://github.com/facebookresearch/DiT/blob/main/models.py#L125-L141
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, wam, output-head, adaln]
build_role: output head variant for a from-scratch WAM denoiser
---

# DiT FinalLayer：最后一步也要吃条件 / DiT FinalLayer: The Output Head Is Conditioned Too

> **一句话 / In one line**: `FinalLayer` 在预测 patch 像素/latent 前，再用条件向量调制一次归一化后的 token。 / `FinalLayer` modulates normalized tokens one more time with the conditioning vector before predicting patch pixels or latents.

## 为什么重要 / Why this matters

WAM 的 denoiser 最终要把 token 还原成视频 latent、动作 latent 或二者的速度场。很多实现会注意中间 block 的条件注入，却忘了输出头也需要知道 timestep。DiT 的 `FinalLayer` 是最小模板：LayerNorm、条件 shift/scale、Linear。

A WAM denoiser ultimately maps tokens back to video latents, action latents, or their velocity fields. It is easy to focus on conditioning inside the middle blocks and forget that the output head also needs the timestep. DiT's `FinalLayer` is the minimal template: LayerNorm, conditional shift/scale, Linear.

## 代码 / The code

`facebookresearch/DiT` — [`models.py`](https://github.com/facebookresearch/DiT/blob/main/models.py#L125-L141)

```python
class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
```

## 逐行讲解 / What's happening

1. **输出宽度按 patch 展开 / Output width expands by patch**: 中文: 每个 token 输出 `patch_size * patch_size * out_channels`，后面才能 `unpatchify`。 / English: Each token predicts `patch_size * patch_size * out_channels`, which can later be unpatchified.
2. **最后的 LayerNorm 也无 affine / Final LayerNorm has no affine**: 中文: 输出头的 shift/scale 仍然来自条件，而不是固定参数。 / English: The output head's shift and scale still come from conditioning, not fixed affine parameters.
3. **只需要两份控制量 / Only two controls are needed**: 中文: 输出头没有残差分支，所以只要 shift 和 scale，不需要 gate。 / English: The output head has no residual branch, so it only needs shift and scale, not gates.

## 类比 / The analogy

像打印照片前的最后一次调色：前面已经完成构图，但送进打印机之前还要根据纸张和光照做一次白平衡。

It is like the last color correction before printing a photo. The composition is done, but the image still needs white balance for the paper and lighting before it goes to the printer.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

这是 `dit-block` / output-head 的交界组件。上游是若干个已经完成视频/动作 token 交互的 DiT block；下游是 `unpatchify`、VAE decode 或动作反归一化。nanoWAM 可以先实现一个共享输出头；生产版通常会为 video latent 和 action head 分开输出维度和 loss。

This sits at the boundary between the `dit-block` stack and the output head. Upstream are DiT blocks where video and action tokens have interacted; downstream are `unpatchify`, VAE decode, or action unnormalization. In nanoWAM, start with one shared output head; a production model often separates output dimensions and losses for video latents and actions.

## 自己跑一遍 / Try it yourself

```python
import torch
import torch.nn as nn

B, tokens, hidden, patch, channels = 2, 4, 8, 2, 3
x = torch.randn(B, tokens, hidden)
c = torch.randn(B, hidden)
norm = nn.LayerNorm(hidden, elementwise_affine=False)
ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))
linear = nn.Linear(hidden, patch * patch * channels)
shift, scale = ada(c).chunk(2, dim=1)
y = linear(norm(x) * (1 + scale[:, None]) + shift[:, None])
print(y.shape)
```

运行 / Run with:

```bash
pip install torch
python try.py
```

预期输出 / Expected output:

```text
torch.Size([2, 4, 12])
```

每个 token 输出一个 `2 x 2 x 3` patch；这就是从 Transformer token 回到空间 latent 的接口。

Each token emits a `2 x 2 x 3` patch. That is the interface from Transformer tokens back to spatial latents.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Wan2.1 的输出层** / **Wan2.1 output layers**: 视频 DiT 同样要把 token 投回 VAE latent 空间。
- **Open-Sora MMDiT** / **Open-Sora MMDiT**: 多流 block 之后也需要一个明确的 image/video 输出头。

## 注意事项 / Caveats / when it breaks

- **patch 维度必须和 unpatchify 对齐** / **Patch dimensions must match unpatchify**: 输出通道错一位，后面 reshape 会直接失败。
- **动作 token 不一定是二维 patch** / **Action tokens are not always 2D patches**: WAM 加动作头时，动作输出通常需要单独的 Linear 头。

## 延伸阅读 / Further reading

- [DiT models.py](https://github.com/facebookresearch/DiT/blob/main/models.py)
- [Wan2.1 repository](https://github.com/Wan-Video/Wan2.1)
