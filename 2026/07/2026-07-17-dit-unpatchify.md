---
date: 2026-07-17
topic: diffusion
source: tracked
repo: facebookresearch/DiT
file: models.py
permalink: https://github.com/facebookresearch/DiT/blob/main/models.py#L230-L239
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, dit, patchify, latent]
---

# DiT unpatchify：把 token 棋盘折回 latent 图 / DiT unpatchify: Fold the Token Board Back into a Latent Image

> **一句话 / In one line**: DiT 的最后一步不是卷积，而是把每个 token 预测的 `p x p x C` 小块重新排列成 `(N, C, H, W)` latent。 / DiT's last step is not a convolution; it rearranges each token's predicted `p x p x C` patch back into an `(N, C, H, W)` latent.

## 为什么重要 / Why this matters

DiT 主干在 token 空间工作，但扩散 scheduler 和 VAE 需要图像/latent 网格。`unpatchify` 是这两个世界的接口：Transformer 输出的是一串 patch token，采样循环需要的是可继续加噪、去噪、解码的 feature map。

The DiT backbone works in token space, while the scheduler and VAE expect an image-like latent grid. `unpatchify` is the adapter between those worlds: the Transformer emits patch tokens, but the denoising loop needs a feature map.

## 代码 / The code

`facebookresearch/DiT` — [`models.py`](https://github.com/facebookresearch/DiT/blob/main/models.py#L230-L239)

```python
def unpatchify(self, x):
    c = self.out_channels
    p = self.x_embedder.patch_size[0]
    h = w = int(x.shape[1] ** 0.5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
    x = torch.einsum("nhwpqc->nchpwq", x)
    imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
    return imgs
```

## 逐行讲解 / What's happening

1. **取出 patch 大小 / Read patch size**: 中文: `p` 决定每个 token 要展开成多大的小格子。 English: `p` tells how large a spatial tile each token owns.
2. **token 数必须是平方数 / Tokens must form a square**: 中文: `h = w = sqrt(T)`，所以普通 DiT 假设方形 latent。 English: `h = w = sqrt(T)`, so this DiT assumes a square latent grid.
3. **先拆 token 通道 / Split token channels**: 中文: `(N, T, p*p*C)` 变成 `(N, h, w, p, p, C)`。 English: `(N, T, p*p*C)` becomes `(N, h, w, p, p, C)`.
4. **`einsum` 只换轴 / `einsum` only reorders axes**: 中文: 没有学习参数，只把 patch 内坐标和 patch 网格坐标交错。 English: no learned parameters, just interleaving patch coordinates with grid coordinates.
5. **回到 latent map / Return to latent map**: 中文: 最终形状是 scheduler/VAE 熟悉的 `(N, C, H, W)`。 English: the final shape is the scheduler/VAE-friendly `(N, C, H, W)`.

## 类比 / The analogy

像把一盒拼图块放回画框：Transformer 给你的是按格子编号的一堆小块，`unpatchify` 只负责把每块转正、放回对应位置。

It is like putting puzzle tiles back into the frame: the Transformer gives numbered tiles, and `unpatchify` rotates them into the right axes and places them back.

## 自己跑一遍 / Try it yourself

```python
import numpy as np

n, h, w, p, c = 1, 2, 2, 2, 1
x = np.arange(n * h * w * p * p * c).reshape(n, h * w, p * p * c)
x = x.reshape(n, h, w, p, p, c)
img = np.einsum("nhwpqc->nchpwq", x).reshape(n, c, h * p, w * p)
print(img[0, 0].tolist())
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]
```

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **MAE / ViT reconstruction**: patch tokens 也要折回图像空间算重建 loss。 / Patch tokens are folded back to image space for reconstruction loss.
- **Video DiT**: 3D 版本会把 token 还原成 `(T, H, W)` latent tube。 / 3D variants restore tokens into a `(T, H, W)` latent tube.

## 注意事项 / Caveats / when it breaks

- **非方形 latent 不适用 / Non-square latents need a different shape contract**: 这里直接设 `h = w`。 / This implementation directly sets `h = w`.
- **patch size 要和 embedder 对齐 / Patch size must match the embedder**: 前后不一致会 reshape 失败。 / A mismatch breaks the reshape.
- **`learn_sigma=True` 会翻倍输出通道 / `learn_sigma=True` doubles output channels**: 后面通常只对 noise 通道做 CFG。 / Later CFG often applies only to selected noise channels.

## 延伸阅读 / Further reading

- [DiT `models.py`](https://github.com/facebookresearch/DiT/blob/main/models.py)
- [DiT paper repository](https://github.com/facebookresearch/DiT)
