---
date: 2026-07-17
topic: wam
source: wam
repo: facebookresearch/DiT
file: models.py
permalink: https://github.com/facebookresearch/DiT/blob/main/models.py#L230-L239
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, wam, dit, output-head, unpatchify]
build_role: output-head advanced variant
---

# DiT latent 输出：预测 patch，再还原成 VAE latent / DiT Latent Output: Predict Patches, Then Restore a VAE Latent

> **一句话 / In one line**: DiT 的输出头让每个 token 预测一个 latent patch，`unpatchify` 再把这些 patch 还原成可交给 VAE/scheduler 的网格。 / DiT's output head makes each token predict a latent patch, then `unpatchify` restores the grid consumed by the VAE or scheduler.

## 为什么重要 / Why this matters

nanoWAM 的主干可以完全在 token 空间里做注意力，但最后必须回到视频 latent。这个组件定义了 output head 的最小合同：上游是 DiT token，下游是 `(C, H, W)` 或视频版 `(C, T, H, W)` latent。没有这层，模型只能预测“token”，不能参与扩散采样。

A nanoWAM backbone can run attention entirely in token space, but the end must return to video latents. This component defines the minimal output-head contract: upstream DiT tokens, downstream image/video latents. Without it, the model predicts tokens but cannot join a diffusion sampler.

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

1. **token 宽度包含 patch 像素 / Token width contains patch pixels**: 中文: 每个 token 的最后一维是 `p*p*C`。 English: each token's last dimension is `p*p*C`.
2. **patch 网格先恢复 / Restore patch grid first**: 中文: token 序列被拆成 `h x w` 网格。 English: the token sequence becomes an `h x w` grid.
3. **patch 内坐标再恢复 / Restore in-patch coordinates**: 中文: `p, p` 两维表示小块内部位置。 English: the two `p` axes represent positions inside each patch.
4. **通道提前 / Move channels forward**: 中文: `einsum` 把 `C` 放到 PyTorch 图像张量的通道位置。 English: `einsum` moves `C` into the PyTorch channel position.
5. **最终给 scheduler / Final output goes to scheduler**: 中文: 采样器只认 dense latent，不认 token list。 English: samplers consume dense latents, not token lists.

## 类比 / The analogy

这是 WAM 的“出片”步骤：前面所有注意力都在剪辑台上操作片段编号，最后要把每个片段渲染回一张连续画面。

This is the WAM's render-out step: attention edits numbered pieces, then the output layer renders them back into a continuous frame.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

中文: 这是 `output-head` 的高级变体。二维 DiT 用 `h*w` token；视频 WAM 会多一个时间维，通常需要把 `T*H*W` token reshape 成 `(T, H, W, p_t, p_h, p_w, C)` 或采用只在空间 patchify 的简化版。

English: This is an advanced `output-head` variant. A 2D DiT uses `h*w` tokens; a video WAM adds time and typically reshapes `T*H*W` tokens into a temporal-spatial patch grid, or uses a simpler spatial-only patching contract.

## 自己跑一遍 / Try it yourself

```python
import numpy as np

tokens = np.arange(16).reshape(1, 4, 4)  # 4 tokens, each predicts 2x2x1
x = tokens.reshape(1, 2, 2, 2, 2, 1)
latent = np.einsum("nhwpqc->nchpwq", x).reshape(1, 1, 4, 4)
print(latent.shape)
print(latent[0, 0, :2, :2].tolist())
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
(1, 1, 4, 4)
[[0, 1], [2, 3]]
```

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Wan2.1 patchify / unpatchify**: 视频 latent 也要在 token 和网格之间来回切换。 / Video latents also move between token and grid forms.
- **MAE decoder**: patch reconstruction 使用同一类轴重排。 / Patch reconstruction uses the same axis logic.

## 注意事项 / Caveats / when it breaks

- **视频维度要显式建模 / Time must be explicit for video**: 不能把所有 token 当方形图片。 / Do not treat all video tokens as one square image.
- **output channels 要和 loss 对齐 / Output channels must match the loss**: `learn_sigma`、velocity、epsilon 目标会改变通道语义。 / `learn_sigma`, velocity, and epsilon targets change channel meaning.
- **reshape bug 很隐蔽 / Reshape bugs are subtle**: 数量对了也可能轴顺序错。 / Element counts can match while axes are wrong.

## 延伸阅读 / Further reading

- [DiT `models.py`](https://github.com/facebookresearch/DiT/blob/main/models.py)
- [Wan2.1 repository](https://github.com/Wan-Video/Wan2.1)
