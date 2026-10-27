---
date: 2026-10-27
topic: diffusion
source: tracked
repo: Wan-Video/Wan2.1
file: wan/modules/vae.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vae.py#L17-L36
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, causal-convolution]
---

# Wan2.1 因果卷积：时间只向过去借帧 / Wan2.1 Causal Convolution: Borrow Frames Only from the Past

> **一句话 / In one line**: 这个 `Conv3d` 把时间 padding 全放在左侧，因此第 t 帧不会偷看未来。 / This `Conv3d` puts temporal padding on the left, so frame t cannot peek ahead.

## 为什么重要 / Why this matters

视频 VAE 若在时间轴两侧补零，编码第 t 帧会读到未来帧，流式生成时就会失效。Wan 把空间 padding 保持对称，却把时间 padding 改成“过去多、未来零”。

A video VAE with symmetric temporal padding leaks future frames and cannot be streamed faithfully. Wan keeps spatial padding symmetric while making time strictly past-facing.

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/modules/vae.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vae.py#L17-L36)

```python
class CausalConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            x = torch.cat([cache_x.to(x.device), x], dim=2)
            padding[4] -= cache_x.shape[2]
        return super().forward(F.pad(x, padding))
```

## 逐行讲解 / What's happening

`F.pad` 的六元组依次覆盖宽、高、时间；时间部分是 `(2 * p, 0)`。分块推理时，`cache_x` 先拼到左边，再扣减仍需补的历史零。

The six-value `F.pad` tuple covers width, height, then time; its temporal pair is `(2 * p, 0)`. Chunked inference prepends cached history and reduces only the missing left padding.

## 类比 / The analogy

像直播字幕员只能引用已经说出口的话，不能引用下一句台词。

It is a live captioner: they may quote what has been said, never the next line.

## 自己跑一遍 / Try it yourself

```python
import torch
import torch.nn.functional as F
x = torch.arange(3.).view(1, 1, 3, 1, 1)
print(F.pad(x, (0, 0, 0, 0, 2, 0)).flatten().tolist())
```

运行 / Run with `pip install torch && python try.py`。输出 / Expected output: `[0.0, 0.0, 0.0, 1.0, 2.0]`。

## 注意事项 / Caveats / when it breaks

- **缓存长度 / Cache length**：缓存必须恰好覆盖卷积感受野需要的过去帧；太短会改变块边界结果。 / The cache must cover the required temporal receptive field or chunk boundaries change the result.
