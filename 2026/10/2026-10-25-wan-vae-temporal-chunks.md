---
date: 2026-10-25
topic: diffusion
source: tracked
repo: Wan-Video/Wan2.1
file: wan/modules/vae.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vae.py#L516-L568
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, video-vae, temporal-cache]
---

# Wan VAE 时间分块：首帧先走，后续四帧接力 / Wan VAE Temporal Chunks: Prime One Frame, Then Relay Four

> **一句话 / In one line**: Wan 的视频 VAE 用首帧加连续四帧的分块协议，在保持因果卷积缓存正确的同时编码长视频。 / Wan's video VAE encodes long clips as one priming frame followed by four-frame chunks so causal-convolution caches remain correct.

## 为什么重要 / Why this matters

视频 VAE 不能像图片模型那样随意把时间维切开：卷积在边界处需要过去的特征。这里把第一帧作为启动块，之后每次输入四帧，并把每层的因果特征缓存交给下一块；这让长视频既不看未来，也不用一次塞进全部显存。

A video VAE cannot split time arbitrarily: convolutions need preceding features at every boundary. The first frame primes the state, then four-frame chunks reuse causal feature caches. This preserves temporal causality without holding the entire clip in memory.

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/modules/vae.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vae.py#L516-L568)

```python
def encode(self, x, scale):
    self.clear_cache()
    t = x.shape[2]
    iter_ = 1 + (t - 1) // 4
    for i in range(iter_):
        self._enc_conv_idx = [0]
        if i == 0:
            out = self.encoder(x[:, :, :1], feat_cache=self._enc_feat_map,
                               feat_idx=self._enc_conv_idx)
        else:
            out_ = self.encoder(x[:, :, 1 + 4 * (i - 1):1 + 4 * i],
                                feat_cache=self._enc_feat_map,
                                feat_idx=self._enc_conv_idx)
            out = torch.cat([out, out_], 2)
    mu, log_var = self.conv1(out).chunk(2, dim=1)
    mu = (mu - scale[0]) * scale[1]
    self.clear_cache()
    return mu
```

## 逐行讲解 / What's happening

1. **第 519-521 行 / Lines 519-521**：`1 + (t - 1) // 4` 明确了“1、4、4…”的时间协议，而不是假装所有块等长。 / The expression makes the `1, 4, 4, ...` protocol explicit instead of pretending that every chunk has equal length.
2. **第 523-534 行 / Lines 523-534**：每一块都从第一个卷积缓存槽开始计数，但缓存数组本身跨块保留；`torch.cat(..., 2)` 沿时间轴拼 latent。 / Each chunk restarts its convolution-slot counter while preserving the cache array across chunks; concatenation happens along the latent time axis.
3. **第 535-542 行 / Lines 535-542**：最后才切分高斯的均值和方差，并将均值映射到扩散模型使用的标准 latent 坐标。 / Only after all chunks are encoded does the code split Gaussian statistics and map the mean into the diffusion model's latent coordinates.

## 类比 / The analogy

这像接力录像：第一位摄影师先校准镜头，之后每位摄影师只拍四秒，却把上一位的机位信息传下去。 / It is relay filming: the first camera operator establishes the shot, and every later operator records only four seconds while inheriting the previous setup.

## 自己跑一遍 / Try it yourself

```python
import torch

x = torch.arange(9).view(1, 1, 9)
chunks = [x[:, :, :1]] + [x[:, :, 1 + 4*i:1 + 4*(i+1)] for i in range(2)]
cache, out = None, []
for chunk in chunks:
    cache = chunk[:, :, -1:] if cache is None else chunk[:, :, -1:]
    out.append(chunk)
print([c.shape[-1] for c in chunks], torch.cat(out, 2).shape[-1])
```

运行 / Run with: `pip install torch && python try.py`

预期输出 / Expected output: ` [1, 4, 4] 9 `。

最关键的是边界协议；真实模型把 `cache` 扩展为每层卷积的历史特征。 / The important part is the boundary protocol; the real model expands `cache` into history for every convolution layer.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **流式 CausalConv3d / Streaming CausalConv3d**：同一仓库的底层卷积缓存过去帧。 / The same repository's lower-level convolution stores past frames.
- **自回归 KV cache / Autoregressive KV cache**：Transformer 也把过去状态留给下一段计算。 / Transformers similarly retain prior state for the next segment.

## 注意事项 / Caveats / when it breaks

- **切块长度是架构契约 / Chunk size is an architectural contract**：`4` 来自 VAE 的时间下采样设计，不能任意修改。 / The `4` follows the VAE's temporal-downsampling design and is not an arbitrary knob.
- **缓存必须在样本间清空 / Clear caches between samples**：否则上一个视频的历史会污染下一个。 / Otherwise history from one video leaks into the next.

## 延伸阅读 / Further reading

- [Wan2.1 VAE source](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vae.py)
