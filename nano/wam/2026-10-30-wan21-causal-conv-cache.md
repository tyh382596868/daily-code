---
date: 2026-10-30
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/modules/vae.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vae.py#L20-L42
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, wam, temporal-compression]
build_role: vae-encoder-decoder
---

# Wan2.1 因果卷积的时间缓存 / Wan2.1 Causal-Convolution Cache

> **一句话 / In one line**: 把前一个视频 chunk 的尾帧接到当前 chunk 前面，让分块推理仍然因果连续。 / Prepend the previous chunk’s tail frames so chunked inference remains causally continuous.

## 为什么重要 / Why this matters

视频 VAE 不能在流式生成时看未来帧；又不能让每个 chunk 都忘记刚过去的运动。这个卷积层把所需历史显式作为 `cache_x` 传入。

A video VAE cannot inspect future frames during streaming, yet each chunk must remember recent motion. This convolution makes the required history explicit as `cache_x`.

## 代码 / The code

`Wan-Video/Wan2.1` — [`vae.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vae.py#L20-L42)

```python
def forward(self, x, cache_x=None):
    padding = list(self._padding)
    if cache_x is not None and self._padding[4] > 0:
        cache_x = cache_x.to(x.device)
        x = torch.cat([cache_x, x], dim=2)
        padding[4] -= cache_x.shape[2]
    x = F.pad(x, padding)
    return super().forward(x)
```

## 逐行讲解 / What's happening

1. 中文：复制 padding 配置，因为本次 chunk 的历史长度会改变前补量。
   English: Copy the padding configuration because the available history changes this chunk’s left pad.
2. 中文：缓存沿时间维拼接到当前帧前。
   English: Cache is concatenated before current frames along time.
3. 中文：真实历史抵消一部分零 padding，卷积看到的时间窗口保持一致。
   English: Real history replaces some zero padding, keeping the convolution’s temporal window consistent.

## 类比 / The analogy

像连续拍电影时先把上一条胶片的最后两格接到新胶片开头，避免剪辑点突然失忆。

Like splicing the last two frames of the prior film reel onto the next reel so the cut does not forget motion.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-WAM

中文：这是 nanoWAM 的视频编码器/解码器时间边界。输入为当前 latent chunk 和可选历史，输出为保持因果性的特征。它依赖 VAE 的时间感受野，供下游 DiT 或解码器消费；生产实现还要为每层维护独立 cache、处理 batch 重置和 dtype/device 迁移。

English: This is the temporal boundary of nanoWAM’s video encoder/decoder. It takes a current latent chunk plus optional history and produces causal features. It depends on the VAE receptive field and feeds a downstream DiT or decoder; production code also needs per-layer caches, batch resets, and dtype/device handling.

## 自己跑一遍 / Try it yourself

```python
history = [8, 9]
chunk = [10, 11, 12]
kernel = 3
windowed = history + chunk
out = [sum(windowed[i:i+kernel]) for i in range(len(chunk))]
print(out)
```

运行 / Run with: `python try.py`

预期输出 / Expected output: `[27, 30, 33]`

中文：第一步使用了历史 8、9；没有它会被零填充替代。 English: The first step uses history 8 and 9; without it, zeros would replace them.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **流式音频卷积 / Streaming audio convolution**: 音频与视频都用历史 buffer 代替未来上下文。 / Audio and video both use history buffers instead of future context.

## 注意事项 / Caveats / when it breaks

- 中文：cache 长度必须与 kernel 的时间感受野匹配。 / English: Cache length must match the kernel’s temporal receptive field.

## 延伸阅读 / Further reading

- [Wan2.1 VAE](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vae.py#L20-L42)
