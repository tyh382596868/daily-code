---
date: 2026-07-08
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/modules/vae.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vae.py#L507-L569
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, vae, temporal-cache]
build_role: vae-encoder-decoder advanced variant
---

# Wan2.1 VAE：按时间块编码，靠 cache 保持连续 / Wan2.1 VAE: Encode Time Chunks While Keeping Continuity with Cache

> **一句话 / In one line**: Wan VAE 不一次性吃完整视频，而是按 `1,4,4,...` 时间块编码，并用卷积 cache 维持因果上下文。 / Wan VAE does not encode a whole video at once; it processes `1,4,4,...` temporal chunks and keeps causal context in convolution caches.

## 为什么重要 / Why this matters

WAM 的视频 VAE 很容易成为显存瓶颈。视频帧越多，3D 卷积中间激活越大。Wan2.1 的做法是把时间维拆成小块，同时每次把 causal conv 需要的尾部特征缓存下来。这样既能流式处理长视频，又不会让每个 chunk 忘掉前面的时间上下文。

A video VAE in a WAM can become the memory bottleneck quickly. More frames mean larger 3D convolution activations. Wan2.1 splits time into small chunks and caches the tail features needed by causal convolutions. This allows streaming long videos without making each chunk forget its temporal context.

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/modules/vae.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vae.py#L507-L569)

```python
def encode(self, x, scale):
    self.clear_cache()
    ## cache
    t = x.shape[2]
    iter_ = 1 + (t - 1) // 4
    ## 对encode输入的x，按时间拆分为1、4、4、4....
    for i in range(iter_):
        self._enc_conv_idx = [0]
        if i == 0:
            out = self.encoder(
                x[:, :, :1, :, :],
                feat_cache=self._enc_feat_map,
                feat_idx=self._enc_conv_idx)
        else:
            out_ = self.encoder(
                x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                feat_cache=self._enc_feat_map,
                feat_idx=self._enc_conv_idx)
            out = torch.cat([out, out_], 2)
    mu, log_var = self.conv1(out).chunk(2, dim=1)
    if isinstance(scale[0], torch.Tensor):
        mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
            1, self.z_dim, 1, 1, 1)
    else:
        mu = (mu - scale[0]) * scale[1]
    self.clear_cache()
    return mu

def decode(self, z, scale):
    self.clear_cache()
    # z: [b,c,t,h,w]
    if isinstance(scale[0], torch.Tensor):
        z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
            1, self.z_dim, 1, 1, 1)
    else:
        z = z / scale[1] + scale[0]
    iter_ = z.shape[2]
    x = self.conv2(z)
    for i in range(iter_):
        self._conv_idx = [0]
        if i == 0:
            out = self.decoder(
                x[:, :, i:i + 1, :, :],
                feat_cache=self._feat_map,
                feat_idx=self._conv_idx)
        else:
            out_ = self.decoder(
                x[:, :, i:i + 1, :, :],
                feat_cache=self._feat_map,
                feat_idx=self._conv_idx)
            out = torch.cat([out, out_], 2)
    self.clear_cache()
    return out
```

## 逐行讲解 / What's happening

1. **第 508 行 / Line 508 (`clear_cache`)**:
   - 中文: 每次 encode 前重置 cache，避免上一段视频的时间上下文泄漏进来。
   - English: The encode pass resets caches so temporal context from a previous video cannot leak in.
2. **第 511-523 行 / Lines 511-523 (时间切块)**:
   - 中文: 第一块只取 1 帧，后面每块取 4 帧。这和 3D VAE 的时间下采样/因果卷积对齐。
   - English: The first chunk uses one frame and later chunks use four frames, matching temporal downsampling and causal convolution.
3. **第 524-530 行 / Lines 524-530 (latent scale)**:
   - 中文: 编码后只返回缩放过的 `mu`，让扩散模型在标准化 latent 空间里工作。
   - English: After encoding, it returns scaled `mu` so the diffusion model operates in normalized latent space.
4. **第 543-563 行 / Lines 543-563 (逐 latent 帧 decode)**:
   - 中文: decode 也逐时间片运行，每次传入同一份 `feat_cache`，让输出帧保持时间连续。
   - English: Decode also runs time slice by time slice, passing the same `feat_cache` so generated frames remain temporally continuous.

## 类比 / The analogy

这像翻译一部长电影的字幕。你不会一次把整部电影塞进脑子，而是一段段翻译；但每段开头要记住上一段最后几句，否则人物称呼和上下文会断。

It is like translating subtitles for a long movie. You do not hold the whole movie in memory; you translate it segment by segment. But each segment needs the last few lines of context, or names and references break.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

这是 `vae-encoder-decoder` 的生产级变体。上游是原始视频帧，下游是 DiT / diffusion backbone 的 latent token。如果 nanoWAM 只做短视频，可以先写无 cache 的 VAE；一旦要处理长视频或流式视频，就需要这种 chunk + cache 设计。生产级实现还要处理 dtype、device、batch 内不同长度和 cache 生命周期。

This is a production-style variant of `vae-encoder-decoder`. Upstream are raw video frames; downstream are latent tokens for the DiT or diffusion backbone. A short-video nanoWAM can start with a non-cached VAE, but long or streaming video needs this chunk-and-cache design. Production code must also handle dtype, device, variable lengths, and cache lifetime.

## 自己跑一遍 / Try it yourself

```python
def wan_encode_chunks(t):
    chunks = []
    for i in range(1 + (t - 1) // 4):
        if i == 0:
            chunks.append((0, min(1, t)))
        else:
            chunks.append((1 + 4 * (i - 1), min(1 + 4 * i, t)))
    return chunks

print(wan_encode_chunks(1))
print(wan_encode_chunks(9))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[(0, 1)]
[(0, 1), (1, 5), (5, 9)]
```

输出展示了 Wan VAE 的时间切块策略：先单帧启动 cache，再用 4 帧为单位推进。

The output shows Wan VAE's temporal chunking strategy: start the cache with one frame, then advance in groups of four.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Open-Sora causal VAE** / **Open-Sora causal VAE**: 同样需要在时间维上防止未来帧泄漏。 / It also prevents future-frame leakage along the temporal dimension.
- **FastWAM video cache** / **FastWAM video cache**: 推理时预填视频上下文，再逐步解码动作。 / It prefills video context at inference time and then decodes actions step by step.

## 注意事项 / Caveats / when it breaks

- **cache 必须按视频清空** / **Caches must be cleared per video**: 否则两个样本会共享时间上下文。 / Otherwise two samples share temporal context accidentally.
- **切块大小绑定模型结构** / **Chunk size is architecture-dependent**: `1,4,4,...` 不是通用常数，它和 Wan VAE 的时间下采样有关。 / `1,4,4,...` is not universal; it is tied to Wan VAE's temporal downsampling.

## 延伸阅读 / Further reading

- [Wan2.1 VAE encode/decode](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vae.py#L507-L569)
