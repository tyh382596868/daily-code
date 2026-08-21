---
date: 2026-08-21
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/modules/vae.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vae.py#L516-L568
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, vae, temporal-chunking, latent-codec]
build_role: vae-encoder-decoder advanced variant
---

# Wan2.1 VAE codec：首帧单独过，后面四帧一组 / Wan2.1 VAE Codec: First Frame Alone, Then Four at a Time

> **一句话 / In one line**: Wan2.1 的 VAE encode 先处理第 1 帧，再按 4 帧 chunk 继续编码并拼接 latent；decode 则逐 latent 帧解码，同时用内部 cache 保持时间卷积上下文。 / Wan2.1's VAE encode handles the first frame alone, then encodes later frames in 4-frame chunks and concatenates latents; decode reconstructs one latent frame at a time while internal caches preserve temporal-conv context.

## 为什么重要 / Why this matters

视频 VAE 不能只当图片 VAE 循环跑。时间卷积需要过去帧上下文，但整段视频一次塞进去又容易占太多显存。Wan2.1 的 codec 把时间轴切成固定节奏：首帧建立 cache，后续 chunk 复用 cache，最后再清理状态。

A video VAE is not just an image VAE in a loop. Temporal convolutions need context from previous frames, but encoding the whole clip at once can be memory-heavy. Wan2.1 splits time into a fixed rhythm: first frame initializes cache, later chunks reuse it, and state is cleared afterward.

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/modules/vae.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vae.py#L516-L568)

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
            out = self.encoder(x[:, :, :1, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
        else:
            out_ = self.encoder(
                x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                feat_cache=self._enc_feat_map,
                feat_idx=self._enc_conv_idx)
            out = torch.cat([out, out_], 2)
    mu, log_var = self.conv1(out).chunk(2, dim=1)
    if isinstance(scale[0], torch.Tensor):
        mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(1, self.z_dim, 1, 1, 1)
    else:
        mu = (mu - scale[0]) * scale[1]
    self.clear_cache()
    return mu

def decode(self, z, scale):
    self.clear_cache()
    # z: [b,c,t,h,w]
    if isinstance(scale[0], torch.Tensor):
        z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(1, self.z_dim, 1, 1, 1)
    else:
        z = z / scale[1] + scale[0]
    iter_ = z.shape[2]
    x = self.conv2(z)
    for i in range(iter_):
        self._conv_idx = [0]
        if i == 0:
            out = self.decoder(x[:, :, i:i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
        else:
            out_ = self.decoder(x[:, :, i:i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
            out = torch.cat([out, out_], 2)
    self.clear_cache()
    return out
```

## 逐行讲解 / What's happening

1. **第 516-522 行 / Lines 516-522 (encode chunk plan)**:
   - 中文: 编码前清 cache，`iter_ = 1 + (t - 1) // 4` 表示首帧单独一组，后面每 4 帧一组。
   - English: Encode clears cache, and `iter_ = 1 + (t - 1) // 4` means first frame alone, then chunks of four.
2. **第 523-534 行 / Lines 523-534 (cached encoder calls)**:
   - 中文: 每个 chunk 都传入 `feat_cache` 和 `feat_idx`，让因果时间卷积能接上前文。
   - English: Each chunk receives `feat_cache` and `feat_idx`, letting causal temporal convs continue from prior context.
3. **第 535-542 行 / Lines 535-542 (latent scale)**:
   - 中文: `conv1` 产出 `mu/log_var`，这里取 `mu` 并做 latent 标准化，返回给扩散模型使用。
   - English: `conv1` yields `mu/log_var`; this path takes `mu`, standardizes latents, and returns them to the diffusion model.
4. **第 544-568 行 / Lines 544-568 (decode one latent frame at a time)**:
   - 中文: decode 先反标准化，再逐 latent 帧过 decoder，并在最后清理 cache。
   - English: Decode unstandardizes latents, runs the decoder one latent frame at a time, then clears cache.

## 类比 / The analogy

像读长篇小说做摘要。第一章要单独建立人物关系，后面每四章一组继续读；每组都带着前面的笔记，但读完要把临时便签收走。

It is like summarizing a long novel. The first chapter establishes the cast, later chapters are read in groups of four, each group uses prior notes, and temporary notes are cleared at the end.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-{VLA,WAM}

在 nanoWAM 里，这是 `vae-encoder-decoder` 的高级版本。上游是原始视频帧，下游是 DiT/flow model 使用的 video latent。最小 nanoWAM 可以先用无 cache 的 2D VAE；生产版需要像这里一样明确时间 chunk、latent scale、cache 生命周期和首帧特殊处理。

In a nanoWAM, this is an advanced `vae-encoder-decoder` component. Upstream is raw video frames; downstream is the video latent consumed by the DiT/flow model. A minimal nanoWAM can begin with a cache-free 2D VAE; a production version needs explicit temporal chunks, latent scaling, cache lifetime, and first-frame handling.

## 自己跑一遍 / Try it yourself

```python
frames = list(range(10))
chunks = [frames[:1]]
for start in range(1, len(frames), 4):
    chunks.append(frames[start:start + 4])

print(chunks)
print([sum(chunk) for chunk in chunks])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[[0], [1, 2, 3, 4], [5, 6, 7, 8], [9]]
[0, 10, 26, 9]
```

中文: toy 版本只展示 Wan2.1 的时间切块节奏：`1, 4, 4, ...`。

English: The toy version only shows Wan2.1's temporal chunk rhythm: `1, 4, 4, ...`.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **streaming conv nets** / **streaming conv nets**: 流式语音/视频模型常缓存上一段尾部上下文。
- **chunked attention** / **chunked attention**: 长序列模型也会分块处理，同时传递 KV 或摘要状态。

## 注意事项 / Caveats / when it breaks

- **cache 生命周期必须成对** / **Cache lifetime must be paired**: encode/decode 前后都要清 cache，否则不同视频会互相污染。
- **scale 是 latent 合同** / **Scale is part of the latent contract**: 下游扩散模型看到的是标准化 latent，不能随意跳过。

## 延伸阅读 / Further reading

- [Wan2.1 VAE `encode` / `decode`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vae.py#L516-L568)
- [Wan2.1 repository](https://github.com/Wan-Video/Wan2.1)
