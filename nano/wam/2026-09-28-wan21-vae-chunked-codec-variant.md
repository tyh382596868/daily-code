---
date: 2026-09-28
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/modules/vae.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vae.py#L516-L568
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, wam, vae-codec]
build_role: vae-encoder-decoder advanced variant
---

# Wan2.1 VAE codec 变体：首帧单独编码，后续按时间块接上 / Wan2.1 VAE Codec Variant: Encode the First Frame Alone, Then Append Temporal Chunks

> **一句话 / In one line**: 视频 VAE 的时间维不能随便整段过；chunk、cache 和 scale 必须一起设计。 / A video VAE cannot treat time as one arbitrary slab; chunking, cache state, and latent scaling have to be designed together.

## 为什么重要 / Why this matters

WAM 要在 latent 空间里想象未来，VAE 是像素世界和 latent 世界之间的门。Wan2.1 这里把首帧和后续 4 帧块分开编码，配合卷积 cache 保持因果时间结构，再在 encode/decode 两侧做同一套 scale 变换。

A WAM imagines futures in latent space, so the VAE is the gate between pixels and latents. Wan2.1 encodes the first frame separately from later four-frame chunks, uses convolution caches to preserve temporal structure, and applies paired scaling in encode and decode.

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

1. **第 516-523 行 / Lines 516-523**:
   - 中文: 进入 encode 先清 cache，再按 `1 + (t-1)//4` 计算时间块数量。
   - English: Encode starts by clearing caches, then computes temporal chunks as `1 + (t-1)//4`.
1. **第 524-534 行 / Lines 524-534**:
   - 中文: 首帧单独过 encoder，后续每 4 帧一组，并沿时间维拼接输出。
   - English: The first frame goes through the encoder alone; later frames are processed in groups of four and concatenated along time.
1. **第 535-542 行 / Lines 535-542**:
   - 中文: 只返回缩放后的 `mu`，说明推理 codec 用确定性 latent，而不是训练时采样。
   - English: It returns scaled `mu`, so inference uses deterministic latents rather than sampling.
1. **第 544-568 行 / Lines 544-568**:
   - 中文: decode 做反向 scale，然后逐 latent frame 解码，同时复用 decoder cache。
   - English: Decode reverses the scale and decodes latent frames one by one while carrying decoder cache.

## 类比 / The analogy

像压缩长视频时先单独保存封面，再每隔几帧做一个 GOP。封面给时间轴定锚，后面的块靠缓存接住连续性。

It is like compressing a long video by storing the cover frame first and then encoding GOP-like chunks. The first frame anchors time, and later chunks carry continuity through cache.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

在 nanoWAM 里，这是 `VideoTokenizer` 或 `LatentCodec` 模块：输入 `[B,C,T,H,W]` 像素视频，输出时间压缩后的 latent；采样器和 DiT 都在 latent 上工作。省掉这个组件就只能在像素空间建模，成本会爆炸。生产版还要处理分辨率、dtype、cache 生命周期和多卡切分。

In nanoWAM, this is the `VideoTokenizer` or `LatentCodec`: pixels `[B,C,T,H,W]` go in, temporally compressed latents come out, and the sampler/DiT operate there. Without it, you model pixels directly and cost explodes. A production version also needs resolution handling, dtype policy, cache lifetime management, and distributed splitting.

## 自己跑一遍 / Try it yourself

```python
frames=list(range(10))
chunks=[frames[:1]]
for i in range(1, len(frames), 4):
    chunks.append(frames[i:i+4])
print(chunks)
latents=[sum(c) for c in chunks]
print(latents)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
[[0], [1, 2, 3, 4], [5, 6, 7, 8], [9]]
[0, 10, 26, 9]
```

这个例子只模拟切块；真实 VAE 里每块会带着卷积 cache 进入下一块。

This only simulates chunking; the real VAE carries convolution cache from chunk to chunk.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **DreamZero VideoVAE cache** / **DreamZero uses temporal chunks and cache for video latents.**
- **Open-Sora causal VAE** / **Open-Sora also protects temporal causality in video compression.**
- **Wan2.1 resample cache** / **Wan2.1 separates spatial resizing from temporal cache state elsewhere.**

## 注意事项 / Caveats / when it breaks

- **cache 必须清理** / **Stale encoder or decoder cache would leak one video into the next.**
- **scale 要成对** / **Encode and decode must use inverse scale transforms.**
- **训练和推理不同** / **Training may use `mu, log_var`; deterministic inference often uses `mu` only.**

## 延伸阅读 / Further reading

- Wan2.1 VAE source permalink above
