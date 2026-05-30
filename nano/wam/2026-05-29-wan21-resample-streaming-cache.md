---
date: 2026-05-29
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/modules/vae.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vae.py#L66-L160
difficulty: advanced
read_time: ~12 min
tags: [code-of-the-day, wam, temporal-compression, streaming, vae, cache]
build_role: Temporal compression — 3D Resample with cross-chunk cache for streaming long-video VAE
---

# Resample 的 feat_cache 让 3D VAE 能流式处理无限长视频 / `feat_cache` lets a 3D VAE process video of arbitrary length, one chunk at a time

> **一句话 / In one line**: Wan2.1 的 `Resample` 在做时间维度上 / 下采样时,会把每个 chunk 的"最后几帧"留作 `feat_cache`,下一个 chunk 来的时候把这份缓存拼到左边再卷积 —— 等价于一次完整长视频的输出,但**显存只跟 chunk 大小有关**。 / Wan2.1's `Resample` keeps the trailing frames of each chunk as `feat_cache`. When the next chunk arrives, the cache is prepended before the temporal conv runs. The output is identical to processing the full clip in one shot, but memory scales with chunk size, not total length.

## 为什么重要 / Why this matters

WAM 的 3D VAE 编码一段 5 分钟视频意味着一次处理 30×128×128×64×16 之类的张量,很容易把 80GB H100 撑爆。生产里必须**分块编码**:一次塞 8 帧,然后下一段 8 帧。但是 3D 时间卷积的 receptive field 跨帧,如果两块之间不维护"上一块尾部"的状态,卷积就会在 chunk 边界产生 zero-padding 伪影,decode 出来视频每 8 帧抖一下。`Resample.feat_cache` 是这个问题的解 —— 用 ~95 行代码实现"时间维度的 streaming 卷积",兼具上采样(`upsample3d`)和下采样(`downsample3d`)两种模式,而且和昨天讲的 `CausalConv3d` 直接配合。

A 3-D VAE that encodes 5 minutes of video must process tensors like `[1, 16, 300, 128, 128]` — an 80 GB H100 won't fit it. Production splits the clip into chunks (say 8 frames at a time). But a temporal conv has cross-frame receptive field, so without carrying state across chunks the conv pads with zeros at every chunk boundary and decoded videos visibly jitter every 8 frames. `Resample.feat_cache` is the cure — 95 lines that implement temporal-streaming convolution with both `upsample3d` and `downsample3d` modes, all designed to compose with yesterday's `CausalConv3d`.

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/modules/vae.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vae.py#L66-L160) (key portions, lightly trimmed)

```python
CACHE_T = 2

class Resample(nn.Module):
    def __init__(self, dim, mode):
        super().__init__()
        self.dim, self.mode = dim, mode
        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == 'downsample2d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == 'downsample3d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1),
                                          stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()

        if self.mode == 'upsample3d' and feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                # first chunk: no cache yet, mark it as 'Rep' = use zero padding once
                feat_cache[idx] = 'Rep'
                feat_idx[0] += 1
            else:
                # Save the last CACHE_T frames of current chunk as the NEXT chunk's cache
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] != 'Rep':
                    cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2),
                                         cache_x], dim=2)
                if cache_x.shape[2] < 2 and feat_cache[idx] == 'Rep':
                    cache_x = torch.cat([torch.zeros_like(cache_x), cache_x], dim=2)
                # Apply the time conv with the prior cache as "past frames"
                if feat_cache[idx] == 'Rep':
                    x = self.time_conv(x)                          # cold start
                else:
                    x = self.time_conv(x, feat_cache[idx])         # streaming
                feat_cache[idx] = cache_x                          # write-back
                feat_idx[0] += 1

                # Interleave temporal upsample by 2× via reshape
                x = x.reshape(b, 2, c, t, h, w)
                x = torch.stack((x[:, 0], x[:, 1]), 3).reshape(b, c, t * 2, h, w)

        # Spatial resample (2D), reshape video to images batch-wise
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        if self.mode == 'downsample3d' and feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = x.clone()
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -1:, :, :].clone()
                # Prepend the last frame of previous chunk, then strided temporal conv
                x = self.time_conv(
                    torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
        return x
```

## 逐行讲解 / What's happening

1. **四种模式 + 可选 time_conv / Four modes and an optional time_conv**:
   - 中文:`upsample2d / downsample2d` 只动空间;`upsample3d / downsample3d` 额外挂一个 `CausalConv3d` 处理时间维度。`upsample3d` 的 time_conv 输出通道是 `dim * 2`,因为后面 reshape 时把通道劈成两份,把"两个相邻帧"在时间上拼出去,等价于沿 t 轴 2× 上采样;`downsample3d` 用 `stride=(2, 1, 1)` 直接在 conv 里降 t。
   - English: `upsample2d/downsample2d` touch only the spatial dims; `upsample3d/downsample3d` add a `CausalConv3d` for time. The upsample version doubles the output channels so reshape can split them into "two adjacent frames", giving 2× temporal upsampling. The downsample version uses `stride=(2,1,1)` to halve t inside the conv directly.

2. **`feat_cache` 是个 list,`feat_idx[0]` 是当前游标 / feat_cache is a list, feat_idx the cursor**:
   - 中文:整个 VAE encoder / decoder 里可能有 N 个 Resample,每个都要自己的缓存。调用方维护一个 list `feat_cache = [None]*N`(N 是 Resample 数量)和一个共享 `feat_idx = [0]`(用列表是因为要在子模块里就地 +=)。每个 Resample 自取 `feat_cache[idx]`、用完 `idx + 1`,确保下一次推理时同一个 Resample 拿到同一个槽位。
   - English: a VAE pass goes through N `Resample` layers; each needs its own slot. The caller maintains `feat_cache = [None] * N` and a shared `feat_idx = [0]` (passed by reference via a list so each child can mutate it). Each `Resample` reads slot `idx`, writes back, increments — the next chunk's pass through the same layer hits the same slot.

3. **首次推理:`feat_cache[idx] = 'Rep'` 表示"复制零"** / **First call: `feat_cache[idx] = 'Rep'`** :
   - 中文:第一个 chunk 没有"过去"可用,用字符串 `'Rep'` 作为 sentinel —— 后续逻辑碰到 `'Rep'` 就退回零填充。第二次起 `feat_cache[idx]` 是真正的过去帧 tensor。
   - English: the first chunk has no past, so the slot is set to the sentinel string `'Rep'`. Later branches see `'Rep'` and fall back to zero-padding for that step. From the second chunk on, the slot holds the actual past-frame tensor.

4. **`cache_x = x[:, :, -CACHE_T:].clone()` 保存尾部 / Save trailing frames as next-chunk cache**:
   - 中文:`CACHE_T = 2` 是 kernel size 3 的副作用 —— 因为时间 conv 看 `{t-2, t-1, t}`,所以下一个 chunk 还需要本 chunk 最后两帧。`clone()` 是必须的:不复制会跟 activation 共享内存,后面被覆盖。
   - English: `CACHE_T = 2` because a kernel-3 temporal conv looks at `{t-2, t-1, t}`, so the next chunk still needs the last two frames of this chunk. `clone()` is mandatory — without it the cache aliases the activation tensor and gets overwritten on the next forward.

5. **`time_conv(x, feat_cache[idx])` 调 CausalConv3d 的 cache_x 入口 / Pass the cache into CausalConv3d**:
   - 中文:这就是昨天 `CausalConv3d` 笔记里那个 `cache_x` 参数的实战。`time_conv(x, cache_x)` 会先把 cache 拼在 x 左边,然后做严格因果的卷积 —— chunk 边界跟一气呵成时**输出完全一致**。
   - English: this is the production caller of yesterday's `CausalConv3d.forward(x, cache_x)`. The cache is concatenated to the left of `x` before the conv runs, so chunk boundaries produce output that's *identical* to the equivalent one-shot run on the full clip.

6. **reshape 实现 2× 时间上采样 / Reshape implements 2× temporal upsample**:
   - 中文:`x.reshape(b, 2, c, t, h, w)` 把通道维劈成 2 份,`torch.stack(..., dim=3)` 把两份交错塞进时间维,reshape 回 `[b, c, t*2, h, w]`。等价于"卷积输出双倍通道,再用 channel-shuffle 升时间维",没有引入插值参数。
   - English: split the channel axis into 2, interleave the two halves along the time axis via `torch.stack(..., dim=3)`, reshape back. Net effect: temporal length doubles, channel count halves, no interpolation parameters introduced.

7. **downsample3d 的简单分支 / The simpler downsample3d path**:
   - 中文:下采样不需要 channel-doubling 那种花活;直接拿上一个 chunk 的最后一帧拼到当前 chunk 左边,然后用 stride=2 的 CausalConv3d 把 t 减半。
   - English: downsampling skips the channel trickery — prepend the previous chunk's last frame and run the strided temporal conv to halve `t`.

## 类比 / The analogy

像电视台播球赛:导演不可能一次性把整场比赛 90 分钟的画面都收进 OBS,只能每 30 秒收一段切给观众。但每段 30 秒之间,如果不带"上一段最后两帧"的接续状态,屏幕会在切片处闪烁。`feat_cache` 就是那个"上段最后两帧缓冲区",`feat_idx` 是"我现在播到第几个轨道(直播/嘉宾/广告)了"的指针。

Imagine broadcasting a 90-minute game: the encoder can't ingest the whole match at once, so OBS chops it into 30-second segments. Without a "tail-of-previous-segment" buffer, the stream flickers at every boundary. `feat_cache` is that buffer, and `feat_idx` is the cursor that tracks which track (main feed, picture-in-picture, ad break) we're currently feeding.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-WAM

中文:在 nanoWAM 里 `Resample` 是 `nano/wam/blocks/resample.py`,跟 `causal_conv3d.py` 紧密耦合 —— Encoder3d 和 Decoder3d 各自堆 4-6 个 Resample(每个减半 t、h、w),`forward` 时通过 `feat_cache` 在 chunks 间传状态。上游是 Encoder3d 的 ResidualBlock + AttentionBlock(本笔记没展开),下游是 VAE 输出 / 输入。如果你**只在训练时跑短视频、推理也只跑短视频**,可以省掉整个 feat_cache 机制,把 Resample 退化成单 chunk 版本。但只要你想:(a) 流式生成长视频,(b) 增量编码已有视频,(c) memory-bounded inference,feat_cache 就是硬需求。生产实现要补:(1) **多 GPU 数据并行时的 cache 同步**(不同 rank 的 cache 是独立的);(2) **bf16 cache vs fp32 cache**(混合精度时要考虑精度衰减);(3) **cache GC**(同一序列结束后清掉,否则下一个序列会拿到过期状态)。

English: in nanoWAM, `Resample` is `nano/wam/blocks/resample.py`, tightly coupled to `causal_conv3d.py`. `Encoder3d` and `Decoder3d` stack 4-6 `Resample` layers each (halving t, h, w step by step). The caller hands a `feat_cache` list and a `feat_idx` cursor to `forward`. Upstream within the VAE are `ResidualBlock`/`AttentionBlock`; downstream is the final encoder/decoder I/O. If you only ever process short clips in both training and inference, you can drop the whole cache machinery. But if you need (a) streaming long-video generation, (b) incremental encoding of existing video, or (c) memory-bounded inference, the cache is hard-required. Production extensions: (1) multi-GPU cache management (each rank holds its own), (2) bf16-vs-fp32 cache precision tradeoffs, (3) cache GC at sequence end to avoid stale state leaking into the next sequence.

## 自己跑一遍 / Try it yourself

```python
# pip install torch
# Verify chunked == one-shot for a temporal-causal Conv3d with a feat_cache.
import torch, torch.nn as nn, torch.nn.functional as F

CACHE_T = 2

class CausalConv3d(nn.Conv3d):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._pad = (self.padding[2], self.padding[2],
                     self.padding[1], self.padding[1],
                     2 * self.padding[0], 0)
        self.padding = (0, 0, 0)
    def forward(self, x, cache=None):
        pad = list(self._pad)
        if cache is not None and self._pad[4] > 0:
            x = torch.cat([cache, x], dim=2)
            pad[4] -= cache.shape[2]
        return super().forward(F.pad(x, pad))

torch.manual_seed(0)
conv = CausalConv3d(1, 1, kernel_size=(3, 1, 1), padding=(1, 0, 0))

# Full clip: 10 frames, 1x1 spatial
x_full = torch.randn(1, 1, 10, 1, 1)
y_full = conv(x_full)                                # one-shot output

# Chunked: 5 + 5 frames, with cache
y_chunks, cache = [], None
for chunk in [x_full[:, :, :5], x_full[:, :, 5:]]:
    y = conv(chunk, cache=cache)
    cache = chunk[:, :, -CACHE_T:].clone()           # next-chunk cache
    y_chunks.append(y)
y_streamed = torch.cat(y_chunks, dim=2)

print("max abs diff:", (y_full - y_streamed).abs().max().item())
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
max abs diff: 0.0
```

中文:chunk 输出和一气呵成的输出完全一致 —— 这就是 `feat_cache` 的存在理由。

English: the chunked output is bit-identical to the one-shot version — that's the whole point of `feat_cache`.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **CogVideoX VAE** / **CogVideoX VAE**: 中文 — 用同样的 `cache` 机制做流式编码,但拆得更细(每个 ResidualBlock 都有自己的 cache)。 / English — same caching idiom but at finer granularity (one cache per ResidualBlock).
- **Open-Sora HunyuanVAE** / **Open-Sora HunyuanVAE**: 中文 — `opensora/models/hunyuan_vae/vae.py` 实现了同样的"上块尾部 + 当前块"拼接。 / English — `opensora/models/hunyuan_vae/vae.py` implements the same prev-chunk-tail + current-chunk pattern.
- **Streaming WaveNet / 流式 TTS** / **Streaming WaveNet / streaming TTS**: 中文 — 一维音频流式合成的祖宗,VAE 的 feat_cache 直接借鉴。 / English — the 1-D ancestor; VAE feat_cache is a direct port of streaming WaveNet's "queue state" trick.
- **今天的 CausalConv3d** / **Today's CausalConv3d**: 中文 — Resample 用 CausalConv3d.cache_x 入口,两段笔记上下衔接。 / English — Resample drives CausalConv3d's `cache_x` parameter; pair the two notes together.

## 注意事项 / Caveats / when it breaks

- **必须 `.clone()`** / **`.clone()` is mandatory**: 中文 — cache 跟 activation 共享 storage 是经典 bug,下次 forward 写入 activation 会同时改写 cache。 / English — sharing storage between cache and activation is the classic bug. The next forward writes the activation, silently corrupting the cache.
- **cache 在 chunks 之间不能跨 batch 重排** / **Don't reorder samples between chunks**: 中文 — `feat_cache[idx][b]` 必须对应同一个 sample b 的过去帧。中途 shuffle batch 会让 cache 和当前帧错位。 / English — `feat_cache[idx][b]` must continue tracking sample `b`. Shuffling the batch mid-sequence misaligns cache and current frames.
- **`feat_idx[0]` 在每次新视频开始必须重置** / **Reset `feat_idx[0]` for each new sequence**: 中文 — 否则下一段视频拿到错误的 cache 槽。新视频开始时调用方要把 `feat_idx[0] = 0` 并把所有 `feat_cache[i] = None`。 / English — at sequence boundaries the caller must reset `feat_idx[0] = 0` and set every `feat_cache[i] = None`. Otherwise the new sequence inherits stale state.
- **上采样的 channel-doubling trick 依赖 `dim * 2` 输出** / **Upsample's channel-doubling assumes `dim * 2` output**: 中文 — 改 time_conv 输出通道会让 reshape 那一步对不上,直接形状错。 / English — changing the time_conv output channel count breaks the reshape-into-doubled-temporal trick; you'll see a shape error immediately.

## 延伸阅读 / Further reading

- [Wan2.1 VAE — full encoder/decoder](https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/vae.py)
- [CogVideoX VAE](https://github.com/THUDM/CogVideoX/tree/main/sat/vae_modules)
- [Streaming convolution / WaveNet queue trick](https://arxiv.org/abs/1611.09482)
- [Open-Sora HunyuanVAE](https://github.com/hpcaitech/Open-Sora/blob/main/opensora/models/hunyuan_vae/vae.py)
