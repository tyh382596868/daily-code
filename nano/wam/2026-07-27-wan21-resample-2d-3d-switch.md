---
date: 2026-07-27
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/modules/vae.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vae.py#L66-L160
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, vae, temporal-compression]
build_role: vae-encoder-decoder advanced variant, temporal resampling switch
---

# Wan2.1 Resample：同一层切换 2D 和 3D 时间缩放 / Wan2.1 Resample: One Layer Switches 2D and 3D Time Scaling

> **一句话 / In one line**: Wan2.1 的 VAE resample 层把空间缩放和时间缩放拆开：普通 2D 用 Conv2d，涉及时间时再加 causal 3D conv 和 cache。 / Wan2.1 splits spatial and temporal resampling: ordinary 2D uses Conv2d, while time-changing paths add causal 3D convolution and cache.

## 为什么重要 / Why this matters

视频 VAE 不能只像图片 VAE 一样缩放 H/W；它还要决定什么时候压缩或扩展时间维，并且分块处理时不能看见未来帧。`Resample` 用 mode 字符串把这些行为集中在一个层里，是 nanoWAM 做 temporal compression 时很值得抄的接口。

A video VAE cannot resample only H/W like an image VAE; it must decide when to compress or expand time, and chunked inference cannot peek at future frames. `Resample` centralizes these behaviors behind a mode string, a useful interface for nanoWAM temporal compression.

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/modules/vae.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vae.py#L66-L160)

```python
class Resample(nn.Module):

    def __init__(self, dim, mode):
        assert mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d',
                        'downsample3d')
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
            self.time_conv = CausalConv3d(
                dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode == 'downsample2d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == 'downsample3d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == 'upsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'
                    feat_idx[0] += 1
                else:

                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] != 'Rep':
                        # cache last frame of last two chunk
                        cache_x = torch.cat([
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                cache_x.device), cache_x
                        ],
                                            dim=2)
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] == 'Rep':
                        cache_x = torch.cat([
                            torch.zeros_like(cache_x).to(cache_x.device),
                            cache_x
                        ],
                                            dim=2)
                    if feat_cache[idx] == 'Rep':
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]),
                                    3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        if self.mode == 'downsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:

                    cache_x = x[:, :, -1:, :, :].clone()
                    # if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx]!='Rep':
                    #     # cache last frame of last two chunk
                    #     cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)

                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x
```

## 逐行讲解 / What's happening

1. **第 68-70 行 / Lines 68-70 (`mode enum`)**:
   - 中文: mode 明确列出 none、2D up/down、3D up/down 五种路径。
   - English: The mode enum lists five paths: none, 2D up/down, and 3D up/down.
2. **第 76-96 行 / Lines 76-96 (`layer construction`)**:
   - 中文: 2D 路径只处理空间；3D 路径额外创建 `time_conv` 来处理时间维。
   - English: 2D paths operate on space only; 3D paths add `time_conv` for the temporal axis.
3. **第 101-137 行 / Lines 101-137 (`upsample3d cache`)**:
   - 中文: 上采样时间维时，cache 补上前一块的尾帧，避免 chunk 边界断裂。
   - English: During temporal upsampling, the cache supplies the previous chunk tail so chunk boundaries remain continuous.
4. **第 138-141 行 / Lines 138-141 (`space as batch`)**:
   - 中文: 代码把 `(b, t)` 合并成 batch，用 2D resample 统一处理每帧空间缩放。
   - English: It folds `(b, t)` into the batch dimension so 2D resampling handles every frame uniformly.
5. **第 143-160 行 / Lines 143-160 (`downsample3d cache`)**:
   - 中文: 下采样时间维也会把上一块尾帧接到当前块前，再更新 cache。
   - English: Temporal downsampling also prepends the previous tail frame before updating the cache.

## 类比 / The analogy

这像剪辑长胶片：每段可以单独进机器，但机器要记住上一段最后一帧，否则拼接处会抖一下。

It is like processing a long film reel in segments: each segment can enter the machine separately, but the machine must remember the last frame of the previous segment or the join will jitter.

## 在 nanoWAM 中的位置 / Where this lives in your nanoWAM

在 nanoWAM 里，这对应 `vae-encoder-decoder` 的 temporal compression 子层。输入是 `B,C,T,H,W` 视频特征，输出是时间/空间尺度改变后的 latent 特征；上游是 encoder residual block，下游是 latent DiT 或 decoder。省掉 cache 会让长视频分块编码在边界处跳变，生产级还要处理多分辨率、混合精度和 chunk 调度。

In nanoWAM this is the temporal-compression sublayer inside `vae-encoder-decoder`. It takes `B,C,T,H,W` video features and emits latent features with changed temporal/spatial scale; encoder residual blocks feed it, and a latent DiT or decoder consumes it. Dropping the cache makes long-video chunk boundaries jump. Production code also needs multi-resolution support, mixed precision, and chunk scheduling.

## 自己跑一遍 / Try it yourself

```python
def temporal_chunks(chunks):
    cache = None
    out = []
    for chunk in chunks:
        work = ([cache] if cache is not None else []) + chunk
        out.append(work[::2])
        cache = chunk[-1]
    return out

print(temporal_chunks([[1, 2, 3], [4, 5, 6]]))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
[[1, 3], [3, 5]]
```

第二个 chunk 的计算看到了第一个 chunk 的尾帧 `3`，这就是 cache 保护时间连续性的核心。

The second chunk sees the first chunk tail `3`; that is the core purpose of the cache for temporal continuity.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Open-Sora temporal VAE** / **Open-Sora temporal VAE**: 中文: Open-Sora 的 causal 3D VAE 也用过去帧约束时间卷积。 / English: Open-Sora causal 3D VAE also constrains temporal convolution to past frames.
- **Wan2.1 chunk VAE** / **Wan2.1 chunk VAE**: 中文: Wan2.1 的 encode/decode 外层也按时间块维护 feat cache。 / English: Wan2.1 encode/decode wrappers also maintain feature cache across time chunks.

## 注意事项 / Caveats / when it breaks

- **默认参数** / **Mutable default**: 中文: `feat_idx=[0]` 是可变默认值，复用时要小心状态串联。 / English: `feat_idx=[0]` is a mutable default; reuse can accidentally share state.
- **时间偶奇** / **Temporal parity**: 中文: 上下采样时间维时，chunk 长度太短会触发额外补帧逻辑。 / English: Very short chunks trigger extra temporal padding logic during up/downsampling.

## 延伸阅读 / Further reading

- [Wan-Video/Wan2.1 source](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vae.py#L66-L160)
