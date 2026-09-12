---
date: 2026-07-31
topic: diffusion
source: tracked
repo: Wan-Video/Wan2.1
file: wan/modules/vae.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vae.py#L17-L220
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, video-vae, causal-conv]
---

# Wan2.1 CausalConv3d：视频 VAE 只看过去 / Wan2.1 CausalConv3d: A Video VAE Looks Only Backward

> **一句话 / In one line**: `CausalConv3d` 把时间 padding 全放在过去侧，并在分块推理时接上上一块的缓存帧。 / `CausalConv3d` puts temporal padding on the past side and stitches in cached frames when processing video chunks.

## 为什么重要 / Why this matters

视频 VAE 经常不能一次吃完整长视频，只能按时间块编码或解码。普通 3D conv 会在时间维两侧 padding，导致当前块“偷看”未来；Wan2.1 的这段代码把时间上下文改成单向，并让相邻块通过 `feat_cache` 连起来。

Video VAEs often cannot encode or decode a long clip in one pass. A normal 3D convolution pads both sides of time, which can leak future context into the current chunk. This Wan2.1 code makes the temporal context one-way and connects adjacent chunks through `feat_cache`.

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/modules/vae.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vae.py#L17-L220)

```python
class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolusion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)

        return super().forward(x)

class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1))
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) \
            if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h
```

## 逐行讲解 / What's happening

1. **第 24-26 行 / Lines 24-26 (`self._padding`)**:
   - 中文: 空间维仍左右对称 padding；时间维变成 `(2 * padding_t, 0)`，也就是只在过去侧补帧。
   - English: Spatial padding stays symmetric; temporal padding becomes `(2 * padding_t, 0)`, meaning frames are added only on the past side.
2. **第 30-34 行 / Lines 30-34 (`cache_x`)**:
   - 中文: 如果上一块有缓存，就先拼到当前块前面，再减少需要人工 padding 的过去帧数。
   - English: If a previous chunk is cached, it is prepended to the current chunk, and the amount of artificial past padding is reduced.
3. **第 205-217 行 / Lines 205-217 (residual cache)**:
   - 中文: 每个 `CausalConv3d` 层都有自己的缓存槽；forward 之后立刻把当前块尾部帧写回对应槽位。
   - English: Each `CausalConv3d` layer owns a cache slot; after the layer runs, the tail frames of the current chunk are written back into that slot.

## 类比 / The analogy

这像看连续剧时只带上一集的结尾片段：你能接上剧情，但不会提前看到下一集的剧透。

It is like watching a series with only the ending recap of the previous episode: you regain context without seeing spoilers from the next episode.

## 自己跑一遍 / Try it yourself

```python
def causal_window(chunk, cache=None, pad=2):
    past = [] if cache is None else list(cache)
    x = past + list(chunk)
    need_pad = max(0, pad - len(past))
    x = ["PAD"] * need_pad + x
    new_cache = list(chunk)[-pad:]
    return x, new_cache

cache = None
for chunk in [[1, 2], [3, 4], [5]]:
    window, cache = causal_window(chunk, cache)
    print(window, "cache=", cache)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
['PAD', 'PAD', 1, 2] cache= [1, 2]
[1, 2, 3, 4] cache= [3, 4]
[3, 4, 5] cache= [5]
```

第二块没有再用假 padding，而是用上一块真实帧补足上下文。

The second chunk no longer needs fake padding; it uses real frames from the previous chunk as context.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Open-Sora CausalConv3d** / **Open-Sora CausalConv3d**: 同样把时间 padding 改成只看过去，用在 causal 3D VAE。 / It also makes temporal padding look backward only inside a causal 3D VAE.
- **Streaming KV cache** / **Streaming KV cache**: Transformer 解码缓存上一段 K/V，思想上也是“保留过去，拒绝未来”。 / Transformer decoding caches previous K/V; conceptually it keeps the past and rejects the future.

## 注意事项 / Caveats / when it breaks

- **缓存槽必须对齐层顺序** / **Cache slots must match layer order**: `feat_idx` 错位会把某层的旧帧喂给另一层。 / If `feat_idx` drifts, frames cached for one layer can be fed into another.
- **训练和分块推理语义不同** / **Training and chunked inference differ**: 全片训练时不一定走 `feat_cache`，上线前要专门测 chunk boundary。 / Full-clip training may not use `feat_cache`; chunk boundaries need dedicated tests before deployment.

## 延伸阅读 / Further reading

- [Wan2.1 VAE source permalink](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vae.py#L17-L220)
