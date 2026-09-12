---
date: 2026-08-17
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/modules/model.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py#L238-L317
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, dit-block]
build_role: dit-block advanced variant
---

# Wan2.1 AttentionBlock：AdaLN 把时间条件塞进 DiT block / Wan2.1 AttentionBlock: AdaLN Injects Time Conditions into the DiT Block

> **一句话 / In one line**: `WanAttentionBlock` 用 6 组 AdaLN 参数分别调制 self-attention、FFN 和残差门控，再接上文本/图像 cross-attention。 / `WanAttentionBlock` uses six AdaLN chunks to modulate self-attention, FFN, and residual gates, then adds text/image cross-attention.

## 为什么重要 / Why this matters

WAM 的 DiT block 不能只是普通 Transformer block。它要知道当前噪声时间步，还要把视频 token 和条件上下文接起来。这里的 `e` 就是时间条件进入每层 block 的入口。

A WAM DiT block cannot be just a plain Transformer block. It must know the current noise timestep and connect video tokens with conditioning context. The `e` tensor here is the entry point for timestep conditioning at every layer.

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/modules/model.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py#L238-L317)

```python
class WanAttentionBlock(nn.Module):
    def __init__(self, cross_attn_type, dim, ffn_dim, num_heads, window_size=(-1, -1), qk_norm=True, cross_attn_norm=False, eps=1e-6):
        super().__init__()
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0], seq_lens, grid_sizes,
            freqs)
        with amp.autocast(dtype=torch.float32):
            x = x + y * e[2]

        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
            with amp.autocast(dtype=torch.float32):
                x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x
```

## 逐行讲解 / What's happening

1. **第 258-276 行 / Lines 258-276**: 中文: block 包含 self-attn、cross-attn、FFN 和一张可学习的 `modulation` 表。 / English: The block contains self-attention, cross-attention, an FFN, and a learned `modulation` table.
2. **第 296-299 行 / Lines 296-299**: 中文: 外部时间条件 `e` 加上每层自己的 modulation，再拆成 6 份。 / English: The external timestep condition `e` is added to per-layer modulation and split into six chunks.
3. **第 301-307 行 / Lines 301-307**: 中文: 前三份分别是 self-attn 的 shift、scale 和 residual gate。 / English: The first three chunks are shift, scale, and residual gate for self-attention.
4. **第 309-316 行 / Lines 309-316**: 中文: cross-attention 先混入条件上下文，后两组 shift/scale/gate 再调制 FFN。 / English: Cross-attention first mixes conditioning context, then the remaining shift/scale/gate chunks modulate the FFN.

## 类比 / The analogy

像调音台上的六个旋钮：两个调 self-attention 的音色，一个管它进主声道的音量；另外两个调 FFN，一个管 FFN 的音量。每个时间步都给这六个旋钮不同位置。

It is like six knobs on a mixing console: two shape self-attention, one controls how much enters the main track; two shape the FFN, and one controls its volume. Each timestep sets those knobs differently.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-{VLA,WAM}

在 nanoWAM 里，这是 `dit-block` 的核心。VAE latent patch 进来形成 `x`，noise scheduler 给出 timestep embedding `e`，文本、图像或动作条件形成 `context`。如果省掉 AdaLN，模型仍能做 attention，但去噪轨迹不再按时间步改变行为。

In a nanoWAM, this is the core `dit-block`. VAE latent patches become `x`, the noise scheduler provides timestep embedding `e`, and text, image, or action conditions become `context`. Without AdaLN, the model can still attend, but its denoising behavior is no longer shaped by timestep.

## 自己跑一遍 / Try it yourself

```python
def toy_block(x, e):
    shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = e
    attn_in = x * (1 + scale_a) + shift_a
    x = x + attn_in * gate_a
    ffn_in = x * (1 + scale_f) + shift_f
    return round(x + (ffn_in * 2) * gate_f, 3)

print(toy_block(1.0, [0.1, 0.0, 0.5, 0.0, 0.0, 0.25]))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
2.325
```

同一个输入 `x`，只要改变 `e` 的 gate 或 scale，block 的残差贡献就会随时间步变化。

For the same input `x`, changing the gates or scales in `e` changes how much each residual branch contributes at that timestep.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **DiT adaLN-Zero** / **DiT adaLN-Zero**: 同样用条件向量生成 shift、scale 和 gate 来调制 Transformer block。 / It also uses condition-derived shift, scale, and gate values to modulate Transformer blocks.
- **Open-Sora MMDiT blocks** / **Open-Sora MMDiT blocks**: 视频 DiT 也把时间条件和 cross-modal context 放进每层 block。 / Video DiTs also inject timestep conditioning and cross-modal context inside each block.

## 注意事项 / Caveats / when it breaks

- **dtype 很敏感** / **The dtype is sensitive**: 代码显式要求 `e` 是 `float32`，调制参数在低精度下容易影响稳定性。 / The code explicitly requires `e` to be `float32`; modulation parameters can be stability-sensitive in low precision.
- **context 类型由 cross_attn_type 决定** / **Context depends on `cross_attn_type`**: T2V 和 I2V 的上下文组织不同，不能只看 block 本身推断全部输入语义。 / T2V and I2V organize context differently, so the block alone does not define all input semantics.

## 延伸阅读 / Further reading

- [Wan2.1 source](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py#L238-L317)
