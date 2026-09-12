---
date: 2026-07-20
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/modules/model.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py#L268-L323
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, wan21, dit-block, adaln]
build_role: dit-block advanced variant
---

# Wan2.1 block 调制：一个条件向量拆成 6 个门控 / Wan2.1 Block Modulation: Split One Condition into Six Gates

> **一句话 / In one line**: `WanAttentionBlock` 把 timestep 条件和可学习偏置相加后切成 6 份，分别控制 self-attn、FFN 的 shift、scale 和 residual gate。 / `WanAttentionBlock` adds timestep conditioning to a learned offset, chunks it into six pieces, and uses them as shift, scale, and residual gates for self-attention and FFN.

## 为什么重要 / Why this matters

扩散 DiT block 不只是“attention + MLP”。噪声时间步、文本条件和视频 token 必须影响每一层的归一化和残差强度。Wan2.1 用生产级 adaLN 风格调制：条件向量不直接加到 token 上，而是改变每层怎么归一化、怎么放大残差。

A diffusion DiT block is not just attention plus MLP. Noise timestep, text condition, and video tokens must influence normalization and residual strength at every layer. Wan2.1 uses production-style adaLN modulation: the condition is not simply added to tokens, but controls how each layer normalizes and gates residuals.

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/modules/model.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py#L268-L323)

```python
# modulation
self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

def forward(
    self,
    x,
    e,
    seq_lens,
    grid_sizes,
    freqs,
    context,
    context_lens,
):
    assert e.dtype == torch.float32
    with amp.autocast(dtype=torch.float32):
        e = (self.modulation + e).chunk(6, dim=1)
    assert e[0].dtype == torch.float32

    # self-attention
    y = self.self_attn(
        self.norm1(x).float() * (1 + e[1]) + e[0], seq_lens, grid_sizes,
        freqs)
    with amp.autocast(dtype=torch.float32):
        x = x + y * e[2]

    # cross-attention & ffn function
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

1. **每层有可学习默认调制 / Each layer has learned default modulation**: 中文: `self.modulation` 是 block 自己的偏置，不同层可以有不同初始门控。 English: `self.modulation` is a block-local offset, so different layers can start with different gates.
2. **条件切成 6 份 / Condition is chunked into six parts**: 中文: self-attn 用 shift/scale/gate 三份，FFN 再用三份。 English: self-attention gets shift, scale, and gate; the FFN gets another three.
3. **先调 norm 后的 token / Modulate normalized tokens first**: 中文: `norm1(x) * (1 + e[1]) + e[0]` 是 adaLN 的核心。 English: `norm1(x) * (1 + e[1]) + e[0]` is the core adaLN move.
4. **残差也被门控 / Residuals are gated too**: 中文: `x + y * e[2]` 让条件决定这一层写回多少。 English: `x + y * e[2]` lets the condition decide how much the layer writes back.
5. **cross-attn 不走同一套门 / Cross-attention is separate**: 中文: 文本/图像 context 先通过 cross-attn 注入，再由 FFN 调制。 English: text or image context enters through cross-attention, then FFN modulation follows.

## 类比 / The analogy

像调音台。音轨本身是 token，条件向量不是另一条音轨，而是一排旋钮：先调 EQ，再调音量推子，决定这一层混进多少声音。

It is like a mixing console. Tokens are the audio tracks; the condition is not another track but a row of knobs: adjust EQ, then volume faders, deciding how much this layer contributes.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

这是 `dit-block` 的高级实现，依赖 `noise-scheduler` 的 timestep embedding、`patchify-positional` 的视频 token 和 `text-conditioning` 的 context。nanoWAM 最小版可以先写一个 `AdaLNBlock(x, cond)`：把 `cond` 投影成 6 个向量，分别给 attention 和 MLP 做 shift/scale/gate。生产版还要处理 fp32 调制、变长视频 token、cross-attention context 和 mixed precision。

This is an advanced `dit-block`, depending on timestep embeddings from `noise-scheduler`, video tokens from `patchify-positional`, and context from `text-conditioning`. A minimal nanoWAM can start with an `AdaLNBlock(x, cond)` that projects `cond` into six vectors for attention and MLP shift/scale/gate. A production version also needs fp32 modulation, variable-length video tokens, cross-attention context, and mixed precision handling.

## 自己跑一遍 / Try it yourself

```python
x = [1.0, 2.0, 3.0]
shift, scale, gate = 0.5, 0.1, 0.25

def norm(v):
    m = sum(v) / len(v)
    return [round(a - m, 3) for a in v]

h = [round(a * (1 + scale) + shift, 3) for a in norm(x)]
y = [round(a + gate * b, 3) for a, b in zip(x, h)]
print(h)
print(y)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[-0.6, 0.5, 1.6]
[0.85, 2.125, 3.4]
```

同一个 hidden state，因为 shift/scale/gate 不同，会写回不同强度的残差。

The same hidden state writes back different residual strength depending on shift, scale, and gate.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **DiT adaLN-Zero** / **DiT adaLN-Zero**: 把 timestep/class 条件变成每层的 shift、scale、gate。 / Timestep and class conditions become per-layer shift, scale, and gate.
- **PixArt / SD3 DiT blocks** / **PixArt / SD3 DiT blocks**: 文本条件和时间条件也常通过 adaLN 调制 transformer block。 / Text and time conditions often modulate transformer blocks through adaLN.

## 注意事项 / Caveats / when it breaks

- **调制保持 fp32 / Keep modulation in fp32**: Wan2.1 显式 `autocast(dtype=torch.float32)`，避免门控数值太脆。 / Wan2.1 explicitly uses fp32 autocast to keep gates numerically stable.
- **gate 初始化很关键 / Gate initialization matters**: 残差门太大，扩散训练早期会不稳。 / Large residual gates can destabilize early diffusion training.
- **条件维度要对齐 / Condition dimensions must align**: `e` 必须能 chunk 成 `[B, 6, dim]`。 / `e` must chunk cleanly into `[B, 6, dim]`.

## 延伸阅读 / Further reading

- [Wan2.1 model.py](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py#L268-L323)
- [Wan2.1 repository](https://github.com/Wan-Video/Wan2.1)
