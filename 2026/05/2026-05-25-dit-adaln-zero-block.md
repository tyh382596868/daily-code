---
date: 2026-05-25
topic: diffusion
source: tracked
repo: facebookresearch/DiT
file: models.py
permalink: https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L101-L122
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, dit, adaln-zero, conditioning]
---

# DiT's adaLN-Zero block: how to inject conditioning into a Transformer

> **In one line**: Replace every LayerNorm's affine parameters with a tiny MLP that maps the conditioning vector to per-token shift, scale, and gate — then zero-initialize that MLP so each block starts as the identity.

## Why this matters

Diffusion Transformers (DiT) replaced the U-Net in modern image and video generators (Stable Diffusion 3, Sora-style models, Cosmos, Wan, CogVideoX — they all use DiT-style backbones). The interesting question DiT had to answer was: *how do you condition a Transformer on the diffusion timestep and class label?* You can't just concat — a U-Net got away with adding embeddings into convolutional features, but a Transformer's residual stream is supposed to be a clean information highway.

DiT's answer is `adaLN-Zero` — adaptive LayerNorm with zero-initialized modulation. It works so well that nearly every diffusion-transformer paper since 2023 uses it.

## The code

`facebookresearch/DiT` — [`models.py`](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L101-L122)

```python
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
```

And the "Zero" half of `adaLN-Zero`, from the model's `initialize_weights`:

```python
# Zero-out adaLN modulation layers in DiT blocks:
for block in self.blocks:
    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
```

## What's happening

1. **`norm1` is `elementwise_affine=False`** — LayerNorm normalizes but does **not** learn its own gain/bias. The gain and bias come from `c` instead.
2. **`adaLN_modulation(c)`** maps the conditioning vector `c` (timestep embedding + class embedding) through `SiLU → Linear(hidden, 6 × hidden)`. That `6×` is split into six per-token vectors: `shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp`.
3. **`modulate(x, shift, scale)`** is the FiLM trick: `x * (1 + scale) + shift`. The `1 +` matters — it means `scale=0` is a no-op (identity), not a zero-out.
4. **`gate_msa` and `gate_mlp`** multiply each sub-block's *output* before it gets added back to the residual stream. This is the magic gating: if the gate is zero, the sub-block contributes nothing.
5. **Zero-init of `adaLN_modulation[-1]`**: the final Linear of the modulation MLP has weight=0 and bias=0, so at step zero the MLP outputs all zeros for every input. That means `shift=0`, `scale=0`, `gate=0`. Combined with point 3 and point 4: every DiTBlock starts as the **identity function** — `x` passes through unchanged. Training then gently learns how much each block should modify the residual.

The combination of zero gates + zero shift/scale on a deep stack (28 blocks in DiT-XL) is what lets a very deep Transformer be trained stably from scratch without warmup tricks.

## The analogy

Think of a sound mixing console with 28 tracks (one per Transformer block). Each track has three knobs controlled by the "conditioning" engineer: tone shift, tone scale, and volume (the gate). Standard initialization is like turning all the knobs to random positions before pressing play — chaos. **adaLN-Zero starts every knob at its neutral position with volume at zero**: the signal passes through every track unaltered, and the engineer learns track-by-track how much each one should contribute, starting from silence.

That neutral-start is why the network doesn't explode when you suddenly stack 28 conditional blocks.

## Try it yourself

```python
# Minimal DiTBlock you can run in 30 lines.
import torch, torch.nn as nn

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TinyDiTBlock(nn.Module):
    def __init__(self, d=64, heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)
        self.attn  = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp   = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.mod   = nn.Sequential(nn.SiLU(), nn.Linear(d, 6*d))
        nn.init.constant_(self.mod[-1].weight, 0)   # the "Zero" in adaLN-Zero
        nn.init.constant_(self.mod[-1].bias,   0)

    def forward(self, x, c):
        s1, sc1, g1, s2, sc2, g2 = self.mod(c).chunk(6, dim=1)
        a, _ = self.attn(modulate(self.norm1(x), s1, sc1), modulate(self.norm1(x), s1, sc1), modulate(self.norm1(x), s1, sc1))
        x = x + g1.unsqueeze(1) * a
        x = x + g2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), s2, sc2))
        return x

block = TinyDiTBlock()
x = torch.randn(2, 16, 64); c = torch.randn(2, 64)
y = block(x, c)
print("identity at init:", torch.allclose(x, y, atol=1e-6))
print("output shape:", y.shape)
```

Run with:
```bash
pip install torch
python try.py
```

Expected output:
```
identity at init: True
output shape: torch.Size([2, 16, 64])
```

The `True` is the punchline: a freshly-initialized adaLN-Zero block is bit-exactly equal to the identity function. Stack a hundred of them and your model still starts from a sane point.

## Where this pattern shows up elsewhere

- **Stable Diffusion 3 (MMDiT)** — same adaLN-Zero, with two separate conditional streams for image and text.
- **Cosmos, CogVideoX, Wan, Open-Sora** — every modern video DiT uses this conditioning style.
- **ConvNeXt-style image classifiers with class conditioning** — same FiLM-style modulation idea, different backbone.
- **FiLM (Feature-wise Linear Modulation)** in 2017 RL papers — the ancestor: `γ * x + β` injected from a side network. DiT adds the "Zero" trick.
- **Zero-init residuals** — ReZero, Fixup, NormFormer all share the philosophy: start each block as identity, learn how much it should do.

## Caveats / when it breaks

- **`elementwise_affine=False` is load-bearing.** If you forget it, you end up with *two* gains and biases per LayerNorm — the LayerNorm's own affine, plus the modulation. Training is still possible but it muddles the zero-init guarantee.
- **The `1 +` in `modulate`.** Easy to forget. Without it, zero-init scale would *kill* the activations (multiply by 0) instead of leaving them alone.
- **6× hidden size is expensive.** For each block you add roughly `6 × hidden²` parameters just for conditioning. In DiT-XL/2 (`hidden=1152`) this is ~8M params/block × 28 blocks ≈ 220M params *just* for the modulation MLPs. Variants like adaLN-single (Pixart-α) share the modulation across blocks to cut this.
- **Only useful when you have a clean conditioning vector** — for cross-attention with long text prompts, you typically pair adaLN-Zero (for the pooled CLS-style condition) with cross-attention layers (for token-wise text).

## Further reading

- DiT paper: ["Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748), Peebles & Xie, 2023 — Section 3.2 is the adaLN-Zero rationale and the ablation showing it beats in-context and cross-attention conditioning.
- FiLM: ["FiLM: Visual Reasoning with a General Conditioning Layer"](https://arxiv.org/abs/1709.07871) — the original `γ*x + β` paper.
- ["Pixart-α"](https://arxiv.org/abs/2310.00426) for the cheaper adaLN-single variant.
- The DiT code itself is wonderfully short — [`models.py`](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py) is one file, 370 lines, the whole model.
