---
date: 2026-06-08
topic: infrastructure
source: tracked
repo: pytorch/torchtune
file: torchtune/modules/peft/dora.py
permalink: https://github.com/pytorch/torchtune/blob/bd2a0fc7c31430972728494fa01aaeeb0ebf1ba1/torchtune/modules/peft/dora.py#L156-L192
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, infrastructure, peft, dora, lora]
---

# DoRA 的 forward 就是一句话:用 (magnitude / ||W + BA||) 重新归一化每一列 / DoRA's whole forward is "renormalize each column by magnitude / ||W + BA||"

> **一句话 / In one line**: DoRA 把权重拆成「方向」和「幅度」两部分,LoRA 改方向,一个可学的标量向量 m 改幅度,forward 里只多了一句 `mag_norm_scale = m / ||W + scale·BA||`。 / DoRA splits a pretrained weight into "direction" and "magnitude": LoRA changes the direction, a learnable scalar vector `m` rescales each output column, and the whole extra cost in forward is one line: `mag_norm_scale = m / ||W + scale·BA||`.

## 为什么重要 / Why this matters

LoRA 已经是 fine-tuning 的事实标准,但 LoRA 论文里漏掉了一个细节:微调 W 时不仅"方向"会变,"幅度"也会变 (向量的范数会偏移)。DoRA 论文 (Liu et al., 2024) 把这件事变成显式参数化:把 `W` 写成 `m · (W / ||W||)`,其中 `m` 是每个输出通道一个的可学标量。这样 LoRA 的 `BA` 只贡献方向,`m` 单独承担幅度变化。torchtune 这版 `DoRALinear` 用 30 行 forward 把整套数学落地,而且和 FSDP / NF4 量化兼容——这是真实生产代码里你能看到的最干净的 DoRA 实现。

LoRA is the de-facto fine-tuning standard, but the original paper quietly conflates two things: fine-tuning `W` changes both its *direction* and its *magnitude* (the per-column norm). DoRA (Liu et al., 2024) makes that explicit by re-parameterizing `W = m · (W / ||W||)`, where `m` is one learnable scalar per output channel. LoRA's `BA` only contributes direction; `m` carries the magnitude. torchtune's `DoRALinear` boils the whole derivation down to a 30-line `forward`, all while remaining FSDP- and NF4-compatible — the cleanest production DoRA implementation I've seen.

## 代码 / The code

`pytorch/torchtune` — [`torchtune/modules/peft/dora.py`](https://github.com/pytorch/torchtune/blob/bd2a0fc7c31430972728494fa01aaeeb0ebf1ba1/torchtune/modules/peft/dora.py#L156-L192)

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x (torch.Tensor): input tensor with shape ``(..., in_dim)``

    Returns:
        Tensor: output tensor with shape ``(..., out_dim)``
    """
    if self._quantize_base:
        base_out = linear_nf4(input=x, weight=self.weight)
        if self.use_bias:
            base_out = base_out + self.bias
    else:
        base_out = F.linear(x, self.weight, self.bias)
    if self.disabled:
        return base_out

    x = self.dropout(x)

    lora_out = self.lora_b(self.lora_a(x))
    # Can't use raw matmul since FSDP hooks are attached to __call__
    # Instead follow the approach in https://github.com/huggingface/peft/pull/1806
    x_eye = torch.eye(
        self.lora_a.weight.shape[1], device=self.lora_a.weight.device, dtype=x.dtype
    )
    lora_weight = self.lora_b(self.lora_a(x_eye)).T
    magnitude = self.magnitude
    weight = self.weight.to(x.dtype)
    weight_norm = self._get_weight_norm(weight, lora_weight.detach())
    weight_norm = weight_norm.detach()
    mag_norm_scale = (magnitude / weight_norm).view(1, -1)

    dora_out = (
        mag_norm_scale - 1
    ) * base_out + mag_norm_scale * lora_out * self.scaling

    return dora_out + base_out
```

(辅助函数 / Helper:)

```python
def _get_weight_norm(self, weight, lora_weight):
    weight = weight + self.scaling * lora_weight
    weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
    return weight_norm
```

## 逐行讲解 / What's happening

1. **`base_out = F.linear(x, self.weight, ...)`**
   - 中文: 先正常算 base output。NF4 量化分支用 `linear_nf4` 走自定义反量化路径,但接口对外完全一样。
   - English: Compute the plain base output first. The NF4-quantized branch goes through `linear_nf4` (a custom dequant-on-the-fly kernel), but it presents the same `(x, W) → out` interface as `F.linear`.

2. **`lora_out = self.lora_b(self.lora_a(x))`**
   - 中文: 这是经典 LoRA 路径:`x → A → B → low-rank ΔW·x`。注意是用 `self.lora_b(...)` 而不是 `lora_b.weight @ ...`——下文会解释为什么。
   - English: Classic LoRA: `x → A → B → low-rank ΔW·x`. Note the call uses `self.lora_b(...)`, not a raw matmul. The reason becomes clear two lines later.

3. **`x_eye = torch.eye(...); lora_weight = self.lora_b(self.lora_a(x_eye)).T`**
   - 中文: 这一步是把 `BA` 矩阵实例化出来——通过给 lora_a / lora_b 输入单位矩阵,`B·A·I = BA`,转置后形状对齐 `(out_dim, in_dim)`。
   - English: This reconstructs the explicit `BA` matrix by passing an identity through the LoRA path: `B · A · I = BA`, then transpose to align `(out_dim, in_dim)` with `self.weight`. Reconstructing it as `lora_b.weight @ lora_a.weight` would be cheaper but bypasses FSDP's `__call__`-hooked all-gather, so the activations would not get sharded gradients. Going through the module call is the FSDP-safe path.

4. **`weight_norm = self._get_weight_norm(weight, lora_weight.detach())`**
   - 中文: 计算「更新后权重的列范数」: `||W + scaling·BA||_2 (dim=1)`。`lora_weight.detach()` 是关键——梯度只通过 `lora_a / lora_b / magnitude` 流回去,而不会通过范数自身的梯度跑回 LoRA 参数 (这是 DoRA 论文里证明能稳定训练的细节)。
   - English: Compute the row norm of the *updated* weight `||W + scaling · BA||₂` along `dim=1` (one norm per output channel). The `.detach()` on `lora_weight` is the critical bit DoRA proves keeps training stable — gradients reach `lora_a/lora_b/magnitude` directly, not through the norm's own derivative. Without that detach the norm gradient creates a feedback loop that ruins convergence.

5. **`mag_norm_scale = (magnitude / weight_norm).view(1, -1)`**
   - 中文: 每个输出通道一个的标量 `m / ||·||`。如果 `m` 初始化得正好等于 `||W||`,这个 scale 就是 1,DoRA 退化成 LoRA——这就是 `initialize_dora_magnitude()` 的作用。
   - English: One scalar per output channel: `m / ||·||`. At init time, `initialize_dora_magnitude()` sets `m = ||W||` so that `mag_norm_scale = 1` and DoRA reduces exactly to LoRA. Training then nudges `m` away from `||W||`.

6. **`dora_out = (mag_norm_scale - 1) * base_out + mag_norm_scale * lora_out * self.scaling`**
   - 中文: 看起来吓人,但展开就是 `mag_norm_scale * (base_out + scaling · lora_out) - base_out`。最后一行 `return dora_out + base_out` 把 `-base_out` 抵消,得到 `mag_norm_scale * (W + scaling · BA) · x` ——这正是 DoRA 论文里的 `m · (W + scaling · BA) / ||W + scaling · BA|| · x`。
   - English: Looks scary, but expand: `mag_norm_scale * (base_out + scaling · lora_out) − base_out`. The trailing `return dora_out + base_out` cancels the `−base_out` and yields `mag_norm_scale * (W + scaling · BA) · x`, which *is* the DoRA paper's `m · (W + scaling · BA) / ||W + scaling · BA|| · x`. The `(scale − 1) · base_out` form is just an algebraic rewrite that avoids materializing the updated weight matrix in the forward.

## 类比 / The analogy

想象一个调音师面前有 N 根弦 (N = `out_dim`)。LoRA 微调改的是每根弦的"指法" (方向),DoRA 多给了一个旋钮:每根弦的张力 (magnitude)。`weight_norm` 是当前弦的物理张力,`m` 是你希望的张力——`m / weight_norm` 就是把这根弦拧到目标张力需要乘的系数。你不必拆掉整把吉他重新做一根弦,只要轻轻拧一下钮就能让每根弦既换指法又换张力。

Picture a piano tuner standing in front of `out_dim` strings. LoRA's fine-tuning changes *how you fret* each string (the direction). DoRA hands you one more knob per string: the *tension* (magnitude). `weight_norm` is the current physical tension, `m` is the tension you want — `m / weight_norm` is just the ratio to multiply by to get there. You don't have to rebuild the string; you turn one tuning peg, and the string now sounds at both a new pitch and a new volume.

## 自己跑一遍 / Try it yourself

```python
# pip install torch
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyDoRA(nn.Module):
    def __init__(self, in_dim, out_dim, rank=4, alpha=8):
        super().__init__()
        self.W = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1, requires_grad=False)
        self.A = nn.Linear(in_dim, rank, bias=False)
        self.B = nn.Linear(rank, out_dim, bias=False)
        nn.init.zeros_(self.B.weight)
        self.m = nn.Parameter(self.W.norm(dim=1))  # init so DoRA ≡ LoRA at step 0
        self.scaling = alpha / rank

    def forward(self, x):
        base = F.linear(x, self.W)
        lora = self.B(self.A(x))
        BA = self.B.weight @ self.A.weight              # (out, in)
        wnorm = (self.W + self.scaling * BA).norm(dim=1).detach()
        scale = (self.m / wnorm).view(1, -1)
        return scale * (base + self.scaling * lora)

layer = TinyDoRA(16, 8)
x = torch.randn(2, 16)
print("init  ", layer(x).std().item(), "≈ pure base?", torch.allclose(layer(x), F.linear(x, layer.W), atol=1e-5))
with torch.no_grad():
    layer.m += 0.5  # bump magnitudes
print("bumped", layer(x).std().item())
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
init   <some-std> ≈ pure base? True
bumped <larger-std>
```

中文:初始时 `m = ||W||` 让 DoRA 等价于 base linear,微调 `m` 之后每个输出通道的幅度就会立刻变化——这就是 DoRA 给 LoRA 增加的那一个自由度。

English: At init `m = ||W||` makes DoRA equivalent to the base linear (the assert prints `True`). After we bump `m`, the per-channel output magnitude shifts immediately — that's the one extra degree of freedom DoRA gives LoRA.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **`huggingface/peft` 的 `DoraLinear`** / **HF PEFT's `DoraLinear`**: 几乎相同的实现,torchtune 这版的 detach 注释直接指向 PEFT PR #1806。 / Nearly identical implementation — torchtune's detach comment points straight to PEFT PR #1806.
- **微软 LoRA repo 的 RSLoRA / VeRA** / **MS LoRA's RSLoRA / VeRA**: 同样是「LoRA + 一个额外标量」的家族,只是标量的接法不同 (RSLoRA 用 `α/sqrt(r)`,VeRA 用共享随机投影 + 可学缩放)。 / Same "LoRA + one extra scalar" family — RSLoRA puts the scalar in `α/sqrt(r)`, VeRA scales a shared random projection.
- **Stable Diffusion 的 LoCon / LoHa** / **SD's LoCon / LoHa**: 在生成模型里也会用类似的「分解 + 重归一化」技巧,只是分解方式换成了 Hadamard / Kronecker。 / The same "decompose + renormalize" trick appears in image-gen, just with Hadamard/Kronecker factorizations instead of magnitude/direction.

## 注意事项 / Caveats / when it breaks

- **必须 detach `lora_weight` 和 `weight_norm`** / **You MUST detach `lora_weight` and `weight_norm`**: 漏掉任一个 detach,DoRA 训练会因为范数对 LoRA 参数的二阶反馈而发散——这就是 DoRA 论文 §3.2 给的稳定性证明的核心。 / Skip either detach and DoRA training diverges because the norm's gradient feeds back into LoRA — this is the core of DoRA paper §3.2's stability argument.
- **`initialize_dora_magnitude()` 必须在 base + LoRA 都加载后调用** / **Call `initialize_dora_magnitude()` only after base + LoRA are materialized**: 在 meta device 上直接调用会抛 `RuntimeError`(torchtune 代码里有显式检查)。 / Calling it on a meta-device weight will raise `RuntimeError` — torchtune explicitly checks `is_meta`.
- **NF4 + DoRA 还要再小心一点** / **NF4 + DoRA needs extra care**: `base_out` 走 `linear_nf4` 反量化,但 `weight.to(x.dtype)` 会把 NF4 权重 dequant 一次以算 `weight_norm`——这是一次显式的全 dtype roundtrip,显存吃得比纯 LoRA 多一点。 / `base_out` uses `linear_nf4`'s dequant, but `weight.to(x.dtype)` materializes the full-precision weight again just to compute `weight_norm` — one extra full-dtype roundtrip per forward, costing more VRAM than plain QLoRA.

## 延伸阅读 / Further reading

- [DoRA: Weight-Decomposed Low-Rank Adaptation (Liu et al., 2024)](https://arxiv.org/abs/2402.09353)
- [HF PEFT PR #1806 — the FSDP-safe `x_eye` trick](https://github.com/huggingface/peft/pull/1806)
- [torchtune DoRA tutorial](https://pytorch.org/torchtune/main/tutorials/dora.html)
