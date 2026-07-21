---
date: 2026-07-03
topic: diffusion
source: tracked
repo: facebookresearch/DiT
file: models.py
permalink: https://github.com/facebookresearch/DiT/blob/main/models.py#L101-L122
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, dit, adaln-zero]
---

# DiT adaLN-Zero：条件不是加进去，而是调制整层 / DiT adaLN-Zero: Conditioning by Modulating the Whole Layer

> **一句话 / In one line**: `DiTBlock` 把 timestep/class 条件变成 6 个向量，分别控制 attention 和 MLP 的 shift、scale、gate。 / `DiTBlock` turns timestep/class conditioning into six vectors that shift, scale, and gate the attention and MLP paths.

## 为什么重要 / Why this matters

扩散 Transformer 不能只看图像 token；每一层都必须知道“现在是哪个噪声步、用什么条件生成”。DiT 的做法很干净：先做无仿射参数的 LayerNorm，再用条件向量产生整层的归一化偏移、缩放和残差门控。

A diffusion Transformer cannot only read image tokens; every layer needs to know the current noise step and conditioning signal. DiT keeps this clean: use LayerNorm without affine parameters, then let the conditioning vector produce shift, scale, and residual gates for the whole block.

## 代码 / The code

`facebookresearch/DiT` — [`models.py`](https://github.com/facebookresearch/DiT/blob/main/models.py#L101-L122)

```python
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

## 逐行讲解 / What's happening

1. **LayerNorm 不带 affine / LayerNorm has no affine**: 中文: `elementwise_affine=False` 把默认的可学习缩放/偏移拿掉，让条件向量接管这件事。 / English: `elementwise_affine=False` removes the default learned scale and bias so the conditioning vector owns that job.
2. **一个 MLP 产 6 份控制量 / One MLP emits six controls**: 中文: `6 * hidden_size` 被拆成 attention 的 shift/scale/gate 和 MLP 的 shift/scale/gate。 / English: `6 * hidden_size` splits into shift/scale/gate for attention and shift/scale/gate for the MLP.
3. **gate 控制残差强度 / Gates control residual strength**: 中文: attention 和 MLP 的输出不是直接加回去，而是先乘以对应 gate。 / English: Attention and MLP outputs are not added directly; each residual path is multiplied by its gate first.

## 类比 / The analogy

这像调音台：原始 token 是乐队，attention 和 MLP 是两组音轨，条件向量不是再加一个乐手，而是在每一层调音量、均衡和开关。

It is like a mixing console. Tokens are the band, attention and MLP are two tracks, and the conditioning vector is not another musician; it adjusts level, EQ, and mute switches at every layer.

## 自己跑一遍 / Try it yourself

```python
import torch
import torch.nn as nn

def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]

B, T, D = 2, 4, 8
x = torch.randn(B, T, D)
c = torch.randn(B, D)
ada = nn.Sequential(nn.SiLU(), nn.Linear(D, 6 * D))
shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = ada(c).chunk(6, dim=1)
attn_like = modulate(nn.LayerNorm(D, elementwise_affine=False)(x), shift_a, scale_a)
out = x + gate_a[:, None] * attn_like
print(out.shape, gate_a.shape)
```

运行 / Run with:

```bash
pip install torch
python try.py
```

预期输出 / Expected output:

```text
torch.Size([2, 4, 8]) torch.Size([2, 8])
```

注意 gate 是 batch 级、token 共享的控制量：同一个样本里的所有 patch token 都接收同一份 timestep/class 调制。

The gate is batch-level and shared across tokens: all patch tokens in one sample receive the same timestep/class modulation.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Wan2.1 的 6 向量 adaLN** / **Wan2.1 six-vector adaLN**: 视频 DiT 也常把条件压成多份 shift/scale/gate。
- **PixArt/SD3 风格 DiT** / **PixArt/SD3-style DiTs**: 同样把条件注入做成每层归一化调制，而不是只拼 token。

## 注意事项 / Caveats / when it breaks

- **条件维度必须对齐** / **Condition width must match**: `c` 的 hidden size 必须和 block hidden size 一致，否则 `chunk(6)` 后无法调制 token。
- **gate 初始化很关键** / **Gate initialization matters**: DiT 论文里的 Zero 初始化让模型一开始接近恒等映射；随便初始化可能让训练早期不稳。

## 延伸阅读 / Further reading

- [DiT repository](https://github.com/facebookresearch/DiT)
- [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
