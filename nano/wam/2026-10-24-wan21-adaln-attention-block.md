---
date: 2026-10-24
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/modules/model.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py#L238-L317
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, dit-block, adaln]
build_role: dit-block
---

# Wan2.1 AdaLN DiT block：六个门控制三段残差 / Wan2.1 AdaLN DiT Block: Six Gates Control Three Residual Stages

> **一句话 / In one line**: Wan 的 timestep 条件拆成六份，分别调制 self-attention 和 FFN 的 shift、scale、gate。 / Wan splits timestep conditioning into six pieces that separately shift, scale, and gate self-attention and FFN.

## 为什么重要 / Why this matters

视频 DiT 不能只把时间步加到 token 上；不同噪声等级应改变“看什么、放大什么、写回多少”。这里把这种控制写成 AdaLN 和门控残差。

A video DiT needs more than adding a timestep to tokens; noise level should change what it attends to, amplifies, and writes back. This block expresses that control through AdaLN and gated residuals.

## 代码 / The code

`Wan-Video/Wan2.1` — [`model.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py#L296-L317)

```python
with amp.autocast(dtype=torch.float32):
    e = (self.modulation + e).chunk(6, dim=1)

y = self.self_attn(
    self.norm1(x).float() * (1 + e[1]) + e[0], seq_lens, grid_sizes, freqs)
with amp.autocast(dtype=torch.float32):
    x = x + y * e[2]

x = x + self.cross_attn(self.norm3(x), context, context_lens)
y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
with amp.autocast(dtype=torch.float32):
    x = x + y * e[5]
```

## 逐行讲解 / What's happening

`chunk(6)` 产生 self-attention 的 shift、scale、gate 与 FFN 的 shift、scale、gate。条件先调制归一化 token，再乘 gate 写回 residual；cross-attention 使用文本或图像 context，但仍由后续 FFN 的条件调制收束。

`chunk(6)` produces shift, scale, and gate for self-attention, then the same trio for the FFN. Conditioning modulates normalized tokens first and gates the residual write-back; cross-attention consumes text or image context before the conditioned FFN completes the block.

## 类比 / The analogy

像录音棚的六个旋钮：两组旋钮分别调均衡和音量，最后各有一个推子决定这一轨实际混入多少。

It is a studio console with six knobs: two groups shape tone and level, with a fader deciding how much of each track reaches the final mix.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

这是 `dit-block` 的 cross-repo advanced variant。patchified video token 与 timestep embedding、text/image context 在这里融合，输出送往下一层或解码头。最小 nanoWAM 可以只保留单次 AdaLN；生产版需要 mixed precision、RoPE、变长序列和并行注意力。

This is a cross-repo advanced `dit-block` variant. Patchified video tokens meet timestep embedding and text/image context here, then flow to the next block or decode head. A nanoWAM can begin with one AdaLN; production needs mixed precision, RoPE, variable lengths, and parallel attention.

## 自己跑一遍 / Try it yourself

```python
import torch
x, shift, scale, gate = torch.ones(1, 2), torch.tensor([[.1, .1]]), torch.tensor([[.2, .2]]), torch.tensor([[.5, .5]])
branch = x * (1 + scale) + shift
print(x + branch * gate)
```

运行 / Run with: `pip install torch && python try.py`

预期输出 / Expected output: `tensor([[1.6500, 1.6500]])`。 / `tensor([[1.6500, 1.6500]])`.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **DiT AdaLN-Zero** / **DiT AdaLN-Zero**: 同样用时间条件调制残差分支。 / It also uses timestep conditioning to modulate residual branches.

## 注意事项 / Caveats / when it breaks

- **gate 的初始化很关键** / **Gate initialization matters**: 过强的初始门会让深层扩散训练不稳定。 / Gates initialized too strongly can destabilize deep diffusion training.

## 延伸阅读 / Further reading

- [Wan2.1 attention block](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py#L238-L317)
