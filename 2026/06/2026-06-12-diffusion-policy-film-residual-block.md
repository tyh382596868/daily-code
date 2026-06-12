---
date: 2026-06-12
topic: robotics
source: tracked
repo: real-stanford/diffusion_policy
file: diffusion_policy/model/diffusion/conditional_unet1d.py
permalink: https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/model/diffusion/conditional_unet1d.py#L14-L66
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, FiLM, diffusion-policy, conditional-residual]
---

# 53 行的 FiLM 残差块 —— Diffusion Policy 的"条件注入"全在这 / 53 lines of FiLM residual block — the whole "conditional injection" of Diffusion Policy lives here

> **一句话 / In one line**: 把 (timestep + observation) 投影成"每通道的 scale / bias",用一次乘法+加法就把条件信息塞进一个 1D 卷积残差块,这就是 Diffusion Policy 整条 U-Net 的搭积木方式 / Project (timestep + observation) into per-channel scale & bias, then a single multiply-add injects the condition into a 1D conv residual block — and that is exactly how the entire Diffusion Policy U-Net is assembled.

## 为什么重要 / Why this matters

2023 年 Cheng Chi 那篇 *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion* 是把"扩散模型"这条线引入机器人控制的转折点。后面 lerobot / openpi / GR00T 全都借用了同样的思想:动作序列被当作"图像",观测当作"条件",一个时间步索引一个 sigma —— 整条 U-Net 就是把动作 chunk 一层层去噪。这个 `ConditionalResidualBlock1D` 就是那条 U-Net 里被反复堆叠的基本单元,53 行写得干净到可以直接背下来。

The 2023 *Diffusion Policy* paper from Cheng Chi et al. is the moment the "diffusion model" line crossed over into robot control. Everything downstream — lerobot, openpi, GR00T — borrows the same recipe: treat the action sequence as an "image", treat the observation as a condition, index one sigma per timestep, and stack a U-Net that denoises the action chunk layer by layer. `ConditionalResidualBlock1D` is the basic unit that gets repeated all the way through that U-Net, and at 53 lines it is short enough to memorize.

## 代码 / The code

`real-stanford/diffusion_policy` — [`diffusion_policy/model/diffusion/conditional_unet1d.py`](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/model/diffusion/conditional_unet1d.py#L14-L66)

```python
class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            cond_dim,
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=False):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1'),
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:,0,...]
            bias = embed[:,1,...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out
```

## 逐行讲解 / What's happening

1. **第 24-27 行 / Lines 24-27 (`self.blocks`)**:
   - 中文: 两个 `Conv1dBlock` 串起来 —— 这俩里头是 `Conv1d → GroupNorm → Mish`。第一块负责把通道数从 `in_channels` 变到 `out_channels`,第二块在 `out_channels` 上继续做一次卷积细化。
   - English: Two `Conv1dBlock` layers in series — each one is `Conv1d → GroupNorm → Mish`. The first one changes the channel count from `in_channels` to `out_channels`; the second refines on the same channel width.

2. **第 31-40 行 / Lines 31-40 (`cond_encoder`)**:
   - 中文: 条件编码器把 `cond` 向量过一个 `Mish → Linear`,输出维度是 `out_channels` 或 `out_channels * 2`(取决于要不要预测 scale)。最后 `Rearrange('batch t -> batch t 1')` 在末尾加了一根长度为 1 的 horizon 维,这样后面广播到 `[B, C, horizon]` 就直接对齐了。
   - English: The conditioner runs `cond` through `Mish → Linear`, with output width `out_channels` or `out_channels * 2` (depending on whether scale is predicted). The final `Rearrange('batch t -> batch t 1')` adds a length-1 horizon dim so it broadcasts cleanly against `[B, C, horizon]`.

3. **第 42-44 行 / Lines 42-44 (`residual_conv`)**:
   - 中文: 残差捷径。如果 `in_channels == out_channels`,直接 `Identity`;否则用一个 1×1 conv 升降维。这是每个 ResNet 风格的 block 都得有的"维度对齐"补丁。
   - English: The residual shortcut. If `in_channels == out_channels`, use `Identity`; otherwise a 1×1 conv re-shapes the channels. Every ResNet-style block needs this "dimension align" patch.

4. **第 54-55 行 / Lines 54-55 (`out = blocks[0](x); embed = cond_encoder(cond)`)**:
   - 中文: 先卷一层得到 `out: [B, out_channels, horizon]`,同时把条件编码成 `embed`。注意条件编码完全独立于 `x`,所以这两步可以并行算。
   - English: First conv produces `out: [B, out_channels, horizon]`; in parallel the condition gets encoded into `embed`. Because the conditioner doesn't depend on `x`, the two computations are independent.

5. **第 56-61 行 / Lines 56-61 (FiLM modulation)**:
   - 中文: 这就是 **FiLM** (Feature-wise Linear Modulation) 的原始公式 —— 把 embed 切成 `[scale, bias]` 两份,然后 `out = scale * out + bias`。每个通道、每条样本拿到一对独立的 `(scale, bias)`,horizon 维上广播。`cond_predict_scale=False` 时退化成纯加法 (`out = out + embed`),省掉 scale 那条腿但少了一半表达力。
   - English: This is the original **FiLM** (Feature-wise Linear Modulation) formula — split `embed` into `[scale, bias]` and do `out = scale * out + bias`. Each channel and each sample gets its own `(scale, bias)` pair, broadcast over horizon. With `cond_predict_scale=False` this degenerates into pure addition (`out = out + embed`), which is simpler but cuts the conditioning capacity in half.

6. **第 64-65 行 / Lines 64-65 (second block + residual)**:
   - 中文: 第二个卷积块在已经 modulate 过的特征上再卷一次,然后加回 `residual_conv(x)`。注意残差是从原始 `x` 来的,**不是**从被 modulate 过的中间结果 —— 这样梯度永远有一条无条件路径流回去。
   - English: The second conv block refines the already-modulated features, then adds back `residual_conv(x)`. The residual comes from the original `x` — **not** the modulated mid-result — so gradient flow always has an unconditional path back.

## 类比 / The analogy

想象你在调一台老式收音机:`out = blocks[0](x)` 是你接收到的原始电波,`scale * out + bias` 就是收音机面板上的两个旋钮 —— scale 控制音量(每个频段独立调),bias 控制电台漂移。FiLM 的精髓就是:不让条件信号自己"广播",而是让它去**调你已有的特征的旋钮**。这样训练时,即使条件向量 `cond` 是高维的,它也只需要影响"调旋钮"这件事,核心特征通道还是由卷积自己学。

Picture an old radio tuner: `out = blocks[0](x)` is the raw waveform you've picked up, and `scale * out + bias` is the pair of knobs on the front panel — `scale` controls volume per band, `bias` controls station drift. The trick of FiLM is not to let the condition broadcast its own signal, but to let it **turn the knobs on features you already have**. So even when the condition vector `cond` is high-dimensional, all it has to do is adjust the knobs; the core feature channels are still learned by the convs themselves.

## 自己跑一遍 / Try it yourself

```python
import torch, torch.nn as nn
from einops.layers.torch import Rearrange

class TinyFiLMBlock(nn.Module):
    def __init__(self, in_c, out_c, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, 3, padding=1)
        self.conv2 = nn.Conv1d(out_c, out_c, 3, padding=1)
        self.cond_enc = nn.Sequential(
            nn.Mish(), nn.Linear(cond_dim, 2 * out_c),
            Rearrange('b t -> b t 1'))
        self.shortcut = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
    def forward(self, x, cond):
        h = self.conv1(x)
        scale, bias = self.cond_enc(cond).reshape(x.shape[0], 2, -1, 1).unbind(dim=1)
        h = scale * h + bias            # FiLM lives here, one line
        h = self.conv2(h)
        return h + self.shortcut(x)

blk = TinyFiLMBlock(in_c=4, out_c=16, cond_dim=8)
x   = torch.randn(2, 4, 32)             # batch=2, channels=4, horizon=32
cond= torch.randn(2, 8)
print(blk(x, cond).shape)               # torch.Size([2, 16, 32])
print('cond=0 →', blk(x, torch.zeros_like(cond)).std().item())
print('cond=10x →', blk(x, 10 * cond).std().item())
```

运行 / Run with:
```bash
pip install torch einops
python try.py
```

预期输出 / Expected output:
```
torch.Size([2, 16, 32])
cond=0 → 0.7...
cond=10x → 12...
```

中文:把 `cond` 放大 10 倍,输出 std 几乎成正比放大 —— 这就是 FiLM 的"杠杆":条件向量是直接乘性地控制输出幅度的。一旦 cond 训练崩了(数值发散),整个 block 会跟着炸。生产里通常会在 `cond_encoder` 最后一层用 zero-init 让 block 初始时退化成 identity。

English: Scale `cond` up 10× and the output std scales almost proportionally — that is the "lever" of FiLM: the condition vector multiplicatively controls the output magnitude. If `cond` diverges during training the whole block explodes with it. Production code typically zero-inits the final layer of `cond_encoder` so the block starts as an identity.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **lerobot 的 ACT policy / lerobot's ACT policy**: 用 cross-attention 注入条件,但 backbone block 内部仍然是同样的"先卷,再 modulate,再卷,再残差"结构 / Uses cross-attention to inject conditions, but the backbone block keeps the same "conv → modulate → conv → residual" structure.
- **DiT 的 adaLN-Zero block / DiT's adaLN-Zero block**: 把 FiLM 用在 LayerNorm 上 —— `scale, shift, gate = cond_mlp(c).chunk(3)`,然后 `x = x + gate * attn(layernorm(x) * (1 + scale) + shift)`。本质上就是 FiLM 套了一层 LayerNorm / Puts FiLM on top of LayerNorm — `scale, shift, gate = cond_mlp(c).chunk(3)`, then `x = x + gate * attn(layernorm(x) * (1 + scale) + shift)`. Same FiLM, with a LayerNorm wrapper.
- **GR00T 的 DiTBlock / GR00T's DiTBlock**: cross-attention 之上叠 adaLN-Zero,相当于"条件走两条路":一条进 cross-attn,一条调 LayerNorm —— 但调 LN 这条还是 FiLM / Stacks cross-attention on top of adaLN-Zero, so the condition takes two paths: one into cross-attn, one onto the LayerNorm. The LN path is still FiLM.
- **StyleGAN2 的 modulated conv / StyleGAN2's modulated conv**: FiLM 的孪生兄弟 —— 把 modulation 折叠进卷积核本身,数学等价但更省一次乘法 / FiLM's twin — folds the modulation into the conv weight itself; mathematically equivalent but saves one multiply.

## 注意事项 / Caveats / when it breaks

- **`Rearrange` 的隐藏维 / The hidden dim of `Rearrange`**: 加的是 horizon=1 的维,只有当 `x` 是 `[B, C, T]` 时广播才对。如果你把 horizon 放在 dim=1 而不是 dim=2,会算错且不报错 / It adds a horizon=1 dim, which only broadcasts correctly when `x` is `[B, C, T]`. If you put horizon on dim=1 instead, the math is wrong and no error fires.
- **`cond_predict_scale=False` 的退化 / Degeneration when `cond_predict_scale=False`**: 退化成纯加法,等价于 `out = out + cond_embed[:, :, None]`。表达力差一截,但在小数据上反而更不容易 overfit。Diffusion Policy 原论文默认是 `False` / Degenerates to pure addition. Less expressive but harder to overfit on small data. The original Diffusion Policy paper actually defaults to `False`.
- **`Mish` 不是必需 / `Mish` is not mandatory**: 这里用 Mish 是因为 2023 年 Mish 还流行;现在 lerobot 已经换成 SiLU/GELU 了。激活函数对最终性能影响 <1%。 / Mish was a 2023 fashion. lerobot has since moved to SiLU/GELU. The activation choice changes final performance by <1%.

## 延伸阅读 / Further reading

- [FiLM: Visual Reasoning with a General Conditioning Layer (Perez et al., 2018)](https://arxiv.org/abs/1709.07871)
- [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion (Chi et al., 2023)](https://arxiv.org/abs/2303.04137)
- [Conditional U-Net implementation in lerobot](https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/diffusion)
- [DiT's adaLN-Zero block (the FiLM-meets-LayerNorm version)](https://github.com/facebookresearch/DiT/blob/main/models.py)
