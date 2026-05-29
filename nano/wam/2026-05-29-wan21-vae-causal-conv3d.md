---
date: 2026-05-29
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/modules/vae.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vae.py#L17-L36
difficulty: intermediate
read_time: ~9 min
tags: [code-of-the-day, wam, vae, causal-3d-conv, temporal]
build_role: VAE / latent encoder-decoder — the pixel ↔ latent compressor that every diffusion-based WAM lives on top of
---

# 一行 padding 把 nn.Conv3d 变成因果 3D 卷积 / One line of padding turns nn.Conv3d into a causal 3D conv

> **一句话 / In one line**: Wan2.1 的整个 3D VAE 全部由 `CausalConv3d` 堆出来,而这个类只在 `nn.Conv3d` 之上做了一件事:把时间维度的 padding 全压到"过去"那一侧,使第 t 帧永远只看 ≤ t 的帧。 / Wan2.1's entire 3D VAE is built from `CausalConv3d`, and the whole class adds *one* trick on top of `nn.Conv3d`: shift all temporal padding to the past so frame `t` can only see frames `≤ t`.

## 为什么重要 / Why this matters

WAM 想用 diffusion 生成视频/世界状态,但直接在 pixel 空间做 diffusion 太贵 —— 一个 4×16×256×256 的 clip 已经 4M 个数。所以所有现代 WAM(Wan/Sora/CogVideo/lingbot)都先用 3D VAE 把视频压成 latent,再在 latent 空间做 diffusion。3D VAE 需要解决一个普通 2D Conv 没有的问题:**时间因果性**。如果训练时 t=0 的 token 能看到 t=10 的未来帧,那 latent 就泄露了"未来",在自回归 / 流式推理时会崩。`CausalConv3d` 用最简洁的写法(20 行)解决这个问题。

Modern world models compress video into latents before running diffusion — denoising 4M raw pixels per clip is infeasible. Every WAM stack (Wan, Sora, CogVideo, lingbot) sits on top of a 3D VAE. The non-trivial requirement compared to a 2D Conv is **temporal causality**: during training, latent token at time `t` must never see frame `t+1`, or your autoregressive / streaming inference will leak the future. `CausalConv3d` solves this in 20 lines by repackaging Conv3d's symmetric padding into one-sided temporal padding.

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/modules/vae.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vae.py#L17-L36)

```python
CACHE_T = 2


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
```

## 逐行讲解 / What's happening

1. **第 24-25 行 / Lines 24-25 (`self._padding = (...)`)**:
   - 中文:`nn.Conv3d` 的 padding 是 `(t, h, w)`,会在两侧各填 t/h/w 个零。这里改成 6-元组 `(w_left, w_right, h_left, h_right, t_left, t_right)`,关键是 **`t_right = 0`、`t_left = 2 * padding[0]`** —— 时间维度只往左 padding(过去),完全不往右 padding(未来)。空间维度 (h, w) 还是对称 padding。
   - English: `nn.Conv3d`'s `padding` is a 3-tuple `(t, h, w)` that pads symmetrically. The class converts that to the 6-tuple form `(w_left, w_right, h_left, h_right, t_left, t_right)` and the key trick is **`t_right = 0`, `t_left = 2 * padding[0]`**: the temporal dim is padded only to the *past* side. Spatial dims (h, w) keep their symmetric padding.

2. **第 26 行 / Line 26 (`self.padding = (0, 0, 0)`)**:
   - 中文:把父类 `nn.Conv3d` 自己的 padding 清零 —— 否则下面 `super().forward(x)` 会再补一次。我们要自己控制 padding。
   - English: zero out the parent class's padding, otherwise `super().forward(x)` would pad again on top of ours. We're taking full control of how padding gets applied.

3. **第 28-34 行 / Lines 28-34 (`forward`)**:
   - 中文:正常情况下直接 `F.pad(x, self._padding)` 然后跑 conv —— 因为左 padding 翻倍、右 padding=0,kernel size=3 的时间 conv 此时输出帧 `t` 用到的输入帧 = `{t-2, t-1, t}`,严格因果。
   - English: in the simple path, `F.pad` applies the asymmetric padding and then `super().forward` runs the kernel. For a temporal kernel of size 3 this means the output at frame `t` was computed from input frames `{t-2, t-1, t}` — strictly causal.

4. **第 30-33 行 / Lines 30-33 (the `cache_x` path)**:
   - 中文:流式推理时,你不会一次性把整段视频喂进来,而是一块一块送入(chunked inference)。这时左侧 padding 不能填 0(那是"没有过去"的语义),而是要填**上一个 chunk 的最后几帧**,这个张量就是 `cache_x`。代码把 `cache_x` 拼在 `x` 左侧,然后从应有的 padding 里**扣掉**已经填好的帧数,剩下的才用 `F.pad` 补 0。这样一个固定权重的 conv 可以无限地处理流式视频,跨 chunk 不漏帧。
   - English: at streaming inference time you don't feed the entire clip at once — you feed chunks. Then "padding with zeros" on the left would lie about the past. Instead the caller passes `cache_x` (the last few frames of the previous chunk); the code prepends those to `x` and *subtracts* the satisfied padding from the count. The result is one set of fixed weights that processes infinite video chunk-by-chunk with no boundary artefacts.

## 类比 / The analogy

像写文章时只允许看左边已经写完的字、不许偷看右边还没写的字。`nn.Conv3d` 默认是"能往两边瞄",`CausalConv3d` 把右边视野关上 —— 但同时把左边视野放宽两倍,这样能看的总信息量不变,只是方向单侧化了。`cache_x` 像是写新一章时,从上一章末尾抄两句到黑板,接着写就能保证语义连贯。

Think of writing a letter where you may only glance to the left at words you've already written, never to the right at what's still blank. `nn.Conv3d` peeks both ways by default; `CausalConv3d` closes the right side but doubles the left to keep total context the same. `cache_x` is your habit of copying the last two sentences of the previous chapter onto a clipboard before starting the next — so continuity holds across chapter boundaries.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-WAM

中文:`CausalConv3d` 是 nanoWAM 的最底层积木,放在 `nano/wam/blocks/causal_conv3d.py` 这个位置。上游是数据加载器(每个 batch 一个 `[B, C, T, H, W]` 的视频张量),它会被 VAE 的 Encoder3d 调用很多次,逐层把 `T, H, W` 都下采样。下游是整个 DiT 主干 —— DiT 在 *latent* 空间工作,所以 DiT 拿到的是 `[B, C_latent, T_latent, H_latent, W_latent]` 而不是 pixel。如果省掉因果性(直接用 nn.Conv3d),你训练时不会立刻翻车,但当你想做"给定历史预测未来"(autoregressive WAM)时,模型已经在 latent 阶段就偷看过未来,效果会比一个用 CausalConv3d 训出来的模型差几个量级。生产级实现还要补:VAE 的 KL 损失、LPIPS 感知损失、GAN 判别器(VAE-GAN 训练),以及 `feat_cache` 机制(下面 Resample 笔记会讲) —— 把它们串起来才是完整的 3D VAE。

English: `CausalConv3d` is the lowest-level brick of nanoWAM — it lives in `nano/wam/blocks/causal_conv3d.py`. Upstream is the dataloader emitting `[B, C, T, H, W]` video tensors; downstream the conv is called over and over inside `Encoder3d` (and later `Decoder3d`) to compress `T, H, W` step by step. The DiT trunk that sits above all this operates entirely in latent space, so it sees `[B, C_lat, T_lat, H_lat, W_lat]`. Skipping causality (i.e. using plain `nn.Conv3d`) does not blow up training immediately, but the moment you want autoregressive future prediction the model has already cheated by peeking at future frames in the VAE stage — autoregressive quality collapses. A production 3D VAE adds KL loss, LPIPS perceptual loss, a GAN discriminator (VAE-GAN training), and the `feat_cache` streaming machinery shown in tomorrow's Resample note.

## 自己跑一遍 / Try it yourself

```python
# pip install torch
import torch, torch.nn as nn, torch.nn.functional as F

class CausalConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2],
                         self.padding[1], self.padding[1],
                         2 * self.padding[0], 0)
        self.padding = (0, 0, 0)
    def forward(self, x):
        return super().forward(F.pad(x, self._padding))

torch.manual_seed(0)
conv = CausalConv3d(1, 1, kernel_size=(3, 1, 1), padding=(1, 0, 0))

# input: T=5 frames, all zero except a spike at t=2
x = torch.zeros(1, 1, 5, 1, 1)
x[0, 0, 2, 0, 0] = 1.0
y = conv(x).detach()
print("output per-frame:", y[0, 0, :, 0, 0].tolist())
# Notice: the spike at input t=2 only affects output t >= 2.
# A plain nn.Conv3d with padding=(1,0,0) would also affect output t=1 (looked into the future).
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
output per-frame: [0.0, 0.0, X, Y, Z]
```

中文:t=0 和 t=1 的输出严格为 0 —— 即"过去帧没有任何信号"。换成 `nn.Conv3d(...)`(对称 padding)你会看到 t=1 也变非零,因为它"提前看到了" t=2 的尖峰。

English: outputs at t=0 and t=1 are exactly 0 — proof that the past was untouched. Swap in plain `nn.Conv3d` with the same padding and you'll see t=1 become non-zero too, because it peeked one frame into the future.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **CogVideoX 3D VAE** / **CogVideoX 3D VAE**: 中文 — 用一样的左侧 padding 思路,只是把缓存机制拆得更细。 / English — same left-padding idea, with a more granular cache abstraction.
- **Open-Sora HunyuanVAE** / **Open-Sora HunyuanVAE**: 中文 — `opensora/models/hunyuan_vae/vae.py` 里有类似的 causal 3D 卷积。 / English — `opensora/models/hunyuan_vae/vae.py` reimplements the same trick.
- **WaveNet (audio)** / **WaveNet (audio)**: 中文 — 一维因果卷积的祖宗,2016 年的论文,所有"流式生成 + 卷积"都从这里来。 / English — the 1-D ancestor from 2016; every streaming-conv generator descends from WaveNet.
- **昨天讲的 lingbot-va FlexAttention mask** / **Yesterday's lingbot-va FlexAttention mask**: 中文 — 那个 mask 在 attention 层强制因果,这里在 conv 层强制因果。两层因果叠加才得到完整的"latent 看不到 latent 未来"。 / English — that mask enforces causality at the attention layer; this enforces it at the conv layer. Both are needed for end-to-end "latent never peeks at latent future".

## 注意事项 / Caveats / when it breaks

- **kernel size 必须是奇数** / **Kernel size should be odd**: 中文 — `2 * padding[0]` 是基于 `padding = (k-1)//2` 推导出来的;偶数 kernel 会偏移一帧。 / English — the `2 * padding[0]` formula assumes `padding = (k-1)//2`. Even kernels need a different recipe or you misalign by one frame.
- **stride > 1 还要额外算** / **Strides > 1 need extra care**: 中文 — Wan2.1 在 `Resample` 里下采样用 `stride=(2, 1, 1)` 配合 `padding=(0, 0, 0)`(不是这里的公式),所以涉及时间下采样时**不要复用同一个 padding 推导**。 / English — Wan2.1's downsampling `Resample` uses `stride=(2, 1, 1)` with `padding=(0, 0, 0)`, *not* this formula. Reusing the symmetric-padding derivation for strided temporal conv breaks the math.
- **cache_x 维度要对** / **cache_x dims must match exactly**: 中文 — `cache_x` 要和 `x` 在 batch、channel、空间维全部一致,否则 `torch.cat` 报错。生产里 cache 通常是上一个 chunk 的最后 `CACHE_T=2` 帧,严格 clone 一份(别共享显存)。 / English — `cache_x` must agree with `x` on every non-temporal dim. Production code keeps `CACHE_T = 2` cloned frames and refuses to share storage with the live activation tensor.

## 延伸阅读 / Further reading

- [Wan2.1 paper / repo](https://github.com/Wan-Video/Wan2.1)
- [CogVideoX 3D VAE](https://github.com/THUDM/CogVideoX)
- [WaveNet (van den Oord et al., 2016)](https://arxiv.org/abs/1609.03499)
- [Diffusion Models for Video — 3D VAE design notes](https://arxiv.org/abs/2403.13802)
