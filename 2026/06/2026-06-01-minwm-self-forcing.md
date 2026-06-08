---
date: 2026-06-01
topic: diffusion
source: trending
repo: shengshu-ai/minWM
file: shared/algorithms/self_forcing.py
permalink: https://github.com/shengshu-ai/minWM/blob/e082c8a3297feaf048f1918f767f96a4ab4a85e8/shared/algorithms/self_forcing.py#L1-L83
difficulty: intermediate
read_time: ~9 min
tags: [code-of-the-day, diffusion, distillation, world-model, gradient-routing]
---

# Self-Forcing 蒸馏的两个支柱:随机退出步 + 软梯度掩码 / The two pillars of Self-Forcing distillation: random exit step + soft gradient mask

> **一句话 / In one line**: 训练时随机挑一步打开梯度、其余 no-grad,只让最后 N 帧产生梯度——用 `x*m + x.detach()*(1-m)` 这条恒等式把"梯度路由"压进 80 行 / At training time, randomly open gradient at one denoising step (others no-grad) and only let the last N frames produce gradient — gradient routing through the identity `x*m + x.detach()*(1-m)`, packed into 80 lines.

## 为什么重要 / Why this matters

实时交互世界模型(minWM、Wan-2.1 蒸馏、Hunyuan-Video 蒸馏)要把 50 步的 diffusion 压到 1-4 步,但完整反传 50 步显存和算力都吃不消。Self-Forcing 的核心是**截断反传 + 帧级梯度掩码**:每次只在随机一步打开梯度,只对"新生成的几帧"算 loss,其他上下文帧停止梯度——80 行就够。读它能同时学到(1)distillation 里 truncated backprop 的实现习惯、(2)那条万能的 "soft gradient mask" 恒等式(它在 STE、partial-freeze、GAN R1 里都出现)。

Real-time interactive world models (minWM, Wan-2.1 distillation, Hunyuan-Video distillation) compress 50-step diffusion down to 1-4 steps, but full backprop through 50 steps is infeasible in memory and compute. Self-Forcing's trick is **truncated backprop + per-frame gradient mask**: only open gradient at one randomly-chosen step, only compute loss on the last few newly-generated frames, freeze gradient on context frames — all in 80 lines. Read it and you simultaneously learn (1) how truncated-backprop distillation is actually wired up, and (2) the universal "soft gradient mask" identity (the same trick reappears in STE, partial freezing, GAN R1).

## 代码 / The code

`shengshu-ai/minWM` — [`shared/algorithms/self_forcing.py`](https://github.com/shengshu-ai/minWM/blob/e082c8a3297feaf048f1918f767f96a4ab4a85e8/shared/algorithms/self_forcing.py#L1-L83)

```python
"""Self-Forcing sub-functions.

Exit step sampling, gradient mask creation/application.
Shared by HY15 and Wan21.
"""

import torch
from torch import Tensor
from typing import List, Optional


def sample_exit_step(
    num_denoising_steps: int,
    num_blocks: int,
    same_across_blocks: bool = True,
    device: Optional[torch.device] = None,
) -> List[int]:
    """Sample exit step indices for self-forcing truncated denoising.

    Each block gets an exit step index in [0, num_denoising_steps).
    At the exit step, the generator runs with grad; other steps are no-grad.
    """
    if same_across_blocks:
        idx = torch.randint(0, num_denoising_steps, (1,), device=device).item()
        return [idx] * num_blocks
    else:
        indices = torch.randint(
            0, num_denoising_steps, (num_blocks,), device=device
        )
        return indices.tolist()


def create_gradient_mask(
    total_frames: int,
    last_n_frames: int,
    shape: List[int],
    frame_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Create a gradient mask that is 1 for the last N frames, 0 elsewhere.

    Used to restrict DMD loss to only the last N generated frames,
    avoiding gradient through early (context) frames.
    """
    mask = torch.zeros(shape, device=device, dtype=dtype)
    start = total_frames - last_n_frames
    if start < 0:
        start = 0
    slices = [slice(None)] * len(shape)
    slices[frame_dim] = slice(start, total_frames)
    mask[tuple(slices)] = 1.0 if dtype != torch.bool else True
    return mask


def apply_gradient_mask(video: Tensor, mask: Tensor) -> Tensor:
    """Apply gradient mask: pass gradient only where mask is nonzero.

    For masked regions, detach the tensor (stop gradient).
    For unmasked regions, keep the gradient.

    result = video * mask + video.detach() * (1 - mask)
    """
    return video * mask + video.detach() * (1.0 - mask)
```

## 逐行讲解 / What's happening

1. **`sample_exit_step`**:
   - 中文: 在 `[0, num_denoising_steps)` 区间随机挑一个 idx。如果 `same_across_blocks=True`(默认),所有时间块共享同一个 exit step(更省显存,实际效果好)。`False` 时每块独立采样(更"覆盖完整 schedule")。`.item()` 把 GPU 上的标量搬回 CPU 做 Python list 构造。
   - English: pick a random `idx` in `[0, num_denoising_steps)`. If `same_across_blocks=True` (default) every temporal block shares it (saves memory, works fine in practice); otherwise each block samples independently for better schedule coverage. `.item()` ferries the scalar back to CPU to build a Python list.

2. **为什么需要 "exit step" / why "exit step" exists**:
   - 中文: 蒸馏目标是让"学生 generator"用 1 步或几步逼近"教师"50 步的输出。如果训练时让 generator 跑完全部 50 步并反传,显存爆。Self-Forcing 的解法:每个 step generator 都跑前向,但**只有随机选中的那一步打开 autograd**,其余 step 用 `torch.no_grad()` 跑。这样反传只穿过常数深度的一步,显存 O(1)。
   - English: distillation aims to make a student generator approximate a 50-step teacher in 1-4 steps. Training the student with full backprop through all 50 steps would blow memory. Self-Forcing's fix: run all forward steps, but **only the randomly selected step has autograd on**; everything else runs inside `torch.no_grad()`. Backprop now traverses a constant-depth single step → O(1) memory.

3. **`create_gradient_mask`**:
   - 中文: 造一个和 video 同 shape 的 0/1 张量,只有"最后 N 帧"对应的位置是 1。`slices = [slice(None)] * len(shape); slices[frame_dim] = slice(start, end)` 是 Python 通用做法——构造一个 N 维 slice tuple 给任意维度的张量打 patch。
   - English: build a 0/1 tensor with the same shape as the video, with 1s only at the last N frames. The `slices = [slice(None)] * len(shape); slices[frame_dim] = slice(start, end)` pattern is a Pythonic way to build an N-D slice tuple to patch any dimension of an arbitrary-rank tensor.

4. **为什么只在最后 N 帧 / why only the last N frames**:
   - 中文: 交互式世界模型生成"上下文 K 帧 + 新 N 帧"的视频块,context 来自历史不应该被梯度污染(否则模型学到"修改历史"),只想让 generator 学习如何"在 context 之后产生看起来对的 N 个新帧"。所以 loss 只在最后 N 帧上算,gradient mask 是实现这一约束的工具。
   - English: an interactive world model generates "K context frames + N new frames" per chunk. The context comes from history and must not receive gradient (else the model learns to rewrite the past). We only want the generator to learn "what's a good next N frames given this context". So the loss applies only to the last N frames; the gradient mask enforces it.

5. **`apply_gradient_mask`: `x*m + x.detach()*(1-m)`**:
   - 中文: **整段代码的灵魂**。`m=1` 的位置:`x*1 + x.detach()*0 = x` —— 完整梯度。`m=0` 的位置:`x*0 + x.detach()*1 = x.detach()` —— 数值上仍是 `x`,但是 `.detach()` 切断了梯度。最终前向输出**等于 `x`**(因为 `x.detach() == x` 数值上),但反向梯度只流过 `m=1` 区域。0/1 之间的连续 mask 还能软路由(部分梯度)。
   - English: **the heart of the whole file**. At `m=1`: `x*1 + x.detach()*0 = x` → full gradient. At `m=0`: `x*0 + x.detach()*1 = x.detach()` → same value, no gradient. The forward output **equals `x`** everywhere numerically, but the backward only flows through positions where `m=1`. With a soft mask in `[0, 1]` you even get partial / scaled gradients for free.

## 类比 / The analogy

想象你是一个动画师,要给一个 50 帧的动画补颜色。导演说"我每次只检查随机一帧,只批评你最后 3 帧画得对不对——前面的请保持不变"。你每天都把整个 50 帧画完(forward),但只在那一帧用红笔留意改进(autograd on),其他帧闭着眼画(no_grad),并且每天结束时只把最后 3 帧交给导演打分(gradient mask)。一周下来,你学会了"在任意时间点接着前面的画风稳稳地画下去"。Self-Forcing 训世界模型的过程一模一样。

Imagine you're an animator coloring a 50-frame piece. The director says "I'll spot-check one random frame per day and only critique the last 3 frames; the rest must stay untouched". Every day you paint all 50 frames (forward), but only that one frame is done with attention to feedback (autograd on); the others you sketch on autopilot (no_grad). At the end of each day you submit only the last 3 frames for grading (gradient mask). Over a week you learn to continue any sequence smoothly from any point. Training a world model with Self-Forcing is exactly that.

## 自己跑一遍 / Try it yourself

```python
# self_forcing_demo.py — pure torch
import torch, torch.nn as nn

def sample_exit_step(T, B, same=True):
    if same: return [torch.randint(0, T, (1,)).item()] * B
    return torch.randint(0, T, (B,)).tolist()

def grad_mask(shape, frame_dim, last_n):
    m = torch.zeros(shape)
    sl = [slice(None)] * len(shape); sl[frame_dim] = slice(shape[frame_dim] - last_n, shape[frame_dim])
    m[tuple(sl)] = 1.0
    return m

def apply_mask(x, m):
    return x * m + x.detach() * (1 - m)

# fake video model: a single Linear over time
torch.manual_seed(0)
B, F, C = 1, 8, 4                 # [batch, frames, channels]
model = nn.Linear(C, C)

x = torch.randn(B, F, C)
# Pretend we ran the model and got a video; route gradients only to last 3 frames
y = model(x)
mask = grad_mask(y.shape, frame_dim=1, last_n=3)
y_routed = apply_mask(y, mask)
loss = (y_routed ** 2).sum()
loss.backward()

# Check: only positions corresponding to last 3 frames see non-zero grad through x
print("model.weight.grad norm :", model.weight.grad.norm().item())

# Same forward value either way
print("max numerical diff     :", (y_routed - y).abs().max().item())

# Truncated backprop simulation
T, num_blocks = 20, 4
exit_steps = sample_exit_step(T, num_blocks, same=True)
print(f"exit step for all {num_blocks} blocks  : {exit_steps[0]}")

ctx = torch.randn(B, F, C, requires_grad=False)
for step in range(T):
    if step == exit_steps[0]:
        with torch.enable_grad():
            out = model(ctx + 0.01 * torch.randn_like(ctx))
            loss = (apply_mask(out, mask) ** 2).sum()
            loss.backward()
    else:
        with torch.no_grad():
            ctx = model(ctx) + 0.01 * torch.randn_like(ctx)
print("model.weight.grad after truncated bp:", model.weight.grad.norm().item())
```

运行 / Run with:
```bash
pip install torch
python self_forcing_demo.py
```

预期输出 / Expected output:
```
model.weight.grad norm : (a positive number, e.g. ~3)
max numerical diff     : 0.0
exit step for all 4 blocks  : (some int in [0, 20))
model.weight.grad after truncated bp: (positive, larger than first round)
```

中文: 关键观察:`max numerical diff = 0.0` —— `apply_mask` 在数值上是恒等;但反传时只在 `mask=1` 的帧位置形成梯度。把 `last_n=3` 改成 `last_n=F`,所有帧都会贡献梯度;改成 `last_n=0`,梯度为零(对应"完全冻结")。

English: key observation: `max numerical diff = 0.0` — `apply_mask` is numerically identity; only the backward differs, with gradient flowing only through `mask=1` frames. Set `last_n=F` and every frame contributes; set `last_n=0` and you get zero gradient (full freeze).

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Straight-Through Estimator (STE)** / **STE in quantization-aware training**: 中文: `y = (x_q - x).detach() + x` 让前向走量化值、反向走原值。同一条恒等式。
- **partial-freeze 微调 / partial freezing**: 中文: `param * mask + param.detach() * (1 - mask)` 可以软冻结指定参数子集,而不需要去改 `requires_grad`。
- **GAN R1 regularizer 的梯度 routing**: 同样靠 `detach()` 把 D 的某一部分参与/不参与 G 的反传。
- **DMD2(Distribution Matching Distillation v2)** / **DMD2 distillation**: 在 minWM 的 `dmd.py` 同目录里;同 generator 同 mask 接口,只是 loss 换成 KL 形式。你之前 2026-05-29 的 Causal-Forcing DMD 笔记就是它的"双 score" 实现。
- **TBPTT(Truncated BPTT)** / **classic truncated BPTT in RNNs**: 中文: 同样思路——不在所有时间步反传,只在固定窗口内反传。Self-Forcing 是"随机化版的 TBPTT for diffusion"。
- **PPO 的 `target_value.detach()`**: critic 更新里截断 actor 的梯度,本质也是 `detach()` 路由。

## 注意事项 / Caveats / when it breaks

- **`same_across_blocks=False` 偶尔会爆显存 / per-block sampling can OOM**: 中文: 每块独立采样意味着可能多块同时落在"早期 step"(梯度链很短)或同时落在"晚期 step"(梯度链很长),实际显存峰值要按最坏情况估。论文默认 `True`。
- **`apply_gradient_mask` 在 inplace 操作下会失效 / inplace ops break mask routing**: 中文: 比如 `video.relu_()` 会覆盖 `detach()` 路径上的 buffer,导致 backward 报错或拿到错误梯度。Self-Forcing 路径上务必避免 inplace。
- **0/1 整数 mask 慢于 float / int mask is slower than float**: 中文: 用 `torch.bool` 通过乘法实际会被隐式 cast,torch 上比 `float32 mask + 乘法` 慢一点。除非显存特别紧才考虑 bool。
- **`mask` 必须 broadcast 对齐 frame_dim / mask shape must align with video**: 中文: 这份代码用完整 shape 是为了避免 broadcast 出错;省事的写法 `[1, F, 1, 1, 1]` 一样工作,但要小心 `apply_mask` 里 `(1 - m)` 的类型。
- **`with torch.no_grad():` 必须包对 / no_grad must wrap the right scope**: 中文: 实战里常见 bug 是把 generator 一半放在 no_grad 一半放在 enable_grad,导致 autograd graph 错连。把"非 exit step 的整个 forward"完整包进 `with torch.no_grad():`。

## 延伸阅读 / Further reading

- [minWM repo (shengshu-ai)](https://github.com/shengshu-ai/minWM) — 框架级文档,Wan21 / HY15 两路 trainer
- [Self-Forcing 原始论文](https://arxiv.org/abs/2506.08009) — 算法出处和 ablation
- [DMD2 paper](https://arxiv.org/abs/2405.14867) — 它的 loss 搭档
- [STE original ref(Bengio 2013)](https://arxiv.org/abs/1308.3432) — 同条恒等式更早的出处
- 你之前的 daily code:[DMD gradient(Causal-Forcing)](../05/2026-05-29-causal-forcing-dmd-gradient.md) ← 蒸馏 loss 的另一面
