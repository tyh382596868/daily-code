---
date: 2026-05-31
topic: diffusion
source: trending
repo: shengshu-ai/minWM
file: shared/algorithms/self_forcing.py
permalink: https://github.com/shengshu-ai/minWM/blob/e082c8a3297feaf048f1918f767f96a4ab4a85e8/shared/algorithms/self_forcing.py#L1-L83
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, world-model, self-forcing, video-diffusion, straight-through]
---

# Self-Forcing 的全部代码就 80 行 / Self-Forcing is 80 lines

> **一句话 / In one line**: 训"实时交互式视频世界模型"时,在所有去噪步里 *随机挑一步* 让 gradient 流过(其他步 no_grad),并且只让最后 N 帧吃 loss —— 用 `x*mask + x.detach()*(1-mask)` 的恒等式实现"按形状路由梯度"。 / Real-time interactive video world-model training picks *one* denoising step at random for gradient flow (all others run no-grad) and restricts loss to the last N frames — using `x*mask + x.detach()*(1-mask)` to "route gradient by shape."

## 为什么重要 / Why this matters

2025 出的 self-forcing 是让"实时交互世界模型"训得动的关键 trick。问题背景:一个交互式视频 WM 必须自回归地生成 —— 用户每给一个 action,模型立刻吐出下一组帧。训练时要模拟这个过程:让模型连着吐若干个 block(每个 block 是 K 步去噪 × M 帧),再算 loss。但如果每一步去噪都流梯度,memory 爆炸;如果只在最后一步流梯度,模型学不到"早期去噪步骤要走对方向"。Self-forcing 的解法是用 *truncated random backprop*:每个 block 在去噪步里随机抽一步当 "exit step",只在那一步开 grad。同时,只对"新生成的"最后 N 帧算 loss(早期帧是已知 context,算 loss 没意义)。这两个机制各对应一个非常通用的工程 idiom —— `randint` 采样 + `mask + detach` 梯度路由 —— 看一遍这 80 行,你就同时学到了 self-forcing 和"如何按形状屏蔽梯度"。minWM 是 2026 年 5 月刚出的"教学型实时交互 WM" 项目,这个文件就是它的算法核心。

The 2025 self-forcing trick is what makes real-time interactive video world models trainable. Setup: an interactive video WM must generate autoregressively — every user action produces the next clip of frames. Training has to mimic this: let the model emit several blocks (each block = K denoising steps × M frames), then take loss. Backprop through every denoising step blows memory; backprop only through the last step means earlier denoising steps don't learn to "head in the right direction." Self-forcing's answer is *truncated random backprop*: in each block, randomly pick one denoising step as the "exit step" and enable grad only there. Also: take loss only on the *newly generated* last N frames (earlier frames are known context, no loss signal). Each mechanism is a transferable engineering idiom — `randint` sampling + `mask + detach` gradient routing. Reading this 80-line file gets you self-forcing *and* the general "mask-shaped gradient gating" idiom. minWM is a May 2026 educational real-time interactive WM project; this file is its algorithmic heart.

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
    # Build slice for the frame dimension
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

1. **`sample_exit_step` 的 `same_across_blocks` 开关 (lines 29-36)**:
   - 中文: `same_across_blocks=True` 时整批 block 共用一个随机退出步(更省 memory,因为你可以一次决定哪些步要存 activation);`False` 时每个 block 独立采样,梯度信号更分散、训练更稳但实现更难(要存更多 activation)。HY15 / Wan21 默认走 True。
   - English: `same_across_blocks=True` means all blocks share one randomly-chosen exit step (memory-cheap — you decide which steps to checkpoint once); `False` samples independently per block, spreading the gradient signal at the cost of more cached activations. HY15 / Wan21 default to True.

2. **`device=device` 的微妙 / The subtle `device=device` (line 30)**:
   - 中文: `torch.randint` 用 GPU 上的随机源生成 —— 这样在多 GPU DDP 下每个 rank 拿到 *相同* 的 exit step(只要 seed 同步),否则不同 rank 在不同步上开 grad,allreduce 就会接到形状不一致的张量直接爆。
   - English: `torch.randint` uses the GPU's RNG so under DDP each rank picks the *same* exit step (assuming synchronized seed). Otherwise ranks would enable grad on different steps and the allreduce would receive mismatched-shape tensors and crash.

3. **`create_gradient_mask` 的 `frame_dim` 参数 (lines 60-68)**:
   - 中文: 它要兼容两种 layout —— `(B, F, C, H, W)`(frame_dim=1)和 `(B, C, F, H, W)`(frame_dim=2,Conv3d 的默认)。`slices = [slice(None)] * len(shape)` 再 `slices[frame_dim] = slice(start, total_frames)` 是个非常通用的"沿指定维取一段"的 Python idiom,值得收藏。
   - English: it has to handle both `(B, F, C, H, W)` (frame_dim=1) and `(B, C, F, H, W)` (frame_dim=2, Conv3d default). The `slices = [slice(None)] * len(shape)` then `slices[frame_dim] = slice(start, total_frames)` is the canonical Python idiom for "slice along an arbitrary axis" — worth memorizing.

4. **`mask` 的 dtype 双形式 (line 67)**:
   - 中文: `dtype=float` 时填 1.0(用于乘法/线性组合),`dtype=bool` 时填 True(用于索引)。同一个函数支持两种用法,调用者自己决定要"软 mask"(乘)还是"硬 mask"(索引)。
   - English: float-typed mask filled with `1.0` (for multiplication / linear combination), bool-typed mask filled with `True` (for indexing). One function, two uses — caller picks soft mask (multiply) or hard mask (index).

5. **`apply_gradient_mask` 的核心恒等式 (line 83)**:
   - 中文: `video * mask + video.detach() * (1 - mask)` 是整个文件的灵魂。前向上这等于 `video`(因为 `mask + (1-mask) = 1`),所以输出数值不变;反向时,只有 `video * mask` 这一项有梯度,而且梯度只在 mask=1 的位置流(其他位置乘 0)。`video.detach()` 那一项不参与 backward。一行代码同时做了"前向不变 + 反向按形状过滤"。
   - English: `video * mask + video.detach() * (1 - mask)` is the soul of the file. Forward it equals `video` exactly (because `mask + (1-mask) = 1`), so values are untouched. Backward only the `video * mask` term has gradient — and it flows only where mask=1 (other positions multiply by 0). The `video.detach()` term contributes nothing. One line: identity forward, mask-shaped backward.

## 类比 / The analogy

想象训练一个连环画家:画家要画 100 格漫画,前 30 格是用户给的剧情大纲(context,他不能改),后 70 格是他自己画的(generated)。老师每次只看一个随机的画格点评(`sample_exit_step` —— 比每格都看省钱),并且只对后 70 格打分(`gradient_mask` —— 前 30 格不是他画的,打分无意义)。难的不是"挑哪格打分",难的是"前 30 格物理上 *存在于* 整张连环画里,怎么让评分时假装它们不存在?" `apply_gradient_mask` 就是那张"评分用的塑料模板":叠在画上之后,数值没变(评委还能看到完整故事),但只有挖空的 70 格能被改动 —— 前 30 格的位置全用胶水封了(detach)。

Picture training a serial comic artist. They draw 100 panels: the first 30 are context the user provided (they can't change those), the last 70 are theirs. The teacher critiques *one* randomly chosen panel per session (`sample_exit_step` — cheaper than every panel) and only grades the last 70 (`gradient_mask` — grading what they didn't draw is pointless). The tricky part isn't "which panel to grade." It's "the first 30 panels *exist* on the same physical sheet — how do we pretend they don't when grading?" `apply_gradient_mask` is the plastic grading template you lay on top: values are unchanged (the grader sees the full story), but only the cut-outs over the last 70 panels are editable — the other positions are glued shut (detached).

## 自己跑一遍 / Try it yourself

```python
# pip install torch
import torch

def apply_gradient_mask(x, mask):
    return x * mask + x.detach() * (1.0 - mask)

# A "video" of 5 frames; we want gradient only on the last 2.
torch.manual_seed(0)
video = torch.randn(1, 5, 3, requires_grad=True)  # (B=1, F=5, C=3)
mask  = torch.zeros_like(video)
mask[:, 3:5] = 1.0                                # last 2 frames

# Run the trick + compute a dummy loss
out  = apply_gradient_mask(video, mask)
loss = (out ** 2).sum()
loss.backward()

print("forward matches input :", torch.equal(out, video))     # True — identity in forward
print("gradient shape        :", video.grad.shape)            # (1, 5, 3)
print("gradient on frames 0-2:", video.grad[0, :3].abs().sum().item())   # 0.0
print("gradient on frames 3-4:", video.grad[0, 3:].abs().sum().item())   # nonzero
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
forward matches input : True
gradient shape        : torch.Size([1, 5, 3])
gradient on frames 0-2: 0.0
gradient on frames 3-4: <some positive number>
```

中文: 前向和原视频一模一样,但反向 *只有* 最后两帧拿到了梯度。这个 `mask + detach` 恒等式是 PyTorch 里"按形状路由梯度"的标准黑科技 —— Straight-Through Estimator、Gumbel-Softmax、scheduled sampling 全是它的近亲。

English: forward is byte-for-byte the input, but backward *only* the last two frames get gradient. This `mask + detach` identity is the canonical PyTorch trick for "routing gradient by shape" — Straight-Through Estimator, Gumbel-Softmax, and scheduled sampling are all close cousins.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Causal-Forcing DMD 蒸馏** / **Causal-Forcing DMD distillation**: 中文: 5-29 教过的 DMD 也用同样的"挑一个 timestep 求梯度,其它 detach"的思路,只是 mask 是"沿 timestep 轴"而不是"沿 frame 轴"。两者本质上都是 truncated backprop。 / English: the DMD gradient covered on 2026-05-29 uses the same "select one timestep, detach the rest" idea — mask is along the timestep axis instead of the frame axis. Both are truncated backprop.
- **Straight-Through Estimator (STE)** / **Straight-Through Estimator**: 中文: VQ-VAE 的量化操作不可导,用 `z + (z_q - z).detach()` 这种相同结构骗过去 —— 前向是量化后的 `z_q`,反向是 `z` 的恒等梯度。 / English: VQ-VAE quantization is non-differentiable; `z + (z_q - z).detach()` works around it — forward returns quantized `z_q`, backward returns identity gradient through `z`. Same mask-detach algebra.
- **Scheduled sampling 在 seq2seq 里的用法** / **Scheduled sampling in seq2seq**: 中文: 训练时随机让一些位置看 ground truth 而不是模型预测,实现完全靠 `gt * mask + pred * (1-mask)` 在 forward 上拼合,再用对应 mask 屏蔽 loss。 / English: train-time mix of teacher forcing and model output uses `gt * mask + pred * (1-mask)` for forward composition and the same mask to gate loss — direct analog of the self-forcing identity.

## 注意事项 / Caveats / when it breaks

- **`mask` 必须 detach 或不带梯度** / **`mask` must be detached or grad-free**: 中文: 如果 `mask` 自己 require_grad,反向时会有一支"对 mask 求梯度"的链路,通常不是你想要的 —— 用 `torch.zeros(...)` / `torch.ones(...)` 创建天然 grad-free。 / English: if `mask` itself requires grad, backward computes a gradient *through* the mask — usually not what you want. Use `torch.zeros(...)` / `torch.ones(...)` (naturally grad-free).
- **`detach()` 在 in-place op 之后** / **`detach()` after an in-place op**: 中文: 如果你的 `video` 之前被 `+=` 之类原地改过,`video.detach()` 的版本号不匹配,backward 会报 "modified by an inplace operation"。 / English: if `video` was previously mutated in place (`+=`), `video.detach()` carries the new version tag; backward may raise "modified by an inplace operation."
- **DDP / FSDP 下 exit-step 必须同步** / **DDP / FSDP require synchronized exit steps**: 中文: 不同 rank 抽到不同 exit step,有的 rank 进入 grad-enabled forward 而其它没有,allreduce 形状不一致直接 deadlock。`device=device` + 同步 seed 是最简单的同步方式;或者 rank 0 决定后 broadcast。 / English: ranks picking different exit steps cause shape-mismatched allreduces and deadlock. Either use a synchronized seed with `device=device`, or rank-0 picks then broadcasts.

## 延伸阅读 / Further reading

- minWM repo: <https://github.com/shengshu-ai/minWM>
- Self-Forcing paper (training real-time interactive video world models): <https://arxiv.org/abs/2506.08009>
- Distribution Matching Distillation (DMD): <https://arxiv.org/abs/2311.18828>
- Past entry on DMD gradient subtraction: `2026/05/2026-05-29-causal-forcing-dmd-gradient.md`
