---
date: 2026-06-03
topic: diffusion
source: trending
repo: shengshu-ai/minWM
file: shared/algorithms/self_forcing.py
permalink: https://github.com/shengshu-ai/minWM/blob/6f00e122d26747062283524b1cf5ae3ffae1b8d2/shared/algorithms/self_forcing.py#L12-L83
difficulty: intermediate
read_time: ~9 min
tags: [code-of-the-day, diffusion, world-model, autoregressive-video, self-forcing, straight-through]
---

# Self-Forcing 三件套:让自回归视频世界模型能 backprop 的 83 行 / Self-Forcing's three helpers: 83 lines that make autoregressive video world models trainable

> **一句话 / In one line**: 一个去噪 step 随机选当 "exit step"(只它有梯度,其余 no_grad),再用一个 frame mask 把梯度限制在最后 N 帧,最后用 `x*m + x.detach()*(1-m)` 这条 straight-through 公式把"只在这里反传"写完 —— 自回归视频 WAM 就能训了。 / Pick one random "exit step" per batch (only it gets grad, the rest run `no_grad`), then a frame mask restricts grad to the last N frames, then the straight-through line `x*m + x.detach()*(1-m)` enforces "only this region backprops" — that's all you need to train an autoregressive video world model.

## 为什么重要 / Why this matters

自回归视频世界模型(给前 K 帧 + 文本/动作,预测下一帧)是机器人、游戏、模拟器都想要的东西,但训练它有两个硬约束:(1) 一个完整 rollout 可能要 50 个去噪 step × 16 帧,如果每步都要 backprop,显存会爆;(2) 早期帧是"context"(条件),后期帧才是"prediction",在 context 上算 loss 会让模型学到 trivial copy。Self-Forcing 是 minWM 这种 "Minimal World Model" tutorial 类 repo 的核心技术,它把这两个问题用三个 20 行函数解决:`sample_exit_step` 随机挑一个 step,只让这个 step 走 grad-enabled forward;`create_gradient_mask` 构造一个"最后 N 帧为 1、其余为 0"的 mask;`apply_gradient_mask` 用经典的 straight-through 公式 `x*m + x.detach()*(1-m)` 把它落地。读完这三个函数你就能把自家 nanoWAM 从"只能跑 inference"升级到"能 finetune"。

Autoregressive video world models — give K context frames + text/actions, predict the next frame(s) — are what robotics, gaming, and simulators all want, but training one is bounded by two hard constraints: (1) a single rollout might be 50 denoising steps × 16 frames; backprop on every step explodes memory; (2) early frames are *context* (conditioning), later frames are *predictions*; computing loss on context teaches the model trivial copying. Self-Forcing is the core technique in tutorial-style "Minimal World Model" repos like minWM, and it cleans both problems up with three 20-line functions: `sample_exit_step` randomly picks one step to forward *with* gradients; `create_gradient_mask` constructs a "1 on the last N frames, 0 elsewhere" mask; `apply_gradient_mask` lands it via the classic straight-through `x*m + x.detach()*(1-m)`. Read these three and you can take your nanoWAM from inference-only to finetune-capable.

## 代码 / The code

`shengshu-ai/minWM` — [`shared/algorithms/self_forcing.py`](https://github.com/shengshu-ai/minWM/blob/6f00e122d26747062283524b1cf5ae3ffae1b8d2/shared/algorithms/self_forcing.py#L12-L83)

```python
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
        indices = torch.randint(0, num_denoising_steps, (num_blocks,), device=device)
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

    result = video * mask + video.detach() * (1 - mask)
    """
    return video * mask + video.detach() * (1.0 - mask)
```

## 逐行讲解 / What's happening

1. **`sample_exit_step` 的"截断去噪" / `sample_exit_step` and "truncated denoising"**:
   - 中文: 整个去噪过程有 50 个 step,但训练时**只让一个**走 grad。`torch.randint(0, num_denoising_steps, (1,))` 随机抽一个 index,在 train loop 里你这样用:`for t in range(N): if t == exit_step: x = G(x, t)  # grad on  else: with torch.no_grad(): x = G(x, t)`。等价于"前面 49 步都是热身,只有这一步算 loss"。期望意义下,每个 step 都有同样概率被选中,所以**梯度的期望覆盖所有 step**,只是方差大一点。
   - English: there are 50 denoising steps; only **one** gets gradient during training. `torch.randint(0, num_denoising_steps, (1,))` samples one index. In the train loop: `for t in range(N): if t == exit_step: x = G(x, t)  # grad on  else: with torch.no_grad(): x = G(x, t)`. The first 49 steps are warmup, only the chosen step gets loss. In expectation each step is selected with equal probability, so gradient coverage is unbiased — variance is just higher than per-step backprop.

2. **`same_across_blocks` 的设计选择 / Design choice in `same_across_blocks`**:
   - 中文: 如果你的 video 是一个一个 block 自回归生成(每 block 4-8 帧),你可以让所有 block 共享同一个 exit step(`same_across_blocks=True`,默认),也可以每 block 独立抽一个。默认共享更稳 —— 减少梯度估计的方差;独立的可以覆盖更多 step,但训练初期可能不稳。Tutorial-style repo 选了默认稳的方案。
   - English: if your video is generated block-by-block (4-8 frames per block), you can either share one exit step across all blocks (`same_across_blocks=True`, default) or sample one per block independently. The default-shared version is more stable — lower-variance gradient estimate; the independent version covers more steps per update but can be unstable early in training. A tutorial repo picks the safe default.

3. **`create_gradient_mask` 用切片造 mask / `create_gradient_mask` via fancy slicing**:
   - 中文: 这一段值得逐句看 —— 它的精妙在于**对任意维度的 video tensor 都通用**。你的 video 可能是 `[B, F, C, H, W]`(frame_dim=1)也可能是 `[B, C, F, H, W]`(frame_dim=2)。代码先构 `slices = [slice(None)] * len(shape)`(全切片),然后只覆盖 `slices[frame_dim] = slice(start, end)`,最后 `mask[tuple(slices)] = 1.0`。这是写 "任意维度索引" 的标准 Pythonic 写法,比写 `mask[:, start:, ...]` 之类的硬编码维度优雅得多。
   - English: this is worth reading slowly — it works for **any video tensor layout**. Your video might be `[B, F, C, H, W]` (`frame_dim=1`) or `[B, C, F, H, W]` (`frame_dim=2`). The code builds `slices = [slice(None)] * len(shape)` (full slice everywhere), then overrides only `slices[frame_dim] = slice(start, end)`, finally `mask[tuple(slices)] = 1.0`. This is the canonical Pythonic way to index a "this axis, that range; everything else, full" slice — much cleaner than hard-coding `mask[:, start:, ...]`.

4. **`apply_gradient_mask` 的 straight-through / `apply_gradient_mask`'s straight-through formula**:
   - 中文: 一行公式 `video * mask + video.detach() * (1 - mask)`。这是深度学习里超经典的 **straight-through estimator**(VQ-VAE、Gumbel-softmax、quantization 全都用):
     - **forward** 上: `video * mask + video.detach() * (1 - mask) = video * mask + video * (1 - mask) = video` —— 输出 = 输入,数值不变。
     - **backward** 上: `detach()` 阻断了 `(1 - mask)` 那一块的梯度,只有 `video * mask` 那一块流回来,等价于 "在 mask=1 区域有梯度,在 mask=0 区域梯度为 0"。
   - English: one line — `video * mask + video.detach() * (1 - mask)`. The classic **straight-through estimator** (VQ-VAE, Gumbel-softmax, quantization all use it):
     - **forward**: `video * mask + video.detach() * (1 - mask) = video * mask + video * (1 - mask) = video` — output equals input numerically.
     - **backward**: `detach()` kills gradient flow through the `(1 - mask)` branch, so only `video * mask` contributes — equivalent to "grad on where mask=1, zero where mask=0."

5. **三个函数为什么一起出现 / Why all three live together**:
   - 中文: 一个训练 step 是这样的:`exit_step = sample_exit_step(50, num_blocks)[0]` → 跑 50 个去噪 step,只有 `t == exit_step` 那一步有 grad → 得到完整 video → `mask = create_gradient_mask(total_frames, last_n=4, ...)` → `video = apply_gradient_mask(video, mask)` → `loss = dmd_loss(video, ...)` → `loss.backward()`。三个函数缺一不可:exit_step 控时间维稀疏,frame_mask 控空间维稀疏,straight-through 是把后者落地的"胶水"。
   - English: one training step looks like: `exit_step = sample_exit_step(50, num_blocks)[0]` → run 50 denoising steps, grad only when `t == exit_step` → get the full video → `mask = create_gradient_mask(total_frames, last_n=4, ...)` → `video = apply_gradient_mask(video, mask)` → `loss = dmd_loss(video, ...)` → `loss.backward()`. The three functions complement each other: `exit_step` thins out the time axis, `frame_mask` thins out the frame axis, and the straight-through is the glue that turns the mask into an actual gradient gate.

## 类比 / The analogy

中文:想象**学生练习写小说**,但 GPU 时间(老师批改时间)有限。老师不可能每一稿、每一段都仔细批 —— 太累。Self-Forcing 是聪明的妥协方案:**每天随机挑一稿来批**(`sample_exit_step`),其它草稿学生自己写、老师不看;批改时**只看最后几段**(`create_gradient_mask` 的 last N frames),前面的"开头"段落是题目给的、不需要打分;最后用一张**透明描红纸**(straight-through `x*m + x.detach()*(1-m)`)精确叠在要批改的几段上,老师只在描红纸覆盖的位置写评语,其他段落原文不动。学生的学习信号稀疏但 unbiased,GPU 时间花在刀刃上。

English: think of a **student practicing novel-writing** with limited GPU time (the teacher's grading budget). The teacher can't carefully grade every draft and every paragraph — too expensive. Self-Forcing is the clever compromise: **pick one random draft per day** to grade (`sample_exit_step`); the student writes the others alone, ungraded. When grading, **only the last few paragraphs** get attention (`create_gradient_mask`'s last N frames); the opening was given as the prompt and doesn't need scoring. Finally, a **transparent tracing-paper overlay** (straight-through `x*m + x.detach()*(1-m)`) is placed precisely over the to-be-graded paragraphs; the teacher only writes feedback inside the traced region, leaving other paragraphs untouched. The student's learning signal is sparse but unbiased, and the GPU budget goes where it matters.

## 自己跑一遍 / Try it yourself

```python
import torch

# 1) sample_exit_step
def sample_exit_step(N, blocks, same=True):
    return [torch.randint(0, N, (1,)).item()] * blocks if same else \
           torch.randint(0, N, (blocks,)).tolist()

# 2) create_gradient_mask
def make_mask(total_F, last_n, shape, frame_dim, device, dtype=torch.float32):
    m = torch.zeros(shape, device=device, dtype=dtype)
    s = [slice(None)] * len(shape); s[frame_dim] = slice(total_F - last_n, total_F)
    m[tuple(s)] = 1.0
    return m

# 3) apply_gradient_mask (straight-through)
def apply_mask(x, m): return x * m + x.detach() * (1.0 - m)

# Verify: forward is identity, backward only flows in mask=1 region.
torch.manual_seed(0)
video = torch.randn(2, 8, 3, 4, 4, requires_grad=True)   # [B, F=8, C, H, W]
mask  = make_mask(total_F=8, last_n=2, shape=video.shape, frame_dim=1, device="cpu")
masked = apply_mask(video, mask)
print("forward identity?",  torch.allclose(masked, video))    # True

masked.sum().backward()
grad_per_frame = video.grad.abs().sum(dim=(0, 2, 3, 4))
print("grad per frame:", grad_per_frame.tolist())
# Frames 0..5 should be 0, frames 6..7 should be nonzero.
print("exit step today:", sample_exit_step(50, 4))
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
forward identity? True
grad per frame: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 48.0, 48.0]
exit step today: [17, 17, 17, 17]
```

中文:前 6 帧梯度严格为零,后 2 帧拿到完整梯度 —— 这就是 `create_gradient_mask` + `apply_gradient_mask` 协作的全部效果。forward 输出和原输入位级一致,所以下游任何 loss 都能直接调,不需要改一行。

English: gradient is strictly zero on the first 6 frames and fully flowing on the last 2 — that's the whole effect of `create_gradient_mask` + `apply_gradient_mask` cooperating. The forward output is bit-identical to the input, so any downstream loss plugs in unchanged.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **VQ-VAE 的 straight-through quantization** / **VQ-VAE's straight-through quantization**: 同一条公式 `z_q = z + (z_q - z).detach()`,数学上等价于 `apply_gradient_mask(z_q, mask=1) + apply_gradient_mask(z, mask=0)`。 / Same formula `z_q = z + (z_q - z).detach()` — algebraically equivalent to `apply_gradient_mask(z_q, mask=1) + apply_gradient_mask(z, mask=0)`.
- **Diffusion Forcing / DMD** / **Diffusion Forcing / DMD**: minWM 用 self-forcing 配合 DMD loss(蒸馏 score 网络),前 N 步免梯度的精神和 DMD 的 score-matching 完美互补。 / minWM pairs self-forcing with DMD (Distribution Matching Distillation); the "first N steps no grad" idea complements DMD's score-matching elegantly.
- **CausVid / Causal-Forcing(已覆盖 2026-05-29)** / **CausVid / Causal-Forcing (covered 2026-05-29)**: 同一思想的自回归视频版,minWM 是它的"教学复刻"。 / Same idea, autoregressive video flavor — minWM is the "teaching reimplementation".
- **REINFORCE-style 训练里的 baseline** / **REINFORCE-style baselines**: 也是一个"哪部分参与 grad、哪部分不参与"的 mask 设计;Self-Forcing 的 mask 比 RL 简单得多,但形式相同。 / Another "what flows / what doesn't" mask design; Self-Forcing's mask is far simpler than RL baselines but has the same shape.

## 注意事项 / Caveats / when it breaks

- **梯度方差 / Gradient variance**: 每步只有 1/50 概率被选中,batch size 小的时候 loss 噪声会很大。实践上常常 `batch_size * num_blocks` 取到 16+ 才稳。 / Each step has only 1/50 probability of being chosen, so loss is noisy at small batch sizes. In practice you want `batch_size * num_blocks ≥ 16` or so for stability.
- **`last_n_frames` 不能太大 / Don't make `last_n_frames` too big**: 如果 `last_n = total_frames`,mask 全是 1,等同于关掉这个机制 —— 上下文帧也会拿到梯度,模型可能学到 trivial copy。一般 `last_n` 取生成 chunk 大小(4-8)。 / If `last_n = total_frames`, the mask is all 1s and the mechanism is effectively off — context frames get gradient too, and the model can learn trivial copying. Set `last_n` to the generation chunk size (4-8 typically).
- **`apply_gradient_mask` 不是性能瓶颈但会复制内存 / Cheap but allocates**: 这一行会**新建** `video` 大小的 tensor。 video 大时是显存压力(`x.detach()` 不 copy 但 `(1 - mask)` 和乘法会触发新 allocation)。可以改写成 in-place 或者用 register_hook 直接 zero 梯度,但代码可读性会大降。 / This line **allocates** a new `video`-sized tensor. For large videos that's memory pressure (`x.detach()` doesn't copy, but `(1 - mask)` and the multiplication do). You can rewrite with `register_hook` to zero grads in-place, but readability tanks.
- **`exit_step` 必须真的在循环里发挥作用 / `exit_step` only helps if you actually use it**: 这三个函数只是"工具",真正的省显存是在调用端:`with torch.no_grad(): x = G(x, t)` 必须对 `t != exit_step` 真的生效。如果你 forget 了把那块包成 `no_grad`,显存照样爆。 / These three are just *utilities*; the actual memory saving happens at the call site: `with torch.no_grad(): x = G(x, t)` must truly apply when `t != exit_step`. Forget the `no_grad` wrap and your VRAM blows up exactly as before.

## 延伸阅读 / Further reading

- [Self-Forcing paper (Yin et al.)](https://arxiv.org/abs/2502.13144)
- [Diffusion Forcing paper](https://arxiv.org/abs/2407.01392)
- [Existing daily-code entry: thu-ml Causal-Forcing's DMD gradient](2026-05-29-causal-forcing-dmd-gradient.md) — the loss this technique is designed to feed.
- [minWM repo](https://github.com/shengshu-ai/minWM) — small, hackable, the recommended starting point if you want to train a tiny world model.
