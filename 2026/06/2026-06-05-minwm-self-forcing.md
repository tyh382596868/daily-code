---
date: 2026-06-05
topic: diffusion
source: trending
repo: shengshu-ai/minWM
file: shared/algorithms/self_forcing.py
permalink: https://github.com/shengshu-ai/minWM/blob/74638bc75a68944cbdb6ef01814456a7534cade4/shared/algorithms/self_forcing.py#L12-L83
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, world-model, training, gradient-mask, self-forcing]
---

# Self-Forcing:只在一个步、只在最后几帧反向传播 / Self-Forcing: backprop through one step, only on the last few frames

> **一句话 / In one line**: 训长视频自回归世界模型时,Self-Forcing 只在一个**随机采样的去噪步**上让梯度流,而且只在视频的**最后 N 帧**上让梯度流 — 用 `x*mask + x.detach()*(1-mask)` 这个三行 trick 把"上下文"和"被监督的目标"分开。 / When training long-context autoregressive video world models, Self-Forcing flows gradient through exactly **one randomly sampled denoising step** and only through the **last N frames** of the video, using the three-line `x*mask + x.detach()*(1-mask)` trick to separate "context" from "supervised target".

## 为什么重要 / Why this matters

自回归视频世界模型(像 Wan2.1, CogVideoX, minWM)动不动就是 49 帧 × 50 去噪步 — 完整 backprop 要保存 49 × 50 = 2450 个中间激活,A100 80G 都不够装。Self-Forcing 给出的回答非常聪明:大部分 step 和大部分帧只是"为了得到正确的上下文",并不需要梯度。我只挑**一个 step 一个尾段**做完整 backprop,其余全部 `detach`。这不是近似 — 因为损失函数(通常是 DMD 蒸馏 loss)本身就只看输出帧,数学上等价。这个技术让 single-GPU 训长视频 WAM 变得可能。

Autoregressive video world models (Wan2.1, CogVideoX, minWM) routinely use 49 frames × 50 denoising steps — full backprop would store 49 × 50 = 2450 intermediate activations and not fit on an A100 80G. Self-Forcing's answer is elegant: most steps and most frames only exist to produce correct context, not to be supervised. Pick **one denoising step and one tail segment** for full backprop, `detach()` everything else. It's not an approximation — the loss (typically DMD distillation) only inspects the output frames, so it's mathematically equivalent. This trick is what makes single-GPU long-video WAM training possible.

## 代码 / The code

`shengshu-ai/minWM` — [`shared/algorithms/self_forcing.py`](https://github.com/shengshu-ai/minWM/blob/74638bc75a68944cbdb6ef01814456a7534cade4/shared/algorithms/self_forcing.py#L12-L83)

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

    For masked regions, detach the tensor (stop gradient).
    For unmasked regions, keep the gradient.

    result = video * mask + video.detach() * (1 - mask)
    """
    return video * mask + video.detach() * (1.0 - mask)
```

## 逐行讲解 / What's happening

1. **`sample_exit_step` 的 `same_across_blocks=True` 默认行为**:
   - 中文: 整个 batch 共用一个随机 step idx。意思是"这一步训练时,我决定只在第 17 步那一刻让 generator 留梯度,其他 49 步全 no_grad"。每个 batch 重抽一次。
   - English: the whole batch shares one randomly sampled step index. Translation: "for this training step, I'm letting the generator keep gradient only at denoising-step 17 — the other 49 steps run `no_grad`". Resampled each batch.

2. **为什么 `[idx] * num_blocks` 而不是直接返回 `idx`?**:
   - 中文: 自回归视频是按"block"分段生成的(每个 block 几帧),`num_blocks` 是 block 数量。函数返回每个 block 一个 exit step;`same_across_blocks=True` 时所有 block 用同一个 step(更稳定),`False` 时每个 block 各自抽样(更多样本但训练抖)。
   - English: autoregressive video generates in "blocks" of frames; `num_blocks` is the block count. The function returns one exit step per block. `True` means all blocks share the step (more stable); `False` samples per block (more diverse but jittery).

3. **`create_gradient_mask` 的 `slices[frame_dim] = slice(start, total_frames)`**:
   - 中文: 用动态构造的 `slice` 列表给任意维度切片 — 比手写 `mask[:, :, -N:]` 通用得多,可以处理 `(B, F, C, H, W)` 也能处理 `(B, C, F, H, W)`,只看 `frame_dim` 是哪一维。
   - English: dynamically constructed `slice` list to index any axis — far more general than `mask[:, :, -N:]`, handles both `(B, F, C, H, W)` and `(B, C, F, H, W)` depending on `frame_dim`.

4. **`mask[tuple(slices)] = 1.0`**:
   - 中文: 在最后 N 帧位置写 1,其他位置保持 0。后面 `apply_gradient_mask` 会读这个 mask 决定哪里要梯度。
   - English: writes 1 at the trailing N frames, 0 elsewhere. `apply_gradient_mask` will consult this mask to route gradient.

5. **`return video * mask + video.detach() * (1.0 - mask)`** ✨:
   - 中文: 这是整个文件的灵魂一行。它在 forward 时数值上是 `video`(因为 `video * mask + video.detach() * (1-mask) == video`,数值上等价于原 tensor)。但在 backward 时,`video.detach()` 的那部分梯度被截断了 — 只有 `mask == 1` 的位置(最后 N 帧)能反传。整张视频在前向完全不变,反向只让尾段流过。
   - English: the soul of the file. Forward-pass value is exactly `video` (`video * mask + video.detach() * (1-mask) == video` numerically). But during backward, the `video.detach()` half has its gradient cut — only positions where `mask == 1` (the last N frames) propagate. The full video stays intact forward; only the tail flows backward.

6. **`dtype != torch.bool` 的判断**:
   - 中文: 如果用 bool mask,后面 `video * mask` 会做隐式类型提升,慢。用 float mask 直接乘更快,但 dtype 由调用方决定。
   - English: with a bool mask, `video * mask` triggers implicit dtype promotion (slow). Float mask multiplies directly; the caller chooses the dtype.

## 类比 / The analogy

想象一个长跑接力比赛 40 棒(40 帧),你要训练第 35 到 40 棒之间交棒的动作。如果让前 34 棒全力以赴跑,然后教练根据 35–40 棒的表现复盘——前 34 棒会累死,而且 80% 的训练精力浪费在他们身上。Self-Forcing 的做法是:前 34 棒像录像回放一样照着跑(forward 正常,但被 `detach`,等于"这一段成绩不计入考核"),只在最后 6 棒开梯度 — 教练只对这 6 棒做调整。结果一样:你拿到了完整的 40 棒交棒数据,但训练成本只剩 15%。

Picture a 40-leg relay race (40 frames) where you want to train the handoffs in legs 35–40. If you make legs 1–34 sprint full effort and then have the coach review only legs 35–40, the first 34 runners exhaust themselves and 80% of training cost is wasted. Self-Forcing makes legs 1–34 run "from playback" (forward proceeds normally but `detach()` says "results don't count toward grading") and only opens the gradient on the last 6 legs — the coach only adjusts those. You still get a complete 40-leg recording, but at 15% of the training cost.

## 自己跑一遍 / Try it yourself

```python
# self_forcing_toy.py — pip install torch
import torch

torch.manual_seed(0)
B, F, C = 1, 8, 4              # 8 frames, 4 channels
video = torch.randn(B, F, C, requires_grad=True)
last_n = 3                     # only backprop through last 3 frames

# Build the gradient mask
mask = torch.zeros_like(video)
mask[:, -last_n:, :] = 1.0

# Apply the trick
masked = video * mask + video.detach() * (1 - mask)

# Pretend the loss is "mean over the whole video"
loss = masked.sum()
loss.backward()

print("video.grad per-frame norm:")
for i in range(F):
    g = video.grad[0, i].abs().sum().item()
    print(f"  frame {i}: grad sum = {g:.4f}")
```

运行 / Run with:
```bash
python self_forcing_toy.py
```

预期输出 / Expected output:
```
video.grad per-frame norm:
  frame 0: grad sum = 0.0000
  frame 1: grad sum = 0.0000
  frame 2: grad sum = 0.0000
  frame 3: grad sum = 0.0000
  frame 4: grad sum = 0.0000
  frame 5: grad sum = 4.0000
  frame 6: grad sum = 4.0000
  frame 7: grad sum = 4.0000
```

中文:前 5 帧梯度精确为 0(被 detach),后 3 帧梯度照常 = 4(因为每帧 4 个通道、每个通道贡献 1)。整个 video 的 forward 输出和 loss 完全正常,只是反向被精确雕琢了。

English: the first 5 frames have exactly zero gradient (detached), the last 3 have normal gradient = 4 (4 channels × 1 each). Forward output and loss are entirely normal; only the backward pass is surgically sculpted.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Causal Forcing (thu-ml, 2026-05-29 trending)** / **Causal Forcing**: 自回归视频蒸馏的兄弟工作,用同样的 `x*mask + x.detach()*(1-mask)` 思路控制 DMD 梯度的范围。 / Sibling work on autoregressive video distillation; uses the same `x*mask + x.detach()*(1-mask)` trick to scope DMD gradient.
- **Reptile / MAML 的 first-order 实现** / **Reptile / first-order MAML**: 也是"通过 detach 控制哪些参数有 inner-loop gradient",虽然语义不同但 trick 一样。 / "Use detach to choose which params get inner-loop gradient" — different semantics, same trick.
- **Reverse KL gradient surgery in RLHF** / **RLHF reverse-KL gradient surgery**: PPO 训练时,对 reference policy 的 logit `.detach()` 是同款做法,确保 reference 不参与训练。 / In PPO, `.detach()` on reference policy logits is the same idiom — ensuring reference isn't trained.
- **Gradient checkpointing 的"only checkpoint these layers"** / **gradient checkpointing's "checkpoint only these layers"**: 这条思路的更老的近亲 — 但前者重算 forward,Self-Forcing 直接砍 backward。 / An older cousin — checkpointing recomputes forward, while Self-Forcing kills backward entirely.

## 注意事项 / Caveats / when it breaks

- **`x * mask + x.detach() * (1 - mask)` ≠ `mask * x` (在 backward 上)** / **`x*mask + x.detach()*(1-mask)` ≠ `mask*x` for backward**: 单写 `mask * x`,前向数值变成 `mask*x`,不再等于 `x`,整个流程就错了。这个 identity 是"前向不变 + 后向选择性截断"的唯一干净写法。 / Just writing `mask * x` changes the forward value to `mask*x` (no longer equal to `x`), which breaks everything. The identity is the only clean way to keep forward intact while choosing the backward path.
- **`mask` 必须和 video 在同一 device、同一 dtype** / **`mask` must match video device/dtype**: 否则 `video * mask` 会触发隐式 cast,小心 bf16 训练里出现 fp32 mask 偷偷把激活转 fp32 的情况。 / Otherwise `video * mask` triggers implicit casts; watch out for fp32 mask silently upcasting activations during bf16 training.
- **跨 `num_blocks` 抽 step 时,顺序很关键** / **block-wise step sampling order matters**: `same_across_blocks=False` 时,不同 block 各自的 exit step 决定了 attention 缓存能否复用。生产代码通常按 step 从小到大排序后再切 block,以减少 KV-cache 重建。 / With `same_across_blocks=False`, different blocks' exit steps determine whether attention caches can be reused. Production code sorts steps ascending before block-slicing to minimize KV-cache rebuilds.
- **只对 last N 帧训不会让早期帧学习吗?** / **"Only training last N frames" — won't early frames stop learning?**: 关键是 N 在 batch 间随机滚动,所有帧都会轮到。但**模型必须有时间一致性的归纳偏置**(causal mask + RoPE),否则后面的帧学到的东西迁移不到前面。 / Yes, because `N` rolls across batches — every frame gets its turn eventually. But **the model must have temporal inductive biases** (causal mask + RoPE), otherwise what's learned on later frames doesn't transfer.

## 延伸阅读 / Further reading

- [minWM repo README](https://github.com/shengshu-ai/minWM) — full-stack open-source video WAM framework, this is one of its training primitives
- [Distribution Matching Distillation (DMD) paper](https://tianweiy.github.io/dmd2/) — the loss this trick is built to serve
- [Causal Forcing (ICML 2026)](https://github.com/thu-ml/Causal-Forcing) — sibling autoregressive distillation method
- [Self-Forcing paper / blog](https://arxiv.org/abs/2502.05772) — the original write-up of this training scheme
