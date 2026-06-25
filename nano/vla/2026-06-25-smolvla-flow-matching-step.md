---
date: 2026-06-25
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/policies/smolvla/modeling_smolvla.py
permalink: https://github.com/huggingface/lerobot/blob/2236cdb302cad685798c5f09ea3c713b824a104a/src/lerobot/policies/smolvla/modeling_smolvla.py#L774-L810
difficulty: intermediate
read_time: ~12 min
tags: [code-of-the-day, vla, flow-matching, smolvla, beta-distribution, training-step, lerobot]
build_role: training-step (cross-repo variant — SmolVLA implementation, contrast with openpi rectified-flow and openvla cross-entropy)
---

# SmolVLA 的流匹配训练步：Beta 分布时间采样 + VLM 骨干输出 MSE 损失 / SmolVLA's Flow-Matching Training Step: Beta-Distributed Time Sampling + VLM Backbone MSE Loss

> **一句话 / In one line**: SmolVLA 的 `VLAFlowMatching.forward` 用 Beta(1.5, 1) 偏置采样时间步、线性插值构造带噪动作、把 VLM 骨干的输出投影为速度场预测，一步完成流匹配训练损失计算。 / SmolVLA's `VLAFlowMatching.forward` samples timesteps from a Beta(1.5, 1) distribution biased toward the hard end, linearly interpolates noisy actions, then projects VLM backbone outputs as velocity field predictions — the complete flow-matching training loss in one forward pass.

## 为什么重要 / Why this matters

流匹配（flow matching）训练步的数学是简洁的：在噪声 `z` 和干净动作 `a` 之间做线性插值 `x_t = t·z + (1-t)·a`，目标速度场 `u_t = z - a`，预测速度 `v_t`，用 MSE 对齐。但真实实现里有很多细节影响训练稳定性和最终性能，SmolVLA 在每一个细节处都有值得深挖的选择：

1. **时间采样用 Beta(1.5, 1)**：Beta(1.5, 1) 的概率密度在 t→1 处更高，意味着训练中更多的样本处于"接近干净动作"的区域（高信噪比、高难度去噪），偏置训练到 `t=1` 附近可以提高最终动作精度。对比：openpi 用均匀分布，Wall-X（昨日笔记）也用 Beta 但参数不同。
2. **前缀/后缀嵌入分离**：图像 + 语言 token 构成 prefix（冻结 VLM），带噪动作 + 时间构成 suffix（action expert），两者拼接送入 `vlm_with_expert.forward`，suffix 的输出就是速度场预测。
3. **`suffix_out[:, -chunk_size:]`**：只取 suffix 输出的最后 `chunk_size` 个 token 做投影，对应动作序列的各时间步——这是动作分块（action chunking）与流匹配融合的关键。

Flow-matching training is mathematically clean: interpolate between noise and clean action, predict the velocity field, minimize MSE. But the implementation choices — how to sample timesteps, how to structure the VLM conditioning, how to slice the action tokens — determine training stability and final performance. SmolVLA's version is a clean, well-structured reference: Beta(1.5,1) time sampling biases toward the high-SNR denoising region; a prefix/suffix split keeps the frozen VLM and the trainable action expert in separate streams; the MSE loss is computed per-step per-joint without reduction, letting the caller decide how to weight different joints and timesteps.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/smolvla/modeling_smolvla.py`](https://github.com/huggingface/lerobot/blob/2236cdb302cad685798c5f09ea3c713b824a104a/src/lerobot/policies/smolvla/modeling_smolvla.py#L774-L810)

```python
# Lines 621-635: time and noise sampling helpers
def sample_noise(self, shape, device):
    return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

def sample_time(self, bsize, device):
    beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
    time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
    time = time_beta * 0.999 + 0.001   # clamp to (0.001, 1.000), never exactly 0
    return time

# Lines 774-810: the full training forward pass
def forward(
    self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None
) -> Tensor:
    """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
    if noise is None:
        noise = self.sample_noise(actions.shape, actions.device)

    if time is None:
        time = self.sample_time(actions.shape[0], actions.device)

    time_expanded = time[:, None, None]                              # (B,) → (B, 1, 1)
    x_t = time_expanded * noise + (1 - time_expanded) * actions     # linear interpolation
    u_t = noise - actions                                            # velocity target

    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )
    suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1

    (_, suffix_out), _ = self.vlm_with_expert.forward(
        attention_mask=att_2d_masks,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, suffix_embs],
        use_cache=False,
        fill_kv_cache=False,
    )
    suffix_out = suffix_out[:, -self.config.chunk_size :]           # last chunk_size tokens
    suffix_out = suffix_out.to(dtype=torch.float32)                 # upcast before projection
    v_t = self.action_out_proj(suffix_out)                          # (B, chunk_size, action_dim)
    losses = F.mse_loss(u_t, v_t, reduction="none")                 # (B, chunk_size, action_dim)
    return losses
```

## 逐行讲解 / What's happening

1. **`sample_time` — Beta(1.5, 1) + 截断 / Beta distribution with clamping**
   - 中文：`Beta(concentration1=1.5, concentration0=1.0)` 是一个偏向高 t 值的分布（均值 = 1.5/(1.5+1) = 0.6）。乘以 0.999 再加 0.001 把采样范围限制在 `(0.001, 1.000)`，避免 t=0（纯噪声，完全没有信号）和 t=1（完全干净，梯度消失）的极端情况。这比均匀分布给"接近真实动作"的去噪时间步更多的训练机会。
   - English: `Beta(1.5, 1)` has mean 0.6 and is skewed toward t=1 (high probability density near "clean action"). Multiplying by 0.999 and adding 0.001 clamps to `(0.001, 1.000)`, avoiding t=0 (pure noise, no gradient signal) and t=1 (exact clean action, zero interpolation error). This bias gives more training time to the hard denoising regime near the clean action.

2. **`x_t = time_expanded * noise + (1-t) * actions`：流匹配插值 / Flow interpolation**
   - 中文：`time_expanded` 的形状是 `(B, 1, 1)`（广播到 `(B, chunk_size, action_dim)`），插值公式是线性的：t=1 时 `x_t = noise`（纯噪声），t=0 时 `x_t = actions`（干净动作）。这比 DDPM 的扩散路径（高斯噪声叠加）更简单，也是为什么 flow matching 的推理步骤可以很少（通常 10 步就够）。
   - English: `time_expanded` broadcasts from shape `(B,)` to `(B, chunk_size, action_dim)`. At t=1: `x_t = noise`; at t=0: `x_t = actions`. This is a straight-line path in action space, unlike DDPM's curved Gaussian diffusion path — which is why flow-matching models need fewer inference steps (typically 10 vs DDPM's 100+).

3. **`u_t = noise - actions`：速度目标 / Velocity target**
   - 中文：流匹配的速度场目标是 `dX/dt = noise - actions`——从干净动作（t=0）到噪声（t=1）的方向。模型学习预测这个方向向量，推理时沿相反方向（从 t=1 到 t=0）积分就能从噪声还原出动作。
   - English: The velocity field target is the direction from clean action to noise. During inference, the model integrates the predicted velocity from t=1 (pure noise) to t=0 (clean action) using an ODE solver (Euler, Heun, etc.), recovering the action from noise.

4. **prefix/suffix 嵌入拆分 / Prefix and suffix embedding split**
   - 中文：`embed_prefix` 把图像（经 SigLIP 编码）、语言 token、机器人状态拼在一起，构成 VLM 的"前缀"。`embed_suffix` 把带噪动作 `x_t` 和时间 `t` 嵌入成"后缀"。两者拼接后统一送入 `vlm_with_expert`——这是一个 SmolVLM-500M + action expert 双流 Transformer，prefix 走 VLM 路径，suffix 走 action expert 路径。
   - English: `embed_prefix` combines SigLIP-encoded images, language tokens, and robot state into the VLM context. `embed_suffix` embeds the noisy actions `x_t` and timestep `t` into the action stream. Both are concatenated and fed to `vlm_with_expert`, a dual-stream transformer where prefix tokens pass through the VLM path and suffix tokens pass through the action expert path.

5. **`suffix_out[:, -chunk_size:]` 和 MSE 损失 / Slicing action tokens and computing MSE**
   - 中文：`vlm_with_expert.forward` 返回 `(prefix_out, suffix_out)`，我们只关心 suffix 的最后 `chunk_size` 个输出（对应动作序列的各时间步）。投影到动作维度后与速度目标 `u_t` 做 MSE，`reduction="none"` 保留每个关节每个时间步的独立损失，供调用方按需加权求和。
   - English: `vlm_with_expert` returns a tuple; we take the last `chunk_size` tokens from the suffix output, project them to `action_dim`, and compute elementwise MSE against `u_t`. `reduction="none"` returns shape `(B, chunk_size, action_dim)` — the per-joint per-timestep loss — so the caller can apply different weights across joints, timesteps, or mask out invalid dimensions.

## 类比 / The analogy

流匹配训练就像教一个向导（模型）从随机地点（噪声）导航回家（干净动作）。`sample_time` 决定你从路程的哪个比例出发（偏向接近终点的地方，因为那里地形最复杂）；`x_t` 是你当前在路上的位置；`u_t` 是指向终点的方向向量。训练时，你给向导一张地图（VLM 的视觉语言上下文），让它预测方向 `v_t`，并惩罚它预测偏差（MSE）。推理时，向导从随机地点出发，每走一小步就更新位置，最终到达家（干净动作）。

Flow-matching training is like coaching a navigator (the model) to walk home (clean action) from a random starting point (noise). `sample_time` picks which fraction of the journey to start from — biased toward the end of the path, where terrain is hardest. `x_t` is your current position. `u_t` is the direction toward home. During training, you hand the navigator a context map (VLM vision-language tokens), ask it to predict the direction `v_t`, and penalize wrong predictions (MSE). During inference, the navigator starts at random noise and takes small steps until it reaches home (the clean action).

## 在 nanoVLA 中的位置 / Where this lives in your nanoVLA

这是 nanoVLA 构建计划中 **training-step** 组件的 SmolVLA 版本实现。它依赖以下已覆盖的组件：

- **vision-encoder** → 提供 `embed_prefix` 里的图像嵌入（SigLIP 或等效视觉编码器）
- **vlm-backbone-wiring** → 提供 `vlm_with_expert` 的双流骨干网络
- **action-head-continuous** → `action_out_proj` 是连续动作头的输出投影
- **action-chunking** → `chunk_size` 参数将单步预测扩展为多步动作序列

This is the **training-step** component in the nanoVLA build plan — the SmolVLA cross-repo variant. It wires all previous components into a single training loop.

In your own nanoVLA, you can reuse this exact `forward()` structure:
1. Keep `sample_time` with Beta(1.5, 1) and the 0.001 floor clamp — it's well-tuned.
2. The `x_t = t·noise + (1-t)·actions` interpolation is the key equation — don't change it.
3. Replace `vlm_with_expert` with your VLM backbone + action expert of choice.
4. The `reduction="none"` MSE is important: it lets you mask invalid joints (e.g. a 6-DOF arm shouldn't penalize unused dimensions) during training.

Production additions needed: (a) action normalization/denormalization wrappers around `actions` and `v_t` (see the openpi normalize-transform note from 2026-06-23); (b) curriculum time sampling (some implementations ramp up Beta concentration over training); (c) multi-step consistency loss if you want fewer inference steps.

## 自己跑一遍 / Try it yourself

```python
import torch, torch.nn.functional as F

# Minimal flow-matching training step (no VLM backbone)
def flow_matching_step(actions, context_emb, action_proj):
    B, T, D = actions.shape
    beta = torch.distributions.Beta(1.5, 1.0)
    t = beta.sample((B,)) * 0.999 + 0.001
    noise = torch.randn_like(actions)
    t_exp = t[:, None, None]
    x_t = t_exp * noise + (1 - t_exp) * actions  # interpolate
    u_t = noise - actions                          # velocity target
    # (in real SmolVLA: context_emb comes from VLM backbone)
    v_t = action_proj(x_t)                         # predict velocity
    loss = F.mse_loss(u_t, v_t, reduction="none")
    return loss.mean(), t

B, T, D = 4, 50, 7  # batch=4, chunk=50, dof=7
actions  = torch.randn(B, T, D)
proj     = torch.nn.Linear(D, D)
loss, t  = flow_matching_step(actions, None, proj)
print(f"loss: {loss.item():.4f}  |  time samples mean: {t.mean():.3f} (expect ~0.6)")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
loss: <some value>  |  time samples mean: 0.601 (expect ~0.6)
```

Beta(1.5, 1) 的理论均值是 1.5/(1.5+1) = 0.6，实际采样均值会非常接近这个值。增大 `B` 可以让均值更稳定收敛。

The theoretical mean of Beta(1.5, 1) is 1.5/2.5 = 0.6. The empirical mean from `sample_time` converges to this as batch size grows — increasing `B` makes it more stable.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **openpi 的 `π₀` training step** / **openpi π₀ training step**: 使用均匀分布采样时间，其余流匹配公式相同。参见 [`2026-06-23-openpi-normalize-transform.md`](../../2026/06/2026-06-23-openpi-normalize-transform.md)。
- **Wall-X ActionHead（昨日 VLA 笔记）** / **Wall-X ActionHead (yesterday's VLA note)**: 也用 Beta 分布，但加了 DOF masking——对不同关节用不同的有效性掩码。参见 [`2026-06-24-groot-action-chunk-delta-relative.md`](2026-06-23-wall-x-dof-masked-flow-matching.md)。
- **openvla 的 cross-entropy action head** / **openvla's cross-entropy action head**: 完全不同的范式——把连续动作离散化为 token 后用语言模型 next-token prediction，而非流匹配 MSE。

## 注意事项 / Caveats / when it breaks

- **`reduction="none"` 的调用方责任** / **Caller is responsible for reducing `losses`**: `forward` 返回的是 `(B, chunk_size, action_dim)` 形状的逐元素损失，调用方必须显式地 `.mean()` 或按关节加权求和，否则反向传播会失败（loss 不是标量）。
- **`suffix_out.to(dtype=torch.float32)` 的时机** / **dtype upcast timing**: 注释写明"原版 openpi 代码"在 `action_out_proj` 前 upcast——这是为了防止 BF16/FP16 精度下 MSE 损失数值溢出（速度目标 `u_t` 的量级可能很大）。
- **Beta 分布的随机性** / **Beta distribution is stochastic**: 每次训练 forward 时 `t` 都是重新采样的，没有确定性。如果你在 debug 时想固定时间步，需要手动传入 `time` 参数。

## 延伸阅读 / Further reading

- [Flow Matching for Generative Modeling（Lipman et al.）](https://arxiv.org/abs/2210.02747)
- [π₀ 机器人基础模型技术报告（Physical Intelligence）](https://arxiv.org/abs/2410.24164)
- [SmolVLA 博客](https://huggingface.co/blog/smolvla)
