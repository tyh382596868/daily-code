---
date: 2026-06-24
topic: diffusion
source: tracked
repo: THUDM/CogVideo
file: sat/sgm/modules/diffusionmodules/sampling.py
permalink: https://github.com/THUDM/CogVideo/blob/main/sat/sgm/modules/diffusionmodules/sampling.py
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, dpm-solver, video-diffusion, multistep-sampler]
---

# DPM-Solver++(2M)：用历史预测做二阶修正的视频扩散采样器 / DPM-Solver++(2M): 2nd-Order Multistep Correction via History Tracking in Video Diffusion

> **一句话 / In one line**: 每步末尾把 denoised 预测存进 `old_denoised`，下一步用两点差分做二阶修正——首步或末步自动退化为 Euler。/ Each step saves the denoised prediction as `old_denoised`; the next step blends current + previous to get a 2nd-order correction — first and last steps automatically fall back to Euler.

## 为什么重要 / Why this matters

一般扩散推理用 DDIM / Euler，每步只看当前的去噪预测。DPM-Solver++(2M) 多记一个量：上一步的去噪输出 `old_denoised`。用两步差分近似二阶导数，让每个时间步的误差从 O(h²) 降到 O(h³)。实际效果是：同等步数质量提升明显，或者同等质量步数砍半。CogVideoX 系列视频生成把它作为默认采样器。

The standard DDIM/Euler sampler looks only at the current denoised estimate. DPM-Solver++(2M) adds a single extra state variable — `old_denoised` from the previous step. Using a two-point finite difference to approximate the second derivative, it upgrades local truncation error from O(h²) to O(h³). In practice this means noticeably better quality at the same step count, or the same quality with half the steps. CogVideoX uses it as the default video generation sampler.

## 代码 / The code

`THUDM/CogVideo` — [`sat/sgm/modules/diffusionmodules/sampling.py`](https://github.com/THUDM/CogVideo/blob/main/sat/sgm/modules/diffusionmodules/sampling.py)

```python
class DPMPP2MSampler(BaseDiffusionSampler):
    """DPM-Solver++(2M) — 2nd-order multistep diffusion sampler."""

    def get_variables(self, sigma, next_sigma, previous_sigma=None):
        t, t_next = [to_neg_log_sigma(s) for s in (sigma, next_sigma)]
        h = t_next - t
        if previous_sigma is not None:
            h_last = t - to_neg_log_sigma(previous_sigma)
            r = h_last / h
        else:
            r = None
        return h, r, t, t_next

    def get_mult(self, h, r, t, t_next, previous_sigma):
        mult1 = to_sigma(t_next) / to_sigma(t)
        mult2 = (-h).expm1()
        if previous_sigma is not None:
            mult3 = 1 + 1 / (2 * r)
            mult4 = 1 / (2 * r)
        else:
            mult3 = None
            mult4 = None
        return mult1, mult2, mult3, mult4

    def sampler_step(
        self,
        old_denoised,
        previous_sigma,
        sigma,
        next_sigma,
        denoiser,
        x,
        cond,
        uc=None,
    ):
        denoised = self.denoise(x, denoiser, sigma, cond, uc)

        h, r, t, t_next = self.get_variables(sigma, next_sigma, previous_sigma)
        mult = [
            append_dims(mult, x.ndim)
            for mult in self.get_mult(h, r, t, t_next, previous_sigma)
        ]

        # 1st-order (Euler) update — always computed as the safe fallback
        x_standard = mult[0] * x - mult[1] * denoised

        if old_denoised is None or torch.sum(next_sigma) < 1e-14:
            # First step, or final step (next_sigma ≈ 0): use Euler
            return x_standard, denoised
        else:
            # 2nd-order correction: blend current and previous denoised
            denoised_d = mult[2] * denoised - mult[3] * old_denoised
            x_advanced = mult[0] * x - mult[1] * denoised_d
            # For any batch item where next_sigma == 0, revert to Euler
            x = torch.where(
                append_dims(next_sigma, x.ndim) > 0.0, x_advanced, x_standard
            )
        return x, denoised

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, **kwargs):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        old_denoised = None
        for i in self.get_sigma_gen(num_sigmas):
            x, old_denoised = self.sampler_step(
                old_denoised,
                None if i == 0 else sigmas[i - 1],
                sigmas[i],
                sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc=uc,
            )
        return x
```

## 逐行讲解 / What's happening

1. **`sampler_step` 签名中的 `old_denoised`**:
   - 中文: 这是二阶多步法的核心——把上一步的 denoised 预测传进来。首步传 `None`。
   - English: This is the key state variable — the previous step's denoised prediction. Pass `None` on the first step.

2. **`denoised = self.denoise(x, denoiser, sigma, cond, uc)`**:
   - 中文: 对当前带噪输入 x 调用去噪器（通常是 CFG 组合后的 U-Net / DiT），得到 x₀ 的预测值。
   - English: Run the denoiser (typically a CFG-combined U-Net/DiT) on the current noisy x to get the x₀ estimate.

3. **`get_variables` / `get_mult`**:
   - 中文: `h = t_next - t`（负对数 sigma 空间里的步长），`r = h_last / h`（前后步长之比，用于二阶混合权重）。`mult1` 到 `mult4` 是 DPM-Solver 论文里的 λ 系数。
   - English: `h` is the step size in log-sigma space, `r` is the ratio of the previous step length to the current step — it sets the blending weights for the 2nd-order correction. `mult1`–`mult4` are the λ coefficients from the DPM-Solver paper.

4. **`x_standard = mult[0] * x - mult[1] * denoised`**:
   - 中文: 标准 Euler 步。无论如何都要算，因为 `torch.where` 需要两个分支。
   - English: The standard 1st-order (Euler) update. Always computed because `torch.where` needs both branches.

5. **`if old_denoised is None or torch.sum(next_sigma) < 1e-14`**:
   - 中文: 两种必须退化为 Euler 的情况：①首步没有历史；②末步 next_sigma≈0，继续高阶修正数值不稳定。
   - English: Two cases that must fall back to Euler: ① first step (no history yet); ② final step (next_sigma≈0 — the correction terms blow up numerically at the boundary).

6. **`denoised_d = mult[2] * denoised - mult[3] * old_denoised`**:
   - 中文: 二阶修正项 D₁。用 `(1 + 1/(2r)) * denoised - (1/(2r)) * old_denoised` 做有限差分估计二阶导数。
   - English: The 2nd-order correction term D₁. Uses `(1 + 1/(2r)) * denoised - (1/(2r)) * old_denoised` — a finite-difference estimate of the second-order derivative of the denoised trajectory.

7. **`x = torch.where(append_dims(next_sigma, x.ndim) > 0.0, x_advanced, x_standard)`**:
   - 中文: 逐 batch item 判断。如果某张图/视频帧的 next_sigma 已经是 0（完成了），就用 Euler 结果；否则用二阶修正。
   - English: Per-batch conditional: if any sample's `next_sigma` is 0 (finished), use the Euler result; otherwise use the 2nd-order result. `append_dims` broadcasts the scalar sigma to match x's shape.

8. **`__call__` 里的 `old_denoised = None` 初始化**:
   - 中文: 采样循环把每步的 denoised 输出传给下一步的 `old_denoised`，实现滑动窗口历史记录。
   - English: The sampling loop feeds each step's denoised output into the next call's `old_denoised`, implementing a sliding-window history of length 1.

## 类比 / The analogy

开车时只看当前车速来预测刹车距离，误差很大。如果同时看"上一秒的车速"，就能估计加速度，预测更准——这就是 `old_denoised`。首步（刚启动）和末步（已经到站）没有加速度参考，退回到只看当前速度（Euler）。

Think of estimating your stopping distance while driving. If you only look at your current speed, your estimate is rough. But if you also recall your speed one second ago, you can estimate acceleration and predict much more accurately — that's `old_denoised`. At the very first step (just starting) and the very last step (nearly stopped), you have no meaningful acceleration reference, so you fall back to the current-speed-only estimate (Euler).

## 自己跑一遍 / Try it yourself

```python
import torch

def simple_dpmpp2m_step(x, sigma, next_sigma, prev_sigma, denoised, old_denoised):
    """Minimal DPM-Solver++(2M) step in log-sigma space."""
    log_s  = -torch.log(sigma.clamp(min=1e-9))
    log_ns = -torch.log(next_sigma.clamp(min=1e-9))
    log_ps = -torch.log(prev_sigma.clamp(min=1e-9)) if old_denoised is not None else None

    h = log_ns - log_s
    mult1 = (next_sigma / sigma)
    mult2 = (-h).expm1()

    x_euler = mult1 * x - mult2 * denoised

    if old_denoised is None or next_sigma.abs() < 1e-7:
        return x_euler

    h_last = log_s - log_ps
    r = h_last / h
    mult3 = 1 + 1 / (2 * r)
    mult4 = 1 / (2 * r)
    denoised_d = mult3 * denoised - mult4 * old_denoised
    return mult1 * x - mult2 * denoised_d

torch.manual_seed(0)
x = torch.randn(1, 4, 8, 8)
sigmas = torch.tensor([1.0, 0.5, 0.1, 0.0])

old_denoised = None
for i in range(len(sigmas) - 1):
    sigma, next_sigma = sigmas[i], sigmas[i + 1]
    prev_sigma = sigmas[i - 1] if i > 0 else None
    denoised = x * 0.9  # toy denoiser: shrink x
    x = simple_dpmpp2m_step(x, sigma, next_sigma, prev_sigma, denoised, old_denoised)
    old_denoised = denoised
    print(f"step {i}: x.norm={x.norm():.4f}")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
step 0: x.norm=1.xxxx   # Euler (no history yet)
step 1: x.norm=1.xxxx   # 2nd-order (old_denoised available)
step 2: x.norm=1.xxxx   # Euler again (next_sigma=0 final step)
```

中文：注意第 0 步（首步，无历史）和第 2 步（末步，next_sigma=0）都走 Euler 路径。

Notice that step 0 (no history) and step 2 (final step, `next_sigma=0`) both take the Euler path. Only the middle steps use the 2nd-order correction.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **`EulerEDMSampler`（同文件）** / **`EulerEDMSampler` (same file)**: 只做 1 阶，没有 `old_denoised`，对比读一目了然 / 1st-order only, no `old_denoised` — compare the two to see the minimal delta.
- **Wan2.1 `FlowDPMSolverMultistepScheduler`** / **Wan2.1 `FlowDPMSolverMultistepScheduler`**: 今天 WAM 笔记的主题——同一思路延伸到 3 阶，并适配 flow-matching（速度预测而非噪声预测）/ today's WAM note — extends the same idea to 3rd order and adapts it to flow-matching (velocity prediction instead of noise prediction).
- **HuggingFace Diffusers `DPMSolverMultistepScheduler`** / **HuggingFace Diffusers `DPMSolverMultistepScheduler`**: 标准库实现，维护一个 `model_outputs` 环形缓冲区，和这里的 `old_denoised` 单值等价 / the canonical library version — maintains a `model_outputs` ring buffer, equivalent to the single `old_denoised` here.

## 注意事项 / Caveats / when it breaks

- **`old_denoised` 必须和当前 x 同 device / `old_denoised` must be on the same device as x**: 跨设备推理（如 CPU offload）时要显式 `.to(device)` / explicitly `.to(device)` if doing CPU-offload inference.
- **末步 `torch.sum(next_sigma) < 1e-14` 条件是全 batch 的判断** / **The `torch.sum(next_sigma) < 1e-14` check is a whole-batch condition**: 只要有一张图没完成，`torch.where` 仍然选二阶路径。`append_dims` 里的逐元素 `> 0.0` 才是真正的 per-sample 分支 / the `torch.where` inside is the real per-sample branch.
- **步数极少（< 5 步）时慎用** / **Avoid at very low step counts (< 5)**: 历史只有 1 步，r 的估计噪声大，此时 Heun 更稳定 / with only 1 step of history, r is noisy — Heun is more stable at very low NFE.

## 延伸阅读 / Further reading

- [DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps (Lu et al., NeurIPS 2022)](https://arxiv.org/abs/2206.00927)
- [DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models (Lu et al., 2022)](https://arxiv.org/abs/2211.01095)
- [CogVideoX technical report](https://arxiv.org/abs/2408.06072)
