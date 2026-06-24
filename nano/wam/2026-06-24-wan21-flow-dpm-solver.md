---
date: 2026-06-24
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/utils/fm_solvers.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/main/wan/utils/fm_solvers.py
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, flow-matching, dpm-solver, ode-solver, video-diffusion, sampler]
build_role: sampler-inference (cross-repo variant — FlowDPMSolverMultistepScheduler in Wan2.1)
---

# FlowDPMSolverMultistepScheduler：阶数自动升级的 ODE 求解器 / FlowDPMSolverMultistepScheduler: An Auto-Order-Escalating ODE Solver for Flow Matching

> **一句话 / In one line**: Wan2.1 把 DPM-Solver++ 的多步校正思想移植到 flow-matching（速度预测），用 `lower_order_nums` 计数器实现冷启动期自动从 1 阶升到 2 阶再升到 3 阶，每步最多只调用一次模型。/ Wan2.1 ports DPM-Solver++'s multistep-correction idea to flow-matching (velocity prediction), using a `lower_order_nums` counter to automatically escalate from 1st → 2nd → 3rd order during warm-up, calling the model only once per step.

## 为什么重要 / Why this matters

Flow-matching 模型预测的是"速度"而不是噪声，导致常见的 DDIM scheduler 不能直接用——DDIM 假设 epsilon-prediction，把 x₀ 和 velocity 的关系搞错了。Wan2.1 针对 flow-matching 重写了 DPM-Solver 的三阶 ODE 求解器，并且增加了两个生产级细节：①前几步（历史不足）自动用低阶；②最后几步（`lower_order_final`）也降回一阶，防止数值抖动。结果是只需 6-10 步就能生成高质量视频，比 Euler 步数减半。

Flow-matching models predict "velocity" rather than noise, so vanilla DDIM doesn't apply — DDIM assumes epsilon-prediction and gets the x₀–velocity relationship wrong. Wan2.1 rewrites DPM-Solver's 3rd-order ODE solver specifically for flow-matching, plus adds two production-quality details: ① automatically use lower-order when history is insufficient; ② fall back to 1st-order at the end (`lower_order_final`) to prevent numerical blowup. The result: high-quality video in just 6–10 steps, half the steps of plain Euler.

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/utils/fm_solvers.py`](https://github.com/Wan-Video/Wan2.1/blob/main/wan/utils/fm_solvers.py)

```python
class FlowDPMSolverMultistepScheduler:
    """
    DPM-Solver++ adapted for flow-matching (velocity-prediction) video generation.
    Supports 1st, 2nd, and 3rd order ODE solvers with automatic order escalation.
    """

    def __init__(self, config):
        self.config = config  # solver_order, lower_order_final, euler_at_final, etc.
        self.model_outputs = [None] * config.solver_order  # ring buffer
        self.lower_order_nums = 0   # how many high-quality steps we've accumulated
        self.step_index = 0
        self.timesteps = None
        self.sigmas = None

    def set_timesteps(self, num_inference_steps, device=None):
        """Compute timestep schedule and reset state."""
        sigmas = get_sampling_sigmas(num_inference_steps, self.config.shift)
        self.sigmas = torch.from_numpy(sigmas).to(device)
        self.timesteps = sigmas_to_timesteps(self.sigmas)
        self.model_outputs = [None] * self.config.solver_order
        self.lower_order_nums = 0
        self.step_index = 0

    def convert_model_output(self, model_output, sample=None):
        """
        Flow-matching models predict velocity v = x_1 - x_0 (or dx/dt at sigma).
        Convert velocity prediction to the x_0 estimate expected by the solver.
        """
        sigma = self.sigmas[self.step_index]
        # flow-matching: x_t = (1 - sigma) * x_0 + sigma * noise
        # => x_0 = (x_t - sigma * velocity) / (1 - sigma)
        alpha_t = 1 - sigma
        x0_pred = (sample - sigma * model_output) / alpha_t
        return x0_pred

    def dpm_solver_first_order_update(self, model_output, sample, noise=None):
        """Euler step in flow-matching ODE."""
        sigma_s = self.sigmas[self.step_index]
        sigma_t = self.sigmas[self.step_index + 1]
        # ODE: dx = (x - x0_pred) / sigma ds  => Euler step
        prev_sample = sigma_t / sigma_s * sample + (1 - sigma_t / sigma_s) * model_output
        return prev_sample

    def multistep_dpm_solver_second_order_update(self, model_outputs, sample, noise=None):
        """
        2nd-order multistep DPM-Solver update.
        Uses current (m0) and previous (m1) model outputs to form a D1 correction.
        """
        sigma_s, sigma_t = self.sigmas[self.step_index], self.sigmas[self.step_index + 1]
        sigma_s1 = self.sigmas[self.step_index - 1]   # previous sigma
        m0, m1 = model_outputs[-1], model_outputs[-2]  # current and previous x0-predictions
        lambda_s = -torch.log(sigma_s)
        lambda_t = -torch.log(sigma_t)
        lambda_s1 = -torch.log(sigma_s1)
        h = lambda_t - lambda_s        # current step size in log-sigma space
        h1 = lambda_s - lambda_s1      # previous step size
        r1 = h1 / h                    # step size ratio
        # D0 = m0 (current prediction)
        # D1 = (1/r1) * (m0 - m1) (finite-difference 2nd-order correction)
        D0 = m0
        D1 = (1.0 / r1) * (m0 - m1)
        # 2nd-order ODE update:
        prev_sample = (
            (sigma_t / sigma_s) * sample
            - (torch.exp(-h) - 1) * D0
            - 0.5 * (torch.exp(-h) - 1) * D1
        )
        return prev_sample

    def multistep_dpm_solver_third_order_update(self, model_outputs, sample):
        """3rd-order update using D0, D1, D2 finite-difference corrections."""
        sigma_s, sigma_t = self.sigmas[self.step_index], self.sigmas[self.step_index + 1]
        sigma_s1 = self.sigmas[self.step_index - 1]
        sigma_s2 = self.sigmas[self.step_index - 2]
        m0, m1, m2 = model_outputs[-1], model_outputs[-2], model_outputs[-3]
        lambda_s  = -torch.log(sigma_s)
        lambda_t  = -torch.log(sigma_t)
        lambda_s1 = -torch.log(sigma_s1)
        lambda_s2 = -torch.log(sigma_s2)
        h  = lambda_t - lambda_s
        h1 = lambda_s - lambda_s1
        h2 = lambda_s1 - lambda_s2
        r1 = h1 / h
        r2 = h2 / h
        D0 = m0
        D1 = (1.0 / r1) * (m0 - m1)
        D2 = (1.0 / (r1 * r2)) * (m0 - m1) - (r1 / r2) * (m1 - m2)
        prev_sample = (
            (sigma_t / sigma_s) * sample
            - (torch.exp(-h) - 1) * D0
            - 0.5 * (torch.exp(-h) - 1) * D1
            - (1.0 / 6.0) * (torch.exp(-h) - 1) * D2
        )
        return prev_sample

    def step(self, model_output, timestep, sample, generator=None, return_dict=True):
        """
        Main per-step dispatch: convert model output, update ring buffer,
        then choose 1st / 2nd / 3rd order based on accumulated history.
        """
        # Decide whether to force 1st order at the very last step.
        lower_order_final = (self.step_index == len(self.timesteps) - 1) and (
            self.config.euler_at_final
            or (self.config.lower_order_final and len(self.timesteps) < 15)
            or self.config.final_sigmas_type == "zero"
        )
        lower_order_second = (
            (self.step_index == len(self.timesteps) - 2)
            and self.config.lower_order_final
            and len(self.timesteps) < 15
        )

        # Convert velocity prediction to x0 prediction for the solver.
        model_output = self.convert_model_output(model_output, sample=sample)

        # Shift the ring buffer: oldest entry is overwritten.
        for i in range(self.config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output

        # Dispatch to the highest order we can support with current history.
        if self.config.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
            prev_sample = self.dpm_solver_first_order_update(model_output, sample=sample)
        elif self.config.solver_order == 2 or self.lower_order_nums < 2 or lower_order_second:
            prev_sample = self.multistep_dpm_solver_second_order_update(
                self.model_outputs, sample=sample)
        else:
            prev_sample = self.multistep_dpm_solver_third_order_update(
                self.model_outputs, sample=sample)

        # Warm up: increment history counter until we reach max order.
        if self.lower_order_nums < self.config.solver_order:
            self.lower_order_nums += 1

        self.step_index += 1
        if not return_dict:
            return (prev_sample,)
        return SchedulerOutput(prev_sample=prev_sample)
```

## 逐行讲解 / What's happening

1. **`model_outputs` 环形缓冲区**:
   - 中文: 长度等于 `solver_order`（最大 3）。每步把老的输出往左移一位，把新的 x₀ 预测放在最右边。这是多步法的历史窗口。
   - English: Length equals `solver_order` (max 3). Each step shifts old outputs left and places the new x₀ prediction at the right. This is the history window for the multistep method.

2. **`convert_model_output`: flow-matching 特有的转换**:
   - 中文: flow-matching 模型预测速度 v。从 `x_t = (1-σ)*x₀ + σ*ε` 反推：`x₀ = (x_t - σ*v) / (1-σ)`。这一步让 DPM-Solver 的数学对 flow-matching 成立。
   - English: A flow-matching model predicts velocity v. Inverting `x_t = (1-σ)*x₀ + σ*ε` gives `x₀ = (x_t - σ*v) / (1-σ)`. This step makes DPM-Solver's math apply correctly to flow-matching.

3. **`lower_order_nums` 计数器**:
   - 中文: 每成功用过一次历史，计数器加 1。在计数器 < solver_order 时，只能用低阶（没有足够历史）。热身期结束后自动升到最高阶。
   - English: Increments each time history was successfully used. While `lower_order_nums < solver_order`, only lower order is possible (insufficient history). Once warmed up, automatically uses max order.

4. **`lower_order_final` 和 `lower_order_second` 的逻辑**:
   - 中文: 末步（sigma 接近 0）时高阶校正项数值可能溢出，强制退回 1 阶（Euler）。如果总步数 < 15，还会把倒数第二步也降到 2 阶。这是生产代码的防抖动设计。
   - English: At the final step (sigma → 0), the higher-order correction terms can blow up numerically, so force 1st-order (Euler). If total steps < 15, also downgrade the second-to-last step to 2nd order. This is production-grade numerical stability.

5. **`D0 / D1 / D2` 差分项**:
   - 中文: D0 是当前 x₀ 预测。D1 = `(m0 - m1) / r1` 是一阶差分，估计 x₀ 的一阶导数。D2 用到三步历史，估计二阶导数。每高一阶，局部截断误差下降一个量级。
   - English: D0 is the current x₀ prediction. D1 = `(m0 - m1) / r1` is the first finite difference, estimating the first derivative of x₀. D2 uses three-step history for the second derivative. Each additional order drops local truncation error by one magnitude.

6. **`step_index` 在每步末尾递增**:
   - 中文: `sigma_s` 和 `sigma_t` 的查表依赖 `step_index`。`set_timesteps` 会重置为 0，确保每次推理从干净状态开始。
   - English: Sigma lookup for `sigma_s` and `sigma_t` relies on `step_index`. `set_timesteps` resets it to 0, ensuring each inference run starts clean.

## 类比 / The analogy

想象你在预测一辆车的下一秒位置：
- **1 阶（Euler）**：只看现在的速度，假设速度不变。
- **2 阶**：还看上一秒的速度，估算加速度，预测更准。
- **3 阶**：再看上上秒，估算加速度的变化率（加加速度），最准。

但刚起步（没有上一秒的数据）只能用 1 阶。快停车时（sigma≈0）高阶误差会放大，也退回 1 阶。`lower_order_nums` 就是记录"你跑了几秒、积累了几秒历史"的计数器。

Imagine predicting a car's position one second ahead:
- **1st order (Euler)**: only look at current velocity — assume constant speed.
- **2nd order**: also look at the previous second's velocity — estimate acceleration.
- **3rd order**: also look two seconds back — estimate the rate of change of acceleration.

But at the very start (no history yet) you must use 1st order. When nearly stopped (sigma ≈ 0), higher-order error terms blow up — fall back to 1st order. `lower_order_nums` tracks "how many seconds of history you've accumulated."

## 在 nanoWAM 中的位置 / Where this lives in your nanoWAM

这个组件属于 nanoWAM 课程的 `sampler-inference` 模块，依赖 `noise-scheduler`（定义 timestep schedule 和 sigma 序列）和 `dit-block`（模型本身，提供每步的 model_output）。

This is the `sampler-inference` component of the nanoWAM curriculum, depending on `noise-scheduler` (timestep schedule and sigma sequence) and `dit-block` (the model that produces `model_output` each step).

中文：在你的 nanoWAM 推理循环里，这个 scheduler 替换掉朴素的 DDIM：先调 `scheduler.set_timesteps(num_steps)`，然后在 sigma 序列上循环调用 `scheduler.step(model_output, t, x_t)`。scheduler 内部处理阶数升级和 CFG 之后的 model_output 转换，你的推理代码不需要知道是 1 阶还是 3 阶。把这个 scheduler 插入后，10 步生成质量相当于朴素 Euler 的 20 步。

In your nanoWAM inference loop, this scheduler replaces naive DDIM: first call `scheduler.set_timesteps(num_steps)`, then loop over `scheduler.step(model_output, t, x_t)` at each sigma. The scheduler handles order escalation and the velocity→x₀ conversion internally — your inference code doesn't need to know whether 1st or 3rd order is running. Plugging this in, 10 steps match naive Euler quality at 20 steps.

上游：model（DiT 块，输出速度 v）、CFG combine（已发生在 model_output 里）
下游：VAE decoder（把 latent x₀ 转成像素）

Upstream: model (DiT blocks outputting velocity v), CFG combine (already applied to model_output)
Downstream: VAE decoder (converts latent x₀ to pixels)

## 自己跑一遍 / Try it yourself

```python
import torch

class ToyFlowScheduler:
    """Minimal 2nd-order flow-matching DPM-Solver."""
    def __init__(self, order=2):
        self.order = order; self.outputs = [None] * order
        self.step_idx = 0; self.low_n = 0; self.sigmas = None

    def set_timesteps(self, n, sigmas=None):
        self.sigmas = sigmas if sigmas is not None else torch.linspace(1, 0, n + 1)
        self.outputs = [None] * self.order; self.step_idx = 0; self.low_n = 0

    def convert(self, v, x):
        s = self.sigmas[self.step_idx]
        return (x - s * v) / (1 - s).clamp(min=1e-7)   # velocity → x0

    def step(self, v, x):
        m = self.convert(v, x)
        self.outputs = self.outputs[1:] + [m]
        ss, st = self.sigmas[self.step_idx], self.sigmas[self.step_idx + 1]
        if self.low_n < 1 or self.order < 2:
            out = (st / ss) * x + (1 - st / ss) * m  # Euler
        else:
            m0, m1 = self.outputs[-1], self.outputs[-2]
            r = (self.sigmas[self.step_idx - 1] - ss) / (st - ss + 1e-9)
            D0, D1 = m0, (m0 - m1) / r
            h = torch.log(st / ss)
            out = (st / ss) * x - (torch.exp(h) - 1) * D0 - 0.5 * (torch.exp(h) - 1) * D1
        if self.low_n < self.order: self.low_n += 1
        self.step_idx += 1
        return out

sched = ToyFlowScheduler(order=2)
x = torch.randn(1, 4); sched.set_timesteps(5)
for t in range(5):
    velocity = -x * 0.5  # toy: push toward zero
    x = sched.step(velocity, x)
    print(f"step {t}: x.norm={x.norm():.4f}")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
step 0: x.norm=...   # 1st order (Euler, no history)
step 1: x.norm=...   # 2nd order kicks in
...
step 4: x.norm=...   # continues shrinking toward 0
```

中文：step 0 因为没有历史只能走 Euler，step 1 开始 `lower_order_nums` 够 1 了，自动升到 2 阶。norm 应该比纯 Euler 收敛更快。

Step 0 is forced Euler (no history); from step 1, `lower_order_nums` reaches 1 and the solver upgrades to 2nd order. The norm should converge faster than pure Euler.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **CogVideo `DPMPP2MSampler`（今天 tracked 笔记）** / **CogVideo `DPMPP2MSampler` (today's tracked note)**: 完全类似的"old_denoised 历史 + 首/末步退化为 Euler"模式，但适配 epsilon-prediction 而不是 velocity-prediction / exactly the same "old-denoised history + first/last step fallback" pattern but for epsilon-prediction instead of velocity-prediction.
- **HuggingFace Diffusers `DPMSolverMultistepScheduler`** / **HuggingFace Diffusers**: 对应的标准库实现，支持 epsilon / v-prediction / flow-matching 模式切换，多步环形缓冲区与这里结构一致 / the canonical library version supporting epsilon/v-prediction/flow-matching switching, same ring-buffer structure.
- **Wan2.1 `FlowMatchEulerDiscreteScheduler`（同目录）** / **Wan2.1 `FlowMatchEulerDiscreteScheduler` (same directory)**: 1 阶 Euler baseline，对比读懂为什么需要多步法 / 1st-order Euler baseline — read alongside to understand why the multistep upgrade matters.

## 注意事项 / Caveats / when it breaks

- **必须在每次推理前调 `set_timesteps()`** / **Must call `set_timesteps()` before each inference run**: 重置 `lower_order_nums`、`step_index` 和环形缓冲区。漏掉这一步会导致历史污染，输出质量降级。
- **`convert_model_output` 在 sigma≈1 时分母很小** / **`convert_model_output` has small denominator when sigma≈1**: `(1 - sigma)` 接近 0，数值不稳定。生产代码加 `clamp(min=1e-7)` 防护；或者第一步强制用 1 阶 Euler。
- **CFG 必须在 `step()` 之前做完** / **CFG must be applied before calling `step()`**: `model_output` 应该是已经用 guidance_scale 混合过的 CFG 输出，scheduler 不会帮你再做 CFG。

## 延伸阅读 / Further reading

- [DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models (Lu et al.)](https://arxiv.org/abs/2211.01095)
- [Flow Matching for Generative Modeling (Lipman et al., 2022)](https://arxiv.org/abs/2210.02747)
- [Wan2.1 Technical Report](https://github.com/Wan-Video/Wan2.1)
