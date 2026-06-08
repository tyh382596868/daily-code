---
date: 2026-06-08
topic: huggingface
source: huggingface
repo: huggingface/diffusers
file: src/diffusers/schedulers/scheduling_flow_match_heun_discrete.py
permalink: https://github.com/huggingface/diffusers/blob/f3d42be118f9af7ed9697b686fba09a8bdcd71d1/src/diffusers/schedulers/scheduling_flow_match_heun_discrete.py#L285-L352
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, diffusers, sampler, flow-matching, heun, ode]
---

# 一个 step 调两次模型:Diffusers 的 Heun 二阶 flow-match 采样 / Two model calls per step: diffusers' Heun 2nd-order flow-match sampler

> **一句话 / In one line**: 用一个 `state_in_first_order` 开关,把 Euler 一阶步骤(预测斜率)和 Heun 二阶修正(再算一次斜率,取平均)塞进同一个 `step()` 函数。 / A single `state_in_first_order` flag squeezes the Euler step (estimate the slope) and the Heun correction (recompute the slope and average) into the same `step()` body.

## 为什么重要 / Why this matters

Flow matching 训练的是一个 ODE:`dx/dt = v(x, t)`。最简单的求解器是 Euler——每步 `x += v * dt`。但 Euler 在大 step 下会偏离真正的轨迹(理论上误差是 `O(dt)`),所以 SD3 / Flux / Wan2.1 这些产品级的采样器更倾向 Heun 二阶(误差 `O(dt²)`)。问题是 Heun 要在同一个时间区间里调两次模型:一次"试探"算斜率,一次在落点再算一次斜率,然后用两个斜率的平均更新。在 diffusers 这种一次 `step()` 只期望一次模型 forward 的接口里,这种"双调用"必须分摊到两次 step 调用上,而中间状态(`prev_derivative`, `dt`, `sample`)要由 scheduler 自己存。这段代码就是这个"对外像 Euler、内部 zig-zag 跑 Heun"的实现技巧。

Flow matching trains an ODE: `dx/dt = v(x, t)`. The simplest solver is Euler — `x += v * dt`. But Euler drifts at large steps (its error is `O(dt)`), so production samplers in SD3 / Flux / Wan2.1 prefer Heun 2nd order (`O(dt²)`). The catch is Heun needs to call the model *twice* over the same interval: once to probe the slope, again at the landing spot to recompute, then update with the average. In a diffusers-style API where each `step()` consumes exactly one model output, this "two-call" sequence is split across two consecutive `step()` invocations, with the in-between state (`prev_derivative`, `dt`, `sample`) carried by the scheduler. That alternation — looks like Euler from the outside, runs Heun zig-zag inside — is what this code teaches.

## 代码 / The code

`huggingface/diffusers` — [`src/diffusers/schedulers/scheduling_flow_match_heun_discrete.py`](https://github.com/huggingface/diffusers/blob/f3d42be118f9af7ed9697b686fba09a8bdcd71d1/src/diffusers/schedulers/scheduling_flow_match_heun_discrete.py#L285-L352)

```python
def step(self, model_output, timestep, sample,
         s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0,
         generator=None, return_dict=True):
    if self.step_index is None:
        self._init_step_index(timestep)

    # Upcast to avoid precision issues when computing prev_sample
    sample = sample.to(torch.float32)

    if self.state_in_first_order:
        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]
    else:
        # 2nd order / Heun's method
        sigma = self.sigmas[self.step_index - 1]
        sigma_next = self.sigmas[self.step_index]

    gamma = min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0
    sigma_hat = sigma * (gamma + 1)

    if gamma > 0:
        noise = randn_tensor(model_output.shape, dtype=model_output.dtype,
                             device=model_output.device, generator=generator)
        eps = noise * s_noise
        sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5

    if self.state_in_first_order:
        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        denoised = sample - model_output * sigma
        # 2. convert to an ODE derivative for 1st order
        derivative = (sample - denoised) / sigma_hat
        # 3. Delta timestep
        dt = sigma_next - sigma_hat

        # store for 2nd order step
        self.prev_derivative = derivative
        self.dt = dt
        self.sample = sample
    else:
        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        denoised = sample - model_output * sigma_next
        # 2. 2nd order / Heun's method
        derivative = (sample - denoised) / sigma_next
        derivative = 0.5 * (self.prev_derivative + derivative)

        # 3. take prev timestep & sample
        dt = self.dt
        sample = self.sample

        # free dt and derivative
        # Note, this puts the scheduler in "first order mode"
        self.prev_derivative = None
        self.dt = None
        self.sample = None

    prev_sample = sample + derivative * dt
    prev_sample = prev_sample.to(model_output.dtype)
    self._step_index += 1
    return FlowMatchHeunDiscreteSchedulerOutput(prev_sample=prev_sample)
```

## 逐行讲解 / What's happening

1. **`state_in_first_order` 分支**:
   - 中文: 这是状态机的两个状态。`True` 时 `step()` 的语义是"我在做 Euler 一阶预测",`False` 时是"我在做 Heun 二阶修正"。`self.state_in_first_order` 由 `prev_derivative` 是否为 None 间接决定——存了就在二阶模式。
   - English: a two-state state machine. `True` means "this `step()` is the Euler first-order probe", `False` means "this `step()` is the Heun second-order correction". `state_in_first_order` is implicitly driven by `prev_derivative` being None or not — populated means we're in second-order mode.

2. **`sigma_hat = sigma * (gamma + 1)` 加噪 ("churn")**:
   - 中文: 可选的"搅动",把 ODE 临时变成 SDE。先把 `sigma` 抬到 `sigma_hat`,然后注入方差为 `sigma_hat² - sigma²` 的高斯噪声"补齐"扩散。`s_churn=0` 时这段是空操作,sampler 就是纯确定性 Heun。
   - English: optional "churn" turns the ODE temporarily into an SDE. Raise `sigma` to `sigma_hat`, then inject Gaussian noise of variance `sigma_hat² - sigma²` to fill the gap. With `s_churn=0` this branch is a no-op and the sampler is purely deterministic Heun.

3. **first-order 分支 `derivative = (sample - denoised) / sigma_hat`**:
   - 中文: 把模型输出(预测的 noise/velocity)转成 ODE 斜率。这一步只用了一次模型,落点先按 Euler 算出 `prev_sample`,但**真正的 Heun 修正还没做**——`prev_derivative` 和 `sample` 被存起来等下一次 step。
   - English: convert the model output (predicted noise / velocity) into an ODE slope. Only one model call so far; the Euler landing point is computed, but **the Heun correction hasn't happened yet** — `prev_derivative` and `sample` are stashed for the next call.

4. **second-order 分支 `derivative = 0.5 * (prev_derivative + derivative)`**:
   - 中文: 这才是 Heun 的精髓——拿"区间起点的斜率"和"区间终点的斜率"取平均,作为整个区间的代表斜率。然后 `prev_sample = sample + derivative * dt` 用的是**上一次**的 `sample` 和 `dt`,本次 step 实际上修正了上一次的输出。
   - English: this is the Heun trick — average the "slope at the interval's start" with the "slope at the interval's end" as a more accurate representative slope. Then `prev_sample = sample + derivative * dt` uses the **previous** `sample` and `dt`, so this call effectively rewrites last call's output with a corrected one.

5. **`self._step_index += 1`**:
   - 中文: 注意 `step_index` 每个 `step()` 都加一次,但有效的"前进时间步"是每两次 step 才前进一次(`sigma_next` 在二阶模式下对齐到上次的 `sigma`)。所以总 step 数是 timesteps 数 × 2 -1。
   - English: `step_index` advances every call, but the *effective* time progress is one logical step per two `step()` calls (`sigma_next` realigns in second-order mode). Total step count is roughly 2× timesteps − 1.

## 类比 / The analogy

Euler 像开车时只看正前方的速度计——这一刻多快就按这速度走 10 秒。Heun 像更老练的司机:看一眼当前速度,凭直觉滑到 10 秒后的位置,看那边的速度计,再回头取两个速度的平均当作平均车速决定真正走多远。同一个里程,Heun 算两次速度但走得更准。Diffusers 把这种"看两眼"分成两个 step 调用,中间把"试探落点的位置"放在抽屉里(`self.sample`),等第二眼回来再用。

Euler is like driving while glancing only at your current speedometer — go that fast for 10 seconds. Heun is the more careful driver: read the current speed, mentally coast 10 seconds ahead, read the speedometer there, then back up and average the two speeds for the actual distance. Same odometer goal, but Heun reads twice and moves more accurately. Diffusers splits this "two glances" into two `step()` calls, parking the probed landing spot in a drawer (`self.sample`) until the second glance comes back.

## 自己跑一遍 / Try it yourself

```python
# try.py — minimal "two-step Heun" loop on a 1D ODE: dy/dt = -y
import torch

class MiniHeun:
    def __init__(self, sigmas):
        self.sigmas = sigmas          # decreasing schedule, e.g. [1.0, 0.5, 0.0]
        self.i = 0
        self.first_order = True
        self.prev_d = self.prev_s = self.prev_dt = None

    def step(self, v_pred, y):
        if self.first_order:
            sig, sig_next = self.sigmas[self.i], self.sigmas[self.i + 1]
            d = v_pred                              # Euler slope
            self.prev_d, self.prev_s, self.prev_dt = d, y, sig_next - sig
            y_new = y + d * (sig_next - sig)
            self.first_order = False
            return y_new                            # an Euler-quality estimate
        else:
            d = 0.5 * (self.prev_d + v_pred)        # average two slopes
            y_new = self.prev_s + d * self.prev_dt  # rewrite last update
            self.i += 1
            self.first_order = True
            return y_new

# ODE: dy/dt = -y, true solution y(t) = exp(-t)
sigmas = torch.linspace(1.0, 0.0, 4)
h = MiniHeun(sigmas)
y = torch.tensor(2.7183)                            # y(0) = e
for _ in range(2 * (len(sigmas) - 1)):
    v_pred = -y                                     # the "model"
    y = h.step(v_pred, y)
print(f"Heun final y    = {y.item():.4f}")
print(f"True y(1)=e^-1  = {torch.exp(torch.tensor(-1.)).item():.4f}")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
Heun final y    ≈ 0.3679
True y(1)=e^-1  ≈ 0.3679
```

注意 Euler 在同样 3 个外部时间步下会算到 ≈ 0.296(偏差 ~20%),Heun 几乎吻合解析解——这就是二阶的差距。

Euler with the same 3 logical time steps lands at ≈ 0.296 (~20% off); Heun matches the analytic solution. That's the second-order win.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **k-diffusion / Karras 系采样器 / k-diffusion samplers (Karras family)**: 同样的 Heun 实现思路,EDM 论文里给出了最佳的 sigma 调度 / Same Heun implementation pattern; the EDM paper specifies the optimal sigma schedule.
- **DPM-Solver / UniPC 的 multi-step / DPM-Solver and UniPC's multi-step variants**: 同样靠 scheduler 缓存历史导数来做高阶矩阵法 / Also cache historical derivatives in the scheduler to enable higher-order multistep methods.
- **流体仿真里的 RK2 / RK2 in fluid simulation**: 数学上的 Heun 就是 RK2 的一个特例;CFD 求解器里到处是 / Heun is a special case of RK2; you'll find it everywhere in CFD solvers.

## 注意事项 / Caveats / when it breaks

- **采样器调用次数翻倍** / **2× number of model calls**: NFE = `2 * len(timesteps) - 1`,生成成本比 Euler 高近一倍 / NFE = `2 * len(timesteps) - 1`, almost double the model calls vs. Euler.
- **混 `state_in_first_order` 与外部 step counter** / **State leaks between samples**: scheduler 是有状态的,一次 sampling 完了必须 `set_timesteps()` 重置,否则下一次会从二阶状态启动 / The scheduler is stateful; call `set_timesteps()` to reset before each new sample, otherwise the next run starts mid-Heun.
- **churn 注入噪声的 sigma 区间要选对** / **Pick the churn window carefully**: `s_tmin/s_tmax` 之外不加噪,设错了等于把 sampler 退化回纯确定性 / Outside the `s_tmin/s_tmax` window no noise is injected; misconfigure them and you silently fall back to a deterministic sampler.

## 延伸阅读 / Further reading

- Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models" (EDM) — the canonical reference for Heun + churn in diffusion
- Flow Matching paper (Lipman et al., 2023) — the formulation this scheduler targets
- diffusers Flow-Match Euler scheduler (same dir) — diff against to see what the 2nd-order branch adds
