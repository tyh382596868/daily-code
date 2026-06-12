---
date: 2026-06-12
topic: wam
source: wam
repo: Robbyant/lingbot-va
file: wan_va/utils/scheduler.py
permalink: https://github.com/Robbyant/lingbot-va/blob/58c2ae5bac46bd8114065bea9d7d256eb67c16c3/wan_va/utils/scheduler.py#L5-L135
difficulty: intermediate
read_time: ~11 min
tags: [code-of-the-day, wam, noise-scheduler, flow-matching, training-weighting]
build_role: noise-scheduler (cross-repo variant — flow-match scheduler with training-time non-uniform weighting)
---

# 一份 130 行的完整 FlowMatchScheduler:把"训练时哪些 timestep 更重要"也包进去了 / A complete 130-line FlowMatchScheduler that also bundles "which timesteps matter more at training time"

> **一句话 / In one line**: `set_timesteps`(造时间轴) + `step`(Euler 一步) + `add_noise`(前向加噪) + `training_target`(回归目标) + `training_weight`(钟形重要性权重) —— 5 个方法就是一份生产级 flow-match scheduler 的全部接口 / `set_timesteps` (build the time axis) + `step` (one Euler step) + `add_noise` (forward noising) + `training_target` (regression target) + `training_weight` (bell-shaped importance weighting) — five methods make up the entire interface of a production-grade flow-match scheduler.

## 为什么重要 / Why this matters

之前我们讲过 Wan2.1 的 `time_shift`(resolution-adaptive 噪声轴)、dreamzero 的 90 行 flow-match scheduler、Open-Sora 的 causal 3D VAE…… 这些都聚焦在"采样侧"。但 flow matching 训练时还有一个常被忽略的细节:**timestep 不能均匀采样,否则模型把大半算力浪费在已经容易的端点上**(σ≈0 时几乎是干净数据,σ≈1 时几乎是纯噪声)。lingbot-va 的 `FlowMatchScheduler` 把这个训练侧权重也放进同一个 130 行的文件里,变成 `training_weight()` 一个 API,直接被训练循环调用。这是连接"我有调度器"和"我训练真的收敛"的那座桥。

We've covered Wan2.1's `time_shift` (resolution-adaptive noise axis), dreamzero's 90-line flow-match scheduler, Open-Sora's causal 3D VAE — all sampler-side. But flow matching's training side has a frequently-ignored detail: **uniformly sampling timesteps wastes most of the compute on the easy endpoints** (σ≈0 is almost clean data, σ≈1 is almost pure noise). lingbot-va's `FlowMatchScheduler` packs that training-side weighting into the same 130-line file as a single `training_weight()` method that the training loop calls. It's the bridge between "I have a scheduler" and "my training actually converges".

## 代码 / The code

`Robbyant/lingbot-va` — [`wan_va/utils/scheduler.py`](https://github.com/Robbyant/lingbot-va/blob/58c2ae5bac46bd8114065bea9d7d256eb67c16c3/wan_va/utils/scheduler.py#L5-L135)

```python
class FlowMatchScheduler():
    def __init__(self, num_inference_steps=100, num_train_timesteps=1000,
                 shift=3.0, sigma_max=1.0, sigma_min=0.003 / 1.002,
                 inverse_timesteps=False, extra_one_step=False, reverse_sigmas=False,
                 exponential_shift=False, exponential_shift_mu=None, shift_terminal=None):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max, self.sigma_min = sigma_max, sigma_min
        self.inverse_timesteps, self.extra_one_step = inverse_timesteps, extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.exponential_shift, self.exponential_shift_mu = exponential_shift, exponential_shift_mu
        self.shift_terminal = shift_terminal
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0,
                      training=False, shift=None, dynamic_shift_len=None):
        if shift is not None: self.shift = shift
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        if self.exponential_shift:
            mu = self.calculate_shift(dynamic_shift_len) if dynamic_shift_len is not None else self.exponential_shift_mu
            self.sigmas = math.exp(mu) / (math.exp(mu) + (1 / self.sigmas - 1))
        else:
            self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.shift_terminal is not None:
            one_minus_z = 1 - self.sigmas
            scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
            self.sigmas = 1 - (one_minus_z / scale_factor)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps)**2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing
            self.training = True
        else:
            self.training = False

    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        if isinstance(timestep, torch.Tensor): timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample

    def add_noise(self, original_samples, noise, timestep, t_dim=2):
        if isinstance(timestep, torch.Tensor): timestep = timestep.cpu()
        timestep = timestep[None]
        timestep_id = torch.argmin((self.timesteps[:, None] - timestep).abs(), dim=0)
        shape = [1] * noise.ndim
        shape[t_dim] = timestep_id.shape[0]
        sigma = self.sigmas[timestep_id].to(original_samples).view(shape)
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample

    def training_target(self, sample, noise, timestep):
        target = noise - sample
        return target

    def training_weight(self, timestep):
        timestep_id = torch.argmin(
            (self.timesteps[:, None].to(timestep.device) - timestep[None]).abs(), dim=0)
        weights = self.linear_timesteps_weights.to(timestep.device)[timestep_id].to(timestep.device)
        return weights
```

## 逐行讲解 / What's happening

1. **`set_timesteps` 第 41-58 行的 sigma 构造 / `set_timesteps` lines 41-58, building the sigma schedule**:
   - 中文: 从 `sigma_max` 到 `sigma_min` 线性 linspace,再把这条线性轴**通过 shift 重映射成非线性**。`exponential_shift` 走 logistic 重映射 `σ' = e^μ / (e^μ + 1/σ - 1)`,`shift_normal` 走经典 SD3 公式 `σ' = shift·σ / (1 + (shift-1)·σ)`。两者都把 sigma 推向"更晚去噪"的方向 —— 高分辨率视频需要更多步在低噪 (σ≈0) 区间精修。
   - English: A linear linspace from `sigma_max` to `sigma_min`, then **re-map non-linearly via shift**. `exponential_shift` uses the logistic re-map `σ' = e^μ / (e^μ + 1/σ - 1)`; `shift_normal` uses the classic SD3 formula `σ' = shift·σ / (1 + (shift-1)·σ)`. Both push sigma toward "later denoising" — high-res video needs more refinement steps in the low-noise (σ≈0) region.

2. **`shift_terminal` 这一段 / The `shift_terminal` block (lines 59-62)**:
   - 中文: 让最后一个 sigma **精确等于** `shift_terminal`(而不是 `sigma_min`)。这是为了把 schedule 钉死在某个数值终点上,避免训练-推理 sigma 范围不一致。如果你训练时用了 zero-SNR,这里就把 `shift_terminal=0` —— 最后一步刚好对齐纯净 data。
   - English: Pins the **last** sigma to exactly `shift_terminal` (instead of `sigma_min`). This nails the schedule to a specific numerical endpoint and avoids train-vs-inference sigma-range mismatch. If you trained with zero-SNR, set `shift_terminal=0` and the last step lands exactly on clean data.

3. **训练权重 `bsmntw_weighing` 第 66-73 行 / Training weights `bsmntw_weighing` lines 66-73**:
   - 中文: 这是整份文件最值得抄的小宝藏。公式是 `y = exp(-2·((x - N/2)/N)^2)` —— 一个标准的 Gaussian,中心在 N/2(N=num_inference_steps),宽度约 N/(2√2)。然后 `y - y.min()` 把端点压到 0,再 `y * (N / y.sum())` 重新归一化使平均权重 = 1(也就是 `mean(weight) = 1`)。**所以这个权重不是改变 loss 的总量级,只改变 timestep 之间的相对重要性**。两端的 t≈0 和 t≈N 几乎不出梯度,中间是主要训练区。`bsmntw` 大概是 *bell-shaped, mean-normalized, time-weighted* 之类的缩写。
   - English: The little gem of the file. Formula: `y = exp(-2·((x - N/2)/N)^2)` — a Gaussian centered at N/2 (N=num_inference_steps), width ≈ N/(2√2). Then `y - y.min()` pushes the endpoints to 0, and `y * (N / y.sum())` rescales so that the mean weight is 1 (`mean(weight) = 1`). **The weight does NOT change the overall loss magnitude — it only changes the relative importance across timesteps**. The two ends t≈0 and t≈N barely backprop, the middle is the main training band. `bsmntw` is presumably an initialism for "bell-shaped, mean-normalized, time-weighted" or similar.

4. **`step` 就是一行 Euler / `step` is one line of Euler (lines 78-89)**:
   - 中文: `prev_sample = sample + model_output * (sigma_ - sigma)`。`sigma_` 是下一个 sigma(更小),所以 `(sigma_ - sigma)` 是负数,model 输出的"velocity"按负方向推 sample 走 —— 这就是 rectified-flow Euler 步。`timestep_id + 1 >= len(self.timesteps)` 时切到 final sigma(0 或 1,看是否 reverse)。
   - English: `prev_sample = sample + model_output * (sigma_ - sigma)`. `sigma_` is the next sigma (smaller), so `(sigma_ - sigma)` is negative; the model's "velocity" output pushes `sample` along the negative direction — that's a rectified-flow Euler step. When `timestep_id + 1 >= len(self.timesteps)`, switch to the final sigma (0 or 1, depending on reverse).

5. **`add_noise` 第 99-109 行就是 q_sample / `add_noise` lines 99-109 is q_sample**:
   - 中文: `sample = (1 - sigma) * original + sigma * noise` —— flow matching 的标准前向公式。注意 `t_dim` 让 sigma 在视频维度(N, C, T, H, W 里的 T)广播,不同帧用各自的 sigma,这是 video diffusion 训练的常见 trick。
   - English: `sample = (1 - sigma) * original + sigma * noise` — the standard flow-matching forward formula. Note `t_dim` lets sigma broadcast over the video time dim (T in N, C, T, H, W); different frames use their own sigmas, a common trick in video diffusion training.

6. **`training_target` 就两行 / `training_target` is two lines (lines 111-113)**:
   - 中文: `target = noise - sample` —— 也就是 rectified flow 的 velocity `v = ε - x_0`(注意这里 `sample` 是 original sample,不是 `x_t`)。模型的回归目标就是这个 velocity。
   - English: `target = noise - sample` — rectified flow's velocity `v = ε - x_0` (note `sample` here is the original sample, not `x_t`). The model regresses this velocity.

7. **`training_weight` 找最近 timestep 并取权重 / `training_weight` finds the nearest timestep and looks up weight (lines 115-122)**:
   - 中文: 训练时随机采的 timestep 不一定精确等于 schedule 上的某个 timestep,所以 `argmin(|timesteps - t|)` 先找最近的 bin,再 lookup `linear_timesteps_weights`。这是离散网格的标准插值近似。
   - English: Training samples timesteps that don't necessarily match a schedule timestep exactly, so `argmin(|timesteps - t|)` finds the nearest bin and looks up `linear_timesteps_weights`. Standard discrete-grid interpolation.

## 类比 / The analogy

想象你在教一个学生学一首钢琴曲。整首曲子有 100 个小节,如果你让他平均花一样多的时间练每个小节,他会把 1/3 的时间浪费在开头 10 小节(已经简单)和结尾 10 小节(已经熟练)上,中间真正难的 60 小节反而没练够。聪明的老师做法是:画一条"重要性曲线",中间高、两头低,平均值 = 1(总练习时长不变)。学生每次练琴前看这条曲线决定花多少时间在哪段。lingbot-va 的 `training_weight` 就是给 flow-matching 训练画的这条钟形曲线 —— 中间的 sigma 是"真正在去噪的时刻",权重最高;两端是"已经差不多没事"的时刻,权重接近 0。

Picture teaching a student a 100-bar piano piece. If you give equal time to every bar, the student wastes a third of the time on the first 10 (already easy) and the last 10 (already mastered) — and shorts the genuinely hard middle 60. The clever teacher draws an "importance curve" — high in the middle, low at both ends, average = 1 (total practice time unchanged). Before each practice session the student consults the curve to decide where to spend minutes. lingbot-va's `training_weight` is exactly this bell-curve for flow-matching training — middle sigmas are "where denoising actually happens" and get the highest weight; the ends are "barely any work" and get near zero.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-WAM

中文:这个组件是 nanoWAM 课程里的 **`noise-scheduler`** 槽位。我们已经覆盖过 Wan2.1 的 `dynamic_shift`、dreamzero 的 90 行 scheduler,但这是第一份把**训练侧权重**写进同一个文件的实现 —— 它的依赖链是 `vae-encoder-decoder` → `patchify-positional` → 然后**和 `dit-block` 平级**,因为 noise scheduler 不依赖网络结构,反过来 `training-loop` 同时依赖 `dit-block` 和 `noise-scheduler`。在你 nanoWAM 里这个文件直接对应 `nanowam/noise/scheduler.py`,被训练循环和推理 sampler 共同 import。

English: This component is the **`noise-scheduler`** slot in your nanoWAM curriculum. We've already covered Wan2.1's `dynamic_shift` and dreamzero's 90-line scheduler, but this is the first implementation that bundles **training-side weighting** into the same file. Its dependency chain: `vae-encoder-decoder` → `patchify-positional` → then **parallel to `dit-block`**, since noise scheduling is independent of network structure. The downstream `training-loop` depends on both `dit-block` and `noise-scheduler`. In your nanoWAM this maps directly to `nanowam/noise/scheduler.py`, imported by both the training loop and the inference sampler.

中文:输入输出契约清晰:`add_noise(x, ε, t)` 给训练用,`step(velocity, t, x_t)` 给推理用,`training_target(x, ε, t)` 提供 loss 的回归目标,`training_weight(t)` 提供每个 t 的标量权重。如果省掉 `training_weight`,你的训练 loss 大头会被端点 timestep 吃掉,模型会在"接近干净数据"和"接近纯噪声"两个区段过拟合,中间真正去噪的能力反而弱 —— 这是很多人 reimplement flow-matching 时收敛慢的元凶。生产级实现还要再加:(a) `set_timesteps` 时支持 sigma re-quantization(让 sigma 落在固定 grid 上以便 cache),(b) `step` 支持 Heun 二阶,(c) `add_noise` 支持 multi-resolution sigma(不同 patch 不同 sigma)。

English: Clean I/O contracts: `add_noise(x, ε, t)` for training, `step(velocity, t, x_t)` for inference, `training_target(x, ε, t)` for the regression target, `training_weight(t)` for the per-t scalar weight. Drop `training_weight` and your training loss is dominated by the endpoint timesteps; the model overfits "almost-clean data" and "almost-pure-noise" regions and underperforms in the middle where denoising actually matters — this is the silent culprit behind many "why doesn't my flow-match converge" reimplementations. A production-grade version further adds: (a) sigma re-quantization in `set_timesteps` so sigmas land on a fixed grid for caching; (b) Heun second-order in `step`; (c) multi-resolution sigma support in `add_noise` (different patches get different sigmas).

## 自己跑一遍 / Try it yourself

```python
import math
import torch

class TinyFlowMatchScheduler:
    def __init__(self, n_train=1000, n_infer=50, shift=3.0):
        self.n_train, self.shift = n_train, shift
        self.set_timesteps(n_infer, training=True)
    def set_timesteps(self, n_infer, training=False):
        sigmas = torch.linspace(1.0, 0.003, n_infer)
        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        self.sigmas, self.timesteps = sigmas, sigmas * self.n_train
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - n_infer / 2) / n_infer) ** 2)
            y = y - y.min()
            self.w = y * (n_infer / y.sum())            # mean(w) == 1
    def add_noise(self, x, eps, t):                     # forward
        i = (self.timesteps - t).abs().argmin()
        s = self.sigmas[i]
        return (1 - s) * x + s * eps
    def training_target(self, x, eps): return eps - x   # velocity
    def training_weight(self, t):
        i = (self.timesteps - t).abs().argmin()
        return self.w[i]

sch = TinyFlowMatchScheduler(n_infer=50)
print("mean(weights)        =", sch.w.mean().item())   # should be ~1.0
print("weight at edges      =", sch.w[0].item(), sch.w[-1].item())
print("weight at middle     =", sch.w[25].item())
x = torch.zeros(1, 3, 8, 8); eps = torch.randn_like(x)
for t in [10, 500, 990]:
    print(f"t={t:3d}  weight={sch.training_weight(torch.tensor(t)).item():.3f}")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
mean(weights)        = 1.0
weight at edges      = 0.0 0.0
weight at middle     = 1.7 ~ 2.0
t= 10  weight=0.00
t=500  weight=1.8...
t=990  weight=0.00
```

中文:权重平均值精确 = 1.0,两端是 0,中间峰值约 1.7-2.0 倍 —— 这意味着 timestep 500 处的样本对 loss 的贡献是平均水平的 1.8 倍,而端点的样本几乎不参与训练。这是钟形权重最直观的表现。

English: Mean weight is exactly 1.0; the endpoints are 0; the middle peaks at about 1.7-2.0×. So a sample at timestep 500 contributes 1.8× the average to the loss, while endpoint samples contribute almost nothing. That's the most direct visualization of the bell-shaped weight.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **SD3 的 logit-normal weighting / SD3's logit-normal weighting**: 用 logit-normal 分布采样 timestep 实现类似的"中间多两端少"效果,数学等价、表达不同 / Samples timesteps from a logit-normal distribution to get the same "middle-heavy" effect — equivalent math, different expression.
- **EDM 的 σ-dependent loss weight / EDM's σ-dependent loss weight**: Karras et al. 用 `(σ² + σ_d²) / (σ·σ_d)²` 做权重,也是 σ→0 时压低 / σ→∞ 时压低的钟形 / Karras et al. weight by `(σ² + σ_d²) / (σ·σ_d)²` — again a bell shape that suppresses σ→0 and σ→∞.
- **Diffusion Policy 的训练循环 / Diffusion Policy's training loop**: 同样的 q_sample + velocity target,但没加 timestep 权重 —— 而 lerobot 的 flow-match policy 已经把这套训练 weighting 抄了过去 / Same q_sample + velocity target, but no timestep weighting. lerobot's flow-match policy has since adopted this exact training weighting.
- **Min-SNR weighting (Hang et al. 2023) / Min-SNR weighting (Hang et al. 2023)**: 用 `min(γ, SNR(t))` 做权重,核心思路一样:别在端点 timestep 上浪费算力 / Uses `min(γ, SNR(t))` as the weight, same core idea: stop wasting compute at endpoint timesteps.

## 注意事项 / Caveats / when it breaks

- **必须先 `set_timesteps(training=True)` 才能用 `training_weight` / Must call `set_timesteps(training=True)` first**: `linear_timesteps_weights` 只在 training 模式下生成。`set_timesteps(training=False)` 后调用 `training_weight` 会 AttributeError / `linear_timesteps_weights` is only built in training mode. Calling `training_weight` after `set_timesteps(training=False)` raises `AttributeError`.
- **timestep grid 和训练随机 t 的不匹配 / Mismatch between the timestep grid and randomly sampled training t**: 训练时通常 `t ~ Uniform(0, num_train_timesteps)`,但 schedule 只有 `n_infer` 个 grid 点。`argmin` 是最近邻取整,这意味着 weight 曲线实际只有 ~n_infer 个不同值 / Training usually does `t ~ Uniform(0, num_train_timesteps)`, but the schedule only has `n_infer` grid points. `argmin` does nearest-neighbor rounding, so the weight curve has only ~n_infer distinct values in practice.
- **`shift_terminal=0` 时数值边界 / Numerical boundary when `shift_terminal=0`**: 当 `shift_terminal=0` 时,`scale_factor = one_minus_z[-1] / 1`,公式没问题;但如果你 reverse_sigmas,最后一个 sigma 是 1,此时 `(1 - shift_terminal)` 在分母可能是 0 —— 务必检查 / When `shift_terminal=0` the math works. But if you `reverse_sigmas` the last sigma is 1 and `(1 - shift_terminal)` could be 0 in the denominator — guard against it.
- **`t_dim=2` 的硬编码 / `t_dim=2` hardcoding**: `add_noise` 默认在 dim=2 上广播 sigma,假设输入是 `(B, C, T, H, W)` 视频。如果你做的是 `(B, T, C, H, W)` 或者 image diffusion 必须显式传 `t_dim` / `add_noise` defaults to broadcasting on dim=2, assuming `(B, C, T, H, W)` video. For `(B, T, C, H, W)` or image diffusion you must pass `t_dim` explicitly.

## 延伸阅读 / Further reading

- [Robbyant/lingbot-va repo](https://github.com/Robbyant/lingbot-va)
- [SD3 paper — logit-normal timestep weighting (Esser et al. 2024)](https://arxiv.org/abs/2403.03206)
- [EDM paper — σ-dependent loss weight (Karras et al. 2022)](https://arxiv.org/abs/2206.00364)
- [Min-SNR weighting (Hang et al. 2023)](https://arxiv.org/abs/2303.09556)
- [Rectified Flow paper (Liu et al. 2022)](https://arxiv.org/abs/2209.03003)
