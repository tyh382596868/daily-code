---
date: 2026-06-09
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/utils/fm_solvers_unipc.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/utils/fm_solvers_unipc.py#L162-L279
difficulty: advanced
read_time: ~15 min
tags: [code-of-the-day, wam, flow-matching, noise-scheduler, resolution-shift]
build_role: noise-scheduler — advanced variant: production-grade flow-matching schedule with resolution-aware dynamic shifting
---

# Wan2.1 的 noise scheduler:同一根 flow-matching 时间轴,被 `time_shift` 重塑成"分辨率自适应" / Wan2.1's noise scheduler: one flow-matching time axis, reshaped by `time_shift` into a resolution-aware schedule

> **一句话 / In one line**: Flow matching 默认 sigma 从 1 线性走到 0,但 1080p 视频需要在"高噪声"段花更多步,480p 则不需要 —— Wan2.1 用一个 4 行的 `time_shift = exp(μ)/(exp(μ) + (1/t-1)^σ)` 公式把时间轴重新弯曲,让一个 checkpoint 跨分辨率都能跑得好。 / Flow matching defaults to a linear sigma schedule from 1 to 0, but 1080p video needs more steps in the high-noise regime while 480p doesn't — Wan2.1 reshapes the schedule with a 4-line `time_shift = exp(μ)/(exp(μ) + (1/t-1)^σ)` curve so a single checkpoint generates well across resolutions.

## 为什么重要 / Why this matters

之前 nanoWAM 课程已经学过 dreamzero 的 90 行 rectified-flow scheduler —— 那是教科书版:`sigma = linspace(1, 0, steps)`,推到 ODE 解就完事。但训练一个真正能用的视频 / world model,你会发现:**同一个模型在 256² 上效果好,推到 1024² 就糊掉了**。原因是高分辨率图像在像素空间里"看起来更接近高斯噪声",所以 sampler 在高噪声段(τ 接近 1)需要花更多功夫;低分辨率刚好相反。Stable Diffusion 3 和 Hunyuan-DiT 提出了 **flow-matching shift trick**:不改训练,只改 inference 时的时间轴弯曲程度,用一个标量 `shift`(或动态的 `μ`)就能搞定。Wan2.1 的 `FlowUniPCMultistepScheduler.set_timesteps` 是这套机制最干净的生产实现 —— 既支持静态 shift(训练时分辨率固定就够了),也支持动态 shift(同一 checkpoint 跑多分辨率)。

The previous nanoWAM curriculum entry covered dreamzero's 90-line rectified-flow scheduler — the textbook version: `sigma = linspace(1, 0, steps)`, plug into an ODE solver, done. But shipping a real video / world model exposes a brutal failure mode: **the same checkpoint that's crisp at 256² goes mushy at 1024²**. The reason: high-res frames look "closer to Gaussian noise" in pixel space, so the sampler needs more iterations in the high-noise regime (τ near 1) where a low-res model is fine with fewer. SD3 and Hunyuan-DiT introduced the **flow-matching shift trick**: don't retrain — just bend the inference-time axis with one scalar `shift` (or a dynamic `μ`) and the same checkpoint works across resolutions. Wan2.1's `FlowUniPCMultistepScheduler.set_timesteps` is the cleanest production implementation of this idea — supporting both static shift (fine when training resolution is fixed) and dynamic shift (one checkpoint, many resolutions).

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/utils/fm_solvers_unipc.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/utils/fm_solvers_unipc.py#L162-L279)

```python
def set_timesteps(self,
                  num_inference_steps=None,
                  device=None,
                  sigmas=None,
                  mu=None,
                  shift=None):
    if self.config.use_dynamic_shifting and mu is None:
        raise ValueError("you have to pass `mu` when `use_dynamic_shifting=True`")

    if sigmas is None:
        sigmas = np.linspace(self.sigma_max, self.sigma_min,
                             num_inference_steps + 1).copy()[:-1]

    if self.config.use_dynamic_shifting:
        sigmas = self.time_shift(mu, 1.0, sigmas)                     # resolution-aware reshape
    else:
        if shift is None:
            shift = self.config.shift
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)          # static reshape

    if self.config.final_sigmas_type == "sigma_min":
        sigma_last = ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]) ** 0.5
    elif self.config.final_sigmas_type == "zero":
        sigma_last = 0

    timesteps = sigmas * self.config.num_train_timesteps
    sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)

    self.sigmas = torch.from_numpy(sigmas)
    self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)
    self.num_inference_steps = len(timesteps)

    self.model_outputs = [None, ] * self.config.solver_order
    self.lower_order_nums = 0
    self.last_sample = None
    if self.solver_p:
        self.solver_p.set_timesteps(self.num_inference_steps, device=device)

    self._step_index = None
    self._begin_index = None
    self.sigmas = self.sigmas.to("cpu")    # avoid CPU<->GPU chatter


def time_shift(self, mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def _sigma_to_alpha_sigma_t(self, sigma):
    return 1 - sigma, sigma
```

## 逐行讲解 / What's happening

1. **`sigmas = np.linspace(sigma_max, sigma_min, steps + 1)[:-1]`**:
   - 中文: 教科书 flow-matching 时间轴:从 `sigma_max ≈ 1` 线性走到 `sigma_min ≈ 0`,N 步均匀切。`[:-1]` 因为我们要的是 N 个步开始点,最后一个 `sigma=0` 是终点不需要采样。
   - English: Textbook flow-matching axis: `sigma_max ≈ 1` linearly down to `sigma_min ≈ 0`, N equal steps. The `[:-1]` slice drops the final `sigma = 0` because we want N step *starts*, not N+1 anchors.

3. **静态 shift 分支:`sigmas = shift * t / (1 + (shift - 1) * t)`**:
   - 中文: 一个分式变换,当 `shift = 1` 时是恒等;当 `shift > 1` 时把曲线"压向高 sigma" —— 也就是让高分辨率训练时花更多步数在高噪声段。SD3 提出 `shift = 3.0` 对 1024² 比较合理。
   - English: A Möbius-style transform. `shift = 1` is the identity; `shift > 1` pushes the curve "up toward high sigma" — i.e. spends more inference steps in the noisy regime, which helps high resolutions. SD3 found `shift ≈ 3.0` reasonable for 1024².

4. **动态 shift 分支:`sigmas = time_shift(mu, 1.0, sigmas)`**:
   - 中文: 看下面 `time_shift` 函数:`exp(μ) / (exp(μ) + (1/t - 1)^σ)`。这是个广义的 logistic 曲线,μ 越大越往"高 sigma 区"偏。**关键是 μ 是从目标分辨率算出来的**(通常 `μ = α · log(W·H / W₀·H₀) + β`),这样不同分辨率的请求自动得到不同的时间轴 —— 一个 checkpoint 通杀多分辨率。
   - English: See `time_shift`: `exp(μ) / (exp(μ) + (1/t - 1)^σ)`. A generalised logistic curve where larger μ pushes more weight to the high-sigma region. **The trick is that μ is computed from the target resolution** (typically `μ = α · log(W·H / W₀·H₀) + β`), so different resolution requests automatically get different schedules and one checkpoint serves many resolutions.

5. **`time_shift(mu, sigma, t) = exp(mu) / (exp(mu) + (1/t - 1)^sigma)`**:
   - 中文: 这是 flow-matching 这套体系里"resolution-aware sigmoid"的标准公式。展开看:当 `t → 1` 时 `(1/t - 1)^σ → 0`,所以 result → 1;当 `t → 0` 时 `(1/t - 1)^σ → ∞`,result → 0。μ 控制曲线"什么时候开始下落":μ 越大,陡降点越靠近 `t = 0`。
   - English: The canonical "resolution-aware sigmoid" of the flow-matching family. Expanding the limits: at `t → 1` you have `(1/t − 1)^σ → 0` so result → 1; at `t → 0` you have `(1/t − 1)^σ → ∞` so result → 0. μ controls *when the curve falls*: larger μ pushes the steep drop toward `t = 0`.

6. **`final_sigmas_type` 分支**:
   - 中文: `"zero"` 意味着 ODE 终点 σ=0 是真零,数学最干净;`"sigma_min"` 意味着用训练时实际的最低 σ(从 `alphas_cumprod` 反推),工程上更稳。Wan2.1 通常用 `"zero"`,因为 flow matching 本身就是用 `x_1 = clean` 训练的。
   - English: `"zero"` lands the ODE end at exact `σ = 0` (mathematically cleanest). `"sigma_min"` uses the lowest σ actually seen at training (recovered from `alphas_cumprod`), which can be more numerically stable. Wan2.1 typically picks `"zero"` because flow matching trains with `x_1` being clean.

7. **`_sigma_to_alpha_sigma_t(sigma) = (1 - sigma, sigma)`**:
   - 中文: 一行就把 flow matching 的核心约定写死:在时间 σ 上,`x_σ = (1-σ)·x_clean + σ·noise`。所以 `alpha_t = 1 - sigma`,纯线性,和 EDM / DDPM 的"复杂 α" 完全不一样 —— flow matching 的代数简单到令人发指。
   - English: One line that fixes the flow-matching convention: at noise time σ, `x_σ = (1 − σ) · x_clean + σ · noise`. So `alpha_t = 1 − sigma`, pure linear — completely unlike EDM / DDPM's elaborate α tables. Flow matching's algebra is unreasonably simple.

## 类比 / The analogy

想象一个老厨师在 10 分钟内做一道菜。如果是简单的炒蛋(低分辨率),前 2 分钟切菜、后 8 分钟翻炒就行;但要是做一道四川水煮鱼(高分辨率),你必须前 6 分钟备料调味(对应高噪声段的精细工作),后 4 分钟才下锅。"做菜总共 10 分钟"是 flow matching 的时间轴 `t ∈ [0, 1]`;但"在哪一段花多少时间"是 `time_shift` 决定的。原始 `linspace` 是"机械地把 10 分钟均分十段";`shift` 是"老厨师默认的菜系节奏";`mu` 则是"根据今天要做的菜临时调整节奏"。同一台 ODE 求解器,搭配不同的 `mu`,就能从早餐做到满汉全席。

Picture a chef with 10 minutes to make a dish. For scrambled eggs (low resolution), 2 minutes of chopping and 8 minutes of stirring is fine; for Sichuan boiled fish (high resolution), you must spend 6 minutes on careful prep and seasoning (the high-noise regime) and only 4 actually cooking. "10 minutes total" is flow matching's `t ∈ [0, 1]` axis. *How* to spend those 10 minutes is what `time_shift` decides. Raw `linspace` is "mechanically split 10 minutes evenly"; static `shift` is "the chef's default rhythm for this cuisine"; `mu` is "today's adjustment based on which dish was ordered." Same ODE solver, different `mu`, anything from breakfast to a banquet.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-WAM

中文:在 nanoWAM 的构件图里,scheduler 是**采样回路的脊椎**:它决定了"从 noise 到 video 一共要 denoise 多少步、每步的 sigma 是多少、每步对 `v_t` 做什么数学变换"。在课程里这是 `noise-scheduler` 组件,**没有任何依赖** —— 你可以在还没写 DiT、VAE 的时候就把它跑通。上游(无):scheduler 不需要任何 model 输出,它只是"时间轴生成器"。下游:`sampler-inference`(每一步具体调 model)、`training-loop`(训练时也要 sample sigma)。**之前已覆盖的简单 rectified-flow scheduler 是教学版**;本节是**生产版**,主要补两块:(1) 静态 `shift` 让你在固定分辨率训练时也能调采样质量;(2) 动态 `mu` 让你**一个 checkpoint 通吃 480p/720p/1080p**。要在 nanoWAM 里集成,先在 `set_timesteps` 里把这套放进去,然后在 inference 入口根据请求的分辨率算 `mu = α · log(W·H / W_train·H_train) + β`(α、β 是两个标量超参,从 SD3 论文借就行)。生产实现还需要补:(a) UniPC 多步预测-校正(看同文件下方 `multistep_uni_p/c_bh_update`);(b) `solver_p`(用低阶 solver 给 UniPC 提供初始猜测);(c) 时间步反向迭代时的 `step_index` 计数器。

English: in nanoWAM's component graph, the scheduler is the **spine of the sampling loop**: it decides how many denoising steps separate noise from video, what sigma each step uses, and how each step transforms `v_t`. In the curriculum this is the `noise-scheduler` component with **no upstream dependencies** — you can implement and unit-test it before writing the DiT or VAE. Upstream: nothing (it's a pure time-axis generator). Downstream: `sampler-inference` (per-step model calls) and `training-loop` (sigma sampling at training time). The **previously covered simple rectified-flow scheduler was the educational version**; this is the **production version**, adding two things: (1) static `shift` to tune sampling quality at a fixed training resolution; (2) dynamic `μ` to let **one checkpoint serve 480p / 720p / 1080p**. To integrate into nanoWAM, drop this `set_timesteps` in, then in your inference entry-point compute `μ = α · log(W·H / W_train·H_train) + β` (α, β are scalar hyperparams; SD3's defaults work). A production implementation additionally needs: (a) UniPC's multi-step predictor-corrector (`multistep_uni_p/c_bh_update` further down the file); (b) a `solver_p` low-order solver to seed UniPC's initial guess; (c) a `step_index` counter for the reversed iteration order.

## 自己跑一遍 / Try it yourself

```python
import math
import numpy as np
import matplotlib.pyplot as plt

def time_shift(mu, sigma, t):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

def static_shift(shift, t):
    return shift * t / (1 + (shift - 1) * t)

steps = 25
linear = np.linspace(1.0, 1e-3, steps + 1)[:-1]
static_3x = static_shift(3.0, linear)
dyn_lowres = time_shift(0.5, 1.0, linear)   # μ for ~256²
dyn_highres = time_shift(1.5, 1.0, linear)  # μ for ~1024²

plt.plot(range(steps), linear, label="linear (sigma_max → sigma_min)")
plt.plot(range(steps), static_3x, label="static shift=3.0")
plt.plot(range(steps), dyn_lowres, label="dynamic μ=0.5 (low res)")
plt.plot(range(steps), dyn_highres, label="dynamic μ=1.5 (high res)")
plt.xlabel("step"); plt.ylabel("sigma"); plt.legend(); plt.grid()
plt.title("Flow-matching sigma schedules under shift")
plt.savefig("/tmp/wam_schedule.png", dpi=120, bbox_inches="tight")
print("saved /tmp/wam_schedule.png")
```

运行 / Run with:
```bash
pip install numpy matplotlib
python try.py
```

预期输出 / Expected output:
```
saved /tmp/wam_schedule.png
```

中文一句:打开图能直观看到 —— μ=1.5 的曲线在前几步几乎水平(在高 sigma 区慢慢走),而 μ=0.5 的曲线更接近线性;这就是为什么同一个模型在 1024² 用 μ=1.5 时不会糊。

English: open the PNG and you can see at a glance — the μ=1.5 curve is nearly flat in the first few steps (slowly traversing the high-sigma region), while the μ=0.5 curve hugs the linear baseline. That visualisation is precisely *why* the same checkpoint stays sharp at 1024² when you bump μ to 1.5.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **同组件其他实现 / Other implementations of the same `noise-scheduler` slot**:
  - **`dreamzero0/dreamzero` rectified-flow scheduler** (2026-05-28 课程): 中文:教科书版 —— `linspace` + Euler step,没有 shift。在固定低分辨率的玩具上够用,扩展性差。 / English: The textbook reference — `linspace` + Euler step, no shift. Fine for fixed-low-resolution toys; scales poorly.
  - **`huggingface/diffusers` `FlowMatchEulerDiscreteScheduler`** (2026-05-31 课程): 中文:diffusers 主仓版本,代码结构和 Wan2.1 这版几乎一样 —— 这版就是从 diffusers 改的(文件注释里 "Modified from")。 / English: The diffusers reference; structurally near-identical to Wan2.1's (the file header says "Modified from"). Wan2.1 added the UniPC integration around it.
  - **`stabilityai/sd3-research` (SD3 paper code)**: 中文:静态 `shift` 公式的源头,但 SD3 用的是 Euler。Wan2.1 把同样的 `shift` 挪到了 UniPC 上。 / English: Where the static `shift` formula originated, but SD3 ships it on top of Euler. Wan2.1 lifts the same `shift` onto a UniPC solver.
  - **`hpcaitech/Open-Sora` `RFScheduler`**: 中文:Open-Sora 自己实现的 rectified-flow,有 shift 但用了不同的公式形式 —— 同一个思想的另一种代数表达。 / English: Open-Sora's own rectified-flow scheduler — same idea, slightly different algebraic form. Useful contrast for reading.
- **Karras 的 EDM noise schedule**: 中文:更早的"自适应 sigma"思想:Karras 2022 用一个 power-law `(sigma_min^{1/ρ} + (sigma_max^{1/ρ} - sigma_min^{1/ρ}) · t)^ρ` 给 sigma 加权重。`time_shift` 是它在 flow matching 范式下的等价物。 / English: An older "adaptive sigma" lineage — Karras 2022 used a power-law `(σ_min^{1/ρ} + (σ_max^{1/ρ} − σ_min^{1/ρ}) · t)^ρ`. `time_shift` is the flow-matching-era equivalent.

## 注意事项 / Caveats / when it breaks

- **μ 计算公式不对就废 / Wrong μ formula nullifies the trick**: 中文:很多人复制了 `time_shift` 函数但 μ 直接硬编码 1.0,等于白蹭了一个常数 —— 必须根据请求分辨率动态算 μ。SD3 给的参考公式是 `μ = log((H_target · W_target) / (H_base · W_base)) / 2`。 / English: A surprising number of forks copy `time_shift` but hard-code `μ = 1.0`, which defeats the purpose. μ must be computed dynamically from the request's resolution; SD3's recipe is `μ = log((H · W) / (H_base · W_base)) / 2`.
- **静态 shift 和动态 mu 不要同时开 / Don't enable static `shift` and dynamic `μ` together**: 中文:代码用 `if/else` 隔开了两条路径,但有些 user-config 同时设置了 `shift` 和 `use_dynamic_shifting=True` —— 这种情况下 shift 会被静默忽略。务必只开一个。 / English: The code's `if/else` only takes one path, but a misconfigured pipeline that sets both `shift` and `use_dynamic_shifting=True` will silently ignore `shift`. Always enable exactly one.
- **`sigmas = sigmas.to("cpu")` 会卡 PyTorch 0.x 流水线 / The trailing `to("cpu")` can deadlock async pipelines**: 中文:文件最后一句把 sigmas 移到 CPU 上是"避免每次 step 都 CPU↔GPU 来回",但如果你跑的是异步 ODE solver,要确保 sigma 读取在主线程。 / English: The trailing `sigmas.to("cpu")` aims to avoid per-step CPU↔GPU chatter, but in async ODE solvers you must ensure the sigma reads happen on the main thread.

## 延伸阅读 / Further reading

- [Stable Diffusion 3 paper — Section 3.3 "Resolution-Dependent Shifting"](https://arxiv.org/abs/2403.03206)
- [Hunyuan-DiT paper — dynamic `mu` formula](https://arxiv.org/abs/2405.08748)
- [diffusers `FlowMatchEulerDiscreteScheduler` source](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py)
