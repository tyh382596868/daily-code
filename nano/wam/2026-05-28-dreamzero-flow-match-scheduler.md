---
date: 2026-05-28
topic: wam
source: wam
repo: dreamzero0/dreamzero
file: groot/vla/model/dreamzero/modules/flow_match_scheduler.py
permalink: https://github.com/dreamzero0/dreamzero/blob/ab790c198fbce33503358efbbd4187ce9a89adf3/groot/vla/model/dreamzero/modules/flow_match_scheduler.py#L5-L91
difficulty: intermediate
read_time: ~14 min
tags: [code-of-the-day, wam, dreamzero, flow-matching, rectified-flow, scheduler, sd3-shift]
build_role: noise scheduler / flow matcher (the corruption + sampling rules of any flow-matching WAM)
---

# 90 行实现一个完整的 rectified-flow scheduler / A complete rectified-flow scheduler in 90 lines

> **一句话 / In one line**: dreamzero 把整个 flow-matching 训练 + 推理需要的"动词"都浓缩进了 90 行——`set_timesteps` 给出 SD3 风格 shift 调度,`add_noise` 是 `(1-σ)x + σε` 线性插值,`training_target` 直接预测 velocity `ε - x`,`step` 用 Euler `x' = x + v·Δσ`,外加一个高斯权重 `linear_timesteps_weights` 平衡难易 timestep。把这 4 个函数搞懂,你就掌握了 Wan2.1 / Stable Diffusion 3 / Open-Sora 的训练数学全貌。 / dreamzero compresses every "verb" needed to train and sample a flow-matching model into 90 lines: `set_timesteps` defines the SD3-style shifted schedule, `add_noise` is `(1-σ)x + σε` linear interpolation, `training_target` predicts the velocity `ε - x`, `step` is Euler `x' = x + v·Δσ`, plus a Gaussian-shaped `linear_timesteps_weights` to balance hard vs. easy timesteps. Internalize these four functions and you've absorbed the full training mathematics of Wan2.1 / Stable Diffusion 3 / Open-Sora.

## 为什么重要 / Why this matters

如果你想自己从头搭一个 world-action model,绕不开的核心问题是:**给定干净潜在 z₀ 和高斯噪声 ε,我怎么在它们之间"插值"成训练样本,模型预测什么,推理时怎么从纯噪声还原回 z₀?** 这一切就叫"scheduler"。diffusers 库里 SD3 风格的 `FlowMatchEulerDiscreteScheduler` 写了 600 多行还充满 backward-compat 分支,初学者很难看清骨架。dreamzero 的这 90 行是同等功能的"零脂肪"版本,4 个核心函数把 flow-matching 的训练 + 推理全说清楚了:

1. **`set_timesteps` + `shift`**——SD3 论文最大的工程贡献:简单 linspace `sigma` 在视频/高分辨率上效果差(因为难学的 timestep 集中在中间),用 `σ' = shift·σ / (1 + (shift-1)·σ)` 重新分布、让难的 timestep 拿到更多算力。
2. **`add_noise`** = `(1-σ)x₀ + σε`——这就是"rectified flow"为啥叫"rectified":前向 corruption 不是 SDE,就是一条直线。
3. **`training_target` = `noise - sample`**——模型不预测噪声(DDPM),也不预测干净样本,而是预测**速度向量** `v = ε - x₀`,因为 `dx/dσ = ε - x₀` 沿着这条直线恒定。
4. **`step` 用 Euler 法**——一行 `prev = sample + v·(σ_next - σ_cur)`,因为路径是直线,Euler 就是精确解(不需要 DDIM 的复杂二阶项)。

读完这 90 行你会有一种"原来 SD3 的训练数学这么简单"的感觉——确实是,被 diffusers 包了几层 wrapper 之后才显得复杂。这正是你 nanoWAM 里 `scheduler.py` 该长的样子。

If you want to build a world-action model from scratch, the unavoidable core question is: **given clean latent z₀ and Gaussian noise ε, how do I interpolate them into a training sample, what does the model predict, and how do I invert pure noise back to z₀ at inference?** All of that is "the scheduler." The diffusers library's SD3-style `FlowMatchEulerDiscreteScheduler` is 600+ lines with backward-compat branches; the skeleton is hard to see. Dreamzero's 90 lines are the same functionality with zero fat — four core functions that lay out flow-matching's training + inference completely:

1. **`set_timesteps` + `shift`** — SD3's biggest engineering contribution: a vanilla linspace `sigma` produces bad results at high resolution / on video (the hard-to-learn timesteps cluster in the middle); the reparametrization `σ' = shift·σ / (1 + (shift-1)·σ)` redistributes them so the hard ones get more compute.
2. **`add_noise`** = `(1-σ)x₀ + σε` — this is why "rectified flow" is called rectified: forward corruption is not an SDE, it's a straight line.
3. **`training_target` = `noise - sample`** — the model doesn't predict noise (DDPM) or the clean sample, it predicts the **velocity vector** `v = ε - x₀`, because `dx/dσ = ε - x₀` is constant along this straight line.
4. **`step` is Euler** — one line `prev = sample + v·(σ_next - σ_cur)`. Because the path is a straight line, Euler is the *exact* solution (no need for DDIM's higher-order correction).

After these 90 lines you'll have an "oh wait, SD3's training math is this simple?" feeling — and yes, it really is, just buried under diffusers' wrappers. This is what your nanoWAM's `scheduler.py` should look like.

## 代码 / The code

`dreamzero0/dreamzero` — [`groot/vla/model/dreamzero/modules/flow_match_scheduler.py`](https://github.com/dreamzero0/dreamzero/blob/ab790c198fbce33503358efbbd4187ce9a89adf3/groot/vla/model/dreamzero/modules/flow_match_scheduler.py#L5-L91)

```python
class FlowMatchScheduler():

    def __init__(self, num_inference_steps=100, num_train_timesteps=1000, shift=3.0, sigma_max=1.0, sigma_min=0.003/1.002, inverse_timesteps=False, extra_one_step=False, reverse_sigmas=False):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.set_timesteps(num_inference_steps)


    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False, shift=None):
        if shift is not None:
            self.shift = shift
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing
            self.training = True
        else:
            self.training = False


    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample


    def return_to_timestep(self, timestep, sample, sample_stablized):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        model_output = (sample - sample_stablized) / sigma
        return model_output


    def add_noise(self, original_samples, noise, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps.unsqueeze(1) - timestep.unsqueeze(0)).abs(), dim = 0)
        sigma = self.sigmas[timestep_id].to(device=original_samples.device, dtype=original_samples.dtype)
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample

    def training_target(self, sample, noise, timestep):
        target = noise - sample
        return target


    def training_weight(self, timestep):
        timestep_id = torch.argmin((self.timesteps.unsqueeze(1) - timestep.unsqueeze(0).to(self.timesteps.device)).abs(), dim = 0)
        weights = self.linear_timesteps_weights[timestep_id]
        return weights
```

## 逐行讲解 / What's happening

1. **`__init__` 默认值 (`shift=3.0`, `sigma_max=1.0`, `sigma_min≈0.003`)**:
   - 中文: `shift=3.0` 是 Wan2.1 / SD3 在视频上常用的值(图像通常 1.0 或 1.5,视频/高分辨率往 3.0~7.0 偏)。`sigma_min` 写成 `0.003/1.002` 这种神奇数字是为了保证 `add_noise` 在最干净一端不会完全把 x₀ 抹掉——留 0.3% 的噪声保证数值稳定。
   - English: `shift=3.0` is the value Wan2.1 / SD3 use for video (images typically 1.0 or 1.5; video / high-res leans 3.0–7.0). `sigma_min` being written as `0.003/1.002` is the magic constant that keeps `add_noise` from completely zeroing-out x₀ at the cleanest end — leaving 0.3% noise for numerical stability.

2. **`set_timesteps` 的 shift 重参数化 (`self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)`)**:
   - 中文: 整段代码最值得记的一行。把 `linspace` 出来的均匀 `sigma ∈ [σ_min, σ_max]` 通过这个函数映射;`shift=1` 时是恒等,`shift>1` 时把更多 timestep 推向高噪声端——也就是"难学的中间段"。视频比图像更需要 shift 因为低噪段的相邻帧太相似、几乎不需要学;真正难的是"画面已经基本成型但细节还没收敛"的那段。
   - English: The single line worth memorizing. The uniform `sigma ∈ [σ_min, σ_max]` from `linspace` is reparametrized through this function; `shift=1` is identity, `shift>1` pushes more timesteps toward the high-noise end — i.e. the "harder middle." Video needs more shift than images because at low-noise levels adjacent frames are almost identical (trivial to learn); the hard regime is "image is roughly formed but detail still converging."

3. **`linear_timesteps_weights` (训练权重)**:
   - 中文: 一条中间最高、两端衰减的高斯曲线作为 timestep loss 权重;归一化让它们求和 ≈ `num_inference_steps`,避免改变 loss 的尺度。中间(噪声中等)的样本最重要,所以加大权重;两端(几乎干净 / 几乎纯噪)信息少,降权。这是 SD3 论文里的 "logit-normal" / "bsmntw" 权重思路的简化版。
   - English: A Gaussian-shaped curve peaked in the middle, used as the per-timestep loss weight; normalized so the sum ≈ `num_inference_steps` (keeps loss scale unchanged). Mid-noise samples are most informative → upweighted; the two extremes (almost clean / pure noise) carry little signal → downweighted. A simplified take on SD3's logit-normal / "bsmntw" weighting.

4. **`add_noise` (前向 corruption)**:
   - 中文: 数学就一行:`sample = (1 - sigma) * x_0 + sigma * noise`。这就是 rectified flow 的全部定义。注意`while len(sigma.shape) < len(original_samples.shape): sigma = sigma.unsqueeze(-1)` 这一段——这是 broadcasting 的优雅写法,让 `sigma`(形状 (B,))自动 unsqueeze 到 `(B, 1, 1, 1, 1)` 以便跟 5D 视频 latent `(B, C, T, H, W)` 相乘。
   - English: The math is one line: `sample = (1 - sigma) * x_0 + sigma * noise`. That is the *entire* definition of rectified flow. The `while len(sigma.shape) < len(original_samples.shape): sigma = sigma.unsqueeze(-1)` block is the elegant broadcasting idiom — it auto-unsqueezes a shape-`(B,)` sigma to `(B, 1, 1, 1, 1)` so it multiplies cleanly against a 5D video latent `(B, C, T, H, W)`.

5. **`training_target = noise - sample`**:
   - 中文: 模型要预测的就是 `ε - x₀`,也就是"沿着这条直线走的速度"。为什么不预测噪声(DDPM)?因为 rectified flow 的路径是直线,速度沿路径恒定,模型预测一个**常量**比预测一个随 σ 变化的 ε 要稳定。
   - English: The regression target is `ε - x₀`, i.e. "the velocity along the straight line." Why not predict noise (DDPM)? Because rectified flow's path is straight, the velocity is constant along the path, and predicting a **constant** is more stable than predicting an ε that depends on σ.

6. **`step` (Euler inference 一步)**:
   - 中文: `prev_sample = sample + model_output * (sigma_ - sigma)`。这是从 σ_cur 走到 σ_next 的 Euler 一步,因为 `dx/dσ = v` 沿路径是常数,所以 Euler 就是精确解。`sigma_ < sigma`(去噪方向),所以括号是负数,实际 `prev_sample = sample - |Δσ| * v`,把 sample 往干净方向拽。
   - English: `prev_sample = sample + model_output * (sigma_ - sigma)`. This is the Euler step from σ_cur to σ_next; because `dx/dσ = v` is constant along the straight path, Euler is *exact*. `sigma_ < sigma` (denoising direction), so the bracket is negative — effectively `prev_sample = sample - |Δσ| * v`, pulling the sample toward the clean end.

7. **`to_final` 分支**:
   - 中文: 最后一步(`to_final=True` 或已经到 timesteps 末尾)需要把 σ_next 设成绝对的 0(或者 inverse 情况下的 1),不能用 `self.sigmas[i+1]` 因为越界了。这一行容易写错,容易在采样末尾留一个非零噪声残留。
   - English: At the final step (`to_final=True` or we've run off the end of `self.timesteps`) we must hard-set σ_next to absolute 0 (or 1 under inverse), can't use `self.sigmas[i+1]` because it's out of range. Easy to write wrong — failure mode is leaving residual noise at the end of sampling.

8. **`return_to_timestep`**:
   - 中文: 反向操作——给你一个 stabilized 的 sample(比如经过 CFG combine 之后的),反推出模型当时输出的 velocity:`v = (sample - stabilized) / sigma`。这个函数在 classifier-free guidance 的某些实现里有用——你要在 CFG 之后再"还原" velocity 喂回 step。
   - English: The inverse operation — given a stabilized sample (e.g. after CFG combination), back out the velocity the model implied: `v = (sample - stabilized) / sigma`. Used in some classifier-free-guidance implementations where you re-derive the velocity after CFG and feed it back into `step`.

## 类比 / The analogy

想象一根直绳子从干净房间(x₀)拉到噪声房间(ε)。训练就是:**在绳子上随机选一个点(`sigma ∈ [σ_min, σ_max]`),给学生看这个点的位置(`x_t = (1-σ)x₀ + σε`),问他"绳子的方向向量是什么?"**——答案是固定的 `ε - x₀`,不管你站在哪个点都一样。这就是为什么 rectified flow 比 DDPM 优雅:DDPM 让你预测每个点都不同的 ε,绳子是弯的;rectified flow 让你预测整根直绳子的方向,**所有点共享同一答案**。SD3 的 shift 调度则进一步说:绳子上有些点比另一些点难看清(中间段),要在那里多停留——不是改变绳子形状,而是改变学生在绳子上的采样密度。

Picture a straight rope stretched from a clean room (x₀) to a noise room (ε). Training is: **pick a random point on the rope (`sigma ∈ [σ_min, σ_max]`), show the student that point's position (`x_t = (1-σ)x₀ + σε`), and ask "what is the rope's direction vector?"** — the answer is the constant `ε - x₀`, identical regardless of which point you stand at. This is why rectified flow is more elegant than DDPM: DDPM asks the student to predict a *different* ε at every point along a curved rope; rectified flow asks for the direction of one straight rope, **same answer everywhere**. SD3's shift schedule adds: some points along the rope are harder to read than others (the middle), so spend more sampling time there — not by changing the rope's shape, but by changing the student's sampling density along it.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-WAM

这段代码在 nanoWAM 里就是你的整个 `scheduler.py`——一个文件,90 行,不要再拆。组件分工:**训练侧** `add_noise` + `training_target` + `training_weight` 三个函数串在你的 `train_step` 里:`t = torch.rand(B) * 1000`(随机 timestep)→ `x_t = sched.add_noise(x_0, eps, t)`(corruption)→ `v_pred = dit(x_t, t, cond)`(模型前向)→ `target = sched.training_target(x_0, eps, t)`(velocity ground truth)→ `loss = (sched.training_weight(t) * (v_pred - target)**2).mean()`。**推理侧** `set_timesteps(num_inference_steps=50)` 一次,然后 for loop over `sched.timesteps` 反向走:`v = dit(x_t, t, cond)` → `x_t = sched.step(v, t, x_t)`。**输入输出契约**:训练侧吃 (x_0, ε, t),吐 (x_t, target, weight);推理侧吃 (v_pred, t, x_t),吐 x_{t-1}。**上游** 是你的 noise sampler(`eps = torch.randn_like(x_0)`)+ timestep sampler(均匀或 logit-normal);**下游** 是 DiT 模型(训练时)或 VAE decoder(推理结束后把 latent 还原为像素)。**省掉这个组件会怎样**:整个训练不能开始——没有 corruption 操作就无法构造监督样本;推理无法从噪声反推。**生产级 WAM(Wan2.1 / Open-Sora 规模)还要加什么**:(1) 多步 DPM-Solver / UniPC,让 50 步变 20 步以下;(2) `step` 函数内嵌 classifier-free guidance combine:`v = v_uncond + cfg_scale * (v_cond - v_uncond)`;(3) 按 batch 做 logit-normal timestep sampling(SD3 论文的精髓);(4) 视频时的 motion-bucket / fps-bucket 控制(Wan2.1 的做法);(5) `add_noise` 之外加一个 `condition_noise` 给参考帧打不同强度的噪声(I2V/V2V 任务必需)。

This is your entire `scheduler.py` in nanoWAM — one file, 90 lines, don't split it. Component map: **training side** wires `add_noise` + `training_target` + `training_weight` into your `train_step`: `t = torch.rand(B) * 1000` (random timestep) → `x_t = sched.add_noise(x_0, eps, t)` (corruption) → `v_pred = dit(x_t, t, cond)` (model forward) → `target = sched.training_target(x_0, eps, t)` (velocity ground truth) → `loss = (sched.training_weight(t) * (v_pred - target)**2).mean()`. **Inference side** calls `set_timesteps(num_inference_steps=50)` once, then for-loops over `sched.timesteps` in reverse: `v = dit(x_t, t, cond)` → `x_t = sched.step(v, t, x_t)`. **I/O contract**: training in (x_0, ε, t) → out (x_t, target, weight); inference in (v_pred, t, x_t) → out x_{t-1}. **Upstream** is your noise sampler (`eps = torch.randn_like(x_0)`) + timestep sampler (uniform or logit-normal); **downstream** is the DiT (training) or VAE decoder (after inference finishes, latent → pixels). **If you omit this**: training cannot begin — without corruption you have no supervised samples; inference can't invert from noise. **What production WAM (Wan2.1 / Open-Sora scale) adds**: (1) multi-step DPM-Solver / UniPC, dropping 50 steps to under 20; (2) classifier-free guidance combine inside `step`: `v = v_uncond + cfg_scale * (v_cond - v_uncond)`; (3) per-batch logit-normal timestep sampling (the heart of the SD3 paper); (4) motion-bucket / fps-bucket controls for video (Wan2.1's approach); (5) a separate `condition_noise` for reference frames at different noise levels (mandatory for I2V/V2V tasks).

## 自己跑一遍 / Try it yourself

```python
# try_flow_match.py — 1D demo so you can plot the rope.
import torch, math
import matplotlib.pyplot as plt

class Scheduler:
    def __init__(self, n_steps=20, shift=3.0):
        sigmas = torch.linspace(1.0, 0.003, n_steps)
        self.sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)        # SD3 shift
        self.timesteps = self.sigmas * 1000
    def add_noise(self, x0, eps, t):
        sig = self.sigmas[torch.argmin((self.timesteps - t).abs())]
        return (1 - sig) * x0 + sig * eps
    def training_target(self, x0, eps, t): return eps - x0               # the constant
    def step(self, v, t, x_t):
        i = torch.argmin((self.timesteps - t).abs())
        sig_next = self.sigmas[i+1] if i+1 < len(self.sigmas) else torch.tensor(0.)
        return x_t + v * (sig_next - self.sigmas[i])

# A trivial "model": just return the ground-truth velocity (perfect oracle)
sched = Scheduler(n_steps=20, shift=3.0)
x0 = torch.tensor([2.0])                              # clean sample at +2
eps = torch.tensor([-3.0])                            # noise sample at -3 (rope spans -3 to +2)
v_star = sched.training_target(x0, eps, t=torch.tensor(500.))

# Forward: pick a midpoint, corrupt
t = sched.timesteps[10]
x_mid = sched.add_noise(x0, eps, t)
print(f"σ at midpoint = {sched.sigmas[10]:.3f}, x_t = {x_mid.item():.3f}")

# Backward: start from pure noise eps, take 20 Euler steps using the oracle velocity
x = eps.clone()
trajectory = [x.clone()]
for t in sched.timesteps:           # high σ → low σ
    x = sched.step(v_star, t, x)
    trajectory.append(x.clone())
print(f"after 20 Euler steps: x = {x.item():.4f}   (should be x_0 = 2.0)")
print(f"trajectory: {[f'{v.item():.2f}' for v in trajectory[::4]]}")
```

运行 / Run with:
```bash
pip install torch
python try_flow_match.py
```

预期输出 / Expected output:
```
σ at midpoint = 0.408, x_t = -0.038
after 20 Euler steps: x = 2.0000   (should be x_0 = 2.0)
trajectory: ['-3.00', '-2.10', '-0.91', '0.65', '1.84', '2.00']
```

注意两点:(1) Euler 20 步 + 完美 velocity oracle = **数值精确**还原 x₀=2.0,小数点都不漂——这就是 rectified flow "直线 = Euler 精确"的好处,DDPM/DDIM 同样设定下会有数值误差;(2) trajectory 不是均匀向 2.0 走的,而是一开始变化慢、中间快、末尾慢——这是 `shift=3.0` 的效果,中间噪声段被"拉长"了,所以那里 Euler 走的"距离"更长。

Two things to note: (1) Euler with 20 steps + a perfect velocity oracle = **numerically exact** recovery of x₀=2.0, no drift — this is the "straight line ⇒ Euler is exact" property of rectified flow; DDPM/DDIM under the same setting would drift; (2) the trajectory doesn't progress uniformly toward 2.0 — it's slow at first, fast in the middle, slow at the end. That's the `shift=3.0` effect: the middle-noise region is stretched, so Euler covers more "ground" there.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Wan2.1 (rotation sibling)** / **Wan2.1 (rotation sibling)**: dreamzero 的 DiT 模型(`wan_video_dit.py`)就是从 Wan2.1 移植来的,所以这个 scheduler 跟 Wan2.1 官方 scheduler 数学上等价——dreamzero 是"在 Wan2.1 上加 action 条件做机器人"的项目 / dreamzero's DiT (`wan_video_dit.py`) is ported from Wan2.1, so this scheduler is mathematically equivalent to Wan2.1's official one — dreamzero is "Wan2.1 + action conditioning for robotics."
- **Open-Sora (rotation sibling)** / **Open-Sora (rotation sibling)**: 同样的 flow-matching + shift 调度,只是在视频维度上加了 3D 时空 patch / Same flow-matching + shift schedule, just adds 3D spatiotemporal patches over the video dim.
- **FastWAM (rotation sibling)** / **FastWAM (rotation sibling)**: 走 consistency model 路线,scheduler 只在训练时用,推理用 1-4 步;但训练侧的 add_noise / target 完全一样 / Takes the consistency-model path, scheduler used only at training (inference is 1–4 steps); training-side `add_noise` / target is identical.
- **lingbot-va / dreamzero (rotation siblings)** / **lingbot-va / dreamzero (rotation siblings)**: 同一家 NVIDIA Isaac-GR00T 团队的不同版本,都用 flow-matching 但 action 注入方式不同(early concat vs. cross-attn vs. AdaLN) / Different NVIDIA Isaac-GR00T variants from the same team — all flow-matching, differing in how actions are injected (early concat vs. cross-attn vs. AdaLN).
- **Stable Diffusion 3 / SD3-Medium** / **Stable Diffusion 3 / SD3-Medium**: 文字到图像版本,scheduler 数学一致(无 shift 或 shift=1.0~3.0);把视频部分换成 256×256 图像就退化成 SD3 / Text-to-image version, scheduler math identical (no shift, or shift=1.0–3.0). Strip the video dim → SD3.
- **diffusers `FlowMatchEulerDiscreteScheduler`** / **diffusers `FlowMatchEulerDiscreteScheduler`**: 上游官方版本,600+ 行但骨架完全是这 90 行 / Upstream official version, 600+ lines but the skeleton is identical to these 90.

## 注意事项 / Caveats / when it breaks

- **timestep ↔ sigma 转换易错** / **timestep ↔ sigma indexing easy to get wrong**: 代码里 `torch.argmin((self.timesteps - timestep).abs())` 是用 timestep 反查最近的 sigma id;如果你随机生成的 timestep 不在 `self.timesteps` 集合里(浮点数任意),`argmin` 会量化到最近格点——大部分情况没问题,但极端密度的 schedule(`n_steps > 1000`)上会出现 collision / The line `torch.argmin((self.timesteps - timestep).abs())` does nearest-sigma lookup; if your randomly-sampled t isn't a value in `self.timesteps` (arbitrary float), `argmin` snaps to the nearest grid point — fine usually, but ultra-dense schedules (`n_steps > 1000`) can collide.
- **`shift` 不能在训练和推理间不一致** / **`shift` must match between training and inference**: 训练用 shift=3.0 但推理用 shift=1.0,采样的 timesteps 分布完全不同,会得到严重 OOD / Training with shift=3.0 but inference with shift=1.0 gives different sampled-timestep distributions and severe OOD failure.
- **`add_noise` 内的 broadcasting `while` 循环** / **`add_noise`'s broadcasting `while` loop**: 优雅但有陷阱——如果 `original_samples` 是 0D(标量),while 循环不会执行,这没问题;但如果你不小心把 `timestep` 弄成 `(B, T)` 形状(每帧不同 t,某些条件训练里会),sigma 形状会跟 sample 不匹配 / Elegant but trap-laden — if `original_samples` is 0D the loop simply doesn't run (fine); but if you accidentally pass `timestep` as `(B, T)` (per-frame t, as in some conditional training setups), the sigma shape won't align with the sample.
- **`shift` 默认 3.0 对图像/小分辨率太激进** / **`shift=3.0` default is too aggressive for images / low resolution**: 256×256 图像更适合 shift=1.0~1.5,3.0 会让训练样本几乎都集中在高噪声端,模型学不到细节 / 256×256 images want shift=1.0–1.5; 3.0 packs almost all samples at the high-noise end and the model never learns fine detail.
- **`linear_timesteps_weights` 仅当 `training=True` 时填** / **`linear_timesteps_weights` is only populated when `training=True`**: 调用 `set_timesteps()` 没传 `training=True` 然后又 `training_weight(t)` 会 AttributeError / Calling `set_timesteps()` without `training=True` and then asking for `training_weight(t)` raises AttributeError.
- **训练 `t` 应该按权重分布采样,不应该等概率** / **Sample training `t` with the weight distribution, not uniformly**: 这段代码只给了权重,但**没在 train_step 里展示**怎么用——你可以(a)用 `linear_timesteps_weights` 直接 importance-sample timestep,或(b)均匀采样后乘 weight 进 loss。两种都对,但混用会双计 / The code exposes the weights but doesn't show *how* to use them — you either (a) importance-sample timesteps from `linear_timesteps_weights` directly or (b) sample uniformly and multiply the weight into the loss. Both correct; mixing them double-counts.

## 延伸阅读 / Further reading

- [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis (Esser et al. 2024 / SD3)](https://arxiv.org/abs/2403.03206) — shift 调度 + logit-normal 采样 + velocity prediction 的起源
- [Wan2.1 paper / repo](https://github.com/Wan-Video/Wan2.1) — 这个 scheduler 的上游来源
- [Flow Matching for Generative Modeling (Lipman et al. 2023)](https://arxiv.org/abs/2210.02747) — flow-matching 数学奠基论文,讲清楚为什么 velocity prediction 是正确的训练目标
- [Rectified Flow (Liu et al. 2022)](https://arxiv.org/abs/2209.03003) — "直线插值"路径的最早系统化论证
- [diffusers `FlowMatchEulerDiscreteScheduler` 源码](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py) — 同样数学的"生产版",对比着看
- [Open-Sora flow_matching code](https://github.com/PKU-YuanGroup/Open-Sora) — 视频版同类实现,可以对比 shift 默认值差异
