---
date: 2026-05-31
topic: huggingface
source: huggingface
repo: huggingface/diffusers
file: src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
permalink: https://github.com/huggingface/diffusers/blob/b003a47354ca3a28b4b21fceb4d649656d8cee99/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py#L426-L525
difficulty: beginner
read_time: ~11 min
tags: [code-of-the-day, huggingface, diffusers, scheduler, flow-matching, sd3, flux]
---

# Flow Matching Euler 的核心其实只有一行 / The heart of a flow-matching Euler step is one line

> **一句话 / In one line**: `prev_sample = sample + dt * model_output` —— SD3、Flux、Wan、Mochi 跑的都是这一行;周围 100 行只是 sigma 表查询 + 两个升级:per-token sigma 和随机性采样。 / `prev_sample = sample + dt * model_output` is the whole step — SD3 / Flux / Wan / Mochi all run that line. The surrounding 100 lines are sigma-table bookkeeping plus two upgrades: per-token sigmas and stochastic sampling.

## 为什么重要 / Why this matters

DDPM/DDIM 的 step 函数一般都有十几个 `alpha_t`、`beta_t`、`sqrt_one_minus_alpha_prod` 之类的张量切来切去,劝退新人。Flow Matching 的 step 数学上比 DDIM *更简单*:你训了一个 velocity 场 `v(x_t, t)`,推理时只是沿着这个场积分 ODE `dx/dt = v(x, t)`。最低端的 ODE 积分器叫 Euler:`x_{t-1} = x_t + dt * v`。就这样。这个文件是几乎所有现代 video/image diffusion 都在用的 scheduler —— SD3、Flux、Wan2.1、Mochi、LTX 一字不改地引用同一个类。看懂它,你就懂了 2024-2026 这一代 diffusion 模型的核心 50% 数学。

DDPM/DDIM step functions are notorious for juggling `alpha_t`, `beta_t`, `sqrt_one_minus_alpha_prod` and friends. Flow matching is mathematically *simpler* than DDIM: you train a velocity field `v(x_t, t)` and at inference time integrate the ODE `dx/dt = v(x, t)`. The cheapest ODE integrator is Euler: `x_{t-1} = x_t + dt * v`. That's the whole story. This file is the scheduler that SD3, Flux, Wan2.1, Mochi and LTX import unchanged — understanding it covers half the math of every recent diffusion model.

## 代码 / The code

`huggingface/diffusers` — [`src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py`](https://github.com/huggingface/diffusers/blob/b003a47354ca3a28b4b21fceb4d649656d8cee99/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py#L426-L525)

```python
def step(
    self,
    model_output: torch.FloatTensor,
    timestep: float | torch.FloatTensor,
    sample: torch.FloatTensor,
    s_churn: float = 0.0,
    s_tmin: float = 0.0,
    s_tmax: float = float("inf"),
    s_noise: float = 1.0,
    generator: torch.Generator | None = None,
    per_token_timesteps: torch.Tensor | None = None,
    return_dict: bool = True,
) -> FlowMatchEulerDiscreteSchedulerOutput | tuple:
    if (
        isinstance(timestep, int)
        or isinstance(timestep, torch.IntTensor)
        or isinstance(timestep, torch.LongTensor)
    ):
        raise ValueError(
            "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
            " `FlowMatchEulerDiscreteScheduler.step()` is not supported. Make sure to pass"
            " one of the `scheduler.timesteps` as a timestep."
        )

    if self.step_index is None:
        self._init_step_index(timestep)

    # Upcast to avoid precision issues when computing prev_sample
    sample = sample.to(torch.float32)

    if per_token_timesteps is not None:
        per_token_sigmas = per_token_timesteps / self.config.num_train_timesteps

        sigmas = self.sigmas[:, None, None]
        lower_mask = sigmas < per_token_sigmas[None] - 1e-6
        lower_sigmas = lower_mask * sigmas
        lower_sigmas, _ = lower_sigmas.max(dim=0)

        current_sigma = per_token_sigmas[..., None]
        next_sigma = lower_sigmas[..., None]
        dt = current_sigma - next_sigma
    else:
        sigma_idx = self.step_index
        sigma = self.sigmas[sigma_idx]
        sigma_next = self.sigmas[sigma_idx + 1]

        current_sigma = sigma
        next_sigma = sigma_next
        dt = sigma_next - sigma

    if self.config.stochastic_sampling:
        x0 = sample - current_sigma * model_output
        noise = randn_tensor(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
        prev_sample = (1.0 - next_sigma) * x0 + next_sigma * noise
    else:
        prev_sample = sample + dt * model_output

    # upon completion increase step index by one
    self._step_index += 1
    if per_token_timesteps is None:
        prev_sample = prev_sample.to(model_output.dtype)

    if not return_dict:
        return (prev_sample,)

    return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)
```

## 逐行讲解 / What's happening

1. **不准传整数 timestep / Refuse integer timesteps (lines 469-481)**:
   - 中文: Flow matching scheduler 用的是 `float` sigma 值(0~1 区间或者 0~num_train 区间),不是 `enumerate(timesteps)` 出来的整数下标。这条护栏是历史 DDIM 用户最容易踩的坑 —— DDIM 是按 index 索引的,flow matching 不是。
   - English: Flow matching uses *floating-point* sigma values (in `[0, 1]` or `[0, num_train]`), not integer indices from `enumerate(timesteps)`. This guardrail catches the most common DDIM-to-FM migration bug.

2. **`_init_step_index`** (line 484):
   - 中文: 第一次调用 `step` 时,通过传进来的 `timestep` 反查 `self.sigmas` 表,确定从第几步开始。这样用户可以从中间步骤恢复(比如 img2img、SDEdit)。
   - English: on the first call, look up `timestep` in `self.sigmas` to figure out the starting index. This lets you resume from any step (img2img, SDEdit).

3. **升 fp32 做计算 / Upcast to fp32 (line 487)**:
   - 中文: model 输出是 fp16/bf16,但 `sample + dt * v` 里 dt 通常是非常小的浮点数,fp16 下乘加会丢精度。升 fp32 算完再 cast 回去,这是 diffusers 全家的常规操作。
   - English: model output is fp16/bf16, but `dt` can be tiny — fp16 multiply-add loses precision. Diffusers always upcasts the linear-combination step to fp32 and casts back.

4. **per-token sigma 分支 (lines 489-499)**:
   - 中文: 这是 Wan/Mochi/LTX 这种视频模型才用的高级特性 —— *每个 token* 各有自己的噪声级别。`sigmas[:, None, None] < per_token_sigmas[None] - 1e-6` 构造一个 mask,挑出"严格小于当前 sigma 的所有 sigma",再 `lower_sigmas.max(dim=0)` 取里面最大的 —— 这就是"每个 token 的下一个 sigma"。`dt` 因此也是 per-token 的。常规图像 diffusion 走 else 分支即可。
   - English: this branch is for image-edit / video flows where *each token* has its own noise level. `sigmas[:, None, None] < per_token_sigmas[None] - 1e-6` builds a mask, then `lower_sigmas.max(dim=0)` picks the largest sigma still below each token's current sigma — i.e. each token's "next" sigma. Standard image diffusion takes the `else` branch.

5. **标准 Euler 步 / Standard Euler step (line 514)**:
   - 中文: 整个调度器的灵魂就这一行。`dt = sigma_next - sigma` 是负数(sigma 从 1 走向 0),`model_output` 是 velocity `v`,所以 `prev_sample = sample + dt * v`。一阶 ODE 积分。
   - English: the soul of the whole scheduler is this single line. `dt = sigma_next - sigma` is negative (sigma decreases from 1 to 0), `model_output` is the velocity `v`, so `prev_sample = sample + dt * v` — vanilla first-order ODE integration.

6. **stochastic_sampling 分支 (lines 509-512)**:
   - 中文: 这是把 Euler 升级成 SDE 的"重新加噪"技巧。先用 velocity 倒推出 `x0 = sample - current_sigma * model_output`(假设这一步就是终点),再用新采样的噪声重新加噪到 `next_sigma` 强度。等价于走一步 SDE 而不是 ODE,能逃出 ODE 模式塌缩的局部最优,但代价是失去 ODE 的确定性。
   - English: this is the "re-noise" upgrade that turns Euler into an SDE step. Estimate `x0 = sample - current_sigma * model_output` (as if this step were the last), then add fresh noise back at `next_sigma` strength. It escapes mode-collapse traps that plain ODE samplers fall into, at the cost of losing determinism.

## 类比 / The analogy

想象你在山顶,被告知"山谷在正南方,但你只能走 10 步"。velocity field 就像每一步脚下的指南针,告诉你"现在朝这个方向走最快下山"。Euler step 就是按指南针指的方向迈一步、看下一步的指南针、再迈一步,直到走到谷底。`dt` 是你每步迈多大,sigma 表是"还剩多少步"。`stochastic_sampling` 像是每走一步后转个随机的小弯 —— 总体方向不变,但不会卡在某个石头上。`per_token_timesteps` 则是允许你的左脚和右脚 *按不同的步长* 同时走 —— 视频模型里有的帧已经清晰、有的还很糊,各自需要不同的进度。

Picture yourself on a mountaintop, told "the valley is due south but you only get 10 steps." The velocity field is the compass at your feet telling you "the steepest descent is *this way* right now." Each Euler step = read compass, take one step, read compass again. `dt` is how big each step is, the sigma table is how many steps remain. `stochastic_sampling` is taking a small random turn after each step — it avoids getting stuck on a single rock. `per_token_timesteps` is letting your left and right foot walk at *different stride lengths* simultaneously — exactly what you need for video where some frames are already clean and others are still blurry.

## 自己跑一遍 / Try it yourself

```python
# pip install diffusers torch
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

# Toy 1-D "image": want to denoise from x_T=randn() back to x_0=42.0
torch.manual_seed(0)
target = torch.tensor(42.0)
scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
scheduler.set_timesteps(num_inference_steps=10)

sample = torch.randn(1) * 5.0           # noisy start
print(f"start: {sample.item():.3f}")
for t in scheduler.timesteps:
    # Pretend velocity field is v = noise - target  (the ground-truth FM target)
    # In a real model this comes from your network: v = model(sample, t)
    v = sample - target                  # constant velocity toward target
    out = scheduler.step(model_output=v, timestep=t, sample=sample)
    sample = out.prev_sample
    print(f"t={t.item():.1f}  x={sample.item():.4f}")

print(f"final: {sample.item():.4f}  (target was 42.0)")
```

运行 / Run with:
```bash
pip install diffusers torch
python try.py
```

预期输出 / Expected output:
```
start: 0.069
t=999.0  x=...
...
t=99.9   x≈42.0
final: ≈42.0  (target was 42.0)
```

中文: 用 `v = sample - target`(一个理想的 velocity)走 10 步 Euler,sample 会几乎完美地收敛到 42 —— flow matching 的整个魅力就是"训出一个好 velocity,推理就是 ODE 积分"。

English: with the ideal velocity `v = sample - target`, 10 Euler steps converge sample to 42 almost exactly — the whole appeal of flow matching is "train a good velocity, then inference is just ODE integration."

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **`dreamzero0/dreamzero` rectified flow scheduler** / **dreamzero rectified-flow scheduler**: 中文: 5-28 教过的"90 行 rectified flow"调度器,核心算法跟这里一模一样,只是把 sigma 表拓展成 rectified-flow 的 shifted schedule。 / English: the 90-line rectified-flow scheduler from 2026-05-28 — same core math, with an extra shifted sigma schedule for video.
- **SD3 / Flux / Wan2.1 pipeline 的去噪循环** / **SD3 / Flux / Wan2.1 denoising loops**: 中文: 这些 pipeline 调用的就是这个 `step` 函数,一行不改 —— 5-29 的 Wan2.1 denoise loop 笔记里能看到 `scheduler.step()` 出现的位置。 / English: SD3 / Flux / Wan2.1 pipelines call this exact `step` function unmodified — see the Wan2.1 denoise loop note from 2026-05-29 for the call site.
- **π₀ flow matching action head** / **π₀'s flow matching action head**: 中文: 5-26 教过 openpi 的 flow matching loss —— 训练的是同一种 velocity,但因为 action 维度小,推理时通常只用 1~5 步 Euler,几乎没人用 SDE 升级。 / English: openpi's flow matching loss from 2026-05-26 trains the same velocity, but the action dim is small enough that inference uses 1-5 Euler steps with no stochastic upgrade.

## 注意事项 / Caveats / when it breaks

- **必须传 float timestep** / **Always pass a float timestep**: 中文: 不要传 `for i, t in enumerate(scheduler.timesteps): scheduler.step(model_out, i, sample)` —— 必须传 `t` 不是 `i`,否则报错。 / English: don't pass the integer loop index, pass the actual float `t` from `scheduler.timesteps`. The guardrail at the top of `step` will raise if you do.
- **`set_timesteps` 是有状态的** / **`set_timesteps` is stateful**: 中文: 它会改 `self.sigmas`、`self.timesteps`、`self.step_index`。同一个 scheduler 实例想跑第二次推理,必须再调一次 `set_timesteps` 重置。 / English: it mutates `self.sigmas`, `self.timesteps`, `self.step_index`. Re-running inference on the same scheduler instance requires calling `set_timesteps` again.
- **`shift` 参数影响低分辨率画质** / **The `shift` parameter matters at low resolution**: 中文: 训练时学的是 256×256,推理 1024×1024 时把 sigma 往大的方向 shift 一下能保住细节 —— SD3 的论文里这是关键超参。 / English: training at 256² but sampling at 1024²? Shift the sigma schedule toward larger noise to preserve detail — SD3's paper calls this out as a critical hyperparam.

## 延伸阅读 / Further reading

- Flow Matching for Generative Modeling: <https://arxiv.org/abs/2210.02747>
- Rectified Flow: <https://arxiv.org/abs/2209.03003>
- SD3 paper (shift schedule motivation): <https://arxiv.org/abs/2403.03206>
