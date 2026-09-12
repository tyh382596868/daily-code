---
date: 2026-07-26
topic: wam
source: wam
repo: hpcaitech/Open-Sora
file: opensora/schedulers/rf/__init__.py
permalink: https://github.com/hpcaitech/Open-Sora/blob/main/opensora/schedulers/rf/__init__.py
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, wam, sampler-inference, rectified-flow]
build_role: sampler-inference advanced variant, rectified-flow denoising loop
---

# Open-Sora RFLOW：采样器负责时间表和 CFG 组合 / Open-Sora RFLOW: The Sampler Owns Timesteps and CFG Mixing

> **一句话 / In one line**: RFLOW sampler 先生成 timestep 序列，再在每一步调用模型并把预测推进到下一噪声级别。 / The RFLOW sampler builds a timestep sequence, calls the model at each step, and advances the latent to the next noise level.

## 为什么重要 / Why this matters

在 WAM 里，sampler 不是附属工具。它决定模型看哪些噪声时间、条件和无条件分支怎么混合、每一步 latent 如何更新。换掉 sampler，运动一致性和速度都会变。

In a WAM, the sampler is not a side utility. It decides which noise times the model sees, how conditional and unconditional branches mix, and how the latent changes at every step. Change the sampler and you change motion quality and speed.

## 代码 / The code

`hpcaitech/Open-Sora` — [`opensora/schedulers/rf/__init__.py`](https://github.com/hpcaitech/Open-Sora/blob/main/opensora/schedulers/rf/__init__.py)

```python
class RFLOW:
    def __init__(self, num_sampling_steps=30, cfg_scale=7.0):
        self.num_sampling_steps = num_sampling_steps
        self.cfg_scale = cfg_scale

    def sample(self, model, z, model_args):
        timesteps = torch.linspace(1.0, 0.0, self.num_sampling_steps + 1, device=z.device)
        for i in range(self.num_sampling_steps):
            t = timesteps[i].expand(z.shape[0])
            pred_cond = model(z, t, **model_args)
            pred_uncond = model(z, t, **{**model_args, "y": None})
            pred = pred_uncond + self.cfg_scale * (pred_cond - pred_uncond)
            dt = timesteps[i + 1] - timesteps[i]
            z = z + dt * pred
        return z
```

## 逐行讲解 / What's happening

1. **`torch.linspace(1.0, 0.0, ...)`**
   - 中文: 时间从纯噪声附近走向干净 latent，`+1` 是为了每一步都有当前和下一时间。
   - English: Time moves from noisy to clean latent; `+1` gives every step both current and next time.
2. **条件/无条件两次 forward / Conditional and unconditional forwards**
   - 中文: CFG 需要同一个 latent 在有 prompt 和无 prompt 下各预测一次。
   - English: CFG needs two predictions for the same latent: one with condition and one without.
3. **`pred_uncond + scale * (...)`**
   - 中文: 这是 classifier-free guidance 的经典组合，把条件方向放大。
   - English: This is the standard CFG mix, amplifying the conditional direction.
4. **`z = z + dt * pred`**
   - 中文: 和 Euler flow 一样，预测是速度，`dt` 是步长。
   - English: As in Euler flow, the prediction is velocity and `dt` is the step size.

## 类比 / The analogy

这像按导航开车：路线表给出每个路口，模型告诉你方向盘怎么打，CFG 像副驾不断强调“更像目标地点”。

It is like driving with navigation: the route gives intersections, the model steers, and CFG is the passenger repeatedly nudging you toward the destination.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

中文: 这是 `sampler-inference` 组件。上游是 noise scheduler 和 DiT block，下游是 VAE decoder。省掉它就只有会预测速度的模型，却没有把噪声 latent 变成视频的过程。

English: This is the `sampler-inference` component. Upstream are the noise scheduler and DiT block; downstream is the VAE decoder. Without it, you have a model that predicts velocity but no procedure that turns noisy latents into video.

## 自己跑一遍 / Try it yourself

```python
def model(z, t, cond):
    return (1 if cond else 0.4) * z

z, scale = 10.0, 2.0
steps = [1.0, 0.5, 0.0]
for a, b in zip(steps, steps[1:]):
    cond, uncond = model(z, a, True), model(z, a, False)
    pred = uncond + scale * (cond - uncond)
    z = z + (b - a) * pred
    print(round(z, 2))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
2.0
0.4
```

中文: CFG 放大了条件速度，所以 latent 变化比无条件预测更剧烈。
English: CFG amplifies the conditional velocity, so the latent moves more strongly than the unconditional prediction.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Wan2.1 T2V sampler** / **Wan2.1 T2V sampler**: 中文: 也会正负 prompt 各跑一次再合成。 / English: It also runs positive and negative prompts before mixing.
- **Diffusers FlowMatch Euler** / **Diffusers FlowMatch Euler**: 中文: 更新式同样是当前样本加上时间差乘速度。 / English: The update is also current sample plus time delta times velocity.

## 注意事项 / Caveats / when it breaks

- **CFG 太大** / **CFG too large**: 中文: 会让视频过饱和、抖动或丢运动一致性。 / English: It can oversaturate video, add jitter, or hurt motion consistency.
- **时间表粗糙** / **Coarse schedule**: 中文: 步数少时速度快，但高频细节更容易坏。 / English: Fewer steps are faster but can damage high-frequency detail.

## 延伸阅读 / Further reading

- [Open-Sora repository](https://github.com/hpcaitech/Open-Sora)
- [Open-Sora schedulers](https://github.com/hpcaitech/Open-Sora/tree/main/opensora/schedulers)
