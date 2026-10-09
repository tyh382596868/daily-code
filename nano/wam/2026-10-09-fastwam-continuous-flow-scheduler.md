---
date: 2026-10-09
topic: wam
source: wam
repo: yuantianyuan01/FastWAM
file: src/fastwam/models/wan22/schedulers/scheduler_continuous.py
permalink: https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/schedulers/scheduler_continuous.py#L4-L88
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, wam, flow-matching-scheduler]
build_role: noise-scheduler advanced variant
---
# FastWAM continuous scheduler：训练抽 t，推理走 delta / FastWAM Continuous Scheduler: Sample t for Training, Step by Delta at Inference

> **一句话 / In one line**: 一个 shift 函数同时塑造训练时间分布、加噪比例、推理时间表和 Euler 更新。 / One shifted time map shapes training samples, noisy interpolation, inference schedule, and Euler updates.

## 为什么重要 / Why this matters

WAM 的视频/动作 latent 不是随便加噪。这个 scheduler 把连续时间 `u` 通过 `_phi` 变形，让训练更偏向某些噪声区间；训练目标固定为 `noise - sample`，推理时则用相邻 sigma 的差值推进样本。小类里藏着训练和采样的一致性契约。

A WAM should not add noise with an ad hoc clock. This scheduler maps uniform time through `_phi`, biases training toward selected noise regions, defines the velocity target as `noise - sample`, and advances inference with neighboring sigma deltas. It is a compact contract between training and sampling.

## 代码 / The code

`yuantianyuan01/FastWAM` — [`src/fastwam/models/wan22/schedulers/scheduler_continuous.py`](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/schedulers/scheduler_continuous.py#L4-L88)

```python
class WanContinuousFlowMatchScheduler:
    """Continuous-time Flow-Matching scheduler with shift-based sampling."""

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 5.0, eps: float = 1e-10):
        if num_train_timesteps <= 0:
            raise ValueError(f"`num_train_timesteps` must be positive, got {num_train_timesteps}")
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}")
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.eps = float(eps)
        self._y_min, self._weight_norm_const = self._precompute_training_weight_stats()

    @staticmethod
    def _phi(u: torch.Tensor, shift: float) -> torch.Tensor:
        return shift * u / (1.0 + (shift - 1.0) * u)

    def _precompute_training_weight_stats(self) -> tuple[float, float]:
        steps = self.num_train_timesteps
        u_grid = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)[:-1]
        t_grid = self._phi(u_grid, self.shift) * float(steps)
        y_grid = torch.exp(-2.0 * ((t_grid - (steps / 2.0)) / steps) ** 2)
        y_min = float(y_grid.min().item())
        y_shifted_grid = y_grid - y_min
        norm_const = float(y_shifted_grid.mean().item())
        return y_min, norm_const

    def sample_training_t(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError(f"`batch_size` must be positive, got {batch_size}")
        u = torch.rand((batch_size,), device=device, dtype=torch.float32)
        sigma = self._phi(u, self.shift)
        timestep = sigma * float(self.num_train_timesteps)
        return timestep.to(dtype=dtype)

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        t = timestep.to(dtype=torch.float32)
        steps = float(self.num_train_timesteps)
        y = torch.exp(-2.0 * ((t - (steps / 2.0)) / steps) ** 2)
        y_shifted = y - self._y_min
        weight = y_shifted / (self._weight_norm_const + self.eps)
        if weight.numel() == 1:
            return weight.reshape(())
        return weight

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        sigma = (timestep / float(self.num_train_timesteps)).to(
            original_samples.device, dtype=original_samples.dtype
        )
        if sigma.ndim == 0:
            return (1 - sigma) * original_samples + sigma * noise
        sigma = sigma.view(-1, *([1] * (original_samples.ndim - 1)))
        return (1 - sigma) * original_samples + sigma * noise

    @staticmethod
    def training_target(sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        del timestep
        return noise - sample

    def build_inference_schedule(
        self,
        num_inference_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        shift_override: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be positive, got {num_inference_steps}")
        shift = self.shift if shift_override is None else float(shift_override)
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}")

        u_steps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device, dtype=torch.float32)
        sigma_steps = self._phi(u_steps, shift)
        timesteps = sigma_steps[:-1] * float(self.num_train_timesteps)
        deltas = sigma_steps[1:] - sigma_steps[:-1]
        return timesteps.to(dtype=dtype), deltas.to(dtype=dtype)

    @staticmethod
    def step(model_output: torch.Tensor, delta: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        delta = delta.to(sample.device, dtype=sample.dtype)
        if delta.ndim == 0:
            return sample + model_output * delta
        delta = delta.view(-1, *([1] * (sample.ndim - 1)))
        return sample + model_output * delta
```

## 逐行讲解 / What's happening

1. **第 17-19 行 / Lines 17-19 (`_phi`)**:
   - 中文: `shift` 把均匀时间弯成非线性 sigma，控制高噪声/低噪声区域的采样密度。
   - English: `shift` bends uniform time into nonlinear sigma values, controlling where samples concentrate.
2. **第 31-37 行 / Lines 31-37 (`sample_training_t`)**:
   - 中文: 训练时先随机抽 `u`，再映射成 scheduler 的 timestep。
   - English: Training samples uniform `u`, then maps it into scheduler timesteps.
3. **第 49-61 行 / Lines 49-61 (noise and target)**:
   - 中文: noisy sample 是 clean 和 noise 的线性插值，目标速度是 `noise - sample`。
   - English: The noisy sample is a linear interpolation between clean data and noise, and the target velocity is `noise - sample`.
4. **第 63-88 行 / Lines 63-88 (inference schedule)**:
   - 中文: 推理得到 timesteps 和 deltas；`step` 用 `sample + velocity * delta` 更新。
   - English: Inference returns timesteps and deltas; `step` updates with `sample + velocity * delta`.

## 类比 / The analogy

像在山路开车。训练阶段你决定哪些坡段多练几次，推理阶段则按路标之间的距离踩油门；两者用的是同一张路线图。

It is like driving on a mountain road. During training you decide which slopes deserve more practice; during inference you drive according to the distances between signs. Both use the same map.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-WAM

中文: 这是 `noise-scheduler` 的高级变体，负责三件事：训练时怎么抽 `t`，如何把 clean latent 和 noise 混成 noisy sample，以及推理时每一步沿速度场走多远。在 nanoWAM 里，它位于 VAE latent 之后、DiT denoiser 之前；如果省掉它，训练 loss 和采样轨迹就没有共同的时间坐标。

English: This is an advanced `noise-scheduler` component. It chooses training times, mixes clean latents with noise, and defines inference step deltas along the velocity field. In a nanoWAM it sits after VAE latents and before the DiT denoiser. Without it, the training target and sampling trajectory do not share a coherent time coordinate.

## 自己跑一遍 / Try it yourself

```python
def phi(u, shift):
    return shift * u / (1 + (shift - 1) * u)

def schedule(steps, shift):
    us = [1 - i / steps for i in range(steps + 1)]
    sigmas = [phi(u, shift) for u in us]
    return sigmas[:-1], [b - a for a, b in zip(sigmas[:-1], sigmas[1:])]

sigmas, deltas = schedule(4, 5.0)
print([round(x, 3) for x in sigmas])
print([round(x, 3) for x in deltas])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
[1.0, 0.938, 0.833, 0.625]
[-0.062, -0.104, -0.208, -0.625]
```

shift 后的时间表不是线性下降，最后一步会更大。

After shifting, the schedule is not linearly spaced; the final step is much larger.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Wan / FlowMatch schedulers** / **Wan / FlowMatch schedulers**: 视频生成常用 shift 改变时间密度。 / Video generation often uses shifted flow schedules.
- **动作世界模型** / **Action world models**: 动作 latent 和视频 latent 共流时，scheduler 是二者共享的时间坐标。 / When action and video latents share a flow, the scheduler is their shared clock.

## 注意事项 / Caveats / when it breaks

- **delta 符号要和速度定义一致** / **Delta sign must match velocity definition**: 这里 delta 是后一个 sigma 减当前 sigma，通常为负。 / Here delta is next sigma minus current sigma, usually negative.
- **权重归一化依赖网格** / **Weight normalization depends on the grid**: `_precompute_training_weight_stats` 用离散训练步估计均值。 / The weight normalization constant is estimated on the discrete training grid.

## 延伸阅读 / Further reading

- [FastWAM continuous scheduler](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/schedulers/scheduler_continuous.py#L4-L88)
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
