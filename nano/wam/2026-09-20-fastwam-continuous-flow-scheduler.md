---
date: 2026-09-20
topic: wam
source: wam
repo: yuantianyuan01/FastWAM
file: src/fastwam/models/wan22/schedulers/scheduler_continuous.py
permalink: https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/schedulers/scheduler_continuous.py#L4-L88
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, flow-matching, scheduler, wan22]
build_role: noise-scheduler advanced variant
---

# FastWAM continuous scheduler：训练时间、加噪目标和推理步长共用一条轴 / FastWAM Continuous Scheduler: Training Time, Noise Targets, and Inference Steps Share One Axis

> **一句话 / In one line**: FastWAM 用一个带 shift 的 flow-matching scheduler 统一训练时的时间采样、loss weight、latent 加噪、velocity target 和推理 Euler 更新。 / FastWAM uses one shifted flow-matching scheduler for training-time sampling, loss weighting, latent noising, velocity targets, and Euler inference updates.

## 为什么重要 / Why this matters

WAM 的视频 latent 和动作 latent 都要在一条连续时间轴上从噪声走向数据。如果训练时按一种 timestep 分布采样，推理时却按另一种 sigma 表走，模型会在它不熟悉的速度场上工作。一个 scheduler 把这些契约放在同一个对象里，更容易保证训练和采样一致。

WAM video and action latents both move along a continuous time axis from noise toward data. If training samples timesteps from one distribution but inference uses another sigma path, the model operates on a velocity field it did not learn. Keeping the contracts in one scheduler makes training and sampling consistent.

这份实现还把两个细节显式化：`_phi` 用 `shift` 改变时间分布，`training_weight` 用高斯形状给中间时间段更合适的权重；推理端则从 `u=1` 走到 `u=0`，将相邻 sigma 的差作为 Euler `delta`。

The implementation makes two details explicit: `_phi` reshapes the time distribution with `shift`, while `training_weight` uses a Gaussian-shaped weighting around the middle of the trajectory. In inference, `u` runs from `1` to `0`, and adjacent sigma differences become Euler deltas.

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

    def build_inference_schedule(self, num_inference_steps, device, dtype, shift_override=None):
        shift = self.shift if shift_override is None else float(shift_override)
        u_steps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)
        sigma_steps = self._phi(u_steps, shift)
        timesteps = sigma_steps[:-1] * float(self.num_train_timesteps)
        deltas = sigma_steps[1:] - sigma_steps[:-1]
        return timesteps.to(dtype=dtype), deltas.to(dtype=dtype)

    @staticmethod
    def step(model_output: torch.Tensor, delta: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        delta = delta.to(sample.device, dtype=sample.dtype)
        return sample + model_output * delta
```

## 逐行讲解 / What's happening

1. **shifted time map / Shifted time map**:
   - 中文: `_phi(u, shift)` 把均匀的 `u` 映射到非均匀 sigma；shift 改变模型在不同噪声区间看到的样本密度。
   - English: `_phi(u, shift)` maps uniform `u` to nonuniform sigma; shift changes how densely the model sees different noise regions.
2. **训练采样 / Training sampling**:
   - 中文: `sample_training_t` 先采样 `[0,1]` 的 `u`，再映射到模型使用的 timestep。
   - English: `sample_training_t` samples `u` in `[0,1]`, then maps it to the timestep used by the model.
3. **加噪 / Adding noise**:
   - 中文: `(1-sigma) * sample + sigma * noise` 形成直线 flow path；batch sigma 会 reshape 成可广播的尾部维度。
   - English: `(1-sigma) * sample + sigma * noise` forms a straight flow path; batch sigma is reshaped for broadcasting over trailing dimensions.
4. **监督速度 / Target velocity**:
   - 中文: 对这条线，速度就是 `noise - sample`，因此模型学习从当前 noisy latent 指向 noise 的方向。
   - English: For this line, velocity is `noise - sample`, so the model learns the direction from the current noisy latent toward noise.
5. **训练权重 / Training weight**:
   - 中文: 中间时间段的权重经过 min-shift 与 normalization，避免权重整体偏移或平均值失控。
   - English: Mid-trajectory weights are min-shifted and normalized so the schedule does not carry an arbitrary offset or uncontrolled mean.
6. **Euler 推理 / Euler inference**:
   - 中文: `delta = sigma_next - sigma_current`，`sample + model_output * delta` 就是最小的 ODE 更新。
   - English: `delta = sigma_next - sigma_current`, and `sample + model_output * delta` is the minimal ODE update.

## 类比 / The analogy

把 latent 想成一辆在雾里开向终点的车。训练时不仅要告诉它当前位置，还要决定在哪些路段多采样；推理时则按同一张路线图给每一段计算里程，模型预测的速度乘以这段路长就是下一站。

Think of the latent as a car driving through fog toward a destination. Training chooses which road segments to visit more often; inference uses the same route map to compute each segment length, and predicted velocity times that length reaches the next stop.

## 在 nanoWAM 中的位置 / Where this lives in your nanoWAM

中文: 这是 `noise-scheduler` 层，位于 VAE/动作 encoder 产生的 clean latent 与 Wan/DiT backbone 之间。上游提供视频 latent、动作 latent 和随机 noise；scheduler 采样 timestep、构造 noisy input、返回 velocity target 和 training weight；下游把 noisy latent 与条件送入 backbone，并在推理循环中反复调用 `step`。从零实现 nanoWAM 时，先用固定线性 sigma 跑通训练和 Euler 采样，再加入 shift、weight normalization、视频/动作不同的 timestep 策略与 CFG；生产实现还要验证 dtype、batch broadcast、长视频 chunk 和 scheduler state 的可复现性。

English: This is the `noise-scheduler` layer between clean video/action latents from the VAE or action encoder and the Wan/DiT backbone. Upstream provides clean latents and noise; the scheduler samples timesteps, builds noisy inputs, returns velocity targets and training weights; downstream feeds the noisy latents to the backbone and calls `step` in the inference loop. For nanoWAM, start with a fixed linear sigma schedule, then add shift, weight normalization, separate video/action policies, and CFG. Production code must validate dtype, batch broadcasting, long-video chunks, and reproducible scheduler state.

## 自己跑一遍 / Try it yourself

```python
import torch

sample = torch.tensor([1.0])
noise = torch.tensor([0.0])
sigma = torch.tensor([0.25])
noisy = (1 - sigma) * sample + sigma * noise
target = noise - sample
model_output = target
next_sample = noisy + model_output * torch.tensor([-0.25])
print(noisy.item(), target.item(), next_sample.item())
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
0.75 -1.0 1.0
```

中文: 从 clean sample 到 noise 的直线路径，在反向 Euler step 中又回到 sample。
English: The straight path goes from the clean sample toward noise, and a reverse Euler step returns toward the sample.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Wan2.1 flow schedulers** / **Wan2.1 flow schedulers**: 中文: 同样把 time shift、sigma 和采样步长绑定起来。 / English: They similarly bind time shift, sigma, and sampling step size.
- **Diffusers flow matching** / **Diffusers flow matching**: 中文: scheduler 通常同时暴露 add_noise、target 和 step，方便训练与推理共享数学约定。 / English: Schedulers often expose add_noise, target, and step together so training and inference share one mathematical contract.
- **ODE solvers** / **ODE solvers**: 中文: Euler 只是最简单的积分器，更高阶 solver 也必须消费同一 velocity field 和时间坐标。 / English: Euler is the simplest integrator; higher-order solvers still consume the same velocity field and time coordinates.

## 注意事项 / Caveats / when it breaks

- **sigma 方向必须统一** / **Sigma direction must be consistent**: 中文: 训练、推理和 target 的方向一旦混用，模型会学到相反速度。 / English: Mixing directions across training, inference, and target construction makes the model learn the opposite velocity.
- **shift 会改变数据分布** / **Shift changes the data distribution**: 中文: 调整 shift 不是纯推理参数，若训练和采样不一致会产生分布偏移。 / English: Shift is not only an inference parameter; a train/inference mismatch creates distribution shift.
- **weight 不能掩盖 padding** / **Weights do not replace padding masks**: 中文: 视频或动作 chunk 的无效位置仍要在 loss reduction 前单独 mask。 / English: Invalid positions in video or action chunks still need explicit masking before loss reduction.

## 延伸阅读 / Further reading

- [FastWAM continuous scheduler](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/schedulers/scheduler_continuous.py#L4-L88)
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Diffusers schedulers](https://huggingface.co/docs/diffusers/main/en/using-diffusers/schedulers)
