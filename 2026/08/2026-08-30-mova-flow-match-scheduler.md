---
date: 2026-08-30
topic: diffusion
source: trending
repo: OpenMOSS/MOVA
file: mova/diffusion/schedulers/flow_match.py
permalink: https://github.com/OpenMOSS/MOVA/blob/d5b7899dd4dd4a92dfca610cafe70b978ead3064/mova/diffusion/schedulers/flow_match.py#L43-L90
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, flow-matching, scheduler]
---

# MOVA FlowMatchScheduler：sigma 时间表决定每一步走多远 / MOVA FlowMatchScheduler: The Sigma Schedule Decides Each Step Size

> **一句话 / In one line**: MOVA 的 flow-match scheduler 先构造可 shift 的 sigma 时间表，再用 `sample + velocity * delta_sigma` 做 Euler 更新。 / MOVA's flow-match scheduler builds a shiftable sigma schedule, then applies an Euler update with `sample + velocity * delta_sigma`.

## 为什么重要 / Why this matters

视频和音频联合生成里，scheduler 是采样循环的节拍器。模型预测的是从噪声到数据的速度场，scheduler 决定当前 sigma、下一步 sigma，以及这次速度应该积分多远。

In joint video-audio generation, the scheduler is the metronome of the sampling loop. The model predicts a velocity field from noise to data; the scheduler decides the current sigma, the next sigma, and how far that velocity should be integrated.

## 代码 / The code

`OpenMOSS/MOVA` — [`mova/diffusion/schedulers/flow_match.py`](https://github.com/OpenMOSS/MOVA/blob/d5b7899dd4dd4a92dfca610cafe70b978ead3064/mova/diffusion/schedulers/flow_match.py#L43-L90)

```python
def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False, shift=None, dynamic_shift_len=None, device=None):
    if shift is not None:
        self.shift = shift
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
    # Cache the initial train timesteps/sigmas the first time we set them.
    if self.train_timesteps is None:
        self.train_timesteps = self.timesteps
        self.train_sigmas = self.sigmas
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
```

## 逐行讲解 / What's happening

1. **第 43-50 行 / Lines 43-50**:
   - 中文: `denoising_strength` 决定起点 sigma，`extra_one_step` 决定是否多建一步再裁掉尾部。
   - English: `denoising_strength` sets the starting sigma, and `extra_one_step` decides whether to create one extra point and trim the tail.
2. **第 51-63 行 / Lines 51-63**:
   - 中文: sigma 可以反向、指数 shift、普通 shift、终点重标定，或整体翻转成 `1 - sigma`。
   - English: Sigmas can be reversed, exponentially shifted, ordinarily shifted, terminal-rescaled, or flipped into `1 - sigma`.
3. **第 64-78 行 / Lines 64-78**:
   - 中文: timestep 只是 sigma 乘训练步数；训练模式还额外生成一个中段更重的权重曲线。
   - English: Timesteps are sigmas multiplied by training steps; training mode also builds a middle-heavy weighting curve.
4. **第 80-90 行 / Lines 80-90**:
   - 中文: `step` 找到最近 timestep，取当前和下一步 sigma，用速度场乘 `delta_sigma` 更新样本。
   - English: `step` finds the nearest timestep, reads current and next sigma, and updates the sample by multiplying the velocity field by `delta_sigma`.

## 类比 / The analogy

这像按路标下山：模型告诉你当前坡度方向，scheduler 决定每两个路标之间隔多远。路标密一点就走得细，shift 后路标会在某些海拔更密集。

It is like hiking downhill by trail markers: the model tells you the slope direction, and the scheduler decides the distance between markers. Denser markers give finer steps, and shifting concentrates markers around certain elevations.

## 自己跑一遍 / Try it yourself

```python
sigmas = [1.0, 0.6, 0.2, 0.0]
sample = 10.0
velocity = -2.0

for i in range(len(sigmas) - 1):
    delta = sigmas[i + 1] - sigmas[i]
    sample = round(sample + velocity * delta, 2)
    print(i, delta, sample)
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
0 -0.4 10.8
1 -0.39999999999999997 11.6
2 -0.2 12.0
```

同一个速度预测，因为 sigma 间隔不同，每一步对样本的实际改变量也不同。

The same velocity prediction changes the sample by different amounts because the sigma gaps differ.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Diffusers FlowMatchEulerDiscreteScheduler** / **Diffusers FlowMatchEulerDiscreteScheduler**: 也用 sigma 时间表和 Euler 步长更新 latent。 / It also updates latents with a sigma schedule and Euler step size.
- **Wan2.1 flow scheduler** / **Wan2.1 flow scheduler**: 采样前会重建 timestep/sigma 表。 / It rebuilds the timestep/sigma table before sampling.
- **Open-Sora RFLOW** / **Open-Sora RFLOW**: sampler 控制 timestep、CFG 和模型调用节奏。 / The sampler controls timesteps, CFG, and model-call rhythm.

## 注意事项 / Caveats / when it breaks

- **timestep 最近邻可能有误差** / **Nearest timestep can introduce error**: 如果传入 timestep 不在表上，会落到最近 sigma。 / If the input timestep is off-grid, it snaps to the nearest sigma.
- **shift 改变采样密度** / **Shift changes sampling density**: 同样步数下，不同区域会得到不同关注度。 / With the same number of steps, different regions receive different attention.
- **方向约定要一致** / **Direction conventions must match**: `reverse_sigmas`、`inverse_timesteps` 和模型输出方向不一致会把采样推反。 / If `reverse_sigmas`, `inverse_timesteps`, and model-output direction disagree, sampling moves the wrong way.

## 延伸阅读 / Further reading

- MOVA scheduler source: https://github.com/OpenMOSS/MOVA/blob/d5b7899dd4dd4a92dfca610cafe70b978ead3064/mova/diffusion/schedulers/flow_match.py
- OpenMOSS/MOVA repository: https://github.com/OpenMOSS/MOVA
