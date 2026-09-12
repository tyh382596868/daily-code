---
date: 2026-08-06
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/utils/fm_solvers.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/utils/fm_solvers.py#L821-L856
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, wam, noise-scheduler]
build_role: noise-scheduler advanced variant, scheduler-aware add_noise for training and img2img/inpainting
---

# Wan2.1 add_noise：同一个加噪函数要懂训练和 inpainting / Wan2.1 add_noise: One Noise Function Must Understand Training and Inpainting

> **一句话 / In one line**: `add_noise` 不只做 `alpha*x + sigma*noise`，还要根据 scheduler 当前状态选对 sigma 索引。 / `add_noise` is more than `alpha*x + sigma*noise`; it must pick sigma indices from the scheduler state.

## 为什么重要 / Why this matters

WAM 训练、img2img、inpainting 都会把干净 latent 加噪，但它们的时间起点不同。训练时按传入 timestep 查表；inpainting 可能已经走过若干 denoise step；img2img 初始 latent 则从 `begin_index` 开始。这个函数把这些分支统一到一条加噪公式里。

WAM training, img2img, and inpainting all add noise to clean latents, but they start from different positions in the schedule. Training indexes by the passed timestep, inpainting may happen after denoising has started, and img2img starts from `begin_index`. This function unifies those cases under one noise formula.

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/utils/fm_solvers.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/utils/fm_solvers.py#L821-L856)

```python
def add_noise(
    self,
    original_samples: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.IntTensor,
) -> torch.Tensor:
    # Make sure sigmas and timesteps have the same device and dtype as original_samples
    sigmas = self.sigmas.to(
        device=original_samples.device, dtype=original_samples.dtype)
    if original_samples.device.type == "mps" and torch.is_floating_point(
            timesteps):
        # mps does not support float64
        schedule_timesteps = self.timesteps.to(
            original_samples.device, dtype=torch.float32)
        timesteps = timesteps.to(
            original_samples.device, dtype=torch.float32)
    else:
        schedule_timesteps = self.timesteps.to(original_samples.device)
        timesteps = timesteps.to(original_samples.device)

    # begin_index is None when the scheduler is used for training or pipeline does not implement set_begin_index
    if self.begin_index is None:
        step_indices = [
            self.index_for_timestep(t, schedule_timesteps)
            for t in timesteps
        ]
    elif self.step_index is not None:
        # add_noise is called after first denoising step (for inpainting)
        step_indices = [self.step_index] * timesteps.shape[0]
    else:
        # add noise is called before first denoising step to create initial latent(img2img)
        step_indices = [self.begin_index] * timesteps.shape[0]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < len(original_samples.shape):
        sigma = sigma.unsqueeze(-1)

    alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
    noisy_samples = alpha_t * original_samples + sigma_t * noise
    return noisy_samples
```

## 逐行讲解 / What's happening

1. **第 823-835 行 / Lines 823-835 (device and dtype)**:
   - 中文: scheduler 表和 sample 必须在同一个 device/dtype 上，MPS 还要避开 float64。
   - English: Scheduler tables must match the sample device and dtype; MPS also needs a float64 workaround.
2. **第 837-848 行 / Lines 837-848 (which sigma)**:
   - 中文: 训练、inpainting、img2img 三种状态分别走不同索引来源。
   - English: Training, inpainting, and img2img each choose sigma indices from a different scheduler state.
3. **第 850-852 行 / Lines 850-852 (broadcast)**:
   - 中文: sigma 是每个样本一个标量，需要一路 `unsqueeze` 到 latent 维度才能广播。
   - English: Sigma starts as one scalar per sample and is repeatedly unsqueezed until it broadcasts across latent dimensions.
4. **第 854-856 行 / Lines 854-856 (mixing)**:
   - 中文: 最后仍然回到扩散核心公式：干净样本乘 `alpha_t`，噪声乘 `sigma_t`。
   - English: The final operation is still the diffusion core: clean sample times `alpha_t`, noise times `sigma_t`.

## 类比 / The analogy

像给咖啡加奶：公式都是“咖啡 + 牛奶”，但你要先知道这是试饮杯、半杯咖啡，还是已经被客人喝过一口的杯子，奶量才不会错。

It is like adding milk to coffee: the formula is always coffee plus milk, but the amount depends on whether it is a sample cup, a half-full cup, or a cup the customer already sipped.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

在 nanoWAM 里，它属于 `noise-scheduler`。训练 loop 用它构造 noisy video/action latents；img2img 或 inpainting sampler 用它从中途时间点初始化 latent。省掉状态分支后，训练也许还能跑，但编辑类推理会在错误噪声强度上开始。

In a nanoWAM, this is part of the `noise-scheduler`. The training loop uses it to create noisy video/action latents; img2img or inpainting samplers use it to initialize from a mid-schedule point. If you omit the state-aware branches, training may still run, but editing-style inference starts at the wrong noise level.

## 自己跑一遍 / Try it yourself

```python
def add_noise(x, noise, sigma):
    alpha = 1 / (sigma * sigma + 1) ** 0.5
    sigma_t = sigma / (sigma * sigma + 1) ** 0.5
    return alpha * x + sigma_t * noise

for sigma in [0.1, 1.0, 3.0]:
    print(sigma, round(add_noise(10.0, -2.0, sigma), 3))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
0.1 9.751
1.0 5.657
3.0 1.265
```

sigma 越大，结果越接近噪声；这就是 scheduler 索引选错会很致命的原因。

As sigma grows, the result moves closer to the noise; that is why choosing the wrong scheduler index is so damaging.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Diffusers schedulers** / **Diffusers schedulers**: `add_noise` 通常也要按 timesteps 查 sigma/alpha 表。 / `add_noise` typically indexes sigma/alpha tables by timesteps.
- **Video inpainting pipelines** / **Video inpainting pipelines**: 中途加噪必须和当前 denoise step 对齐。 / Mid-schedule noising must align with the current denoising step.

## 注意事项 / Caveats / when it breaks

- **scheduler 状态必须先设置** / **Scheduler state must be initialized**: `begin_index`、`step_index` 来自 pipeline 调用顺序。 / `begin_index` and `step_index` depend on pipeline call order.
- **广播维度别偷懒** / **Do not shortcut broadcasting**: 视频 latent 通常是 `[B,C,T,H,W]`，sigma 必须扩到可广播形状。 / Video latents are often `[B,C,T,H,W]`, so sigma must be expanded to a broadcastable shape.

## 延伸阅读 / Further reading

- [Wan2.1 Flow solver](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/utils/fm_solvers.py)
- Diffusers scheduler design notes
