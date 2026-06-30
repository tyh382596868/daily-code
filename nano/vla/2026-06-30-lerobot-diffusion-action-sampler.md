---
date: 2026-06-30
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/policies/diffusion/modeling_diffusion.py
permalink: https://github.com/huggingface/lerobot/blob/2f2b5679510a35aa83fdd8e9f986e134666618bc/src/lerobot/policies/diffusion/modeling_diffusion.py#L235-L333
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, diffusion-policy, action-head]
build_role: action-head-continuous
---

# LeRobot Diffusion Policy：从噪声动作轨迹反推可执行 chunk / LeRobot Diffusion Policy: Denoise a Noisy Action Trajectory into an Executable Chunk

> **一句话 / In one line**: `conditional_sample` 生成整段 horizon，再由 `generate_actions` 切出当前要执行的 `n_action_steps`。 / `conditional_sample` generates the whole horizon, then `generate_actions` slices out the `n_action_steps` to execute now.

## 为什么重要 / Why this matters

VLA 不一定要把动作当 token 输出。Diffusion Policy 把动作序列当连续信号，从高斯噪声开始，用观测特征作为条件一步步去噪。这适合机器人控制，因为动作天然是连续向量，而且一次预测多个未来步能减少抖动。

A VLA does not have to emit action tokens. Diffusion Policy treats the action sequence as a continuous signal, starts from Gaussian noise, and denoises it under observation conditioning. That fits robot control because actions are continuous vectors, and predicting multiple future steps reduces jitter.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/diffusion/modeling_diffusion.py`](https://github.com/huggingface/lerobot/blob/2f2b5679510a35aa83fdd8e9f986e134666618bc/src/lerobot/policies/diffusion/modeling_diffusion.py#L235-L333)

```python
def conditional_sample(
    self,
    batch_size: int,
    global_cond: Tensor | None = None,
    generator: torch.Generator | None = None,
    noise: Tensor | None = None,
) -> Tensor:
    device = get_device_from_parameters(self)
    dtype = get_dtype_from_parameters(self)

    sample = (
        noise
        if noise is not None
        else torch.randn(
            size=(batch_size, self.config.horizon, self.config.action_feature.shape[0]),
            dtype=dtype,
            device=device,
            generator=generator,
        )
    )

    self.noise_scheduler.set_timesteps(self.num_inference_steps)

    for t in self.noise_scheduler.timesteps:
        model_output = self.unet(
            sample,
            torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
            global_cond=global_cond,
        )
        sample = self.noise_scheduler.step(model_output, t, sample, generator=generator).prev_sample

    return sample

def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
    batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
    global_cond_feats = [batch[OBS_STATE]]
    if self.config.image_features:
        if self.config.use_separate_rgb_encoder_per_camera:
            images_per_camera = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
            img_features_list = torch.cat(
                [
                    encoder(images)
                    for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)
                ]
            )
            img_features = einops.rearrange(
                img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
            )
        else:
            img_features = self.rgb_encoder(
                einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
            )
            img_features = einops.rearrange(
                img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
            )
        global_cond_feats.append(img_features)

    if self.config.env_state_feature:
        global_cond_feats.append(batch[OBS_ENV_STATE])
    return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

def generate_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
    batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
    assert n_obs_steps == self.config.n_obs_steps
    global_cond = self._prepare_global_conditioning(batch)
    actions = self.conditional_sample(batch_size, global_cond=global_cond, noise=noise)
    start = n_obs_steps - 1
    end = start + self.config.n_action_steps
    actions = actions[:, start:end]
    return actions
```

## 逐行讲解 / What's happening

1. **第 243-253 行 / Lines 243-253**:
   - 中文: 如果调用者没有给固定噪声，就采样一个 `(B, horizon, action_dim)` 的动作轨迹噪声。
   - English: If no fixed noise is provided, it samples a `(B, horizon, action_dim)` noisy action trajectory.
2. **第 257-265 行 / Lines 257-265**:
   - 中文: 每个 scheduler timestep 调一次 U-Net，预测当前噪声动作该怎么更新。
   - English: At each scheduler timestep, the U-Net predicts how to update the current noisy action.
3. **第 270-310 行 / Lines 270-310**:
   - 中文: 图像、机器人状态、环境状态被拼成一个全局条件向量。
   - English: Images, robot state, and environment state are concatenated into one global conditioning vector.
4. **第 328-331 行 / Lines 328-331**:
   - 中文: 模型生成完整 horizon，但只执行当前观测之后的一小段。
   - English: The model generates the full horizon, but only the short slice after the current observation is executed.

## 类比 / The analogy

这像导航软件先规划整条路线，但司机接下来只需要看下一个路口到下两个路口的指令。

It is like a navigation app planning the whole route while the driver only follows the next few turns.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

在 nanoVLA 里，这就是 `action-head-continuous` 的一种实现。上游是视觉/状态 encoder 输出的 `global_cond`，下游是机器人控制器实际发送的 action chunk。如果不用扩散动作头，可以换成 Gaussian head 或离散 action tokenizer，但连续高维动作通常会少一些量化损失。

In a nanoVLA, this is one implementation of `action-head-continuous`. Upstream encoders produce `global_cond`; downstream robot control consumes the action chunk. You can replace it with a Gaussian head or a discrete action tokenizer, but continuous high-dimensional actions often avoid quantization loss.

## 自己跑一遍 / Try it yourself

```python
import torch

B, horizon, action_dim, cond_dim = 2, 8, 3, 5
sample = torch.randn(B, horizon, action_dim)
cond = torch.randn(B, cond_dim)
for t in range(4, 0, -1):
    pred_noise = sample * 0.2 + cond.mean(dim=1).view(B, 1, 1) * 0.01
    sample = sample - pred_noise / t
actions = sample[:, 1:4]
print(actions.shape)
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```text
torch.Size([2, 3, 3])
```

核心现象是：采样器维护整段轨迹，但控制器只拿一段去执行。

The key point is that the sampler maintains a whole trajectory while the controller executes only a slice.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **OpenPI flow matching** / **OpenPI flow matching**: 也是连续动作头，但用 flow matching 时间轴替代 DDPM/DDIM scheduler。 / It is also a continuous action head, but uses a flow-matching time axis instead of a DDPM/DDIM scheduler.
- **GR00T diffusion head** / **GR00T diffusion head**: 同样把语言/视觉条件注入动作去噪网络。 / It similarly injects language/vision conditioning into an action denoising network.

## 注意事项 / Caveats / when it breaks

- **`horizon` 必须覆盖执行窗口** / **`horizon` must cover the execution window**: `n_action_steps <= horizon - n_obs_steps + 1`。 / The action slice must fit inside the generated horizon.
- **闭环延迟很关键** / **Closed-loop latency matters**: 去噪步数越多，动作质量可能更好，但控制频率会下降。 / More denoising steps may improve quality but reduce control frequency.

## 延伸阅读 / Further reading

- [LeRobot diffusion policy](https://github.com/huggingface/lerobot/blob/2f2b5679510a35aa83fdd8e9f986e134666618bc/src/lerobot/policies/diffusion/modeling_diffusion.py)

