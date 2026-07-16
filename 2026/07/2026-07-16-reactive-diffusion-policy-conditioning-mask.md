---
date: 2026-07-16
topic: robotics
source: trending
repo: xiaoxiaoxh/reactive_diffusion_policy
file: reactive_diffusion_policy/model/diffusion/mask_generator.py
permalink: https://github.com/xiaoxiaoxh/reactive_diffusion_policy/blob/824c5e8de1fd1811106907a04b5f0186e0138c0b/reactive_diffusion_policy/model/diffusion/mask_generator.py#L34-L91
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, diffusion-policy, mask, conditioning]
---

# Reactive Diffusion Policy mask：观测可见，动作待去噪 / Reactive Diffusion Policy Mask: Observations Visible, Actions Denoised

> **一句话 / In one line**: Reactive Diffusion Policy 用布尔 mask 标出哪些时间步和维度是条件，训练和采样时都把这些位置强制写回真值。 / Reactive Diffusion Policy uses a boolean mask to mark conditioned timesteps and dimensions, then forces those positions back to ground truth during training and sampling.

## 为什么重要 / Why this matters

扩散策略不是从空白生成整条轨迹。它通常知道前几步观测，只需要生成未来动作。这个 mask generator 把“哪些维度是观测、哪些维度是动作、前几步可见”变成一个 `[B, T, D]` 布尔张量。后续 loss 只监督未被条件锁住的位置。

A diffusion policy usually does not generate an entire trajectory from nothing. It knows the first observation steps and generates future actions. This mask generator turns "which dimensions are observations, which are actions, and how many steps are visible" into a `[B, T, D]` boolean tensor. The later loss supervises only positions not locked by conditioning.

## 代码 / The code

`xiaoxiaoxh/reactive_diffusion_policy` — [`mask_generator.py`](https://github.com/xiaoxiaoxh/reactive_diffusion_policy/blob/824c5e8de1fd1811106907a04b5f0186e0138c0b/reactive_diffusion_policy/model/diffusion/mask_generator.py#L34-L91), [`diffusion_unet_image_policy.py`](https://github.com/xiaoxiaoxh/reactive_diffusion_policy/blob/824c5e8de1fd1811106907a04b5f0186e0138c0b/reactive_diffusion_policy/policy/diffusion_unet_image_policy.py#L78-L116)

```python
class LowdimMaskGenerator(ModuleAttrMixin):
    def __init__(self, action_dim, obs_dim, max_n_obs_steps=2, fix_obs_steps=True, action_visible=False):
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.max_n_obs_steps = max_n_obs_steps
        self.fix_obs_steps = fix_obs_steps
        self.action_visible = action_visible

    @torch.no_grad()
    def forward(self, shape, seed=None):
        B, T, D = shape
        assert D == (self.action_dim + self.obs_dim)

        dim_mask = torch.zeros(size=shape, dtype=torch.bool, device=device)
        is_action_dim = dim_mask.clone()
        is_action_dim[..., :self.action_dim] = True
        is_obs_dim = ~is_action_dim

        if self.fix_obs_steps:
            obs_steps = torch.full((B,), fill_value=self.max_n_obs_steps, device=device)
        else:
            obs_steps = torch.randint(low=1, high=self.max_n_obs_steps + 1, size=(B,), generator=rng, device=device)

        steps = torch.arange(0, T, device=device).reshape(1, T).expand(B, T)
        obs_mask = (steps.T < obs_steps).T.reshape(B, T, 1).expand(B, T, D)
        obs_mask = obs_mask & is_obs_dim

        mask = obs_mask
        if self.action_visible:
            mask = mask | action_mask
        return mask
```

```python
trajectory[condition_mask] = condition_data[condition_mask]
model_output = model(trajectory, t, local_cond=local_cond, global_cond=global_cond)
trajectory = scheduler.step(model_output, t, trajectory, generator=generator, **kwargs).prev_sample
trajectory[condition_mask] = condition_data[condition_mask]
```

## 逐行讲解 / What's happening

1. **维度先切开 / Dimensions are split first**: 中文: 前 `action_dim` 是动作，其余是观测。 English: the first `action_dim` channels are action, the rest are observation.
2. **obs steps 可固定可随机 / Observation steps can be fixed or random**: 中文: 固定步数稳定训练，随机步数能增强不同历史长度。 English: fixed steps stabilize training; random steps augment history length.
3. **mask 是 `[B,T,D]` / The mask is `[B,T,D]`**: 中文: 它同时编码 batch、时间和维度。 English: it encodes batch, time, and dimension together.
4. **condition 强制写回 / Conditions are written back**: 中文: 采样每一步都把可见位置恢复成真值。 English: every sampling step restores visible positions to ground truth.
5. **loss 用反 mask / Loss uses the inverse mask**: 中文: 训练时只惩罚需要模型生成的位置。 English: training penalizes only positions the model must generate.

## 类比 / The analogy

这像填空题。题干里的观测已经给出，不能被模型改写；空格里的未来动作才需要生成。mask 就是那张“哪些是题干、哪些是空格”的透明胶片。

It is like a fill-in-the-blank test. Observations in the prompt are given and must not be rewritten; future actions are the blanks. The mask is the transparency showing which cells are prompt and which are blanks.

## 自己跑一遍 / Try it yourself

```python
import numpy as np

B, T, action_dim, obs_dim = 1, 4, 2, 3
D = action_dim + obs_dim
steps = np.arange(T)[None, :]
obs_steps = np.array([2])
is_action = np.zeros((B, T, D), dtype=bool)
is_action[..., :action_dim] = True
obs_mask = (steps < obs_steps[:, None])[:, :, None] & ~is_action
loss_mask = ~obs_mask
print(obs_mask.astype(int)[0])
print(loss_mask.astype(int)[0])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[[0 0 1 1 1]
 [0 0 1 1 1]
 [0 0 0 0 0]
 [0 0 0 0 0]]
[[1 1 0 0 0]
 [1 1 0 0 0]
 [1 1 1 1 1]
 [1 1 1 1 1]]
```

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Diffusion Policy inpainting** / **Diffusion Policy inpainting**: 已知观测作为条件，未知动作作为生成目标。 / known observations condition the model while unknown actions are generated.
- **image inpainting diffusion** / **image inpainting diffusion**: 每步采样都把 mask 内像素写回原图。 / every denoising step restores masked-known pixels.

## 注意事项 / Caveats / when it breaks

- **维度顺序必须一致 / Dimension order must match**: 如果 trajectory 不是 `[action, obs]`，mask 会锁错维度。 / If trajectory is not `[action, obs]`, the mask locks the wrong dimensions.
- **`action_visible=True` 会泄漏历史动作 / `action_visible=True` reveals past actions**: 适合某些设置，但要确认训练目标。 / Useful in some settings, but changes the target.
- **condition 每步都要写回 / Conditions must be restored every step**: 只在开始写一次会被 scheduler 更新冲掉。 / Writing only once lets scheduler updates drift conditioned values.

## 延伸阅读 / Further reading

- [Reactive Diffusion Policy mask generator](https://github.com/xiaoxiaoxh/reactive_diffusion_policy/blob/824c5e8de1fd1811106907a04b5f0186e0138c0b/reactive_diffusion_policy/model/diffusion/mask_generator.py)
- [Reactive Diffusion Policy sampling loop](https://github.com/xiaoxiaoxh/reactive_diffusion_policy/blob/824c5e8de1fd1811106907a04b5f0186e0138c0b/reactive_diffusion_policy/policy/diffusion_unet_image_policy.py)
