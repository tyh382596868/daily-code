---
date: 2026-08-06
topic: diffusion
source: trending
repo: amazon-science/Spherical_Diffusion_Policy
file: sdp/policy/diffusion_equi_unet_cnn_enc_policy_se2.py
permalink: https://github.com/amazon-science/Spherical_Diffusion_Policy/blob/ea25aa5a0f16c1499448a05cd751bccec4b3e916/sdp/policy/diffusion_equi_unet_cnn_enc_policy_se2.py#L131-L168
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, conditional-sampling]
---

# Spherical Diffusion Policy：每步都把条件钉回去 / Spherical Diffusion Policy: Pin the Conditions Back at Every Step

> **一句话 / In one line**: 条件扩散采样不是只在开头写入条件，而是在每个 denoise step 前都用 mask 把已知值钉回轨迹。 / Conditional diffusion sampling does not write conditions only once; before every denoise step it pins known values back into the trajectory with a mask.

## 为什么重要 / Why this matters

机器人动作扩散经常只生成未知未来动作，同时固定某些已知观测、历史动作或约束槽位。这个采样循环的关键是 `trajectory[condition_mask] = condition_data[condition_mask]`：不管 scheduler 上一步怎么改，下一步问模型前都先恢复硬条件。

Robot action diffusion often generates only unknown future actions while fixing known observations, past actions, or constrained slots. The key line is `trajectory[condition_mask] = condition_data[condition_mask]`: regardless of what the scheduler changed in the previous step, hard conditions are restored before querying the model again.

## 代码 / The code

`amazon-science/Spherical_Diffusion_Policy` — [`sdp/policy/diffusion_equi_unet_cnn_enc_policy_se2.py`](https://github.com/amazon-science/Spherical_Diffusion_Policy/blob/ea25aa5a0f16c1499448a05cd751bccec4b3e916/sdp/policy/diffusion_equi_unet_cnn_enc_policy_se2.py#L131-L168)

```python
def conditional_sample(self,
        condition_data, condition_mask,
        local_cond=None, global_cond=None,
        generator=None,
        # keyword arguments to scheduler.step
        **kwargs
        ):
    model = self.diff
    scheduler = self.noise_scheduler

    trajectory = torch.randn(
        size=condition_data.shape,
        dtype=condition_data.dtype,
        device=condition_data.device,
        generator=generator)

    # set step values
    scheduler.set_timesteps(self.num_inference_steps)

    for t in scheduler.timesteps:
        # 1. apply conditioning
        trajectory[condition_mask] = condition_data[condition_mask]

        # 2. predict model output
        model_output = model(trajectory, t,
            local_cond=local_cond, global_cond=global_cond)

        # 3. compute previous image: x_t -> x_t-1
        trajectory = scheduler.step(
            model_output, t, trajectory,
            generator=generator,
            **kwargs
            ).prev_sample

    # finally make sure conditioning is enforced
    trajectory[condition_mask] = condition_data[condition_mask]

    return trajectory
```

## 逐行讲解 / What's happening

1. **第 141-145 行 / Lines 141-145 (random start)**:
   - 中文: 整条轨迹先从标准噪声开始，形状完全跟 `condition_data` 对齐。
   - English: The whole trajectory starts as random noise with the same shape as `condition_data`.
2. **第 147-148 行 / Lines 147-148 (scheduler setup)**:
   - 中文: 采样前先让 scheduler 生成离散时间表，后面循环只按这张表走。
   - English: Before sampling, the scheduler builds the discrete timestep table that the loop will follow.
3. **第 150-156 行 / Lines 150-156 (condition then predict)**:
   - 中文: 每一步先用 mask 覆盖已知槽位，再把完整轨迹交给 denoiser。
   - English: Each step overwrites known slots via the mask before sending the full trajectory to the denoiser.
4. **第 159-166 行 / Lines 159-166 (step and final clamp)**:
   - 中文: scheduler 更新可能会改到条件位置，所以循环结束后还要最后钉回一次。
   - English: The scheduler update may modify conditioned positions, so the loop pins them back one final time.

## 类比 / The analogy

像在填数独：空格可以反复擦写，但题目里给定的数字每一轮检查都必须重新确认，最后提交前也要再看一遍。

It is like solving Sudoku: empty cells can change repeatedly, but the given numbers must be rechecked every round and once more before submission.

## 自己跑一遍 / Try it yourself

```python
traj = [0.0, 0.0, 0.0, 0.0]
cond = [9.0, 0.0, 0.0, -3.0]
mask = [True, False, False, True]

for step in range(3):
    traj = [c if m else x for x, c, m in zip(traj, cond, mask)]
    traj = [x * 0.5 + step for x in traj]
traj = [c if m else x for x, c, m in zip(traj, cond, mask)]
print(traj)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[9.0, 2.5, 2.5, -3.0]
```

即使中间的 step 会动到所有位置，最后被 mask 标记的位置仍然保持硬条件。

Even if intermediate steps update every position, the masked positions still end as hard constraints.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Diffusion Policy inpainting** / **Diffusion Policy inpainting**: 用 mask 固定历史动作，只生成未来动作。 / Masks fix past actions while future actions are generated.
- **Video inpainting samplers** / **Video inpainting samplers**: 每步把已知帧或 mask 区域写回 latent。 / Known frames or masked regions are written back into latents at each step.

## 注意事项 / Caveats / when it breaks

- **硬条件可能产生边界不连续** / **Hard constraints can create seams**: 条件槽位和自由槽位之间需要模型学会平滑过渡。 / The model must learn smooth transitions between fixed and free slots.
- **mask shape 必须精确** / **Mask shape must be exact**: mask 和 trajectory 广播错了，会固定错误维度。 / A wrongly broadcast mask pins the wrong dimensions.

## 延伸阅读 / Further reading

- [Spherical Diffusion Policy repository](https://github.com/amazon-science/Spherical_Diffusion_Policy)
- Spherical Diffusion Policy: A SE(3) Equivariant Visuomotor Policy with Spherical Fourier Representation
