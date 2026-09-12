---
date: 2026-08-29
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/text2video.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/main/wan/text2video.py#L208-L230
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, sampler-inference, classifier-free-guidance]
build_role: sampler-inference advanced variant
---

# Wan2.1 CFG loop：正负提示词各走一遍 / Wan2.1 CFG Loop: Run Positive and Negative Prompts Together

> **一句话 / In one line**: Wan2.1 的采样循环每个 timestep 同时跑条件和无条件预测，再用 guidance scale 拉开两者。 / Wan2.1's sampling loop runs conditional and unconditional predictions at each timestep, then separates them with a guidance scale.

## 为什么重要 / Why this matters

World Action Model 生成未来时，常常既要听动作或文本条件，又不能完全失去生成模型的先验。CFG 把“按条件走”和“按无条件先验走”配成一对，用两者差值控制生成方向。

When a World Action Model imagines the future, it often needs to follow actions or text while keeping the generative prior intact. CFG pairs a conditional prediction with an unconditional one and uses their difference to steer generation.

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/text2video.py`](https://github.com/Wan-Video/Wan2.1/blob/main/wan/text2video.py#L208-L230)

```python
for _, t in enumerate(tqdm(timesteps)):
    latent_model_input = [latents[0]]
    timestep = [t]

    timestep = torch.stack(timestep)

    noise_pred_cond = self.model(
        latent_model_input, t=timestep, context=context, seq_len=max_seq_len
    )[0]
    noise_pred_uncond = self.model(
        latent_model_input, t=timestep, context=context_null, seq_len=max_seq_len
    )[0]

    noise_pred = noise_pred_uncond + guide_scale * (
        noise_pred_cond - noise_pred_uncond
    )

    temp_x0 = sample_scheduler.step(
        noise_pred.unsqueeze(0),
        t,
        latents[0].unsqueeze(0),
        return_dict=False,
        generator=seed_g,
    )[0]
    latents = [temp_x0.squeeze(0)]
```

## 逐行讲解 / What's happening

1. **timestep 包装 / Timestep wrapping**:
   - 中文: 当前 `t` 被放进 tensor，和 latent 一起喂给 denoiser。
   - English: The current `t` is wrapped as a tensor and passed to the denoiser with the latent.
2. **条件预测 / Conditional prediction**:
   - 中文: `context` 来自正向 prompt，告诉模型这一步应该朝目标语义去噪。
   - English: `context` comes from the positive prompt and steers denoising toward the target semantics.
3. **无条件预测 / Unconditional prediction**:
   - 中文: `context_null` 来自空/负提示词，提供模型自己的基础生成方向。
   - English: `context_null` comes from the empty or negative prompt and provides the model's baseline direction.
4. **CFG 合成 / CFG merge**:
   - 中文: `uncond + scale * (cond - uncond)` 把条件差值放大，再交给 scheduler 更新 latent。
   - English: `uncond + scale * (cond - uncond)` amplifies the conditional delta before the scheduler updates the latent.

## 类比 / The analogy

这像摄影师拿两张参考图调色：一张是“自然默认效果”，一张是“客户想要的效果”。guidance scale 决定最终照片朝客户版本推多远。

It is like color grading with two references: one is the natural default look, the other is the client's target. The guidance scale decides how far the final image moves toward the client's version.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

在 nanoWAM 中，这属于 `sampler-inference` 的高级变体。上游是文本、动作或状态条件编码器，下游是 scheduler/decoder。没有 CFG，你可以生成未来，但很难稳定控制“遵循条件”和“保持自然动态”之间的平衡。

In a nanoWAM, this is an advanced variant of `sampler-inference`. Upstream are text, action, or state condition encoders; downstream are the scheduler and decoder. Without CFG, you can still generate futures, but it is harder to balance condition following against natural dynamics.

## 自己跑一遍 / Try it yourself

```python
latent = 1.0
guide_scale = 3.0

for t in [3, 2, 1]:
    cond = latent - 0.2 * t
    uncond = latent - 0.05 * t
    guided = round(uncond + guide_scale * (cond - uncond), 3)
    latent = round(latent - 0.1 * guided, 3)
    print(t, guided, latent)
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
3 -0.5 1.05
2 0.05 1.045
1 0.545 0.99
```

同一个 latent 每步都由条件和无条件预测共同决定，而不是只看 prompt 分支。

The same latent is updated from both conditional and unconditional predictions at every step, not only the prompt-conditioned branch.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Stable Diffusion CFG** / **Stable Diffusion CFG**: 图像扩散也用正负 prompt 差值引导噪声预测。 / Image diffusion also steers noise prediction with positive-minus-negative prompt deltas.
- **Open-Sora reference conditioning** / **Open-Sora reference conditioning**: 条件帧约束采样方向。 / Reference frames constrain the sampling trajectory.
- **action-conditioned rollout** / **action-conditioned rollout**: 动作条件可以像 prompt 一样提供条件分支。 / Action conditions can serve the same role as the prompt-conditioned branch.

## 注意事项 / Caveats / when it breaks

- **scale 太大会过饱和** / **Too much scale overshoots**: 生成可能变得僵硬或出现伪影。 / Generation can become rigid or artifact-heavy.
- **cond/uncond 必须同形状** / **Cond and uncond must match shape**: 否则差值没有明确语义。 / The delta has no clean meaning if shapes differ.
- **负提示词不是免费午餐** / **Negative prompts are not free**: 它们改变的是 baseline 分支，可能也会压掉有用内容。 / They change the baseline branch and may suppress useful content too.

## 延伸阅读 / Further reading

- Wan2.1 text-to-video source: https://github.com/Wan-Video/Wan2.1/blob/main/wan/text2video.py
- Wan2.1 repository: https://github.com/Wan-Video/Wan2.1
