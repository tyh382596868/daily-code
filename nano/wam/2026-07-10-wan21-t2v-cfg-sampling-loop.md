---
date: 2026-07-10
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/text2video.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/text2video.py#L158-L260
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, wam, sampler-inference, classifier-free-guidance]
build_role: sampler-inference advanced variant
---

# Wan2.1 T2V sampler：正负 prompt 各跑一次，再做 CFG / Wan2.1 T2V Sampler: Run Positive and Negative Prompts, Then Apply CFG

> **一句话 / In one line**: Wan2.1 每个采样步分别预测 conditional 和 unconditional noise，再用 `uncond + scale * (cond - uncond)` 推 latent 前进。 / Wan2.1 predicts conditional and unconditional noise at every sampling step, then advances the latent with `uncond + scale * (cond - uncond)`.

## 为什么重要 / Why this matters

视频扩散采样的核心不只是 scheduler。prompt adherence 来自 classifier-free guidance：同一个 latent、同一个 timestep，模型看一次正 prompt，再看一次负 prompt，两者差值就是“朝文本靠近”的方向。

The core of video diffusion sampling is not only the scheduler. Prompt adherence comes from classifier-free guidance: for the same latent and timestep, the model sees the positive prompt once and the negative prompt once, and their difference is the direction toward the text condition.

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/text2video.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/text2video.py#L158-L260)

```python
# preprocess
F = frame_num
target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                size[1] // self.vae_stride[1],
                size[0] // self.vae_stride[2])

seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                    (self.patch_size[1] * self.patch_size[2]) *
                    target_shape[1] / self.sp_size) * self.sp_size

if n_prompt == "":
    n_prompt = self.sample_neg_prompt
seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
seed_g = torch.Generator(device=self.device)
seed_g.manual_seed(seed)

context = self.text_encoder([input_prompt], self.device)
context_null = self.text_encoder([n_prompt], self.device)

noise = [
    torch.randn(
        target_shape[0],
        target_shape[1],
        target_shape[2],
        target_shape[3],
        dtype=torch.float32,
        device=self.device,
        generator=seed_g)
]

with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
    if sample_solver == 'unipc':
        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False)
        sample_scheduler.set_timesteps(
            sampling_steps, device=self.device, shift=shift)
        timesteps = sample_scheduler.timesteps
    elif sample_solver == 'dpm++':
        sample_scheduler = FlowDPMSolverMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False)
        sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
        timesteps, _ = retrieve_timesteps(
            sample_scheduler,
            device=self.device,
            sigmas=sampling_sigmas)
    else:
        raise NotImplementedError("Unsupported solver.")

    # sample videos
    latents = noise
    arg_c = {'context': context, 'seq_len': seq_len}
    arg_null = {'context': context_null, 'seq_len': seq_len}

    for _, t in enumerate(tqdm(timesteps)):
        latent_model_input = latents
        timestep = torch.stack([t])
        self.model.to(self.device)
        noise_pred_cond = self.model(
            latent_model_input, t=timestep, **arg_c)[0]
        noise_pred_uncond = self.model(
            latent_model_input, t=timestep, **arg_null)[0]

        noise_pred = noise_pred_uncond + guide_scale * (
            noise_pred_cond - noise_pred_uncond)

        temp_x0 = sample_scheduler.step(
            noise_pred.unsqueeze(0),
            t,
            latents[0].unsqueeze(0),
            return_dict=False,
            generator=seed_g)[0]
        latents = [temp_x0.squeeze(0)]
```

## 逐行讲解 / What's happening

1. **第 158-166 行 / Lines 158-166 (`target_shape`, `seq_len`)**:
   - 中文: 先把帧数和分辨率换成 VAE latent 的时间、高、宽，再算 DiT 需要的 token 长度。
   - English: Frame count and resolution are converted into VAE latent time, height, and width, then into the token length expected by the DiT.
2. **第 168-184 行 / Lines 168-184 (positive and negative context)**:
   - 中文: 正 prompt 和负 prompt 各自编码，后面每一步都成对使用。
   - English: Positive and negative prompts are encoded separately and used as a pair at every step.
3. **第 206-225 行 / Lines 206-225 (scheduler choice)**:
   - 中文: Wan2.1 支持 UniPC 和 DPM++ 两条采样器路径，但都产出同一类 timestep 序列。
   - English: Wan2.1 supports UniPC and DPM++ paths, but both produce the timestep sequence consumed by the loop.
4. **第 239-254 行 / Lines 239-254 (CFG and step)**:
   - 中文: 同一个 latent 跑两次模型，CFG 合成后的 `noise_pred` 才交给 scheduler 更新 latent。
   - English: The same latent runs through the model twice; the CFG-combined `noise_pred` is what the scheduler uses to update the latent.

## 类比 / The analogy

这像拍电影时同时听导演和审片人的意见：导演说“更靠近这个画面”，审片人说“别出现这些元素”。CFG 把两种意见相减后放大成下一步修改方向。

It is like editing a film while listening to a director and a reviewer. The director says what to move toward; the reviewer says what to avoid. CFG subtracts and scales those two signals into the next edit direction.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

这是 `sampler-inference` 的高级变体，并复用了 `classifier-free-guidance` 组件。上游要有 text encoder、negative prompt 和初始 latent noise；这里循环调用 DiT 并通过 scheduler 积分；下游 VAE decode 把最终 latent 还原成视频。

This is an advanced `sampler-inference` variant that reuses the `classifier-free-guidance` component. Upstream needs a text encoder, a negative prompt, and initial latent noise; this loop calls the DiT and integrates through the scheduler; downstream VAE decoding turns final latents into video.

## 自己跑一遍 / Try it yourself

```python
cond = [0.2, 0.8, -0.1]
uncond = [0.0, 0.5, 0.1]
scale = 5.0
guided = [u + scale * (c - u) for c, u in zip(cond, uncond)]
print([round(x, 2) for x in guided])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[1.0, 2.0, -0.9]
```

scale 越大，正负 prompt 的差值被放得越大，画面越贴 prompt 但也更容易过冲。

The larger the scale, the more the positive-minus-negative difference is amplified, improving prompt adherence but increasing overshoot risk.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Diffusers pipelines** / **Diffusers pipelines**: 经典 CFG 也是 cond/uncond 两次 forward 后线性组合。 / Classic CFG also linearly combines conditional and unconditional forward passes.
- **Open-Sora samplers** / **Open-Sora samplers**: 视频 DiT 推理同样围绕 latent loop、scheduler step 和 text guidance 组织。 / Video DiT inference is similarly organized around a latent loop, scheduler step, and text guidance.

## 注意事项 / Caveats / when it breaks

- **两次 forward 成本翻倍** / **Two forwards double cost**: CFG 质量更稳，但每步要跑 cond/uncond 两次模型。 / CFG is stable, but each step needs two model forwards.
- **负 prompt 是条件的一部分** / **The negative prompt is real conditioning**: 默认负 prompt 不合适时，会把模型推向奇怪的避让方向。 / A poor default negative prompt can push the model toward odd avoidance behavior.

## 延伸阅读 / Further reading

- [Wan2.1 `WanT2V.generate` sampling loop](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/text2video.py#L158-L260)
