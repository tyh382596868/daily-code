---
date: 2026-10-13
topic: diffusion
source: tracked
repo: Wan-Video/Wan2.1
file: wan/image2video.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/image2video.py#L184-L330
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, image-to-video, classifier-free-guidance]
---

# Wan2.1 I2V loop：首帧变条件，噪声一路被 CFG 拉回视频 / Wan2.1 I2V Loop: Turn the First Frame into Condition, Then Pull Noise into Video with CFG

> **一句话 / In one line**: Wan2.1 先把首帧编码成 mask+latent 条件，再在每步用正负 prompt 做 CFG 更新。 / Wan2.1 first encodes the first frame as mask+latent condition, then applies CFG from positive and negative prompts at every step.

## 为什么重要 / Why this matters

图生视频不是把图片贴到第 0 帧就结束。采样器还要知道 latent 网格、首帧 mask、文本条件、CLIP 视觉条件，以及每一步如何把噪声推回视频。

Image-to-video is more than pasting an image into frame zero. The sampler needs the latent grid, first-frame mask, text condition, CLIP visual condition, and a denoising update for every step.

## 代码 / The code

`Wan-Video/Wan2.1` - [`wan/image2video.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/image2video.py#L184-L330)

```python
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]

        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16, (F - 1) // 4 + 1,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ],
                           dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # preprocess
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        self.clip.model.to(self.device)
        clip_context = self.clip.visual([img[:, None, :, :]])
        if offload_model:
            self.clip.model.cpu()

        y = self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                        0, 1),
                torch.zeros(3, F - 1, h, w)
            ],
                         dim=1).to(self.device)
        ])[0]
        y = torch.concat([msk, y])
            # sample videos
            latent = noise

            arg_c = {
                'context': [context[0]],
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': [y],
            }

            arg_null = {
                'context': context_null,
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': [y],
            }

            if offload_model:
                torch.cuda.empty_cache()

            self.model.to(self.device)
            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = [latent.to(self.device)]
                timestep = [t]

                timestep = torch.stack(timestep).to(self.device)

                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0].to(
                        torch.device('cpu') if offload_model else self.device)
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0].to(
                        torch.device('cpu') if offload_model else self.device)
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                latent = latent.to(
                    torch.device('cpu') if offload_model else self.device)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)
```

## 逐行讲解 / What's happening

1. **第 184-197 行 / Lines 184-197 (latent canvas)**:
   - 中文: 从宽高比推出 latent 网格，并把序列长度补齐到并行粒度。
   - English: It derives the latent grid from aspect ratio and rounds sequence length for parallelism.
2. **第 210-249 行 / Lines 210-249 (first-frame condition)**:
   - 中文: mask 标出首帧，VAE 把首帧和空白未来帧编码成条件 latent。
   - English: The mask marks the first frame, and the VAE encodes that frame plus blank future frames as condition latents.
3. **第 284-319 行 / Lines 284-319 (CFG pair)**:
   - 中文: 正负 prompt 共享图像条件，只替换文本 context，然后按 CFG 公式合成噪声。
   - English: Positive and negative prompts share image condition, swap text context, and combine noise with the CFG formula.
4. **第 324-330 行 / Lines 324-330 (scheduler step)**:
   - 中文: scheduler 接收 guided noise、timestep 和当前 latent，产出下一版 latent。
   - English: The scheduler consumes guided noise, timestep, and current latent to produce the next latent.

## 类比 / The analogy

这像拿一张照片续画漫画分镜：第一格必须保留，剧情提示告诉你往哪里画，负提示提醒你别跑偏。

It is like extending a photo into a comic strip: the first panel stays anchored, the prompt gives the plot, and the negative prompt says what to avoid.

## 自己跑一遍 / Try it yourself

```python
latent = 0.0
guide_scale = 3.0
for step, cond, uncond in [(0.5, 2.0, 0.5), (0.25, 1.0, 0.25)]:
    guided = uncond + guide_scale * (cond - uncond)
    latent -= step * guided
    print(round(guided, 3), round(latent, 3))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
5.0 -2.5
2.5 -3.125
```

中文: CFG 放大的是条件和无条件预测的差。

English: CFG amplifies the difference between conditional and unconditional predictions.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Diffusers pipelines** / **Diffusers pipelines**: Diffusers pipelines。 / Diffusers pipelines.
- **Open-Sora inference** / **Open-Sora inference**: Open-Sora inference。 / Open-Sora inference.

## 注意事项 / Caveats / when it breaks

- **显存压力** / **Two CFG forwards are expensive under limited VRAM**: 显存压力。 / Two CFG forwards are expensive under limited VRAM.
- **尺寸取整** / **Latent stride and patch rounding can change exact output size**: 尺寸取整。 / Latent stride and patch rounding can change exact output size.

## 延伸阅读 / Further reading
- [Wan2.1 image2video.py](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/image2video.py#L184-L330)
