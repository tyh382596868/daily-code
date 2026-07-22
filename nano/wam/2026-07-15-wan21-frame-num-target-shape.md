---
date: 2026-07-15
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/text2video.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/text2video.py#L154-L252
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, wam, sampler-inference, target-shape]
build_role: sampler-inference advanced variant
---

# Wan2.1 frame_num：先从帧数推 latent 形状 / Wan2.1 frame_num: Derive Latent Shape from Frame Count First

> **一句话 / In one line**: Wan2.1 的 T2V sampler 用 `frame_num` 推导 latent 时间长度和 `seq_len`，保证非默认帧数也能对齐 DiT 输入。 / Wan2.1's T2V sampler derives latent temporal length and `seq_len` from `frame_num`, keeping non-default frame counts aligned with the DiT input contract.

## 为什么重要 / Why this matters

视频扩散不是只把 `frame_num` 传给解码器就完事。帧数会影响 VAE latent 的时间维、patch token 数、sequence parallel 对齐长度、初始噪声张量和最终 decode。只要其中一个地方仍假设默认 81 帧，就会出现 tensor shape mismatch。这个片段把 `frame_num` 放在采样入口处，先算 `target_shape`，后面所有张量都跟着它走。

Video diffusion cannot just pass `frame_num` to the decoder. Frame count affects the VAE latent temporal dimension, patch-token count, sequence-parallel alignment, initial noise tensor, and final decode. If any one of those still assumes the default 81 frames, tensor shapes diverge. This sampler puts `frame_num` at the entrance, computes `target_shape`, and lets every later tensor follow it.

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/text2video.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/text2video.py#L154-L252)

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
        timestep = [t]
        timestep = torch.stack(timestep)

        self.model.to(self.device)
        noise_pred_cond = self.model(
            latent_model_input, t=timestep, **arg_c)[0]
        noise_pred_uncond = self.model(
            latent_model_input, t=timestep, **arg_null)[0]
```

## 逐行讲解 / What's happening

1. **`(F - 1) // stride + 1`**:
   - 中文: 视频 VAE 的时间下采样通常保留首帧对齐，所以 latent 帧数不是简单 `F // stride`。
   - English: Video VAE temporal downsampling often preserves first-frame alignment, so latent frames are not simply `F // stride`.
2. **`seq_len` 对齐到 `sp_size`**:
   - 中文: token 数要能被 sequence parallel 分组整除，`ceil(... / sp_size) * sp_size` 是对齐垫片。
   - English: Token count must align with sequence-parallel grouping; `ceil(... / sp_size) * sp_size` pads to that boundary.
3. **noise 用 `target_shape`**:
   - 中文: 初始噪声就是 latent 视频本体，它必须和模型预测、scheduler step、VAE decode 的形状一致。
   - English: Initial noise is the latent video itself, so its shape must match model prediction, scheduler steps, and VAE decode.
4. **cond/uncond 共用 `seq_len`**:
   - 中文: CFG 的正负 prompt 两次 forward 必须拥有相同 token shape，否则差分无法相减。
   - English: CFG's conditional and unconditional forwards must share token shape, or their predictions cannot be subtracted.

## 类比 / The analogy

这像定制相册：你先决定照片张数，才能知道需要几页、每页几个格子、最后一页要不要补空格。不能等装订完才告诉印刷厂“其实我要 97 张”。

It is like making a custom photo album. You choose the number of photos first, then derive pages, slots per page, and blank slots on the final page. You cannot tell the printer after binding that the album actually needs 97 photos.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

中文: 这是 `sampler-inference` 的高级变体，位于 prompt encode 之后、denoise loop 之前。nanoWAM 的 sampler 应该先有一个 `make_latent_shape(frame_num, size)`，所有 noise、position ids、attention mask、decode shape 都从这里派生。生产版还要处理多分辨率、多 batch 和分布式对齐。

English: This is an advanced `sampler-inference` component, after prompt encoding and before the denoise loop. A nanoWAM sampler should start with `make_latent_shape(frame_num, size)`, deriving noise, position ids, attention masks, and decode shape from it. A production version also handles multiple resolutions, batching, and distributed alignment.

## 自己跑一遍 / Try it yourself

```python
import math

def target(frame_num, size=(1280, 720), vae_stride=(4, 8, 8), patch=(1, 2, 2), sp=8, z=16):
    t = (frame_num - 1) // vae_stride[0] + 1
    h = size[1] // vae_stride[1]
    w = size[0] // vae_stride[2]
    seq = math.ceil((h * w) / (patch[1] * patch[2]) * t / sp) * sp
    return (z, t, h, w), seq

for f in [81, 97]:
    print(f, target(f))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
81 ((16, 21, 90, 160), 75600)
97 ((16, 25, 90, 160), 90000)
```

中文: 帧数从 81 到 97 时，latent 时间维和 `seq_len` 都变了；这两个值必须一起传播。

English: Moving from 81 to 97 frames changes both latent time and `seq_len`; both values must propagate together.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **DiT patchify** / **DiT patchify**: 图像尺寸先决定 patch grid，再决定位置编码长度。 / Image size determines the patch grid, then position-embedding length.
- **LLM paged attention** / **LLM paged attention**: request length 先决定 block 数，再决定 cache 分配。 / Request length determines block count before cache allocation.

## 注意事项 / Caveats / when it breaks

- **`frame_num` 通常应是 `4n+1` / `frame_num` usually should be `4n+1`**: VAE temporal stride 和训练分布都依赖这个约束。
- **`seq_len` 是 padded 长度 / `seq_len` is padded length**: 模型可能看到 padding token，attention/mask 必须同步处理。
- **多卡时更敏感 / More sensitive in distributed runs**: `sp_size` 对齐错会变成跨 rank shape mismatch。

## 延伸阅读 / Further reading

- [Wan2.1 `text2video.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/text2video.py)
- [Wan2.1 commit 841fe52](https://github.com/Wan-Video/Wan2.1/commit/841fe5237bbf8724e08870b1e83213e669cda0d1)
