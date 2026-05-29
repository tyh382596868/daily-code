---
date: 2026-05-29
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/text2video.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/text2video.py#L196-L262
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, wam, sampler, inference, scheduler-step]
build_role: Inference sampler — the denoising loop that turns noise + prompt into a clip
---

# 60 行推理循环就是 WAM 的"生成"过程 / 60 lines of denoise loop is the entire WAM "generate"

> **一句话 / In one line**: Wan2.1 的视频生成推理就是"build scheduler → noise = randn → for t in timesteps: model(x_t) → CFG combine → scheduler.step → decode latent",任何 DiT-based WAM 部署到生产时都是这套骨架。 / Wan2.1's video generation is "build scheduler → noise = randn → for t in timesteps: model(x_t) → CFG combine → scheduler.step → VAE-decode" — that 60-line skeleton is what every DiT-based WAM serves in production.

## 为什么重要 / Why this matters

训练靠 `compute_loss` 反向传梯度,**推理**完全是另一回事:你只有一个 randn 噪声 + prompt,要走 25-50 步 scheduler.step 才能从噪声爬到一段视频。这一步里所有东西必须正确组合:scheduler 类型(UniPC、DPM++、Euler)、shift 参数、timesteps 序列、CFG 两 forward、`torch.no_grad()`、autocast、最终 VAE.decode 解回 pixel。任何一个环节出错都不是"loss 偏高",而是直接输出花屏。Wan2.1 这 60 行把这套套路压缩到极致,几乎是 nanoWAM 的 `sample.py` 草稿。

Training propagates gradients with `compute_loss`; **inference** has nothing to do with that. You start from `randn` noise + prompt and walk 25-50 scheduler steps to climb from noise to a clip. Every choice along the way must line up: scheduler family (UniPC, DPM++, Euler), shift parameter, the timesteps schedule, two-forward CFG, `torch.no_grad`, autocast, and the final `vae.decode` back to pixels. Any single mistake here doesn't show as elevated loss — it shows as broken output. Wan2.1's 60-line `generate` is the most compact correct template; treat it as a working draft of nanoWAM's `sample.py`.

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/text2video.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/text2video.py#L196-L262)

```python
@contextmanager
def noop_no_sync():
    yield

no_sync = getattr(self.model, 'no_sync', noop_no_sync)

with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

    # === 1) Build the sampler-side scheduler ===
    if sample_solver == 'unipc':
        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps,
            shift=1, use_dynamic_shifting=False)
        sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
        timesteps = sample_scheduler.timesteps
    elif sample_solver == 'dpm++':
        sample_scheduler = FlowDPMSolverMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps,
            shift=1, use_dynamic_shifting=False)
        sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
        timesteps, _ = retrieve_timesteps(sample_scheduler,
                                          device=self.device,
                                          sigmas=sampling_sigmas)
    else:
        raise NotImplementedError("Unsupported solver.")

    # === 2) Start from pure noise ===
    latents = noise

    arg_c    = {'context': context,      'seq_len': seq_len}
    arg_null = {'context': context_null, 'seq_len': seq_len}

    # === 3) The denoise loop ===
    for _, t in enumerate(tqdm(timesteps)):
        latent_model_input = latents
        timestep = torch.stack([t])

        self.model.to(self.device)
        noise_pred_cond   = self.model(latent_model_input, t=timestep, **arg_c)[0]
        noise_pred_uncond = self.model(latent_model_input, t=timestep, **arg_null)[0]

        # CFG combine (see today's CFG note)
        noise_pred = noise_pred_uncond + guide_scale * (
            noise_pred_cond - noise_pred_uncond)

        # === 4) One scheduler step: x_t → x_{t-1} ===
        temp_x0 = sample_scheduler.step(
            noise_pred.unsqueeze(0), t, latents[0].unsqueeze(0),
            return_dict=False, generator=seed_g)[0]
        latents = [temp_x0.squeeze(0)]

    # === 5) Decode latents back to pixels via the 3D VAE ===
    x0 = latents
    if offload_model:
        self.model.cpu()
        torch.cuda.empty_cache()
    if self.rank == 0:
        videos = self.vae.decode(x0)
```

## 逐行讲解 / What's happening

1. **autocast + no_grad + no_sync 三重保护 / Three guards**:
   - 中文:`amp.autocast(dtype=self.param_dtype)` 强制走 bf16/fp16 算子,推理省一半显存;`torch.no_grad()` 禁掉 autograd,显存再省一截;`no_sync()` 是 FSDP/DDP 下"不要在每次 forward 都同步参数"的 context(单机时是 noop)。三者叠加是大模型推理的标配。
   - English: `autocast(dtype=self.param_dtype)` forces every op into bf16/fp16, halving activation memory; `torch.no_grad()` disables autograd and saves a chunk more; `no_sync()` tells FSDP/DDP not to sync params on every forward (noop on single-GPU). All three are baseline for inference of a large model.

2. **Scheduler 的两个分支:UniPC 和 DPM++ / Scheduler choice**:
   - 中文:UniPC 是 1-step multistep,质量稳定;DPM++ 是 2-3 step multistep,少几步就能出好图。`shift` 是 flow matching scheduler 特有的"时间步密度倾斜"参数,大 shift 让 timestep 集中在 t 接近 0 的清晰端(对高分辨率有利)。`set_timesteps(sampling_steps, ...)` 是核心 —— 训练时有 1000 个 timestep,推理只取 25-50 个。
   - English: UniPC is a stable 1-step multistep solver; DPM++ is a faster 2-3 step variant. `shift` is a flow-matching-specific knob that biases timesteps toward the clean end (helps high-resolution). The crucial call is `set_timesteps(sampling_steps)`: training has 1000 timesteps, inference samples only 25-50 of them.

3. **`latents = noise` 起点 / Start from pure noise**:
   - 中文:推理输入就是 `randn(B, C_lat, T_lat, H_lat, W_lat)`,跟训练时 `add_noise(latent, noise, t=1)` 的结果分布一致。
   - English: inference starts from `randn(B, C_lat, T_lat, H_lat, W_lat)` — the same distribution as training's `add_noise` at `t=1`.

4. **CFG 在循环里两次 forward / Two CFG forwards inside the loop**:
   - 中文:每一步都跑两次模型,这是 CFG 的"代价"(详见今天的 CFG 笔记)。蒸馏方法(Causal-Forcing、DMD)就是为了让一次 forward 就能输出 guided noise,把这段开销砍一半。
   - English: every step runs the model twice — the price of CFG (see today's CFG note). Distillation methods like Causal-Forcing/DMD exist to fold guided output into a single forward, halving this cost.

5. **`scheduler.step` 是真正的"前进一步" / `scheduler.step` is the actual integrator**:
   - 中文:`step(noise_pred, t, x_t)` 在 flow-matching 里就是欧拉法 `x_{t-1} = x_t - dt * v`(其中 `v = noise_pred`),UniPC/DPM++ 内部用高阶组合多步历史。注意 Wan 这里只传当前 `latents[0]`,scheduler 内部维护多步历史。
   - English: `step(noise_pred, t, x_t)` for flow matching is Euler `x_{t-1} = x_t - dt * v` where `v = noise_pred`. UniPC/DPM++ internally combine the current and previous predictions for higher-order accuracy. Wan only hands in the current latent; scheduler stores its own history.

6. **VAE.decode 把 latent 翻回 pixel / Decode latents to pixels**:
   - 中文:整段循环都在 latent 空间(VAE 编码的低维空间),最后一步 `self.vae.decode(x0)` 把 `[B, C_lat, T_lat, H_lat, W_lat]` 翻回 `[B, 3, T, H, W]` 的视频。这一步用前面 CausalConv3d 笔记里讲的 3D VAE decoder。
   - English: the entire loop operates in latent space (VAE-encoded low-dim); the final `vae.decode(x0)` lifts `[B, C_lat, T_lat, H_lat, W_lat]` back to a pixel video `[B, 3, T, H, W]` using the 3D VAE decoder from today's CausalConv3d note.

7. **`offload_model` 提示流式部署 / `offload_model` hints at streaming deployment**:
   - 中文:推理结束后把 model 搬回 CPU 是 hint —— 大 WAM 模型 + VAE 在一张 GPU 上可能放不下,生产部署常常是"推理时 model 上 GPU、decode 时把 model 卸到 CPU,VAE 上 GPU"。
   - English: pushing the model to CPU after the loop hints at the deployment pattern — large WAM + VAE often don't co-fit on one GPU. Production typically swaps: model on GPU during the loop, then offload to CPU and bring the VAE up to decode.

## 类比 / The analogy

像考古修复一幅模糊壁画:开始是一片随机噪点(整面墙糊掉);你按照"先修轮廓再修细节"的顺序(timesteps 从 1.0 到 0),每一笔用两份参考(prompt 描述 + 通用美术原理),把当前模糊的墙稍微擦干净一些。25-50 步之后壁画就清晰了 —— 这是 latent;最后请印刷厂(VAE decoder)把它放大成真实尺寸的画。

Picture restoring a blurred mural. You start with random noise (the whole wall is washed out). The schedule says "fix big shapes first, then details" (timesteps from 1.0 down to 0). Each stroke uses two references — the specific prompt and generic art-history priors (CFG cond and uncond) — to scrub the wall slightly less blurry. After 25-50 strokes the latent mural is clear; finally the print shop (VAE decoder) blows it up to true size.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-WAM

中文:这是 nanoWAM 的 `nano/wam/sample.py` 完整骨架。上游全部用得着:VAE(CausalConv3d + Resample)、DiT(3D RoPE + text cross-attn + action conditioning + adaLN)、scheduler(yesterday 的 dreamzero FlowMatchScheduler 或自实现 UniPC/DPM++)、CFG(今天的笔记)。下游是文件保存(`save_video(videos)`)或者真实机器人(把 action latent 解码成关节指令)。如果省掉这段:你训练好了一切但没法用 —— **训练 + 推理是两套循环**,training_step 永远生不出新视频。生产实现还要补:(1) **batch 推理** —— 把 B 多个 prompt 一起跑,但 CFG 那两 forward 要 batched;(2) **inference 时间 KV cache** —— 自回归推长视频时,过去帧的 K/V 缓存可以跨步骤复用;(3) **多 GPU pipeline** —— DiT forward 太大要切到多卡。

English: this is the full skeleton of nanoWAM's `nano/wam/sample.py`. Every upstream piece is required: VAE (CausalConv3d + Resample), DiT (3-D RoPE + text cross-attn + action conditioning + adaLN), scheduler (yesterday's dreamzero `FlowMatchScheduler` or your own UniPC/DPM++), and CFG (today's note). Downstream is saving the video (`save_video(videos)`) or, for action latents, decoding into joint commands. Skip this loop and your trained model produces nothing — training and inference are two different control flows. Production additions: (1) **batched inference** with CFG forwards batched together; (2) **KV cache across denoise steps** when extending to long autoregressive videos; (3) **pipeline parallelism** across GPUs because the DiT forward is too big for one card.

## 自己跑一遍 / Try it yourself

```python
# pip install torch
# 1-D toy WAM "sampler" — start from noise, denoise to a Gaussian-shaped target.
import torch

target_mu = 5.0                                # the "prompt" target
def model_score(x_t, cond=True):               # toy "model"
    mu = target_mu if cond else 0.0            # uncond returns score of N(0,1)
    return -(x_t - mu)                         # flow-matching velocity toward mu

steps  = 20
sigma  = torch.linspace(1.0, 0.02, steps)
x_t    = torch.randn(8) * sigma[0]             # start from noise
guide  = 5.0                                   # CFG scale

for i in range(steps - 1):
    v_cond   = model_score(x_t, cond=True)
    v_uncond = model_score(x_t, cond=False)
    v        = v_uncond + guide * (v_cond - v_uncond)   # CFG combine
    dt       = sigma[i + 1] - sigma[i]                   # negative (decreasing sigma)
    x_t      = x_t + dt * v                              # Euler step

print(f"final samples (target = {target_mu}): {x_t.tolist()}")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
final samples (target = 5.0): [4.8..., 4.9..., 5.1..., 4.7..., 5.0..., 4.9..., 5.1..., 4.8...]
```

中文:20 步从随机噪点(均值 0,方差 1)推到接近 5.0 的目标 —— 这就是 sampler 的最小工作单元。生产代码用 UniPC/DPM++ 替换 Euler、用真正的 model 替换玩具 score,但骨架完全一样。

English: 20 steps walk samples from noise centred at 0 to a distribution around 5.0 — that's the minimal sampler. Production swaps Euler for UniPC/DPM++ and the toy `model_score` for a real model, but the skeleton is unchanged.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Stable Diffusion txt2img** / **Stable Diffusion txt2img**: 中文 — 完全相同的 `for t in timesteps: ... CFG ... scheduler.step()`,只是 VAE 是 2D 的。 / English — identical `for t in timesteps: ... CFG ... scheduler.step()`, just with a 2-D VAE.
- **Open-Sora sample.py** / **Open-Sora sample.py**: 中文 — 几乎一字不差,UniPC + flow matching + CFG。 / English — almost line-for-line the same: UniPC + flow matching + CFG.
- **Causal-Forcing 的 inference_with_trajectory** / **Causal-Forcing's `inference_with_trajectory`**: 中文 — 把这个循环改成"逐 chunk + KV cache"形式,变成 self-forcing 的因果采样。 / English — restructures this loop into chunked, KV-cache-aware self-forcing inference.
- **Yesterday's WAM scheduler note** / **Yesterday's WAM scheduler note**: 中文 — 那个 FlowMatchScheduler 的 `step()` 就是这里 `scheduler.step()` 的内部实现。两条笔记上下衔接。 / English — yesterday's `FlowMatchScheduler.step()` is the internal implementation of the `scheduler.step()` call here. Pair the two notes together.

## 注意事项 / Caveats / when it breaks

- **shift 选错就崩** / **Wrong `shift` ruins quality**: 中文 — 视频分辨率高时 shift 要调大(3-7),低分辨率小一点。`shift=1` 是"无 shift",适合训练 timestep 分布,但推理常常要重调。 / English — high-resolution video benefits from larger `shift` (3-7); low-res needs less. `shift=1` matches the training distribution but inference often re-tunes.
- **`set_timesteps` 必须在 forward 之前** / **`set_timesteps` must run before the loop**: 中文 — scheduler 是有状态的,timesteps、sigmas 在 `set_timesteps` 之后才填充。漏调或者循环里调都会出错。 / English — schedulers are stateful; `timesteps` and `sigmas` are only populated after `set_timesteps`. Calling it inside the loop (or skipping it) breaks the math.
- **autocast 必开** / **Autocast is not optional**: 中文 — 大 DiT 的 forward 在 fp32 下显存会爆;但 VAE.decode 如果用 fp16 / bf16 可能有 NaN,生产时 VAE 单独保持 fp32 是常见做法。 / English — DiT forward in fp32 OOMs immediately, but VAE.decode in bf16/fp16 can NaN. Production frequently keeps the VAE in fp32 even when DiT runs bf16.
- **VAE.decode 之前要 model.cpu()** / **Offload the model before VAE.decode if memory is tight**: 中文 — 7B DiT + VAE 同时驻留 GPU 经常 OOM。先卸载,再 decode。 / English — a 7 B DiT + 3-D VAE often won't co-fit. Offload the DiT to CPU, then call decode.

## 延伸阅读 / Further reading

- [Wan2.1 generate.py](https://github.com/Wan-Video/Wan2.1/blob/main/generate.py)
- [FlowMatchEulerDiscreteScheduler (diffusers)](https://huggingface.co/docs/diffusers/api/schedulers/flow_match_euler_discrete)
- [DPM-Solver++ (Lu et al., 2022)](https://arxiv.org/abs/2211.01095)
- [UniPC sampling](https://arxiv.org/abs/2302.04867)
