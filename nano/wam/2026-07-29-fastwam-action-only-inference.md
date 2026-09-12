---
date: 2026-07-29
topic: wam
source: wam
repo: huggingface/lerobot
file: src/lerobot/policies/fastwam/wan/modular.py
permalink: https://github.com/huggingface/lerobot/blob/f37be3edbee60f3a09a5183788b91eb19f0c07d1/src/lerobot/policies/fastwam/wan/modular.py#L1786-L1871
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, sampler-inference, kv-cache]
build_role: sampler-inference advanced variant
---

# FastWAM infer_action：缓存视频，只采样动作 / FastWAM infer_action: Cache Video, Sample Only Actions

> **一句话 / In one line**: FastWAM 先把首帧视频 token 预填到 MoT cache，再只对 action latent 跑去噪循环。 / FastWAM pre-fills the first-frame video tokens into the MoT cache, then runs the denoising loop only over action latents.

## 为什么重要 / Why this matters

WAM 不一定每次都要生成完整视频。有时机器人只需要“看一帧，预测未来动作”。这段代码展示了高效路径：视频作为条件缓存住，动作序列作为待采样 latent 逐步更新。

A WAM does not always need to generate a full video. Sometimes the robot only needs to "see one frame, predict future actions." This code shows the efficient path: cache video as conditioning and iteratively update the action sequence latent.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/fastwam/wan/modular.py`](https://github.com/huggingface/lerobot/blob/f37be3edbee60f3a09a5183788b91eb19f0c07d1/src/lerobot/policies/fastwam/wan/modular.py#L1786-L1871)

```python
@torch.no_grad()
def infer_action(
    self,
    prompt: str | None,
    input_image: torch.Tensor,
    action_horizon: int,
    proprio: torch.Tensor | None = None,
    context: torch.Tensor | None = None,
    context_mask: torch.Tensor | None = None,
    negative_prompt: str | None = None,
    text_cfg_scale: float = 1.0,
    num_inference_steps: int = 20,
    sigma_shift: float | None = None,
    seed: int | None = None,
    rand_device: str = "cpu",
    tiled: bool = False,
) -> dict[str, Any]:
    self.eval()
    if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
        raise ValueError("`infer_action` requires `video_attention_mask_mode='first_frame_causal'.")

    input_image, _, _ = self._normalize_infer_input_image(input_image)
    proprio = self._normalize_infer_proprio(proprio)
    latents_action = self._make_action_latents(action_horizon, seed, rand_device)

    input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
    first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
    fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

    context, context_mask = self._prepare_infer_context(prompt, context, context_mask, proprio)

    timestep_video = torch.zeros(
        (first_frame_latents.shape[0],),
        dtype=first_frame_latents.dtype,
        device=self.device,
    )
    video_pre = self.video_expert.pre_dit(
        x=first_frame_latents,
        timestep=timestep_video,
        context=context,
        context_mask=context_mask,
        action=None,
        fuse_vae_embedding_in_latents=fuse_flag,
    )
    video_seq_len = int(video_pre["tokens"].shape[1])
    attention_mask = self._build_mot_attention_mask(
        video_seq_len=video_seq_len,
        action_seq_len=latents_action.shape[1],
        video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
        device=video_pre["tokens"].device,
    )
    video_kv_cache = self.mot.prefill_video_cache(
        video_tokens=video_pre["tokens"],
        video_freqs=video_pre["freqs"],
        video_t_mod=video_pre["t_mod"],
        video_context_payload={
            "context": video_pre["context"],
            "mask": video_pre["context_mask"],
        },
        video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
    )

    infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
        num_inference_steps=num_inference_steps,
        device=self.device,
        dtype=latents_action.dtype,
        shift_override=sigma_shift,
    )
    for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action, strict=True):
        timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
        pred_action = self._predict_action_noise_with_cache(
            latents_action=latents_action,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

    return {"action": latents_action[0].detach().to(device="cpu", dtype=torch.float32)}
```

## 逐行讲解 / What's happening

1. **第 1803-1809 行 / Lines 1803-1809 (setup)**:
   - 中文: 进入 eval/no-grad，检查 attention mask 模式，并初始化 action latent。
   - English: The method enters eval/no-grad mode, checks the attention-mask mode, and initializes action latents.
2. **第 1811-1837 行 / Lines 1811-1837 (video prefill)**:
   - 中文: 首帧被 VAE 编码并送进 video expert，随后构造 MoT 的视频+动作 attention mask。
   - English: The first frame is VAE-encoded and passed through the video expert, then a video+action MoT attention mask is built.
3. **第 1837-1845 行 / Lines 1837-1845 (`prefill_video_cache`)**:
   - 中文: 视频 token 的 KV cache 只算一次，后面的 action denoise 复用它。
   - English: Video-token KV cache is computed once and reused by the following action denoising loop.
4. **第 1848-1871 行 / Lines 1848-1871 (scheduler loop)**:
   - 中文: action scheduler 给出每步 timestep/delta，模型预测 action noise，scheduler 再更新 latent。
   - English: The action scheduler supplies timesteps/deltas, the model predicts action noise, and the scheduler updates the latent.

## 类比 / The analogy

这像厨师先把汤底熬好放在锅里，后面每一轮只调整面条的火候；汤底不需要每次重做。

It is like a cook preparing the broth once, then only adjusting the noodles each round; the broth does not need to be rebuilt every time.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-{VLA,WAM}

在 nanoWAM 里，这属于 `sampler-inference`。上游是 VAE/视觉 tokenizer、文本/状态条件和动作 latent 初始化；下游是 robot policy 的 action queue。省掉 video cache 也能跑，但每个 denoise step 都重复处理首帧条件，推理成本会高很多。

In a nanoWAM, this is `sampler-inference`. Upstream are the VAE/visual tokenizer, text/state conditioning, and action-latent initialization; downstream is the robot policy's action queue. You can omit the video cache, but then every denoising step repeats first-frame conditioning and inference becomes much more expensive.

## 自己跑一遍 / Try it yourself

```python
def scheduler_steps(n):
    return [(1 - i / n, 1 / n) for i in range(n)]

def predict_noise(action, cached_video):
    return [0.5 * x + 0.1 * cached_video for x in action]

action = [1.0, -0.5, 0.25]
cached_video = 2.0
for t, delta in scheduler_steps(4):
    noise = predict_noise(action, cached_video)
    action = [round(x - delta * n, 4) for x, n in zip(action, noise)]
print(action)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[0.4207, -0.4586, -0.0189]
```

这个 toy loop 把真实代码里的三件事简化出来：固定条件、预测噪声、用 scheduler delta 更新动作 latent。

This toy loop isolates the three real operations: fixed conditioning, noise prediction, and scheduler-delta updates to the action latent.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LeRobot action chunking** / **LeRobot action chunking**: 一次预测未来多步，然后逐步消费。 / Predict a multi-step future once, then consume it step by step.
- **LLM prefix cache** / **LLM prefix cache**: prompt prefix 的 KV cache 固定，decode token 逐步追加。 / Prompt-prefix KV cache stays fixed while decode tokens are appended.

## 注意事项 / Caveats / when it breaks

- **mask 模式是硬前提** / **Mask mode is a hard precondition**: 代码要求 `first_frame_causal`，否则 action-only cache 语义不成立。 / The code requires `first_frame_causal`; otherwise action-only cache semantics do not hold.
- **视频和动作 schedule 要协调** / **Video and action schedules must align**: 训练时如果视频/动作噪声定义不同，推理 scheduler 也要匹配。 / If training uses different video/action noise definitions, inference schedulers must match them.

## 延伸阅读 / Further reading

- [FastWAM in LeRobot](https://github.com/huggingface/lerobot)
- [Source permalink](https://github.com/huggingface/lerobot/blob/f37be3edbee60f3a09a5183788b91eb19f0c07d1/src/lerobot/policies/fastwam/wan/modular.py#L1786-L1871)
