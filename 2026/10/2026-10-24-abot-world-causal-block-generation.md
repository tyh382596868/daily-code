---
date: 2026-10-24
topic: robotics
source: trending
repo: amap-cvlab/ABot-World
file: pipeline/causal_inference.py
permalink: https://github.com/amap-cvlab/ABot-World/blob/5c3f7260f2346ebb60a49efb81a7db5bfe15d7e5/pipeline/causal_inference.py#L342-L460
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, robotics, world-model, causal-generation]
---

# ABot-World 因果块生成：首帧永远是干净条件 / ABot-World Causal Blocks: Keep the First Frame as Clean Context

> **一句话 / In one line**: 每个视频块在去噪时重写首帧，并在结束后用干净 context 更新 KV cache。 / Each video block restores its first frame during denoising and refreshes the KV cache with clean context afterwards.

## 为什么重要 / Why this matters

流式世界模型要连续生成未来，但不能让最初的观测逐步被噪声污染。ABot-World 把首帧视为固定锚点，并将当前块的位置写进 transformer 调用。

A streaming world model must generate the future without gradually corrupting its original observation. ABot-World treats the first frame as an anchor and passes the current block position into the transformer.

## 代码 / The code

`amap-cvlab/ABot-World` — [`causal_inference.py`](https://github.com/amap-cvlab/ABot-World/blob/5c3f7260f2346ebb60a49efb81a7db5bfe15d7e5/pipeline/causal_inference.py#L361-L460)

```python
replace_first = (self.current_start_frame - self.num_input_frames == 0) and (first_frame_latents is not None)
if replace_first:
    noisy_input[:, 0:1] = first_frame_latents

for index, current_timestep in enumerate(self.denoising_step_list):
    timestep = torch.ones([B, F], device=noise_block.device, dtype=torch.int64) * current_timestep
    if replace_first: timestep[:, 0] = 0
    _, denoised_pred = self.generator(
        noisy_image_or_video=noisy_input, conditional_dict=self.conditional_dict,
        timestep=timestep, kv_cache=self.kv_cache1, crossattn_cache=self.crossattn_cache,
        current_start=self.current_start_frame * self.frame_seq_length,
        replace_first_timestep_and_noise_latents=replace_first)
    noisy_input = denoised_pred
    if replace_first: noisy_input[:, 0:1] = first_frame_latents

self.current_start_frame += F
```

## 逐行讲解 / What's happening

仅在第一个生成块时，代码把 frame 0 设成干净 latent、其时间步设为 0，并在每轮去噪后再次写回。`current_start_frame * frame_seq_length` 让位置编码知道这是连续流中的哪一段，最后推进全局帧游标。

Only for the first generated block, the code makes frame zero a clean latent, gives it timestep zero, and restores it after every denoising iteration. `current_start_frame * frame_seq_length` tells positional encoding where this block belongs in the stream, then the global frame cursor advances.

## 类比 / The analogy

像画长卷时把第一张参考照片钉在画板上：每次加新笔触都能覆盖新区域，却不能把参考照片涂花。

It is pinning the first reference photo to a long canvas: new strokes can cover new regions, but never paint over the reference.

## 自己跑一遍 / Try it yourself

```python
import torch
x = torch.randn(1, 3, 2); first = torch.ones(1, 1, 2)
for _ in range(3):
    x = x * .5
    x[:, :1] = first
print(x[:, 0])
```

运行 / Run with: `pip install torch && python try.py`

预期输出 / Expected output: `tensor([[1., 1.]])`。首帧在每一步后仍然不变。

Expected output: `tensor([[1., 1.]])`. The first frame remains unchanged after every step.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **image-to-video diffusion** / **image-to-video diffusion**: 首帧 latent 与 mask 常被重复注入以保持条件。 / First-frame latents and masks are often reinjected to preserve conditioning.

## 注意事项 / Caveats / when it breaks

- **缓存必须对应干净上下文** / **Caches must match clean context**: 若用含噪首帧建立 KV cache，后续块会继承错误条件。 / A KV cache built from a noisy first frame passes the wrong condition downstream.

## 延伸阅读 / Further reading

- [ABot-World causal inference](https://github.com/amap-cvlab/ABot-World/blob/5c3f7260f2346ebb60a49efb81a7db5bfe15d7e5/pipeline/causal_inference.py#L342-L460)
