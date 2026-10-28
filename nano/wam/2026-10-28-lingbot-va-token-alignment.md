---
date: 2026-10-28
topic: wam
source: wam
repo: Robbyant/lingbot-va
file: wan_va/modules/model.py
permalink: https://github.com/Robbyant/lingbot-va/blob/main/wan_va/modules/model.py#L3203-L3274
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, wam, attention]
build_role: dit-block
---

# LingBot-VA token 对齐：把混合流补到 attention 友好长度 / LingBot-VA Token Alignment: Pad Mixed Streams for Attention

> **一句话 / In one line**: LingBot-VA 拼接视频与动作 token 后，把总长度补到 128 的倍数，再一并补齐位置与时间条件。 / After joining video and action tokens, LingBot-VA pads the total length to a multiple of 128 and pads position/time conditions in lockstep.

## 为什么重要 / Why this matters

高效 attention kernel 常要求特定块尺寸；只补 hidden states 而漏掉 RoPE 或 timestep 条件，会造成 token 与条件错位。

Efficient attention kernels often prefer block sizes. Padding only hidden states while forgetting RoPE or timesteps misaligns conditions and tokens.

## 代码 / The code

`Robbyant/lingbot-va` — [`model.py`](https://github.com/Robbyant/lingbot-va/blob/main/wan_va/modules/model.py#L3240-L3274)

```python
total_length = hidden_states.shape[1]
padded_length = (128 - total_length % 128) % 128
hidden_states = F.pad(hidden_states, (0, 0, 0, padded_length))
rotary_emb = F.pad(rotary_emb, (0, 0, 0, 0, 0, padded_length))
temb = F.pad(temb, (0, 0, 0, padded_length))
timestep_proj = F.pad(timestep_proj, (0, 0, 0, 0, 0, padded_length))
```

## 逐行讲解 / What's happening

公式产生 `0..127` 个补齐 token。四个序列用同一 `padded_length` 补齐，确保每个 token 仍对应自己的 RoPE 和时间调制。

The formula yields 0–127 padding tokens. All four sequences use the same amount, preserving each token’s RoPE and timestep modulation.

## 在 nanoWAM 中的位置 / Where this lives in your nanoWAM

它位于 DiT block 前的打包层：视频 latent、条件帧与动作 token 合流后，先做 kernel 对齐和 mask，再送入 attention。

It is the packing layer before DiT blocks: merge video latents, condition frames, and action tokens; align for the kernel and mask; then attend.

## 自己跑一遍 / Try it yourself

```python
length = 259
pad = (128 - length % 128) % 128
print(pad, length + pad)  # 125 384
```

运行 / Run with `python try.py`。输出 / Expected output: `125 384`。

## 注意事项 / Caveats / when it breaks

- **attention mask / Mask**：补齐 token 必须不可见，否则零向量仍可能污染注意力归一化。 / Padded tokens must be masked out.
