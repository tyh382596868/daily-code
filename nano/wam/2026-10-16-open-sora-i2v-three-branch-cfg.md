---
date: 2026-10-16
topic: wam
source: wam
repo: hpcaitech/Open-Sora
file: opensora/utils/sampling.py
permalink: https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/utils/sampling.py#L158-L245
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, classifier-free-guidance, i2v]
build_role: classifier-free guidance, advanced image-to-video variant
---

# Open-Sora 三分支 CFG：文字和图像条件分别拉动视频 / Open-Sora Three-Branch CFG: Let Text and Image Conditions Pull Video Separately

> **一句话 / In one line**: I2V 去噪把条件、无文字和无条件预测分成三份，再分别施加文本与图像 guidance。 / I2V denoising creates conditional, no-text, and unconditional predictions, then applies text and image guidance independently.

## 为什么重要 / Why this matters

普通 CFG 只有“有提示词/无提示词”两条路；I2V 还要区分保留参考帧和遵循文字。三路预测把这两个旋钮拆开，允许视频同时忠于首帧又听从文本。

Ordinary CFG has only prompt and no-prompt branches. I2V must also distinguish preserving a reference frame from following text. Three predictions separate those knobs so a video can honor both.

## 代码 / The code

`hpcaitech/Open-Sora` — [`sampling.py`](https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/utils/sampling.py#L158-L245)

```python
kwargs["cond"] = torch.cat([cond, cond, torch.zeros_like(cond)], dim=0)
img = torch.cat([cond_x, cond_x, cond_x], dim=0)
pred = model(img=img, **kwargs, timesteps=t_vec, guidance=guidance_vec)
cond, uncond, uncond_2 = pred.chunk(3, dim=0)
pred = uncond_2 + image_gs * (uncond - uncond_2) + text_gs * (cond - uncond)
img = img + (t_prev - t_curr) * pred
```

## 逐行讲解 / What's happening

1. 中文: batch 被复制三次：有文字+有图像、无文字+有图像、无文字+无图像。
   English: The batch is tripled: text+image, no-text+image, and no-text+no-image.
2. 中文: `uncond - uncond_2` 是图像条件增量，`cond - uncond` 是文本条件增量。
   English: `uncond - uncond_2` is the image-condition delta, while `cond - uncond` is the text-condition delta.
3. 中文: 两个 scale 分开加权，随后用 flow update 走到下一噪声时间。
   English: The two scales are independently weighted before a flow update moves to the next noise time.

## 类比 / The analogy

像导演同时对摄影和剧本下指令：先确保场景不跑偏，再决定故事该往哪里发展。

It is a director giving separate instructions to cinematography and script: preserve the scene first, then choose where the story goes.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

中文: 这是 `classifier-free-guidance` 的 I2V 进阶版本，位于 DiT 预测 velocity 之后、scheduler 更新 latent 之前。输入是噪声视频、文本/参考帧条件和两个 guidance scale；输出是合成后的 velocity。没有它时模型仍能生成，但不能在同一采样中独立调节文字与视觉条件。生产版要避免三倍 batch 峰值显存，并校验条件复制的排序。

English: This advanced `classifier-free-guidance` variant sits after DiT velocity prediction and before the scheduler updates video latents. It consumes noisy video, text/reference conditions, and two scales, emitting a composed velocity. Without it the model can still generate, but cannot independently tune text and visual conditioning. Production implementations must control the threefold peak batch memory and verify branch ordering.

## 自己跑一遍 / Try it yourself

```python
uncond2, uncond, cond = 0.0, 2.0, 5.0
image_scale, text_scale = 0.5, 1.5
pred = uncond2 + image_scale * (uncond - uncond2) + text_scale * (cond - uncond)
print(pred)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
5.5
```

中文: 两个差分项各自有旋钮。  
English: Each difference term has its own knob.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Wan2.1 CFG loop** / **Wan2.1 CFG loop**: 中文: 使用正负提示词差分控制文本。 / English: It uses positive/negative prompt differences for text control.
- **Open-Sora distilled guidance embedding** / **Open-Sora distilled guidance embedding**: 中文: 把 guidance 强度输入模型。 / English: It feeds guidance strength into the model.

## 注意事项 / Caveats / when it breaks

- 中文: 三个 branch 的排列一旦错位，结果会稳定地错。 / English: A branch-order mismatch produces consistently wrong results.
- 中文: 更高 scale 不总是更好，可能造成闪烁或过饱和。 / English: Higher scales are not always better and can cause flicker or oversaturation.

## 延伸阅读 / Further reading

- [Open-Sora I2V denoiser](https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/utils/sampling.py#L158-L245)
