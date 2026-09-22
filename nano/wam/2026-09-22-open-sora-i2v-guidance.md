---
date: 2026-09-22
topic: wam
source: wam
repo: hpcaitech/Open-Sora
file: opensora/utils/sampling.py
permalink: https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/utils/sampling.py#L120-L245
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, classifier-free-guidance, image-to-video, sampling]
build_role: Classifier-free guidance and multimodal conditioning
---

# Open-Sora I2V guidance：文字和图像条件分两条方向推 / Open-Sora I2V Guidance: Push Text and Image Conditions Along Separate Directions

> **一句话 / In one line**: Open-Sora 用三路预测同时得到条件、文字无条件和图像无条件结果，再分别用两个 guidance scale 组合它们。 / Open-Sora runs three predictions for conditional, text-unconditional, and image-unconditional states, then combines them with two separate guidance scales.

## 为什么重要 / Why this matters

普通 CFG 常写成 `uncond + scale * (cond - uncond)`，但 image-to-video 同时有文字和参考图两种条件。Open-Sora 把它拆成两条向量：先从 image-unconditional 走向 image-conditioned，再从 image-conditioned 走向 text-conditioned。这样文字强度和图像保持强度可以独立调节。

Classic CFG is often written as `uncond + scale * (cond - uncond)`, but image-to-video has both text and reference-image conditions. Open-Sora separates the two directions: move from image-unconditional to image-conditioned, then from image-conditioned to text-conditioned. Text strength and image-preservation strength become independent knobs.

实现上它没有把三个分支拆成三次 model call，而是把输入沿 batch 维复制三份。这样共享一次 kernel launch 形状，也让后面的 `pred.chunk(3)` 变得非常清楚。更特别的是，image guidance 可以沿视频时间轴变化，避免首帧和后续帧被同样强度地钉死。

Implementation-wise, it does not issue three separate model calls. It repeats the inputs along the batch dimension, keeping one kernel shape and making `pred.chunk(3)` explicit. Image guidance can also vary along the temporal axis, so the first frame and later frames do not have to be constrained equally.

## 代码 / The code

`hpcaitech/Open-Sora` — [`opensora/utils/sampling.py`](https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/utils/sampling.py#L120-L245)

```python
def get_oscillation_gs(guidance_scale: float, i: int, force_num=10):
    """Get oscillation guidance for CFG."""
    if i < force_num or (i >= force_num and i % 2 == 0):
        gs = guidance_scale
    else:
        gs = 1.0
    return gs


class I2VDenoiser(Denoiser):
    def denoise(self, model: MMDiTModel, **kwargs) -> Tensor:
        img = kwargs.pop("img")
        timesteps = kwargs.pop("timesteps")
        guidance = kwargs.pop("guidance")
        guidance_img = kwargs.pop("guidance_img")
        masks = kwargs.pop("masks")
        masked_ref = kwargs.pop("masked_ref")
        kwargs.pop("sigma_min")
        text_osci = kwargs.pop("text_osci", False)
        image_osci = kwargs.pop("image_osci", False)
        scale_temporal_osci = kwargs.pop("scale_temporal_osci", False)
        patch_size = kwargs.pop("patch_size", 2)

        guidance_vec = torch.full(
            (img.shape[0],), guidance, device=img.device, dtype=img.dtype
        )
        for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            t_vec = torch.full(
                (img.shape[0],), t_curr, dtype=img.dtype, device=img.device
            )
            b, c, t, w, h = masked_ref.size()
            cond = torch.cat((masks, masked_ref), dim=1)
            cond = pack(cond, patch_size=patch_size)
            kwargs["cond"] = torch.cat([cond, cond, torch.zeros_like(cond)], dim=0)
            cond_x = img[: len(img) // 3]
            img = torch.cat([cond_x, cond_x, cond_x], dim=0)
            pred = model(img=img, **kwargs, timesteps=t_vec, guidance=guidance_vec)

            text_gs = get_oscillation_gs(guidance, i) if text_osci else guidance
            image_gs = get_oscillation_gs(guidance_img, i) if image_osci else guidance_img
            cond, uncond, uncond_2 = pred.chunk(3, dim=0)
            if image_gs > 1.0 and scale_temporal_osci:
                step_upper_image_gs = torch.linspace(image_gs, 1.0, len(timesteps))[i]
                image_gs = torch.linspace(1.0, step_upper_image_gs, t)[
                    None, None, :, None, None
                ].repeat(b, c, 1, h, w)
                image_gs = pack(image_gs, patch_size=patch_size).to(cond.device, cond.dtype)

            pred = uncond_2 + image_gs * (uncond - uncond_2) + text_gs * (cond - uncond)
            pred = torch.cat([pred, pred, pred], dim=0)
            img = img + (t_prev - t_curr) * pred

        img = img[: len(img) // 3]
        return img

    def prepare_guidance(
        self,
        text: list[str],
        optional_models: dict[str, nn.Module],
        device: torch.device,
        dtype: torch.dtype,
        **kwargs,
    ) -> tuple[list[str], dict[str, Tensor]]:
        ret = {}
        neg = kwargs.get("neg", None)
        ret["guidance_img"] = kwargs.pop("guidance_img")
        if neg is None:
            neg = [""] * len(text)
        text = text + neg + neg
        return text, ret
```

## 逐行讲解 / What's happening

1. **第 120-133 行 / Lines 120-133 (oscillation)**:
   - 中文: 前若干步保持原 guidance；之后隔步把 scale 降到 1，给采样轨迹留一点回弹空间。
   - English: The original scale is kept for the early steps, then every other step falls back to 1.0, giving the trajectory room to recover.
2. **第 178-200 行 / Lines 178-200 (three-way batch)**:
   - 中文: `cond`、`uncond`、`uncond_2` 的条件沿 batch 维排列，图像 latent 也复制三份，因此一次 forward 得到三种预测。
   - English: The conditional, text-unconditional, and image-unconditional inputs are arranged along the batch dimension, so one forward produces all three predictions.
3. **第 203-219 行 / Lines 203-219 (two guidance directions)**:
   - 中文: `uncond - uncond_2` 是图像条件方向，`cond - uncond` 是文字条件方向；两个 scale 可以独立变化。
   - English: `uncond - uncond_2` is the image-conditioning direction, while `cond - uncond` is the text-conditioning direction. The scales can vary independently.
4. **第 209-216 行 / Lines 209-216 (temporal scale)**:
   - 中文: image guidance 可以从视频前部到后部逐渐变化，适合首帧强约束、后续帧更自由的 I2V。
   - English: Image guidance can change from the beginning to the end of the video, which fits I2V settings where the first frame is tightly anchored and later frames are freer.
5. **第 228-245 行 / Lines 228-245 (negative prompts)**:
   - 中文: 如果没有显式 negative prompt，就用空字符串复制两份，把文本 batch 对齐到三路分支。
   - English: When no negative prompt is supplied, empty strings are duplicated twice so the text batch lines up with the three branches.

## 类比 / The analogy

像调一支合奏乐队：`uncond_2` 是没有参考图的底鼓，`uncond` 加入参考图的旋律，`cond` 再加入文字指挥。两个音量旋钮分别控制“像不像这张图”和“听不听文字”。

Think of a three-part band: `uncond_2` is the beat without the reference image, `uncond` adds the image melody, and `cond` adds the text conductor. Two volume knobs control image faithfulness and textual obedience separately.

## 在 nanoWAM 中的位置 / Where this lives in your nanoWAM

这是 `classifier-free-guidance` 组件，位于文本/图像条件编码和 DiT forward 之后、latent sampler update 之内。输入是正负文本、参考帧 mask、latent 和 guidance scales；输出是组合后的 velocity/noise prediction，交给 Euler 或 flow-matching step。

This is the `classifier-free-guidance` component. It sits after text/image conditioning and the DiT forward, inside the latent sampling loop. Inputs are positive/negative text, reference-frame masks, latents, and guidance scales; its output is the combined velocity/noise prediction passed to the Euler or flow-matching step.

从零实现时，先让 denoiser 接受一个 `(3B, ...)` batch，再按 `chunk(3)` 合成预测即可。省掉 CFG 后，模型仍能生成，但 prompt adherence 和 reference-frame fidelity 会失去一个可调的推力。生产版还要加入负 prompt 缓存、不同条件的 mask 对齐、显存预算、动态 guidance 和采样器稳定性检查。

For a from-scratch implementation, start with a denoiser that accepts a `(3B, ...)` batch and combine the three predictions with `chunk(3)`. Without CFG the model can still generate, but prompt adherence and reference fidelity lose a controllable steering force. Production adds negative-prompt caching, condition-mask alignment, memory budgeting, dynamic guidance, and sampler stability checks.

## 自己跑一遍 / Try it yourself

```python
import torch

cond = torch.tensor([[3.0, 2.0]])
uncond = torch.tensor([[1.0, 1.0]])
uncond_img = torch.tensor([[0.0, 0.5]])
image_gs, text_gs = 2.0, 1.5
guided = uncond_img + image_gs * (uncond - uncond_img) + text_gs * (cond - uncond)
sample = torch.zeros_like(guided)
sample = sample + 0.1 * guided
round_rows = lambda x: [[round(v, 3) for v in row] for row in x.tolist()]
print(round_rows(guided), round_rows(sample))
```

运行 / Run with:

```bash
pip install torch
python try.py
```

预期输出 / Expected output:

```text
[[5.0, 3.0]] [[0.5, 0.3]]
```

中文: 两个方向的向量可以单独放大；这就是多条件 CFG 比一个总 scale 更有表现力的原因。 / English: The two conditioning directions can be amplified independently, which is why multimodal CFG is more expressive than one global scale.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **DiT label dropout** / **DiT label dropout**: 中文: 训练时制造无条件分支，采样时再把条件方向放大。 / English: Training-time label dropout creates an unconditional branch that can be amplified during sampling.
- **Wan2.1 CFG loop** / **Wan2.1 CFG loop**: 中文: 文本到视频通常只需要正负文本两路；I2V 多出来的第三路专门保存图像条件。 / English: Text-to-video commonly uses positive and negative text branches; I2V adds a third branch to preserve image conditioning.
- **FastWAM action sampling** / **FastWAM action sampling**: 中文: action-only denoising 也能复用“条件预填充、逐步修正”的采样结构。 / English: Action-only denoising reuses the same “prefill conditions, then refine step by step” sampling shape.

## 注意事项 / Caveats / when it breaks

- **三路输入必须严格对齐** / **All three branches must align**: 中文: batch、mask、timestep 和 text token 的顺序错一位，CFG 会悄悄把错误条件相减。 / English: A one-position mismatch among batches, masks, timesteps, or text tokens silently subtracts the wrong conditions.
- **高 guidance 不是免费质量** / **High guidance is not free quality**: 中文: scale 太大可能带来过饱和、运动僵硬或参考图粘连。 / English: Excessive scales can cause oversaturation, rigid motion, or over-attachment to the reference image.
- **时间轴 guidance 要看任务** / **Temporal guidance is task-dependent**: 中文: 首帧强、后帧弱适合 I2V，但 loop 或 video-to-video 可能需要另一条 schedule。 / English: Strong-first-frame, weak-later guidance fits I2V, but loop and video-to-video tasks may need a different schedule.

## 延伸阅读 / Further reading

- [Open-Sora I2V denoiser](https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/utils/sampling.py#L120-L245)
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
- [Open-Sora repository](https://github.com/hpcaitech/Open-Sora)
