---
date: 2026-08-11
topic: wam
source: wam
repo: hpcaitech/Open-Sora
file: opensora/utils/inference.py
permalink: https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/utils/inference.py#L283-L351
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, wam, visual-conditioning]
build_role: action-conditioning advanced variant, reference-frame mask construction for image/video-conditioned generation
---

# Open-Sora reference mask：条件帧要钉在 latent 时间轴上 / Open-Sora Reference Mask: Pin Condition Frames onto the Latent Timeline

> **一句话 / In one line**: `prepare_inference_condition()` 根据 `i2v/v2v` 模式生成 mask，并把参考 latent 写到首帧、尾帧或一段时间窗口。 / `prepare_inference_condition()` builds a mask for `i2v/v2v` modes and writes reference latents into the head, tail, or a temporal window.

## 为什么重要 / Why this matters

视频 diffusion 不是只有纯文本生成。I2V、V2V、首尾帧约束都需要告诉模型：“这些时间位置已经有条件，不要当成普通噪声。”这段代码把条件语义落成两个张量：`masks` 说明哪里被钉住，`masked_z` 存放钉住的 latent 值。

Video diffusion is not only text-to-video. I2V, V2V, and first/last-frame constraints need to tell the model which time positions are conditioned rather than free noise. This code turns that semantics into two tensors: `masks` says where conditions apply, and `masked_z` stores the conditioned latents.

## 代码 / The code

`hpcaitech/Open-Sora` — [`opensora/utils/inference.py`](https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/utils/inference.py#L283-L351)

```python
def prepare_inference_condition(
    z: torch.Tensor,
    mask_cond: str,
    ref_list: list[list[torch.Tensor]] = None,
    causal: bool = True,
) -> torch.Tensor:
    """
    Prepare the visual condition for the model, using causal vae.
    """
    B, C, T, H, W = z.shape

    masks = torch.zeros(B, 1, T, H, W)
    masked_z = torch.zeros(B, C, T, H, W)

    if ref_list is None:
        assert mask_cond == "t2v", f"reference is required for {mask_cond}"

    for i in range(B):
        ref = ref_list[i]

        if ref is None and mask_cond != "t2v":
            print("no reference found. will default to cond_type t2v!")

        if ref is not None and T > 1:  # video
            if mask_cond == "i2v_head":  # equivalent to masking the first timestep
                masks[i, :, 0, :, :] = 1
                masked_z[i, :, 0, :, :] = ref[0][:, 0, :, :]
            elif mask_cond == "i2v_tail":  # mask the last timestep
                masks[i, :, -1, :, :] = 1
                masked_z[i, :, -1, :, :] = ref[-1][:, -1, :, :]
            elif mask_cond == "v2v_head":
                k = 8 + int(causal)
                masks[i, :, :k, :, :] = 1
                masked_z[i, :, :k, :, :] = ref[0][:, :k, :, :]
            elif mask_cond == "v2v_tail":
                k = 8 + int(causal)
                masks[i, :, -k:, :, :] = 1
                masked_z[i, :, -k:, :, :] = ref[0][:, -k:, :, :]
            elif mask_cond == "v2v_head_easy":
                k = 16 + int(causal)
                masks[i, :, :k, :, :] = 1
                masked_z[i, :, :k, :, :] = ref[0][:, :k, :, :]
            elif mask_cond == "v2v_tail_easy":
                k = 16 + int(causal)
                masks[i, :, -k:, :, :] = 1
                masked_z[i, :, -k:, :, :] = ref[0][:, -k:, :, :]
            elif mask_cond == "i2v_loop":  # mask first and last timesteps
                masks[i, :, 0, :, :] = 1
                masks[i, :, -1, :, :] = 1
                masked_z[i, :, 0, :, :] = ref[0][:, 0, :, :]
                masked_z[i, :, -1, :, :] = ref[-1][:, -1, :, :]
            else:
                assert mask_cond == "t2v", f"Unknown mask condition {mask_cond}"

    masks = masks.to(z.device, z.dtype)
    masked_z = masked_z.to(z.device, z.dtype)
    return masks, masked_z
```

## 逐行讲解 / What's happening

1. **第 300-304 行 / Lines 300-304 (condition tensors)**:
   - 中文: `masks` 只有 1 个 channel，用来广播标记条件位置；`masked_z` 和 latent 同形状，存放条件值。
   - English: `masks` has one channel and marks conditioned positions; `masked_z` matches the latent shape and stores the conditioned values.
2. **第 306-315 行 / Lines 306-315 (reference guard)**:
   - 中文: 没有 reference 时只允许 `t2v`；其他模式缺条件会打印 fallback 提醒。
   - English: Without references, only `t2v` is allowed; other modes emit a fallback warning.
3. **第 318-344 行 / Lines 318-344 (mode-specific placement)**:
   - 中文: `i2v_head/tail` 钉单帧，`v2v_head/tail` 钉一段窗口，`i2v_loop` 同时钉首尾帧。
   - English: `i2v_head/tail` pins one frame, `v2v_head/tail` pins a window, and `i2v_loop` pins both first and last frames.
4. **第 349-351 行 / Lines 349-351 (device/dtype alignment)**:
   - 中文: 返回前把条件张量搬到 `z` 的 device/dtype，避免后续 sampler 混精度报错。
   - English: Before returning, condition tensors move to `z`'s device/dtype to avoid mixed-device or mixed-precision failures.

## 类比 / The analogy

像给时间轴上的胶片夹夹子：夹住第一帧、最后一帧或前几帧，剩下的位置让扩散模型自由补全。

It is like putting clips on a film strip: clamp the first frame, last frame, or a prefix window, and let diffusion fill the rest.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

这是 `action-conditioning` 的视觉条件版本：`reference encoder -> masked_z + masks -> denoiser/sampler`。如果你做机器人 world model，它可以换成“关键帧条件”“当前观测条件”或“动作导致的未来约束”。省掉 mask，模型就不知道哪些 latent 是必须保留的观测，哪些是要生成的未来。

This is the visual-conditioning version of `action-conditioning`: `reference encoder -> masked_z + masks -> denoiser/sampler`. In a robot world model, the same slot can hold keyframe conditions, current observations, or action-induced future constraints.

## 自己跑一遍 / Try it yourself

```python
T = 6
mask = [0] * T
values = ["noise"] * T
ref = ["A", "B", "C", "D", "E", "F"]
mode = "i2v_loop"
if mode == "i2v_loop":
    mask[0] = mask[-1] = 1
    values[0], values[-1] = ref[0], ref[-1]
print(mask)
print(values)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[1, 0, 0, 0, 0, 1]
['A', 'noise', 'noise', 'noise', 'noise', 'F']
```

这个 toy 例子展示 `i2v_loop` 的语义：首尾帧是硬条件，中间帧交给生成过程。

This toy shows the meaning of `i2v_loop`: first and last frames are hard conditions, middle frames are generated.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Diffusion inpainting** / **Diffusion inpainting**: 图像 inpainting 也用 mask 区分保留区域和生成区域。 / Image inpainting also uses masks to separate preserved and generated regions.
- **Control-conditioned WAM** / **Control-conditioned WAM**: 动作条件可以用类似 mask 指定哪些未来状态受已知控制约束。 / Action-conditioned WAMs can use similar masks to mark future states constrained by known controls.

## 注意事项 / Caveats / when it breaks

- **reference 长度要够** / **Reference length must be enough**: `v2v_head_easy` 会取 16 或 17 帧，reference 太短会在上游准备阶段失败。 / `v2v_head_easy` needs 16 or 17 frames; too-short references fail upstream.
- **mask channel 需要可广播** / **Mask channel must broadcast**: `masks` 是 `[B,1,T,H,W]`，后续代码要按 channel 广播使用。 / `masks` is `[B,1,T,H,W]` and downstream code must broadcast it over channels.

## 延伸阅读 / Further reading

- [Open-Sora `prepare_inference_condition`](https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/utils/inference.py#L283-L351)

