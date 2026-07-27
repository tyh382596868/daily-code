---
date: 2026-07-27
topic: vla
source: vla
repo: Physical-Intelligence/openpi
file: src/openpi/shared/image_tools.py
permalink: https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/shared/image_tools.py#L55-L126
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, vla, vision-encoder, preprocessing]
build_role: vision-encoder advanced variant, camera preprocessing contract
---

# openpi resize_with_pad：视觉入口先守住长宽比 / openpi resize_with_pad: Preserve Aspect Ratio at the Vision Door

> **一句话 / In one line**: 相机图像进 VLA 前，先等比缩放再居中 padding，避免把物体几何形状拉坏。 / Before camera frames enter a VLA, resize with aspect ratio preserved and center-pad them so object geometry is not distorted.

## 为什么重要 / Why this matters

VLA 的视觉编码器通常要求固定分辨率，但机器人相机可能来自不同宽高比。openpi 这段 PyTorch 版预处理把输入格式、dtype、缩放比例和 padding 值都固定下来，让后面的 ViT 或 PaliGemma 看到稳定的图像契约。

VLA vision encoders usually require a fixed resolution, while robot cameras may use different aspect ratios. This PyTorch preprocessing path in openpi standardizes layout, dtype handling, resize ratio, and padding value before a ViT or PaliGemma sees the image.

## 代码 / The code

`Physical-Intelligence/openpi` — [`src/openpi/shared/image_tools.py`](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/shared/image_tools.py#L55-L126)

```python
def resize_with_pad_torch(
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """PyTorch version of resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with black. If the image is float32, it must be in the range [-1, 1].

    Args:
        images: Tensor of shape [*b, h, w, c] or [*b, c, h, w]
        height: Target height
        width: Target width
        mode: Interpolation mode ('bilinear', 'nearest', etc.)

    Returns:
        Resized and padded tensor with same shape format as input
    """
    # Check if input is in channels-last format [*b, h, w, c] or channels-first [*b, c, h, w]
    if images.shape[-1] <= 4:  # Assume channels-last format
        channels_last = True
        # Convert to channels-first for torch operations
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension
        images = images.permute(0, 3, 1, 2)  # [b, h, w, c] -> [b, c, h, w]
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension

    batch_size, channels, cur_height, cur_width = images.shape

    # Calculate resize ratio
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Resize
    resized_images = F.interpolate(
        images, size=(resized_height, resized_width), mode=mode, align_corners=False if mode == "bilinear" else None
    )

    # Handle dtype-specific clipping
    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    # Calculate padding
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # Pad
    constant_value = 0 if images.dtype == torch.uint8 else -1.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),  # left, right, top, bottom
        mode="constant",
        value=constant_value,
    )

    # Convert back to original format if needed
    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]
        if batch_size == 1 and images.shape[0] == 1:
            padded_images = padded_images.squeeze(0)  # Remove batch dimension if it was added

    return padded_images
```

## 逐行讲解 / What's happening

1. **第 73-84 行 / Lines 73-84 (`layout detection`)**:
   - 中文: 函数根据最后一维是否像 channel 判断 NHWC/NCHW，并把 torch 运算统一到 NCHW。
   - English: The function infers NHWC versus NCHW from the last dimension and converts computation to NCHW.
2. **第 87-95 行 / Lines 87-95 (`ratio resize`)**:
   - 中文: 用 `max(cur_width / width, cur_height / height)` 保证缩放后不会超过目标框。
   - English: `max(cur_width / width, cur_height / height)` ensures the resized image fits inside the target box.
3. **第 97-103 行 / Lines 97-103 (`dtype policy`)**:
   - 中文: `uint8` 回到 0-255，`float32` 限到 [-1, 1]，其他 dtype 直接拒绝。
   - English: `uint8` returns to 0-255, `float32` is clipped to [-1, 1], and other dtypes are rejected.
4. **第 105-126 行 / Lines 105-126 (`center padding`)**:
   - 中文: 上下左右 padding 用 `divmod` 均分，最后再恢复原来的 channel layout。
   - English: Padding is split with `divmod` across both sides, then the original channel layout is restored.

## 类比 / The analogy

这像把不同尺寸照片装进同一个相框：不能把照片拉扁，只能等比缩放后在空边补黑边。

It is like putting photos of different sizes into one frame: you should not stretch the photo, only scale it proportionally and fill the empty sides.

## 在 nanoVLA 中的位置 / Where this lives in your nanoVLA

在 nanoVLA 里，这属于 `vision-encoder` 前面的 camera preprocessing 子模块。输入是单帧或 batch 相机图像，输出是固定 H/W 的图像张量；上游是机器人 camera driver 或 dataset，下游是 ViT patch embed。省掉它会让不同相机的物体尺度和形状不稳定，生产级还要补相机内参、多相机同步和增强策略。

In nanoVLA this sits just before the `vision-encoder` as camera preprocessing. It takes single-frame or batched camera tensors and emits fixed-H/W images; camera drivers or datasets feed it, and the ViT patch embed consumes it. Without this step, object scale and shape vary across cameras. A production version also needs camera intrinsics, multi-camera synchronization, and augmentation policy.

## 自己跑一遍 / Try it yourself

```python
def resize_plan(h, w, target_h, target_w):
    ratio = max(w / target_w, h / target_h)
    rh, rw = int(h / ratio), int(w / ratio)
    top, extra_h = divmod(target_h - rh, 2)
    left, extra_w = divmod(target_w - rw, 2)
    return (rh, rw), (top, top + extra_h, left, left + extra_w)

print(resize_plan(480, 640, 224, 224))
print(resize_plan(300, 600, 224, 224))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
((168, 224), (28, 28, 0, 0))
((112, 224), (56, 56, 0, 0))
```

宽图会先贴满目标宽度，剩余高度被平均补到上下两侧。

A wide image fills the target width first, and the remaining height is split across top and bottom padding.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **SmolVLA resize** / **SmolVLA resize**: 中文: LeRobot/SmolVLA 也用 resize-with-pad 保持相机几何。 / English: LeRobot/SmolVLA also uses resize-with-pad to preserve camera geometry.
- **OpenVLA processors** / **OpenVLA processors**: 中文: OpenVLA 的 processor 也把图像标准化放在语言/动作融合之前。 / English: OpenVLA processors also normalize images before language/action fusion.

## 注意事项 / Caveats / when it breaks

- **通道判断** / **Channel heuristic**: 中文: `shape[-1] <= 4` 是实用判断，特殊张量布局要显式测试。 / English: `shape[-1] <= 4` is pragmatic; unusual layouts need explicit tests.
- **padding 值** / **Padding value**: 中文: float 图像假定范围是 [-1, 1]，所以黑边使用 -1.0。 / English: Float images are assumed to be in [-1, 1], so black padding uses -1.0.

## 延伸阅读 / Further reading

- [Physical-Intelligence/openpi source](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/shared/image_tools.py#L55-L126)
