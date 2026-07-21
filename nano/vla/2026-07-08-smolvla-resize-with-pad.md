---
date: 2026-07-08
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/policies/smolvla/modeling_smolvla.py
permalink: https://github.com/huggingface/lerobot/blob/8a74e0ac6d01706d67fddfed682a09d694d9c8c0/src/lerobot/policies/smolvla/modeling_smolvla.py#L132-L153
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, vla, vision-encoder, preprocessing]
build_role: vision-encoder advanced variant
---

# SmolVLA resize_with_pad：相机图像先等比缩放再补齐 / SmolVLA resize_with_pad: Resize Camera Frames, Then Pad

> **一句话 / In one line**: `resize_with_pad` 把任意相机画面变成固定尺寸，同时保留长宽比。 / `resize_with_pad` converts arbitrary camera frames to a fixed size while preserving aspect ratio.

## 为什么重要 / Why this matters

VLA 的视觉编码器通常要求固定分辨率，但机器人相机可能来自不同设备、不同裁剪和不同宽高比。直接拉伸会改变几何关系，尤其会影响抓取、末端位置和桌面物体的形状。SmolVLA 用“等比缩放 + 左上 padding”的小函数，把输入规整到模型尺寸，同时尽量不扭曲画面。

A VLA vision encoder usually expects a fixed resolution, while robot cameras may have different devices, crops, and aspect ratios. Direct stretching changes geometry, which is especially harmful for grasping, end-effector position, and object shape. SmolVLA uses a small "resize then pad" helper to standardize inputs without distorting the image.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/smolvla/modeling_smolvla.py`](https://github.com/huggingface/lerobot/blob/8a74e0ac6d01706d67fddfed682a09d694d9c8c0/src/lerobot/policies/smolvla/modeling_smolvla.py#L132-L153)

```python
def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img
```

## 逐行讲解 / What's happening

1. **第 134-135 行 / Lines 134-135 (shape check)**:
   - 中文: 函数只接受 `[B, C, H, W]`，因为后续 `F.interpolate` 和 `F.pad` 都按 batch 图像处理。
   - English: The function accepts only `[B, C, H, W]`, matching how `F.interpolate` and `F.pad` operate on batched images.
2. **第 139 行 / Line 139 (`ratio`)**:
   - 中文: 取宽缩放比和高缩放比的较大值，保证缩放后不会超过目标画布。
   - English: It takes the larger width/height ratio so the resized image never exceeds the target canvas.
3. **第 142-144 行 / Lines 142-144 (`F.interpolate`)**:
   - 中文: 用双线性插值等比缩小或放大，保留图像几何比例。
   - English: Bilinear interpolation resizes the image while preserving geometric proportions.
4. **第 149-151 行 / Lines 149-151 (`F.pad`)**:
   - 中文: 剩余空白补在左侧和上侧，padding 值默认为 `-1`，通常对应归一化图像的背景。
   - English: The remaining area is padded on the left and top with `-1`, often matching the normalized image background.

## 类比 / The analogy

这像把不同尺寸的照片贴进同一个相框：你先按比例缩放照片，不能把人脸拉长；空出来的边用底纸补齐，相框尺寸才统一。

It is like placing different photos into the same frame: scale each photo proportionally so faces are not stretched, then fill the empty area with backing paper.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

这是 `vision-encoder` 前的输入规整层。上游是多相机观测和图像增强，下游是 patch embedding / SigLIP / DINO 这类视觉 backbone。如果省掉它，你的 nanoVLA 会把“相机宽高比差异”误当成真实世界几何变化。生产级实现还要记录 padding mask，或者保证训练和部署的 padding 方向完全一致。

This belongs immediately before the `vision-encoder`. Upstream are camera observations and augmentations; downstream are patch embedding or a visual backbone such as SigLIP or DINO. If you omit it, your nanoVLA may confuse camera aspect-ratio differences with real-world geometry. A production implementation should also track masks or keep padding direction identical between training and deployment.

## 自己跑一遍 / Try it yourself

```python
def resize_shape(cur_w, cur_h, width, height):
    ratio = max(cur_w / width, cur_h / height)
    rw, rh = int(cur_w / ratio), int(cur_h / ratio)
    return {"resized": (rw, rh), "pad_left": width - rw, "pad_top": height - rh}

print(resize_shape(640, 480, 224, 224))
print(resize_shape(320, 640, 224, 224))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
{'resized': (224, 168), 'pad_left': 0, 'pad_top': 56}
{'resized': (112, 224), 'pad_left': 112, 'pad_top': 0}
```

宽图会补高度，竖图会补宽度；真正被保持住的是图像比例，而不是填满整个方形画布。

Wide images get height padding, tall images get width padding. The preserved quantity is aspect ratio, not full canvas occupancy.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **OpenVLA image processors** / **OpenVLA image processors**: 也会把机器人观测转成 VLM backbone 期望的固定图像规格。 / They also convert robot observations into the fixed image format expected by the VLM backbone.
- **nanoVLM patch embedding** / **nanoVLM patch embedding**: patch encoder 需要稳定的空间网格，输入尺寸变化会直接改变 token 数。 / Patch encoders need a stable spatial grid; changing input size changes token count.

## 注意事项 / Caveats / when it breaks

- **padding 方向是建模选择** / **Padding direction is a modeling choice**: 左上补齐会改变物体在画布里的绝对位置，训练和部署必须一致。 / Left/top padding changes absolute object position on the canvas, so train and deploy must match.
- **mask 可能更干净** / **A mask may be cleaner**: 如果 backbone 支持 image mask，显式告诉模型哪里是 padding 会更稳。 / If the backbone supports an image mask, explicitly marking padded pixels is more robust.

## 延伸阅读 / Further reading

- [SmolVLA `resize_with_pad`](https://github.com/huggingface/lerobot/blob/8a74e0ac6d01706d67fddfed682a09d694d9c8c0/src/lerobot/policies/smolvla/modeling_smolvla.py#L132-L153)
