---
date: 2026-09-01
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/vace.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/vace.py#L139-L210
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, wam, action-conditioning, vace, masks]
build_role: action-conditioning advanced variant
---

# Wan2.1 VACE mask：把可控区域变成双通道 latent / Wan2.1 VACE Mask: Turn Editable Regions into Paired Latents

> **一句话 / In one line**: VACE 先用 mask 把视频分成 inactive/reactive 两份，再把 mask 下采样到 VAE latent 尺度作为条件一起送给模型。 / VACE splits video into inactive and reactive parts with a mask, then downsamples the mask to VAE-latent scale as conditioning.

## 为什么重要 / Why this matters

World Action Model 需要区分“必须保留的世界状态”和“动作可以改变的区域”。如果条件只是一整段视频，模型不知道哪些像素应该钉住，哪些地方可以响应控制。Wan2.1 VACE 把这个约束显式编码：inactive 部分保留背景，reactive 部分承载可变区域，mask 再跟 latent 对齐，告诉 DiT 哪里该听控制信号。

A World Action Model must separate world state that should stay fixed from regions an action may change. If conditioning is just a video, the model cannot know which pixels are anchored and which may react. Wan2.1 VACE encodes the constraint explicitly: inactive latents preserve background, reactive latents carry editable regions, and the mask is aligned to the latent grid so the DiT knows where control should apply.

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/vace.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/vace.py#L139-L210)

```python
def vace_encode_frames(self, frames, ref_images, masks=None, vae=None):
    vae = self.vae if vae is None else vae
    if ref_images is None:
        ref_images = [None] * len(frames)
    else:
        assert len(frames) == len(ref_images)

    if masks is None:
        latents = vae.encode(frames)
    else:
        masks = [torch.where(m > 0.5, 1.0, 0.0) for m in masks]
        inactive = [i * (1 - m) + 0 * m for i, m in zip(frames, masks)]
        reactive = [i * m + 0 * (1 - m) for i, m in zip(frames, masks)]
        inactive = vae.encode(inactive)
        reactive = vae.encode(reactive)
        latents = [
            torch.cat((u, c), dim=0) for u, c in zip(inactive, reactive)
        ]

    cat_latents = []
    for latent, refs in zip(latents, ref_images):
        if refs is not None:
            if masks is None:
                ref_latent = vae.encode(refs)
            else:
                ref_latent = vae.encode(refs)
                ref_latent = [
                    torch.cat((u, torch.zeros_like(u)), dim=0)
                    for u in ref_latent
                ]
            assert all([x.shape[1] == 1 for x in ref_latent])
            latent = torch.cat([*ref_latent, latent], dim=1)
        cat_latents.append(latent)
    return cat_latents

def vace_encode_masks(self, masks, ref_images=None, vae_stride=None):
    vae_stride = self.vae_stride if vae_stride is None else vae_stride
    if ref_images is None:
        ref_images = [None] * len(masks)
    else:
        assert len(masks) == len(ref_images)

    result_masks = []
    for mask, refs in zip(masks, ref_images):
        c, depth, height, width = mask.shape
        new_depth = int((depth + 3) // vae_stride[0])
        height = 2 * (int(height) // (vae_stride[1] * 2))
        width = 2 * (int(width) // (vae_stride[2] * 2))

        # reshape
        mask = mask[0, :, :, :]
        mask = mask.view(depth, height, vae_stride[1], width,
                         vae_stride[1])  # depth, height, 8, width, 8
        mask = mask.permute(2, 4, 0, 1, 3)  # 8, 8, depth, height, width
        mask = mask.reshape(vae_stride[1] * vae_stride[2], depth, height,
                            width)  # 8*8, depth, height, width

        # interpolation
        mask = F.interpolate(
            mask.unsqueeze(0),
            size=(new_depth, height, width),
            mode='nearest-exact').squeeze(0)

        if refs is not None:
            length = len(refs)
            mask_pad = torch.zeros_like(mask[:, :length, :, :])
            mask = torch.cat((mask_pad, mask), dim=1)
        result_masks.append(mask)
    return result_masks

def vace_latent(self, z, m):
    return [torch.cat([zz, mm], dim=0) for zz, mm in zip(z, m)]
```

## 逐行讲解 / What's happening

1. **第 146-156 行 / Lines 146-156 (split by binary mask)**:
   - 中文: mask 先二值化；`inactive` 把可编辑区域清零，`reactive` 把不可编辑区域清零，然后两份分别过 VAE。
   - English: The mask is binarized; `inactive` zeros editable regions, `reactive` zeros non-editable regions, and both are encoded by the VAE.
2. **第 154-156 行 / Lines 154-156 (channel concat)**:
   - 中文: 两份 latent 沿通道维拼起来，让模型同时看到背景约束和反应区域。
   - English: The two latents are concatenated along channels so the model sees both background constraints and reactive regions.
3. **第 158-171 行 / Lines 158-171 (reference frames)**:
   - 中文: 如果有参考图，参考 latent 被拼到时间维前面；有 mask 时还给参考 latent 补一份零 reactive 通道。
   - English: Reference-image latents are prepended along time; with masks, they also receive a zero reactive channel.
4. **第 181-200 行 / Lines 181-200 (latent-scale mask)**:
   - 中文: mask 被按 VAE stride 重排并最近邻插值到 latent 的时空分辨率，避免条件和 latent 网格错位。
   - English: The mask is reshaped by VAE stride and nearest-neighbor resized to latent spatiotemporal resolution so conditioning aligns with the latent grid.
5. **第 209-210 行 / Lines 209-210 (final conditioning tensor)**:
   - 中文: 最后 `z` 和 mask 沿通道拼接，形成 DiT 读取的控制输入。
   - English: Finally, `z` and the mask are concatenated by channel into the control input consumed by the DiT.

## 类比 / The analogy

像修照片时先画蒙版：桌子区域锁住，杯子区域允许修改。软件不是靠猜，而是同时拿到原图、可编辑图层和蒙版。

It is like editing a photo with a mask: the table is locked while the cup can change. The software does not guess; it receives the original layer, editable layer, and mask together.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

中文: 这是 `action-conditioning` 的 advanced variant，依赖 VAE latent 网格和 DiT 条件入口。nanoWAM 里可以把机器人动作、接触区域、目标 mask 或首帧约束都转成同样的 latent-aligned condition。输入是视频帧、参考图和 mask；输出是拼通道 latent。省掉它，模型会把“哪里能变”混成隐式知识，控制会不稳定；生产版还要处理软 mask、多对象 mask、相机坐标到图像 mask 的投影和 mask 时序平滑。

English: This is an advanced variant of `action-conditioning`, depending on the VAE latent grid and the DiT conditioning path. In a nanoWAM, robot actions, contact regions, goal masks, or first-frame constraints can all become latent-aligned conditions. Inputs are frames, references, and masks; output is a channel-concatenated latent. Without this, the model must infer "what may change" implicitly, making control unstable. Production versions add soft masks, multi-object masks, camera-to-image projection, and temporal smoothing.

## 自己跑一遍 / Try it yourself

```python
frames = [10, 20, 30, 40]
masks = [0, 1, 1, 0]
inactive = [x * (1 - m) for x, m in zip(frames, masks)]
reactive = [x * m for x, m in zip(frames, masks)]
latent_channels = inactive + reactive
mask_latent = masks[::2]
conditioning = latent_channels + mask_latent
print(inactive)
print(reactive)
print(conditioning)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[10, 0, 0, 40]
[0, 20, 30, 0]
[10, 0, 0, 40, 0, 20, 30, 0, 0, 1]
```

中文: 示例把空间维简化成一维，但核心不变：内容和 mask 必须在同一个 latent 尺度上对齐。

English: The example collapses space into one dimension, but the core idea is unchanged: content and mask must align on the same latent scale.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Wan2.1 I2V conditioning** / **Wan2.1 I2V conditioning**: 中文: 首帧条件也是把确定区域钉到 latent 时间轴。 / English: First-frame conditioning also pins known regions onto the latent timeline.
- **Open-Sora reference masks** / **Open-Sora reference masks**: 中文: 参考帧控制同样需要把 mask 映射到模型实际采样的 latent 分辨率。 / English: Reference-frame control also maps masks to the latent resolution used by sampling.

## 注意事项 / Caveats / when it breaks

- **mask 必须对齐 VAE stride / Masks must match VAE stride**: 中文: 高宽不能随便取，否则 `view` 和 latent 尺寸会错。 / English: Height and width cannot be arbitrary; otherwise `view` and latent dimensions break.
- **硬二值 mask 会丢边界 / Hard masks lose boundaries**: 中文: `torch.where(m > 0.5)` 简单可靠，但边缘细节可能需要软权重。 / English: `torch.where(m > 0.5)` is simple and robust, but fine boundaries may need soft weights.

## 延伸阅读 / Further reading

- Wan2.1 VACE source: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/vace.py
