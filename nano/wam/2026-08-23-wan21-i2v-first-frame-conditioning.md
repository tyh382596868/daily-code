---
date: 2026-08-23
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/image2video.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/main/wan/image2video.py#L167-L229
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, action-conditioning, image-to-video]
build_role: action-conditioning advanced variant
---

# Wan2.1 I2V conditioning：首帧钉住，后续交给噪声 / Wan2.1 I2V Conditioning: Pin the First Frame, Let Noise Fill the Rest

> **一句话 / In one line**: Wan2.1 把输入图像编码成 latent 条件，并配一个 mask，告诉 DiT 第一帧是已知条件，其余帧从噪声生成。 / Wan2.1 encodes the input image as a latent condition and attaches a mask telling the DiT that the first frame is fixed while later frames are generated from noise.

## 为什么重要 / Why this matters

World Action Model 经常要从一个已知状态出发想象未来。I2V 的第一帧就是这个已知状态：如果模型不知道哪一段是条件、哪一段是待生成目标，它会把条件也当成噪声一起改掉。

A World Action Model often imagines the future from a known state. In image-to-video, the first frame is that known state. If the model cannot distinguish condition from target, it may rewrite the initial state as if it were noise.

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/image2video.py`](https://github.com/Wan-Video/Wan2.1/blob/main/wan/image2video.py#L167-L229)

```python
img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
F = frame_num
h, w = img.shape[1:]
aspect_ratio = h / w
lat_h = round(
    np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
    self.patch_size[1] * self.patch_size[1])
lat_w = round(
    np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
    self.patch_size[2] * self.patch_size[2])
h = lat_h * self.vae_stride[1]
w = lat_w * self.vae_stride[2]
max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
    self.patch_size[1] * self.patch_size[2])
max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size
seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
seed_g = torch.Generator(device=self.device)
seed_g.manual_seed(seed)
noise = torch.randn(
    16, (F - 1) // 4 + 1,
    lat_h,
    lat_w,
    dtype=torch.float32,
    generator=seed_g,
    device=self.device)
msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
msk[:, 1:] = 0
msk = torch.concat([
    torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
],
                   dim=1)
msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
msk = msk.transpose(1, 2)[0]

y = self.vae.encode([
    torch.concat([
        torch.nn.functional.interpolate(
            img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                0, 1),
        torch.zeros(3, F - 1, h, w)
    ],
                 dim=1).to(self.device)
])[0]
y = torch.concat([msk, y])
```

## 逐行讲解 / What's happening

1. **第 167-181 行 / Lines 167-181**:
   - 中文: 输入图像先归一化，再根据面积、宽高比、VAE stride 和 patch size 推出 latent 网格尺寸。
   - English: The input image is normalized, then latent grid size is derived from area, aspect ratio, VAE stride, and patch size.
2. **第 185-191 行 / Lines 185-191**:
   - 中文: 目标视频 latent 先从随机噪声开始，时间长度是 `(F - 1) // 4 + 1`。
   - English: The target video latent starts as random noise with temporal length `(F - 1) // 4 + 1`.
3. **第 192-200 行 / Lines 192-200**:
   - 中文: mask 第一帧为 1，其余为 0；随后重排成 VAE temporal chunk 对齐的形状。
   - English: The mask marks the first frame as 1 and later frames as 0, then reshapes to match VAE temporal chunks.
4. **第 219-229 行 / Lines 219-229**:
   - 中文: 条件视频由“缩放后的输入图 + 后续全零帧”组成，经过 VAE 编码后与 mask 拼接。
   - English: The conditioning video is the resized input image followed by zero frames. It is VAE-encoded and concatenated with the mask.

## 类比 / The analogy

这像漫画分镜：第一格已经画好，后面几格还是空白。mask 是贴在画纸上的便签，告诉助手“第一格不要改，后面按故事继续画”。

It is like a comic storyboard: the first panel is already drawn, and the later panels are blank. The mask is a sticky note saying, "Do not change panel one; continue the story in the rest."

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

在 nanoWAM 中，这属于 `action-conditioning` 的视觉条件变体：把当前观测状态变成一个 latent 条件通道，并让生成模型知道哪些时刻已知、哪些时刻要预测。上游是 VAE encoder 和当前观测，下游是 DiT denoiser。生产版本还会加入动作 token、相机位姿、接触状态或多视角条件。

In a nanoWAM, this is a visual-conditioning variant of `action-conditioning`: it turns the current observation into a latent condition channel and tells the generator which timesteps are known versus predicted. Upstream are the VAE encoder and current observation; downstream is the DiT denoiser. A production version may add action tokens, camera pose, contact state, or multi-view conditioning.

## 自己跑一遍 / Try it yourself

```python
frames = ["start", "?", "?", "?", "?"]
mask = [1] + [0] * (len(frames) - 1)
condition = ["start"] + ["zero"] * (len(frames) - 1)

packed = list(zip(mask, condition))
for i, (m, c) in enumerate(packed):
    role = "fixed condition" if m else "to generate"
    print(i, c, role)
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
0 start fixed condition
1 zero to generate
2 zero to generate
3 zero to generate
4 zero to generate
```

这个最小例子保留了 Wan2.1 的关键契约：条件内容和条件 mask 要一起传。

This minimal example preserves Wan2.1's key contract: condition content and condition mask travel together.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Open-Sora reference masks** / **Open-Sora reference masks**: 用 mask 固定参考帧或首尾帧。 / Masks pin reference, head, or tail frames.
- **inpainting diffusion** / **inpainting diffusion**: mask 区分已知像素和要补全的像素。 / A mask separates known pixels from regions to fill.
- **robot rollout models** / **robot rollout models**: 初始状态固定，后续状态由动作条件展开。 / The initial state is fixed while future states roll out under actions.

## 注意事项 / Caveats / when it breaks

- **时间维要对齐 VAE stride** / **Time must align to VAE stride**: mask 被 repeat 和 view，错一格就会钉错帧。 / The mask is repeated and reshaped; one offset pins the wrong frame.
- **零帧不是无条件** / **Zero frames are not no condition**: 真正的“已知/未知”由 mask 决定。 / The mask, not the zero tensor alone, defines known versus unknown.
- **多视角更复杂** / **Multi-view is more complex**: 多相机条件需要额外通道或 token 标识来源。 / Multi-camera conditioning needs additional channels or source tokens.

## 延伸阅读 / Further reading

- Wan2.1 I2V source: https://github.com/Wan-Video/Wan2.1/blob/main/wan/image2video.py
- Wan2.1 repository: https://github.com/Wan-Video/Wan2.1
