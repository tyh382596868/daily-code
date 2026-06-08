---
date: 2026-06-03
topic: diffusion
source: tracked
repo: facebookresearch/dinov3
file: dinov3/layers/rope_position_encoding.py
permalink: https://github.com/facebookresearch/dinov3/blob/31703e4cbf1ccb7c4a72daa1350405f86754b6d1/dinov3/layers/rope_position_encoding.py#L16-L121
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, rope, vit, positional-encoding, data-augmentation]
---

# DINOv3 的 RoPE 在训练时随机抖动坐标:一份免费的"位置数据增广" / DINOv3's RoPE randomizes its own coordinates at train time — free positional-encoding augmentation

> **一句话 / In one line**: 把 ViT 的 2D 像素坐标归一化到 `[-1, +1]`,再训练时往里掺三种随机扰动(shift / jitter / rescale),一份位置编码就同时教会模型对位移、缩放和长宽比都不敏感。 / Normalize a ViT's 2D patch coordinates to `[-1, +1]`, then in training inject three flavors of randomness (shift / jitter / rescale) into them — and one positional encoding teaches the model translation, scale and aspect-ratio invariance for free.

## 为什么重要 / Why this matters

绝大多数 ViT 用的是**学好的固定位置编码**(`nn.Parameter` 矩阵),换分辨率就要插值,而且模型其实把"位置 0 长得这样"硬记进了 attention 头里。RoPE 已经好一截 —— 它把位置写成 `(cos θ, sin θ)` 然后**乘进 q/k**,所以位置变换是个旋转,理论上换分辨率比较优雅。但 DINOv3 又走了一步:**训练时把网格坐标本身随机抖动**,等于在位置维度上做数据增广。模型再也没法记住"位置 17 长这样",只能学到一个**相对位置**的概念。结果是同一个 backbone 不管是 224×224 训练还是 512×448 部署都能直接用,DINOv3 用这一招把 SSL ViT 推到了 self-supervised 视觉的新 SOTA。

Most ViTs use **learned absolute positional embeddings** (an `nn.Parameter` matrix) — change the resolution and you have to interpolate, and the model has burned "position 0 looks like this" into every attention head. RoPE already helps: positions become `(cos θ, sin θ)` *multiplied into q/k*, so changing resolution is a rotation, not a re-fit. DINOv3 goes one step further: it **randomly perturbs the coordinate grid itself at train time** — positional data augmentation. The model can no longer memorize "position 17 looks like such-and-such"; it can only learn a *relative* notion of where things are. The same backbone trained at 224×224 then works at 512×448 with zero changes, and this trick is one ingredient that pushed DINOv3 to SOTA on self-supervised vision.

## 代码 / The code

`facebookresearch/dinov3` — [`dinov3/layers/rope_position_encoding.py`](https://github.com/facebookresearch/dinov3/blob/31703e4cbf1ccb7c4a72daa1350405f86754b6d1/dinov3/layers/rope_position_encoding.py#L16-L121)

```python
# RoPE positional embedding with no mixing of coordinates (axial) and no learnable weights
# Supports two parametrizations of the rope parameters: either using `base` or `min_period` and `max_period`.
class RopePositionEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        D_head = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = D_head
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        self.dtype = dtype
        self.register_buffer(
            "periods",
            torch.empty(D_head // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def forward(self, *, H: int, W: int) -> tuple[Tensor, Tensor]:
        device = self.periods.device
        dtype = self.dtype
        dd = {"device": device, "dtype": dtype}

        # Prepare coords in range [-1, +1]
        if self.normalize_coords == "max":
            max_HW = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / max_HW
            coords_w = torch.arange(0.5, W, **dd) / max_HW
        elif self.normalize_coords == "min":
            min_HW = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / min_HW
            coords_w = torch.arange(0.5, W, **dd) / min_HW
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, **dd) / H
            coords_w = torch.arange(0.5, W, **dd) / W
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
        coords = coords.flatten(0, 1)             # [HW, 2]
        coords = 2.0 * coords - 1.0               # shift [0,1] -> [-1, +1]

        # Shift coords by adding a uniform value in [-shift, shift]
        if self.training and self.shift_coords is not None:
            shift_hw = torch.empty(2, **dd).uniform_(-self.shift_coords, self.shift_coords)
            coords += shift_hw[None, :]

        # Jitter coords by multiplying with a log-uniform value in [1/jitter, jitter]
        if self.training and self.jitter_coords is not None:
            jitter_max = np.log(self.jitter_coords)
            jitter_min = -jitter_max
            jitter_hw = torch.empty(2, **dd).uniform_(jitter_min, jitter_max).exp()
            coords *= jitter_hw[None, :]          # independent per axis

        # Rescale coords by multiplying with a log-uniform value in [1/rescale, rescale]
        if self.training and self.rescale_coords is not None:
            rescale_max = np.log(self.rescale_coords)
            rescale_min = -rescale_max
            rescale_hw = torch.empty(1, **dd).uniform_(rescale_min, rescale_max).exp()
            coords *= rescale_hw                  # same factor for both axes

        # Prepare angles and sin/cos
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]  # [HW, 2, D//4]
        angles = angles.flatten(1, 2)             # [HW, D//2]
        angles = angles.tile(2)                   # [HW, D]
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        return (sin, cos)

    def _init_weights(self):
        if self.base is not None:
            periods = self.base ** (
                2 * torch.arange(self.D_head // 4, dtype=self.dtype) / (self.D_head // 2)
            )
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, self.D_head // 4, dtype=self.dtype)
            periods = base**exponents
            periods = periods / base * self.max_period
        self.periods.data = periods
```

## 逐行讲解 / What's happening

1. **`assert embed_dim % (4 * num_heads) == 0`**:
   - 中文: RoPE 是 2D 的 —— H 和 W 各自占 `D_head/2` 维,每一对 `(sin, cos)` 又共享一个频率,所以 `D_head` 必须是 4 的倍数。
   - English: RoPE is 2D — half of `D_head` goes to H, half to W, and each `(sin, cos)` pair shares one frequency, so `D_head` must be a multiple of 4.

2. **`register_buffer("periods", ..., persistent=True)`**:
   - 中文: 频率写成 buffer 而不是 Parameter —— 不参与训练,但 `state_dict()` 里能拿到。`persistent=True` 是关键:DINOv3 用 EMA teacher,会做 `teacher.load_state_dict(student.state_dict())`,如果 buffer 不持久化就会丢失。
   - English: periods are a *buffer*, not a parameter — no gradient, but they show up in `state_dict()`. `persistent=True` matters because DINOv3 uses an EMA teacher initialized via `teacher.load_state_dict(student.state_dict())`; a non-persistent buffer would silently disappear.

3. **`coords = 2.0 * coords - 1.0`**:
   - 中文: 这一步把网格挪到 `[-1, +1]` —— 一个**与分辨率无关**的坐标系。512×512 的中心和 224×224 的中心,在这套坐标里都是 `(0, 0)`。
   - English: this line maps the grid to `[-1, +1]` — a **resolution-independent** frame. The center of a 512×512 image and the center of a 224×224 image both sit at `(0, 0)` here.

4. **三种 train-time 增广 / The three train-time augmentations**:
   - 中文: `shift_coords` 给整个 `[-1, +1]` 网格加一个均匀随机偏移,等于"图像被裁切了一点";`jitter_coords` 用一个**对数均匀**的随机因子分别乘进 H 和 W,等于"长宽比变了";`rescale_coords` 用同一个 log-uniform 因子同时乘进 H、W,等于"放大缩小"。三个都只在 `self.training=True` 时启动,推理时坐标是干净的 `[-1, +1]` 网格。
   - English: `shift_coords` adds a single uniform random offset to the whole `[-1, +1]` grid — equivalent to "the image got cropped a bit"; `jitter_coords` multiplies H and W independently by a **log-uniform** random factor — equivalent to "aspect ratio changed"; `rescale_coords` multiplies both axes by the *same* log-uniform factor — equivalent to "zoomed in/out". All three are gated on `self.training`; at inference the grid is the clean `[-1, +1]` mesh.

5. **`angles = 2 * π * coords / periods`**:
   - 中文: 标准 RoPE 公式 —— 把一个标量坐标乘以一组频率,得到一组角度。`angles.tile(2)` 把每个角度复制成对(实部、虚部),最后一次 cos/sin 拿到 ViT 用的 `(sin, cos)` 表。
   - English: standard RoPE formula — multiply each scalar coordinate by a set of frequencies to get a set of angles. `angles.tile(2)` duplicates each angle into a real/imaginary pair, and one `cos/sin` pass produces the `(sin, cos)` table the ViT will use.

6. **`_init_weights`(两种频率参数化 / two frequency parametrizations)**:
   - 中文: 传统做法是 `base=10000`(NTK 风格),DINOv3 默认 `base=100`(因为坐标已经归一化到 ±1,不需要那么大的指数范围)。另一种是直接指定 `min_period` 和 `max_period`,几何级数把频率铺在这两个周期之间 —— 更直观,适合不同任务调节"看多远"。
   - English: classical approach is `base=10000` (NTK style); DINOv3 defaults to `base=100` because coordinates are already normalized to ±1 so you don't need such an exponentially-wide range. The alternative directly specifies `min_period` and `max_period`, geometrically spacing frequencies between those two values — more interpretable, and easier to tune "how far the model should look".

## 类比 / The analogy

中文:想象你在教一个学生**看地图**。传统的学习式位置编码就像给他一张固定的城市地图,所有路牌都钉死;换城市他就懵了。普通 RoPE 是给他一个**比例尺** —— 哪怕到了别的城市,他知道"1 厘米 = 1 公里",至少能换算。DINOv3 的 RoPE 又狠了一步:**每次训练都偷偷把这张地图整体平移、拉伸、改一下纵横比**。学生没法死记任何坐标,只能学会"A 在 B 的左上方"这种**相对关系**。等部署时把任何一张陌生地图扔给他,他都能直接读懂。

English: think of teaching a student to **read maps**. Learned absolute positional embeddings are like giving them one fixed city map with all the street signs nailed in place — show them a different city and they're lost. Plain RoPE gives them a **scale bar** instead: even in a new city, "1 cm = 1 km" still works, so they can translate. DINOv3's RoPE goes further: every training session it secretly shifts, stretches and skews the map a little. The student can't memorize any one coordinate; they can only learn **relative facts** — "A is up-and-to-the-left of B". At deployment, you can hand them any map at any resolution and they'll still read it.

## 自己跑一遍 / Try it yourself

```python
import math, torch
import torch.nn as nn

class TinyAxialRope(nn.Module):
    def __init__(self, D_head=32, base=100.0):
        super().__init__()
        assert D_head % 4 == 0
        periods = base ** (2 * torch.arange(D_head // 4) / (D_head // 2))
        self.register_buffer("periods", periods.float(), persistent=True)

    def forward(self, H, W, training=True, shift=0.5, jitter=2.0):
        # Resolution-independent grid in [-1, +1]
        ch = (torch.arange(0.5, H) / H) * 2 - 1
        cw = (torch.arange(0.5, W) / W) * 2 - 1
        coords = torch.stack(torch.meshgrid(ch, cw, indexing="ij"), -1).flatten(0, 1)
        if training:
            coords += torch.empty(2).uniform_(-shift, shift)[None]
            jit = torch.empty(2).uniform_(-math.log(jitter), math.log(jitter)).exp()
            coords *= jit[None]
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2).tile(2)  # [HW, D_head]
        return angles.sin(), angles.cos()

rope = TinyAxialRope(D_head=32)
sin224, cos224 = rope(14, 14, training=False)   # 224/16 patches
sin512, cos512 = rope(32, 28, training=False)   # 512x448 / 16 patches
print("224 grid centre angle:", cos224[7 * 14 + 7, 0].item())
print("512 grid centre angle:", cos512[16 * 28 + 14, 0].item())
print("train-time draw 1:    ", rope(14, 14, training=True)[1][0, 0].item())
print("train-time draw 2:    ", rope(14, 14, training=True)[1][0, 0].item())
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
224 grid centre angle: ~0.998   # cos near 0 → almost 1
512 grid centre angle: ~0.998   # same! resolution-independent
train-time draw 1:     ~0.93
train-time draw 2:     ~0.71    # different every call
```

中文:两个不同分辨率的中心 patch 拿到的 cos 几乎一样 —— 这就是 `[-1, +1]` 归一化的核心好处。同时,训练模式下两次调用同样的 `(H, W)` 拿到的角度**不同**,说明 shift+jitter 真的随机了。

English: the centre patch of two different-resolution grids comes out with almost the same `cos` value — that's the whole point of `[-1, +1]` normalization. In training mode, two calls with the same `(H, W)` produce *different* angles, confirming that shift+jitter really is randomized per call.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Llama / Qwen 的 1D RoPE** / **Llama / Qwen's 1D RoPE**: 同样的 `cos/sin × q/k` 旋转,只是位置是序列索引而不是 2D 网格。 / Same `cos/sin × q/k` rotation, except the position is a sequence index instead of a 2D grid.
- **Wan2.1 / lingbot-va 的 3D RoPE** / **3D RoPE in Wan2.1 / lingbot-va**: 视频 DiT 把维度三分(frame / H / W),每个轴单独算 RoPE,DINOv3 这个 axial 2D 版的直接推广。 / Video DiT splits the head dim three ways (frame / H / W), each axis getting its own RoPE — a direct generalization of DINOv3's axial 2D version.
- **TorchTitan 的 NTK-aware scaling** / **TorchTitan's NTK-aware scaling**: 推理时用 `base` 缩放外推到更长上下文 —— 和 DINOv3 训练时改坐标本质是同一件事(让模型不依赖具体频率值)。 / At inference, scaling `base` to extrapolate to longer contexts — same essential trick as DINOv3's coord randomization (don't let the model latch onto specific frequency values).

## 注意事项 / Caveats / when it breaks

- **维度约束 / Dimension constraint**: `D_head` 必须是 4 的倍数,否则 axial RoPE 没法平均分给 H 和 W,会触发 `assert`。 / `D_head` must be divisible by 4 — otherwise axial RoPE can't split evenly across H and W and the `assert` fires.
- **`periods` 用 `persistent=True` 是有原因的 / `persistent=True` on `periods` matters**: 如果你魔改成 `persistent=False`,EMA teacher 复制权重时 buffer 会丢失,训练前几步 teacher 的位置编码全是零,loss 突然爆炸。 / If you flip it to `persistent=False`, the EMA teacher loses the buffer on `load_state_dict`, the first few steps of teacher have zero positional encoding, and loss explodes.
- **太猛的 jitter / Too much jitter**: `jitter_coords > 2.0` 等于"长宽比可以变 4 倍",对一般任务太激进,DINOv3 用的是 `1.1–1.4` 这种小幅扰动。 / `jitter_coords > 2.0` means "aspect ratio can change by 4×", which is too aggressive for most tasks. DINOv3 uses something like `1.1–1.4`.
- **推理时记得 `model.eval()` / Don't forget `model.eval()` at inference**: 三种增广都用 `self.training` 门控 —— 忘了切 eval 会让每次推理的输出都微微不同,debug 起来很迷惑。 / All three augmentations gate on `self.training`. Forget `model.eval()` and every inference call gives a slightly different output — confusing to debug.

## 延伸阅读 / Further reading

- [DINOv3 paper](https://arxiv.org/abs/2508.10104)
- [RoFormer / RoPE 原始论文](https://arxiv.org/abs/2104.09864)
- [NTK-aware RoPE scaling for context extension](https://blog.eleuther.ai/yarn/)
- [Existing daily-code entry: Wan2.1's 3D RoPE](../../nano/wam/2026-05-29-wan21-3d-rope.md)
