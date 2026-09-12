---
date: 2026-08-14
topic: diffusion
source: tracked
repo: facebookresearch/dinov3
file: dinov3/layers/rope_position_encoding.py
permalink: https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/layers/rope_position_encoding.py#L16-L121
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, rope]
---

# DINOv3 RoPE 坐标增强：位置编码也能做数据增强 / DINOv3 RoPE Coordinate Augmentation: Position Encoding Can Be Augmented Too

> **一句话 / In one line**: `RopePositionEmbedding` 先把 patch 坐标归一到 `[-1, 1]`，训练时再对坐标做 shift、jitter、rescale，最后生成轴向 RoPE 的 `sin/cos`。 / `RopePositionEmbedding` normalizes patch coordinates to `[-1, 1]`, optionally augments them during training, then emits axial RoPE `sin/cos` tables.

## 为什么重要 / Why this matters

视觉 backbone 经常要吃不同长宽比、不同分辨率的图像。DINOv3 这段代码把 RoPE 从“固定网格查表”改成“按当前 `H, W` 即时生成坐标”，还允许训练期轻微扰动坐标，让模型不要把位置尺度记死。

Vision backbones often see images with changing aspect ratios and resolutions. This code turns RoPE from a fixed lookup table into coordinates generated for the current `H, W`, with training-time perturbations that keep the model from overfitting to one exact spatial scale.

## 代码 / The code

`facebookresearch/dinov3` — [`dinov3/layers/rope_position_encoding.py`](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/layers/rope_position_encoding.py#L16-L121)

```python
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
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
        coords = coords.flatten(0, 1)
        coords = 2.0 * coords - 1.0

        if self.training and self.shift_coords is not None:
            shift_hw = torch.empty(2, **dd).uniform_(-self.shift_coords, self.shift_coords)
            coords += shift_hw[None, :]

        if self.training and self.jitter_coords is not None:
            jitter_max = np.log(self.jitter_coords)
            jitter_min = -jitter_max
            jitter_hw = torch.empty(2, **dd).uniform_(jitter_min, jitter_max).exp()
            coords *= jitter_hw[None, :]

        if self.training and self.rescale_coords is not None:
            rescale_max = np.log(self.rescale_coords)
            rescale_min = -rescale_max
            rescale_hw = torch.empty(1, **dd).uniform_(rescale_min, rescale_max).exp()
            coords *= rescale_hw

        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2)
        angles = angles.tile(2)
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        return (sin, cos)

    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype
        if self.base is not None:
            periods = self.base ** (
                2 * torch.arange(self.D_head // 4, device=device, dtype=dtype) / (self.D_head // 2)
            )
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, self.D_head // 4, device=device, dtype=dtype)
            periods = base**exponents
            periods = periods / base
            periods = periods * self.max_period
        self.periods.data = periods
```

## 逐行讲解 / What's happening

1. **第 33-36 行 / Lines 33-36**: 中文: `embed_dim` 必须能分给每个 head 的 x/y 两个轴和 sin/cos 两半；周期参数只能用 `base` 或 `min_period/max_period` 二选一。 / English: `embed_dim` must divide cleanly across heads, axes, and sin/cos halves; period configuration is kept unambiguous.
2. **第 63-78 行 / Lines 63-78**: 中文: 三种归一化方式决定长边、短边或各轴单独缩放，随后网格坐标被摊平成 `HW` 个 token。 / English: The normalization mode decides whether the long side, short side, or each axis sets the scale, then the grid is flattened into `HW` tokens.
3. **第 80-97 行 / Lines 80-97**: 中文: 训练时的 shift、jitter、rescale 都发生在坐标上，不改模型权重。 / English: Shift, jitter, and rescale perturb the coordinates during training without changing the learned parameters.
4. **第 99-106 行 / Lines 99-106**: 中文: 坐标除以周期得到角度，最后输出可以直接旋转 attention head 的 `sin/cos`。 / English: Coordinates divided by periods become angles, and the returned `sin/cos` tables can rotate attention-head channels directly.

## 类比 / The analogy

像给城市地图换比例尺：同一条街可以按“长边对齐”“短边对齐”或“横纵各自对齐”来画。训练时轻轻挪动比例尺，是让司机别只会背一张固定地图。

Think of redrawing a city map with different rulers: one ruler follows the long side, one follows the short side, and one scales x/y separately. Training-time perturbations keep the driver from memorizing a single printed map.

## 自己跑一遍 / Try it yourself

```python
import math

def rope_angles(h, w, periods):
    coords = []
    for y in range(h):
        for x in range(w):
            cy = 2 * ((y + 0.5) / h) - 1
            cx = 2 * ((x + 0.5) / w) - 1
            coords.append((round(cy, 2), round(cx, 2)))
    return [[round(2 * math.pi * c / p, 2) for c in pair for p in periods] for pair in coords]

print(rope_angles(2, 3, [1.0])[:3])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[[-3.14, -4.21], [-3.14, 0.0], [-3.14, 4.21]]
```

第一行三个 patch 的 y 坐标相同，x 坐标从左到右变化；这就是轴向 RoPE 的基本形状。

The first row has the same y coordinate while x changes left to right; that is the basic shape of axial RoPE.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **DiT 固定二维 sin/cos** / **DiT fixed 2D sin/cos**: 同样把二维网格拆成两个频率轴，但通常不做训练期坐标增强。 / It also splits a 2D grid into frequency axes, usually without coordinate augmentation.
- **Wan2.1 3D RoPE** / **Wan2.1 3D RoPE**: 视频模型把轴从 x/y 扩展到 t/h/w。 / Video models extend the axes from x/y to t/h/w.

## 注意事项 / Caveats / when it breaks

- **维度合同很硬** / **The dimension contract is strict**: `embed_dim % (4 * num_heads) == 0` 不满足就没法平均分给轴和 sin/cos。 / If `embed_dim % (4 * num_heads) != 0`, axes and sin/cos halves cannot be allocated cleanly.
- **增强只应训练时开** / **Augmentation belongs to training**: 推理时扰动坐标会让相同图像得到不稳定位置编码。 / Perturbing coordinates at inference makes the same image receive unstable positions.

## 延伸阅读 / Further reading

- [facebookresearch/dinov3 source](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/layers/rope_position_encoding.py#L16-L121)
