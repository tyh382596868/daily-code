---
date: 2026-09-12
topic: diffusion
source: tracked
repo: hpcaitech/Open-Sora
file: opensora/utils/sampling.py
permalink: https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/utils/sampling.py#L295-L332
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, flow-matching, timestep-schedule, video-resolution]
---

# Open-Sora 时间表：分辨率和帧数一起改变采样节奏 / Open-Sora Timesteps: Let Resolution and Frame Count Change the Sampling Pace

> **一句话 / In one line**: Open-Sora 先生成标准的线性时间表，再用空间 token 数和视频帧数共同决定一个 `shift_alpha`。 / Open-Sora first builds a linear timestep grid, then lets spatial token count and frame count determine a `shift_alpha`.

## 为什么重要 / Why this matters

中文：视频扩散不是把图片的采样时间表直接复制到更长的视频上。视频的 latent token 数会随着分辨率和帧数一起增长，模型在高信号区域需要更合适的时间分布。`get_schedule` 用一个很小的函数，把“采样多少步”和“每一步落在哪里”分开，再用分辨率与时间长度去调整后者。

English: Video diffusion should not blindly reuse an image schedule for a longer clip. The latent token count grows with both spatial resolution and frame count, so the model benefits from a schedule that spends time differently. `get_schedule` keeps the number of steps separate from the locations of those steps, then adjusts the latter using spatial and temporal scale.

## 代码 / The code

`hpcaitech/Open-Sora` — [`opensora/utils/sampling.py`](https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/utils/sampling.py#L295-L332)

```python
def time_shift(alpha: float, t: Tensor) -> Tensor:
    return alpha * t / (1 + (alpha - 1) * t)


def get_res_lin_function(
    x1: float = 256, y1: float = 1, x2: float = 4096, y2: float = 3
) -> callable:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    num_frames: int,
    shift_alpha: float | None = None,
    base_shift: float = 1,
    max_shift: float = 3,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        if shift_alpha is None:
            # estimate mu based on linear estimation between two points
            # spatial scale
            shift_alpha = get_res_lin_function(y1=base_shift, y2=max_shift)(
                image_seq_len
            )
            # temporal scale
            shift_alpha *= math.sqrt(num_frames)
        # calculate shifted timesteps
        timesteps = time_shift(shift_alpha, timesteps)

    return timesteps.tolist()
```

## 逐行讲解 / What's happening

1. **第 295-296 行 / Lines 295-296 (`time_shift`)**:
   - 中文：这个分式把 `[0, 1]` 里的时间点重新映射。`alpha > 1` 时，靠近高噪声端的时间会被重新拉开，改变采样密度但不改变起点和终点。
   - English: The rational map remaps points inside `[0, 1]`. When `alpha > 1`, it changes how much resolution the schedule gives to the high-noise side while preserving the endpoints.
2. **第 299-304 行 / Lines 299-304 (`get_res_lin_function`)**:
   - 中文：空间规模没有直接做复杂拟合，而是在 `(256, 1)` 和 `(4096, 3)` 两个锚点之间线性插值，返回一个稍后可以调用的函数。
   - English: Spatial scale is not fitted with a complicated model. The code linearly interpolates between `(256, 1)` and `(4096, 3)` and returns a callable.
3. **第 307-317 行 / Lines 307-317 (`torch.linspace`)**:
   - 中文：`num_steps + 1` 很关键，因为每个 Euler/flow step 需要一对 `(t_curr, t_prev)`，最后的 `0` 是显式终点。
   - English: `num_steps + 1` matters because each Euler/flow step needs a pair `(t_curr, t_prev)`. The final `0` is an explicit endpoint.
4. **第 320-328 行 / Lines 320-328 (`shift_alpha`)**:
   - 中文：如果调用方没有提供固定的 `shift_alpha`，代码先根据空间 token 数估计，再乘上 `sqrt(num_frames)`，让更长的视频也影响时间表。
   - English: When the caller does not provide a fixed `shift_alpha`, the code estimates it from spatial token count and then multiplies by `sqrt(num_frames)`, so longer videos also affect the schedule.
5. **第 329-332 行 / Lines 329-332 (the final remap)**:
   - 中文：整个时间表一次性变换，随后 sampler 只消费这个 list，不需要知道 shift 是怎样算出来的。
   - English: The whole grid is transformed once. The sampler consumes the resulting list without knowing how the shift was derived.

## 类比 / The analogy

中文：像给不同长度的登山路线安排路标。短路线可以均匀放路标；路线变长后，你会在最容易迷路的上坡段多放一些，而不是把所有路标简单拉伸。

English: Imagine placing markers along hiking routes of different lengths. A short route can use evenly spaced markers, but a longer route needs more attention around the steep, confusing sections instead of merely stretching the old map.

## 自己跑一遍 / Try it yourself

```python
import math

def schedule(steps, image_tokens, frames):
    alpha = 1 + (3 - 1) * (image_tokens - 256) / (4096 - 256)
    alpha *= math.sqrt(frames)
    raw = [1 - i / steps for i in range(steps + 1)]
    return [alpha * t / (1 + (alpha - 1) * t) for t in raw]

print([round(t, 3) for t in schedule(4, 256, 1)])
print([round(t, 3) for t in schedule(4, 4096, 16)])
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
[1.0, 0.75, 0.5, 0.25, 0.0]
[1.0, 0.973, 0.923, 0.8, 0.0]
```

中文：第二组时间点更靠近高值端，说明更大的空间和时间规模会改变采样节奏，而不是改变总步数。

English: The second schedule stays higher for longer, showing that larger spatial and temporal scale changes the pacing without changing the number of steps.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Wan2.1 dynamic shift** / **Wan2.1 dynamic shift**: 用 `time_shift` 让 flow-matching 时间轴对分辨率更敏感。 / Uses a similar `time_shift` idea to make the flow-matching timeline resolution-aware.
- **Diffusers FlowMatch schedulers** / **Diffusers FlowMatch schedulers**: 把 `sigma`、`timestep` 和一步更新的速度场拆开。 / Separate `sigma`, `timestep`, and the velocity-field update.
- **MIRA sampling grids** / **MIRA sampling grids**: 根据动作与视频任务调整采样点密度，而不是只改模型输出。 / Adjust sample-point density for the task instead of changing only the model output.

## 注意事项 / Caveats / when it breaks

- **`image_seq_len` 必须和 latent token 化一致** / **`image_seq_len` must match latent tokenization**: 如果 patch size 或 VAE 压缩率变了，旧的尺度锚点就不再表达真实计算量。
- **`sqrt(num_frames)` 是经验缩放** / **`sqrt(num_frames)` is an empirical scale**: 它不是物理定律，换数据分布或 sampler 后需要重新验证。
- **不要把 shift 和 CFG 混为一谈** / **Do not confuse shift with CFG**: shift 改时间网格，CFG 改条件预测的组合方式。

## 延伸阅读 / Further reading

- [Open-Sora sampling utilities](https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/utils/sampling.py)
- [Wan2.1 flow-match dynamic shift](https://github.com/Wan-Video/Wan2.1)
