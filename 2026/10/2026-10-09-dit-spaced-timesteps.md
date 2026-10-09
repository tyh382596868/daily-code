---
date: 2026-10-09
topic: diffusion
source: tracked
repo: facebookresearch/DiT
file: diffusion/respace.py
permalink: https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/diffusion/respace.py#L12-L62
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, timestep-respacing]
---
# DiT timestep respacing：少采样几步，也要覆盖整条噪声路 / DiT Timestep Respacing: Fewer Steps, Still Cover the Whole Noise Path

> **一句话 / In one line**: 把原始 diffusion 时间轴切成几段，每段按比例抽点，得到一组更短但分布均匀的采样步。 / Split the original diffusion timeline into sections, then sample each section with a fractional stride.

## 为什么重要 / Why this matters

扩散模型训练常常有 1000 个噪声步，但推理时你不一定愿意走满 1000 步。这段 `space_timesteps` 的价值在于：它不是简单地从头到尾每隔 N 步取一个点，而是允许你给不同区段分配不同预算，例如前 100 步取 10 个点，中间取 15 个点，末尾取 20 个点。这样采样器可以更细地照顾某些噪声区间。

Diffusion models are often trained on a long noise schedule, but inference needs a shorter route. `space_timesteps` turns that long schedule into a smaller set of retained steps while still spreading samples across the full process. The section interface is small, but it encodes a useful design idea: inference speed is a schedule design problem, not just a loop-count knob.

## 代码 / The code

`facebookresearch/DiT` — [`diffusion/respace.py`](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/diffusion/respace.py#L12-L62)

```python
def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)
```

## 逐行讲解 / What's happening

1. **第 31-40 行 / Lines 31-40 (`section_counts`)**:
   - 中文: 字符串输入先被解析；`ddimN` 是特殊模式，会寻找一个整数 stride，刚好留下 N 个时间步。
   - English: String input is normalized first. The `ddimN` branch searches for an integer stride that produces exactly N retained steps.
2. **第 41-47 行 / Lines 41-47 (section sizing)**:
   - 中文: 原始时间轴被平均切成若干段，余数交给靠前的段；如果某段比想抽的点还短，直接报错。
   - English: The original timeline is divided into sections, with extra steps assigned to early sections. A section cannot request more retained steps than it contains.
3. **第 51-60 行 / Lines 51-60 (`frac_stride`)**:
   - 中文: 每段内部用浮点步长前进，再 `round` 到整数时间步，所以首尾都能被覆盖。
   - English: Each section advances by a fractional stride and rounds to integer timesteps, which keeps both ends of the section represented.

## 类比 / The analogy

像把一部长电影剪成预告片。你不会只从开头连续截 30 秒，而是从开场、冲突、高潮、结尾各挑几帧，让观众仍然看懂完整弧线。

It is like cutting a movie trailer. You do not take one continuous slice from the beginning; you sample the opening, conflict, climax, and ending so the shorter version still preserves the story arc.

## 自己跑一遍 / Try it yourself

```python
def space_timesteps(num_timesteps, section_counts):
    if isinstance(section_counts, str):
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx, all_steps = 0, []
    for i, count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        stride = 1 if count <= 1 else (size - 1) / (count - 1)
        cur = 0.0
        for _ in range(count):
            all_steps.append(start_idx + round(cur))
            cur += stride
        start_idx += size
    return sorted(set(all_steps))

print(space_timesteps(30, "3,4,3"))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
[0, 4, 9, 10, 13, 16, 19, 20, 24, 29]
```

注意输出不是固定间隔，而是在三个区段里分别均匀取点。

Notice that the result is not one global stride; it is evenly sampled within each section.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **DDIM / DDPM sampler** / **DDIM / DDPM samplers**: 采样器经常把长训练时间表压成短推理时间表。 / Samplers often compress the long training schedule into a shorter inference schedule.
- **视频扩散** / **Video diffusion**: 高分辨率视频推理更贵，分段预算能把更多步留给敏感区间。 / Video generation is expensive, so section budgets let you spend more steps where they matter.

## 注意事项 / Caveats / when it breaks

- **重复时间步** / **Duplicate timesteps**: `round` 后再转成 `set`，极端配置可能减少实际步数。 / Rounding plus `set` can reduce the effective count in edge cases.
- **区段语义要自己定** / **Section meaning is manual**: 代码只负责抽点，不知道哪个噪声区间最重要。 / The function samples steps; it does not know which noise region deserves more budget.

## 延伸阅读 / Further reading

- [DiT `respace.py`](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/diffusion/respace.py#L12-L62)
- [DDIM paper](https://arxiv.org/abs/2010.02502)
