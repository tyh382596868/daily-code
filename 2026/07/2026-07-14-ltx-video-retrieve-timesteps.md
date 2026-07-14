---
date: 2026-07-14
topic: diffusion
source: trending
repo: Lightricks/LTX-Video
file: ltx_video/pipelines/pipeline_ltx_video.py
permalink: https://github.com/Lightricks/LTX-Video/blob/4b2d053057623ddd4d0a1d3e9cd28890e9ef487f/ltx_video/pipelines/pipeline_ltx_video.py#L135-L205
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, video-generation, scheduler, timesteps]
---

# LTX-Video retrieve_timesteps：自定义采样步也要重设 scheduler / LTX-Video retrieve_timesteps: Custom Steps Still Reset the Scheduler

> **一句话 / In one line**: LTX-Video 先让 scheduler 生成 timesteps，再按跳过首尾步裁剪，并把裁剪后的列表重新写回 scheduler。 / LTX-Video asks the scheduler for timesteps, trims skipped head/tail steps, then writes the trimmed list back into the scheduler.

## 为什么重要 / Why this matters

视频生成常需要跳过早期或末尾采样步：图生视频可能从中等噪声开始，快速预览可能提前结束。只裁剪本地 `timesteps` 不够，因为 scheduler 内部状态仍然以旧列表为准。这段代码的关键是“裁剪后再 `set_timesteps` 一次”。

Video generation often skips early or late sampling steps: image-to-video may start from medium noise, and previews may stop early. Trimming a local `timesteps` list is not enough because the scheduler still holds the old internal state. The key move here is calling `set_timesteps` again after trimming.

## 代码 / The code

`Lightricks/LTX-Video` — [`ltx_video/pipelines/pipeline_ltx_video.py`](https://github.com/Lightricks/LTX-Video/blob/4b2d053057623ddd4d0a1d3e9cd28890e9ef487f/ltx_video/pipelines/pipeline_ltx_video.py#L135-L205)

```python
def retrieve_timesteps(
    scheduler,
    num_inference_steps=None,
    device=None,
    timesteps=None,
    skip_initial_inference_steps=0,
    skip_final_inference_steps=0,
    **kwargs,
):
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(...)
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps

        if (
            skip_initial_inference_steps < 0
            or skip_final_inference_steps < 0
            or skip_initial_inference_steps + skip_final_inference_steps >= num_inference_steps
        ):
            raise ValueError(...)

        timesteps = timesteps[
            skip_initial_inference_steps : len(timesteps) - skip_final_inference_steps
        ]
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        num_inference_steps = len(timesteps)

    return timesteps, num_inference_steps
```

## 逐行讲解 / What's happening

1. **显式 timesteps 先验能力检查 / Explicit timesteps need capability checking**:
   - 中文: 不是每个 scheduler 都支持传入自定义列表，所以用 `inspect.signature` 先看参数。
   - English: Not every scheduler accepts a custom list, so `inspect.signature` checks support first.
2. **默认路径先生成完整列表 / Default path builds the full list first**:
   - 中文: 没有自定义列表时，先调用 scheduler 原生 spacing 策略。
   - English: Without a custom list, the scheduler's native spacing strategy is used first.
3. **跳步合法性 / Skip validation**:
   - 中文: 首尾跳过步数不能为负，也不能把整个采样序列删光。
   - English: Head/tail skips cannot be negative or remove the entire sampling sequence.
4. **裁剪后重新设置 / Reset after trimming**:
   - 中文: 裁剪后的 `timesteps` 再写回 scheduler，保证内部 sigmas/order 等状态同步。
   - English: The trimmed `timesteps` are written back so internal sigmas/order state stays in sync.

## 类比 / The analogy

像导航先规划完整路线，再决定跳过前两站。你不能只在纸上划掉站名，还要让导航系统重新按剩下路线计算到站时间。

It is like planning a full route and then skipping the first two stops. You cannot only cross out stops on paper; the navigation system must recalculate using the remaining route.

## 自己跑一遍 / Try it yourself

```python
def retrieve(steps, skip_start, skip_end):
    if skip_start < 0 or skip_end < 0 or skip_start + skip_end >= len(steps):
        raise ValueError("bad skip")
    trimmed = steps[skip_start: len(steps) - skip_end]
    scheduler_state = {"timesteps": trimmed, "num_steps": len(trimmed)}
    return scheduler_state

print(retrieve([9, 7, 5, 3, 1], 1, 1))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
{'timesteps': [7, 5, 3], 'num_steps': 3}
```

真正重要的是返回列表和 scheduler 内部状态一致，而不是只拿到一个被切片的 Python list。

The important part is consistency between the returned list and the scheduler's internal state, not merely having a sliced Python list.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **image-to-image strength** / **image-to-image strength**: 很多 pipeline 会根据 strength 截掉高噪声或低噪声部分。 / Many pipelines trim high-noise or low-noise regions according to strength.
- **fast preview sampling** / **fast preview sampling**: 预览模式常跳过末尾精修步以换取速度。 / Preview modes often skip late refinement steps for speed.

## 注意事项 / Caveats / when it breaks

- **自定义列表和 scheduler 不兼容 / Custom lists may be unsupported**: 不检查签名会在运行中抛出更难懂的错误。 / Without signature checking, failures are harder to interpret.
- **跳过太多会删空采样 / Too much skipping empties sampling**: 首尾跳步之和必须小于总步数。 / Head and tail skips together must be less than total steps.

## 延伸阅读 / Further reading

- [LTX-Video `retrieve_timesteps`](https://github.com/Lightricks/LTX-Video/blob/4b2d053057623ddd4d0a1d3e9cd28890e9ef487f/ltx_video/pipelines/pipeline_ltx_video.py#L135-L205)
