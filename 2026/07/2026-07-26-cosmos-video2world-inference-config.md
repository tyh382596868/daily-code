---
date: 2026-07-26
topic: diffusion
source: tracked
repo: nvidia-cosmos/cosmos-predict2
file: examples/video2world.py
permalink: https://github.com/nvidia-cosmos/cosmos-predict2/blob/main/examples/video2world.py
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, video2world, inference-contract]
---

# Cosmos Video2World：先把采样任务写成契约 / Cosmos Video2World: Write the Sampling Task as a Contract First

> **一句话 / In one line**: Video2World 推理入口把 prompt、条件帧、采样步数和输出路径收束成一个明确任务。 / The Video2World entry point turns prompt text, conditional frames, sampling steps, and output paths into one explicit job contract.

## 为什么重要 / Why this matters

世界模型推理最容易乱在“参数散落各处”：prompt 在命令行里，视频路径在数据加载器里，采样步在 scheduler 里，输出路径又在保存逻辑里。这个入口的价值不是某个数学公式，而是把一次视频续写任务的边界先写清楚。

World-model inference often gets messy because parameters live in different places: prompts in CLI flags, videos in loaders, steps in schedulers, and outputs in save code. This entry is useful because it makes the boundary of one video continuation job explicit before any heavy model call runs.

## 代码 / The code

`nvidia-cosmos/cosmos-predict2` — [`examples/video2world.py`](https://github.com/nvidia-cosmos/cosmos-predict2/blob/main/examples/video2world.py)

```python
@dataclass
class Video2WorldJob:
    prompt: str
    input_video: str
    output_path: str
    num_conditional_frames: int = 1
    num_video_frames: int = 121
    guidance: float = 7.0
    num_steps: int = 35


def build_video2world_request(args):
    return Video2WorldJob(
        prompt=args.prompt,
        input_video=args.input_video,
        output_path=args.output_path,
        num_conditional_frames=args.num_conditional_frames,
        num_video_frames=args.num_video_frames,
        guidance=args.guidance,
        num_steps=args.num_steps,
    )


def run_video2world(pipeline, job: Video2WorldJob):
    conditioning = pipeline.load_conditioning_video(
        job.input_video,
        num_frames=job.num_conditional_frames,
    )
    samples = pipeline.generate(
        prompt=job.prompt,
        video_condition=conditioning,
        num_frames=job.num_video_frames,
        num_steps=job.num_steps,
        guidance=job.guidance,
    )
    pipeline.save_video(samples, job.output_path)
```

## 逐行讲解 / What's happening

1. **`Video2WorldJob`**
   - 中文: 把一次采样的所有外部输入放进一个对象，后面的函数只接收这个对象，避免隐式全局状态。
   - English: It packages every external input for one sampling run into a single object, so later code does not depend on scattered globals.
2. **`num_conditional_frames` 与 `num_video_frames`**
   - 中文: 前者决定看多少历史帧，后者决定生成多长的未来片段，两者分开能避免把条件窗口误当成输出长度。
   - English: One controls the observed history window, the other controls generated duration; keeping them separate prevents a common off-by-window bug.
3. **`pipeline.generate(...)`**
   - 中文: sampler 只看到已经整理好的 prompt、条件视频和采样超参，推理核心因此可以保持纯粹。
   - English: The sampler receives normalized prompt, video condition, and scheduler settings, keeping the inference core focused.

## 类比 / The analogy

这像摄影棚的通告单：演员、场景、镜头长度、交片位置都先写在一张纸上，摄影师拿到以后只负责执行，不用边拍边问制片人。

Think of it as a film-shoot call sheet: cast, location, shot length, and delivery path are written down first, so the camera crew can execute without chasing producers mid-shot.

## 自己跑一遍 / Try it yourself

```python
from dataclasses import dataclass

@dataclass
class Job:
    prompt: str
    cond_frames: int
    out_frames: int
    steps: int

def plan(job):
    return [f"condition:{i}" for i in range(job.cond_frames)] + [f"denoise:{s}" for s in range(job.steps)] + [f"save:{job.out_frames}"]

print(plan(Job("robot pushes cube", 2, 8, 3)))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
['condition:0', 'condition:1', 'denoise:0', 'denoise:1', 'denoise:2', 'save:8']
```

中文: 注意条件帧和输出帧在计划里是两种不同角色。
English: Notice that conditional frames and output frames play different roles in the plan.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Diffusers pipelines** / **Diffusers pipelines**: 中文: `__call__` 先规范化 prompt、timesteps 和 latents。 / English: `__call__` normalizes prompts, timesteps, and latents before denoising.
- **Wan2.1 sampler** / **Wan2.1 sampler**: 中文: 先从帧数推 latent 形状，再进入去噪循环。 / English: It derives latent shape from frame count before entering the denoising loop.

## 注意事项 / Caveats / when it breaks

- **配置不是验证** / **Configuration is not validation**: 中文: 生产代码还要检查路径、分辨率、fps 和模型最大帧数。 / English: Production code must still validate paths, resolution, FPS, and model frame limits.
- **条件窗口太短** / **Short condition windows**: 中文: 只给一帧会弱化速度信息。 / English: A single frame weakens motion cues.

## 延伸阅读 / Further reading

- [nvidia-cosmos/cosmos-predict2](https://github.com/nvidia-cosmos/cosmos-predict2)
- [Cosmos Predict2 examples](https://github.com/nvidia-cosmos/cosmos-predict2/tree/main/examples)
