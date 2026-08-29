---
date: 2026-08-29
topic: robotics
source: tracked
repo: huggingface/lerobot
file: src/lerobot/rollout/strategies/core.py
permalink: https://github.com/huggingface/lerobot/blob/4aaff99be4a1d81568c08c8f0296b41b40c99ec4/src/lerobot/rollout/strategies/core.py#L33-L47
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, rollout, video-budget]
---

# LeRobot video budget：先估时长，再管存储 / LeRobot Video Budget: Estimate Time Before Managing Storage

> **一句话 / In one line**: LeRobot 用控制频率、episode 数量和每帧大小估算录制视频的上限，让 rollout 在开始前就知道磁盘压力。 / LeRobot estimates the upper bound of recorded video from control frequency, episode count, and frame size so rollout code knows the storage pressure before it starts.

## 为什么重要 / Why this matters

机器人数据采集不是只跑 policy，还要持续写视频。如果录制参数没有提前换算成秒数和容量，采集跑到一半才发现磁盘不够，前面的 episode 也可能白跑。

Robot collection is not just policy execution; it continuously writes video. If recording settings are not translated into seconds and capacity up front, a job may discover storage pressure mid-run and waste earlier episodes.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/rollout/strategies/core.py`](https://github.com/huggingface/lerobot/blob/4aaff99be4a1d81568c08c8f0296b41b40c99ec4/src/lerobot/rollout/strategies/core.py#L33-L47)

```python
def estimate_max_episode_seconds(
    env: Robot,
    episodes: int | None,
    episode_time_s: int | float | None,
    max_steps_per_episode: int | None,
) -> int:
    """Estimate maximum episode seconds."""
    if episodes and episode_time_s:
        return int(episodes * episode_time_s)
    if episodes and max_steps_per_episode:
        return int(episodes * max_steps_per_episode / env.fps)
    return 1


def estimate_max_videos_size_gb(max_episode_seconds: int, video_size_per_frame_gb: float, fps: int) -> float:
    """Estimate maximum videos size in GB."""
    return max_episode_seconds * fps * video_size_per_frame_gb
```

## 逐行讲解 / What's happening

1. **第 39-42 行 / Lines 39-42**:
   - 中文: 如果调用方直接给了 episode 数和每个 episode 的秒数，就用最清楚的上界。
   - English: If the caller provides episode count and seconds per episode, the function uses that direct upper bound.
2. **第 43-44 行 / Lines 43-44**:
   - 中文: 如果只有 step 数，就除以 `env.fps` 把控制步转换成真实时间。
   - English: If only step count is available, it divides by `env.fps` to convert control ticks into wall-clock recording time.
3. **第 45-47 行 / Lines 45-47**:
   - 中文: 没有足够信息时返回一个保守的非零秒数；视频容量再由秒数、fps 和单帧大小相乘得到。
   - English: With insufficient information it returns a small non-zero fallback; video budget is then seconds times fps times per-frame size.

## 类比 / The analogy

这像出门拍延时摄影前先算电池和存储卡：每秒几张、拍多久、每张多大，三个数乘起来就是你能不能拍完整段。

It is like checking battery and SD-card capacity before a time-lapse shoot: frames per second, duration, and size per frame decide whether the whole run fits.

## 自己跑一遍 / Try it yourself

```python
class Env:
    fps = 20

def seconds(episodes=None, episode_time_s=None, max_steps=None):
    if episodes and episode_time_s:
        return int(episodes * episode_time_s)
    if episodes and max_steps:
        return int(episodes * max_steps / Env.fps)
    return 1

video_size_per_frame_gb = 0.000002
budget = seconds(episodes=3, max_steps=200) * Env.fps * video_size_per_frame_gb
print(seconds(episodes=3, max_steps=200))
print(round(budget, 3))
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
30
0.001
```

这个 toy 例子把 step 预算转换成视频秒数，核心就是不要把控制步和真实时间混在一起。

The toy example converts a step budget into video seconds. The key is not to confuse control ticks with real time.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **ROS bag recorder** / **ROS bag recorder**: 录制前常用话题频率估算 bag 大小。 / Recorders often estimate bag size from topic frequency before capture.
- **dataset sharding** / **dataset sharding**: 先估样本大小，再决定 shard 数量。 / Dataset writers estimate sample size before choosing shard counts.
- **training checkpointing** / **training checkpointing**: 先算 checkpoint 频率和大小，再设保留策略。 / Checkpoint systems combine save frequency and file size before setting retention.

## 注意事项 / Caveats / when it breaks

- **fps 必须真实** / **FPS must be real**: 配置 fps 和相机实际 fps 不一致会低估容量。 / A configured FPS that differs from actual camera FPS underestimates storage.
- **单帧大小会变** / **Frame size changes**: 压缩格式、分辨率和场景内容都会改变每帧大小。 / Codec, resolution, and scene content all change per-frame size.
- **fallback 不是容量保证** / **The fallback is not a guarantee**: 返回 `1` 只是让下游不崩，并不是合理预算。 / Returning `1` only keeps downstream code alive; it is not a real budget.

## 延伸阅读 / Further reading

- LeRobot rollout strategies: https://github.com/huggingface/lerobot/blob/main/src/lerobot/rollout/strategies/core.py
- LeRobot docs: https://huggingface.co/docs/lerobot/
