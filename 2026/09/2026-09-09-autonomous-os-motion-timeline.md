---
date: 2026-09-09
topic: diffusion
source: trending
repo: autonomous-ai/autonomous-os
file: hal/drivers/motors/recording_timing.py
permalink: https://github.com/autonomous-ai/autonomous-os/blob/1fd93a7e0e78f3f9d9ca603f56c157589e7676d0/hal/drivers/motors/recording_timing.py#L51-L126
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, robotics, motion-safety, resampling]
---

# Autonomous OS motion timeline：超速片段自动拉长 / Autonomous OS Motion Timeline: Stretch Only the Unsafe Segments

> **一句话 / In one line**: 这段代码不改动作目标，只拉长超过舵机能力的时间间隔，再把轨迹重采样到播放频率。 / The code keeps joint targets intact, stretches only intervals beyond the servo limit, and then resamples the trajectory to the playback rate.

## 为什么重要 / Why this matters

中文：机器人动作录制通常来自一个时间网格，但真实硬件有速度上限。简单地把整段动作慢放会改变用户设计的节奏；完全不处理又会让舵机落后、追赶甚至产生机械冲击。Autonomous OS 的做法是逐段计算最大关节位移，只拉长真正超速的 segment，其他时间保持原样。

English: Recorded robot motions live on an authored time grid, but hardware has a maximum joint speed. Slowing the whole clip changes its intended rhythm; ignoring the limit makes servos lag and snap. Autonomous OS computes the largest joint displacement per segment and stretches only the intervals that exceed the limit.

## 代码 / The code

`autonomous-ai/autonomous-os` — [`hal/drivers/motors/recording_timing.py`](https://github.com/autonomous-ai/autonomous-os/blob/1fd93a7e0e78f3f9d9ca603f56c157589e7676d0/hal/drivers/motors/recording_timing.py#L51-L126)

```python
def stretch_timeline(
    times: List[float], frames: List[Dict[str, float]], policy: Any = None
) -> List[float]:
    """Widen the gaps that demand more joint speed than the servo can deliver.

    Returns a new, still-monotonic time axis. Only over-speed segments grow;
    everything else keeps its authored timing, so a recording slows down
    exactly where it was impossible and nowhere else.
    """
    max_dps = effective_max_dps(policy)
    if max_dps <= 0:
        return times

    out = [times[0]]
    for i in range(1, len(frames)):
        authored_dt = max(times[i] - times[i - 1], 1e-3)
        peak_delta = max(
            (abs(frames[i][j] - frames[i - 1][j]) for j in frames[i]),
            default=0.0,
        )
        needed_dt = peak_delta / max_dps
        out.append(out[-1] + max(authored_dt, needed_dt))
    return out


def resample_recording(
    times: List[float],
    frames: List[Dict[str, float]],
    name: str,
    fps: float,
    policy: Any = None,
    geometry: Any = None,
) -> List[Dict[str, float]]:
    """Put frames on a playback loop's own 1/fps grid.

    The loop steps exactly one frame per tick, so a list sampled at fps plays at
    real time by construction — no timing logic in the hot path.

    Also the gate for whole-body stability: raises if a pose reaches far enough
    off the base axis to tip the body, per the body's own declared ceiling and
    geometry. Checked here because both motion drivers come through this
    function, so the simulator refuses the same clip a body would. Resampling
    only stretches time, never moves a joint, so checking the authored frames
    covers the played ones.
    """
    check_stable(frames, name, policy, geometry)
    stretched = stretch_timeline(times, frames, policy)
    duration = stretched[-1] - stretched[0]
    if duration <= 0:
        return frames

    joints = list(frames[0].keys())
    step = 1.0 / fps
    total = max(1, int(round(duration / step)))

    out: List[Dict[str, float]] = []
    src = 0
    for k in range(total + 1):
        t = stretched[0] + min(k * step, duration)
        # stretched[] is monotonic and t only advances, so this walk is O(n).
        while src < len(stretched) - 2 and stretched[src + 1] < t:
            src += 1
        span = stretched[src + 1] - stretched[src]
        p = 0.0 if span <= 0 else (t - stretched[src]) / span
        p = max(0.0, min(1.0, p))
        a, b = frames[src], frames[src + 1]
        out.append({j: a[j] + (b[j] - a[j]) * p for j in joints})

    authored = times[-1] - times[0]
    max_dps = effective_max_dps(policy)
    if max_dps > 0 and duration > authored * 1.01:
        logger.info(
            "recording %r stretched %.2fs -> %.2fs to stay under %.0f deg/s",
            name, authored, duration, max_dps,
        )
    return out
```

## 逐行讲解 / What's happening

1. **第 60-73 行 / Lines 60-73 (`stretch_timeline`)**:
   - 中文: 每段取所有关节位移的最大值，计算达到该位移所需的最短时间，并与原始时间间隔取最大值。
   - English: Each segment uses the largest joint displacement, derives the minimum safe duration, and keeps the larger of authored and required time.
2. **第 64-72 行 / Lines 64-72**:
   - 中文: `out` 保持单调递增；只要某一段安全，原始节奏就原样保留。
   - English: `out` remains monotonic, and safe segments preserve their authored timing exactly.
3. **第 96-117 行 / Lines 96-117 (`resample_recording`)**:
   - 中文: 先做全身稳定性检查，再把拉长后的时间轴映射到固定 `1 / fps` 网格，并用线性插值生成每个播放帧。
   - English: Stability is checked first, then the stretched timeline is sampled on a fixed `1 / fps` grid with linear interpolation.
4. **第 120-126 行 / Lines 120-126**:
   - 中文: 如果总时长被拉长，就记录一条日志，便于发现哪些录音触发了硬件限速。
   - English: A log is emitted when total duration grows, making hardware-limited recordings observable.

## 类比 / The analogy

中文：像把一段舞蹈录像放进只能按固定速度转动的唱片机。慢动作只加在舞者突然跨大步的地方，其他节拍不动。

English: It is like playing a dance recording on a turntable with a speed limit. Slow motion is inserted only where the dancer takes an oversized step; the rest of the rhythm stays intact.

## 自己跑一遍 / Try it yourself

```python
def stretch(times, frames, max_speed):
    out = [times[0]]
    for i in range(1, len(frames)):
        authored = max(times[i] - times[i - 1], 1e-3)
        peak = max(abs(frames[i][j] - frames[i - 1][j]) for j in frames[i])
        out.append(out[-1] + max(authored, peak / max_speed))
    return out

times = [0.0, 0.1, 0.2]
frames = [{"joint": 0.0}, {"joint": 5.0}, {"joint": 5.5}]
print(stretch(times, frames, max_speed=25.0))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[0.0, 0.2, 0.30000000000000004]
```

中文：第一段原本要求 `5 / 0.1 = 50` 单位每秒，所以被拉到 `0.2` 秒；第二段没有超速，保留原节奏。

English: The first segment demands `5 / 0.1 = 50` units per second, so it is stretched to `0.2` seconds. The second segment is safe and keeps its original pace.

## 注意事项 / Caveats / when it breaks

- **它只保证速度，不自动保证加速度舒适** / **It limits speed, not acceleration comfort**: 真实执行器还可能需要 jerk、扭矩和碰撞约束。
- **插值不会改变原始关节端点** / **Interpolation does not change authored endpoints**: 如果端点本身越过机械限位，需要单独做位置裁剪或拒绝。
- **`fps` 必须为正数** / **`fps` must be positive**: 否则 `1.0 / fps` 无法定义。

## 延伸阅读 / Further reading

- [Autonomous OS recording_timing.py](https://github.com/autonomous-ai/autonomous-os/blob/1fd93a7e0e78f3f9d9ca603f56c157589e7676d0/hal/drivers/motors/recording_timing.py)
- [Autonomous OS repository](https://github.com/autonomous-ai/autonomous-os)
