---
date: 2026-09-01
topic: robotics
source: trending
repo: Hebbian-Robotics/hflow
file: src/hflow/checks.py
permalink: https://github.com/Hebbian-Robotics/hflow/blob/616494d17916d74adcaa9a9b0dc6a8a60d7fe7dc/src/hflow/checks.py#L79-L142
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, data-quality, motion-profile, intervals]
---

# HFlow motion checks：一次差分，多处复用 / HFlow Motion Checks: Differentiate Once, Reuse the Evidence

> **一句话 / In one line**: HFlow 先把关节位置和时间戳变成统一的运动 profile，再把连续异常 step 合并成可查询的时间区间。 / HFlow first turns joint positions and timestamps into one motion profile, then merges contiguous anomalous steps into queryable time intervals.

## 为什么重要 / Why this matters

机器人数据清洗不能只给一个 pass/fail。你需要知道哪里时间戳倒退、哪里关节速度突变、异常持续多久，以及不同阈值下是否该剔除 episode。HFlow 的做法是先产出事实证据：速度、非正时间间隔计数、异常区间。阈值和是否 reject 留给后面的 gate。

Robotics data curation should not stop at pass/fail. You need to know where timestamps went backward, where joint velocity spiked, how long the anomaly lasted, and whether a given threshold should reject the episode. HFlow first emits factual evidence: velocities, nonpositive time deltas, and anomalous intervals. Thresholding and rejection are left to later gates.

## 代码 / The code

`Hebbian-Robotics/hflow` — [`src/hflow/checks.py`](https://github.com/Hebbian-Robotics/hflow/blob/616494d17916d74adcaa9a9b0dc6a8a60d7fe7dc/src/hflow/checks.py#L79-L142)

```python
@dataclass(frozen=True)
class _JointMotionProfile:
    """Finite-difference motion facts of one state channel, computed once for
    every check that reasons about joint speed."""

    stamps_ns: np.ndarray
    deltas_s: np.ndarray
    per_step_max_speed: np.ndarray  # max over joints of |dq/dt|, one per step
    nonpositive_dt_count: int


def _joint_motion_profile(
    episode: Episode, topic: str, field: str | None
) -> _JointMotionProfile | None:
    """``None`` when the channel has fewer than two messages (no motion to
    profile); callers record the message count and stop."""
    channel = episode.channel(topic)
    positions = channel.to_numpy(field)
    if positions.ndim == 1:
        positions = positions[:, np.newaxis]
    stamps_ns = channel.timestamps
    if len(stamps_ns) < 2:
        return None
    deltas_s = np.diff(stamps_ns) / 1e9
    safe_deltas_s = np.where(deltas_s > 0, deltas_s, np.nan)
    velocities = np.abs(np.diff(positions, axis=0)) / safe_deltas_s[:, np.newaxis]
    return _JointMotionProfile(
        stamps_ns=stamps_ns,
        deltas_s=deltas_s,
        per_step_max_speed=np.nanmax(velocities, axis=1),
        nonpositive_dt_count=int(np.sum(deltas_s <= 0)),
    )


def _mask_run_intervals(
    stamps_ns: np.ndarray,
    step_mask: np.ndarray,
    label: str,
    *,
    min_duration_s: float = 0.0,
) -> list[Interval]:
    """Contiguous True runs of a per-step mask as labeled intervals.

    Step ``i`` spans ``stamps_ns[i]``..``stamps_ns[i + 1]``, so a run of steps
    ``r``..``i - 1`` spans ``stamps_ns[r]``..``stamps_ns[i]``.
    """
    intervals: list[Interval] = []

    def append_run(run_start_index: int, run_end_index: int) -> None:
        start_ns = int(stamps_ns[run_start_index])
        end_ns = int(stamps_ns[run_end_index])
        if (end_ns - start_ns) / 1e9 >= min_duration_s:
            intervals.append(Interval(start_ns=start_ns, end_ns=end_ns, label=label))

    run_start: int | None = None
    for index, in_run in enumerate(step_mask):
        if in_run and run_start is None:
            run_start = index
        elif not in_run and run_start is not None:
            append_run(run_start, index)
            run_start = None
    if run_start is not None:
        append_run(run_start, len(stamps_ns) - 1)
    return intervals
```

## 逐行讲解 / What's happening

1. **第 79-88 行 / Lines 79-88 (`_JointMotionProfile`)**:
   - 中文: dataclass 保存的是可复用事实，不保存某个阈值下的结论。
   - English: The dataclass stores reusable facts, not a threshold-specific decision.
2. **第 95-104 行 / Lines 95-104 (positions and deltas)**:
   - 中文: 关节位置统一成二维，时间戳差转成秒；非正时间间隔被换成 `nan`，避免除以零或负时间。
   - English: Joint positions are normalized to 2D and timestamp deltas become seconds; nonpositive deltas become `nan` to avoid divide-by-zero or negative-time speeds.
3. **第 104-110 行 / Lines 104-110 (max speed evidence)**:
   - 中文: 每一步取所有关节里最大的绝对速度，变成一个易查的 episode 级质量信号。
   - English: Each step keeps the maximum absolute joint speed across joints, yielding a compact episode-quality signal.
4. **第 113-131 行 / Lines 113-131 (interval builder)**:
   - 中文: 布尔 mask 里的连续 True 被合并成一个 `Interval`，并且可以按最短持续时间过滤。
   - English: Contiguous `True` spans in a boolean mask become one `Interval`, optionally filtered by minimum duration.
5. **第 133-142 行 / Lines 133-142 (run state machine)**:
   - 中文: `run_start` 记录异常段起点；False 结束一段，循环结束还要补提交尾段。
   - English: `run_start` records where an anomalous span begins; `False` closes it, and the final open span is committed after the loop.

## 类比 / The analogy

像质检员检查传送带上的零件。她不会只说“这箱不好”，而是先记录每个零件经过的时间、偏差最大的位置、连续出问题的时间段，最后再按客户标准判定。

It is like an inspector watching parts on a conveyor belt. She does not only say "this box is bad"; she records timestamps, largest deviations, and continuous bad spans, then applies the customer's threshold later.

## 自己跑一遍 / Try it yourself

```python
stamps = [0, 1_000_000_000, 2_000_000_000, 4_000_000_000]
positions = [0.0, 0.2, 2.2, 2.4]
deltas = [(b - a) / 1e9 for a, b in zip(stamps, stamps[1:])]
speeds = [abs(b - a) / dt for a, b, dt in zip(positions, positions[1:], deltas)]
mask = [speed > 1.0 for speed in speeds]
intervals = []
start = None
for i, bad in enumerate(mask):
    if bad and start is None:
        start = i
    elif not bad and start is not None:
        intervals.append((stamps[start], stamps[i]))
        start = None
if start is not None:
    intervals.append((stamps[start], stamps[-1]))
print(speeds)
print(intervals)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[0.2, 2.0, 0.09999999999999987]
[(1000000000, 2000000000)]
```

中文: 速度尖峰被保留为一个具体时间段，而不是丢成一个模糊的失败标签。

English: The velocity spike becomes a concrete time span, not a vague failure label.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **dataset curation manifests** / **Dataset curation manifests**: 中文: 先产出证据表，再用 SQL 和阈值选择训练集。 / English: First produce evidence tables, then use SQL and thresholds to select training data.
- **robot safety monitors** / **Robot safety monitors**: 中文: 在线安全系统也会把连续异常合并成事件，方便报警和复盘。 / English: Online safety systems also merge contiguous anomalies into events for alerting and review.

## 注意事项 / Caveats / when it breaks

- **`nan` 不是 verdict / `nan` is not a verdict**: 中文: 非正时间间隔让速度变 `nan`，调用方还要单独记录和解释。 / English: Nonpositive time deltas make speed `nan`; callers still need to record and explain them.
- **最大关节速度会隐藏维度 / Max joint speed hides dimensions**: 中文: 聚合后很适合筛选，但定位具体哪个关节还需要附加证据。 / English: The aggregate is good for filtering, but identifying the exact joint needs extra evidence.

## 延伸阅读 / Further reading

- HFlow checks source: https://github.com/Hebbian-Robotics/hflow/blob/616494d17916d74adcaa9a9b0dc6a8a60d7fe7dc/src/hflow/checks.py
