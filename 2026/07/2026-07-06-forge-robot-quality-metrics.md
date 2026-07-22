---
date: 2026-07-06
topic: robotics
source: trending
repo: arpitg1304/forge
file: forge/quality/metrics.py
permalink: https://github.com/arpitg1304/forge/blob/461a0179115c7f2dc763ff4b1a1d2de02f5a1e69/forge/quality/metrics.py#L24-L117
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, robotics, data-quality]
---

# Forge robotics quality metrics：先筛掉坏轨迹，再训练策略 / Forge Robotics Quality Metrics: Filter Bad Episodes Before Training Policies

> **一句话 / In one line**: 这些小函数把 dead action、轨迹 jerk、夹爪抖动变成可量化的数据质量信号。 / These small functions turn dead actions, trajectory jerk, and gripper chatter into measurable data-quality signals.

## 为什么重要 / Why this matters

机器人策略常常不是被模型结构拖垮，而是被坏 demonstration 拖垮：动作全零、时间戳抖动、夹爪疯狂开合。Forge 的 `metrics.py` 值得看，因为它用普通 numpy 函数把这些问题变成可以批处理、可报警、可过滤的指标。

Robot policies are often hurt less by architecture and more by bad demonstrations: all-zero actions, jittery timestamps, or a gripper rapidly opening and closing. Forge's `metrics.py` is useful because plain NumPy functions turn these issues into batchable, alertable, filterable metrics.

## 代码 / The code

`arpitg1304/forge` — [`forge/quality/metrics.py`](https://github.com/arpitg1304/forge/blob/461a0179115c7f2dc763ff4b1a1d2de02f5a1e69/forge/quality/metrics.py#L24-L117)

```python
def dead_action_detection(
    actions: NDArray, config: QualityConfig
) -> tuple[float, list[tuple[int, int]]]:
    eps = config.dead_action_eps

    # Zero actions
    is_zero = np.all(np.abs(actions) < eps, axis=1)

    # Constant actions (same as first timestep)
    is_constant = np.all(np.abs(actions - actions[0:1]) < eps, axis=1)

    is_dead = is_zero | is_constant
    dead_fraction = float(np.mean(is_dead))

    # Find contiguous dead ranges
    dead_ranges = _find_runs(is_dead)

    return dead_fraction, dead_ranges


def log_dimensionless_jerk(
    positions: NDArray, dt: float
) -> float | None:
    if len(positions) < 4 or dt <= 0:
        return None

    velocity = np.gradient(positions, dt, axis=0)
    acceleration = np.gradient(velocity, dt, axis=0)
    jerk = np.gradient(acceleration, dt, axis=0)

    t_total = (len(positions) - 1) * dt
    peak_vel = float(np.max(np.linalg.norm(velocity, axis=1)))

    if peak_vel < 1e-10 or t_total < 1e-10:
        return None

    jerk_magnitude_sq = float(np.sum(jerk**2) * dt)

    ldlj_arg = t_total**3 / peak_vel**2 * jerk_magnitude_sq
    ldlj = -np.log(max(ldlj_arg, 1e-10))

    return float(ldlj)


def gripper_chatter(
    actions: NDArray, duration: float, config: QualityConfig
) -> tuple[int, float, bool]:
    gripper_dim = config.gripper_dim
    gripper = actions[:, gripper_dim]

    # Auto-detect binarization threshold: midpoint of observed range
    g_min, g_max = float(np.min(gripper)), float(np.max(gripper))
    if g_max - g_min < 1e-6:
        # Gripper never moves
        return 0, 0.0, False

    threshold = (g_min + g_max) / 2.0
    binary = (gripper > threshold).astype(np.int32)
    transitions = int(np.sum(np.abs(np.diff(binary))))

    chatter_rate = transitions / max(duration, 1e-6)
    is_chattery = chatter_rate > config.chatter_threshold

    return transitions, chatter_rate, is_chattery
```

## 逐行讲解 / What's happening

1. **第 5-11 行 / Lines 5-11 (dead action masks)**:
   - 中文: 同时检查“全接近 0”和“始终等于第一帧”，覆盖静止和卡死两种坏轨迹。
   - English: it checks both near-zero actions and actions identical to the first frame, covering still and stuck trajectories.
2. **第 24-29 行 / Lines 24-29 (`np.gradient`)**:
   - 中文: 位置先变速度，再变加速度，再变 jerk；jerk 越大，动作越不平滑。
   - English: positions become velocity, acceleration, then jerk; larger jerk means less smooth motion.
3. **第 35-38 行 / Lines 35-38 (`ldlj`)**:
   - 中文: LDLJ 用总时长和峰值速度做无量纲化，方便不同长度轨迹比较。
   - English: LDLJ normalizes by duration and peak velocity, making trajectories of different lengths comparable.
4. **第 50-60 行 / Lines 50-60 (gripper threshold)**:
   - 中文: 夹爪信号先自动二值化，再数开合跳变次数。
   - English: the gripper signal is binarized automatically, then open/close transitions are counted.

## 类比 / The analogy

像质检员检查流水线产品：先挑出完全没动的机器，再看机械臂动作是不是抖，最后数夹爪有没有乱按开关。

It is like a quality inspector on a factory line: first catch machines that never moved, then check whether the arm moved smoothly, and finally count whether the gripper kept toggling.

## 自己跑一遍 / Try it yourself

```python
import numpy as np

actions = np.array([[0, 0], [0, 0], [1, 1], [1, -1], [1, 1]], dtype=float)
is_zero = np.all(np.abs(actions) < 1e-6, axis=1)
is_constant = np.all(np.abs(actions - actions[0:1]) < 1e-6, axis=1)
dead_fraction = np.mean(is_zero | is_constant)
gripper = actions[:, 1]
binary = (gripper > (gripper.min() + gripper.max()) / 2).astype(int)
transitions = np.sum(np.abs(np.diff(binary)))
print(round(dead_fraction, 2), int(transitions))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
0.4 3
```

40% 的 timestep 是 dead action，夹爪信号发生了 3 次跳变。

Forty percent of timesteps are dead actions, and the gripper signal changes state three times.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LeRobot dataset validation** / **LeRobot dataset validation**: 训练前检查 episode 字段和时间序列完整性。 / It checks episode fields and sequence integrity before training.
- **DROID / Open X-Embodiment preprocessing** / **DROID / Open X-Embodiment preprocessing**: 大规模机器人数据都需要先做质量门控。 / Large robot datasets need quality gates before model training.

## 注意事项 / Caveats / when it breaks

- **阈值不是通用真理** / **Thresholds are not universal truth**: 不同机器人动作尺度不同，`eps` 和 chatter 阈值要按数据集调。 / Different robots use different action scales, so `eps` and chatter thresholds need dataset tuning.
- **指标只负责报警** / **Metrics only raise alerts**: 删除样本前最好抽查可视化，避免误杀慢速精细操作。 / Before deleting samples, inspect visualizations to avoid removing slow but valid manipulation.

## 延伸阅读 / Further reading

- [Forge robotics data toolkit](https://github.com/arpitg1304/forge)
