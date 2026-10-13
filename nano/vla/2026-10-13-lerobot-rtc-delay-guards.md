---
date: 2026-10-13
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/rollout/inference/rtc.py
permalink: https://github.com/huggingface/lerobot/blob/24f15b92b556cb861ca65e989462e23505a701bd/src/lerobot/rollout/inference/rtc.py#L105-L181
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, action-chunking, rtc]
build_role: action-chunking advanced variant, latency-aware chunk reuse guards
---

# LeRobot RTC guards：动作 chunk 能不能接上，要先看延迟 / LeRobot RTC Guards: Check Latency Before Reusing an Action Chunk

> **一句话 / In one line**: RTC 会裁剪、补齐并检查上个 chunk 的剩余动作，避免延迟超出训练范围。 / RTC trims, pads, and validates leftover actions from the previous chunk so latency does not exceed the trained range.

## 为什么重要 / Why this matters

异步 VLA 推理里，新 chunk 算出来时机器人已经执行了几步；这段代码把 prefix 长度和真实延迟变成显式规则。

In asynchronous VLA inference, the robot has already executed steps before the new chunk arrives; this code makes prefix length and real delay explicit rules.

## 代码 / The code

`huggingface/lerobot` - [`src/lerobot/rollout/inference/rtc.py`](https://github.com/huggingface/lerobot/blob/24f15b92b556cb861ca65e989462e23505a701bd/src/lerobot/rollout/inference/rtc.py#L105-L181)

```python
def _normalize_prev_actions_length(prev_actions: torch.Tensor, target_steps: int) -> torch.Tensor:
    """Pad (holding the last action) or truncate RTC prefix actions to a fixed length.

    Zero-padding would decode to the dataset mean inside the RTC guided region.
    """
    if prev_actions.ndim != 2:
        raise ValueError(f"Expected 2D [T, A] tensor, got shape={tuple(prev_actions.shape)}")
    steps, _ = prev_actions.shape
    if steps == target_steps:
        return prev_actions
    if steps > target_steps:
        return prev_actions[:target_steps]
    if steps == 0:
        raise ValueError("Cannot pad an empty prefix: no last action to hold.")
    hold = prev_actions[-1:].expand(target_steps - steps, -1)
    return torch.cat([prev_actions, hold], dim=0)


def _trained_rtc_chunk_can_merge(
    *,
    conditioned_delay: int,
    measured_delay: int,
    training_max_delay: int,
    has_previous_actions: bool,
) -> bool:
    """Whether a trained RTC chunk still covers the overlap that actually elapsed.

    A chunk is unusable either because inference outran the prefix it was conditioned on, or
    because the elapsed delay left the range the checkpoint was trained for. Both are transient
    by nature (a latency spike), so this reports them the same way and lets the caller retry;
    only a persistent run of unusable chunks is fatal.
    """
    if not has_previous_actions:
        return True
    if measured_delay > training_max_delay:
        return False
    return measured_delay <= conditioned_delay


def _estimate_rtc_delay(
    *,
    latency: float,
    time_per_step: float,
    mode: str,
    training_max_delay: int,
    has_previous_actions: bool,
) -> int:
    """Estimate overlap, using the trained capacity to bootstrap the first transition."""
    if latency:
        return math.ceil(latency / time_per_step)
    if mode == "trained" and has_previous_actions:
        return training_max_delay
    return 0


def _clamp_trained_rtc_delay(*, conditioned_delay: int, available_steps: int, training_max_delay: int) -> int:
    """Clamp the hard prefix to what both the checkpoint and the queue can back.

    Past ``training_max_delay`` the model has never seen a prefix that long, and past
    ``available_steps`` ``_normalize_prev_actions_length`` pads the tail by holding the last
    action, so the extra steps would be inpainted as if a frozen hold had been committed.
    Clamping keeps the chunk usable; ``_trained_rtc_chunk_can_merge`` still discards it if the
    delay that actually elapsed outran this prefix.
    """
    clamped = min(conditioned_delay, training_max_delay, available_steps)
    if clamped < conditioned_delay:
        logger.warning(
            "Trained RTC wanted a %d-step prefix but the checkpoint supports %d and the queue "
            "holds %d committed actions; conditioning on %d. Raise --inference.queue_threshold "
            "and --inference.rtc.execution_horizon, or retrain with a larger "
            "--policy.rtc_training_max_delay, to keep the full overlap.",
            conditioned_delay,
            training_max_delay,
            available_steps,
            clamped,
        )
    return clamped
```

## 逐行讲解 / What's happening

1. **第 105-120 行 / Lines 105-120 (prefix length)**:
   - 中文: 太长截断，太短保持最后动作补齐。
   - English: Long prefixes are truncated; short prefixes are padded by holding the last action.
2. **第 123-141 行 / Lines 123-141 (can merge)**:
   - 中文: 真实延迟超过训练上限或条件 delay 就拒绝合并。
   - English: A chunk is rejected if measured delay exceeds training limit or conditioned delay.
3. **第 144-181 行 / Lines 144-181 (estimate and clamp)**:
   - 中文: wall-clock latency 被换算成步数，再按训练能力和队列长度夹住。
   - English: Wall-clock latency becomes steps, then is clamped by training capacity and queue length.

## 类比 / The analogy

这像接力铺铁轨：看不见足够多上一段轨道时，宁可停一下，也不要硬接歪。

It is like laying railway tracks in shifts: if you cannot see enough of the previous segment, pause rather than attach a crooked segment.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

中文: 这是 `action-chunking` 的高级边界层，位于 chunk policy 和单步执行器之间；省掉它，异步推理遇到 latency spike 时容易把新旧动作错位拼接。

English: This is an advanced `action-chunking` boundary between the chunk policy and the single-step actuator; without it, latency spikes can splice old and new actions out of phase.

## 自己跑一遍 / Try it yourself

```python
def normalize(prev, target):
    return prev[:target] if len(prev) > target else prev + [prev[-1]] * (target - len(prev))
def can_merge(conditioned, measured, max_delay):
    return measured <= max_delay and measured <= conditioned
print(normalize([10, 11], 4))
print(can_merge(3, 2, 4))
print(can_merge(3, 5, 4))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[10, 11, 11, 11]
True
False
```

中文: 连续控制里，hold last action 往往比补 0 更安全。

English: In continuous control, holding the last action is often safer than zero padding.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **openpi ActionChunkBroker** / **openpi ActionChunkBroker**: openpi ActionChunkBroker。 / openpi ActionChunkBroker.
- **ACT temporal ensemble** / **ACT temporal ensemble**: ACT temporal ensemble。 / ACT temporal ensemble.

## 注意事项 / Caveats / when it breaks

- **训练分布外延迟** / **Chunks outside trained delay need retry or fallback**: 训练分布外延迟。 / Chunks outside trained delay need retry or fallback.
- **空 prefix** / **A hold-prefix cannot be built without any previous action**: 空 prefix。 / A hold-prefix cannot be built without any previous action.

## 延伸阅读 / Further reading
- [LeRobot RTC inference](https://github.com/huggingface/lerobot/blob/24f15b92b556cb861ca65e989462e23505a701bd/src/lerobot/rollout/inference/rtc.py#L105-L181)
