---
date: 2026-09-01
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/policies/pi05/memory.py
permalink: https://github.com/huggingface/lerobot/blob/d36d404b65315139b7601a707f260a3db736462f/src/lerobot/policies/pi05/memory.py#L68-L121
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, short-term-observation-memory, temporal-mask]
build_role: short-term-observation-memory advanced variant
---

# LeRobot Pi05 memory：固定步幅取历史帧 / LeRobot Pi05 Memory: Sample History Frames at a Fixed Stride

> **一句话 / In one line**: Pi05 把最近的观测队列采成固定长度历史窗口，并为 episode 开头的补帧生成 temporal mask。 / Pi05 samples a fixed-length window from the recent observation queue and builds a temporal mask for padded early-episode frames.

## 为什么重要 / Why this matters

VLA 不只看当前相机帧。拿取、接近、放置这些动作都依赖短时间运动线索：物体刚才往哪儿动，夹爪是不是已经闭合，机器人是否在抖。Pi05 的 memory 工具把这个问题压成三个小件：按年龄取帧、给时间位置编码、用 mask 阻止模型看见还不存在的历史。

A VLA rarely needs only the current camera frame. Grasping, approaching, and placing depend on short-term motion cues: where the object was moving, whether the gripper already closed, and whether the robot is shaking. Pi05 compresses the problem into three small pieces: sample by age, encode temporal position, and mask history that does not exist yet.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/pi05/memory.py`](https://github.com/huggingface/lerobot/blob/d36d404b65315139b7601a707f260a3db736462f/src/lerobot/policies/pi05/memory.py#L68-L121)

```python
def sample_observation_history(
    history: list[Tensor], *, num_frames: int, stride: int, steps_seen: int
) -> tuple[Tensor, Tensor]:
    """Subsample a homogeneous-batch inference queue and mark pre-episode padding.

    Frames are addressed by age relative to ``history[-1]``, the newest observation,
    so the current frame is always the last sampled frame regardless of queue length.

    ``steps_seen`` applies to every batch row. Pi05 clears the queue at each
    batched rollout boundary; independently resetting vector rows is unsupported.
    """
    # Descending ages, so the returned frames run oldest -> current.
    required_ages = list(range((num_frames - 1) * stride, -1, -stride))
    if len(history) <= required_ages[0]:
        raise ValueError(
            f"observation history holds {len(history)} frames, need at least "
            f"{required_ages[0] + 1} to sample {num_frames} frames at stride {stride}"
        )
    values = torch.stack([history[-1 - age] for age in required_ages], dim=1)
    valid = torch.tensor([steps_seen > age for age in required_ages], dtype=torch.bool, device=values.device)
    padding_mask = (~valid)[None, :].expand(values.shape[0], -1)
    return values, padding_mask


def temporal_sinusoidal_embedding(
    num_frames: int, hidden_size: int, *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """Return fixed temporal embeddings whose current position is exactly zero."""
    if hidden_size % 2:
        raise ValueError(f"hidden_size must be even, got {hidden_size}")
    positions = torch.arange(1 - num_frames, 1, device=device, dtype=torch.float32)[:, None]
    frequencies = torch.exp(
        torch.arange(0, hidden_size, 2, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / hidden_size)
    )[None, :]
    angles = positions * frequencies
    embedding = torch.zeros(num_frames, hidden_size, device=device, dtype=torch.float32)
    embedding[:, 0::2] = torch.sin(angles)
    embedding[:, 1::2] = torch.cos(angles) - 1.0
    return embedding.to(dtype=dtype)


def causal_temporal_mask(frame_mask: Tensor, *, dtype: torch.dtype, num_patches: int) -> Tensor:
    """Build an additive causal and key-padding mask for temporal attention."""
    if frame_mask.ndim != 2:
        raise ValueError(f"frame_mask must have shape (batch, frames), got {tuple(frame_mask.shape)}")
    batch_size, num_frames = frame_mask.shape
    allowed = torch.ones(num_frames, num_frames, dtype=torch.bool, device=frame_mask.device).tril()
    allowed = allowed[None] & frame_mask[:, None, :].bool()
    # Keep padded query rows numerically safe; valid queries still cannot see padded keys.
    allowed |= torch.eye(num_frames, dtype=torch.bool, device=frame_mask.device)[None]
    mask = torch.zeros(batch_size, 1, num_frames, num_frames, dtype=dtype, device=frame_mask.device)
    mask.masked_fill_(~allowed[:, None], torch.finfo(dtype).min)
    return mask.repeat_interleave(num_patches, dim=0)
```

## 逐行讲解 / What's happening

1. **第 79-86 行 / Lines 79-86 (age-based sampling)**:
   - 中文: `required_ages` 从最老到当前排列，例如 4 帧、stride=2 会取 `6,4,2,0` 这些 age。
   - English: `required_ages` runs oldest to current; for four frames with stride 2, it samples ages `6,4,2,0`.
2. **第 86-89 行 / Lines 86-89 (values and padding mask)**:
   - 中文: `values` 固定成 `(B,T,...)`；`steps_seen` 不够老的历史会在 `padding_mask` 里标成无效。
   - English: `values` has a fixed `(B,T,...)` shape; history older than `steps_seen` is marked invalid in `padding_mask`.
3. **第 92-107 行 / Lines 92-107 (current frame as zero)**:
   - 中文: 时间位置从 `1-num_frames` 到 `0`，所以当前帧的 sin 为 0、cos-1 也为 0。
   - English: Temporal positions run from `1-num_frames` to `0`, so the current frame has sin 0 and cos-minus-1 equal to 0.
4. **第 110-121 行 / Lines 110-121 (causal temporal mask)**:
   - 中文: temporal attention 只能看自己和过去的有效帧；padding query 行保留对角线，避免全是 `-inf`。
   - English: Temporal attention can only see the current and valid past frames; padded query rows keep the diagonal to avoid all-`-inf` rows.

## 类比 / The analogy

像倒车影像不是只看当前一帧，而是按固定间隔看最近几张截图。刚启动时历史不够，就要贴上“这张是假补的”标签，别让驾驶员把它当真。

It is like a reversing camera that shows several recent snapshots at fixed intervals, not only the current frame. Right after startup, missing history is padded and must be labeled as fake so it is not treated as real evidence.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

中文: 这是 `short-term-observation-memory` 的 advanced variant，依赖 `vision-encoder`。在 nanoVLA 里，它放在相机预处理之后、VLM/action expert 前面：输入是最近 N 帧图像或视觉 token，输出是固定长度历史张量和 temporal mask。省掉它，模型只能从单帧猜速度和接触状态；生产版还要支持每个环境单独 reset、多相机同步、丢帧插值和实时缓存上限。

English: This is an advanced variant of the `short-term-observation-memory` item, depending on `vision-encoder`. In a nanoVLA it sits after camera preprocessing and before the VLM/action expert: it takes recent frames or visual tokens and returns a fixed history tensor plus temporal mask. Without it, the model must infer velocity and contact state from one frame. A production version adds per-environment reset, multi-camera synchronization, dropped-frame interpolation, and bounded realtime caches.

## 自己跑一遍 / Try it yourself

```python
history = ["f0", "f1", "f2", "f3", "f4", "f5", "f6"]
num_frames, stride, steps_seen = 4, 2, 3
ages = list(range((num_frames - 1) * stride, -1, -stride))
sampled = [history[-1 - age] for age in ages]
valid = [steps_seen > age for age in ages]
padding = [not x for x in valid]
print(ages)
print(sampled)
print(padding)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[6, 4, 2, 0]
['f0', 'f2', 'f4', 'f6']
[True, True, False, False]
```

中文: 注意采样形状固定，但前两帧因为 episode 还没真的经历那么久，被 mask 标成 padding。

English: Notice that the sampled shape is fixed, but the first two frames are marked padding because the episode has not truly lived that long.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LeRobot StreamingVideoEncoder** / **LeRobot StreamingVideoEncoder**: 中文: 采集侧也会保留短期视频上下文，只是目标是压缩存储。 / English: The collection side also preserves short video context, but for compressed storage.
- **openpi inference loops** / **openpi inference loops**: 中文: 部署时通常也维护观测队列，再把最近窗口喂给 policy。 / English: Deployment loops often maintain an observation queue and feed the recent window to the policy.

## 注意事项 / Caveats / when it breaks

- **batch 同步假设 / Batch reset assumption**: 中文: 源码注释说明 `steps_seen` 作用于整个 batch；如果并行环境不同步 reset，需要按行记录。 / English: The source notes that `steps_seen` applies to the whole batch; asynchronous vector environments need per-row counters.
- **stride 是信息取舍 / Stride is an information tradeoff**: 中文: stride 太大看不到快速接触，太小又浪费 token 在相似帧上。 / English: A large stride misses fast contacts; a small stride spends tokens on near-duplicate frames.

## 延伸阅读 / Further reading

- Pi05 memory source: https://github.com/huggingface/lerobot/blob/d36d404b65315139b7601a707f260a3db736462f/src/lerobot/policies/pi05/memory.py
