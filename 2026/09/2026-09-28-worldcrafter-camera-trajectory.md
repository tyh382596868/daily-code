---
date: 2026-09-28
topic: diffusion
source: trending
repo: TencentARC/WorldCrafter
file: worldcrafter/camera.py
permalink: https://github.com/TencentARC/WorldCrafter/blob/ba1bbe57cf407ac24a2f45b87c5827ad81869815/worldcrafter/camera.py#L212-L363
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, camera-trajectory]
---

# WorldCrafter camera trajectory：把键盘动作采样成连续相机路径 / WorldCrafter Camera Trajectory: Sample Keyboard Actions into Continuous Camera Paths

> **一句话 / In one line**: 可控世界模型需要的不是“按了 W”，而是一段逐帧相机位姿。 / A controllable world model does not need “W was pressed”; it needs a frame-by-frame camera pose trajectory.

## 为什么重要 / Why this matters

WorldCrafter 的交互式探索要把离散动作变成视频生成条件。这里的代码把 forward/right/yaw/orbit/reverse 等事件采样成每个 chunk 的相机矩阵，并保存 logical start/end，方便后续反向和连续生成。

WorldCrafter's interactive exploration turns discrete controls into video-generation conditions. This code samples events such as forward, right, yaw, orbit, and reverse into camera matrices per chunk, while recording logical start/end poses for reversal and continuity.

## 代码 / The code

`TencentARC/WorldCrafter` — [`worldcrafter/camera.py`](https://github.com/TencentARC/WorldCrafter/blob/ba1bbe57cf407ac24a2f45b87c5827ad81869815/worldcrafter/camera.py#L212-L363)

```python
def sample_chunk(
    start: np.ndarray, action: Action, *, fractions: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one action; the logical endpoint starts the next chunk."""
    action.validate()
    alpha = np.arange(CHUNK_FRAMES, dtype=np.float64) / CHUNK_FRAMES if fractions is None else fractions
    poses = np.repeat(start[None], len(alpha), axis=0)
    end = start.copy()
    if action.yaw or action.pitch:
        def rotated(fraction):
            if action.yaw:
                return rotation_y(action.yaw * fraction) @ start[:3, :3]
            return start[:3, :3] @ rotation_x(action.pitch * fraction)

        for index, fraction in enumerate(alpha):
            poses[index, :3, :3] = rotated(fraction)
        end[:3, :3] = rotated(1.0)
        if action.orbit and action.orbit_radius:
            radius = action.orbit_radius
            angle = math.radians(abs(action.yaw or action.pitch))
            if radius * angle > MAX_TRANSLATION + 1e-12:
                raise ValueError("Orbit arc length must not exceed 5 per chunk; reduce radius or angle")
            center = start[:3, 3] + radius * start[:3, 2]
            poses[:, :3, 3] = center - radius * poses[:, :3, 2]
            if alpha[0] == 0:
                poses[0, :3, 3] = start[:3, 3]
            end[:3, 3] = center - radius * end[:3, 2]
    else:
        if action.up:
            direction = np.array([0.0, -action.up, 0.0])
        elif action.forward:
            direction = action.forward * horizontal_direction(start[:3, :3], True)
        else:
            direction = action.right * horizontal_direction(start[:3, :3], False)
        delta = action.speed * direction
        if np.linalg.norm(delta) > MAX_TRANSLATION + 1e-12:
            raise ValueError(f"Translation must not exceed {MAX_TRANSLATION:g} per chunk")
        poses[:, :3, 3] = start[:3, 3] + alpha[:, None] * delta
        end[:3, 3] = start[:3, 3] + delta
    return poses, end


def _sample_event(start: np.ndarray, event: str, fractions: np.ndarray, orbit_radius=DEFAULT_ORBIT_RADIUS) -> np.ndarray:
    components = event.split("&")
    if len(components) == 1:
        return sample_chunk(start, action_from_event(event, orbit_radius), fractions=fractions)[0]
    if any(part.startswith("orbit_") for part in components):
        raise ValueError("Use orbit as a separate action")
    poses = np.repeat(start[None], len(fractions), axis=0)
    delta = np.zeros(3)
    for component in components:
        action = action_from_event(component)
        if action.yaw:
            for index, fraction in enumerate(fractions):
                poses[index, :3, :3] = rotation_y(action.yaw * fraction) @ poses[index, :3, :3]
        elif action.pitch:
            for index, fraction in enumerate(fractions):
                poses[index, :3, :3] = poses[index, :3, :3] @ rotation_x(action.pitch * fraction)
        else:
            _, end = sample_chunk(start, action)
            delta += end[:3, 3] - start[:3, 3]
    if np.linalg.norm(delta) > MAX_TRANSLATION + 1e-12:
        raise ValueError(f"{event}: combined translation must not exceed {MAX_TRANSLATION:g} per chunk")
    poses[:, :3, 3] = start[:3, 3] + fractions[:, None] * delta
    return poses


def _sample_curve(start, event, tangent_start, tangent_end, times, *, orbit_radius=DEFAULT_ORBIT_RADIUS):
    fractions = times
    if tangent_start is not None:
        fractions = (
            -2 * times**3 + 3 * times**2
            + (times**3 - 2 * times**2 + times) * tangent_start
            + (times**3 - times**2) * tangent_end
        )
    return _sample_event(start, event, fractions, orbit_radius)


def _reverse_curve(curve, times):
    return curve(1.0 - times)


def _reverse_sampled_curve(curve, last_time, times):
    return curve(last_time * (1.0 - times))


def build_trajectory(
    events: list[str], *, dtype: str = "float64", sampling: str = "linear",
    last_frame: str = "exclude", orbit_radius: float = DEFAULT_ORBIT_RADIUS,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    if not events:
        raise ValueError("Provide at least one camera action")
    if not math.isfinite(orbit_radius) or orbit_radius < 0:
        raise ValueError("Orbit radius must be finite and non-negative")
    world = np.eye(4, dtype=np.float64)
    chunks, records, curves, sample_times = [], [], [], []

    def append(event, curve, times, poses=None):
        nonlocal world
        start, end = curve(np.array([0.0, 1.0]))
        chunks.append(curve(times) if poses is None else poses)
        records.append({
            "chunk_index": len(records), "event": event,
            "logical_start_c2w": start.tolist(), "logical_end_c2w": end.tolist(),
        })
        curves.append(curve)
        sample_times.append(times)
        world = end

    for index, event in enumerate(events):
        if event.startswith("reverse"):
            count = int(re.search(r"[0-9]+$", event).group())
            if count > len(chunks):
                raise ValueError(f"{event} needs {count} preceding chunks; only {len(chunks)} exist")
            indices = list(range(len(chunks) - 1, len(chunks) - count - 1, -1))
            for source in indices:
                if event.startswith("reverse_frames"):
                    curve = partial(_reverse_sampled_curve, curves[source], sample_times[source][-1])
                    times = np.linspace(0.0, 1.0, CHUNK_FRAMES)
                    append(event, curve, times, chunks[source][::-1].copy())
                else:
                    curve = partial(_reverse_curve, curves[source])
                    include_end = last_frame == "include" and index == len(events) - 1 and source == indices[-1]
                    times = (
                        np.linspace(0.0, 1.0, CHUNK_FRAMES) if include_end
                        else np.arange(CHUNK_FRAMES, dtype=np.float64) / CHUNK_FRAMES
                    )
                    poses = None
                    original = records[source]["event"]
                    if sampling == "linear" and "&" not in original and not original.startswith("reverse"):
                        action = action_from_event(original)
                        if not action.yaw and not action.pitch and sample_times[source][-1] < 1.0 and not include_end:
                            # Reuse linear translation samples without another interpolation roundoff.
                            endpoint = np.array(records[source]["logical_end_c2w"])
                            poses = np.concatenate([endpoint[None], chunks[source][1:][::-1]])
                    append(event, curve, times, poses)
            continue
        smooth = sampling == "smooth_turns"
        entering = index > 0 and events[index - 1] == event
        leaving = index + 1 < len(events) and events[index + 1] == event
        include_end = (smooth and not leaving) or (last_frame == "include" and index == len(events) - 1)
        times = (
            np.linspace(0.0, 1.0, CHUNK_FRAMES) if include_end
            else np.arange(CHUNK_FRAMES, dtype=np.float64) / CHUNK_FRAMES
        )
        curve = partial(
            _sample_curve, world.copy(), event,
            float(entering) if smooth else None, float(leaving),
            orbit_radius=orbit_radius,
        )
        append(event, curve, times)
    return np.concatenate(chunks).astype(dtype), records
```

## 逐行讲解 / What's happening

1. **第 212-251 行 / Lines 212-251**:
   - 中文: 单个 action 被采样为 `CHUNK_FRAMES` 个 pose；旋转、orbit 和平移各有不同几何。
   - English: A single action becomes `CHUNK_FRAMES` poses; rotation, orbit, and translation use different geometry.
1. **第 254-276 行 / Lines 254-276**:
   - 中文: 复合事件先累积旋转和平移，再检查合并位移是否越界。
   - English: Compound events accumulate rotation and translation, then validate the combined displacement.
1. **第 279-287 行 / Lines 279-287**:
   - 中文: smooth turns 用 Hermite-like 曲线改变 fraction，让重复动作进入/离开更平滑。
   - English: Smooth turns use a Hermite-like curve over fractions so repeated actions enter and leave more gently.
1. **第 321-347 行 / Lines 321-347**:
   - 中文: reverse 不是简单改字符串，而是复用历史 curve 或已采样帧，保证回放路径一致。
   - English: Reverse is not just a string flag; it reuses prior curves or sampled frames so the return path is consistent.

## 类比 / The analogy

像给摄影师写分镜表：不能只写“往前走一下”，要写每一帧机位在哪、镜头朝哪、这一段从哪里接到哪里。

It is like writing a shot plan for a camera operator. “Move forward” is not enough; every frame needs a position, direction, and start/end continuity.

## 自己跑一遍 / Try it yourself

```python
frames=4
start=[0.0,0.0,0.0]
delta=[1.0,0.0,0.0]
poses=[
    [round(start[j]+(i/frames)*delta[j],2) for j in range(3)]
    for i in range(frames)
]
end=[start[j]+delta[j] for j in range(3)]
print(poses)
print(end)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
[[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0], [0.75, 0.0, 0.0]]
[1.0, 0.0, 0.0]
```

注意采样帧不一定包含终点；终点是下一段 chunk 的 logical start。

Notice sampled frames may exclude the endpoint; the endpoint becomes the next chunk's logical start.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Astra relative pose** / **Astra-style systems also translate trajectories into local motion conditions.**
- **World model rollouts** / **Rollout-based world models need action sequences aligned to frames.**
- **Camera-conditioned video diffusion** / **Camera paths are a common conditioning channel for controllable video generation.**

## 注意事项 / Caveats / when it breaks

- **端点包含策略很重要** / **Including the last frame can duplicate the first frame of the next chunk.**
- **复合动作要限幅** / **Combined translations can exceed the model's training distribution.**
- **反向路径要复用历史** / **Recomputing reverse curves naively can introduce small geometric drift.**

## 延伸阅读 / Further reading

- WorldCrafter source permalink above
- WorldCrafter README and project page
