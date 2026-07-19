---
date: 2026-07-19
topic: vla
source: vla
repo: huggingface/lerobot
file: examples/dataset/load_lerobot_dataset.py
permalink: https://github.com/huggingface/lerobot/blob/main/examples/dataset/load_lerobot_dataset.py
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, lerobot, temporal-window, dataset]
build_role: short-term-observation-memory advanced variant
---

# LeRobot delta_timestamps：用相对时间窗取历史观测和未来动作 / LeRobot delta_timestamps: Use Relative Time Windows for History and Future Actions

> **一句话 / In one line**: `delta_timestamps` 让一个当前 index 同时取历史图像、历史 state 和未来 action chunk。 / `delta_timestamps` lets one current index fetch past images, past states, and future action chunks.

## 为什么重要 / Why this matters

VLA 不只看当前帧。真实机器人需要短期记忆来估计速度、接触变化和动作趋势；训练动作 chunk 时还要拿到未来多个 action。LeRobot 把这些时间窗写成相对秒数，dataset 负责对齐帧。

A VLA does not only need the current frame. Real robots need short-term memory for velocity, contact changes, and action trends; action-chunk training also needs future actions. LeRobot writes those windows as relative seconds, and the dataset aligns frames.

## 代码 / The code

`huggingface/lerobot` — [`examples/dataset/load_lerobot_dataset.py`](https://github.com/huggingface/lerobot/blob/main/examples/dataset/load_lerobot_dataset.py)

```python
delta_timestamps = {
    camera_key: [-1, -0.5, -0.20, 0],
    "observation.state": [-1.5, -1, -0.5, -0.20, -0.10, 0],
    "action": [t / dataset.fps for t in range(64)],
}

dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)

print(f"{dataset[0][camera_key].shape=}")          # (4, c, h, w)
print(f"{dataset[0]['observation.state'].shape=}") # (6, c)
print(f"{dataset[0]['action'].shape=}")            # (64, c)
```

## 逐行讲解 / What's happening

1. **图像看过去 / Images look backward**: 中文: `[-1, -0.5, -0.20, 0]` 取当前帧和三个历史帧。 English: `[-1, -0.5, -0.20, 0]` takes the current frame plus three past frames.
2. **state 更密 / State is denser**: 中文: proprioception 比图像便宜，可以取更多历史点。 English: proprioception is cheaper than images, so it can use more history points.
3. **action 看未来 / Actions look forward**: 中文: `[t / fps for t in range(64)]` 变成未来 64 步动作监督。 English: `[t / fps for t in range(64)]` becomes supervision for a 64-step future action chunk.
4. **dataset 做对齐 / The dataset aligns timestamps**: 中文: policy 不需要知道 parquet 行号怎么换成帧。 English: the policy does not need to know how parquet rows map to frames.
5. **输出多一个时间维 / Outputs gain a time dimension**: 中文: 每个 key 返回的是窗口堆叠，不再是单帧。 English: each key returns a stacked window, not a single frame.

## 类比 / The analogy

像开车看后视镜和前方路线。你不是只看挡风玻璃这一瞬间，还会瞄一眼刚刚的位置，并提前计划接下来几秒的方向盘。

It is like driving with mirrors and a route ahead. You do not only look through the windshield at this instant; you glance at recent history and plan the next few seconds of steering.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

这是 `short-term-observation-memory` 的 dataset 版本。上游是 episode storage，下游是 vision encoder、state encoder 和 action head。最小 nanoVLA 可以先把这些窗口简单 stack；生产实现还要处理缺帧、跨 episode 边界、不同相机 fps 和 padding mask。

This is the dataset-level version of `short-term-observation-memory`. The upstream component is episode storage; downstream consumers are the vision encoder, state encoder, and action head. A minimal nanoVLA can simply stack these windows; a production version also needs missing-frame handling, episode-boundary guards, mixed camera FPS, and padding masks.

## 自己跑一遍 / Try it yourself

```python
fps = 10
frames = list(range(100))

def window(current, deltas):
    out = []
    for dt in deltas:
        idx = current + round(dt * fps)
        out.append(frames[max(0, min(idx, len(frames) - 1))])
    return out

delta_timestamps = {
    "image": [-1, -0.5, 0],
    "action": [t / fps for t in range(4)],
}

print(window(20, delta_timestamps["image"]))
print(window(20, delta_timestamps["action"]))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[10, 15, 20]
[20, 21, 22, 23]
```

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **ACT action chunking** / **ACT action chunking**: 训练时直接监督一段未来动作。 / Training supervises a future action segment.
- **SmolVLA image preprocessing** / **SmolVLA image preprocessing**: 多相机图像先按统一 schema 进入模型。 / Multi-camera images enter the model under a unified schema.

## 注意事项 / Caveats / when it breaks

- **delta 必须贴合 fps / Deltas must align with FPS**: 不能落在不存在的帧间隔上。 / Deltas should land on valid frame intervals.
- **跨 episode 要截断 / Clip across episode boundaries**: 历史窗口不能偷看上一个 episode。 / History windows must not leak into the previous episode.
- **未来 action 只在训练可用 / Future actions are training-only**: 部署时只能由 policy 预测。 / Future actions are only available as training targets.

## 延伸阅读 / Further reading

- [LeRobot dataset example](https://github.com/huggingface/lerobot/blob/main/examples/dataset/load_lerobot_dataset.py)
- [LeRobot dataset v3 docs](https://github.com/huggingface/lerobot/blob/main/docs/source/lerobot-dataset-v3.mdx)
