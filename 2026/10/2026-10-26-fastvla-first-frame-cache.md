---
date: 2026-10-26
topic: robotics
source: trending
repo: Garfield-kh/FastVLA
file: fastvla/policy.py
permalink: https://github.com/Garfield-kh/FastVLA/blob/0cb8693d333d4168f875c711bbf0aa1f9e0c527d/fastvla/policy.py#L74-L105
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, vla, kv-cache]
---

# FastVLA 首帧缓存：任务只预填一次，控制循环只跑轻量 actor / FastVLA First-Frame Cache: Prefill the Task Once, Run Only the Lightweight Actor

> **一句话 / In one line**: 首次 `act` 构建 VLM KV cache，后续控制步复用它。 / The first `act` builds VLM KV cache; later control steps reuse it.

## 为什么重要 / Why this matters

高频控制不能每一步都重跑大 VLM。FastVLA 将指令和首帧变成 episode 级任务记忆，在线路径只编码当前观测、状态并查询 cache。

High-rate control cannot rerun a large VLM every step. FastVLA turns instruction plus first frame into episode-level task memory.

## 代码 / The code

```python
def act(self, views, state, instruction, hist_views=None):
    if self.cache is None:
        self.build_cache(instruction, views)
    slots = []
    for c, off in zip(self.cam_of_slot, self.slot_time):
        src = views if off == 0 else (hist_views or {}).get(off, views)
        slots.append(to_float_image(src[c], self.img_hw))
    imgs = torch.stack(slots)[None].to(self.device)
    with torch.autocast(torch.device(self.device).type, dtype=torch.bfloat16):
        out = self.model(imgs=imgs, state=s.to(self.device), role=self.role, **self.cache)
```

来源 / Source: [`fastvla/policy.py`](https://github.com/Garfield-kh/FastVLA/blob/0cb8693d333d4168f875c711bbf0aa1f9e0c527d/fastvla/policy.py#L74-L105)。该项目于 2026-10 发布。 / The project was released in October 2026.

## 逐行讲解 / What's happening

`cache is None` 把昂贵 prefill 限定到 episode 第一拍；历史槽缺失时退回当前画面；actor 接收 cache 作为额外 KV 条件。

`cache is None` limits costly prefill to the first step; missing history falls back to current views; the actor receives cache as KV conditioning.

## 类比 / The analogy

像厨师先读一次整张菜单，随后每道菜只看当前锅里的状态。

Like a chef reading the whole menu once, then watching only the pan state for each dish.

## 自己跑一遍 / Try it yourself

```python
cache = None
for frame in [0, 1, 2]:
    if cache is None: cache = "instruction+first-frame"
    print(frame, cache)
```

运行 / Run with `python try.py`。预期 / Expected: 三步都复用同一 cache。

## 注意事项 / Caveats / when it breaks

- **任务变化 / Task changes**：换指令、相机标定或 episode 时必须 `reset()`。 / Reset when the instruction, calibration, or episode changes.
