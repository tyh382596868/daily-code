---
date: 2026-10-28
topic: diffusion
source: trending
repo: AMAP-ML/DreamX-World
file: inference_ar_forcing.py
permalink: https://github.com/AMAP-ML/DreamX-World/blob/master/inference_ar_forcing.py#L978-L1073
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, world-model, camera]
---

# DreamX-World 相机条件：每帧位姿扩展成每个 patch 的条件 / DreamX-World Camera Conditioning: Expand Poses Per Patch

> **一句话 / In one line**: DreamX-World 把 latent 帧的相机外参和内参复制到该帧的每个空间 patch，使几何条件与视频 token 一一对齐。 / DreamX-World repeats each latent-frame camera pose and intrinsics for every spatial patch, aligning geometry with video tokens.

## 为什么重要 / Why this matters

视频 DiT 通常按 patch 而非整帧工作；帧级相机参数必须变成 token 级条件，才能与 RoPE/attention 序列同长度。

Video DiTs operate on patches, not whole frames. Frame-level camera parameters must become token-level conditions to match the RoPE/attention sequence.

## 代码 / The code

`AMAP-ML/DreamX-World` — [`inference_ar_forcing.py`](https://github.com/AMAP-ML/DreamX-World/blob/master/inference_ar_forcing.py#L978-L1073)

```python
c2ws = torch.as_tensor(c2w_poses, dtype=dtype, device=device)
viewmats = _invert_SE3(c2ws)  # [T_latent, 4, 4]
viewmats = viewmats.unsqueeze(1).expand(-1, 880, -1, -1)
viewmats = viewmats.reshape(1, -1, 4, 4)
K = K.unsqueeze(1).expand(-1, viewmats.shape[1], -1, -1)
```

## 逐行讲解 / What's happening

代码先把相机到世界坐标转换为 world-to-camera，再为每帧的 `22×40=880` 个 patch 复用同一矩阵，最后压成 `[1, token_count, ...]`。

It converts camera-to-world into world-to-camera, reuses each matrix for `22×40=880` patches, then flattens to `[1, token_count, ...]`.

## 类比 / The analogy

一帧是一张地图；相机参数是地图图例。模型需要把同一份图例贴到该地图的每一小格上。

A frame is a map and its camera parameters are the legend; the model needs that legend on every cell.

## 自己跑一遍 / Try it yourself

```python
import torch
pose = torch.eye(4).unsqueeze(0)
per_patch = pose.unsqueeze(1).expand(-1, 4, -1, -1).reshape(1, -1, 4, 4)
print(per_patch.shape)  # torch.Size([1, 4, 4, 4])
```

运行 / Run with `pip install torch && python try.py`。输出 / Expected output: 每个 patch 都有一份 pose。 / Every patch receives a pose.

## 注意事项 / Caveats / when it breaks

- **patch 网格 / Patch grid**：`880` 是该模型的 `22×40` 网格；换分辨率或 VAE stride 时必须重新计算。 / `880` is this model’s grid; recompute it for another resolution or VAE stride.
