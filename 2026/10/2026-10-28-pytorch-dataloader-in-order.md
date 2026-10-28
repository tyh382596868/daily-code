---
date: 2026-10-28
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/utils/data/dataloader.py
permalink: https://github.com/pytorch/pytorch/blob/main/torch/utils/data/dataloader.py
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, dataloader]
---

# PyTorch DataLoader 顺序开关：吞吐和样本顺序的取舍 / PyTorch DataLoader `in_order`: Throughput Versus Order

> **一句话 / In one line**: 多 worker DataLoader 默认按发出顺序交付 batch；关掉 `in_order` 后，先完成的 worker 可先返回。 / Multi-worker DataLoader returns batches FIFO by default; with `in_order=False`, the first worker to finish can return first.

## 为什么重要 / Why this matters

慢样本会拖住默认 FIFO 队列；但乱序也可能让类别不均衡数据在训练早期产生偏斜。

A slow sample can stall FIFO delivery, while out-of-order delivery can bias the early stream of imbalanced data.

## 代码 / The code

`pytorch/pytorch` — [`dataloader.py`](https://github.com/pytorch/pytorch/blob/main/torch/utils/data/dataloader.py)

```python
in_order: bool = True

# Documentation contract:
# If False, batches are not enforced to be returned FIFO.
# It only applies when num_workers > 0.
```

## 逐行讲解 / What's happening

worker 为已派发的 batch 生产结果。默认模式会暂存“后到”的结果，等待较早编号完成；关闭后省掉等待，但返回顺序不再等于 sampler 顺序。

Workers produce dispatched batches. The default buffers later results until earlier indices arrive; disabling it removes that wait, but delivery no longer matches sampler order.

## 类比 / The analogy

默认像按叫号取餐；`in_order=False` 像哪个窗口先做好就先端走。

Default mode is numbered pickup; `in_order=False` is whichever counter finishes first.

## 自己跑一遍 / Try it yourself

```python
from torch.utils.data import DataLoader, TensorDataset
import torch
loader = DataLoader(TensorDataset(torch.arange(8)), num_workers=2, in_order=False)
print(next(iter(loader))[0])
```

运行 / Run with `pip install torch && python try.py`。输出 / Expected output: 一个 batch；跨运行顺序不应作为契约。 / A batch; do not treat cross-run order as a contract.

## 注意事项 / Caveats / when it breaks

- **可复现性 / Reproducibility**：只在能容忍乱序时开启，并固定 worker seed。 / Enable only when reordering is acceptable and worker seeds are controlled.
