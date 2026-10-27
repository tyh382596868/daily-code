---
date: 2026-10-27
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/utils/data/dataset.py
permalink: https://github.com/pytorch/pytorch/blob/f6e5bb36a9b3739e2b774bb94463704af1e1c06f/torch/utils/data/dataset.py#L73-L114
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, pytorch, dataloader]
---

# PyTorch 流式数据：每个 worker 必须各走一段 / PyTorch Iterable Data: Each Worker Needs Its Own Range

> **一句话 / In one line**: `IterableDataset` 被多 worker 复制，数据范围也必须显式切开。 / `IterableDataset` is copied into every worker, so its range must be sharded explicitly.

## 为什么重要 / Why this matters

多 worker 不会自动让普通 Python 迭代器“分工”；若每个副本都从同一流读取，训练样本会悄悄重复。

Multiple workers do not automatically divide an ordinary Python stream; every copy can silently replay the same examples.

## 代码 / The code

`pytorch/pytorch` — [`torch/utils/data/dataset.py`](https://github.com/pytorch/pytorch/blob/f6e5bb36a9b3739e2b774bb94463704af1e1c06f/torch/utils/data/dataset.py#L103-L114)

```python
worker_info = torch.utils.data.get_worker_info()
if worker_info is None:
    iter_start, iter_end = self.start, self.end
else:
    per_worker = int(math.ceil((self.end - self.start) / worker_info.num_workers))
    iter_start = self.start + worker_info.id * per_worker
    iter_end = min(iter_start + per_worker, self.end)
return iter(range(iter_start, iter_end))
```

## 逐行讲解 / What's happening

主进程保留全范围；worker 按向上取整的块大小取得不重叠区间，`min` 防止最后一块越界。

The main process keeps the full range. Workers receive disjoint ceil-sized blocks, and `min` clips the tail.

## 类比 / The analogy

像仓库主管把拣货清单切成连续区段，而不是给每个员工同一张清单。

It is a warehouse manager assigning contiguous pick lists instead of handing everyone the same list.

## 自己跑一遍 / Try it yourself

```python
import math
start, end, workers = 0, 10, 3
n = math.ceil((end - start) / workers)
print([list(range(start+i*n, min(start+(i+1)*n, end))) for i in range(workers)])
```

运行 / Run with `python try.py`。输出 / Expected output: `[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]`。

## 注意事项 / Caveats / when it breaks

- **分布式训练 / Distributed training**：worker 分片之外还要按 rank 分片，否则不同 GPU 仍会重复。 / Worker sharding is not rank sharding; distributed jobs need both.
