---
date: 2026-10-29
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/utils/data/dataloader.py
permalink: https://github.com/pytorch/pytorch/blob/e9295c692a07e709c9070b7fb55cf76e8f2d79c1/torch/utils/data/dataloader.py#L783-L788
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, pytorch, dataloader]
---

# PyTorch 单进程 DataLoader / PyTorch Single-Process DataLoader

> **一句话 / In one line**: 一次取 batch 是索引、抓取、可选 pin-memory 的小流水线。 / Fetching a batch is a small pipeline: index, fetch, optionally pin.

## 为什么重要 / Why this matters

最简单的 DataLoader 路径把接口拆成三步，因此 sampler、dataset 和设备传输可以各自演进。

The simplest DataLoader path separates three concerns, so samplers, datasets, and device transfer can evolve independently.

## 代码 / The code

`pytorch/pytorch` — [`torch/utils/data/dataloader.py`](https://github.com/pytorch/pytorch/blob/e9295c692a07e709c9070b7fb55cf76e8f2d79c1/torch/utils/data/dataloader.py#L783-L788)

```python
def _next_data(self):
    index = self._next_index()
    data = self._dataset_fetcher.fetch(index)
    if self._pin_memory:
        data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)
    return data
```

## 逐行讲解 / What's happening

1. 中文：索引先由 sampler 决定，dataset 无需知道随机策略。 English: The sampler selects the index first, so the dataset need not know the sampling policy.
2. 中文：`fetch` 同时覆盖单样本和 batch 索引。 English: `fetch` handles both single examples and batched indices.
3. 中文：pin 是可选的最后一步，不污染数据集语义。 English: Pinning is an optional final step and does not pollute dataset semantics.

## 类比 / The analogy

像餐厅出餐：点单决定编号，厨房制作，最后把餐盘放到取餐架。

Like a restaurant: the order chooses a ticket, the kitchen prepares it, then the plate goes to the pickup shelf.

## 自己跑一遍 / Try it yourself

```python
ids = iter([2, 0])
dataset = ["zero", "one", "two"]
def next_batch():
    i = next(ids)
    return dataset[i]
print(next_batch(), next_batch())
```

运行 / Run with: `python try.py`

预期输出 / Expected output: `two zero`

中文：换 sampler 不必改 dataset。 English: Changing the sampler does not require changing the dataset.

## 注意事项 / Caveats / when it breaks

- 中文：pin memory 主要帮助 CPU→CUDA 的异步传输；纯 CPU 工作负载通常不获益。 / English: Pinning mainly helps CPU-to-CUDA transfer; CPU-only workloads usually do not benefit.

## 延伸阅读 / Further reading

- [DataLoader iterator source](https://github.com/pytorch/pytorch/blob/e9295c692a07e709c9070b7fb55cf76e8f2d79c1/torch/utils/data/dataloader.py#L783-L788)
