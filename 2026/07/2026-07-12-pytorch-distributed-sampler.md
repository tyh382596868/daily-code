---
date: 2026-07-12
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/utils/data/distributed.py
permalink: https://github.com/pytorch/pytorch/blob/45e286e38836873fcbff6d13262e97faf1e8bfaa/torch/utils/data/distributed.py#L103-L151
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, pytorch, distributed-sampler]
---

# PyTorch DistributedSampler：先补齐，再按 rank 切片 / PyTorch DistributedSampler: Pad First, Then Slice by Rank

> **一句话 / In one line**: 分布式采样器先构造一个全局 index 序列，再用 `rank : total_size : num_replicas` 给每张卡取互斥子序列。 / The distributed sampler builds one global index list, then uses `rank : total_size : num_replicas` to give each worker a disjoint slice.

## 为什么重要 / Why this matters

分布式训练的数据问题不是“随机”这么简单，而是“每个 rank 数量一样、样本不重叠、每个 epoch 顺序可变”。这段 `__iter__` 把这些约束压成几步：确定性 shuffle、补齐或截断、最后 stride 切片。

Distributed data loading is not just randomness; each rank needs the same number of samples, non-overlapping work, and a new order each epoch. This `__iter__` turns those constraints into deterministic shuffle, pad-or-trim, and stride slicing.

## 代码 / The code

`pytorch/pytorch` — [`torch/utils/data/distributed.py`](https://github.com/pytorch/pytorch/blob/45e286e38836873fcbff6d13262e97faf1e8bfaa/torch/utils/data/distributed.py#L103-L151)

```python
def __iter__(self) -> Iterator[_T_co]:
    if self.shuffle:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()
    else:
        indices = list(range(len(self.dataset)))

    if not self.drop_last:
        padding_size = self.total_size - len(indices)
        if padding_size <= len(indices):
            indices += indices[:padding_size]
        else:
            indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
    else:
        indices = indices[: self.total_size]
    if len(indices) != self.total_size:
        raise AssertionError(
            f"Number of indices ({len(indices)}) does not match total_size ({self.total_size})"
        )

    indices = indices[self.rank : self.total_size : self.num_replicas]
    if len(indices) != self.num_samples:
        raise AssertionError(
            f"Number of subsampled indices ({len(indices)}) does not match num_samples ({self.num_samples})"
        )

    return iter(indices)
```

## 逐行讲解 / What's happening

1. **统一随机源 / Shared random source**:
   - 中文: 每个 rank 都用同一个 `seed + epoch`，所以大家看到的全局 shuffle 顺序一致。
   - English: Every rank uses the same `seed + epoch`, so all workers see the same global shuffled order.
2. **补齐总长度 / Make length divisible**:
   - 中文: 不 `drop_last` 时会把开头样本复制到尾部，让总数能被 rank 数整除。
   - English: Without `drop_last`, the beginning of the list is repeated at the end so total length divides evenly by replica count.
3. **按 rank stride / Stride by rank**:
   - 中文: rank 0 取 0、N、2N，rank 1 取 1、N+1、2N+1；这比手写分块更少边界错误。
   - English: Rank 0 takes 0, N, 2N; rank 1 takes 1, N+1, 2N+1. This avoids many manual partitioning edge cases.

## 类比 / The analogy

像给三个人发一叠洗好的扑克牌：先把牌洗成同一个顺序，不够整除就从牌头补几张，然后轮流发牌。每个人拿到的张数一样，也不会抢同一个位置。

It is like dealing cards to three people: shuffle one deck order, pad from the front if needed, then deal round-robin. Everyone gets the same count and no one competes for the same slot.

## 自己跑一遍 / Try it yourself

```python
import math, random

def shard(n, replicas, rank, seed=0, epoch=2):
    rng = random.Random(seed + epoch)
    indices = list(range(n))
    rng.shuffle(indices)
    total = math.ceil(n / replicas) * replicas
    indices += indices[: total - n]
    return indices[rank:total:replicas]

print(shard(10, 3, 0))
print(shard(10, 3, 1))
print(shard(10, 3, 2))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
三个长度相同的列表 / three lists with equal length
```

关键现象是三个 rank 的长度一致，补齐样本只出现在必要位置。

The important behavior is that all ranks get equal lengths, with repeated samples only when padding is required.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **HF datasets sharding** / **HF datasets sharding**: 流式数据也需要先定义全局顺序，再按 worker 拆分。 / Streaming datasets also define a global order before partitioning across workers.
- **训练 checkpoint resume** / **training checkpoint resume**: `set_epoch` 这种显式 epoch 状态让恢复训练时顺序可重建。 / Explicit epoch state makes sample order reproducible after resuming.

## 注意事项 / Caveats / when it breaks

- **忘记 `set_epoch`** / **forgetting `set_epoch`**: 每个 epoch 会重复同一套 shuffle。 / Each epoch will reuse the same shuffled order.
- **数据集长度变化** / **changing dataset length**: 这个 sampler 假设 dataset 长度稳定。 / This sampler assumes a stable dataset length.

## 延伸阅读 / Further reading

- PyTorch source — https://github.com/pytorch/pytorch/blob/45e286e38836873fcbff6d13262e97faf1e8bfaa/torch/utils/data/distributed.py
