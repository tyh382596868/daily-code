---
date: 2026-07-13
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/utils/data/sampler.py
permalink: https://github.com/pytorch/pytorch/blob/84cc6889912b47a5a6161d9697530949c1eb8fb5/torch/utils/data/sampler.py#L120-L188
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, pytorch, data-loading]
---

# PyTorch RandomSampler：可变长度数据集也能延迟求长度 / PyTorch RandomSampler: Delay Dataset Length Until Iteration

> **一句话 / In one line**: `RandomSampler` 把 `num_samples` 做成属性，迭代时才读 `len(data_source)`，所以数据集长度运行中变化也能被看见。 / `RandomSampler` makes `num_samples` a property and reads `len(data_source)` at iteration time, so runtime dataset-size changes can be observed.

## 为什么重要 / Why this matters

很多人以为 sampler 初始化后就冻结了数据集长度。PyTorch 这里更谨慎：默认 `num_samples=None` 时不缓存长度，而是在每次访问 `num_samples` 时重新读 `len(data_source)`。这对动态数据集、在线采样、或者训练中替换底层索引的场景很实用。

Many users assume a sampler freezes dataset length at construction. PyTorch is more careful: when `num_samples=None`, it does not cache the length and instead asks `len(data_source)` whenever `num_samples` is read. That matters for dynamic datasets and online index refresh.

## 代码 / The code

`pytorch/pytorch` — [`torch/utils/data/sampler.py`](https://github.com/pytorch/pytorch/blob/84cc6889912b47a5a6161d9697530949c1eb8fb5/torch/utils/data/sampler.py#L120-L188)

```python
class RandomSampler(Sampler[int]):
    def __init__(self, data_source, replacement=False, num_samples=None, generator=None) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator
        if not isinstance(self.replacement, bool):
            raise TypeError(f"replacement should be a boolean value, but got replacement={self.replacement}")
        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(f"num_samples should be a positive integer value, but got num_samples={self.num_samples}")

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
        else:
            for _ in range(self.num_samples // n):
                yield from torch.randperm(n, generator=generator).tolist()
            yield from torch.randperm(n, generator=generator).tolist()[: self.num_samples % n]
```

## 逐行讲解 / What's happening

1. **`_num_samples` 保存用户显式值 / `_num_samples` stores explicit input**:
   - 中文: 用户传了固定样本数就用固定值；没传就每次回到数据集问长度。
   - English: If the user passes an explicit count, use it; otherwise ask the dataset for its current length.
2. **generator 懒创建 / Lazy generator creation**:
   - 中文: 没有外部 generator 时，采样器从 PyTorch 随机源取一个 seed，创建本次迭代的 generator。
   - English: Without an external generator, the sampler draws a seed and creates a generator for this iterator.
3. **replacement 分两条路 / Replacement splits the algorithm**:
   - 中文: 有放回采样走 `randint`；无放回采样走 `randperm`，需要更多样本时就多轮 permutation。
   - English: Sampling with replacement uses `randint`; sampling without replacement uses `randperm`, repeating permutations if more samples are requested.

## 类比 / The analogy

像电影院检票员不在开门前死记座位数，而是每次放人前看一眼当前售票系统。临时加座、退票、换厅，都不会让检票逻辑过期。

It is like a theater usher checking the live ticketing system instead of memorizing capacity before doors open. Added seats, refunds, or room changes do not stale the admission logic.

## 自己跑一遍 / Try it yourself

```python
class Dynamic:
    def __init__(self, n): self.n = n
    def __len__(self): return self.n

class Sampler:
    def __init__(self, data, num_samples=None):
        self.data, self._num_samples = data, num_samples
    @property
    def num_samples(self):
        return len(self.data) if self._num_samples is None else self._num_samples

d = Dynamic(3)
s = Sampler(d)
print(s.num_samples)
d.n = 5
print(s.num_samples)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
3
5
```

重点是 `num_samples` 不是初始化时写死的字段，而是一个小查询。

The point is that `num_samples` is not a frozen field; it is a small query.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **DistributedSampler** / **DistributedSampler**: 也把“如何补齐、如何切分”推迟到每个 epoch 的迭代路径。 / It also pushes padding and rank slicing into the per-epoch iteration path.
- **Streaming datasets** / **Streaming datasets**: 动态长度和惰性读取通常比提前物化更稳。 / Dynamic length and lazy reads are usually safer than early materialization.

## 注意事项 / Caveats / when it breaks

- **无放回路径要求 `n > 0`** / **the no-replacement path needs `n > 0`**: 空数据集没有可排列的索引。 / An empty dataset has no indices to permute.
- **多 worker 仍要管 seed** / **multi-worker still needs seed discipline**: 可复现训练要显式传 generator 或控制全局 seed。 / Reproducible training should pass a generator or control global seeding.

## 延伸阅读 / Further reading

- PyTorch sampler source — https://github.com/pytorch/pytorch/blob/84cc6889912b47a5a6161d9697530949c1eb8fb5/torch/utils/data/sampler.py
