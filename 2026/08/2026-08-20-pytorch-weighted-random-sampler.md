---
date: 2026-08-20
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/utils/data/sampler.py
permalink: https://github.com/pytorch/pytorch/blob/bcbe9717ec3afca19c9026f1ad49f0ff6bda69a4/torch/utils/data/sampler.py#L213-L283
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, pytorch, data-loader, sampling]
---

# PyTorch WeightedRandomSampler：权重变成抽样概率 / PyTorch WeightedRandomSampler: Turn Weights into Draw Probabilities

> **一句话 / In one line**: `WeightedRandomSampler` 只保存一维权重、抽样次数和是否放回，真正的采样交给 `torch.multinomial`。 / `WeightedRandomSampler` stores a 1D weight vector, draw count, and replacement flag, then delegates the actual sampling to `torch.multinomial`.

## 为什么重要 / Why this matters

数据不平衡时，你不一定要复制少数类样本。更简单的办法是让 sampler 产生有偏索引流：重要样本权重大，被抽到的概率高；普通 dataset 和 DataLoader 完全不用改。

When data is imbalanced, you do not have to physically duplicate minority examples. A sampler can produce a biased stream of indices: high-weight samples are drawn more often, while the dataset and DataLoader stay unchanged.

## 代码 / The code

`pytorch/pytorch` — [`torch/utils/data/sampler.py`](https://github.com/pytorch/pytorch/blob/bcbe9717ec3afca19c9026f1ad49f0ff6bda69a4/torch/utils/data/sampler.py#L213-L283)

```python
class WeightedRandomSampler(Sampler[int]):
    r"""Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).

    Args:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
        generator (Generator): Generator used in sampling.
    """

    weights: torch.Tensor
    num_samples: int
    replacement: bool

    def __init__(
        self,
        weights: Sequence[float],
        num_samples: int,
        replacement: bool = True,
        generator=None,
    ) -> None:
        if (
            not isinstance(num_samples, int)
            or isinstance(num_samples, bool)
            or num_samples <= 0
        ):
            raise ValueError(
                f"num_samples should be a positive integer value, but got num_samples={num_samples}"
            )
        if not isinstance(replacement, bool):
            raise ValueError(
                f"replacement should be a boolean value, but got replacement={replacement}"
            )

        weights_tensor = torch.as_tensor(weights, dtype=torch.double)
        if len(weights_tensor.shape) != 1:
            raise ValueError(
                "weights should be a 1d sequence but given "
                f"weights have shape {tuple(weights_tensor.shape)}"
            )

        self.weights = weights_tensor
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        rand_tensor = torch.multinomial(
            self.weights, self.num_samples, self.replacement, generator=self.generator
        )
        yield from iter(rand_tensor.tolist())

    def __len__(self) -> int:
        return self.num_samples
```

## 逐行讲解 / What's happening

1. **第 251-262 行 / Lines 251-262 (argument checks)**:
   - 中文: `num_samples` 必须是正整数，`replacement` 必须是 bool，避免把 sampler 放进 DataLoader 后才爆错。
   - English: `num_samples` must be a positive integer and `replacement` must be a bool, so bad configs fail early.
2. **第 264-270 行 / Lines 264-270 (weight normalization contract)**:
   - 中文: 权重转成 double tensor，但不要求和为 1；`multinomial` 会按相对大小解释。
   - English: Weights become a double tensor, but they do not need to sum to 1; `multinomial` interprets relative scale.
3. **第 276-280 行 / Lines 276-280 (`__iter__`)**:
   - 中文: 每次迭代一次性抽出 `num_samples` 个索引，再 `yield from` 变成 Python 索引流。
   - English: Each iteration draws `num_samples` indices in one tensor call, then yields them as a Python index stream.

## 类比 / The analogy

像抽奖箱。每个样本不是只放一张票，而是按权重放不同数量的票；抽完放回就是 `replacement=True`，抽完不放回就是 `False`。

It is like a raffle box. Each example contributes tickets proportional to its weight. Drawing and returning the ticket is `replacement=True`; drawing without returning it is `False`.

## 自己跑一遍 / Try it yourself

```python
import random

weights = [1, 1, 8]
names = ["common_a", "common_b", "rare"]
draws = random.choices(range(len(weights)), weights=weights, k=10)
print([names[i] for i in draws])
print("rare_count", sum(1 for i in draws if i == 2))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
['rare', ...]
rare_count usually larger than the others
```

中文: 输出有随机性，但 `rare` 的权重大约占 80%，所以会频繁出现。

English: The output is random, but `rare` has about 80% of the weight mass and should appear frequently.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **imbalanced classification** / **imbalanced classification**: 少数类样本权重大，训练 batch 更常看到它们。 / Minority-class examples get larger weights, so batches see them more often.
- **preference data mixing** / **preference data mixing**: 高质量或新鲜数据可以通过 sampler 提高出现频率。 / High-quality or fresh data can be upsampled by sampler weight.

## 注意事项 / Caveats / when it breaks

- **不等于 loss reweighting** / **Not the same as loss reweighting**: sampler 改变看到样本的频率，loss 权重改变每个样本的梯度大小。 / Sampling changes example frequency; loss weights change gradient scale.
- **无放回有硬上限** / **Without replacement has a hard limit**: `replacement=False` 时不能抽出超过非零权重样本数量的有效唯一索引。 / Without replacement, you cannot draw more unique useful indices than the positive-weight population supports.

## 延伸阅读 / Further reading

- [PyTorch `WeightedRandomSampler`](https://github.com/pytorch/pytorch/blob/bcbe9717ec3afca19c9026f1ad49f0ff6bda69a4/torch/utils/data/sampler.py#L213-L283)
- [PyTorch `torch.multinomial`](https://pytorch.org/docs/stable/generated/torch.multinomial.html)
