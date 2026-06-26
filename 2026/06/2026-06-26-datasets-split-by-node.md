---
date: 2026-06-26
topic: huggingface
source: huggingface
repo: huggingface/datasets
file: src/datasets/distributed.py
permalink: https://github.com/huggingface/datasets/blob/b713dcdffa92ada37c569e6f1419ce94fc170b0c/src/datasets/distributed.py
difficulty: beginner
read_time: ~8 min
tags: [code-of-the-day, huggingface, datasets, distributed-training, data-loading, ddp, fsdp]
---

# `split_dataset_by_node`：一个 API 背后藏着两种完全不同的分布式数据切分策略 / `split_dataset_by_node`: One API, Two Fundamentally Different Distribution Strategies Hidden Inside

> **一句话 / In one line**: `split_dataset_by_node(dataset, rank, world_size)` 为 DDP/FSDP 的每个节点切分出专属的数据子集，但 map-style 数据集用连续索引块（磁盘局部性优先），iterable 数据集用 shard 整除分配或 1-of-N 跳跃采样（流式吞吐优先）。 / `split_dataset_by_node(dataset, rank, world_size)` assigns each DDP/FSDP node its own data slice, but map-style datasets use contiguous index chunks (disk-locality first) while iterable datasets use shard-level or 1-in-N strided assignment (streaming throughput first).

## 为什么重要 / Why this matters

多 GPU 训练时，最容易踩的坑之一就是**数据重复或遗漏**：8 块 GPU 全都加载相同的数据，或者某些 batch 被跳过了。`HuggingFace Trainer` 内部就是用这个函数来保证每块 GPU 只看到属于自己的那份数据，不多不少。

理解这个函数能帮你回答三个实际问题：(1) 为什么不能对 IterableDataset shuffle 后忘记固定 seed？(2) 为什么 iterable 数据集的 shard 数量最好是 GPU 数量的整数倍？(3) map-style 和 iterable 数据集在分布式场景下的本质区别是什么？

One of the most common distributed training bugs is **data duplication or gaps**: all 8 GPUs load the same data, or some batches get dropped. HuggingFace Trainer uses this function internally to ensure each GPU sees exactly its own portion of the data, no more, no less.

Understanding this function answers three practical questions: (1) Why can't you shuffle an IterableDataset without fixing the seed in a distributed setup? (2) Why should the number of iterable shards ideally be a multiple of the GPU count? (3) What is the fundamental difference between map-style and iterable datasets in distributed settings?

## 代码 / The code

`huggingface/datasets` — [`src/datasets/distributed.py`](https://github.com/huggingface/datasets/blob/b713dcdffa92ada37c569e6f1419ce94fc170b0c/src/datasets/distributed.py)

```python
from typing import TypeVar

from .arrow_dataset import Dataset, _split_by_node_map_style_dataset
from .iterable_dataset import IterableDataset, _split_by_node_iterable_dataset


DatasetType = TypeVar("DatasetType", Dataset, IterableDataset)


def split_dataset_by_node(dataset: DatasetType, rank: int, world_size: int) -> DatasetType:
    """
    Split a dataset for the node at rank `rank` in a pool of nodes of size `world_size`.

    For map-style datasets:

    Each node is assigned a chunk of data, e.g. rank 0 is given the first chunk of the dataset.
    To maximize data loading throughput, chunks are made of contiguous data on disk if possible.

    For iterable datasets:

    If the dataset has a number of shards that is a factor of `world_size` (i.e. if
    `dataset.num_shards % world_size == 0`), then the shards are evenly assigned across
    the nodes, which is the most optimized.
    Otherwise, each node keeps 1 example out of `world_size`, skipping the other examples.

    > [!WARNING]
    > If you shuffle your iterable dataset in a distributed setup, make sure to set a fixed
    > `seed` in `IterableDataset.shuffle` so the same shuffled list of shards is used on
    > every node to know which shards the node should skip.

    Args:
        dataset (`Dataset` or `IterableDataset`): The dataset to split by node.
        rank (`int`): Rank of the current node.
        world_size (`int`): Total number of nodes.

    Returns:
        `Dataset` or `IterableDataset`: The dataset to be used on the node at rank `rank`.
    """
    if isinstance(dataset, Dataset):
        return _split_by_node_map_style_dataset(dataset, rank=rank, world_size=world_size)
    else:
        return _split_by_node_iterable_dataset(dataset, rank=rank, world_size=world_size)
```

## 逐行讲解 / What's happening

1. **`DatasetType = TypeVar("DatasetType", Dataset, IterableDataset)`**
   - 中文: 限制性 TypeVar（不是无约束的 `T`）——调用时传入 `Dataset` 则返回 `Dataset`，传入 `IterableDataset` 则返回 `IterableDataset`，类型检查器能正确推断。
   - English: A constrained TypeVar — pass in a `Dataset` and you get back a `Dataset`, pass an `IterableDataset` and you get an `IterableDataset`. Type checkers can infer the return type precisely, unlike an unconstrained `T`.

2. **`if isinstance(dataset, Dataset):`  →  `_split_by_node_map_style_dataset`**
   - 中文: map-style 数据集（Arrow 格式存在磁盘上）的切分策略是**连续索引块**：rank 0 拿索引 `[0, N//W)`，rank 1 拿 `[N//W, 2*N//W)`，以此类推。这样每个节点读的都是磁盘上连续的一段，利用操作系统预读（read-ahead）最大化 I/O 吞吐。
   - English: Map-style datasets (stored as Arrow on disk) are split into **contiguous index chunks**: rank 0 gets indices `[0, N//W)`, rank 1 gets `[N//W, 2*N//W)`, etc. Each node reads a contiguous region on disk, letting OS read-ahead maximize I/O throughput.

3. **`else:` → `_split_by_node_iterable_dataset`**
   - 中文: Iterable 数据集（流式、无随机访问）有两种策略：
     - **整除优先（最优）**: 如果 `dataset.num_shards % world_size == 0`，则把 shard 整块分配给各节点（rank 0 拿 shard 0~k，rank 1 拿 shard k+1~2k……）。每个节点只需顺序读它的那几个 shard，零额外 I/O 开销。
     - **1-of-N 跳跃采样（降级策略）**: 如果 shard 数量不整除，退化为：rank r 的节点只保留 `index % world_size == r` 的样本，丢弃其他的。代价是每个节点要读取**全量**数据流，但只用其中 1/N。
   - English: Iterable datasets (streaming, no random access) have two strategies:
     - **Shard-divisible (optimal)**: if `num_shards % world_size == 0`, assign whole shards to each rank. Rank r reads only its assigned shards sequentially — zero extra I/O.
     - **1-in-N striding (fallback)**: if shards aren't divisible, rank r keeps only examples where `index % world_size == r`, skipping the rest. Cost: each node reads the **full** data stream, using only 1/N of it.

4. **WARNING: 固定 `seed` 的必要性**
   - 中文: 这个警告非常重要。如果你 shuffle 了 IterableDataset（内部会打乱 shard 顺序），但没有固定 seed，那么 rank 0 和 rank 1 看到的 shard 顺序不同，1-of-N 跳跃采样就会基于不同的顺序计算 `index % world_size`，导致两个节点拿到的是同一批数据（或者有遗漏）。
   - English: Critical warning. If you shuffle an IterableDataset (which shuffles shard order) without a fixed seed, rank 0 and rank 1 see different shard orderings. The 1-in-N strided sampler then computes `index % world_size` against different orderings, causing data duplication or gaps across ranks.

## 类比 / The analogy

想象一个图书馆的 8 名馆员要分头整理 1000 本书。如果书已经上架（map-style），最聪明的做法是每人负责书架上连续的一段（馆员 A 负责第 1-125 本，馆员 B 负责第 126-250 本……），这样每人走最少的路（磁盘连续读）。如果书还在运货卡车上一件一件卸下来（iterable），最好的情况是 8 辆卡车各装了一批书（8 个整数倍的 shard），每辆车直接分给一名馆员；如果只有 5 辆卡车（不整除），就只能让 8 人都站在一条传送带旁边，第 1 本给馆员 A，第 2 本给馆员 B……每人都得看完所有书，只拿属于自己的那些。

Imagine 8 library assistants splitting 1,000 books. If the books are already shelved (map-style), the smart move is for each person to cover a contiguous shelf section — minimal walking (contiguous disk reads). If books are still being unloaded from trucks one by one (iterable), the best case is 8 trucks each carrying a separate batch (shard-divisible), so each truck goes directly to one assistant. If there are only 5 trucks (not divisible), all 8 assistants must stand at the same conveyor belt — book 1 goes to assistant A, book 2 to B, etc. — everyone sees everything but only keeps their 1-in-8.

## 自己跑一遍 / Try it yourself

```python
from datasets import Dataset, IterableDataset
from datasets.distributed import split_dataset_by_node

# 1. Map-style: contiguous chunks
ds = Dataset.from_dict({"x": list(range(100))})
rank0 = split_dataset_by_node(ds, rank=0, world_size=4)
rank1 = split_dataset_by_node(ds, rank=1, world_size=4)
print(rank0[0], rank0[-1])   # {'x': 0}  {'x': 24}
print(rank1[0], rank1[-1])   # {'x': 25} {'x': 49}
assert len(rank0) + len(rank1) == 50  # each node gets N/world_size

# 2. Iterable: stride fallback (since we only have 1 implicit shard)
it_ds = IterableDataset.from_generator(lambda: ({"x": i} for i in range(20)))
it_rank0 = split_dataset_by_node(it_ds, rank=0, world_size=4)
print([row["x"] for row in it_rank0])  # [0, 4, 8, 12, 16]  (1-in-4 stride)
```

运行 / Run with:
```bash
pip install datasets
python try.py
```

预期输出 / Expected output:
```
{'x': 0} {'x': 24}
{'x': 25} {'x': 49}
[0, 4, 8, 12, 16]
```

中文：map-style 的 rank0 拿 [0,24]，rank1 拿 [25,49]，连续分布。iterable（单 shard）退化为 1-in-4 跳跃：rank0 只得到索引 0, 4, 8, 12, 16，这意味着每个节点要遍历全部 20 个样本才能筛选出自己的 5 个。

English: Map-style rank0 gets [0,24] and rank1 gets [25,49] — contiguous. The iterable dataset (single shard, not divisible by 4) falls back to 1-in-4 striding: rank0 sees only indices 0, 4, 8, 12, 16, meaning every node iterates all 20 examples just to pick its 5.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **PyTorch `DistributedSampler`** / **PyTorch `DistributedSampler`**: `torch.utils.data.distributed.DistributedSampler` 也用类似的连续/交错切分，但只支持 map-style（有 `__len__`），不支持 iterable。`split_dataset_by_node` 统一了两种情况。 / `torch.utils.data.distributed.DistributedSampler` uses a similar contiguous/interleaved split but only for map-style datasets with `__len__`. `split_dataset_by_node` unifies both.
- **WebDataset 的 shard 策略** / **WebDataset shard strategy**: `webdataset.split_by_node` 同样优先整 shard 分配，退化时用交错采样——和这里的 iterable 路径几乎完全相同。 / `webdataset.split_by_node` also prefers whole-shard assignment and falls back to interleaved sampling — nearly identical to the iterable path here.
- **HuggingFace Trainer 内部** / **Inside HuggingFace Trainer**: `Trainer._get_train_sampler()` 在 `is_world_process_zero()` 为 `False` 时调用 `split_dataset_by_node`，保证所有 rank 的数据集加起来不重叠、不遗漏地覆盖整个训练集。 / `Trainer._get_train_sampler()` calls `split_dataset_by_node` on non-zero ranks to ensure all ranks' datasets partition the full training set without overlap or gaps.

## 注意事项 / Caveats / when it breaks

- **IterableDataset + shuffle 必须固定 seed** / **IterableDataset + shuffle requires a fixed seed**: 如代码中的 WARNING 所述，`dataset.shuffle(seed=42)` 的 `seed` 参数是必须的，否则各节点的 shard 顺序不同，1-of-N 采样失效。 / As the WARNING states, `dataset.shuffle(seed=42)` requires the `seed` argument; otherwise different ranks see different shard orderings and the 1-in-N sampler produces duplicates or gaps.
- **shard 数量建议是 world_size 的整数倍** / **Shard count should be a multiple of world_size**: 如果 shard 数量不整除，退化到 1-of-N 跳跃采样，每个节点要读 N 倍的数据量。对于 TB 级数据集这是不可接受的。建议在数据集准备阶段就按 `world_size` 对齐 shard 数量。 / If shards aren't divisible, the fallback reads N× more data per node. For TB-scale datasets this is unacceptable. Align shard counts to multiples of `world_size` during dataset preparation.
- **最后一个 batch 不足 world_size 时** / **When the last batch is smaller than world_size**: map-style 的连续切分在数据量不整除 world_size 时，最后一个 rank 可能少几个样本。`_split_by_node_map_style_dataset` 内部会通过 `drop_last` 或 padding 来处理，具体行为取决于下游的 DataLoader 配置。 / For map-style datasets when total examples isn't divisible by world_size, the last rank may have fewer examples. The internal `_split_by_node_map_style_dataset` handles this via `drop_last` or padding depending on downstream DataLoader config.

## 延伸阅读 / Further reading

- [HuggingFace Datasets distributed training guide](https://huggingface.co/docs/datasets/en/use_with_pytorch#distributed-setups)
- [PyTorch DistributedSampler docs](https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler)
- [WebDataset shard-based distributed training](https://github.com/webdataset/webdataset#splitting-shards-across-workers-and-nodes)
