---
date: 2026-10-12
topic: huggingface
source: huggingface
repo: huggingface/datasets
file: src/datasets/utils/sharding.py
permalink: https://github.com/huggingface/datasets/blob/9e7496a4d11caf68da2846d84c9596cda4a4f984/src/datasets/utils/sharding.py#L4-L92
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, datasets, sharding]
---

# Datasets gen_kwargs sharding：多文件参数要一起切 / Datasets gen_kwargs Sharding: Split Multi-File Arguments Together

> **一句话 / In one line**: Datasets 把 `gen_kwargs` 里的等长列表视作一组 shard，并按 job 均匀切分，同时拒绝长度不一致的歧义输入。 / Datasets treats equal-length lists in `gen_kwargs` as aligned shards, splits them evenly across jobs, and rejects ambiguous inputs with mismatched lengths.

## 为什么重要 / Why this matters

数据集 builder 常常不只是一个文件列表。它可能有 `files`、`metadata_files`、`checksums` 这些并行列表，每个位置彼此对应。并行加载时不能只切 `files`，否则 metadata 会和数据错位。这里的 sharding 工具先确认所有列表长度一致，再按连续 range 切分，让每个 worker 拿到对齐的一小组参数。

A dataset builder often has more than one list. It may carry `files`, `metadata_files`, and `checksums`, where each position corresponds to the same shard. Parallel loading cannot split only `files`, or metadata will drift away from data. This sharding utility first checks that list lengths agree, then splits contiguous ranges so each worker receives aligned arguments.

## 代码 / The code

`huggingface/datasets` - [`src/datasets/utils/sharding.py`](https://github.com/huggingface/datasets/blob/9e7496a4d11caf68da2846d84c9596cda4a4f984/src/datasets/utils/sharding.py#L4-L92)

```python
def _number_of_shards_in_gen_kwargs(gen_kwargs: dict) -> int:
    """Return the number of possible shards according to the input gen_kwargs"""
    # Having lists of different sizes makes sharding ambigious, raise an error in this case
    # until we decide how to define sharding without ambiguity for users
    lists_lengths = {key: len(value) for key, value in gen_kwargs.items() if isinstance(value, list)}
    if len(set(lists_lengths.values())) > 1:
        raise RuntimeError(
            "Sharding is ambiguous for this dataset: "
            + "we found several data sources lists of different lengths, and we don't know over which list we should parallelize:\n"
            + "\n".join(f"\t- key {key} has length {length}" for key, length in lists_lengths.items())
            + "\nTo fix this, check the 'gen_kwargs' and make sure to use lists only for data sources, "
            + "and use tuples otherwise. In the end there should only be one single list, or several lists with the same length."
        )
    max_length = max(lists_lengths.values(), default=0)
    return max(1, max_length)


def _distribute_shards(num_shards: int, max_num_jobs: int) -> list[range]:
    """
    Get the range of shard indices per job.
    If num_shards<max_num_jobs, then num_shards jobs are given a range of one shard.
    The shards indices order is preserved: e.g. all the first shards are given the first job.
    Moreover all the jobs are given approximately the same number of shards.

    Example:

    ```python
    >>> _distribute_shards(2, max_num_jobs=4)
    [range(0, 1), range(1, 2)]
    >>> _distribute_shards(10, max_num_jobs=3)
    [range(0, 4), range(4, 7), range(7, 10)]
    ```
    """
    shards_indices_per_group = []
    for group_idx in range(max_num_jobs):
        num_shards_to_add = num_shards // max_num_jobs + (group_idx < (num_shards % max_num_jobs))
        if num_shards_to_add == 0:
            break
        start = shards_indices_per_group[-1].stop if shards_indices_per_group else 0
        shard_indices = range(start, start + num_shards_to_add)
        shards_indices_per_group.append(shard_indices)
    return shards_indices_per_group


def _split_gen_kwargs(gen_kwargs: dict, max_num_jobs: int) -> list[dict]:
    """Split the gen_kwargs into `max_num_job` gen_kwargs"""
    # Having lists of different sizes makes sharding ambigious, raise an error in this case
    num_shards = _number_of_shards_in_gen_kwargs(gen_kwargs)
    if num_shards == 1:
        return [dict(gen_kwargs)]
    else:
        shard_indices_per_group = _distribute_shards(num_shards=num_shards, max_num_jobs=max_num_jobs)
        return [
            {
                key: [value[shard_idx] for shard_idx in shard_indices_per_group[group_idx]]
                if isinstance(value, list)
                else value
                for key, value in gen_kwargs.items()
            }
            for group_idx in range(len(shard_indices_per_group))
        ]


def _merge_gen_kwargs(gen_kwargs_list: list[dict]) -> dict:
    return {
        key: [value for gen_kwargs in gen_kwargs_list for value in gen_kwargs[key]]
        if isinstance(gen_kwargs_list[0][key], list)
        else gen_kwargs_list[0][key]
        for key in gen_kwargs_list[0]
    }


def _shuffle_gen_kwargs(rng: np.random.Generator, gen_kwargs: dict) -> dict:
    """Return a shuffled copy of the input gen_kwargs"""
    # We must shuffle all the lists, and lists of the same size must have the same shuffling.
    # This way entangled lists of (shard, shard_metadata) are still in the right order.

    # First, let's generate the shuffled indices per list size
    list_sizes = {len(value) for value in gen_kwargs.values() if isinstance(value, list)}
    indices_per_size = {}
    for size in list_sizes:
        indices_per_size[size] = list(range(size))
        rng.shuffle(indices_per_size[size])
    # Now let's copy the gen_kwargs and shuffle the lists based on their sizes
    shuffled_kwargs = dict(gen_kwargs)
    for key, value in shuffled_kwargs.items():
        if isinstance(value, list):
            shuffled_kwargs[key] = [value[i] for i in indices_per_size[len(value)]]
    return shuffled_kwargs
```

## 逐行讲解 / What's happening

1. **第 4-18 行 / Lines 4-18 (`_number_of_shards_in_gen_kwargs`)**:
   - 中文: 它只把 `list` 当成可切分数据源，tuple 等其他对象会原样复制。如果多个 list 长度不同，就直接报错，因为系统不知道应该沿哪条列表切。
   - English: Only `list` values are treated as shardable data sources; tuples and other values are copied as constants. If lists have different lengths, the function errors because there is no unambiguous split axis.

2. **第 21-45 行 / Lines 21-45 (`_distribute_shards`)**:
   - 中文: 每个 job 拿连续 range，余数从前面的 job 开始多分一个。`10` 个 shard 给 `3` 个 job 时就是 `4,3,3`。
   - English: Each job receives a contiguous range. Remainders go to earlier jobs, so `10` shards over `3` jobs become `4,3,3`.

3. **第 48-64 行 / Lines 48-64 (`_split_gen_kwargs`)**:
   - 中文: 对每个 job，它遍历所有 key；list 就按该 job 的 range 切，非 list 就复制同一个值。
   - English: For each job, it walks every key. Lists are sliced by that job's range; non-lists are copied unchanged.

4. **第 76-92 行 / Lines 76-92 (`_shuffle_gen_kwargs`)**:
   - 中文: shuffle 时按“列表长度”共享同一个 permutation，这样同长度的文件列表和 metadata 列表仍然对齐。
   - English: Shuffling shares one permutation per list length, so equal-length file and metadata lists stay aligned.

## 类比 / The analogy

这像把一摞成套文件分给几个人。每套里有申请表、附件和收据，分发时要按套切，不能把申请表给 A、附件给 B。代码里的等长列表就是“同一套文件”的不同页。

It is like distributing packets of paperwork. Each packet has a form, an attachment, and a receipt. You split by packet, not by giving forms to one person and receipts to another. Equal-length lists are the pages of the same packet.

## 自己跑一遍 / Try it yourself

```python
def distribute(n, jobs):
    out = []
    for i in range(jobs):
        count = n // jobs + (i < n % jobs)
        if count == 0:
            break
        start = out[-1].stop if out else 0
        out.append(range(start, start + count))
    return out

def split_kwargs(kwargs, jobs):
    ranges = distribute(len(next(v for v in kwargs.values() if isinstance(v, list))), jobs)
    return [{k: [v[i] for i in r] if isinstance(v, list) else v for k, v in kwargs.items()} for r in ranges]

print(split_kwargs({"files": ["a", "b", "c"], "meta": [1, 2, 3], "split": "train"}, 2))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[{'files': ['a', 'b'], 'meta': [1, 2], 'split': 'train'}, {'files': ['c'], 'meta': [3], 'split': 'train'}]
```

`files` 和 `meta` 被同一组索引切开，`split` 则复制到每个 job。并行加载时，这种小心翼翼的对齐比“平均切文件”更重要。

`files` and `meta` are cut with the same indices, while `split` is copied to every job. In parallel loading, that alignment matters more than simply dividing files evenly.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **WebDataset tar shards** / **WebDataset tar shards**: tar 文件、样本 key、decoder 配置也需要保持同一个 shard 边界。 / Tar files, sample keys, and decoder options need to preserve shard boundaries too.
- **Distributed sampler** / **Distributed sampler**: PyTorch `DistributedSampler` 也先定义全局顺序，再按 rank 切片，避免不同 rank 看到错位数据。 / PyTorch `DistributedSampler` also defines a global order first and slices by rank so ranks do not see misaligned samples.

## 注意事项 / Caveats / when it breaks

- **只有 list 会被切** / **Only lists are split**: 如果你把数据源放进 tuple，它会被当成常量复制。 / If a data source is stored in a tuple, it is copied as a constant.
- **不同长度列表会报错** / **Mismatched list lengths error out**: 这是设计选择，不是限制。它宁愿早失败，也不猜哪个列表才是主 shard。 / This is intentional. The utility fails early instead of guessing which list defines the primary shard axis.

## 延伸阅读 / Further reading

- [Source permalink](https://github.com/huggingface/datasets/blob/9e7496a4d11caf68da2846d84c9596cda4a4f984/src/datasets/utils/sharding.py#L4-L92)
- [Hugging Face Datasets repository](https://github.com/huggingface/datasets)

