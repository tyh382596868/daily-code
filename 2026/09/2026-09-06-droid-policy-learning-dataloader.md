---
date: 2026-09-06
topic: robotics
source: tracked
repo: droid-dataset/droid_policy_learning
file: examples/droid_dataloader.py
permalink: https://github.com/droid-dataset/droid_policy_learning/blob/9a29c832b4c81bf38401111f5e4cdddaca217581/examples/droid_dataloader.py#L393-L531
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, rlds, dataloader, success-filter]
---

# DROID dataloader：先筛成功样本，再把多源轨迹交给 RLDS / DROID Dataloader: Filter Success First, Then Hand Multi-Source Trajectories to RLDS

> **一句话 / In one line**: 这段 loader 先做 success filter 和 dataset 配置合并，再把多源轨迹拼成一个 interleaved TorchRLDSDataset。 / This loader first applies success filtering and dataset config merging, then builds one interleaved TorchRLDSDataset from multiple sources.

## 为什么重要 / Why this matters

中文：机器人数据管道最容易乱在两件事上：哪些 episode 算“可训练”，以及多个子集怎么按比例混在一起。这里把成功筛选、统计汇总、混合权重和 PyTorch 数据管线全部收在一个入口里，训练侧就只看一个 DataLoader。

English: Robot data pipelines usually get messy in two places: which episodes count as trainable, and how to mix multiple subsets by ratio. This code keeps success filtering, stat aggregation, mix weights, and the PyTorch pipeline in one entry point, so training only sees one DataLoader.

## 代码 / The code

`droid-dataset/droid_policy_learning` — [`examples/droid_dataloader.py`](https://github.com/droid-dataset/droid_policy_learning/blob/9a29c832b4c81bf38401111f5e4cdddaca217581/examples/droid_dataloader.py#L393-L531)

```python
tf.config.set_visible_devices([], "GPU")

DATA_PATH = Path("/path/to/droid")
DATASET_NAMES = ["droid"]

BASE_DATASET_KWARGS = {
    "shuffle_buffer_size": 1000,
    "image_obs_keys": ("image",),
    "state_obs_keys": ("state",),
}

success_filter = lambda episode: episode["is_success"]

dataset_kwargs_list = [
    {**BASE_DATASET_KWARGS, "name": name, "split": split}
    for name, split in [("droid", "train"), ("droid", "val")]
]

combined_stats = {
    "num_episodes": sum(item["num_episodes"] for item in stats_list),
    "num_success": sum(item["num_success"] for item in stats_list),
}

dataset = make_interleaved_dataset(
    DATA_PATH,
    dataset_kwargs_list,
    success_filter=success_filter,
    stats=combined_stats,
)

dataset = dataset.map(augment_example)
torch_dataset = TorchRLDSDataset(dataset)
dataloader = DataLoader(torch_dataset, batch_size=32, num_workers=0)
```

## 逐行讲解 / What's happening

1. **第 393-437 行 / Lines 393-437**:
   - 中文: 先禁用 GPU 可见性，再把基础数据参数、成功判断和数据源名字定下来。
   - English: The code first hides GPUs, then fixes the base dataset knobs, success predicate, and dataset names.
2. **第 439-460 行 / Lines 439-460**:
   - 中文: 每个 split 都从同一份基础参数派生，统计量也先合并成全局视图。
   - English: Each split derives from the same base config, and the statistics are merged into one global view.
3. **第 462-531 行 / Lines 462-531**:
   - 中文: 真正的拼装发生在 `make_interleaved_dataset()`，最后只留下一个标准 PyTorch dataloader。
   - English: The real assembly happens in `make_interleaved_dataset()`, and the end result is just one standard PyTorch dataloader.

## 类比 / The analogy

中文：像先把不合格零件剔掉，再把不同工厂送来的箱子按配比混到同一条产线。

English: It is like rejecting bad parts first, then mixing boxes from different factories by ratio onto one assembly line.

## 自己跑一遍 / Try it yourself

```python
def interleave(items, keep):
    return [x for x in items if keep(x)]

episodes = [
    {"name": "a", "is_success": True},
    {"name": "b", "is_success": False},
    {"name": "c", "is_success": True},
]
print([e["name"] for e in interleave(episodes, lambda e: e["is_success"])])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
['a', 'c']
```

中文：这个 loader 的核心不是“读文件”，而是把训练前的数据选择和混合规则写成一处。

English: The core trick is not file reading; it is putting data selection and mixing rules in one place before training starts.
