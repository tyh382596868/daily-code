---
date: 2026-09-22
topic: robotics
source: tracked
repo: droid-dataset/droid
file: droid/data_loading/trajectory_sampler.py
permalink: https://github.com/droid-dataset/droid/blob/33ae6a67274f36d2e29525b86f23a56616ef43a7/droid/data_loading/trajectory_sampler.py#L10-L111
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, droid, trajectory-sampling, dataloader]
---

# DROID 轨迹采样：先找可用 episode，再按 worker 分片 / DROID Trajectory Sampling: Find Episodes, Then Partition Workers

> **一句话 / In one line**: DROID 把嵌套数据目录递归收集成轨迹路径，再让每个 DataLoader worker 只在自己的路径区间里随机抽一条轨迹。 / DROID recursively turns a nested data tree into trajectory paths, then lets each DataLoader worker sample only from its own path range.

## 为什么重要 / Why this matters

机器人数据通常不是一个平整的文件列表，而是按日期、任务、机器人和 episode 层层嵌套。`crawler` 先把真正包含 `trajectory.h5` 的目录找出来，还可以在打开 HDF5 属性后过滤失败样本；后面的 `TrajectorySampler` 再把“选哪条轨迹”和“如何处理每个 timestep”分开。

Robot datasets are rarely flat file lists. They are nested by date, task, robot, and episode. `crawler` discovers directories that actually contain `trajectory.h5` and can filter them through HDF5 attributes; `TrajectorySampler` then separates trajectory selection from per-timestep processing.

最值得学的是 worker 边界。worker 不共享一个全局随机游标，而是先根据 `worker_info.id` 算出自己的 `[range_low, range_high)`，再独立抽样。这样多进程加载时，每个 worker 都有清楚的责任范围。

The key idea is the worker boundary. Workers do not share one global random cursor. Each computes its own `[range_low, range_high)` interval from `worker_info.id`, then samples independently inside that interval. The ownership rule is simple and inspectable.

## 代码 / The code

`droid-dataset/droid` — [`droid/data_loading/trajectory_sampler.py`](https://github.com/droid-dataset/droid/blob/33ae6a67274f36d2e29525b86f23a56616ef43a7/droid/data_loading/trajectory_sampler.py#L10-L111)

```python
import os

import h5py
import numpy as np

from droid.data_processing.timestep_processing import TimestepProcesser
from droid.trajectory_utils.misc import load_trajectory


def crawler(dirname, filter_func=None):
    subfolders = [f.path for f in os.scandir(dirname) if f.is_dir()]
    traj_files = [f.path for f in os.scandir(dirname) if (f.is_file() and "trajectory.h5" in f.path)]

    if len(traj_files):
        if filter_func is None:
            use_data = True
        else:
            hdf5_file = h5py.File(traj_files[0], "r")
            use_data = filter_func(hdf5_file.attrs)
            hdf5_file.close()

        if use_data:
            return [dirname]

    all_folderpaths = []
    for child_dirname in subfolders:
        child_paths = crawler(child_dirname, filter_func=filter_func)
        all_folderpaths.extend(child_paths)

    return all_folderpaths


def generate_train_test_split(filter_func=None, remove_failures=True, train_p=0.9):
    all_folderpaths = collect_data_folderpaths(filter_func=filter_func, remove_failures=remove_failures)
    num_train_traj = int(train_p * len(all_folderpaths))
    all_ind = np.random.permutation(len(all_folderpaths))
    training_ind, test_ind = all_ind[:num_train_traj], all_ind[num_train_traj:]
    train_folderpaths, test_folderpaths = [], []

    for i in range(len(all_folderpaths)):
        folderpath = all_folderpaths[i]
        if i in training_ind:
            train_folderpaths.append(folderpath)
        if i in test_ind:
            test_folderpaths.append(folderpath)

    return train_folderpaths, test_folderpaths


def collect_data_folderpaths(filter_func=None, remove_failures=True):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(dir_path, "../../data")
    if remove_failures:
        data_dir = os.path.join(data_dir, "success")
    return crawler(data_dir, filter_func=filter_func)


class TrajectorySampler:
    def __init__(
        self,
        all_folderpaths,
        recording_prefix="",
        traj_loading_kwargs={},
        timestep_filtering_kwargs={},
        image_transform_kwargs={},
        camera_kwargs={},
    ):
        self._all_folderpaths = all_folderpaths
        self.recording_prefix = recording_prefix
        self.traj_loading_kwargs = traj_loading_kwargs
        self.timestep_processer = TimestepProcesser(
            **timestep_filtering_kwargs, image_transform_kwargs=image_transform_kwargs
        )
        self.camera_kwargs = camera_kwargs

    def fetch_samples(self, worker_info=None):
        if worker_info is None:
            range_low, range_high = 0, len(self._all_folderpaths)
        else:
            slice_size = len(self._all_folderpaths) // worker_info.num_workers
            range_low = slice_size * worker_info.id
            range_high = slice_size * (worker_info.id + 1)

        traj_ind = np.random.randint(low=range_low, high=range_high)
        folderpath = self._all_folderpaths[traj_ind]
        filepath = os.path.join(folderpath, "trajectory.h5")
        recording_folderpath = os.path.join(folderpath, "recordings", self.recording_prefix)
        if not os.path.exists(recording_folderpath):
            recording_folderpath = None

        traj_samples = load_trajectory(
            filepath,
            recording_folderpath=recording_folderpath,
            camera_kwargs=self.camera_kwargs,
            **self.traj_loading_kwargs,
        )
        return [self.timestep_processer.forward(t) for t in traj_samples]
```

## 逐行讲解 / What's happening

1. **第 10-31 行 / Lines 10-31 (`crawler`)**:
   - 中文: 先看当前目录有没有轨迹文件；有就把当前目录当作叶子，否则继续递归。过滤器只读属性，不必把整条轨迹加载进内存。
   - English: The crawler treats a directory containing a trajectory file as a leaf; otherwise it recurses. The filter reads only HDF5 attributes, so rejected episodes do not need to be fully loaded.
2. **第 34-51 行 / Lines 34-51 (`generate_train_test_split`)**:
   - 中文: 用一个随机 permutation 生成索引，再按索引分成 train/test。数据路径和随机切分逻辑保持解耦。
   - English: One random permutation creates the split indices. Dataset discovery stays separate from the train/test policy.
3. **第 86-94 行 / Lines 86-94 (`fetch_samples`)**:
   - 中文: 没有 worker 时覆盖全部路径；有 worker 时先计算自己的区间，再从区间内抽一条 trajectory。
   - English: Without worker metadata the sampler can use every path; with it, the worker computes its interval first and samples one trajectory inside it.
4. **第 97-109 行 / Lines 97-109 (load and transform)**:
   - 中文: 录像目录不存在时传 `None`，加载器可以走无视频分支；整条轨迹加载后，再逐 timestep 交给处理器。
   - English: A missing recording directory becomes `None`, allowing the loader to use a no-video path. The full trajectory is loaded once, then each timestep passes through the processor.

## 类比 / The analogy

把数据集想成一栋按楼层和房间组织的仓库。`crawler` 是盘点员，只给“真的有一箱货”的房间登记；多个 worker 像不同搬运工，各自负责一段房间号，抽取一个完整箱子后再逐件贴标签。

Think of the dataset as a warehouse organized by floors and rooms. `crawler` registers only rooms that contain a real shipment; each worker is a mover assigned a room interval, who picks one complete box and labels its contents item by item.

## 自己跑一遍 / Try it yourself

```python
import numpy as np

def worker_range(total, worker_id, workers):
    width = total // workers
    return width * worker_id, width * (worker_id + 1)

np.random.seed(7)
for worker_id in range(3):
    lo, hi = worker_range(12, worker_id, 3)
    print(worker_id, (lo, hi), np.random.randint(lo, hi))
```

运行 / Run with:

```bash
pip install numpy
python try.py
```

预期输出 / Expected output:

```text
0 (0, 4) 3
1 (4, 8) 4
2 (8, 12) 11
```

中文: 先切责任区、再随机抽样，随机性不会改变 worker 的边界。 / English: Randomness chooses the sample, but it never changes which worker owns which interval.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **RLDS episode builders** / **RLDS episode builders**: 中文: 先确定 episode 边界，再把 timestep 映射成训练样本。 / English: They establish episode boundaries first, then map timesteps into training examples.
- **PyTorch `IterableDataset` workers** / **PyTorch `IterableDataset` workers**: 中文: `get_worker_info()` 同样把共享数据源变成 worker-local 的迭代责任。 / English: `get_worker_info()` similarly turns one source into worker-local iteration responsibilities.
- **LeRobot dataset samplers** / **LeRobot dataset samplers**: 中文: 多相机、多 episode 数据也需要把路径发现、分片和 transform 分成三层。 / English: Multi-camera, multi-episode datasets benefit from the same separation of discovery, sharding, and transforms.

## 注意事项 / Caveats / when it breaks

- **整除余数** / **Remainders**: 中文: `len(paths) // num_workers` 会让最后一部分路径可能没有被分到 worker；生产实现应显式处理余数。 / English: Integer division can leave a tail of paths unassigned; production code should handle the remainder explicitly.
- **空区间** / **Empty ranges**: 中文: 路径少于 worker 数时，`np.random.randint` 会失败。 / English: If there are fewer paths than workers, `np.random.randint` can receive an empty interval.
- **默认可变参数** / **Mutable defaults**: 中文: 源码保留了 `{}` 默认值；新代码更适合用 `None` 再在函数体里创建字典。 / English: The source keeps `{}` defaults; new code should prefer `None` and allocate dictionaries inside the function.
- **完整轨迹成本** / **Whole-trajectory cost**: 中文: 一次返回整条轨迹方便时序处理，但也会抬高单个 worker 的峰值内存。 / English: Returning a whole trajectory simplifies temporal processing but raises peak memory per worker.

## 延伸阅读 / Further reading

- [DROID trajectory sampler](https://github.com/droid-dataset/droid/blob/33ae6a67274f36d2e29525b86f23a56616ef43a7/droid/data_loading/trajectory_sampler.py#L10-L111)
- [PyTorch worker information](https://pytorch.org/docs/stable/data.html#torch.utils.data.get_worker_info)
- [DROID dataset repository](https://github.com/droid-dataset/droid)
