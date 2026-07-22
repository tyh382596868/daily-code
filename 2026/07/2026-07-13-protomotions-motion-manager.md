---
date: 2026-07-13
topic: robotics
source: trending
repo: NVlabs/ProtoMotions
file: protomotions/envs/motion_manager/motion_manager.py
permalink: https://github.com/NVlabs/ProtoMotions/blob/49fe5ad69de67ebbc07ea2b25d41b0f622c15c3c/protomotions/envs/motion_manager/motion_manager.py#L1-L170
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, motion-imitation]
---

# ProtoMotions MotionManager：给并行环境分配参考动作 / ProtoMotions MotionManager: Assign Reference Motions to Parallel Environments

> **一句话 / In one line**: `MotionManager` 维护每个环境当前模仿哪条 motion、走到哪个时间，并支持 subset/exclusion 让评估和训练更可控。 / `MotionManager` tracks which motion each environment imitates and at what time, while subset/exclusion controls make training and evaluation more targeted.

## 为什么重要 / Why this matters

人形机器人模仿学习不是只把 motion dataset 喂给 policy。并行仿真里每个 environment 都要有自己的 motion id、motion time、采样权重和排除规则；否则你很难复现实验，也很难只测某一组动作。

Humanoid imitation learning is not just feeding a motion dataset to a policy. In parallel simulation, each environment needs its own motion id, motion time, sampling weights, and exclusion rules; otherwise experiments are hard to reproduce and targeted evaluation is awkward.

## 代码 / The code

`NVlabs/ProtoMotions` — [`protomotions/envs/motion_manager/motion_manager.py`](https://github.com/NVlabs/ProtoMotions/blob/49fe5ad69de67ebbc07ea2b25d41b0f622c15c3c/protomotions/envs/motion_manager/motion_manager.py#L1-L170)

```python
class MotionManager:
    def __init__(self, config, num_envs, env_dt, device, motion_lib, fixed_motion_ids_per_env=None):
        self.config = config
        self.num_envs = num_envs
        self.device = device
        self.motion_lib = motion_lib
        self.env_dt = env_dt

        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_times = torch.zeros(num_envs, device=device)

        self.init_start_probs = (
            torch.ones(num_envs, dtype=torch.float, device=device)
            * self.config.init_start_prob
        )
        self.motion_weights = self.motion_lib.motion_weights.clone().to(device=device)

        self._setup_motion_subset()
        self._setup_motion_exclusion()
        self._setup_fixed_motion_ids(fixed_motion_ids_per_env)

    def _setup_motion_subset(self):
        subset_method = self.config.subset_method
        total_motions = self.motion_lib.num_motions()
        if subset_method is None or self.num_envs > total_motions:
            self.available_motion_ids = None
            return

        if isinstance(subset_method, str):
            if subset_method == "first":
                n_motions = min(self.num_envs, total_motions)
                self.available_motion_ids = torch.arange(n_motions, device=self.device)
            elif subset_method == "last":
                n_motions = min(self.num_envs, total_motions)
                self.available_motion_ids = torch.arange(total_motions - n_motions, total_motions, device=self.device)
            elif subset_method == "random":
                n_motions = min(self.num_envs, total_motions)
                self.available_motion_ids = torch.randperm(total_motions, device=self.device)[:n_motions]
            else:
                raise ValueError(f"Unknown subset_method string: {subset_method}. Must be 'first', 'last', or 'random'")
        elif isinstance(subset_method, (list, ListConfig)):
            motion_ids = list(subset_method)
            if len(motion_ids) != self.num_envs:
                raise ValueError("When using list for subset_method, length must equal num_envs")
            self.available_motion_ids = torch.tensor(motion_ids, dtype=torch.long, device=self.device)
```

## 逐行讲解 / What's happening

1. **每个环境一份状态 / Per-environment state**:
   - 中文: `motion_ids` 和 `motion_times` 都是长度为 `num_envs` 的 tensor，适合 GPU 并行仿真。
   - English: `motion_ids` and `motion_times` are tensors of length `num_envs`, matching GPU-parallel simulation.
2. **采样权重从 motion lib 克隆 / Clone sampling weights from the motion library**:
   - 中文: manager 可以修改本轮采样策略，但不直接污染 motion library 的原始权重。
   - English: The manager can adjust sampling behavior without mutating the motion library's original weights.
3. **subset 是评估控制阀 / Subset is an evaluation control valve**:
   - 中文: `first`、`last`、`random` 或显式列表都能把实验限制到一组动作。
   - English: `first`, `last`, `random`, or an explicit list can constrain experiments to a chosen motion set.

## 类比 / The analogy

像舞蹈课老师给每个学生分配不同示范视频和播放时间。有人从第一支舞开始，有人只练最后几支；老师还要记住每个人视频播放到第几秒。

It is like a dance instructor assigning each student a reference video and timestamp. Some students start from the first routine, others practice the last few; the instructor tracks where each video currently is.

## 自己跑一遍 / Try it yourself

```python
def subset(method, total, num_envs):
    if method == "first":
        return list(range(min(num_envs, total)))
    if method == "last":
        n = min(num_envs, total)
        return list(range(total - n, total))
    if isinstance(method, list):
        if len(method) != num_envs:
            raise ValueError("length must equal num_envs")
        return method

print(subset("first", 10, 3))
print(subset("last", 10, 3))
print(subset([2, 5, 8], 10, 3))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
[0, 1, 2]
[7, 8, 9]
[2, 5, 8]
```

这个例子展示了 subset 的用途：同一套环境，可以快速切换“全量训练”和“固定动作评估”。

This example shows why subsets are useful: the same environment can switch between broad training and fixed-motion evaluation.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Isaac Gym vectorized envs** / **Isaac Gym vectorized envs**: 每个环境的 episode 状态通常都是一个 batch tensor。 / Per-environment episode state is usually stored as batch tensors.
- **RL curriculum sampling** / **RL curriculum sampling**: 常通过权重、白名单、黑名单控制下一个任务分布。 / Task distributions are often controlled through weights, allowlists, and blocklists.

## 注意事项 / Caveats / when it breaks

- **显式列表长度必须匹配环境数** / **explicit lists must match environment count**: 否则无法一一分配 motion。 / Otherwise there is no one-to-one assignment.
- **subset 不等于修改权重** / **subset is not weight mutation**: subset 限制可采样集合，权重仍是另一层策略。 / A subset limits availability; weights remain a separate sampling policy.

## 延伸阅读 / Further reading

- ProtoMotions source — https://github.com/NVlabs/ProtoMotions/blob/49fe5ad69de67ebbc07ea2b25d41b0f622c15c3c/protomotions/envs/motion_manager/motion_manager.py

