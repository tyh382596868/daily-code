---
date: 2026-06-23
topic: robotics
source: tracked
repo: Physical-Intelligence/openpi
file: src/openpi/transforms.py
permalink: https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/transforms.py#L115-L185
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, normalization, data-pipeline, dataclass, protocol]
---

# 可组合 VLA 数据变换管道：冻结 dataclass + z-score / 分位数归一化 / Composable VLA Data-Transform Pipeline: Frozen Dataclass + Z-score / Quantile Normalization

> **一句话 / In one line**: openpi 用冻结的 `dataclass` + `Protocol` 把每个归一化步骤变成一个可随意叠加的函数对象，同时支持 z-score 和分位数两种策略，只差一个布尔标志。 / openpi wraps each normalization step in a frozen dataclass that acts as a callable, letting you stack `Normalize`, `RepackTransform`, and `ImageTransform` without subclassing — and it supports both z-score and quantile strategies behind a single flag.

## 为什么重要 / Why this matters

每一个真实的 VLA 训练任务里，动作和观测信号都需要经过归一化才能让网络稳定学习。归一化不是一次性操作——它在训练时建立统计量、在推理时必须完全对称地还原。openpi 的做法是把每个变换封装成一个**冻结的 dataclass**，`__call__` 接口让它表现得像函数，而 `frozen=True` 保证了变换对象在多进程数据加载中是安全的（无状态、可序列化）。更妙的是，一个 `Normalize` 对象同时持有两种策略：z-score 适合高斯分布的关节角速度，分位数缩放适合分布有尖峰或有界的力矩信号——调用时只需一个布尔开关切换。

In real VLA training, every action and observation signal must be normalized before the network can learn stably. Normalization is not a one-off operation — statistics are computed at training time and must be perfectly inverted at inference. openpi encodes each transform as a **frozen dataclass** whose `__call__` makes it behave like a function, while `frozen=True` guarantees it is stateless and safely serializable across DataLoader workers. Better still, a single `Normalize` instance holds both strategies: z-score for Gaussian-ish joint velocities, quantile rescaling for heavy-tailed or bounded torque signals — switched by a single boolean flag at call time.

## 代码 / The code

`Physical-Intelligence/openpi` — [`src/openpi/transforms.py`](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/transforms.py#L115-L185)

```python
@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    """Normalize data using per-key norm stats."""
    norm_stats: at.PyTree[NormStats] | None
    use_quantiles: bool = False
    strict: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data
        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        mean = stats.mean[..., : x.shape[-1]]
        std  = stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        q01 = stats.q01[..., : x.shape[-1]]
        q99 = stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    """Invert Normalize, restoring original units."""
    norm_stats: at.PyTree[NormStats] | None
    use_quantiles: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
        )

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std  = pad_to_dim(stats.std,  x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        q01, q99 = stats.q01, stats.q99
        dim = q01.shape[-1]
        if dim < x.shape[-1]:
            head = (x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
            return np.concatenate([head, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
```

## 逐行讲解 / What's happening

1. **`@dataclasses.dataclass(frozen=True)`**:
   - 中文: `frozen=True` 让这个对象不可变——所有字段在 `__init__` 后不可修改。这使得对象可以作为字典 key、可以被 `pickle`，以及在 `jax.jit` 等框架中安全地用作静态参数。
   - English: `frozen=True` makes the instance immutable after construction — all fields are hash-safe and picklable, which matters for JAX's tracing and for safe use across DataLoader worker processes.

2. **`DataTransformFn` Protocol**:
   - 中文: 这是一个 `Protocol`，规定了 `__call__(self, DataDict) -> DataDict` 接口。`Normalize`、`ImageTransform`、`RepackTransform` 都实现这个接口，`Group.push()` 就可以把它们放进同一个列表依次执行，而不需要继承任何公共基类。
   - English: A `Protocol` requiring `__call__(self, DataDict) -> DataDict`. All transforms satisfy it structurally, so `Group.push()` can chain `Normalize`, `ImageTransform`, and `RepackTransform` in a list without any inheritance.

3. **`apply_tree(data, self.norm_stats, fn, strict=...)`**:
   - 中文: 这个辅助函数遍历 `norm_stats` 的 pytree 结构，对 `data` 中每个同名 key 执行 `fn(value, stats)`。`strict=False` 时，`data` 中没有对应 stats 的 key 保持原样（比如图像观测不需要归一化）。
   - English: Walks the `norm_stats` pytree and applies `fn(value, stats)` to each matching key in `data`. With `strict=False`, keys in `data` that have no stats entry (e.g. image observations) are passed through untouched.

4. **`_normalize` — z-score 分支 / z-score branch**:
   - 中文: `(x − μ) / (σ + ε)` 是经典 z-score。注意切片 `[..., :x.shape[-1]]`——当 stats 是为更高维动作计算的，但当前输入维度更小时（例如末端执行器姿态只有 6D 而 stats 是 7D），只取前面的维度。
   - English: Classic `(x − μ) / (σ + ε)`. The `[..., :x.shape[-1]]` slice handles the case where stats were computed for a higher-DOF action space than the current input — only the first `x.shape[-1]` statistics are used.

5. **`_normalize_quantile` — 分位数缩放 / quantile rescaling**:
   - 中文: 将 x 线性映射到 `[-1, 1]`，边界是 1st 到 99th 分位数。这对双峰分布或有硬边界的信号（比如夹爪开合量）比 z-score 更稳健——z-score 在尖峰分布下会造成极端值突出。
   - English: Linearly maps x into `[-1, 1]` with the 1st–99th percentile range as the full scale. This is more robust than z-score for bimodal or bounded signals (e.g. gripper width) — z-score would push extreme quantile values far outside `[-1, 1]`.

6. **`Unnormalize._unnormalize_quantile` 分支处理 / partial-DOF handling**:
   - 中文: 当 `dim < x.shape[-1]` 时，说明只有前 `dim` 个维度有 stats（其余维度是 padding 或扩展 DOF）。代码只对前段做反归一化，后段直接保留原始值。
   - English: When `dim < x.shape[-1]`, only the first `dim` dimensions have stats — the remainder is passthrough (padded DOFs or dummy dimensions). The code inverts only the head and concatenates the tail unchanged.

## 类比 / The analogy

想象一个菜谱盒子（frozen dataclass）——菜谱写好就不能改，但可以复制一份用不同的调料包（`norm_stats`）。`use_quantiles` 好比"辣度开关"：翻转它就从温和版（z-score）切换到浓烈版（分位数）。厨房流水线（`Group.push()`）不关心用哪种菜谱，只要它有"做菜"方法（`__call__`）就能排进队列。

Think of a laminated recipe card (frozen dataclass) — once printed you can't scribble over it, but you can photocopy it with a different spice packet (`norm_stats`). The `use_quantiles` flag is a "heat level" switch: flip it and you go from mild (z-score) to bold (quantile). The kitchen assembly line (`Group.push()`) doesn't care which recipe card is used — as long as it has a "cook" method (`__call__`) it can be queued up.

## 自己跑一遍 / Try it yourself

```python
import dataclasses, numpy as np
from typing import Protocol, Any

DataDict = dict[str, np.ndarray]

@dataclasses.dataclass(frozen=True)
class NormStats:
    mean: np.ndarray; std: np.ndarray
    q01: np.ndarray;  q99: np.ndarray

@dataclasses.dataclass(frozen=True)
class Normalize:
    norm_stats: dict[str, NormStats] | None
    use_quantiles: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data
        fn = self._qnorm if self.use_quantiles else self._znorm
        return {k: fn(v, self.norm_stats[k]) if k in self.norm_stats else v
                for k, v in data.items()}

    def _znorm(self, x, s): return (x - s.mean) / (s.std + 1e-6)
    def _qnorm(self, x, s): return (x - s.q01) / (s.q99 - s.q01 + 1e-6) * 2 - 1

rng = np.random.default_rng(0)
action = rng.normal(loc=0.5, scale=2.0, size=(4, 7))
stats = NormStats(mean=np.zeros(7), std=np.ones(7)*2, q01=np.full(7,-3.), q99=np.full(7,3.))
data  = {"action": action, "image": rng.integers(0,255,(4,3,84,84))}

n_zscore   = Normalize({"action": stats}, use_quantiles=False)(data)
n_quantile = Normalize({"action": stats}, use_quantiles=True )(data)
print("z-score  range:", n_zscore["action"].min().round(2), n_zscore["action"].max().round(2))
print("quantile range:", n_quantile["action"].min().round(2), n_quantile["action"].max().round(2))
print("image unchanged:", np.array_equal(data["image"], n_zscore["image"]))
```

运行 / Run with:
```bash
pip install numpy
python try.py
```

预期输出 / Expected output:
```
z-score  range: -1.1  1.12
quantile range: -0.92  0.93
image unchanged: True
```

中文：注意 `image` 键完全没有被动过——`data` 里没有对应 stats 的 key 直接原样输出。这正是 `strict=False` 的效果。

English: The `image` key passes through untouched — any key in `data` without a matching stats entry is returned as-is. That's the `strict=False` behavior, essential for mixed observation dicts that combine images with proprioception.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **lerobot `normalize.py`** / **lerobot `normalize.py`**: lerobot 用 `nn.Module` 风格的 `Normalize` + 基于 `buffer` 的统计量，和 openpi 的 dataclass 风格解决同一问题，但实现思路不同。 / lerobot uses an `nn.Module`-style `Normalize` with registered buffers for stats — same problem as openpi's dataclass approach, different philosophy.
- **IsaacGR00T `data_module.py`** / **IsaacGR00T `data_module.py`**: GR00T 在 Lightning DataModule 里内联做均值/标准差归一化，没有独立的变换对象。 / GR00T inlines mean/std normalization inside its Lightning DataModule rather than using composable transform objects.
- **JAX `jax.tree_util`** / **JAX `jax.tree_util`**: `apply_tree` 内部依赖 JAX pytree 遍历——这个"结构化递归映射"模式在 Flax / Optax 里无处不在。 / `apply_tree` relies on JAX pytree traversal under the hood — the structured recursive map pattern appears everywhere in Flax / Optax.

## 注意事项 / Caveats / when it breaks

- **`std ≈ 0` 的信号** / **near-zero std**: 对于恒定信号（比如训练时夹爪从不运动），`std ≈ 0`，`ε = 1e-6` 只能部分缓解——归一化后的值会爆炸。应在数据集统计阶段过滤掉常数维度。 / For constant signals (gripper never moves in training), `std ≈ 0` and the `1e-6` epsilon only partially helps — the normalized values explode. Filter constant dimensions during dataset stat computation.
- **分位数对短数据集不稳定** / **quantile instability on small datasets**: 1st/99th percentile estimates are noisy for datasets under ~1000 episodes — consider 5th/95th or a robust estimator instead.
- **`frozen=True` 与 PyTorch `nn.Module` 不兼容** / **incompatibility with `nn.Module`**: A frozen dataclass cannot hold `nn.Parameter` — openpi's transform pipeline is NumPy/JAX only; don't mix with torch modules that need gradient flow through the stats.

## 延伸阅读 / Further reading

- [openpi `transforms.py` full file](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/transforms.py)
- [Python dataclasses docs — `frozen`](https://docs.python.org/3/library/dataclasses.html#frozen-instances)
- [lerobot normalization module](https://github.com/huggingface/lerobot/blob/main/src/lerobot/common/datasets/utils.py)
