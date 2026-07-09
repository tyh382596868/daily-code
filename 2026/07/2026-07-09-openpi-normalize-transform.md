---
date: 2026-07-09
topic: robotics
source: tracked
repo: Physical-Intelligence/openpi
file: src/openpi/transforms.py
permalink: https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/transforms.py#L108-L159
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, robotics, normalization, transforms]
---

# openpi Normalize：同一个 transform 同时支持 z-score 和分位数归一化 / openpi Normalize: One Transform for Z-Score and Quantile Scaling

> **一句话 / In one line**: openpi 把动作、状态等树状数据的归一化写成一个可组合 transform，并用同一套 norm stats 支持 z-score 和 quantile 两种路径。 / openpi packages normalization for nested state/action data as a composable transform, with one stats tree driving either z-score or quantile scaling.

## 为什么重要 / Why this matters

机器人数据不是一张扁平表：不同相机、关节、夹爪和动作维度经常嵌在一个 nested dict 里。openpi 这里的设计重点不是公式本身，而是把公式放进 transform pipeline：上游 repack 成标准 key，下游模型只看到已经归一化的 `state` / `actions`。

Robot data is not a flat table. Cameras, joints, grippers, and action dimensions often live in a nested dictionary. The useful idea here is not the formula alone, but where it lives: inside the transform pipeline, after repacking and before the model consumes normalized `state` / `actions`.

## 代码 / The code

`Physical-Intelligence/openpi` — [`src/openpi/transforms.py`](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/transforms.py#L108-L159)

```python
@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

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
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
```

## 逐行讲解 / What's happening

1. **第 109-113 行 / Lines 109-113 (`norm_stats`, `use_quantiles`, `strict`)**:
   - 中文: 这个 transform 的行为完全由统计量树和两个开关决定；它不关心数据来自哪个机器人。
   - English: The transform is driven by a stats tree and two switches; it does not need robot-specific branches.
2. **第 115-117 行 / Lines 115-117 (`__post_init__`)**:
   - 中文: 如果走 quantile 路径，构造时就验证 `q01/q99` 是否存在，避免训练到一半才炸。
   - English: The quantile path validates `q01/q99` at construction time, so failures show up before training starts.
3. **第 122-128 行 / Lines 122-128 (`apply_tree`)**:
   - 中文: 真正的工程点在这里：递归遍历 data 和 stats，把同一个归一化函数应用到匹配 leaf。
   - English: The engineering trick is here: recurse through data and stats, applying the same leaf function wherever keys match.
4. **第 131-138 行 / Lines 131-138 (two formulas)**:
   - 中文: z-score 保留均值方差解释；quantile 把 `q01..q99` 映射到 `[-1, 1]`，对离群值更稳。
   - English: Z-score keeps a mean/std interpretation; quantile scaling maps `q01..q99` to `[-1, 1]`, which is often more robust to outliers.

## 类比 / The analogy

这像给不同规格的螺丝都贴上同一套货架标签：你不改变螺丝本身，只是把它们放进统一坐标系，仓库系统就能按同一套规则取货。

It is like labeling different screw sizes with one warehouse coordinate system. The screws do not change, but once they share a coordinate convention, the rest of the warehouse can handle them uniformly.

## 自己跑一遍 / Try it yourself

```python
def zscore(x, mean, std):
    return [(v - m) / (s + 1e-6) for v, m, s in zip(x, mean, std)]

def quantile(x, q01, q99):
    return [(v - lo) / (hi - lo + 1e-6) * 2 - 1 for v, lo, hi in zip(x, q01, q99)]

print([round(v, 2) for v in zscore([3, 10], [1, 8], [2, 4])])
print([round(v, 2) for v in quantile([3, 10], [1, 0], [5, 20])])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[1.0, 0.5]
[-0.0, -0.0]
```

第一行是标准差单位；第二行表示两个值都落在各自分位区间的中点附近。

The first line is measured in standard deviations; the second says both values sit near the middle of their quantile spans.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LeRobot normalizers** / **LeRobot normalizers**: 也把每个 observation/action field 的统计量和数据 leaf 对齐。 / They also align per-field observation/action stats with matching data leaves.
- **Diffusion Policy LinearNormalizer** / **Diffusion Policy LinearNormalizer**: 用参数 store 管理多路观测的 scale/offset。 / It stores scale/offset parameters for many observation streams.

## 注意事项 / Caveats / when it breaks

- **维度裁剪要小心** / **Dimension slicing needs care**: `stats.mean[..., : x.shape[-1]]` 允许 action 维度变短，但也可能掩盖错误的 stats 文件。 / `stats.mean[..., : x.shape[-1]]` allows shorter action vectors, but can hide a wrong stats file.
- **quantile 不是物理单位** / **Quantiles are not physical units**: 它更稳，但解释性弱于 z-score。 / It is robust, but less directly interpretable than z-score.

## 延伸阅读 / Further reading

- [openpi `Normalize`](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/transforms.py#L108-L159)
