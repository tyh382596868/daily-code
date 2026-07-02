---
date: 2026-07-02
topic: robotics
source: tracked
repo: Physical-Intelligence/openpi
file: src/openpi/transforms.py
permalink: https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/transforms.py#L203-L225
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, robotics, action-representation]
---

# openpi DeltaActions：把绝对动作改写成相对控制 / openpi DeltaActions: Rewrite Absolute Actions into Relative Control

> **一句话 / In one line**: 用一个布尔 mask，只把需要相对化的动作维度减去当前 state。 / A boolean mask selects only the action dimensions that should subtract the current state.

## 为什么重要 / Why this matters

机器人动作既可以表示成“去到这个绝对位置”，也可以表示成“从当前状态移动这么多”。后者常让跨初始姿态的数据更容易学习。`DeltaActions` 把这个选择压成一个数据变换：有 mask 才改、只改前 `dims` 个维度、一次广播覆盖整个 action horizon。

Robot actions can mean either "go to this absolute pose" or "move this much from the current state." The second form is often easier to learn across different starting poses. `DeltaActions` makes the choice a data transform: do nothing without a mask, touch only the first `dims`, and broadcast over the whole action horizon.

## 代码 / The code

`Physical-Intelligence/openpi` — [`src/openpi/transforms.py`](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/transforms.py#L203-L225)

```python
@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data
```

## 逐行讲解 / What's happening

1. **第 203-211 行 / Lines 203-211**: 中文: frozen dataclass 让这个变换像配置项一样可组合，`mask=None` 明确表示 no-op。 / English: The frozen dataclass makes the transform configuration-like and composable; `mask=None` explicitly means no-op.
2. **第 214-215 行 / Lines 214-215**: 中文: 没有 `actions` 或 mask 时直接返回，推理和训练可以共用同一条 transform chain。 / English: Missing `actions` or a mask returns immediately, so inference and training can share one transform chain.
3. **第 217-221 行 / Lines 217-221**: 中文: `np.where(mask, state, 0)` 保留要相减的 state 维度，`expand_dims(..., axis=-2)` 把当前 state 广播到整个 action 序列。 / English: `np.where(mask, state, 0)` keeps only selected state dimensions, and `expand_dims(..., axis=-2)` broadcasts the current state over the action sequence.

## 类比 / The analogy

像导航软件切换“目的地坐标”和“从当前位置往前 200 米”。同一条路线可以用绝对地址表达，也可以用相对位移表达；mask 决定哪些轴用相对说法。

It is like switching a map app from a destination address to "go 200 meters from here." The route is the same, but the coordinate system changes; the mask chooses which axes speak relative language.

## 自己跑一遍 / Try it yourself

```python
import numpy as np
state = np.array([10.0, 20.0, 0.5])
actions = np.array([[11.0, 19.0, 0.8], [12.0, 18.0, 0.2]])
mask = np.array([True, True, False])
dims = mask.shape[-1]
actions[..., :dims] -= np.expand_dims(np.where(mask, state[:dims], 0), axis=-2)
print(actions)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[[ 1.  -1.   0.8]
 [ 2.  -2.   0.2]]
```

中文: 前两个维度变成相对位移，第三个维度保持绝对命令。

English: The first two dimensions become relative deltas; the third stays an absolute command.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Diffusion Policy 动作归一化 / Diffusion Policy action normalization**: 中文: 先统一动作坐标系，再让策略学习残差。 / English: Normalize action coordinates first, then let the policy learn residuals.
- **GR00T action chunk transforms / GR00T action chunk transforms**: 中文: 相对/绝对动作通常是数据层决策，不应该散落在模型 forward 里。 / English: Relative versus absolute action is usually a data-layer choice, not logic scattered through model `forward`.

## 注意事项 / Caveats / when it breaks

- **原地修改 / In-place mutation**: 中文: `actions -= ...` 会改传入数组；复用同一 batch 时要先 copy。 / English: `actions -= ...` mutates the input array; copy first if the same batch is reused.
- **mask 维度 / Mask dimensionality**: 中文: mask 可以短于 action 维度，但不能比 state/action 前缀还长。 / English: The mask can be shorter than the action vector, but it must match an existing prefix.

## 延伸阅读 / Further reading

- Source permalink above.
- `src/openpi/transforms.py` also contains the inverse `AbsoluteActions`.
