---
date: 2026-08-17
topic: vla
source: vla
repo: Physical-Intelligence/openpi
file: src/openpi/transforms.py
permalink: https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/transforms.py#L328-L430
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, vla, action-dimension]
build_role: action-head-continuous advanced variant
---

# openpi PadStatesAndActions：不同机器人先补齐到同一动作维度 / openpi PadStatesAndActions: Pad Different Robots into One Action Width

> **一句话 / In one line**: `PadStatesAndActions` 把 state 和 action 的最后一维补到模型统一动作维度，让多机器人数据能共用一个 action head。 / `PadStatesAndActions` pads the last dimension of state and action to the model's shared action width so multiple robots can use one action head.

## 为什么重要 / Why this matters

不同机器人有不同自由度。训练一个通用 VLA 时，模型通常需要固定宽度的 tensor，而不是每个 embodiment 一套 head。这段代码选择了最朴素但很有效的合同：真实维度放前面，不足的尾部补零。

Different robots have different degrees of freedom. A general VLA usually wants fixed-width tensors rather than one head per embodiment. This code uses the simplest useful contract: real dimensions first, zero padding at the tail.

## 代码 / The code

`Physical-Intelligence/openpi` — [`src/openpi/transforms.py`](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/transforms.py#L328-L430)

```python
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x
```

## 逐行讲解 / What's happening

1. **第 328-331 行 / Lines 328-331**: 中文: transform 只持有一个配置：模型侧的统一动作维度。 / English: The transform has one configuration value: the model-side shared action dimension.
2. **第 333-337 行 / Lines 333-337**: 中文: `state` 必补，`actions` 只有训练样本里存在时才补；推理侧可能只有 state。 / English: `state` is always padded, while `actions` is padded only when present; inference may provide state only.
3. **第 423-429 行 / Lines 423-429**: 中文: `pad_width` 对所有轴默认不补，只在指定轴尾部追加零。 / English: `pad_width` leaves every axis unchanged except the selected axis, where zeros are appended at the tail.
4. **第 430 行 / Line 430**: 中文: 如果输入已经足够宽，函数直接返回原数组，不做截断。 / English: If the input is already wide enough, the function returns it as-is and does not truncate.

## 类比 / The analogy

像给不同尺寸的表格统一装进同一宽度的文件夹：真实列放左边，右边空白列补齐。文件夹宽度固定，内容仍保留原来的顺序。

It is like putting spreadsheets of different widths into folders with one fixed width: real columns stay on the left, blank columns fill the right side. The folder is fixed, while the content order stays intact.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-{VLA,WAM}

在 nanoVLA 里，这是 `action-head-continuous` 周围的数据合同层。vision/language/state token 进入模型前，state 需要变成统一宽度；训练 action target 也要同宽，loss 再用 mask 忽略 padding 维度。省掉这层，多 embodiment batch 会在 collate 或 head 输出处直接形状不一致。

In a nanoVLA, this is the data-contract layer around `action-head-continuous`. Before vision/language/state tokens enter the model, state must have one width; training action targets need the same width, and the loss should mask padded dimensions. Without this layer, multi-embodiment batches fail at collation or action-head output shape checks.

## 自己跑一遍 / Try it yourself

```python
def pad_to_dim(row, target, value=0):
    return row + [value] * max(0, target - len(row))

states = [[1, 2], [3, 4, 5]]
actions = [[0.1], [0.2, 0.3, 0.4]]
print([pad_to_dim(x, 4) for x in states])
print([pad_to_dim(x, 4) for x in actions])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[[1, 2, 0, 0], [3, 4, 5, 0]]
[[0.1, 0, 0, 0], [0.2, 0.3, 0.4, 0]]
```

统一宽度并不表示所有维度都有效；真正训练时还需要 action mask 配合 loss 使用。

A shared width does not mean every dimension is valid; real training still needs an action mask in the loss.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LeRobot Evo1 action mask** / **LeRobot Evo1 action mask**: 也把动作维度有效性显式交给 action head。 / It also makes action-dimension validity explicit for the action head.
- **Isaac-GR00T embodiment adapters** / **Isaac-GR00T embodiment adapters**: 更复杂的系统会在补齐外再加 embodiment-specific 投影或 mask。 / More complex systems add embodiment-specific projections or masks on top of padding.

## 注意事项 / Caveats / when it breaks

- **不负责裁剪** / **It does not truncate**: 如果输入比 `model_action_dim` 更宽，代码直接返回原数组，后续 shape 可能不匹配。 / If the input is wider than `model_action_dim`, it is returned unchanged and may break later shape checks.
- **padding 不是语义** / **Padding is not semantics**: 补零维度必须在 loss 和控制输出里被 mask 掉。 / Padded zero dimensions must be masked out in both loss computation and control output.

## 延伸阅读 / Further reading

- [openpi source](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/transforms.py#L328-L430)
