---
date: 2026-07-17
topic: vla
source: vla
repo: Physical-Intelligence/openpi
file: src/openpi/transforms.py
permalink: https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/transforms.py
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, openpi, data-pipeline, transforms]
build_role: training-step advanced variant
---

# openpi RepackTransform：用路径表重排机器人 batch / openpi RepackTransform: Repack Robot Batches with a Path Table

> **一句话 / In one line**: `RepackTransform` 把嵌套 observation/action 字典先 flatten，再按显式路径表组装成 policy 需要的 schema。 / `RepackTransform` flattens a nested observation/action dictionary, then rebuilds the policy schema from explicit paths.

## 为什么重要 / Why this matters

VLA 最容易失控的地方不是模型层，而是数据 schema。不同数据集会把图像、state、action 放在不同路径下；模型却需要固定键名。openpi 用 `RepackTransform` 把“数据集原始结构”翻译成“policy 输入结构”，避免在模型里散落大量 dataset-specific if。

The easiest place to lose control in a VLA is not the model layer; it is the data schema. Datasets put images, state, and actions under different paths, while the policy expects fixed keys. openpi uses `RepackTransform` to translate raw dataset structure into policy input structure.

## 代码 / The code

`Physical-Intelligence/openpi` — [`src/openpi/transforms.py`](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/transforms.py)

```python
@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary."""

    # Example:
    # {
    #   "images": {
    #     "cam_high": "observation.images.top",
    #     "cam_low": "observation.images.bottom",
    #   },
    #   "state": "observation.state",
    #   "actions": "action",
    # }
    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)
```

## 逐行讲解 / What's happening

1. **配置是 PyTree / The config is a PyTree**: 中文: 输出结构可以嵌套，不只是扁平 dict。 English: the output structure can be nested, not just flat.
2. **路径是字符串 / Paths are strings**: 中文: 每个叶子告诉 transform 去原始 batch 的哪个路径取值。 English: each leaf says which original path to read.
3. **先 flatten 原始数据 / Flatten raw data first**: 中文: 嵌套字典变成 `path -> value` 查找表。 English: nested data becomes a `path -> value` lookup table.
4. **`jax.tree.map` 保留目标形状 / `jax.tree.map` preserves target shape**: 中文: 输出长得和 `structure` 一样，只是叶子换成真实数组。 English: output has the same shape as `structure`, with leaves replaced by arrays.
5. **缺 key 直接失败 / Missing keys fail fast**: 中文: schema 错误不会悄悄变成 None。 English: schema errors do not silently become `None`.

## 类比 / The analogy

这像一张装箱清单：左边写新箱子的格子名，右边写旧仓库货架号。搬运工不理解机器人，只按清单把东西搬到正确格子。

It is like a packing list: the left side names the new box slot, the right side names the old shelf path. The mover does not understand robotics; it just follows the manifest.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

中文: 这是 `training-step` 前的数据契约层。nanoVLA 应该让 dataset adapter 产出原始嵌套 batch，再用一个 repack transform 统一到 `image/state/actions/prompt`。这样模型 forward 永远只处理统一 schema。

English: This is the data-contract layer before `training-step`. A nanoVLA should let dataset adapters emit raw nested batches, then use one repack transform to normalize them into `image/state/actions/prompt`.

## 自己跑一遍 / Try it yourself

```python
def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, path))
        else:
            out[path] = v
    return out

data = {"observation": {"images": {"top": "rgb0"}, "state": [1, 2]}, "action": [0.1]}
structure = {"image": {"cam_high": "observation.images.top"}, "state": "observation.state", "actions": "action"}
flat = flatten(data)
packed = {"image": {"cam_high": flat[structure["image"]["cam_high"]]}, "state": flat[structure["state"]], "actions": flat[structure["actions"]]}
print(packed)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
{'image': {'cam_high': 'rgb0'}, 'state': [1, 2], 'actions': [0.1]}
```

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **DROID timestep processor**: 先把原始数据规整成训练契约。 / It normalizes raw records into a training contract.
- **LeRobot dataset features**: dataset schema 与 policy schema 分离。 / Dataset schema and policy schema are kept separate.

## 注意事项 / Caveats / when it breaks

- **路径字符串要稳定 / Path strings must be stable**: 数据集字段重命名会立即失败。 / Dataset field renames fail immediately.
- **不要在模型里 repack / Do not repack inside the model**: schema 逻辑应留在数据层。 / Schema logic belongs in the data layer.
- **输出结构就是模型合同 / Output structure is the model contract**: 改它等于改 forward API。 / Changing it changes the forward API.

## 延伸阅读 / Further reading

- [openpi transforms](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/transforms.py)
- [openpi repository](https://github.com/Physical-Intelligence/openpi)
