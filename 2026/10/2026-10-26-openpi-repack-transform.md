---
date: 2026-10-26
topic: robotics
source: tracked
repo: Physical-Intelligence/openpi
file: src/openpi/transforms.py
permalink: https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/transforms.py#L80-L100
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, data-contract]
---

# openpi 重打包：用路径表把机器人数据变成模型契约 / openpi Repacking: Turn Robot Data into a Model Contract with a Path Table

> **一句话 / In one line**: `RepackTransform` 用一个声明式树把各机器人数据格式投影到统一输入。 / `RepackTransform` projects many robot schemas into one input contract with a declarative tree.

## 为什么重要 / Why this matters

机器人数据集字段千差万别；把字段适配散落在训练循环里，会让新数据集接入变成高风险修改。这里把“输出字段 → 原始路径”集中为一张表，模型只看统一结构。

Robot datasets disagree on field names and nesting. Keeping the adaptation in one path table lets the model consume one stable schema.

## 代码 / The code

`Physical-Intelligence/openpi` — [`src/openpi/transforms.py`](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/transforms.py#L80-L100)

```python
@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)
```

## 逐行讲解 / What's happening

`structure` 保留目标嵌套形状，叶子却是源字段路径；先扁平化输入，再用同一棵树逐叶查表。因此结构和取值规则可以分别审阅。

`structure` preserves the desired nesting while its leaves name source paths. Flatten once, then map each leaf through the lookup table.

## 类比 / The analogy

这像机场行李转运牌：行李不必改名，转运牌只声明它该去哪个登机口。

It is an airport baggage board: bags stay unchanged; the board says where each one goes.

## 自己跑一遍 / Try it yourself

```python
def get(d, path):
    for key in path.split("."): d = d[key]
    return d
raw = {"obs": {"joint": [1, 2]}, "cmd": [3]}
contract = {"state": "obs.joint", "actions": "cmd"}
print({k: get(raw, p) for k, p in contract.items()})
```

运行 / Run with `python try.py`。预期输出 / Expected output: `{'state': [1, 2], 'actions': [3]}`。

## 注意事项 / Caveats / when it breaks

- **缺失路径 / Missing paths**：默认查找会立刻失败；生产实现应在数据导入期验证契约。 / Validate the contract during ingestion, not halfway through training.

## 延伸阅读 / Further reading

- [openpi transforms](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/transforms.py)
