---
date: 2026-08-12
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/utils/data/_utils/pin_memory.py
permalink: https://github.com/pytorch/pytorch/blob/main/torch/utils/data/_utils/pin_memory.py#L51-L100
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, pytorch, dataloader]
---

# PyTorch pin_memory：递归搬 batch，但保住容器形状 / PyTorch pin_memory: Recursively Move a Batch While Preserving Containers

> **一句话 / In one line**: `pin_memory()` 像 `collate()` 的后处理，递归走过 mapping、tuple、sequence，只把真正能 pin 的叶子交给对象自己处理。 / `pin_memory()` is a post-processing walk over `collate()` output: recurse through mappings, tuples, and sequences, and let pinnable leaves handle themselves.

## 为什么重要 / Why this matters

DataLoader 的 batch 很少只是一个 tensor；它通常是字典套列表、列表套 namedtuple。`pin_memory=True` 如果只处理顶层 tensor 就没用。PyTorch 的实现把容器结构复制一份，同时递归转换叶子节点，这样 CUDA 拷贝能加速，用户代码看到的 batch 形状仍然不变。

A DataLoader batch is rarely a single tensor. It is often a dictionary of lists or namedtuples. If `pin_memory=True` only handled top-level tensors, it would miss most real batches. PyTorch clones the container shape and recursively transforms leaves, so CUDA transfer can speed up without changing the batch contract.

## 代码 / The code

`pytorch/pytorch` — [`torch/utils/data/_utils/pin_memory.py`](https://github.com/pytorch/pytorch/blob/main/torch/utils/data/_utils/pin_memory.py#L51-L100)

```python
def pin_memory(data, device=None):
    if isinstance(data, torch.Tensor):
        return data.pin_memory()

    if hasattr(data, "pin_memory"):
        return data.pin_memory()

    if isinstance(data, (str, bytes)):
        return data
    if isinstance(data, collections.abc.Mapping):
        try:
            if isinstance(data, collections.abc.MutableMapping):
                clone = copy.copy(data)
                clone.update(
                    {k: pin_memory(sample, device) for k, sample in data.items()}
                )
                return clone
            else:
                return type(data)(
                    {k: pin_memory(sample, device) for k, sample in data.items()}
                )
        except TypeError:
            return {k: pin_memory(sample, device) for k, sample in data.items()}
    if isinstance(data, tuple):
        if hasattr(data, "_fields"):  # namedtuple
            return type(data)(*(pin_memory(sample, device) for sample in data))
        return type(data)(pin_memory(sample, device) for sample in data)
    if isinstance(data, collections.abc.Sequence):
        try:
            if isinstance(data, collections.abc.MutableSequence):
                clone = copy.copy(data)
                for i, item in enumerate(data):
                    clone[i] = pin_memory(item, device)
                return clone
            return type(data)([pin_memory(sample, device) for sample in data])
        except TypeError:
            return [pin_memory(sample, device) for sample in data]
    return data
```

## 逐行讲解 / What's happening

1. **第 52-56 行 / Lines 52-56 (leaf first)**:
   - 中文: tensor 和自定义对象优先自己 `pin_memory()`，这是扩展点。
   - English: Tensors and custom objects get first chance to handle `pin_memory()`, which is the extension point.
2. **第 58-80 行 / Lines 58-80 (mapping path)**:
   - 中文: 字典类先尝试保留原类型；可变 mapping 用浅拷贝再更新，不可变 mapping 用构造器重建。
   - English: Mapping-like objects try to keep their original type; mutable mappings are shallow-copied and updated, immutable ones are rebuilt through their constructor.
3. **第 81-84 行 / Lines 81-84 (tuple path)**:
   - 中文: namedtuple 要用 `type(data)(*items)` 恢复字段语义，普通 tuple 用生成器即可。
   - English: A namedtuple needs `type(data)(*items)` to preserve fields; a plain tuple can be rebuilt from a generator.
4. **第 85-100 行 / Lines 85-100 (sequence fallback)**:
   - 中文: sequence 如果构造器不接受列表，就退回普通 list，宁可丢一点类型也不能丢数据。
   - English: If a sequence type cannot be reconstructed from a list, PyTorch falls back to a plain list. Losing a bit of type information is better than losing data.

## 类比 / The analogy

像搬家时给所有易碎品贴标签：箱子、抽屉、收纳盒的层级不变，真正贴标签的是里面的杯子和盘子。

It is like labeling fragile items while moving house: boxes, drawers, and containers keep their hierarchy, while the actual cups and plates receive the labels.

## 自己跑一遍 / Try it yourself

```python
from collections import namedtuple

class Leaf:
    def __init__(self, x): self.x = x
    def pin_memory(self): return Leaf(f"pinned:{self.x}")
    def __repr__(self): return self.x

Pair = namedtuple("Pair", "left right")

def pin(data):
    if hasattr(data, "pin_memory"): return data.pin_memory()
    if isinstance(data, dict): return {k: pin(v) for k, v in data.items()}
    if isinstance(data, tuple) and hasattr(data, "_fields"):
        return type(data)(*(pin(x) for x in data))
    if isinstance(data, (list, tuple)): return type(data)(pin(x) for x in data)
    return data

batch = {"obs": [Leaf("image"), Leaf("state")], "meta": Pair(Leaf("a"), "id")}
print(pin(batch))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
{'obs': [pinned:image, pinned:state], 'meta': Pair(left=pinned:a, right='id')}
```

这个例子展示了关键点：算法不是“处理 tensor”，而是“保持容器不乱，递归处理叶子”。

The key point is that the algorithm is not just "process tensors"; it preserves the container layout and recursively handles leaves.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **PyTorch collate** / **PyTorch collate**: 递归合 batch 时也先看类型，再决定怎么重建容器。 / Recursive batching also inspects type first, then decides how to rebuild containers.
- **pytree utilities** / **pytree utilities**: JAX/PyTorch pytree 同样把“结构”和“叶子操作”分开。 / JAX and PyTorch pytrees also separate structure from leaf operations.

## 注意事项 / Caveats / when it breaks

- **自定义容器构造器可能失败** / **Custom constructors may fail**: 所以源码有 `TypeError` fallback，不能假设 `type(data)(iterable)` 总是合法。
- **字符串必须停住** / **Strings must stop recursion**: `str` 是 sequence，但不能被拆成字符列表处理。

## 延伸阅读 / Further reading

- [PyTorch `pin_memory`](https://github.com/pytorch/pytorch/blob/main/torch/utils/data/_utils/pin_memory.py#L51-L100)
