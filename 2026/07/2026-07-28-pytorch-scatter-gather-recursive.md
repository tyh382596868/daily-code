---
date: 2026-07-28
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/nn/parallel/scatter_gather.py
permalink: https://github.com/pytorch/pytorch/blob/e0de08abca327e4e7a2d3f6421d5e461a9dc1c08/torch/nn/parallel/scatter_gather.py#L49-L87
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, data-parallel, recursion]
---

# PyTorch scatter：递归拆开嵌套 batch / PyTorch scatter: Recursively Split a Nested Batch

> **一句话 / In one line**: `scatter` 不只切 tensor，它会递归走进 tuple、list、dict 和 namedtuple，让 DataParallel 保住输入结构。 / `scatter` does not only split tensors; it recursively walks tuples, lists, dicts, and namedtuples so DataParallel preserves input structure.

## 为什么重要 / Why this matters

多 GPU 训练时，输入 batch 往往不是单个 tensor，而是 `({"image": tensor, "meta": ...}, labels)` 这种嵌套结构。PyTorch 这里的技巧是：tensor 真切片，非 tensor 复制引用，容器则递归重建同样的形状。

In multi-GPU training, an input batch is often not a single tensor but a nested object like `({"image": tensor, "meta": ...}, labels)`. PyTorch's trick is simple: actually split tensors, duplicate references for non-tensors, and recursively rebuild containers with the same shape.

## 代码 / The code

`pytorch/pytorch` — [`torch/nn/parallel/scatter_gather.py`](https://github.com/pytorch/pytorch/blob/e0de08abca327e4e7a2d3f6421d5e461a9dc1c08/torch/nn/parallel/scatter_gather.py#L49-L87)

```python
def scatter(inputs, target_gpus, dim=0):
    r"""Slice tensors into approximately equal chunks and distributes them across given GPUs.

    Duplicates references to objects that are not tensors.
    """

    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            return Scatter.apply(target_gpus, None, dim, obj)
        if _is_namedtuple(obj):
            return [
                type(obj)(*args)
                for args in zip(*map(scatter_map, obj), strict=False)
            ]
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj), strict=False))
        if isinstance(obj, list) and len(obj) > 0:
            return [list(i) for i in zip(*map(scatter_map, obj), strict=False)]
        if isinstance(obj, dict) and len(obj) > 0:
            return [
                type(obj)(i)
                for i in zip(*map(scatter_map, obj.items()), strict=False)
            ]
        return [obj for _ in target_gpus]

    try:
        res = scatter_map(inputs)
    finally:
        scatter_map = None  # type: ignore[assignment]
    return res
```

## 逐行讲解 / What's happening

1. **第 55-57 行 / Lines 55-57 (`Tensor`)**:
   - 中文: 只有 tensor 走真正的 `Scatter.apply`，也就是切 batch 维并分发到设备。
   - English: Only tensors go through real `Scatter.apply`, which slices the batch dimension and distributes chunks.
2. **第 58-75 行 / Lines 58-75 (`containers`)**:
   - 中文: namedtuple、tuple、list、dict 都先递归拆子元素，再用 `zip(*)` 按 GPU 重新打包。
   - English: Namedtuples, tuples, lists, and dicts recursively scatter children, then use `zip(*)` to repack per GPU.
3. **第 76 行 / Line 76 (`duplicate refs`)**:
   - 中文: 普通对象不会复制深层内容，只给每个目标设备一份引用。
   - English: Ordinary objects are not deep-copied; each target device receives a reference.
4. **第 83-87 行 / Lines 83-87 (`clear refcycle`)**:
   - 中文: 递归内部函数会形成闭包引用环，`finally` 里把名字清掉帮助释放。
   - English: The recursive inner function creates a closure cycle, so `finally` clears the name to help release it.

## 类比 / The analogy

这像给多张桌子分餐：米饭真的分成几碗，菜单和调料瓶只放同样的引用，托盘结构还保持原来的分层。

It is like serving several tables: rice is actually divided into bowls, menus and condiment bottles are shared by reference, and each tray keeps the same nested layout.

## 自己跑一遍 / Try it yourself

```python
def scatter_like(obj, n):
    if isinstance(obj, list):
        return [list(x) for x in zip(*(scatter_like(v, n) for v in obj))]
    if isinstance(obj, tuple):
        return list(zip(*(scatter_like(v, n) for v in obj)))
    if isinstance(obj, dict):
        parts = [scatter_like(item, n) for item in obj.items()]
        return [dict(x) for x in zip(*parts)]
    if isinstance(obj, range):
        xs = list(obj)
        step = (len(xs) + n - 1) // n
        return [xs[i * step:(i + 1) * step] for i in range(n)]
    return [obj for _ in range(n)]

batch = {"x": range(5), "meta": "same"}
print(scatter_like(batch, 2))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
[{'x': [0, 1, 2], 'meta': 'same'}, {'x': [3, 4], 'meta': 'same'}]
```

例子里 `x` 被切开，但 `meta` 保持同一个普通对象语义。

In the example, `x` is split, while `meta` keeps ordinary shared-object semantics.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **PyTorch `default_collate`** / **PyTorch `default_collate`**: 中文: DataLoader 也递归保持 batch 的容器结构。 / English: DataLoader also recursively preserves batch container structure.
- **DTensor sharding** / **DTensor sharding**: 中文: 分布式张量 API 也把“切 tensor”和“保结构”分开。 / English: Distributed tensor APIs also separate tensor slicing from structure preservation.

## 注意事项 / Caveats / when it breaks

- **非 tensor 共享引用** / **Shared non-tensors**: 中文: 可变普通对象会被多个副本共享，写入会互相影响。 / English: Mutable non-tensors are shared across replicas, so writes can leak between them.
- **容器长度** / **Container length**: 中文: `zip` 会按最短子结果对齐，奇怪的自定义容器要额外测试。 / English: `zip` aligns by the shortest child result, so unusual custom containers need tests.

## 延伸阅读 / Further reading

- [pytorch/pytorch source](https://github.com/pytorch/pytorch/blob/e0de08abca327e4e7a2d3f6421d5e461a9dc1c08/torch/nn/parallel/scatter_gather.py#L49-L87)
