---
date: 2026-09-10
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/utils/checkpoint.py
permalink: https://github.com/pytorch/pytorch/blob/main/torch/utils/checkpoint.py#L1405-L1418
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, pytorch, checkpoint, autograd, torch-compile]
---

# PyTorch SAC storage key：同名 op 也要分执行区 / PyTorch SAC Storage Key: Same Op, Separate Compiled Regions

> **一句话 / In one line**: selective activation checkpointing 用 `(func, compiled_callable_id)` 给编译区域里的缓存分桶，避免重算时拿错同名 op 的输出。 / Selective activation checkpointing keys compiled-region cache entries by `(func, compiled_callable_id)` so recomputation does not reuse the wrong output from the same operator.

## 为什么重要 / Why this matters

中文：SAC 允许你按 op 决定“保存还是重算”。问题是同一个 ATen op 可能在多个编译区域里出现，如果只用 `func` 当 key，反向重算跳过某个区域后，FIFO 缓存会错位。PyTorch 的 `_sac_storage_key` 给 Inductor 编译代码追加 callable id，让每个编译区域有自己的队列。

English: SAC lets a policy decide whether each operator output is saved or recomputed. The trap is that the same ATen operator may appear in multiple compiled regions. If the key is only `func`, a skipped region can shift the FIFO cache. PyTorch's `_sac_storage_key` adds the Inductor callable id so every compiled region gets its own queue.

## 代码 / The code

`pytorch/pytorch` — [`torch/utils/checkpoint.py`](https://github.com/pytorch/pytorch/blob/main/torch/utils/checkpoint.py#L1405-L1418)

```python
def _sac_storage_key(func, args):
    """Compute the SAC storage key for a given op.
    For inductor_compiled_code, each compiled region gets its own FIFO queue
    keyed by the callable's unique idx.  Without this, all compiled regions
    share one queue and a cache-hit that skips a region during recompute
    causes the queue to return the wrong entry (see gh-175258).
    """
    from torch._higher_order_ops.wrap import (
        _resolve_inductor_callable,
        inductor_compiled_code as _inductor_compiled_code,
    )
    if func is _inductor_compiled_code and args:
        return (func, _resolve_inductor_callable(args[0]).idx)
    return func
```

The key is consumed by the save/replay modes in the same file:

```python
key = _sac_storage_key(func, args)
idx = self.func_counter[key]
self.func_counter[key] += 1
```

## 逐行讲解 / What's happening

1. **第 1405-1411 行 / Lines 1405-1411**:
   - 中文: docstring 直接说明 bug 形态：多个 compiled region 共享队列会在 recompute cache hit 后错取条目。
   - English: The docstring names the failure mode: multiple compiled regions sharing one queue can return the wrong entry after a recompute cache hit.
2. **第 1412-1415 行 / Lines 1412-1415**:
   - 中文: 只在函数内部导入 Inductor wrapper，避免普通 checkpoint 路径提前依赖编译模块。
   - English: The Inductor wrapper imports stay inside the helper so ordinary checkpoint paths do not eagerly depend on compile internals.
3. **第 1416-1418 行 / Lines 1416-1418**:
   - 中文: Inductor 编译代码用二元 key；其他 op 仍保持老行为，只按 `func` 分桶。
   - English: Inductor compiled code gets a two-part key; ordinary ops keep the legacy bucket keyed by `func`.
4. **counter use / counter use**:
   - 中文: 每个 key 还有独立 invocation index，同一 op 的第 0/1/2 次调用也能区分。
   - English: Each key also has its own invocation index, so the 0th, 1st, and 2nd calls of the same op stay distinct.

## 类比 / The analogy

中文：像仓库里很多箱子都写着“矩阵乘法”。只看箱名会拿错货；给每条生产线再加一个编号，才能从正确货架取第 N 个箱子。

English: Imagine many warehouse boxes labeled "matmul." The label alone is ambiguous; adding a production-line id lets the loader retrieve the Nth box from the correct shelf.

## 自己跑一遍 / Try it yourself

```python
def key(func, region=None):
    return (func, region) if func == "compiled" and region is not None else func

storage = {}
for func, region, value in [
    ("compiled", 7, "region-7-output"),
    ("compiled", 9, "region-9-output"),
    ("aten.add", None, "plain-output"),
]:
    storage.setdefault(key(func, region), []).append(value)

print(storage[key("compiled", 7)][0])
print(storage[key("compiled", 9)][0])
print(storage[key("aten.add")][0])
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
region-7-output
region-9-output
plain-output
```

中文：SAC 的正确性依赖“保存时”和“重算时”生成完全相同的 key 和调用序号。

English: SAC correctness depends on producing the exact same key and call index during save and recompute.

## 注意事项 / Caveats / when it breaks

- **控制流必须一致** / **Control flow must match**: forward 和 recompute 中 op 序列不同会让 invocation index 错位。
- **compiled region 需要额外身份** / **Compiled regions need extra identity**: 同一个 wrapper 函数不能代表所有编译片段。
- **用户 hooks 仍要考虑** / **User hooks still matter**: SAC 保存 tensor 时还要和 `saved_tensors_hooks` 的语义对齐。

## 延伸阅读 / Further reading

- [PyTorch checkpoint.py](https://github.com/pytorch/pytorch/blob/main/torch/utils/checkpoint.py)
- [Selective activation checkpointing docs](https://pytorch.org/docs/stable/checkpoint.html)
