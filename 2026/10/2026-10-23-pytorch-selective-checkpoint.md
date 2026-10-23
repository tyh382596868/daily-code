---
date: 2026-10-23
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/utils/checkpoint.py
permalink: https://github.com/pytorch/pytorch/blob/fed346df557d144bc8a48f2a018ca2403fcdb7e7/torch/utils/checkpoint.py#L1755-L1845
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, pytorch, activation-checkpointing]
---

# PyTorch 选择性 checkpoint：只把昂贵算子留在记忆里 / PyTorch Selective Checkpointing: Save Only the Expensive Ops

> **一句话 / In one line**: 选择性 checkpoint 把「存还是重算」从整个函数细化到单个 ATen 算子。 / Selective checkpointing moves the save-versus-recompute decision from a whole function down to individual ATen operators.

## 为什么重要 / Why this matters

普通 activation checkpoint 用算力换显存；这里允许把昂贵的矩阵乘保留、把便宜操作重算，形成更细的预算控制。

Ordinary activation checkpointing trades compute for memory. This API can retain expensive matmuls while recomputing cheap operations, giving a finer budget.

## 代码 / The code

`pytorch/pytorch` — [`checkpoint.py`](https://github.com/pytorch/pytorch/blob/fed346df557d144bc8a48f2a018ca2403fcdb7e7/torch/utils/checkpoint.py#L1785-L1845)

```python
def create_selective_checkpoint_contexts(policy_fn_or_list,
                                         allow_cache_entry_mutation=False):
    if isinstance(policy_fn_or_list, list):
        for op in policy_fn_or_list:
            if not isinstance(op, (torch._ops.OpOverload,
                                   torch._ops.HigherOrderOperator)):
                raise ValueError("Expected op in op_list to be an OpOverload")
```

## 逐行讲解 / What's happening

列表是简写：列表中的算子等价于 `MUST_SAVE`，其他算子偏向重算；函数形式则能根据 context、operator 和参数作决定。校验的是具体 overload（如 `aten.mm.default`），不是模糊的 overload packet。

A list is shorthand: listed ops are `MUST_SAVE` and others prefer recomputation. A function can instead inspect context, operator, and arguments. Validation requires a concrete overload such as `aten.mm.default`, not an ambiguous overload packet.

## 类比 / The analogy

像登山打包：把难以在山上重新取得的水带着，把随处可捡的树枝留给路上再找。

It is packing for a hike: carry water that is costly to replace, and gather easy-to-find sticks later.

## 自己跑一遍 / Try it yourself

```python
import functools, torch
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts
x = torch.randn(4, 4, requires_grad=True)
ctx = functools.partial(create_selective_checkpoint_contexts, [torch.ops.aten.mm.default])
out = checkpoint(lambda a: (a @ a).relu(), x, use_reentrant=False, context_fn=ctx)
out.sum().backward(); print(x.grad.shape)
```

运行 / Run with: `pip install torch && python try.py`

预期输出 / Expected output: `torch.Size([4, 4])`。

要点是策略围绕真正的 operator，而非 Python 函数名。 / The key is that the policy targets the actual operator, not a Python function name.

## 注意事项 / Caveats / when it breaks

- cache 条目被原地修改默认会报错，因为反向传播的重放必须可复现。 / Mutating cached entries errors by default because backward replay must be reproducible.

## 延伸阅读 / Further reading

- [PyTorch checkpoint source](https://github.com/pytorch/pytorch/blob/fed346df557d144bc8a48f2a018ca2403fcdb7e7/torch/utils/checkpoint.py#L1755-L1845)
