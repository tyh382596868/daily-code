---
date: 2026-10-22
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/utils/_pytree.py
permalink: https://github.com/pytorch/pytorch/blob/main/torch/utils/_pytree.py#L195-L264
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, pytree]
---

# PyTorch pytree 注册：给容器一份可逆说明书 / PyTorch Pytree Registration: Give a Container a Reversible Recipe

> **一句话 / In one line**: 注册时同时提供 flatten 与 unflatten，框架才能安全地把任意容器送进 transform。 / Register both flatten and unflatten so transforms can safely handle custom containers.

## 为什么重要 / Why this matters

`torch.func`、导出和批处理都需要把嵌套输入暂时摊平；只会拆开、不知道怎样装回去，变换就不能保持用户的容器语义。

`torch.func`, export, and batching must flatten nested inputs temporarily. A transform that cannot rebuild the container cannot preserve user-visible semantics.

## 代码 / The code

`pytorch/pytorch` — [`_pytree.py`](https://github.com/pytorch/pytorch/blob/main/torch/utils/_pytree.py#L195-L264)

```python
def register_pytree_node(cls, flatten_fn, unflatten_fn, *, serialized_type_name=None,
                         to_dumpable_context=None, from_dumpable_context=None,
                         flatten_with_keys_fn=None):
    with _NODE_REGISTRY_LOCK:
        if cls in SUPPORTED_NODES:
            raise ValueError(f"{cls} is already registered as pytree node.")
    _private_register_pytree_node(cls, flatten_fn, unflatten_fn,
        serialized_type_name=serialized_type_name,
        to_dumpable_context=to_dumpable_context,
        from_dumpable_context=from_dumpable_context,
        flatten_with_keys_fn=flatten_with_keys_fn)
```

## 逐行讲解 / What's happening

1. **锁 / Lock**：先检查注册表，避免两个线程为同一类型留下不同规则。 / The lock makes the type-to-rule mapping deterministic.
2. **双向契约 / Bidirectional contract**：`flatten_fn` 返回叶子与 context，`unflatten_fn` 用两者复原对象。 / Leaves plus context are sufficient to reconstruct the object.
3. **路径版 flatten / Key-aware flatten**：可选 key path 让错误报告和选择性变换知道叶子来自哪里。 / Optional key paths retain leaf provenance.

## 类比 / The analogy

像搬家清单：箱子里的物品是 leaves，房间编号是 context；没有编号，搬回去只能得到一堆东西。 / Like a moving inventory: leaves are belongings and context is the room label.

## 自己跑一遍 / Try it yourself

```python
from dataclasses import dataclass
from torch.utils import _pytree
@dataclass
class Pair: left: object; right: object
_pytree.register_dataclass(Pair)
leaves, spec = _pytree.tree_flatten(Pair(1, 2))
print(leaves, _pytree.tree_unflatten(leaves, spec))
```

运行 / Run with: `pip install torch && python try.py`  
预期输出 / Expected output: `[1, 2]` 以及重建出的 `Pair`。 / `[1, 2]` and the rebuilt `Pair`.

## 注意事项 / Caveats / when it breaks

- **可逆性 / Reversibility**：context 必须包含重建所需的结构信息。 / Context must contain all structural information needed for reconstruction.
