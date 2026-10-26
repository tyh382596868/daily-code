---
date: 2026-10-26
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/autograd/graph.py
permalink: https://github.com/pytorch/pytorch/blob/6d5f66bd5ba59d8d7a428b15b261df292b03dbd9/torch/autograd/graph.py#L297-L389
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, pytorch, autograd, memory]
---

# PyTorch 保存张量钩子：给反向传播换一套仓库 / PyTorch Saved-Tensor Hooks: Give Backward a Different Warehouse

> **一句话 / In one line**: forward 保存中间张量时可先打包，backward 需要时再解包。 / Pack intermediates at save time and unpack them only when backward needs them.

## 为什么重要 / Why this matters

训练显存往往被反向传播需要的激活占据。这个上下文把“如何保存”从算子逻辑中抽出，`save_on_cpu` 就是同一接口的内存换带宽实例。

Activation memory dominates many training jobs. This context separates storage policy from operator math; `save_on_cpu` is its memory-for-bandwidth implementation.

## 代码 / The code

```python
class saved_tensors_hooks:
    def __init__(self, pack_hook, unpack_hook):
        self.pack_hook = pack_hook
        self.unpack_hook = unpack_hook
    def __enter__(self):
        torch._C._autograd._push_saved_tensors_default_hooks(
            self.pack_hook, self.unpack_hook)
    def __exit__(self, *args):
        torch._C._autograd._pop_saved_tensors_default_hooks()
```

来源 / Source: [`torch/autograd/graph.py`](https://github.com/pytorch/pytorch/blob/6d5f66bd5ba59d8d7a428b15b261df292b03dbd9/torch/autograd/graph.py#L297-L389)。

## 逐行讲解 / What's happening

进入上下文时，C++ autograd 引擎取得一对默认 hook；退出时必定弹栈，因此嵌套上下文只影响自己的动态作用域。

Entering installs a hook pair in the autograd engine; exiting pops it, so the policy has a bounded dynamic scope.

## 类比 / The analogy

像把仓库货架改成压缩箱：入库时压缩，取货时还原，配方本身不用知道仓库在哪。

Like swapping warehouse shelves for compression crates: pack on arrival, restore on pickup, without changing the recipe.

## 自己跑一遍 / Try it yourself

```python
import torch
seen = []
with torch.autograd.graph.saved_tensors_hooks(
    lambda x: (seen.append("pack"), x.detach())[1],
    lambda x: (seen.append("unpack"), x):
    x = torch.ones(2, requires_grad=True); (x * x).sum().backward()
print(seen)
```

运行 / Run with `pip install torch && python try.py`。预期 / Expected: 包含 `pack` 和 `unpack`。

## 注意事项 / Caveats / when it breaks

- **顺序不是 API / Ordering is not an API**：不要假设解包顺序等于打包的反序。 / Do not rely on reverse packing order.
