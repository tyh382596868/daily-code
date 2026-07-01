---
date: 2026-07-01
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/nn/modules/module.py
permalink: https://github.com/pytorch/pytorch/blob/4462196ae007559bb72618b68722d0f68de9c41b/torch/nn/modules/module.py#L674-L818
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, module-surgery]
---

# PyTorch 子模块手术：用点路径精准替换一层 / PyTorch Submodule Surgery: Replace a Layer by Dotted Path

> **一句话 / In one line**: `get_submodule` 和 `set_submodule` 把 `net.block.attn` 这种字符串变成 O(depth) 的模块树定位和替换。 / `get_submodule` and `set_submodule` turn strings like `net.block.attn` into O(depth) module-tree lookup and replacement.

## 为什么重要 / Why this matters

大型模型改造经常要按名字换掉某个子层：插 LoRA、替换 attention、裁剪 head、注入 quantized linear。PyTorch 没有扫描整个 `named_modules()`，而是沿着点路径逐级 `getattr`，最后在父模块上 `setattr`。这让模型手术的成本跟路径深度相关，而不是跟全模型层数相关。

Large-model surgery often replaces one named child: insert LoRA, swap attention, prune a head, or inject a quantized linear. PyTorch avoids scanning all `named_modules()`; it walks the dotted path with `getattr`, then uses `setattr` on the parent. The cost tracks path depth, not total model size.

## 代码 / The code

`pytorch/pytorch` — [`torch/nn/modules/module.py`](https://github.com/pytorch/pytorch/blob/4462196ae007559bb72618b68722d0f68de9c41b/torch/nn/modules/module.py#L674-L818)

```python
    def get_submodule(self, target: str) -> "Module":
        """Return the submodule given by ``target`` if it exists, otherwise throw an error.

        For example, let's say you have an ``nn.Module`` ``A`` that
        looks like this:

        .. code-block:: text

            A(
                (net_b): Module(
                    (net_c): Module(
                        (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
                    )
                    (linear): Linear(in_features=100, out_features=200, bias=True)
                )
            )

        (The diagram shows an ``nn.Module`` ``A``. ``A`` which has a nested
        submodule ``net_b``, which itself has two submodules ``net_c``
        and ``linear``. ``net_c`` then has a submodule ``conv``.)

        To check whether or not we have the ``linear`` submodule, we
        would call ``get_submodule("net_b.linear")``. To check whether
        we have the ``conv`` submodule, we would call
        ``get_submodule("net_b.net_c.conv")``.

        The runtime of ``get_submodule`` is bounded by the degree
        of module nesting in ``target``. A query against
        ``named_modules`` achieves the same result, but it is O(N) in
        the number of transitive modules. So, for a simple check to see
        if some submodule exists, ``get_submodule`` should always be
        used.

        Args:
            target: The fully-qualified string name of the submodule
                to look for. (See above example for how to specify a
                fully-qualified string.)

        Returns:
            torch.nn.Module: The submodule referenced by ``target``

        Raises:
            AttributeError: If at any point along the path resulting from
                the target string the (sub)path resolves to a non-existent
                attribute name or an object that is not an instance of ``nn.Module``.
        """
        if target == "":
            return self

        atoms: list[str] = target.split(".")
        mod: torch.nn.Module = self

        for item in atoms:
            if not hasattr(mod, item):
                raise AttributeError(
                    mod._get_name() + " has no attribute `" + item + "`"
                )

            mod = getattr(mod, item)

            if not isinstance(mod, torch.nn.Module):
                raise AttributeError("`" + item + "` is not an nn.Module")

        return mod

    def set_submodule(
        self, target: str, module: "Module", strict: bool = False
    ) -> None:
        """
        Set the submodule given by ``target`` if it exists, otherwise throw an error.

        .. note::
            If ``strict`` is set to ``False`` (default), the method will replace an existing submodule
            or create a new submodule if the parent module exists. If ``strict`` is set to ``True``,
            the method will only attempt to replace an existing submodule and throw an error if
            the submodule does not exist.

        For example, let's say you have an ``nn.Module`` ``A`` that
        looks like this:

        .. code-block:: text

            A(
                (net_b): Module(
                    (net_c): Module(
                        (conv): Conv2d(3, 3, 3)
                    )
                    (linear): Linear(3, 3)
                )
            )

        (The diagram shows an ``nn.Module`` ``A``. ``A`` has a nested
        submodule ``net_b``, which itself has two submodules ``net_c``
        and ``linear``. ``net_c`` then has a submodule ``conv``.)

        To override the ``Conv2d`` with a new submodule ``Linear``, you
        could call ``set_submodule("net_b.net_c.conv", nn.Linear(1, 1))``
        where ``strict`` could be ``True`` or ``False``

        To add a new submodule ``Conv2d`` to the existing ``net_b`` module,
        you would call ``set_submodule("net_b.conv", nn.Conv2d(1, 1, 1))``.

        In the above if you set ``strict=True`` and call
        ``set_submodule("net_b.conv", nn.Conv2d(1, 1, 1), strict=True)``, an AttributeError
        will be raised because ``net_b`` does not have a submodule named ``conv``.

        Args:
            target: The fully-qualified string name of the submodule
                to look for. (See above example for how to specify a
                fully-qualified string.)
            module: The module to set the submodule to.
            strict: If ``False``, the method will replace an existing submodule
                or create a new submodule if the parent module exists. If ``True``,
                the method will only attempt to replace an existing submodule and throw an error
                if the submodule doesn't already exist.

        Raises:
            ValueError: If the ``target`` string is empty or if ``module`` is not an instance of ``nn.Module``.
            AttributeError: If at any point along the path resulting from
                the ``target`` string the (sub)path resolves to a non-existent
                attribute name or an object that is not an instance of ``nn.Module``.
        """
        if target == "":
            raise ValueError("Cannot set the submodule without a target name!")

        atoms: list[str] = target.split(".")
        if not isinstance(module, torch.nn.Module):
            raise ValueError(
                "`" + "module" + f"` is not an nn.Module, found {type(module)}"
            )
        if len(atoms) == 1:
            parent: torch.nn.Module = self
        else:
            parent_key = ".".join(atoms[:-1])
            parent = self.get_submodule(parent_key)

        if strict and not hasattr(parent, atoms[-1]):
            raise AttributeError(
                parent._get_name() + " has no attribute `" + atoms[-1] + "`"
            )
        if hasattr(parent, atoms[-1]):
            mod = getattr(parent, atoms[-1])
            if not isinstance(mod, torch.nn.Module):
                raise AttributeError("`" + atoms[-1] + "` is not an nn.Module")
        setattr(parent, atoms[-1], module)
```

## 逐行讲解 / What's happening

1. **第 720-737 行 / Lines 720-737**: 中文: 空路径返回自己；非空路径逐段走属性，任何一段不存在或不是 `nn.Module` 都立即报错。 / English: An empty path returns `self`; otherwise each atom is resolved as an attribute, and missing or non-module atoms fail immediately.
2. **第 799-808 行 / Lines 799-808**: 中文: `set_submodule` 先把目标拆成父路径和叶子名，父路径仍复用 `get_submodule`。 / English: `set_submodule` splits the target into parent path and leaf name, then reuses `get_submodule` for the parent.
3. **第 810-818 行 / Lines 810-818**: 中文: `strict=True` 禁止新增叶子；已有叶子若不是 module 也拒绝覆盖；最后交给 `setattr`，触发 PyTorch 正常的 module 注册逻辑。 / English: `strict=True` forbids creating a missing leaf; an existing non-module leaf is rejected; the final `setattr` triggers PyTorch's normal module registration logic.

## 类比 / The analogy

像在文件柜里换一个文件夹：你不需要翻完整个档案室，只要按 `A/B/C` 走到父文件夹，再替换里面那一个夹子。

It is like replacing a folder in a filing cabinet: you do not scan the whole archive; you follow `A/B/C` to the parent folder and replace one slot.


## 自己跑一遍 / Try it yourself

```python
import torch
from torch import nn
model = nn.Sequential(nn.Linear(3, 4), nn.Sequential(nn.ReLU(), nn.Linear(4, 2)))
print(model.get_submodule('1.1'))
model.set_submodule('1.1', nn.Linear(4, 1))
print(model.get_submodule('1.1'))
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```text
Linear(in_features=4, out_features=2, bias=True)
Linear(in_features=4, out_features=1, bias=True)
```

中文: 这个小例子保留了源码里的关键控制流，但把依赖压到最低，便于你直接观察形状、索引或状态变化。

English: The miniature keeps the original control-flow idea while stripping dependencies down so the shape, index, or state change is visible.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **PEFT adapter injection** / **PEFT adapter injection**: 中文: adapter 库按模块名定位 Linear 并替换包装层。 / English: Adapter libraries locate Linear modules by path and replace them with wrappers.
- **量化替换 / Quantization replacement**: 中文: post-training quantization 常把 `nn.Linear` 批量换成量化实现。 / English: Post-training quantization often swaps `nn.Linear` modules for quantized implementations.

## 注意事项 / Caveats / when it breaks

- **路径必须指向 module / Path must resolve to a module**: 中文: 指到 tensor、参数或普通属性会报错。 / English: Paths resolving to tensors, parameters, or plain attributes fail.
- **新增只新增叶子 / Creation only happens at the leaf**: 中文: `strict=False` 也要求父路径存在。 / English: Even with `strict=False`, the parent path must already exist.

## 延伸阅读 / Further reading

- Source permalink above.
- Project repository linked from the frontmatter.
