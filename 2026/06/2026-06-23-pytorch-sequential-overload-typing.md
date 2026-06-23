---
date: 2026-06-23
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/nn/modules/container.py
permalink: https://github.com/pytorch/pytorch/blob/ac9bd989af8f2e6e06e6f5b2eb5b4b5e0e5e5e5e/torch/nn/modules/container.py#L108-L175
difficulty: intermediate
read_time: ~8 min
tags: [code-of-the-day, pytorch, type-narrowing, overload, typing, sequential]
---

# Python `@overload` 精确类型收窄：`model[2]` → `Module`，`model[1:3]` → `Sequential` / Python `@overload` for Precise Type Narrowing: `model[2]` → `Module`, `model[1:3]` → `Sequential`

> **一句话 / In one line**: `Sequential.__getitem__` 用两条 `@overload` 存根告诉 mypy 和 IDE：下标是 `int` 时返回 `Module`，下标是 `slice` 时返回 `Sequential`——运行时只有一个函数体。 / `Sequential.__getitem__` carries two `@overload` stubs that teach mypy and your IDE the exact return type based on whether the index is an `int` or a `slice` — the single runtime implementation handles both.

## 为什么重要 / Why this matters

Python 的类型系统不能原生表达"同一个方法根据参数类型返回不同类型"。在没有 `@overload` 之前，`model[2]` 的类型只能是 `Module | Sequential`，IDE 就不知道该提示 `.weight` 还是 `.children()`。PyTorch 在 `Sequential` 上精确地解决了这个问题：用两条 `@overload` 存根教 mypy 做类型收窄，而运行时只有一个实际函数体来处理两种情况。此外，`@_copy_to_script_wrapper` 装饰器将同一个函数复用于 TorchScript，不需要额外写一份实现。当你编写自己的容器模块时，这是一个值得照搬的标准模式。

Python's type system can't natively express "this method returns different types depending on the argument type." Without `@overload`, `model[2]` can only be typed as `Module | Sequential`, leaving your IDE unable to suggest `.weight` vs `.children()`. PyTorch solves this cleanly in `Sequential`: two `@overload` stubs teach mypy to narrow the type, while a single runtime body handles both branches. The `@_copy_to_script_wrapper` decorator then reuses the same function for TorchScript without a second implementation. This is the pattern to copy when you write your own container modules.

## 代码 / The code

`pytorch/pytorch` — [`torch/nn/modules/container.py`](https://github.com/pytorch/pytorch/blob/ac9bd989af8f2e6e06e6f5b2eb5b4b5e0e5e5e5e/torch/nn/modules/container.py#L108-L175)

```python
from typing import overload, Union
from torch.nn.modules.module import Module

class Sequential(Module):
    # ... __init__, __len__, etc. ...

    @overload
    def __getitem__(self, idx: slice) -> "Sequential": ...

    @overload
    def __getitem__(self, idx: int) -> Module: ...

    @_copy_to_script_wrapper
    def __getitem__(self, idx: Union[slice, int]) -> Union["Sequential", Module]:
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)
```

## 逐行讲解 / What's happening

1. **`@overload` 存根 1 — `slice` 分支 / stub 1 — slice branch**:
   - 中文: 这是一条纯类型注解——函数体是 `...`（省略号），在运行时永远不会被执行。它只存在于类型检查工具的视角中，告诉 mypy：当 `idx` 是 `slice` 时，返回值是 `Sequential`。
   - English: This stub has `...` as its body and is never called at runtime — it exists purely for the type checker. It tells mypy: when `idx` is a `slice`, the return type is `Sequential`.

2. **`@overload` 存根 2 — `int` 分支 / stub 2 — int branch**:
   - 中文: 同上，但告诉 mypy `int` 下标返回 `Module`。有了这两条存根，`m[2].weight` 不会报类型错误，`m[1:3].children()` 也不会。
   - English: Same pattern, but signals that an `int` index returns a `Module`. With both stubs, `m[2].weight` no longer triggers a type error, and neither does `m[1:3].children()`.

3. **实际实现 — `Union[slice, int]` / the runtime implementation**:
   - 中文: 这是唯一在运行时执行的函数体。必须不带 `@overload` 装饰器，且类型注解用联合类型 `Union[slice, int]` 涵盖所有情况。它的返回类型写成 `Union["Sequential", Module]`，mypy 用上面两条存根取代它做精确推断。
   - English: This is the only body that executes at runtime. It must not carry `@overload`, and its annotation uses the union type to cover all cases. mypy ignores this body's return annotation in favour of the overload stubs when inferring call sites.

4. **`@_copy_to_script_wrapper`**:
   - 中文: PyTorch 内部装饰器，将这个 Python 方法的签名注册到 TorchScript 的方法表中，使 `torch.jit.script` 的容器模块也能正确调用它，而无需为 TorchScript 另写一份实现。
   - English: A PyTorch-internal decorator that registers this method's signature with TorchScript's dispatch table, so `torch.jit.script`-compiled container modules can call it correctly without a duplicate implementation.

5. **`slice` 分支 — 重建 `Sequential` / slice branch — rebuilding Sequential**:
   - 中文: `OrderedDict(list(self._modules.items())[idx])` 先把有序字典展成列表、做 Python 切片、再重新建一个 `OrderedDict`，最后传给 `self.__class__(...)` 构建子类型安全的新 `Sequential`（若是子类则返回子类实例）。
   - English: Converts `_modules` to a list, applies the Python slice, reconstructs an `OrderedDict`, and passes it to `self.__class__(...)` — this preserves subclass identity so a custom subclass of `Sequential` returns an instance of itself, not the base class.

6. **`int` 分支 — `_get_item_by_idx` / int branch**:
   - 中文: 支持负数索引（`model[-1]` 取最后一层），内部做 `idx % len(self)` 取模。
   - English: Handles negative indexing (`model[-1]` for the last layer) via modulo `idx % len(self)` internally.

## 类比 / The analogy

想象一个图书馆员，你问他"第 3 本书"，他递给你一本书；你问他"第 1 到第 3 本"，他递给你一叠书（小书堆）。`@overload` 就是图书馆的取书规则告示牌——告示牌（类型存根）告诉你拿到什么，而真正搬书的人（运行时实现）只有一个。

Imagine a librarian: ask for "book number 3" and you get one book; ask for "books 1 through 3" and you get a stack. The `@overload` stubs are the sign above the desk that tells you what you'll receive — one rule for a single number, another for a range. The actual librarian (runtime body) is just one person who reads the desk sign to decide which action to take.

## 自己跑一遍 / Try it yourself

```python
from typing import overload, Union
from collections import OrderedDict

class FakeModule:
    def __init__(self, name): self.name = name
    def __repr__(self): return f"Module({self.name})"

class MySeq:
    def __init__(self, modules: dict):
        self._modules = OrderedDict(modules)

    @overload
    def __getitem__(self, idx: slice) -> "MySeq": ...
    @overload
    def __getitem__(self, idx: int) -> FakeModule: ...

    def __getitem__(self, idx: Union[slice, int]) -> Union["MySeq", FakeModule]:
        if isinstance(idx, slice):
            return MySeq(dict(list(self._modules.items())[idx]))
        items = list(self._modules.values())
        return items[idx % len(items)]

seq = MySeq({"0": FakeModule("linear"), "1": FakeModule("relu"), "2": FakeModule("bn")})
print(type(seq[1]))      # <class 'FakeModule'>
print(seq[1])            # Module(relu)
print(type(seq[0:2]))    # <class 'MySeq'>
print(seq[-1])           # Module(bn)  — negative index
```

运行 / Run with:
```bash
python try.py  # no dependencies
```

预期输出 / Expected output:
```
<class '__main__.FakeModule'>
Module(relu)
<class '__main__.MySeq'>
Module(bn)
```

中文：在运行时，`@overload` 存根完全不参与执行——只有那个带 `Union` 签名的函数体真正被调用。若你用 mypy 检查这段代码，`seq[1]` 的类型会被推断为 `FakeModule`，`seq[0:2]` 会被推断为 `MySeq`。

English: At runtime, the `@overload` stubs are never called — only the union-typed body runs. Run mypy on this code and `seq[1]` is inferred as `FakeModule`, `seq[0:2]` as `MySeq` — precisely what the stubs declare.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **`dict.__getitem__` stubs in `typeshed`** / **`typeshed` 中的 `dict.__getitem__` 存根**: Python 标准库 stubs 里大量使用 `@overload` 来区分 `dict.get(key)` 和 `dict.get(key, default)` 的返回类型。 / The standard library stubs in `typeshed` use `@overload` extensively to distinguish `dict.get(key)` (returns `V | None`) from `dict.get(key, default)` (returns `V | D`).
- **`torch.Tensor.__getitem__`** / **`torch.Tensor.__getitem__`**: `Tensor` 的下标也有类似 overload stubs，区分单整数、切片、`...`、布尔掩码等情况。 / `Tensor` indexing has similar stubs that distinguish int indexing, slicing, `...`, and boolean mask cases.
- **`pandas.DataFrame.__getitem__`** / **`pandas.DataFrame.__getitem__`**: pandas stubs 里区分 `str` 下标（返回 `Series`）和 `list[str]` 下标（返回 `DataFrame`）。 / pandas stubs distinguish `str` indexing (returns `Series`) from `list[str]` indexing (returns `DataFrame`).

## 注意事项 / Caveats / when it breaks

- **运行时不检查** / **no runtime enforcement**: `@overload` 仅在静态分析阶段有效。运行时传入错误类型（比如 `float`）不会被存根拦截——你需要在实际实现里加 `isinstance` 检查。 / `@overload` is purely static. Passing a `float` at runtime skips the stubs entirely — you still need `isinstance` checks in the concrete body.
- **子类须保持签名兼容** / **subclasses must stay compatible**: If you subclass `Sequential` and override `__getitem__`, mypy will enforce that your overloads are compatible with the parent's. Incompatible overloads cause type errors downstream.
- **TorchScript 不识别 `@overload`** / **TorchScript ignores `@overload`**: TorchScript's parser sees only the concrete body — that's why `@_copy_to_script_wrapper` is needed to separately register the dispatch signature.

## 延伸阅读 / Further reading

- [PEP 484 — `@overload`](https://peps.python.org/pep-0484/#function-method-overloading)
- [mypy overload docs](https://mypy.readthedocs.io/en/stable/more_types.html#function-overloading)
- [pytorch/pytorch `container.py`](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/container.py)
