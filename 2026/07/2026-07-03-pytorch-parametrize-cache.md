---
date: 2026-07-03
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/nn/utils/parametrize.py
permalink: https://github.com/pytorch/pytorch/blob/main/torch/nn/utils/parametrize.py#L24-L65
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, parametrization, cache]
---

# PyTorch parametrization cache：一次 forward 里别重复算同一个权重 / PyTorch Parametrization Cache: Do Not Recompute the Same Weight in One Forward

> **一句话 / In one line**: `parametrize.cached()` 用一个上下文管理器临时缓存参数化后的 tensor，退出时自动清空。 / `parametrize.cached()` uses a context manager to temporarily cache parametrized tensors and clears them on exit.

## 为什么重要 / Why this matters

权重参数化常见于谱归一化、正交约束、共享 recurrent kernel。参数本体每次访问都可能触发一次变换；如果同一个 forward 里多次读取同一个参数，重复计算既慢又容易让代码变得隐蔽。

Weight parametrizations show up in spectral normalization, orthogonal constraints, and shared recurrent kernels. Reading the parameter may run a transform each time; if one forward reads the same parameter repeatedly, recomputation is wasteful and hard to notice.

## 代码 / The code

`pytorch/pytorch` — [`torch/nn/utils/parametrize.py`](https://github.com/pytorch/pytorch/blob/main/torch/nn/utils/parametrize.py#L24-L65)

```python
_cache_enabled = 0
_cache: dict[tuple[int, str], Tensor | None] = {}


@contextmanager
def cached():
    r"""Context manager that enables the caching system within parametrizations registered with :func:`register_parametrization`.

    The value of the parametrized objects is computed and cached the first time
    they are required when this context manager is active. The cached values are
    discarded when leaving the context manager.
    """
    global _cache
    global _cache_enabled
    _cache_enabled += 1
    try:
        yield
    finally:
        _cache_enabled -= 1
        if not _cache_enabled:
            _cache = {}
```

## 逐行讲解 / What's happening

1. **全局层数计数 / Global nesting counter**: 中文: `_cache_enabled` 不是布尔值，而是计数器，所以嵌套 `with cached()` 也能正确工作。 / English: `_cache_enabled` is a counter, not a boolean, so nested `with cached()` blocks behave correctly.
2. **key 是模块和名字 / Keys are module and name**: 中文: `_cache` 用 `(id(module), tensor_name)` 这类键区分不同参数化对象。 / English: `_cache` distinguishes parametrized objects with keys such as `(id(module), tensor_name)`.
3. **退出最外层才清空 / Clear only after the outermost exit**: 中文: `finally` 保证异常时也递减；只有计数归零才丢掉缓存。 / English: `finally` decrements even on exceptions; cached values are dropped only when the counter reaches zero.

## 类比 / The analogy

这像餐厅后厨的临时备菜台：一轮晚餐服务期间常用的配料先放手边，服务结束就全部清掉，避免下一轮拿到旧东西。

It is like a prep station in a kitchen. Frequently used ingredients stay close during one service, then everything is cleared before the next service starts.

## 自己跑一遍 / Try it yourself

```python
from contextlib import contextmanager

cache_enabled = 0
cache = {}

@contextmanager
def cached():
    global cache_enabled, cache
    cache_enabled += 1
    try:
        yield
    finally:
        cache_enabled -= 1
        if not cache_enabled:
            cache = {}

with cached():
    cache["weight"] = "computed once"
    print(cache)
print(cache)
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
{'weight': 'computed once'}
{}
```

核心现象是“缓存只活在上下文里”。这比把缓存挂在模块上更安全，因为 forward 之外不会留下旧的派生权重。

The key behavior is that the cache only lives inside the context. That is safer than storing it on the module, because derived weights do not leak outside the forward.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **`torch.no_grad()`** / **`torch.no_grad()`**: 同样用上下文管理器改变一段代码的运行语义。
- **activation checkpointing** / **Activation checkpointing**: 也是用显式边界控制中间值生命周期。

## 注意事项 / Caveats / when it breaks

- **不要跨 forward 复用** / **Do not reuse across forwards**: 参数可能被 optimizer 更新，旧缓存会错。
- **只缓存派生值** / **Cache derived values only**: 原始参数仍应由模块和 optimizer 管理。

## 延伸阅读 / Further reading

- [PyTorch parametrizations docs](https://pytorch.org/docs/stable/nn.utils.parametrize.html)
- [Source file](https://github.com/pytorch/pytorch/blob/main/torch/nn/utils/parametrize.py)
