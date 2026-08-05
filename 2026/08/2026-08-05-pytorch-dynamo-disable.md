---
date: 2026-08-05
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/_dynamo/decorators.py
permalink: https://github.com/pytorch/pytorch/blob/e27c7219bd432b1048e1e8427010fa159d0ab6b6/torch/_dynamo/decorators.py#L87-L144
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, torch-compile, dynamo]
---

# PyTorch Dynamo disable：跳过当前函数，还是跳过整棵调用树 / PyTorch Dynamo disable: Skip One Function or the Whole Call Tree

> **一句话 / In one line**: `disable(recursive=True)` 用 `DisableContext` 让 Dynamo 完全跳过函数树；`recursive=False` 只给当前 wrapper 打标记，让内部调用仍可继续被追踪。 / `disable(recursive=True)` uses `DisableContext` to skip a whole function tree; `recursive=False` only marks the current wrapper so inner calls can still be traced.

## 为什么重要 / Why this matters

`torch.compile` 遇到 Python 边界、debug 逻辑或第三方对象时，经常需要人为告诉 Dynamo “这里别编译”。但“别编译”有两种语义：有时整段业务逻辑都不该进图，有时只是一个外壳不该进图，里面的 tensor 计算仍然值得编译。这个函数把两种语义做成一个显式开关。

When `torch.compile` hits Python boundaries, debug logic, or third-party objects, you often need to tell Dynamo to stay away. But "stay away" has two meanings: sometimes the whole business function should be outside the graph; sometimes only the wrapper is untraceable while inner tensor code is still worth compiling. This function makes that distinction explicit.

## 代码 / The code

`pytorch/pytorch` — [`torch/_dynamo/decorators.py`](https://github.com/pytorch/pytorch/blob/e27c7219bd432b1048e1e8427010fa159d0ab6b6/torch/_dynamo/decorators.py#L87-L144)

```python
def disable(fn=None, recursive=True, *, reason=None, wrapping=True):  # type: ignore[no-untyped-def]
    """
    Decorator to disable TorchDynamo

    If recursive=True, Dynamo is completely skipped on the decorated function
    frame as well as the recursively invoked functions.

    If recursive=False, Dynamo skips frames associated with the function code,
    but still process recursively invoked frames.

    If reason is provided, it will be printed when Dynamo attempts to trace the disabled function.
    """
    if recursive:
        if fn is not None:
            fn = innermost_fn(fn)
            if not callable(fn):
                raise AssertionError("fn must be callable")
            return DisableContext(msg=reason, wrapping=wrapping)(fn)
        return DisableContext(msg=reason, wrapping=wrapping)
    else:

        def wrap(fn: Callable[_P, _R]) -> Callable[_P, _R]:
            fn = innermost_fn(fn)
            if not callable(fn):
                raise AssertionError("fn must be callable")

            nonrecursive_disable_wrapper = get_nonrecursive_disable_wrapper(fn)
            nonrecursive_disable_wrapper._torchdynamo_disable = True  # type: ignore[attr-defined]
            nonrecursive_disable_wrapper._torchdynamo_disable_msg = reason  # type: ignore[attr-defined]
            nonrecursive_disable_wrapper._torchdynamo_orig_callable = fn  # type: ignore[attr-defined]
            nonrecursive_disable_wrapper._torchdynamo_wrapper_id = id(  # type: ignore[attr-defined]
                nonrecursive_disable_wrapper
            )
            nonrecursive_disable_wrapper._torchdynamo_disable_recursive = False  # type: ignore[attr-defined]
            return nonrecursive_disable_wrapper

        if fn is None:
            return wrap
        return wrap(fn)


_nonrecursive_disable_wrapper_code = disable(lambda: None, recursive=False).__code__  # type: ignore[attr-defined]
skip_code(_nonrecursive_disable_wrapper_code)


def skip(fn: Callable[_P, _R] | None = None) -> Callable[..., Any]:
    """
    Skip frames associated with the function code, but still process recursively
    invoked frames
    """
    if fn is None:
        return skip
    fn = innermost_fn(fn)
    if not callable(fn):
        raise AssertionError("fn must be callable")
    skip_code(fn.__code__)
    fn._torchdynamo_disable = True  # type: ignore[attr-defined]
    return fn
```

## 逐行讲解 / What's happening

1. **第 99-105 行 / Lines 99-105 (recursive path)**:
   - 中文: 默认路径直接返回 `DisableContext`。如果传了函数，它会立刻包住函数；如果没传函数，它返回一个可当 decorator/context manager 用的对象。
   - English: The default path returns `DisableContext`. If a function is supplied, it wraps the function immediately; otherwise it returns an object usable as a decorator or context manager.
2. **第 108-125 行 / Lines 108-125 (non-recursive wrapper)**:
   - 中文: 非递归模式创建一个 wrapper，并在 wrapper 上挂 `_torchdynamo_disable_recursive = False`。这不是普通注释，而是 Dynamo 后续识别语义的标记。
   - English: Non-recursive mode creates a wrapper and stamps `_torchdynamo_disable_recursive = False` on it. This is not decorative; Dynamo reads those attributes later.
3. **第 128-129 行 / Lines 128-129 (skip the wrapper code)**:
   - 中文: PyTorch 先造一个假的非递归 wrapper，拿到它的 `__code__`，再把这个 wrapper 自己也加入 skip 表，避免追踪器钻进包装层。
   - English: PyTorch builds one dummy non-recursive wrapper, grabs its `__code__`, and skips that wrapper implementation too, so the tracer does not waste time inside the wrapper layer.
4. **第 132-144 行 / Lines 132-144 (`skip`)**:
   - 中文: `skip` 是更直接的版本：按函数 `__code__` 跳过当前 frame，同时给函数对象打 `_torchdynamo_disable`。
   - English: `skip` is the direct version: it skips the function's `__code__` and marks the function object with `_torchdynamo_disable`.

## 类比 / The analogy

这像给办公楼贴施工标识。`recursive=True` 是“整栋楼今天不开门”；`recursive=False` 是“前台在装修，请从后门进，楼上的办公室照常办公”。

It is like posting construction signs in an office building. `recursive=True` says the whole building is closed. `recursive=False` says the lobby is closed, but the offices upstairs are still open through another entrance.

## 自己跑一遍 / Try it yourself

```python
def disable(fn=None, recursive=True, reason=None):
    def wrap(f):
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        wrapper.disabled = True
        wrapper.recursive = recursive
        wrapper.reason = reason
        return wrapper
    return wrap if fn is None else wrap(fn)

@disable(recursive=False, reason="python wrapper")
def outer(x):
    return x + 1

print(outer(4), outer.disabled, outer.recursive, outer.reason)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
5 True False python wrapper
```

中文: 小例子只模拟“给 wrapper 打标记”，真实 Dynamo 会用这些标记决定 frame 追踪边界。

English: The toy version only simulates stamping metadata on the wrapper; real Dynamo uses those flags to decide tracing boundaries.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **`torch.compiler.allow_in_graph`** / **`torch.compiler.allow_in_graph`**: 同样通过给 callable 注册语义，让编译器改变处理方式。 / It also registers callable-level semantics to change compiler behavior.
- **pytest marks** / **pytest marks**: 测试函数上挂 metadata，runner 再按 metadata 决定是否跳过或改变运行方式。 / Test functions carry metadata, and the runner decides whether to skip or alter execution.

## 注意事项 / Caveats / when it breaks

- **非递归不是“什么都不编译” / Non-recursive does not mean compile nothing**: 内部调用仍可能被 Dynamo 处理。 / Inner calls can still be handled by Dynamo.
- **标记依赖函数身份 / Marks depend on function identity**: 包装、重新绑定或复制函数对象时，标记语义可能不再落在你以为的对象上。 / Wrapping, rebinding, or copying callables can move the mark away from the object you expected.

## 延伸阅读 / Further reading

- PyTorch Dynamo decorators: https://github.com/pytorch/pytorch/blob/e27c7219bd432b1048e1e8427010fa159d0ab6b6/torch/_dynamo/decorators.py#L87-L144
