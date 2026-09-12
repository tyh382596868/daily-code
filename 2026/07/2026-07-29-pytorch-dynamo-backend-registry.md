---
date: 2026-07-29
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/_dynamo/backends/registry.py
permalink: https://github.com/pytorch/pytorch/blob/26314d9476d043c95f33a9cc40c468b5a1643428/torch/_dynamo/backends/registry.py#L87-L160
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, torch-compile, registry]
---

# PyTorch Dynamo backend：字符串最后要变成 callable / PyTorch Dynamo Backends: Strings Eventually Become Callables

> **一句话 / In one line**: Dynamo 用一个 registry 把 `torch.compile(..., backend="inductor")` 里的字符串延迟解析成真正的编译函数。 / Dynamo uses a registry to lazily resolve a string such as `backend="inductor"` into the real compiler function.

## 为什么重要 / Why this matters

`torch.compile` 要支持内置后端、调试后端、实验后端和第三方 entry point。如果每次 import PyTorch 都加载所有后端，启动成本会很高；如果直接接受字符串又不校验，错误会很难定位。

`torch.compile` has to support built-in backends, debug backends, experimental backends, and third-party entry points. Loading all of them during PyTorch import would be expensive; accepting arbitrary strings without validation would make errors confusing.

## 代码 / The code

`pytorch/pytorch` — [`torch/_dynamo/backends/registry.py`](https://github.com/pytorch/pytorch/blob/26314d9476d043c95f33a9cc40c468b5a1643428/torch/_dynamo/backends/registry.py#L87-L160)

```python
def register_backend(
    compiler_fn: CompilerFn | None = None,
    name: str | None = None,
    tags: Sequence[str] = (),
) -> Callable[..., Any]:
    """
    Decorator to add a given compiler to the registry to allow calling
    `torch.compile` with string shorthand.  Note: for projects not
    imported by default, it might be easier to pass a function directly
    as a backend and not use a string.
    """
    if compiler_fn is None:
        return functools.partial(register_backend, name=name, tags=tags)  # type: ignore[return-value]
    if not callable(compiler_fn):
        raise AssertionError(f"compiler_fn must be callable, got {type(compiler_fn)}")
    name = name or compiler_fn.__name__
    if name in _COMPILER_FNS:
        raise AssertionError(f"duplicate name: {name}")
    if name not in _BACKENDS:
        _BACKENDS[name] = None
    _COMPILER_FNS[name] = compiler_fn
    _BACKEND_TAGS[name] = tuple(tags)
    return compiler_fn


register_debug_backend = functools.partial(register_backend, tags=("debug",))
register_experimental_backend = functools.partial(
    register_backend, tags=("experimental",)
)


def lookup_backend(compiler_fn: str | CompilerFn) -> CompilerFn:
    """Expand backend strings to functions"""
    if isinstance(compiler_fn, str):
        if compiler_fn not in _BACKENDS:
            _lazy_import()
        if compiler_fn not in _BACKENDS:
            import difflib

            from ..exc import InvalidBackend

            suggestions = difflib.get_close_matches(compiler_fn, list_backends(), n=2)
            raise InvalidBackend(name=compiler_fn, suggestions=suggestions)

        if compiler_fn not in _COMPILER_FNS:
            entry_point = _BACKENDS[compiler_fn]
            if entry_point is not None:
                register_backend(compiler_fn=entry_point.load(), name=compiler_fn)
        compiler_fn = _COMPILER_FNS[compiler_fn]
    return compiler_fn
```

## 逐行讲解 / What's happening

1. **第 87-105 行 / Lines 87-105 (`register_backend`)**:
   - 中文: 同一个函数同时支持 `@register_backend` 和 `@register_backend(name="x")` 两种装饰器写法。
   - English: The same function supports both `@register_backend` and `@register_backend(name="x")` decorator styles.
2. **第 106-115 行 / Lines 106-115 (validation and tables)**:
   - 中文: 非 callable、重复名字都会直接报错；成功后写入函数表和 tag 表。
   - English: Non-callables and duplicate names fail immediately; successful registrations populate the function and tag tables.
3. **第 124-141 行 / Lines 124-141 (`lookup_backend`)**:
   - 中文: 只有查找字符串时才 `_lazy_import()`，找不到时用 `difflib` 给近似建议。
   - English: `_lazy_import()` runs only when a string needs resolution; unknown names get close-match suggestions through `difflib`.

## 类比 / The analogy

这像机场登机口屏幕：旅客看到的是航班号，系统内部最后要找到具体登机口；如果输错航班号，屏幕会提示最像的候选，而不是让你猜。

It is like an airport gate display: passengers see a flight number, but the system must resolve it to a concrete gate; if the flight number is wrong, it suggests close matches instead of leaving you guessing.

## 自己跑一遍 / Try it yourself

```python
import difflib
REGISTRY = {}

def register(fn=None, name=None):
    if fn is None:
        return lambda real_fn: register(real_fn, name)
    key = name or fn.__name__
    if key in REGISTRY:
        raise AssertionError(f"duplicate name: {key}")
    REGISTRY[key] = fn
    return fn

@register(name="fast")
def compile_fast(graph, inputs):
    return "compiled"

def lookup(name):
    if name not in REGISTRY:
        print("did you mean", difflib.get_close_matches(name, REGISTRY, n=1))
        return None
    return REGISTRY[name]

print(lookup("fast")("g", []))
lookup("fas")
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
compiled
did you mean ['fast']
```

这个小例子保留了 PyTorch 代码的核心：用户接口可以是字符串，但执行层必须拿到函数。

This toy version keeps the core PyTorch idea: the user-facing API can accept a string, but the execution layer needs a function.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Flask route registry** / **Flask route registry**: URL 字符串最终映射到 view function。 / URL strings eventually map to view functions.
- **插件系统** / **Plugin systems**: 配置文件里写名字，运行时再加载 entry point。 / Config files store names, while runtime loads entry points.

## 注意事项 / Caveats / when it breaks

- **名字是全局命名空间** / **Names share one namespace**: 重名后端会被拒绝。 / Duplicate backend names are rejected.
- **lazy import 隐藏成本** / **Lazy import hides cost**: 第一次 lookup 可能比后续 lookup 慢。 / The first lookup can be slower than later ones.

## 延伸阅读 / Further reading

- [torch.compile documentation](https://pytorch.org/docs/stable/generated/torch.compile.html)
- [Source permalink](https://github.com/pytorch/pytorch/blob/26314d9476d043c95f33a9cc40c468b5a1643428/torch/_dynamo/backends/registry.py#L87-L160)
