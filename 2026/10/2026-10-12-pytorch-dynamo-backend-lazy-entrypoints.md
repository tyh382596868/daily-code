---
date: 2026-10-12
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/_dynamo/backends/registry.py
permalink: https://github.com/pytorch/pytorch/blob/d2f3a9d2e41e2fb201224ec61f74aa000a37e513/torch/_dynamo/backends/registry.py#L87-L188
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, torch-compile, backend-registry]
---

# PyTorch Dynamo backend：字符串后端延迟变成函数 / PyTorch Dynamo Backends: Backend Strings Become Functions Lazily

> **一句话 / In one line**: `torch.compile(backend="name")` 背后是一张懒加载注册表，只有真正查找时才 import 内置后端或加载 entry point。 / Behind `torch.compile(backend="name")` is a lazy registry that imports built-in backends or loads entry points only when lookup actually happens.

## 为什么重要 / Why this matters

编译后端注册表要同时满足两个目标：启动快，以及可扩展。PyTorch 不能在 import `torch` 时把所有编译后端都加载一遍，但又要让用户能用字符串选择内置后端或第三方后端。这段代码展示了一个常见框架模式：先登记名字和 entry point，等用户真的点名时再把名字解析成 callable。

A backend registry has to be both fast at startup and extensible. PyTorch cannot import every compiler backend when `torch` imports, but users still need string names for built-in and third-party backends. This code shows a common framework pattern: register names and entry points first, resolve a name into a callable only when someone asks for it.

## 代码 / The code

`pytorch/pytorch` - [`torch/_dynamo/backends/registry.py`](https://github.com/pytorch/pytorch/blob/d2f3a9d2e41e2fb201224ec61f74aa000a37e513/torch/_dynamo/backends/registry.py#L87-L188)

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

    Args:
        compiler_fn: Callable taking a FX graph and fake tensor inputs
        name: Optional name, defaults to `compiler_fn.__name__`
        tags: Optional set of string tags to categorize backend with
    """
    if compiler_fn is None:
        # @register_backend(name="") syntax
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


# NOTE: can't type this due to public api mismatch; follow up with dev team
def list_backends(exclude_tags=("debug", "experimental")) -> list[str]:  # type: ignore[no-untyped-def]
    """
    Return valid strings that can be passed to:

        torch.compile(..., backend="name")
    """
    _lazy_import()
    exclude_tags_set = set(exclude_tags or ())

    backends = [
        name
        for name in _BACKENDS
        if not exclude_tags_set.intersection(_BACKEND_TAGS.get(name, ()))
    ]
    return sorted(backends)


@functools.cache
def _lazy_import() -> None:
    from .. import backends
    from ..utils import import_submodule

    import_submodule(backends)

    from ..repro.after_dynamo import dynamo_minifier_backend

    if dynamo_minifier_backend is None:
        raise AssertionError("dynamo_minifier_backend failed to load")

    _discover_entrypoint_backends()


@functools.cache
def _discover_entrypoint_backends() -> None:
    # importing here so it will pick up the mocked version in test_backends.py
    from importlib.metadata import entry_points

    group_name = "torch_dynamo_backends"
    eps = entry_points(group=group_name)
    # pyrefly: ignore [bad-index]
    eps_dict = {name: eps[name] for name in eps.names}
    for backend_name in eps_dict:
        _BACKENDS[backend_name] = eps_dict[backend_name]
```

## 逐行讲解 / What's happening

1. **第 87-115 行 / Lines 87-115 (`register_backend`)**:
   - 中文: 这个函数既能直接注册 callable，也能当 decorator 用。注册时会防重名，把名字放进 `_BACKENDS`，把真正函数放进 `_COMPILER_FNS`。
   - English: The function can register a callable directly or act as a decorator. Registration rejects duplicate names, records the name in `_BACKENDS`, and stores the real callable in `_COMPILER_FNS`.

2. **第 124-142 行 / Lines 124-142 (`lookup_backend`)**:
   - 中文: 如果用户传入字符串，先看 `_BACKENDS`。没见过就触发 `_lazy_import()`；还没见过就用 `difflib` 给相近名字建议。
   - English: If the user passes a string, lookup checks `_BACKENDS`. Unknown names trigger `_lazy_import()`; still-unknown names raise an error with nearby suggestions from `difflib`.

3. **第 137-141 行 / Lines 137-141 (entry point load)**:
   - 中文: entry point 只在第一次被点名时 `load()`，再通过 `register_backend` 变成普通 compiler function。
   - English: An entry point is loaded only when first requested, then registered as an ordinary compiler function.

4. **第 146-160 行 / Lines 146-160 (`list_backends`)**:
   - 中文: 列表接口默认排除 `debug` 和 `experimental` 标签，避免普通用户看到内部或实验后端。
   - English: The listing API excludes `debug` and `experimental` tags by default, keeping internal or unstable backends out of the normal list.

## 类比 / The analogy

这像餐厅菜单。菜单上写着菜名，但厨房不会在开门时把每道菜都炒好。客人点了某道菜，后厨才拿出对应配方开始做；如果菜名拼错，服务员会猜两个最像的名字。

It is like a restaurant menu. The menu lists dishes, but the kitchen does not cook every dish at opening time. When a diner orders one, the kitchen loads that recipe. If the name is misspelled, the server suggests the closest matches.

## 自己跑一遍 / Try it yourself

```python
from functools import partial

BACKENDS, FNS, TAGS = {}, {}, {}
def register(fn=None, name=None, tags=()):
    if fn is None:
        return partial(register, name=name, tags=tags)
    name = name or fn.__name__
    if name in FNS:
        raise AssertionError(f"duplicate name: {name}")
    BACKENDS.setdefault(name, None)
    FNS[name] = fn
    TAGS[name] = tuple(tags)
    return fn

@register(name="toy", tags=("debug",))
def toy_backend(graph, inputs):
    return "compiled"

def list_backends(exclude=("debug",)):
    return sorted(k for k in BACKENDS if not (set(exclude) & set(TAGS[k])))

print(list_backends())
print(FNS["toy"](None, None))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[]
compiled
```

`toy` 被注册了，但默认列表因为 `debug` 标签把它藏起来。标签不是装饰，它会影响用户能看到的 backend 集合。

`toy` is registered, but the default listing hides it because of the `debug` tag. Tags are not decorative; they shape which backends users see.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **插件系统** / **Plugin systems**: 很多框架先登记 entry point 名字，真正使用时才 import 插件模块。 / Many frameworks record entry point names first and import plugin modules only when used.
- **模型 registry** / **Model registries**: Hugging Face `Auto*` 类也把字符串/config 映射到具体类，区别是它通常围绕模型类型而不是 compiler backend。 / Hugging Face `Auto*` classes also map strings or configs to concrete classes, but around model types rather than compiler backends.

## 注意事项 / Caveats / when it breaks

- **懒加载隐藏 import 错误** / **Lazy loading defers import errors**: 后端模块坏了，可能到第一次 lookup 才爆炸。测试要覆盖字符串 lookup，而不只是直接传 callable。 / If a backend module is broken, the failure may appear only at first lookup. Tests should cover string lookup, not only direct callables.
- **名字是全局资源** / **Names are global resources**: duplicate name 会被拒绝。第三方后端最好用清晰前缀，避免和内置名字撞车。 / Duplicate names are rejected. Third-party backends should use clear names to avoid collisions with built-ins.

## 延伸阅读 / Further reading

- [Source permalink](https://github.com/pytorch/pytorch/blob/d2f3a9d2e41e2fb201224ec61f74aa000a37e513/torch/_dynamo/backends/registry.py#L87-L188)
- [torch.compile documentation](https://pytorch.org/docs/stable/generated/torch.compile.html)

