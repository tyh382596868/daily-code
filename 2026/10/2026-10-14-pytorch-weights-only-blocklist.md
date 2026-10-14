---
date: 2026-10-14
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/_weights_only_unpickler.py
permalink: https://github.com/pytorch/pytorch/blob/8ed12848bd2b7baba59d5dd8056d861173cd7702/torch/_weights_only_unpickler.py#L78-L158
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, pytorch, serialization, allowlist]
---

# PyTorch weights-only：安全 allowlist 也要有 blocklist / PyTorch Weights-Only: Even Safe Allowlists Need a Blocklist

> **一句话 / In one line**: `weights_only` unpickler 允许用户加安全类型，但永远拒绝来自危险模块的对象。 / The `weights_only` unpickler lets users add safe globals, but permanently rejects objects from dangerous modules.

## 为什么重要 / Why this matters

模型 checkpoint 本质上可能是 pickle；安全加载不能只问“用户 allowlist 了什么”，还要问“有没有一些模块即使被 allowlist 也不该加载”。

A model checkpoint may be a pickle stream. Safe loading cannot only ask what the user allowlisted; it also needs modules that remain forbidden even if someone tries to allowlist them.

## 代码 / The code

`pytorch/pytorch` - [`torch/_weights_only_unpickler.py`](https://github.com/pytorch/pytorch/blob/8ed12848bd2b7baba59d5dd8056d861173cd7702/torch/_weights_only_unpickler.py#L78-L158)

```python
# modules in this list are never allowed, even if the user attempts to allowlist
# functions/classes from them
_blocklisted_modules = [
    "sys",
    "os",
    "posix",
    "nt",
    # traceback.FrameSummary and linecache read arbitrary files
    "traceback",
    "linecache",
]

_marked_safe_globals_set: set[Callable | tuple[Callable, str]] = set()


def _add_safe_globals(safe_globals: list[Callable | tuple[Callable, str]]):
    global _marked_safe_globals_set
    _marked_safe_globals_set = _marked_safe_globals_set.union(set(safe_globals))


def _get_safe_globals() -> list[Callable | tuple[Callable, str]]:
    global _marked_safe_globals_set
    return list(_marked_safe_globals_set)


def _clear_safe_globals():
    global _marked_safe_globals_set
    _marked_safe_globals_set = set()


def _remove_safe_globals(
    globals_to_remove: list[Callable | tuple[Callable, str]],
):
    global _marked_safe_globals_set
    _marked_safe_globals_set = _marked_safe_globals_set - set(globals_to_remove)


class _safe_globals:
    def __init__(self, safe_globals: list[Callable | tuple[Callable, str]]):
        self.safe_globals = safe_globals

    def __enter__(self):
        _add_safe_globals(self.safe_globals)

    def __exit__(self, type, value, tb):
        _remove_safe_globals(self.safe_globals)


def _get_user_allowed_globals():
    rc: dict[str, Any] = {}
    for f in _marked_safe_globals_set:
        if isinstance(f, tuple):
            if len(f) != 2:
                raise ValueError(
                    f"Expected tuple of length 2 (global, str of callable full path), but got tuple of length: {len(f)}"
                )
            if type(f[1]) is not str:
                raise TypeError(
                    f"Expected second item in tuple to be str of callable full path, but got: {type(f[1])}"
                )
            f, name = f
        else:
            name = f"{f.__module__}.{f.__qualname__}"
        # Blocklisted modules can't be allowlisted, even under an alias.
        if getattr(f, "__module__", None) in _blocklisted_modules:
            continue
        rc[name] = f
    return rc
```

## 逐行讲解 / What's happening

1. **第 78-88 行 / Lines 78-88 (`_blocklisted_modules`)**:
   - 中文: `sys`、`os`、`posix`、`nt` 这类系统模块直接封死，`traceback` 和 `linecache` 也因为能读任意文件被加入。
   - English: System modules such as `sys`, `os`, `posix`, and `nt` are permanently blocked; `traceback` and `linecache` are included because they can read arbitrary files.
2. **第 90-123 行 / Lines 90-123 (safe globals state)**:
   - 中文: 用户 allowlist 存在一个集合里，并提供增删、清空和上下文管理器。
   - English: User allowlists live in a set with add, remove, clear, and context-manager helpers.
3. **第 139-158 行 / Lines 139-158 (`_get_user_allowed_globals`)**:
   - 中文: 入口允许 `(callable, name)` 别名，但最后仍按对象真实 `__module__` 做 blocklist 检查。
   - English: The API accepts `(callable, name)` aliases, but the final blocklist check still uses the object's real `__module__`.

## 类比 / The analogy

这像公司访客系统：员工可以预登记客人，但“禁止携带危险品”这条规则不会因为有人填了名字就失效。

It is like an office visitor system: employees can pre-register guests, but the no-dangerous-items rule does not disappear because a name appears on the list.

## 自己跑一遍 / Try it yourself

```python
blocked = {"os", "posix", "linecache"}
safe = set()

def add(x): safe.add(x)
def allowed():
    out = {}
    for fn in safe:
        if fn.__module__ in blocked:
            continue
        out[f"{fn.__module__}.{fn.__name__}"] = fn
    return out

import os
add(os.getcwd)
add(len)
print(sorted(allowed()))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
['builtins.len']
```

中文: allowlist 不是绝对通行证；blocklist 是最后一道不可绕过的门。

English: An allowlist is not an unconditional pass; the blocklist is the non-bypassable final gate.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Web upload filters** / **Web upload filters**: MIME allowlist 之外还会禁止脚本扩展名。 / MIME allowlists are often paired with script-extension blocklists.
- **Plugin systems** / **Plugin systems**: 插件可扩展，但核心危险 API 通常不可暴露。 / Plugins may extend behavior, while dangerous core APIs remain hidden.

## 注意事项 / Caveats / when it breaks

- **别名不能成为后门** / **Aliases must not become backdoors**: 如果只检查传入 name，危险对象可以伪装成安全路径。 / If only the provided name is checked, a dangerous object can masquerade as a safe path.
- **全局状态要小心测试隔离** / **Global state needs test isolation**: safe globals 是进程级集合，测试和库代码要及时清理。 / Safe globals are process-level state, so tests and libraries must clean them up.

## 延伸阅读 / Further reading

- [PyTorch weights-only unpickler](https://github.com/pytorch/pytorch/blob/8ed12848bd2b7baba59d5dd8056d861173cd7702/torch/_weights_only_unpickler.py#L78-L158)
