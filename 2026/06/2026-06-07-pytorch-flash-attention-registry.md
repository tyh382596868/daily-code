---
date: 2026-06-07
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/nn/attention/_registry.py
permalink: https://github.com/pytorch/pytorch/blob/56964c25c21235cf3a06679d2e400195087f64fb/torch/nn/attention/_registry.py#L1-L137
difficulty: beginner
read_time: ~8 min
tags: [code-of-the-day, pytorch, registry, plugin-system, flash-attention, sdpa]
---

# PyTorch 怎么让 FA3、FA4 这种外部后端"插进"SDPA 调度器:一个 137 行的注册表 / How PyTorch lets external backends (FA3, FA4) plug into the SDPA dispatcher: a 137-line registry

> **一句话 / In one line**: 全部机制就是两个模块级全局变量 + 四个小函数:provider 调 `register_flash_attention_impl(name, register_fn=...)` 把 callable 放进 dict,用户调 `activate_flash_attention_impl(name)` 时才真正去注册自定义 kernel,返回的 handle 由模块全局保活直到进程结束. / The whole mechanism is two module-level globals plus four small functions: providers call `register_flash_attention_impl(name, register_fn=...)` to stash a callable in a dict, users explicitly `activate_flash_attention_impl(name)` to actually register the custom kernel into the dispatcher, and the returned handle stays alive in module-level state for the process lifetime.

## 为什么重要 / Why this matters

PyTorch 2.11 把 `F.scaled_dot_product_attention` 的后端做成了"开放生态":FA3、FA4 现在不是内置的,它们是**独立的 PyPI 包**,但又要能让用户一行代码切换到它们提供的高速 kernel. 这件事难就难在——provider 不能在 `import` 时偷偷生效(那样多个包共存会冲突),又必须能保活注册状态(handle 被 GC 掉的瞬间,dispatcher 就还原了). 这 137 行就是教科书级的解决方案:provider 注册 *callable*,user 显式 *activate*,返回 *handle* 由模块全局保活. 这种"注册 vs 激活"两步走的设计——以及 `restore_flash_attention_impl()` 自带的退回路径——是任何想做插件系统都该抄的模板.

PyTorch 2.11 opens up `F.scaled_dot_product_attention` as a plug-in surface: FA3 and FA4 are now **separately distributed PyPI packages** that can slot their kernels into the dispatcher — but only when the user opts in. The hard part is that providers must NOT activate on import (otherwise two installed providers would fight), yet the activation must persist for the rest of the process. The solution in these 137 lines is textbook: providers register a *callable* under a name, users explicitly *activate* a name, and the returned handle is kept alive in module-level state. The two-step "register vs activate" split — plus a built-in `restore_*` escape hatch — is a template worth copying for any extensible PyTorch surface.

## 代码 / The code

`pytorch/pytorch` — [`torch/nn/attention/_registry.py#L1-L137`](https://github.com/pytorch/pytorch/blob/56964c25c21235cf3a06679d2e400195087f64fb/torch/nn/attention/_registry.py#L1-L137)

```python
# mypy: allow-untyped-defs
"""Registry for flash attention implementations.

This module contains the registration system for flash attention implementations.
It has no torch dependencies to avoid circular imports during initialization.
"""

import logging
from collections.abc import Callable
from typing import Literal, Protocol


logger = logging.getLogger(__name__)


class FlashAttentionHandle(Protocol):
    def remove(self) -> None: ...


_RegisterFn = Callable[..., FlashAttentionHandle | None]
_FlashAttentionImpl = Literal["FA3", "FA4"]

_FLASH_ATTENTION_IMPLS: dict[str, _RegisterFn] = {}

_FLASH_ATTENTION_ACTIVE: tuple[str, FlashAttentionHandle] | None = None


def register_flash_attention_impl(
    impl: str | _FlashAttentionImpl,
    *,
    register_fn: _RegisterFn,
) -> None:
    """
    Register the callable that activates a flash attention impl.

    Args:
        impl: Implementation identifier (e.g., ``"FA4"``).
        register_fn: Callable that performs the actual dispatcher registration.
            This function will be invoked by :func:`activate_flash_attention_impl`
            and should register custom kernels with the PyTorch dispatcher.
            It may optionally return a handle implementing
            :class:`FlashAttentionHandle` to keep any necessary state alive.
    """
    global _FLASH_ATTENTION_IMPLS
    _FLASH_ATTENTION_IMPLS[impl] = register_fn


def activate_flash_attention_impl(
    impl: str | _FlashAttentionImpl,
) -> None:
    """
    Activate into the dispatcher a previously registered flash attention impl.

    .. note::
        Backend providers should NOT automatically activate their implementation
        on import. Users should explicitly opt-in by calling this function or via
        environment variables to ensure multiple provider libraries can coexist.
    """
    global _FLASH_ATTENTION_ACTIVE, _FLASH_ATTENTION_IMPLS

    restore_flash_attention_impl(
        _raise_warn=False
    )  # first restore any prev overrides (if any) to default

    register_fn = _FLASH_ATTENTION_IMPLS.get(impl)
    if register_fn is None:
        raise ValueError(
            f"Unknown flash attention impl '{impl}'. "
            f"Available implementations: {list_flash_attention_impls()}"
        )

    handle = register_fn()
    if handle is not None:
        _FLASH_ATTENTION_ACTIVE = (impl, handle)


def list_flash_attention_impls() -> list[str]:
    """Return the names of all available flash attention implementations."""
    return sorted(_FLASH_ATTENTION_IMPLS.keys())


def current_flash_attention_impl() -> str | None:
    """
    Return the currently activated flash attention impl name, if any.

    ``None`` indicates that no custom impl has been activated.
    """
    return (
        _FLASH_ATTENTION_ACTIVE[0]
        if _FLASH_ATTENTION_ACTIVE is not None
        else _FLASH_ATTENTION_ACTIVE
    )


def restore_flash_attention_impl(_raise_warn: bool = True) -> None:
    """
    Restore the default FA2 implementation
    """
    global _FLASH_ATTENTION_ACTIVE

    handle = None
    if _FLASH_ATTENTION_ACTIVE is not None:
        handle = _FLASH_ATTENTION_ACTIVE[1]

    if handle is not None:
        handle.remove()
    elif _raise_warn:
        logger.warning(
            "Trying to restore default FA2 impl when no custom impl was activated"
        )

    _FLASH_ATTENTION_ACTIVE = None  # default
```

## 逐行讲解 / What's happening

1. **`FlashAttentionHandle(Protocol)` 用结构子类型化定义"句柄是什么" / `FlashAttentionHandle(Protocol)` defines what a handle is by structural typing**:
   - 中文: 只要你的对象有 `.remove()` 方法,就符合协议——provider 不需要 import PyTorch 的某个具体类,只用 duck typing 就行. 这是为什么文件顶上注释强调"no torch dependencies"——这个文件可以在 `import torch` 还没跑完的时候被加载.
   - English: any object with a `.remove()` method satisfies the protocol — providers don't have to import a concrete PyTorch class, structural typing is enough. That's why the docstring stresses "no torch dependencies": this file is safe to load before `import torch` even finishes resolving.

2. **两个模块级全局 / The two module-level globals**:
   - 中文: `_FLASH_ATTENTION_IMPLS: dict[str, callable]` 存"还没激活的注册函数",`_FLASH_ATTENTION_ACTIVE: tuple[str, handle] | None` 存"当前激活的那一个的句柄". 一个是"目录",另一个是"当前已生效的项". 拆开两个变量后,`register` 和 `activate` 就完全解耦了——前者无副作用,后者才真正动 dispatcher.
   - English: `_FLASH_ATTENTION_IMPLS: dict[str, callable]` is the catalogue of "registered but not yet activated" callables; `_FLASH_ATTENTION_ACTIVE: tuple[str, handle] | None` holds the currently-active one's handle. Two variables, two responsibilities — `register` has no side effects on the dispatcher; only `activate` does.

3. **`register_flash_attention_impl(impl, *, register_fn)` 是纯字典 setattr / `register_flash_attention_impl` is just a dict write**:
   - 中文: 完全没有 torch、没有 dispatcher、没有副作用. provider 包导入时调一句就行——这一步不会动到任何 PyTorch 运行时状态.
   - English: no torch, no dispatcher, no side effect. The provider package can call this at import time and it changes nothing about the PyTorch runtime.

4. **`activate_flash_attention_impl` 先 `restore_*(_raise_warn=False)` / `activate` calls `restore_*(_raise_warn=False)` first**:
   - 中文: 切到新后端前,先把上一个 handle 的 `.remove()` 调掉——避免"FA3 dispatcher 注册没撤、FA4 又叠了一层"这种烂状态. `_raise_warn=False` 是因为切换路径不该打 warning(只有用户显式 restore 时才警告).
   - English: before switching, tear down the previous handle via `.remove()` so dispatcher state doesn't accumulate (FA3 hooks still live while FA4 piles on top). `_raise_warn=False` suppresses the "you restored when nothing was active" warning that's only useful on the user-facing `restore` path.

5. **`handle = register_fn()` 真正生效在这里 / `handle = register_fn()` is where activation actually happens**:
   - 中文: provider 提供的 `register_fn` 才知道怎么用 `torch.library.impl` / `c10d` / `torch.dispatcher` 把 kernel 挂进去. 这个 callable 可能返回一个 `handle`(里面持有 dispatcher 句柄、CUDA stream 等需要保活的东西),也可能返回 `None`(若 provider 不需要清理状态).
   - English: the provider-supplied `register_fn` is the one that knows how to splice kernels into `torch.library.impl` / `c10d` / `torch.dispatcher`. It may return a `handle` that holds dispatcher hooks, CUDA streams, or anything else that must outlive this call — or it returns `None` if no cleanup is needed.

6. **`_FLASH_ATTENTION_ACTIVE = (impl, handle)` 保活 / Pinning to keep alive**:
   - 中文: 这一句是整个机制最关键的一行——如果不把 `handle` 存到模块级全局,Python GC 会立刻回收它,handle 析构时 dispatcher 注册可能跟着撤. 模块级 dict / tuple 是 Python 里最稳的"进程级生命周期".
   - English: this single line is the linchpin — if you don't pin `handle` into module-level state, Python GC reclaims it the moment `activate_*` returns, and the dispatcher registration might be torn down with it. A module-level dict/tuple is the canonical "process-lifetime" anchor in Python.

7. **`restore_flash_attention_impl` 通过 `handle.remove()` 退回 / `restore_*` tears down via `handle.remove()`**:
   - 中文: 显式退回到 FA2 默认实现. handle 是 protocol,只需要 `.remove()` 就够——具体怎么从 dispatcher 撤是 provider 的实现细节,这个文件完全不关心.
   - English: explicit return to the default FA2. The handle is a protocol — only `.remove()` is required — and *how* it unregisters from the dispatcher is the provider's secret. This file deliberately knows nothing about it.

## 类比 / The analogy

想象一栋公寓楼大门口的"对讲机面板"——它就是 `_FLASH_ATTENTION_IMPLS`. 物业先在面板上把每户的门铃按钮安上去(`register`),但安按钮的时候**不会去敲谁家的门**. 你按下某个按钮 (`activate`),门铃才真正响,而且把上一次响着的铃先停掉(`restore` 先调)——避免两个门铃同时响. 物业还给你一根备用电话线(`handle`),挂在墙上的钩子上(`_FLASH_ATTENTION_ACTIVE`)——只要钩子上有这根线,门铃就一直能响;什么时候你想恢复原样,把这根线从钩子上摘下来 `.remove()`,门铃就静音了.

Think of the apartment-block intercom panel out front — that's `_FLASH_ATTENTION_IMPLS`. The super installs a doorbell button for every unit (`register`), but installing the button **doesn't ring anyone's door**. When you press a button (`activate`), the bell actually rings — and first it silences whichever bell was previously ringing (`restore` is called first), so two bells never sound at once. The super also gives you a spare phone wire (`handle`) clipped to a hook on the wall (`_FLASH_ATTENTION_ACTIVE`) — as long as the wire stays on the hook, the bell keeps working; the moment you `.remove()` it from the hook, the bell goes quiet.

## 自己跑一遍 / Try it yourself

```python
# pip install nothing — this is pure stdlib
from typing import Protocol

class Handle(Protocol):
    def remove(self) -> None: ...

_IMPLS = {}
_ACTIVE = None

def register(name, *, register_fn):
    _IMPLS[name] = register_fn

def activate(name):
    global _ACTIVE
    if _ACTIVE is not None:
        _ACTIVE[1].remove()      # tear down previous
        _ACTIVE = None
    handle = _IMPLS[name]()
    if handle is not None:
        _ACTIVE = (name, handle)

# Provider A registers, but does not activate.
class HandleA:
    def remove(self): print("FA-A torn down")
def reg_a():
    print("FA-A kernels patched into dispatcher")
    return HandleA()
register("FA-A", register_fn=reg_a)

print("active:", _ACTIVE)        # -> None
activate("FA-A")
print("active:", _ACTIVE[0])     # -> 'FA-A'
activate("FA-A")                 # switching to same: prev torn down, then re-registered
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
active: None
FA-A kernels patched into dispatcher
active: FA-A
FA-A torn down
FA-A kernels patched into dispatcher
```

中文: 注意 `register` 本身没有任何输出——它真的就是 dict 写一下. 只有 `activate` 才执行 `reg_a()` 并 `print` 出来. 这就是"导入即安静、激活才生效"的设计.

Note the silence after `register` — it's literally a dict write. Only `activate` invokes `reg_a()` and prints. This *is* the "import is silent, activation is loud" design.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **`torch.library.impl` 自身**: 中文: 同样是 register vs override 的两段式,但暴露面更底层. 这个 `_registry.py` 是包装在它之上的高阶模式. / English: same register-vs-override two-step at a lower level. This `_registry.py` is the high-level wrapper around `torch.library`.
- **HuggingFace `AutoModel.register`**: 中文: register 把 `(config_class, model_class)` 塞进字典,实际类只在 `from_pretrained` 拿到 config 后才被实例化. 同样是"目录 vs 当前激活"的分离. / English: register stuffs `(config_class, model_class)` into a dict; the actual class is only instantiated when `from_pretrained` matches a config. Catalogue vs current-instance separation, same shape.
- **PyTorch `torch.compile` 的后端注册**: 中文: `register_backend("my_backend", lambda gm, ex: ...)` 也是 lazy 注册,直到用户在 `torch.compile(model, backend="my_backend")` 才真正调用 callable. / English: `register_backend("my_backend", lambda gm, ex: ...)` is also lazy; the callable only runs when the user passes `backend="my_backend"` to `torch.compile`.
- **Python `entry_points`**: 中文: 同样的"注册后台静默、激活时才 import"思想,但生效域是整个 pip 生态,跨进程也能用. / English: same "register quietly, dispatch loudly" idea, but at the package level across pip-installed processes.

## 注意事项 / Caveats / when it breaks

- **`register_fn` 必须返回 handle 才能保活**: 中文: 如果 provider 写的 `register_fn` 返回 `None`,但内部用 weakref 注册 kernel,那一回 `activate_*` 函数返回,GC 立马回收,dispatcher 状态丢失. provider 务必返回一个真实的对象.
- **`register_fn` must return a handle to persist**: English: if a provider returns `None` but registered the kernel via a weakref internally, GC will reclaim it the instant `activate_*` returns and the dispatcher state is gone. Providers must return a real, retained object.
- **多个 provider 同时 register 是允许的,同时 activate 不允许**: 中文: 字典里可以有 FA3 和 FA4 两个 entry,但 `_FLASH_ATTENTION_ACTIVE` 是单值. 这设计上就是"用户负责选哪个". 如果你的应用需要按 op 选不同 backend,要在更高一层做路由,不能指望这个 registry.
- **Multiple `register`s are fine, only one can be `activate`d**: English: the dict tolerates both FA3 and FA4 entries, but `_FLASH_ATTENTION_ACTIVE` holds a single tuple. By design, the user picks one. If you need per-op backend routing, layer that above this registry.
- **没有线程锁**: 中文: 这个 registry 本身不是线程安全的——多线程同时 `activate` 不同后端会乱. PyTorch dispatcher 也是全局状态,所以这个限制是天然的,不是设计缺陷.
- **No thread lock**: English: the registry itself is not thread-safe — concurrent `activate` calls from different threads will race. The PyTorch dispatcher is global state anyway, so this is fundamental, not a flaw.

## 延伸阅读 / Further reading

- [PyTorch 2.11 release notes: SDPA backend registry](https://github.com/pytorch/pytorch/releases) — the user-facing API
- [`torch/nn/attention/_fa3.py`](https://github.com/pytorch/pytorch/blob/main/torch/nn/attention/_fa3.py) — FA3 provider implementation
- [`torch/nn/attention/_fa4.py`](https://github.com/pytorch/pytorch/blob/main/torch/nn/attention/_fa4.py) — FA4 provider implementation
- [PEP 544 — Protocols (structural subtyping)](https://peps.python.org/pep-0544/) — why `Protocol` lets handles stay duck-typed
