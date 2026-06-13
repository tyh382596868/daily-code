---
date: 2026-06-13
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/nn/attention/_registry.py
permalink: https://github.com/pytorch/pytorch/blob/77e8ad08177a8af2cff1cd18ea8f996245e2ad33/torch/nn/attention/_registry.py#L16-L137
difficulty: intermediate
read_time: ~9 min
tags: [code-of-the-day, pytorch, flash-attention, dispatcher, plugin-system]
---

# PyTorch 终于把"自带 FA + FA3 + FA4 + 你家自定义 attention"做成了插件 / PyTorch ships a real plugin system for "built-in FA + FA3 + FA4 + your custom attention"

> **一句话 / In one line**: 五个小函数 + 一个 Protocol,PyTorch 给 SDPA dispatcher 安上了"插件总线",任何后端 (FA3 / FA4 / 第三方) 都可以在运行时用 `activate_flash_attention_impl("FA4")` 切上去,再用 `restore_flash_attention_impl()` 安全卸下。/ Five small functions and one Protocol turn PyTorch's SDPA dispatcher into a true plugin bus — any backend (FA3 / FA4 / third-party) can swap in at runtime via `activate_flash_attention_impl("FA4")` and safely unwind via `restore_flash_attention_impl()`.

## 为什么重要 / Why this matters

过去两年所有人都在抢 Flash-Attention v3、v4、ROCm Tri Dao 自家实现、Triton 版、CUTLASS 版……导致一个尴尬局面:**你的 PyTorch 一旦装上某个第三方 FA 包,所有 SDPA 调用都会被无声替换**,而且想换回原始 FA2 还得重启进程。这次 PyTorch 把这个混乱局面收束成一个干净的插件注册表 —— 后端只是登记一个 callable,用户显式 opt-in 才会激活,激活时返回一个 `FlashAttentionHandle`,卸载时 `handle.remove()`。这是教科书级别的"dispatcher 友好的插件设计"。

The past two years saw everyone scrambling to slot in Flash-Attention v3, v4, a ROCm port, a Triton variant, a CUTLASS variant… and the result was a mess: **install one third-party FA wheel and every SDPA call in your process silently rerouted**, with no clean way back to baseline FA2 short of restarting Python. This file ends the mess: backends merely *register* a callable, users explicitly *activate*, the activation returns a `FlashAttentionHandle`, and unwind calls `handle.remove()`. Textbook dispatcher-friendly plugin design.

## 代码 / The code

`pytorch/pytorch` — [`torch/nn/attention/_registry.py`](https://github.com/pytorch/pytorch/blob/77e8ad08177a8af2cff1cd18ea8f996245e2ad33/torch/nn/attention/_registry.py#L16-L137)

```python
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
    """Register the callable that activates a flash attention impl.

    Intended for SDPA backend providers. End users should call
    activate_flash_attention_impl() instead.
    """
    global _FLASH_ATTENTION_IMPLS
    _FLASH_ATTENTION_IMPLS[impl] = register_fn


def activate_flash_attention_impl(impl: str | _FlashAttentionImpl) -> None:
    """Activate a previously registered flash attention impl.

    Backend providers should NOT auto-activate on import. Users opt in
    explicitly so multiple provider libraries can coexist.
    """
    global _FLASH_ATTENTION_ACTIVE, _FLASH_ATTENTION_IMPLS

    restore_flash_attention_impl(_raise_warn=False)  # restore any previous override first

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
    """Return the currently activated flash attention impl name, if any."""
    return (
        _FLASH_ATTENTION_ACTIVE[0]
        if _FLASH_ATTENTION_ACTIVE is not None
        else _FLASH_ATTENTION_ACTIVE
    )


def restore_flash_attention_impl(_raise_warn: bool = True) -> None:
    """Restore the default FA2 implementation."""
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

1. **第 16-17 行 / Lines 16-17 (`FlashAttentionHandle` Protocol)**:
   - 中文: Python 3.8 引入的 `Protocol` 类型让 PyTorch 不需要后端继承任何具体基类,只要后端注册时返回的对象有一个 `.remove() -> None` 方法,就符合契约。这就是所谓"结构子类型" (structural subtyping) —— 第三方库可以零依赖地实现这个接口。
   - English: `Protocol` (Python 3.8+) lets PyTorch *not* demand backends inherit from a concrete base class — any object with `.remove() -> None` satisfies the contract. That's structural subtyping: third-party libs can implement the interface with zero PyTorch import surface.

2. **第 20-25 行 / Lines 20-25 (the two module globals)**:
   - 中文: `_FLASH_ATTENTION_IMPLS` 是一个 `{name → callable}` 字典,**装注册过的后端**;`_FLASH_ATTENTION_ACTIVE` 是一个 `(name, handle)` 元组,**装当前激活的那一个**。注意 active 是单数 —— 任何时刻最多只有一个自定义后端在跑。
   - English: `_FLASH_ATTENTION_IMPLS` is a `{name → callable}` dict that **tracks every registered backend**; `_FLASH_ATTENTION_ACTIVE` is a `(name, handle)` tuple that **tracks the single currently-active one**. Note "active" is singular — at most one custom backend at a time.

3. **第 28-58 行 / Lines 28-58 (`register_flash_attention_impl`)**:
   - 中文: 一个 `dict[impl] = register_fn` 的简单写法,但配了 keyword-only 的 `register_fn` 参数 —— 这种 API 设计让调用方必须写 `register_flash_attention_impl("FA4", register_fn=...)`,避免位置参数搞混。docstring 还明确警告"backend providers 不要在 import 时自动激活"。
   - English: It's just `dict[impl] = register_fn`, but with a keyword-only `register_fn` argument — the API forces callers to write `register_flash_attention_impl("FA4", register_fn=...)`, killing positional-argument bugs. The docstring also explicitly warns "backend providers must NOT auto-activate on import".

4. **第 61-99 行 / Lines 61-99 (`activate_flash_attention_impl`)**:
   - 中文: 三步动作 —— (1) 先 `restore_flash_attention_impl(_raise_warn=False)` 把之前可能还在跑的后端撤掉,这是**重入安全**的关键;(2) 找出对应的 `register_fn` 并调用,真正向 dispatcher 注册自己的 kernel;(3) 如果 register_fn 返回了一个 handle 就存起来,后面卸载时用。
   - English: Three moves — (1) `restore_flash_attention_impl(_raise_warn=False)` unwinds any backend already in place, which is **the key to re-entrant safety**; (2) look up the matching `register_fn` and invoke it to actually register kernels with the dispatcher; (3) if `register_fn` returned a handle, stash it for later removal.

5. **第 102-117 行 / Lines 102-117 (`list_*` and `current_*`)**:
   - 中文: 两个只读 getter,主要是给文档生成和调试用的。`list_flash_attention_impls` 还排了序 —— 这是个小细节但很关键:用户看到的可用后端列表总是稳定的。
   - English: Two read-only getters, mostly for docs and debugging. `list_flash_attention_impls` sorts the keys — small detail, big impact: users see a stable, deterministic list of available backends.

6. **第 120-137 行 / Lines 120-137 (`restore_flash_attention_impl`)**:
   - 中文: 调用 `handle.remove()` 把 dispatcher 里的覆盖撤掉,然后清空全局状态。**这才是真正区别"插件"和"猴子补丁"的地方** —— monkey-patch 没有卸载逻辑,但这里 dispatcher 注册可以被撤回,从而恢复到原始 FA2。`_raise_warn` 参数控制"撤回没有 active 后端时要不要 warning",方便给 `activate` 调用时静默使用。
   - English: Calls `handle.remove()` to undo the dispatcher override, then clears global state. **This is what separates a real "plugin" from a monkey-patch** — monkey-patches have no unwind path, but the dispatcher registration here can be cleanly removed, restoring the original FA2. The `_raise_warn` flag controls whether unwinding-with-nothing-active should warn, so `activate` can call it silently.

## 类比 / The analogy

中文: 把 PyTorch 的 SDPA dispatcher 想象成一台**家庭影院里的 HDMI 切换器**。FA2 是默认那条永远接好的电视输入。FA3、FA4、你自己写的 Triton attention 是各自一台外接设备 —— 它们**先在切换器上登记一下"我叫 FA4,你按这个键能找到我"** (`register_flash_attention_impl`),但**没人按键之前都不抢屏幕** (不 auto-activate)。你按下 FA4 那个键 (`activate_flash_attention_impl("FA4")`),切换器先把目前显示的源切掉 (`restore_*`),再把信号路由到 FA4。一切 SDPA 调用从此走 FA4 的 kernel。看完想回 FA2?再按 `restore_*`,切换器把"撤回票根" (`handle.remove()`) 撕掉,屏幕回到默认输入。

English: Picture PyTorch's SDPA dispatcher as the **HDMI switch on a home theatre**. FA2 is the default input that's always wired up. FA3, FA4, your handwritten Triton attention are external devices — they **first announce themselves on the switch** ("I'm FA4, press this button to find me", via `register_flash_attention_impl`), **but no one grabs the screen until a button is pressed** (no auto-activate). You press the FA4 button (`activate_flash_attention_impl("FA4")`); the switch unhooks whatever was on (`restore_*`), then routes the signal to FA4. All SDPA calls now go through FA4's kernel. Done watching? Hit `restore_*`, the switch tears the "rollback receipt" (`handle.remove()`) and the screen reverts to the default input.

## 自己跑一遍 / Try it yourself

```python
# fake_attention_backend.py — register and activate a stub backend
from torch.nn.attention._registry import (
    register_flash_attention_impl,
    activate_flash_attention_impl,
    current_flash_attention_impl,
    list_flash_attention_impls,
    restore_flash_attention_impl,
)

class StubHandle:                                       # satisfies FlashAttentionHandle Protocol
    removed = False
    def remove(self):
        self.removed = True
        print("[StubHandle] removed -> restored to FA2")

def register_fn():
    print("[register_fn] would push a custom kernel into the dispatcher")
    return StubHandle()

register_flash_attention_impl("MyAttn", register_fn=register_fn)
print("available:", list_flash_attention_impls())       # ['MyAttn']

activate_flash_attention_impl("MyAttn")
print("current:", current_flash_attention_impl())       # 'MyAttn'

restore_flash_attention_impl()
print("current:", current_flash_attention_impl())       # None
```

运行 / Run with:
```bash
pip install "torch>=2.5"
python fake_attention_backend.py
```

预期输出 / Expected output:
```
available: ['MyAttn']
[register_fn] would push a custom kernel into the dispatcher
current: MyAttn
[StubHandle] removed -> restored to FA2
current: None
```

中文: 注意 **stub handle 满足 Protocol 完全不需要 import torch 的任何东西** —— 这就是 structural typing 的好处。第三方包可以零依赖地实现 `FlashAttentionHandle`。

English: Note that the **stub handle satisfies the Protocol without importing anything from torch** — that's the structural-typing payoff. A third-party package can implement `FlashAttentionHandle` with zero torch import surface.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **`torch.compile` backend registry** / **`torch.compile` backend registry**: 同样的"register → activate"模式,`torch._dynamo.register_backend("inductor", fn)`,然后 `torch.compile(model, backend="inductor")` 激活。/ Same "register → activate" pattern. `torch._dynamo.register_backend("inductor", fn)`, then `torch.compile(model, backend="inductor")` flips it on.
- **HuggingFace `diffusers` scheduler registry** / **HuggingFace `diffusers` scheduler registry**: 用 `from_config("DPMSolverMultistepScheduler")` 来切换求解器,本质上是同一个模式。/ `from_config("DPMSolverMultistepScheduler")` swaps solvers — same idea.
- **JAX `pallas` kernel attestation** / **JAX `pallas` kernel attestation**: pallas 在 TPU 上也是允许用户登记自定义 kernel 并显式激活。/ Pallas on TPU lets users register custom kernels and explicitly activate them.

## 注意事项 / Caveats / when it breaks

- **单实例约束 / Single-active constraint**:
  - 中文: `_FLASH_ATTENTION_ACTIVE` 是一个标量元组,**任何时刻最多一个自定义后端激活**。所以"我想同一台机器上的不同进程用不同 FA 后端"是 OK 的 (每个进程独立),"同一进程同时跑 FA3 和 FA4 在不同 attention 层"目前**不支持**。
  - English: `_FLASH_ATTENTION_ACTIVE` is a scalar tuple — **at most one custom backend per process**. So "two processes on the same box use different FAs" is fine (each is its own globals), but "one process runs FA3 in some layers and FA4 in others" is **not** currently supported.
- **handle 必须 idempotent / `handle.remove()` must be idempotent**:
  - 中文: `activate` 一开始就先 `restore_*(_raise_warn=False)`,意味着 `handle.remove()` 在多次 activate 间会反复被调用 —— 后端实现这个方法时一定要确保**重复调用安全** (e.g. 用 flag 标记 removed)。
  - English: `activate` calls `restore_*(_raise_warn=False)` first, meaning `handle.remove()` may be invoked multiple times across consecutive activates — backends **must make it idempotent** (e.g. with a `removed` flag).
- **是 private API / This is private**:
  - 中文: 文件名 `_registry.py` 带下划线,模块路径下两个变量也是 `_FLASH_ATTENTION_*` 开头,说明现阶段是给"第三方 FA 后端作者"看的,不是给用户日常调用的。**API 名字可能在 PyTorch 2.x → 3.x 之间被改名**,别在生产代码里直接 import。
  - English: The leading underscore in `_registry.py` plus the `_FLASH_ATTENTION_*` globals say this is for "third-party FA backend authors" today, not for everyday user code. **The names may rename across PyTorch 2.x → 3.x**, so don't import this in production yet.

## 延伸阅读 / Further reading

- [PyTorch SDPA blog — "Accelerated Generation with Flash Attention"](https://pytorch.org/blog/accelerating-generative-ai/)
- [FlashAttention-3 paper — "Fast and accurate attention with asynchrony and low-precision"](https://arxiv.org/abs/2407.08608)
- [Python `Protocol` docs — structural subtyping](https://docs.python.org/3/library/typing.html#typing.Protocol)
- [PyTorch issue tracker — search "flash attention impl"](https://github.com/pytorch/pytorch/issues?q=flash+attention+impl)
