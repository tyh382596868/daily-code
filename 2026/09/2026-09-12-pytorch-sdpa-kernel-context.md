---
date: 2026-09-12
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/nn/attention/__init__.py
permalink: https://github.com/pytorch/pytorch/blob/86195a6314dcf93964663b4df36b081c837e9845/torch/nn/attention/__init__.py#L106-L174
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, scaled-dot-product-attention, context-manager, backend-dispatch]
---

# PyTorch SDPA context：临时选择 attention kernel / PyTorch SDPA Context: Select an Attention Kernel Temporarily

> **一句话 / In one line**: `sdpa_kernel` 把全局 backend flags 包进一个可恢复的 context manager。 / `sdpa_kernel` wraps global backend flags in a restorable context manager.

## 为什么重要 / Why this matters

中文：`scaled_dot_product_attention` 可能落到 math、memory-efficient、FlashAttention、cuDNN 或外部注册实现。模型代码通常只想调用一个 attention API，不想在每个 forward 里手动开关一堆全局状态。PyTorch 用 context manager 保存旧状态、设置临时状态、退出时恢复，让局部实验不会污染后续计算。

English: `scaled_dot_product_attention` may use math, memory-efficient, FlashAttention, cuDNN, or a registered external implementation. Model code should call one attention API instead of manually toggling global flags in every forward. PyTorch uses a context manager to snapshot state, install a temporary policy, and restore it on exit.

## 代码 / The code

`pytorch/pytorch` — [`torch/nn/attention/__init__.py`](https://github.com/pytorch/pytorch/blob/86195a6314dcf93964663b4df36b081c837e9845/torch/nn/attention/__init__.py#L106-L174)

```python
def _sdpa_kernel(backends: Iterable, set_priority: bool = False) -> None:
    for name, val in _backend_names.items():
        enabled = getattr(SDPBackend, val) in backends
        getattr(torch._C, f"_set_sdp_use_{name}")(enabled)
    if set_priority:
        # backends should be a unique list
        user_priority = [int(backend) for backend in backends]
        previous_priority = torch._C._get_sdp_priority_order()
        for backend in previous_priority:
            if backend not in user_priority:
                user_priority.append(int(backend))
        torch._C._set_sdp_priority_order(user_priority)


@contextlib.contextmanager
def sdpa_kernel(backends: list[SDPBackend] | SDPBackend, set_priority: bool = False):
    r"""
    Context manager to select which backend to use for scaled dot product attention.

    .. warning:: This function is beta and subject to change.

    Args:
        backends (Union[List[SDPBackend], SDPBackend]): A backend or list of backends for scaled dot product attention.
        set_priority (bool=False): Whether the ordering of the backends is interpreted as their priority order.
    """
    if not isinstance(backends, (list, SDPBackend)):
        raise AssertionError(
            f"Backend must be an instance of SDPBackend or a list of SDPBackend instances, got {type(backends).__name__}"
        )

    if isinstance(backends, SDPBackend):
        backends = [backends]
    backends = list(dict.fromkeys(backends))

    previous_backends = _cur_sdpa_kernel_backends(with_priority=set_priority)
    priority_token = _sdpa_kernel_uses_priority.set(
        set_priority or _sdpa_kernel_uses_priority.get()
    )
    try:
        _sdpa_kernel(backends, set_priority)
        yield {}
    finally:
        _sdpa_kernel_uses_priority.reset(priority_token)
        _sdpa_kernel(previous_backends, set_priority)
```

## 逐行讲解 / What's happening

1. **第 106-109 行 / Lines 106-109 (`_sdpa_kernel`)**:
   - 中文：backend 名字来自 `_backend_names`，每个名字都变成一个 C++ flag 的 setter；当前 backend 不在列表里就被关闭。
   - English: `_backend_names` turns backend names into C++ flag setters. A backend not in the selected list is disabled.
2. **第 110-117 行 / Lines 110-117 (`set_priority`)**:
   - 中文：启用列表不仅可以表达“哪些可用”，还可以表达“优先级顺序”。用户指定的顺序先放进去，旧列表里剩下的 backend 再补到后面。
   - English: The list can express both availability and priority. User-selected backends come first, and remaining old backends are appended.
3. **第 155-158 行 / Lines 155-158 (input validation)**:
   - 中文：接受单个 enum 或 list，但拒绝任意对象，避免把字符串或错误类型一路传进底层 flag。
   - English: A single enum or a list is accepted, while arbitrary objects are rejected before touching low-level flags.
4. **第 160-163 行 / Lines 160-163 (normalization)**:
   - 中文：单个 backend 被包装成 list，`dict.fromkeys` 去掉重复项，同时保持用户给出的顺序。
   - English: A single backend becomes a list, and `dict.fromkeys` removes duplicates without losing order.
5. **第 165-169 行 / Lines 165-169 (snapshot and context state)**:
   - 中文：先保存旧 backend，再用 `ContextVar` 记录嵌套 context 是否已经进入 priority 模式；这比普通全局布尔值更适合嵌套调用。
   - English: The old backend set is captured first, and a `ContextVar` tracks whether an enclosing context already uses priority mode, which behaves correctly under nesting.
6. **`try/finally` / `try/finally`**:
   - 中文：即使 attention kernel 抛异常，`finally` 也会恢复旧状态。这是 context manager 最重要的工程价值。
   - English: Even if the attention call raises, `finally` restores the old state. That is the most important production property of the context manager.

## 类比 / The analogy

中文：像实验室的电源总闸。你可以在一个实验房间里暂时只开 FlashAttention，离开房间时总闸自动回到原来的配置，不会让隔壁实验也突然失去 math backend。

English: It is like a lab power panel. You can enable only FlashAttention inside one experiment room, and the panel returns to its previous configuration when you leave so the next room is unaffected.

## 自己跑一遍 / Try it yourself

```python
from contextlib import contextmanager

state = {"math": True, "flash": True}

@contextmanager
def only(*names):
    old = state.copy()
    state.update({name: name in names for name in state})
    try:
        yield
    finally:
        state.clear()
        state.update(old)

print(state)
with only("flash"):
    print(state)
print(state)
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
{'math': True, 'flash': True}
{'math': False, 'flash': True}
{'math': True, 'flash': True}
```

中文：重点不是“关掉某个开关”，而是把临时策略的生命周期绑定到代码块。

English: The important idea is not toggling a flag; it is binding a temporary policy to the lifetime of a code block.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **PyTorch autocast** / **PyTorch autocast**: 进入上下文时改 dtype policy，退出时恢复。 / Temporarily changes dtype policy and restores it on exit.
- **FlashAttention backend registries** / **FlashAttention backend registries**: 把外部 kernel 注册到统一 dispatcher。 / Register external kernels behind a unified dispatcher.
- **OpenWAM attention selection** / **OpenWAM attention selection**: 在进程启动时选择 backend，再让模型使用统一 callable。 / Select a backend at startup and expose one callable to the model.

## 注意事项 / Caveats / when it breaks

- **backend 可用不代表输入一定支持** / **Availability does not mean input compatibility**: dtype、device、mask 和 head dimension 仍可能让 kernel 回退。
- **这是进程状态** / **This is process state**: 不要跨线程随意共享未封装的 flag；优先使用官方 context。
- **priority 会改变选择顺序** / **Priority changes selection order**: 性能 benchmark 要固定 `set_priority` 和硬件环境。

## 延伸阅读 / Further reading

- [PyTorch `sdpa_kernel`](https://github.com/pytorch/pytorch/blob/86195a6314dcf93964663b4df36b081c837e9845/torch/nn/attention/__init__.py)
- [PyTorch scaled dot product attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
