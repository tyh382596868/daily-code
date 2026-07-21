---
date: 2026-07-16
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/utils/__init__.py
permalink: https://github.com/pytorch/pytorch/blob/6d0950b4277092132463d743df98cd03009986b2/torch/utils/__init__.py#L35-L62
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, pytorch, weakref, dynamo, tensor]
---

# PyTorch swap_tensors：只放行内部 TensorWeakRef / PyTorch swap_tensors: Allow Only Internal TensorWeakRef

> **一句话 / In one line**: PyTorch 让 `swap_tensors` 接受 Dynamo/Inductor 自己创建的安全 tensor weakref，同时继续拒绝用户 weakref。 / PyTorch lets `swap_tensors` tolerate Dynamo/Inductor's own safe tensor weakrefs while still rejecting user weakrefs.

## 为什么重要 / Why this matters

`swap_tensors` 会保留 Python tensor 对象身份，但交换它背后的 tensor 内容。普通 weakref 可能观察到“同一个对象突然换芯”的状态，所以以前直接拒绝所有 weakref。问题是 Dynamo guard 也会持有内部 weakref；如果把它们也拒绝，编译后的 forward 还活着时做参数转换会误伤。这个改动把 weakref 分成安全内部引用和未知外部引用。

`swap_tensors` preserves the Python tensor object identity while swapping the tensor payload underneath. A normal weakref could observe that identity/content mismatch, so PyTorch used to reject all weakrefs. But Dynamo guards also hold internal weakrefs. Rejecting those blocks legitimate compiled-forward/module-conversion paths. This patch splits weakrefs into safe internal ones and unknown external ones.

## 代码 / The code

`pytorch/pytorch` — [`torch/utils/__init__.py`](https://github.com/pytorch/pytorch/blob/6d0950b4277092132463d743df98cd03009986b2/torch/utils/__init__.py#L35-L62), [`torch/utils/weak.py`](https://github.com/pytorch/pytorch/blob/6d0950b4277092132463d743df98cd03009986b2/torch/utils/weak.py#L348-L370)

```python
def swap_tensors(t1, t2):
    # Ensure there are no weakrefs that could observe swapped tensor contents.
    # Import lazily because torch.utils is imported while torch is still initializing.
    from torch.utils.weak import _TensorWeakRef

    def has_unsafe_weakrefs(t):
        # TensorWeakRef owns _TensorWeakRef, the internal tensor-only weakref
        # used by Dynamo and Inductor. It has no user callback and fixes the
        # tensor weakref before returning it. Keep other weakref types,
        # including WeakIdRef-backed caches, on the conservative path.
        return any(not isinstance(wr, _TensorWeakRef) for wr in weakref.getweakrefs(t))

    if has_unsafe_weakrefs(t1):
        raise RuntimeError("Cannot swap t1 because it has weakref associated with it")
    if has_unsafe_weakrefs(t2):
        raise RuntimeError("Cannot swap t2 because it has weakref associated with it")
```

```python
class _TensorWeakRef(weakref.ref):
    """Tensor-only weakref that fixes Tensor weakrefs before returning them."""

    __slots__ = ()

    def __call__(self):
        out = super().__call__()
        if out is None:
            return out
        if not isinstance(out, Tensor):
            raise AssertionError(f"expected torch.Tensor, got {type(out)}.")
        out._fix_weakref()
        return out


class TensorWeakRef:
    def __init__(self, tensor: Tensor) -> None:
        self.ref = _TensorWeakRef(tensor)
```

## 逐行讲解 / What's happening

1. **lazy import**: 中文: `torch.utils` 初始化很早，所以 `_TensorWeakRef` 延迟导入，避免 import 环。 English: `_TensorWeakRef` is imported lazily because `torch.utils` initializes early.
2. **只看真实 weakref 对象 / Inspect real weakref objects**: 中文: `weakref.getweakrefs(t)` 返回对象当前所有 weakref。 English: `weakref.getweakrefs(t)` lists every weakref currently attached to the tensor.
3. **内部引用白名单 / Internal refs are whitelisted**: 中文: 只有 `_TensorWeakRef` 被认为安全。 English: only `_TensorWeakRef` is considered safe.
4. **未知引用保守拒绝 / Unknown refs stay blocked**: 中文: 用户 callback、普通 `weakref.ref`、`WeakIdRef` 都会阻止 swap。 English: user callbacks, plain `weakref.ref`, and `WeakIdRef` still block swapping.
5. **`_fix_weakref()` 是关键 / `_fix_weakref()` is the key**: 中文: tensor weakref 被取回时先修复内部状态。 English: tensor weakrefs repair tensor-side state before returning the object.

## 类比 / The analogy

这像给仓库换货架标签。内部盘点员知道“货架号没变但货物换了”，可以继续工作；外部客户拿着旧标签来找货，就可能看到错东西，所以要挡住。

It is like swapping warehouse contents while keeping shelf labels. Internal inventory systems know the shelf stayed but the contents changed; outside customers with stale labels might see the wrong thing, so they stay blocked.

## 自己跑一遍 / Try it yourself

```python
class InternalRef: pass
class UserRef: pass

def has_unsafe(refs):
    return any(not isinstance(ref, InternalRef) for ref in refs)

print(has_unsafe([InternalRef(), InternalRef()]))
print(has_unsafe([InternalRef(), UserRef()]))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
False
True
```

中文: toy 例子保留核心规则：内部引用不阻塞，任何未知引用都阻塞。

English: The toy keeps the core rule: internal refs do not block, any unknown ref does.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Dynamo guard internals** / **Dynamo guard internals**: guard 需要弱引用追踪对象存活，但不该改变用户可见语义。 / Guards need weakrefs for liveness without changing user-visible semantics.
- **Autograd use-count checks** / **Autograd use-count checks**: PyTorch 常用保守检查避免对象身份和底层 storage 语义分离。 / PyTorch often uses conservative checks when object identity and storage semantics can diverge.

## 注意事项 / Caveats / when it breaks

- **用户 weakref 仍然拒绝 / User weakrefs are still rejected**: 这是语义保护，不是性能限制。 / This protects semantics, not performance.
- **只适合 tensor weakref / Tensor weakref only**: `_TensorWeakRef` 明确检查返回对象是 `Tensor`。 / `_TensorWeakRef` asserts the returned object is a `Tensor`.
- **swap 后不要跨 forward/backward 改 dtype/device / Avoid dtype/device swaps between forward and backward**: 后面还有 use-count/autograd prehook 保护。 / Later use-count and autograd checks still guard unsafe swaps.

## 延伸阅读 / Further reading

- [PyTorch `swap_tensors`](https://github.com/pytorch/pytorch/blob/6d0950b4277092132463d743df98cd03009986b2/torch/utils/__init__.py)
- [PyTorch `TensorWeakRef`](https://github.com/pytorch/pytorch/blob/6d0950b4277092132463d743df98cd03009986b2/torch/utils/weak.py)
