---
date: 2026-06-26
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/_functorch/apis.py
permalink: https://github.com/pytorch/pytorch/blob/40c2703d252fcc339bfc3aef80626efdeeb331b5/torch/_functorch/apis.py#L67-L258
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, functional-transforms, vmap, vectorization, per-sample-gradients]
---

# `torch.func.vmap`：把任意函数向量化，批量维度随心所欲 / `torch.func.vmap`: Vectorize Any Function, Put the Batch Dimension Anywhere

> **一句话 / In one line**: `vmap(func)` 返回一个新函数，它把 `func` 在输入的某个维度上"自动 for 循环"，但编译成一次向量化内核而不是真正循环，同时支持与 `grad` 组合得到 per-sample 梯度。 / `vmap(func)` returns a new function that maps `func` over a chosen dimension of its inputs — compiled into a single vectorized kernel rather than a real loop — and composes with `grad` to yield per-sample gradients.

## 为什么重要 / Why this matters

PyTorch 的 `vmap` 解决了一个普遍的痛点：你写好了一个处理**单个**样本的函数，但现在需要在 batch 上跑。传统做法要么手写 `torch.stack` + `for` 循环（慢），要么重构函数加 batch 维度（改接口、易出错）。`vmap` 允许你保留单样本视角的函数不变，用一行代码升维到 batch。

更重要的是，`vmap` 和其他函数式变换（`grad`、`jvp`、`vjp`）可以自由组合。最常见的用法：`vmap(grad(loss))` — 对 batch 里每个样本单独计算梯度，不改 loss 函数本身，不扩展内存里的 batch 维度循环。这对元学习、影响函数、LoRA per-sample 梯度估计等场景极其有用。

`vmap` solves a perennial PyTorch annoyance: you write a function for a **single** example, but now you need to run it over a batch. Options were either a slow Python `for` loop or refactoring the function to accept an explicit batch dimension. `vmap` lets you keep the single-example function as-is and lift it to batches in one line.

More importantly, `vmap` composes freely with other functional transforms (`grad`, `jvp`, `vjp`). The most common pattern: `vmap(grad(loss))` — compute per-sample gradients for every example in a batch without modifying the loss function or writing any loops. This is critical for meta-learning, influence functions, and per-sample LoRA gradient estimation.

## 代码 / The code

`pytorch/pytorch` — [`torch/_functorch/apis.py`](https://github.com/pytorch/pytorch/blob/40c2703d252fcc339bfc3aef80626efdeeb331b5/torch/_functorch/apis.py#L67-L258)

```python
@exposed_in("torch.func")
def vmap(
    func: Callable[_P, _R],
    in_dims: in_dims_t = 0,
    out_dims: out_dims_t = 0,
    randomness: str = "error",
    *,
    chunk_size: int | None = None,
) -> Callable[_P, _R]:
    """
    vmap is the vectorizing map; ``vmap(func)`` returns a new function that
    maps ``func`` over some dimension of the inputs. Semantically, vmap
    pushes the map into PyTorch operations called by ``func``, effectively
    vectorizing those operations.

    # Key signature patterns (from docstring):
    #
    # Basic batch:
    #   batched_dot = vmap(torch.dot)        # [D],[D]->[]  becomes  [N,D],[N,D]->[N]
    #
    # Non-first batch dim:
    #   batched_dot = vmap(torch.dot, in_dims=1)   # [N,D],[N,D] -> [D]
    #
    # Mixed batch dims — first arg batched, second not:
    #   batched_dot = vmap(torch.dot, in_dims=(0, None))  # [N,D],[D] -> [N]
    #
    # Nested vmap for 2-D batches:
    #   batched_dot = vmap(vmap(torch.dot))   # [N1,N0,D],[N1,N0,D] -> [N1,N0]
    #
    # Per-sample gradients (key composition with grad):
    #   grad_weight_per_example = vmap(grad(compute_loss), in_dims=(None, 0, 0))(*inputs)
    """
    from torch.compiler import is_compiling

    _check_randomness_arg(randomness)
    if not (chunk_size is None or chunk_size > 0):
        raise ValueError(
            f"vmap: chunk_size should be None or greater than 0. (got {chunk_size})"
        )

    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        return vmap_impl(
            func,
            in_dims,
            out_dims,
            randomness,
            chunk_size,
            *args,
            **kwargs,
        )

    if not is_compiling():
        wrapped = _wraps_without_dynamo_attrs(func)(wrapped)

    return wrapped
```

## 逐行讲解 / What's happening

1. **`in_dims: in_dims_t = 0`**
   - 中文: 告诉 vmap 每个输入的**哪个维度**是 batch 维度。默认 `0` 表示所有输入的第 0 维都是 batch。可以传 `tuple` 为每个输入单独指定：`(0, None)` 表示第一个参数第 0 维是 batch，第二个参数没有 batch 维（会被每个 batch 元素共享）。也可以传 `dict`，与 Python struct 输入配合使用。
   - English: Specifies which dimension of **each input** is the batch dimension. Default `0` means all inputs' dim-0 is the batch. Pass a tuple to specify per-input: `(0, None)` means the first arg has a batch dim at 0, the second has no batch dim (broadcast across all elements). You can also pass a dict to match a struct input.

2. **`out_dims: out_dims_t = 0`**
   - 中文: 控制 batch 维度在**输出**中出现的位置。默认 `0` 把 batch 放在最前面。传 `1` 可以让输出的第 1 维是 batch，例如 `vmap(f, out_dims=1)(x)` 返回 `[5, 2]` 而不是 `[2, 5]`。
   - English: Controls where the batch dimension appears **in the output**. Default `0` puts it first. Passing `1` puts it at position 1, e.g. `vmap(f, out_dims=1)(x)` returns `[5, 2]` instead of `[2, 5]`.

3. **`randomness: str = "error"`**
   - 中文: 控制 vmap 内部随机操作的行为。`"error"` 遇到随机操作直接报错（安全默认）；`"different"` 让每个 batch 元素得到不同的随机数；`"same"` 让所有 batch 元素得到相同的随机数。注意：这只管 PyTorch 随机操作，不管 `random` 模块或 NumPy。
   - English: Controls randomness inside vmap. `"error"` raises on any random op (safe default); `"different"` gives each batch element independent randomness; `"same"` broadcasts the same random draw. Note: this only covers PyTorch random ops, not `random` or NumPy.

4. **`chunk_size: int | None = None`**
   - 中文: 内存救星。当 batch 太大导致 vmap 一次性把所有 batch 展开会 OOM 时，传入 `chunk_size=16` 让 vmap 每次只处理 16 个样本，等价于在 Python 层做 for 循环但不改 func 的接口。`chunk_size=1` 完全等价于逐样本 for 循环。
   - English: A memory escape hatch. When fully unrolling the batch would OOM, `chunk_size=16` processes 16 examples at a time — equivalent to a for-loop without touching `func`'s interface. `chunk_size=1` is exactly a per-example loop.

5. **`if not is_compiling(): wrapped = _wraps_without_dynamo_attrs(func)(wrapped)`**
   - 中文: 不在 `torch.compile` 跟踪期间，才把原始函数的 `__name__`、`__doc__` 等元数据复制到包装函数上，并**移除** `_torchdynamo_inline` 等 Dynamo 属性。这样 Dynamo 不会把 vmap 的包装函数误认为是一个已编译的可调用对象，避免产生错误的重编译决策。
   - English: Only outside `torch.compile` tracing does it copy metadata (`__name__`, `__doc__`) from the original to the wrapper and **strip** Dynamo attrs like `_torchdynamo_inline`. Without stripping, Dynamo would treat the wrapper as a compiled callable and make wrong recompilation decisions.

## 类比 / The analogy

想象你是一位厨师，写好了处理**一颗**土豆的食谱（削皮 → 切块 → 称重）。现在来了一箱 32 颗土豆。普通做法：你拿出食谱、处理第 1 颗、记录结果、再拿食谱、处理第 2 颗……（Python for 循环，慢）。`vmap` 相当于把你的单颗土豆食谱"工厂化"——生产线上 32 个工位同时处理 32 颗，你的食谱本身一个字都不用改。`in_dims=(0, None)` 就像说"这批土豆每颗不同，但用的刀（第二个参数）所有工位共享同一把"。

Think of a chef with a recipe for **one** potato (peel → cube → weigh). Now a crate of 32 arrives. The slow approach: follow the recipe 32 times in sequence. `vmap` is like turning the single-potato recipe into a production line with 32 simultaneous workstations — the recipe itself doesn't change. `in_dims=(0, None)` means "each potato is different, but all workstations share the same knife (the second argument)."

## 自己跑一遍 / Try it yourself

```python
import torch
from torch.func import vmap, grad

# 1. Basic: batch dot products
x, y = torch.randn(4, 5), torch.randn(4, 5)
batched_dot = vmap(torch.dot)
print(batched_dot(x, y).shape)  # [4]

# 2. Per-sample gradients (the killer use case)
def loss_fn(w, x, t):
    return ((x @ w) - t).pow(2).mean()

W = torch.randn(5, requires_grad=True)
X = torch.randn(4, 5)   # 4 examples
T = torch.randn(4)

# Gradient w.r.t. W for each example independently:
per_sample_grads = vmap(grad(loss_fn), in_dims=(None, 0, 0))(W, X, T)
print(per_sample_grads.shape)  # [4, 5]  — one grad per example

# 3. chunk_size for memory control
big_X = torch.randn(1024, 5)
big_T = torch.randn(1024)
chunked_grads = vmap(grad(loss_fn), in_dims=(None, 0, 0), chunk_size=64)(W, big_X, big_T)
print(chunked_grads.shape)     # [1024, 5]
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
torch.Size([4])
torch.Size([4, 5])
torch.Size([1024, 5])
```

中文：注意 `per_sample_grads` 的 shape 是 `[4, 5]` —— 4 个样本、每个样本对 5 维权重向量各有一个梯度。如果用普通 `.backward()` 你只能得到 batch 平均后的梯度（shape `[5]`），要得到逐样本梯度需要循环 4 次。`vmap` 把这 4 次变成了一次向量化内核。

English: `per_sample_grads` has shape `[4, 5]` — one gradient per example over the 5-dimensional weight vector. A regular `.backward()` would give only the batch-averaged gradient (shape `[5]`). To get per-sample gradients without `vmap` you'd loop 4 times. `vmap` collapses those 4 passes into a single vectorized kernel.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **JAX `jax.vmap`** / **JAX `jax.vmap`**: PyTorch vmap 的直接灵感来源，API 高度相似。JAX 的 `jit(vmap(grad(f)))` 对应 PyTorch 的 `torch.compile(vmap(grad(f)))`。 / The direct inspiration; the API is nearly identical. JAX's `jit(vmap(grad(f)))` maps to PyTorch's `torch.compile(vmap(grad(f)))`.
- **LoRA per-sample gradient noise estimation** / **LoRA per-sample gradient noise estimation**: `vmap(grad(lora_loss))` 可以高效估计 LoRA 参数对每个样本的影响，用于 DataComp 或影响函数数据选择。 / `vmap(grad(lora_loss))` efficiently estimates per-example influence on LoRA parameters, useful for DataComp-style data selection and influence functions.
- **`torch.func.hessian`** / **`torch.func.hessian`**: 内部实现就是 `vmap(jacrev(jacrev(f)))` —— 先对外层 grad 做 vmap 向量化行，再对内层 grad 做 jacrev 逐列计算，最终得到 Hessian 矩阵。 / Implemented as `vmap(jacrev(jacrev(f)))` — vectorize rows with vmap, compute columns with jacrev, producing the full Hessian.

## 注意事项 / Caveats / when it breaks

- **有状态副作用** / **Side effects break vmap**: `func` 里不能有 Python 级别的状态更新（`dict.update()`、`global` 变量等）。vmap 会把 `func` 向量化地执行，副作用的执行次数不确定。 / `func` must be pure: no Python-level state mutations (`dict.update()`, globals). vmap vectorizes execution and side-effect count is undefined.
- **`randomness="error"` 是默认值** / **`randomness="error"` is the default**: 如果 `func` 内部有 `torch.randn()`、`torch.dropout()` 等，忘记设置 `randomness="different"` 会直接报错，而不是静默地给所有 batch 元素相同的随机数。 / If `func` calls `torch.randn()` or `torch.dropout()` and you forget `randomness="different"`, you get an explicit error — not the silent wrong result of identical random draws.
- **`in_dims=None` 表示广播，不是跳过** / **`in_dims=None` means broadcast, not skip**: `None` 告诉 vmap "这个参数在所有 batch 元素间共享"，函数仍然会接收到整个张量，不是 slice。 / `None` tells vmap "this arg is shared across all batch elements" — the function still sees the full tensor, not a slice.

## 延伸阅读 / Further reading

- [PyTorch functorch transforms overview](https://pytorch.org/docs/stable/func.html)
- [JAX sharp bits: vmap and side effects](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html)
- [Per-sample gradients tutorial](https://pytorch.org/tutorials/intermediate/per_sample_grads.html)
