---
date: 2026-06-21
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/_functorch/apis.py
permalink: https://github.com/pytorch/pytorch/blob/15883c6209fcd2893ac53113a483e368bab4d47c/torch/_functorch/apis.py#L354-L464
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, functional-transforms, grad, vmap, per-sample-gradients]
---

# `torch.func.grad`：把函数变成它自己的梯度函数 / `torch.func.grad`: Turn Any Function Into Its Own Gradient Function

> **一句话 / In one line**: `grad(f)` 返回一个新函数，调用它等价于对 `f` 的输入求偏导数，可以像普通函数一样嵌套和组合，也可以和 `vmap` 组合实现单次 forward 计算批量逐样本梯度。 / `grad(f)` returns a new function whose call computes the partial derivative of `f` with respect to its inputs — composable like any function, and paired with `vmap` it computes per-sample gradients in a single forward pass.

## 为什么重要 / Why this matters

传统 PyTorch 自动微分的范式是"计算图 + `.backward()`"：你运行 forward，构建图，调用 `.backward()` 把梯度写进 `.grad` 属性，再手动读取。这个模式对单个损失很高效，但有两个痛点：

1. **逐样本梯度**需要循环（每次 forward 一个样本），或者用复杂的 Jacobian 技巧。
2. **高阶梯度**（梯度的梯度）需要 `create_graph=True`，容易出错。

`torch.func.grad` 采用 JAX 风格的"函数变换"范式：梯度本身就是一个函数变换——输入一个函数，输出它的梯度函数。这让梯度变成了可组合的一等公民：`grad(grad(sin))` 就是二阶导，`vmap(grad(loss))` 就是向量化的逐样本梯度，一行代码，一次 kernel 调用。

The traditional PyTorch autodiff paradigm is "compute graph + `.backward()`": run forward, build graph, call `.backward()` to write gradients into `.grad` attributes, then read them manually. This works well for a single loss, but has two pain points:

1. **Per-sample gradients** require a loop (one forward per sample) or complex Jacobian tricks.
2. **Higher-order gradients** require `create_graph=True`, which is error-prone.

`torch.func.grad` adopts a JAX-style "function transform" paradigm: the gradient is itself a function transform — you give it a function, you get back its gradient function. This makes gradients first-class and composable: `grad(grad(sin))` is the second derivative, `vmap(grad(loss))` is vectorized per-sample gradients — one line of code, one kernel invocation.

## 代码 / The code

`pytorch/pytorch` — [`torch/_functorch/apis.py`](https://github.com/pytorch/pytorch/blob/15883c6209fcd2893ac53113a483e368bab4d47c/torch/_functorch/apis.py#L354-L464)

```python
@exposed_in("torch.func")
def grad(
    func: Callable[_P, Any], argnums: argnums_t = 0, has_aux: bool = False
) -> Callable[_P, Any]:
    """``grad`` operator helps computing gradients of ``func`` with respect to the
    input(s) specified by ``argnums``. This operator can be nested to
    compute higher-order gradients.

    Args:
        func (Callable): A Python function that takes one or more arguments.
            Must return a single-element Tensor. If specified ``has_aux`` equals ``True``,
            function can return a tuple of single-element Tensor and other auxiliary objects:
            ``(output, aux)``.
        argnums (int or Tuple[int]): Specifies arguments to compute gradients with respect to.
            ``argnums`` can be single integer or tuple of integers. Default: 0.
        has_aux (bool): Flag indicating that ``func`` returns a tensor and other
            auxiliary objects: ``(output, aux)``. Default: False.

    Returns:
        Function to compute gradients with respect to its inputs. By default, the output of
        the function is the gradient tensor(s) with respect to the first argument.
        If specified ``has_aux`` equals ``True``, tuple of gradients and output auxiliary objects
        is returned. If ``argnums`` is a tuple of integers, a tuple of output gradients with
        respect to each ``argnums`` value is returned.

    Example of using ``grad``:

        >>> from torch.func import grad
        >>> x = torch.randn([])
        >>> cos_x = grad(lambda x: torch.sin(x))(x)
        >>> assert torch.allclose(cos_x, x.cos())
        >>>
        >>> # Second-order gradients
        >>> neg_sin_x = grad(grad(lambda x: torch.sin(x)))(x)
        >>> assert torch.allclose(neg_sin_x, -x.sin())

    When composed with ``vmap``, ``grad`` can be used to compute per-sample-gradients:

        >>> from torch.func import grad, vmap
        >>> batch_size, feature_size = 3, 5
        >>>
        >>> def model(weights, feature_vec):
        >>>     assert feature_vec.dim() == 1
        >>>     return feature_vec.dot(weights).relu()
        >>>
        >>> def compute_loss(weights, example, target):
        >>>     y = model(weights, example)
        >>>     return ((y - target) ** 2).mean()
        >>>
        >>> weights = torch.randn(feature_size, requires_grad=True)
        >>> examples = torch.randn(batch_size, feature_size)
        >>> targets = torch.randn(batch_size)
        >>> grad_weight_per_example = vmap(grad(compute_loss), in_dims=(None, 0, 0))(
        ...     weights, examples, targets
        ... )
    """
    import torch._functorch.eager_transforms as eager_transforms
    from torch.compiler import is_compiling

    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> tuple[Any, torch.Tensor]:
        return eager_transforms.grad_impl(func, argnums, has_aux, args, kwargs)

    if not is_compiling():
        wrapper = _wraps_without_dynamo_attrs(func)(wrapper)

    return wrapper
```

## 逐行讲解 / What's happening

1. **`@exposed_in("torch.func")` 装饰器**:
   - 中文: 把内部实现函数注册为 `torch.func.grad` 的公开名称。PyTorch 的功能分层结构：公开 API 在 `torch.func`，实现在 `torch._functorch`，底层在 `eager_transforms`。
   - English: Registers this internal function as the public `torch.func.grad` name. PyTorch's layering: public API in `torch.func`, implementation in `torch._functorch`, core engine in `eager_transforms`.

2. **`argnums: argnums_t = 0`**:
   - 中文: 默认对第 0 个参数（即第一个输入）求梯度。传 `argnums=(0, 1)` 则同时对前两个参数求梯度，返回梯度元组。
   - English: Defaults to differentiating with respect to argument 0 (the first input). Pass `argnums=(0, 1)` to get gradients with respect to the first two arguments as a tuple.

3. **`has_aux: bool = False`**:
   - 中文: 若为 `True`，`func` 可以返回 `(scalar_loss, aux_data)` 元组，`grad` 只对 `scalar_loss` 求梯度，同时把 `aux_data`（例如中间激活、调试信息）透传出来。这在需要同时返回梯度和额外信息时非常有用。
   - English: When `True`, `func` can return `(scalar_loss, aux_data)`; `grad` differentiates only through `scalar_loss` and passes `aux_data` (e.g. intermediate activations, metrics) through unchanged. Useful when you need both the gradient and side-channel information.

4. **`def wrapper(...)`** 和 **`eager_transforms.grad_impl`**:
   - 中文: 整个 `grad` 函数本身只有 5 行实质逻辑：创建一个闭包 `wrapper`，调用时把 `func`、`argnums`、`has_aux` 连同实际参数一起转发给 `eager_transforms.grad_impl`。真正的梯度计算在 `grad_impl` 里通过 functorch 的 `grad` 变换实现。
   - English: The entire `grad` function body is just 5 lines of substance: create a closure `wrapper` that, when called, forwards `func`, `argnums`, `has_aux`, and the actual arguments to `eager_transforms.grad_impl`. The actual gradient computation happens inside `grad_impl` via functorch's grad transform.

5. **`if not is_compiling(): wrapper = _wraps_without_dynamo_attrs(func)(wrapper)`**:
   - 中文: 只在非 `torch.compile` 上下文中，把原始函数的文档字符串和签名复制给 wrapper。在 `torch.compile` 编译期间跳过这步，避免干扰 Dynamo 的追踪。
   - English: Only outside `torch.compile` context, copies the original function's docstring and signature to the wrapper. Skipped during `torch.compile` tracing to avoid interfering with Dynamo's graph capture.

## 类比 / The analogy

`grad` 就像数学里的微分算子 $\frac{d}{dx}$：你把一个函数（例如 $\sin$）交给它，得到的是另一个函数（$\cos$）。你可以把 $\cos$ 再交给微分算子，得到 $-\sin$。这就是 `grad(grad(sin))`。`vmap` 则像把这个操作并行地施加在一个向量上——不是一个 $x$，而是同时处理一批 $x$。

`grad` is like the mathematical differential operator $\frac{d}{dx}$: you hand it a function (like $\sin$) and get back another function ($\cos$). You can hand $\cos$ back to the operator and get $-\sin$. That's `grad(grad(sin))`. `vmap` is like applying this operation in parallel across a vector — not one $x$, but a whole batch of $x$s at once.

## 自己跑一遍 / Try it yourself

```python
import torch
from torch.func import grad, vmap

# 1. 基本用法：sin 的导数是 cos
f_prime = grad(torch.sin)
x = torch.tensor(1.0)
print("grad(sin)(1.0):", f_prime(x))          # 应等于 cos(1) ≈ 0.5403
print("cos(1.0)      :", torch.cos(x))

# 2. 二阶梯度：sin 的二阶导是 -sin
f_double_prime = grad(grad(torch.sin))
print("grad(grad(sin))(1.0):", f_double_prime(x))  # 应等于 -sin(1) ≈ -0.8415

# 3. vmap + grad：逐样本梯度
def loss_fn(w, x, y):
    return ((w.dot(x) - y) ** 2)

w = torch.randn(4)
xs = torch.randn(8, 4)   # batch of 8
ys = torch.randn(8)
per_sample_grads = vmap(grad(loss_fn), in_dims=(None, 0, 0))(w, xs, ys)
print("per-sample grad shape:", per_sample_grads.shape)  # (8, 4)
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
grad(sin)(1.0): tensor(0.5403)
cos(1.0)      : tensor(0.5403)
grad(grad(sin))(1.0): tensor(-0.8415)
per-sample grad shape: torch.Size([8, 4])
```

中文：最关键的是第 3 个输出——`per_sample_grads.shape == (8, 4)`，意味着我们用一次 `vmap(grad(...))` 调用同时得到了 8 个样本各自对权重的梯度，形状为 `(batch, feature_dim)`，没有写任何 for 循环。

English: The key result is the third output: `per_sample_grads.shape == (8, 4)`, meaning we got 8 per-sample weight gradients in a single `vmap(grad(...))` call — shape `(batch, feature_dim)` — with no for loop written anywhere.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **JAX `jax.grad`** / **JAX**: `torch.func.grad` 直接对标 JAX 的 `jax.grad`，两者 API 几乎一一对应，概念完全相同。 / `torch.func.grad` directly mirrors `jax.grad`; the two APIs are nearly 1:1 in concept.
- **差分隐私训练（DP-SGD）** / **Differential Privacy (DP-SGD)**: Opacus 库在支持 `vmap(grad(...))` 之后，把逐样本梯度计算速度提升了 5-10x，因为可以摆脱 sample-level for loop。 / The Opacus library sped up per-sample gradient computation 5-10x after adding `vmap(grad(...))` support, eliminating the per-sample loop.
- **元学习（MAML）** / **Meta-learning (MAML)**: `grad(grad(loss))` 实现内环梯度 + 外环对内环梯度求导，比 `create_graph=True` 更易读。 / `grad(grad(loss))` implements inner-loop gradient + outer-loop differentiation through the inner loop, much cleaner than `create_graph=True`.

## 注意事项 / Caveats / when it breaks

- **`func` 必须返回标量** / **`func` must return a scalar**: `grad` 只能对标量输出求梯度。如果需要 Jacobian 矩阵，用 `torch.func.jacrev` 或 `torch.func.jacfwd`。 / `grad` only differentiates scalar outputs. For Jacobian matrices, use `torch.func.jacrev` or `torch.func.jacfwd`.
- **不支持 in-place 操作** / **In-place ops unsupported**: functorch 的变换不兼容 in-place tensor 操作（如 `x.add_(1)`），会报错。 / functorch transforms are incompatible with in-place tensor operations (`x.add_(1)`) — these will error.
- **`torch.no_grad()` 的语义** / **`torch.no_grad()` semantics**: 如函数内部有 `torch.no_grad()` 块，`grad` 会尊重它（内部 no_grad 不会被计算梯度）；但外部套 `torch.no_grad()` 不会阻止 `grad` 计算，因为 `grad` 是函数变换而非计算图。 / A `torch.no_grad()` block inside the function is respected by `grad`; but wrapping the `grad` call in `torch.no_grad()` does NOT prevent gradient computation, because `grad` is a function transform, not a graph operation.

## 延伸阅读 / Further reading

- [torch.func docs](https://pytorch.org/docs/stable/func.html)
- [functorch tutorial: per-sample gradients](https://pytorch.org/tutorials/intermediate/per_sample_grads.html)
- [JAX autodiff cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html)
