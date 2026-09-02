---
date: 2026-09-02
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/nn/modules/normalization.py
permalink: https://github.com/pytorch/pytorch/blob/73c27d87e075f3831e2462bd61cbbb37d0d82d69/torch/nn/modules/normalization.py#L343-L427
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, rmsnorm, normalization, functional]
---

# PyTorch RMSNorm：最后一维做均方根归一化 / PyTorch RMSNorm: Normalize by Root Mean Square Along the Last Dimension

> **一句话 / In one line**: `nn.RMSNorm` 只是给 `F.rms_norm()` 包了一层模块外壳，但它把 shape、参数初始化和可选 affine 都收得很整。 / `nn.RMSNorm` is mostly a thin module wrapper around `F.rms_norm()`, but it packages shape handling, parameter init, and optional affine cleanly.

## 为什么重要 / Why this matters

中文：很多人把 RMSNorm 记成“一个 normalize 函数”，但真正工程化时，麻烦在于 shape 怎么收、参数怎么建、最后怎么接到模块系统里。PyTorch 这段代码把这些边角收得很干净：单个 `normalized_shape` 会自动升成 tuple，是否学习仿射参数由 `elementwise_affine` 决定，前向只剩一行 functional 调用。

English: It is easy to remember RMSNorm as “just a normalization formula,” but in production the awkward parts are shape bookkeeping, parameter construction, and module integration. PyTorch handles those edges neatly here: a scalar `normalized_shape` is promoted to a tuple, affine parameters are optional, and the forward path collapses to a single functional call.

## 代码 / The code

`pytorch/pytorch` — [`torch/nn/modules/normalization.py`](https://github.com/pytorch/pytorch/blob/73c27d87e075f3831e2462bd61cbbb37d0d82d69/torch/nn/modules/normalization.py#L343-L427)

```python
class RMSNorm(Module):
    r"""Applies Root Mean Square Layer Normalization over a mini-batch of inputs.

    This layer implements the operation as described in
    the paper `Root Mean Square Layer Normalization <https://arxiv.org/pdf/1910.07467.pdf>`__

    .. math::
        y_i = \frac{x_i}{\mathrm{RMS}(x)} * \gamma_i, \quad
        \text{where} \quad \text{RMS}(x) = \sqrt{\epsilon + \frac{1}{n} \sum_{i=1}^{n} x_i^2}

    The RMS is taken over the last ``D`` dimensions, where ``D``
    is the dimension of :attr:`normalized_shape`. For example, if :attr:`normalized_shape`
    is ``(3, 5)`` (a 2-dimensional shape), the RMS is computed over
    the last 2 dimensions of the input.
    """

    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: tuple[int, ...]
    eps: float | None
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float | None = None,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in __init__.
        """
        if self.elementwise_affine:
            init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the forward pass.
        """
        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )
```

## 逐行讲解 / What's happening

1. **第 343-351 行 / Lines 343-351**:
   - 中文: 注释先把数学定义写清楚，核心就是 `x / RMS(x)` 再乘缩放参数 `gamma`。
   - English: The docstring states the math directly: divide by `RMS(x)`, then apply the learned scale `gamma`.
2. **第 392-414 行 / Lines 392-414**:
   - 中文: 构造函数把整数 shape 变 tuple，并在需要时创建 `weight`；不需要仿射时就注册空参数。
   - English: The constructor normalizes integer shapes into tuples and creates `weight` only when affine parameters are enabled.
3. **第 416-427 行 / Lines 416-427**:
   - 中文: 前向路径没有手写公式，直接交给 `F.rms_norm()`，模块层只负责参数和接口。
   - English: The forward path does not reimplement the math; it delegates to `F.rms_norm()` and lets the module own parameters and shape contracts.

## 类比 / The analogy

中文：像一台带自动校准的电子秤。秤本体只负责读数和显示，真正烦人的零点校准、单位设定、记忆功能，都被机身外壳包好了。

English: It is like a digital scale with auto-calibration. The weighing surface only reads the weight; all the annoying parts like zeroing, units, and saved settings live in the enclosure around it.

## 自己跑一遍 / Try it yourself

```python
import math

def rms_norm(xs, eps=1e-5):
    rms = math.sqrt(eps + sum(x * x for x in xs) / len(xs))
    return [round(x / rms, 3) for x in xs]

print(rms_norm([3.0, 4.0]))
print(rms_norm([1.0, 2.0, 2.0]))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[0.6, 0.8]
[0.577, 1.155, 1.155]
```

中文：RMSNorm 的关键不是“做标准化”本身，而是它不减均值、只管尺度，所以更像一个稳定放大器。

English: The point of RMSNorm is not normalization in the generic sense; it does not subtract the mean, so it behaves more like a stable gain control.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Wan2.1 WanRMSNorm** / **Wan2.1 WanRMSNorm**: 中文: 生产代码里常把同一条 RMS 公式包成自己的轻量模块。 / English: Production code often wraps the same RMS formula in a lightweight project-specific module.
- **Diffusers / Transformers layer norm wrappers** / **Diffusers / Transformers layer norm wrappers**: 中文: 许多库都把 functional kernel 和模块参数管理分开。 / English: Many libraries separate the functional kernel from module-side parameter management.

## 注意事项 / Caveats / when it breaks

- **最后一维形状必须对得上 / The tail shape must match**: 中文: `normalized_shape` 不是装饰参数，而是模块契约。 / English: `normalized_shape` is a contract, not a cosmetic knob.
- **`elementwise_affine=False` 就没有可学习缩放 / No affine means no learned scale**: 中文: 这会让模块完全退回到纯函数归一化。 / English: The module collapses to a pure functional normalization.

## 延伸阅读 / Further reading

- PyTorch source: https://github.com/pytorch/pytorch/blob/73c27d87e075f3831e2462bd61cbbb37d0d82d69/torch/nn/modules/normalization.py
