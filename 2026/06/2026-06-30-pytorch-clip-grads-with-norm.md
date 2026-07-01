---
date: 2026-06-30
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/nn/utils/clip_grad.py
permalink: https://github.com/pytorch/pytorch/blob/6e78fad4760d4e471b8bf025a55e8d36eda006d7/torch/nn/utils/clip_grad.py#L115-L221
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, gradients, foreach]
---

# PyTorch 梯度裁剪的第二步：只缩小，不放大 / PyTorch Gradient Clipping Step Two: Scale Down, Never Up

> **一句话 / In one line**: `_clip_grads_with_norm_` 先算缩放系数，再 clamp 到 1，避免小梯度被意外放大。 / `_clip_grads_with_norm_` computes one scale factor, clamps it to 1, and avoids accidentally amplifying small gradients.

## 为什么重要 / Why this matters

梯度裁剪看起来是一行 API，但 PyTorch 把它拆成“求总范数”和“按已知范数缩放”两个函数。这个拆分让调用者可以复用总范数，也让多 tensor foreach 路径只做一次分组。

Gradient clipping looks like a one-line API, but PyTorch splits it into "compute total norm" and "scale with a known norm." That split lets callers reuse the norm and lets the foreach path group tensors only once per operation.

## 代码 / The code

`pytorch/pytorch` — [`torch/nn/utils/clip_grad.py`](https://github.com/pytorch/pytorch/blob/6e78fad4760d4e471b8bf025a55e8d36eda006d7/torch/nn/utils/clip_grad.py#L115-L221)

```python
@_no_grad
def _clip_grads_with_norm_(
    parameters: _tensor_or_tensors,
    max_norm: float,
    total_norm: torch.Tensor,
    foreach: bool | None = None,
) -> None:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    if len(grads) == 0:
        return
    grouped_grads: dict[
        tuple[torch.device, torch.dtype], tuple[list[list[Tensor]], list[int]]
    ] = _group_tensors_by_device_and_dtype([grads])  # type: ignore[assignment]

    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for (device, _), ([device_grads], _) in grouped_grads.items():
        if (foreach is None and _has_foreach_support(device_grads, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            torch._foreach_mul_(device_grads, clip_coef_clamped.to(device))
        elif foreach:
            raise RuntimeError(
                f"foreach=True was passed, but can't use the foreach API on {device.type} tensors"
            )
        else:
            clip_coef_clamped_device = clip_coef_clamped.to(device)
            for g in device_grads:
                g.mul_(clip_coef_clamped_device)


@_no_grad
def clip_grad_norm_(
    parameters: _tensor_or_tensors,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        is_generator = isinstance(parameters, types.GeneratorType)
        parameters = list(parameters)
        if is_generator and len(parameters) == 0:
            warnings.warn(
                "`parameters` is an empty generator, no gradient clipping will occur.",
                stacklevel=3,
            )
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = _get_total_norm(grads, norm_type, error_if_nonfinite, foreach)
    _clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm
```

## 逐行讲解 / What's happening

1. **第 124-128 行 / Lines 124-128**:
   - 中文: 只收集已有 `grad` 的参数，没有梯度就直接返回。
   - English: It only gathers parameters with existing gradients and returns early when there is nothing to scale.
2. **第 129-132 行 / Lines 129-132**:
   - 中文: 按 device/dtype 分组，给 foreach kernel 喂同质 tensor。
   - English: Gradients are grouped by device and dtype so foreach kernels receive homogeneous tensors.
3. **第 134-138 行 / Lines 134-138**:
   - 中文: `max_norm / total_norm` 被 clamp 到最大 1；大梯度缩小，小梯度保持原样。
   - English: `max_norm / total_norm` is clamped to at most 1; large gradients shrink, small gradients remain unchanged.
4. **第 175-221 行 / Lines 175-221**:
   - 中文: 公共 API 把 generator 先转成 list，避免遍历一次后参数消失。
   - English: The public API materializes generators into lists so parameters do not disappear after one pass.

## 类比 / The analogy

这像给一群人过限高门：太高的人要低头，已经够矮的人不会被垫高。

It is like passing through a height gate: tall people duck, but short people are not put on a platform.

## 自己跑一遍 / Try it yourself

```python
import torch

w = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
w.grad = torch.tensor([30.0, 40.0])
before = w.grad.norm()
after = torch.nn.utils.clip_grad_norm_([w], max_norm=5.0)
print(round(before.item(), 1), round(after.item(), 1), w.grad.tolist())
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```text
50.0 50.0 [2.999999..., 3.999999...]
```

返回值是裁剪前的范数，梯度本身被原地缩放到范数约等于 5。

The return value is the pre-clipping norm, while the gradients are scaled in-place to norm about 5.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **优化器 foreach 更新** / **Foreach optimizer updates**: AdamW 等优化器也先按 device/dtype 分桶，再批量调用 foreach。 / Optimizers such as AdamW also bucket tensors by device and dtype before calling foreach kernels.
- **AMP 梯度缩放** / **AMP gradient scaling**: 同样把“算比例”和“应用比例”分开，方便处理溢出和同步。 / AMP similarly separates computing a scale from applying it, which helps with overflow handling and synchronization.

## 注意事项 / Caveats / when it breaks

- **`foreach=True` 是强制要求** / **`foreach=True` is a hard request**: 不支持的设备会抛错，而不是静默 fallback。 / Unsupported devices raise instead of silently falling back.
- **返回值不是裁剪后范数** / **The return value is not the clipped norm**: 训练日志里要看清楚语义。 / Training logs should treat it as the original norm.

## 延伸阅读 / Further reading

- [PyTorch `clip_grad.py`](https://github.com/pytorch/pytorch/blob/6e78fad4760d4e471b8bf025a55e8d36eda006d7/torch/nn/utils/clip_grad.py)

