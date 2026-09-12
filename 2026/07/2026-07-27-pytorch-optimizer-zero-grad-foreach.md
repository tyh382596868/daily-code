---
date: 2026-07-27
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/optim/optimizer.py
permalink: https://github.com/pytorch/pytorch/blob/9985428cfce1b55d1df7e0f85363ce17a39b0644/torch/optim/optimizer.py#L1023-L1085
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, pytorch, optimizer, foreach]
---

# PyTorch zero_grad：清空梯度也能批处理 / PyTorch zero_grad: Batch the Gradient Clearing Too

> **一句话 / In one line**: `zero_grad(set_to_none=False)` 不只是逐个清零；在 foreach/fused optimizer 下，它会按 device 和 dtype 分组后批量清零。 / `zero_grad(set_to_none=False)` is not only a per-parameter loop; under foreach or fused optimizers, it groups gradients by device and dtype before zeroing them in batches.

## 为什么重要 / Why this matters

训练循环每一步都会清梯度，这条路径的开销会被放大很多次。PyTorch 在保持 `None` 与零张量语义差异的同时，把可批量处理的 dense grad 收集起来交给 `_foreach_zero_`，减少 Python 循环和 kernel 启动。

Gradient clearing runs every training step, so small overheads repeat constantly. PyTorch preserves the semantic difference between `None` and zero tensors, then batches dense gradients through `_foreach_zero_` to reduce Python looping and kernel launches.

## 代码 / The code

`pytorch/pytorch` — [`torch/optim/optimizer.py`](https://github.com/pytorch/pytorch/blob/9985428cfce1b55d1df7e0f85363ce17a39b0644/torch/optim/optimizer.py#L1023-L1085)

```python
    @torch._disable_dynamo
    def zero_grad(self, set_to_none: bool = True) -> None:
        r"""Reset the gradients of all optimized :class:`torch.Tensor` s.

        Args:
            set_to_none (bool, optional): Instead of setting to zero, set the grads to None. Default: ``True``

                This will in general have lower memory footprint, and can modestly improve performance.
                However, it changes certain behaviors. For example:

                1. When the user tries to access a gradient and perform manual ops on it,
                   a None attribute or a Tensor full of 0s will behave differently.
                2. If the user requests ``zero_grad(set_to_none=True)`` followed by a backward pass, ``.grad``\ s
                   are guaranteed to be None for params that did not receive a gradient.
                3. ``torch.optim`` optimizers have a different behavior if the gradient is 0 or None
                   (in one case it does the step with a gradient of 0 and in the other it skips
                   the step altogether).
        """
        foreach = self.defaults.get("foreach", False) or self.defaults.get(
            "fused", False
        )

        if not hasattr(self, "_zero_grad_profile_name"):
            self._patch_step_function()

        per_device_and_dtype_grads: (
            defaultdict[torch.device, defaultdict[torch.dtype, list[torch.Tensor]]]
            | None
        )
        if foreach:
            per_device_and_dtype_grads = defaultdict(lambda: defaultdict(list))
        else:
            per_device_and_dtype_grads = None

        with torch.autograd.profiler.record_function(self._zero_grad_profile_name):
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        if set_to_none:
                            p.grad = None
                        else:
                            if p.grad.grad_fn is not None:
                                p.grad.detach_()
                            else:
                                p.grad.requires_grad_(False)
                            if not foreach or p.grad.is_sparse:
                                p.grad.zero_()
                            else:
                                if per_device_and_dtype_grads is None:
                                    raise AssertionError(
                                        "Expected per_device_and_dtype_grads to be set"
                                    )
                                per_device_and_dtype_grads[p.grad.device][
                                    p.grad.dtype
                                ].append(p.grad)
            if foreach:
                if per_device_and_dtype_grads is None:
                    raise AssertionError(
                        "Expected per_device_and_dtype_grads to be set"
                    )
                for per_dtype_grads in per_device_and_dtype_grads.values():
                    for grads in per_dtype_grads.values():
                        torch._foreach_zero_(grads)
```

## 逐行讲解 / What's happening

1. **第 1041-1043 行 / Lines 1041-1043 (`foreach flag`)**:
   - 中文: 只要 optimizer 默认启用 `foreach` 或 `fused`，清零路径也跟着走批处理。
   - English: If the optimizer defaults enable `foreach` or `fused`, gradient clearing follows the batched path too.
2. **第 1052-1055 行 / Lines 1052-1055 (`group table`)**:
   - 中文: 批处理模式创建 `device -> dtype -> grads` 的二级表。
   - English: Batched mode builds a two-level `device -> dtype -> grads` table.
3. **第 1058-1069 行 / Lines 1058-1069 (`semantic branch`)**:
   - 中文: `set_to_none=True` 直接丢掉 grad；否则先 detach/关掉 requires_grad，再决定是否立刻清零。
   - English: `set_to_none=True` drops the grad; otherwise PyTorch detaches or disables grad tracking before deciding whether to zero immediately.
4. **第 1071-1085 行 / Lines 1071-1085 (`foreach zero`)**:
   - 中文: dense grad 被收集到同设备同 dtype 的桶里，最后一次 `_foreach_zero_` 批量处理。
   - English: Dense grads are collected into same-device, same-dtype buckets and cleared with one `_foreach_zero_` per bucket.

## 类比 / The analogy

这像收餐盘：散客的盘子可以一个个擦，但同材质同尺寸的盘子会先叠成一摞，再一起送进洗碗机。

It is like clearing plates in a cafeteria: odd plates can be wiped one by one, but matching plates are stacked and sent through the dishwasher together.

## 自己跑一遍 / Try it yourself

```python
from collections import defaultdict

grads = [("cuda", "fp16", 1), ("cuda", "fp16", 2), ("cpu", "fp32", 3)]
buckets = defaultdict(lambda: defaultdict(list))
for device, dtype, grad in grads:
    buckets[device][dtype].append(grad)

for device, per_dtype in buckets.items():
    for dtype, values in per_dtype.items():
        print(device, dtype, "zero", values)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
cuda fp16 zero [1, 2]
cpu fp32 zero [3]
```

分组的关键不是参数属于哪个 layer，而是底层 foreach kernel 能否一次处理同 device、同 dtype 的张量。

The grouping key is not the layer; it is whether the backend foreach kernel can process tensors with the same device and dtype together.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **clip_grad foreach** / **clip_grad foreach**: 中文: `clip_grad_norm_` 也按 device/dtype 分组后调用 foreach API。 / English: `clip_grad_norm_` also groups by device and dtype before using foreach APIs.
- **optimizer step** / **Optimizer step**: 中文: 多个 optimizer 的 foreach step 都复用同样的批处理思想。 / English: Many foreach optimizer steps use the same batching idea.

## 注意事项 / Caveats / when it breaks

- **稀疏梯度** / **Sparse gradients**: 中文: sparse grad 不进 foreach bucket，会走逐个清零。 / English: Sparse grads skip the foreach bucket and are zeroed individually.
- **None 语义** / **`None` semantics**: 中文: `None` 会让 optimizer step 跳过该参数，零梯度则可能执行一次零更新。 / English: `None` makes the optimizer skip the parameter; a zero grad may still perform a zero update.

## 延伸阅读 / Further reading

- [pytorch/pytorch source](https://github.com/pytorch/pytorch/blob/9985428cfce1b55d1df7e0f85363ce17a39b0644/torch/optim/optimizer.py#L1023-L1085)
