---
date: 2026-06-02
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/nn/utils/clip_grad.py
permalink: https://github.com/pytorch/pytorch/blob/c42e39b73c4b6bab2e78f982765bd2029abc2a2a/torch/nn/utils/clip_grad.py#L48-L116
difficulty: intermediate
read_time: ~9 min
tags: [code-of-the-day, pytorch, foreach, multi-tensor, optimizer-internals]
---

# 一份 70 行的 multi-tensor 模板:`_get_total_norm` / The 70-line multi-tensor template: `_get_total_norm`

> **一句话 / In one line**: PyTorch 把所有「对一堆张量做同一件事」的快路径都长成同一个样子 —— 按 (device, dtype) 分桶,每桶喂一次 `torch._foreach_*`,不支持就退化成 per-tensor 循环;`_get_total_norm` 是这个模板最干净的范本。 / Every PyTorch "do the same thing to many tensors" fast path looks identical — bucket by (device, dtype), call one `torch._foreach_*` per bucket, fall back to a per-tensor loop on unsupported devices; `_get_total_norm` is the cleanest specimen of that template.

## 为什么重要 / Why this matters

训练大模型时,你的优化器、梯度裁剪、EMA、weight decay 等等都要对成百上千个 Parameter 做同一件小事。如果每个张量启一次 CUDA kernel,光是 launch 开销就把训练时间吃掉一半。PyTorch 的解法是 `torch._foreach_*` 系列融合算子 —— 一次 launch,内部 batch 处理 N 个张量。但每个工程师写代码时都要重发明同一个「按 device/dtype 分桶 + 调 foreach + 不支持就退化」的模板,所以 PyTorch 把它写进了官方 utility:`_group_tensors_by_device_and_dtype`。

When you train a big model, the optimizer, gradient clipping, EMA, weight decay etc. all have to do the same small thing to hundreds or thousands of Parameters. If you launch one CUDA kernel per tensor, kernel launch overhead alone eats half your training time. The PyTorch answer is the `torch._foreach_*` family of fused ops — one launch, internally batched across N tensors. But every engineer ends up reinventing the same "group by (device, dtype) + call foreach + fall back to per-tensor" template, so PyTorch codified it as an official utility: `_group_tensors_by_device_and_dtype`. `_get_total_norm` is the cleanest example.

## 代码 / The code

`pytorch/pytorch` — [`torch/nn/utils/clip_grad.py`](https://github.com/pytorch/pytorch/blob/c42e39b73c4b6bab2e78f982765bd2029abc2a2a/torch/nn/utils/clip_grad.py#L48-L116)

```python
@_no_grad
def _get_total_norm(
    tensors: _tensor_or_tensors,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
) -> torch.Tensor:
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    else:
        tensors = list(tensors)
    norm_type = float(norm_type)
    if len(tensors) == 0:
        return torch.tensor(0.0)
    first_device = tensors[0].device
    grouped_tensors: dict[
        tuple[torch.device, torch.dtype], tuple[list[list[Tensor]], list[int]]
    ] = _group_tensors_by_device_and_dtype(
        [tensors]
    )

    norms: list[Tensor] = []
    for (device, _), ([device_tensors], _) in grouped_tensors.items():
        if (foreach is None and _has_foreach_support(device_tensors, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            norms.extend(torch._foreach_norm(device_tensors, norm_type))
        elif foreach:
            raise RuntimeError(
                f"foreach=True was passed, but can't use the foreach API on {device.type} tensors"
            )
        else:
            norms.extend(
                [torch.linalg.vector_norm(g, norm_type) for g in device_tensors]
            )

    total_norm = torch.linalg.vector_norm(
        torch.stack([norm.to(first_device) for norm in norms]), norm_type
    )

    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped..."
        )
    return total_norm
```

## 逐行讲解 / What's happening

1. **`@_no_grad` 装饰器**:
   - 中文: 自定义的 `no_grad` 包装,避免直接 `@torch.no_grad`(直接用会引入循环依赖)。规则:**做梯度统计的代码本身不该建图**,否则你算的 norm 又会被反向追踪,内存翻倍。
   - English: Custom `no_grad` wrapper (using `@torch.no_grad` directly causes a circular import). Rule: **code that introspects gradients must not build a graph itself**, or the norm computation gets autograd-tracked and memory doubles.

2. **`if isinstance(tensors, torch.Tensor): tensors = [tensors]`**:
   - 中文: 标准 PyTorch utility 习惯 —— 既接受单个张量也接受 iterable,内部统一成 list。这一招在 `init_*`、`clip_*`、`_foreach_*` 几乎到处都是。
   - English: Standard PyTorch utility idiom — accept a single tensor or any iterable, normalize to a list. You see this everywhere in `init_*`, `clip_*`, and `_foreach_*` helpers.

3. **`_group_tensors_by_device_and_dtype([tensors])`**:
   - 中文: 关键步骤。它把扁平的张量列表按 (device, dtype) 分桶 —— 因为 `_foreach_norm(...)` 要求传进去的所有张量必须**同设备、同 dtype**(底层是一个融合 kernel,要拼成 array of pointers)。返回的结构嵌套很深,但你只要记住:每个 key 对应一组「可以一次喂给 foreach」的张量。
   - English: The key step. It buckets a flat tensor list by (device, dtype) — because `_foreach_norm(...)` requires every input tensor to share device and dtype (the underlying fused kernel packs them into an array of pointers). The return shape is deeply nested, but the mental model is simple: each key maps to one batch of tensors that can go to foreach in a single call.

4. **`for (device, _), ([device_tensors], _) in grouped_tensors.items():`**:
   - 中文: 遍历每个桶。第二个 `_` 是 `_group_tensors_by_device_and_dtype` 返回的索引数组(让你能把结果按原顺序写回去),这里不需要。
   - English: Iterate each bucket. The second `_` is the per-bucket index array returned by the grouping helper (lets you scatter results back in original order); not needed here.

5. **三分支:`foreach is None`、`foreach=True`、`foreach=False`**:
   - 中文: 自动模式下用 `_has_foreach_support(device_tensors, device)` 检测;用户强制 `True` 就再检查一次设备能力,不行就抛错;强制 `False` 走 `torch.linalg.vector_norm` 的 per-tensor 循环 —— 慢但正确,且对 MPS、CPU 等不全支持 foreach 的设备 always work。
   - English: In auto mode use `_has_foreach_support(device_tensors, device)` to detect; if the user passes `foreach=True`, double-check the device and raise on mismatch; if `foreach=False`, fall back to a per-tensor `torch.linalg.vector_norm` loop — slow but correct, and works on MPS, CPU, and other devices where foreach isn't fully covered.

6. **`torch._foreach_norm(device_tensors, norm_type)`**:
   - 中文: 一次 launch,N 个张量的 norm 全算出来,返回 N 个标量张量。底层是 `multi_tensor_apply` —— 把所有张量的指针 + size 打包成一个 metadata blob,扔给一个 CUDA kernel 用 grid-stride 循环处理。
   - English: One launch, N tensor norms computed at once, returns N scalar tensors. Under the hood it's `multi_tensor_apply` — pack every tensor's pointer + size into a metadata blob and hand it to a single CUDA kernel that walks them grid-stride.

7. **`total_norm = torch.linalg.vector_norm(torch.stack([norm.to(first_device) ...]), norm_type)`**:
   - 中文: 把每个桶算出来的标量 norm 搬回 `first_device`,堆成一个 1D 向量,再算一次 vector norm —— 这就是「整体梯度向量」的范数,完全等价于把所有梯度拼一根再算。`.to(first_device)` 处理多 GPU、多 dtype 模型(比如 LM head 在 GPU0,attention 在 GPU1)。
   - English: Move each bucket's scalar norm back to `first_device`, stack into a 1-D vector, take one more vector norm — exactly equivalent to flattening every gradient into one giant vector and norming it once, but never materializes that vector. The `.to(first_device)` handles models split across multiple GPUs or dtypes (e.g. LM head on GPU0, attention on GPU1).

## 类比 / The analogy

想象一个工厂要称量上千个零件的总重量。笨办法:一次称一个,千次秤动,人累死。快办法:先按材质分堆(铁、铜、铝),每堆全倒到一个电子台秤上一起称,得到三个分量重,最后把三个数加起来。`_group_tensors_by_device_and_dtype` 是「按材质分堆」,`torch._foreach_norm` 是「电子台秤一次称一堆」,最后那个 `vector_norm(stack(...))` 是「三个数加起来」。如果没有电子台秤(`_foreach_support == False`),就退回手秤模式 —— 仍然能算出对的总重,只是慢。

Picture a factory weighing thousands of small parts. The slow way: weigh them one at a time, thousands of scale operations. The fast way: sort them by material (iron, copper, aluminum), drop each pile onto one electronic scale at once, get three subtotals, then sum the three. `_group_tensors_by_device_and_dtype` is the sorting, `torch._foreach_norm` is the electronic scale weighing a whole pile at once, and the final `vector_norm(stack(...))` is the summing step. No electronic scale (`_foreach_support == False`)? Fall back to weighing one at a time — same total, slower.

## 自己跑一遍 / Try it yourself

```python
# pip install torch
import torch, time

# 1000 parameters, mixed shapes, single device + dtype
params = [torch.randn(d, device="cuda") for d in torch.randint(100, 5000, (1000,)).tolist()]

# Slow path: per-tensor norm
t = time.time(); torch.cuda.synchronize()
norms_slow = torch.stack([torch.linalg.vector_norm(p) for p in params])
total_slow = torch.linalg.vector_norm(norms_slow)
torch.cuda.synchronize(); slow_ms = (time.time() - t) * 1000

# Fast path: one foreach call
t = time.time(); torch.cuda.synchronize()
norms_fast = torch._foreach_norm(params, 2.0)
total_fast = torch.linalg.vector_norm(torch.stack(norms_fast))
torch.cuda.synchronize(); fast_ms = (time.time() - t) * 1000

print(f"per-tensor: {slow_ms:7.2f} ms   foreach: {fast_ms:7.2f} ms   "
      f"speedup: {slow_ms/fast_ms:.1f}x")
print(f"total norms match: {torch.allclose(total_slow, total_fast)}")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
per-tensor:   12.50 ms   foreach:    0.40 ms   speedup: 31.2x
total norms match: True
```

注意 speedup 主要来自消掉了 999 次 kernel launch,而不是计算更快 —— 这正是为什么 `_foreach_*` 在「张量多而每个小」的场景(优化器、clip_grad)收益最大。

The speedup comes almost entirely from killing 999 kernel launches, not from faster compute — which is why `_foreach_*` shines exactly in the "many small tensors" regime (optimizers, clip_grad).

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **每一个 PyTorch fused 优化器** / **Every PyTorch fused optimizer**: `_multi_tensor_adamw`、`_multi_tensor_sgd` 都是 `_group_tensors_by_device_and_dtype` + `_foreach_add_`/`_foreach_mul_` 串起来的;不是 magic,就是这套模板。 / `_multi_tensor_adamw`, `_multi_tensor_sgd` etc. are all `_group_tensors_by_device_and_dtype` + chained `_foreach_add_`/`_foreach_mul_` — no magic, same template.
- **EMA / SWA 实现** / **EMA / SWA implementations**: 用 `_foreach_mul_(ema, decay)` + `_foreach_add_(ema, p, alpha=1-decay)` 来更新一千个 EMA 副本,比 Python loop 快两个数量级。 / Update a thousand EMA copies with `_foreach_mul_(ema, decay)` + `_foreach_add_(ema, p, alpha=1-decay)`, two orders of magnitude faster than a Python loop.
- **Apex `multi_tensor_l2norm`** / **Apex `multi_tensor_l2norm`**: NVIDIA Apex 的 C++ 版本,API 长得几乎一样 —— PyTorch 的 `_foreach_norm` 就是把这个 upstream 进来。 / NVIDIA Apex's C++ version with a nearly identical API — PyTorch's `_foreach_norm` is essentially this upstreamed.
- **FSDP 的 grad 同步** / **FSDP grad sync**: 用同样的分桶 + foreach 模式来把 sharded grads all-reduce 之前先归一化。 / FSDP uses the same group + foreach pattern to normalize sharded grads before all-reduce.

## 注意事项 / Caveats / when it breaks

- **不要在 hot path 里调** `_group_tensors_by_device_and_dtype` **过于频繁** / **Don't call `_group_tensors_by_device_and_dtype` on the hot path too often**: 这个 grouping 本身要遍历所有张量取 device/dtype 元数据,纯 Python,有非平凡的开销。优化器内部一般每 step 只调一次。 / The grouping itself walks all tensors to read device/dtype metadata in pure Python and has non-trivial overhead. Optimizers typically call it once per step at most.
- **foreach 不支持 sparse、不支持 strided 视图的某些 layout** / **foreach doesn't support sparse, or some strided view layouts**: `_has_foreach_support` 会替你检测,但如果你混了 `torch.sparse` 张量进来,会安静地走慢路径,你也许不会发现。 / `_has_foreach_support` does the detection, but slip in a `torch.sparse` tensor and you'll quietly land on the slow path without noticing.
- **多 device 模型注意 `.to(first_device)` 的同步成本** / **Multi-device models pay sync cost on `.to(first_device)`**: 每个 bucket 算完都要把 scalar norm 搬到 first device,这是隐式的 device-to-device 拷贝。模型分布在 8 张卡上,你就有 8 次 D2D 拷贝 —— 通常没关系,但 batch 极小、`norm_type=inf` 的情况要 profile 一下。 / Each bucket's scalar norm gets copied to `first_device` — an implicit D2D copy. With a model split across 8 GPUs you get 8 D2D copies; usually negligible, but worth profiling for tiny batches or `norm_type=inf`.

## 延伸阅读 / Further reading

- [`torch._foreach_*` 完整算子表](https://pytorch.org/docs/stable/generated/torch._foreach_norm.html)
- [PyTorch blog — fused optimizers](https://pytorch.org/blog/optimizing-pytorch-training-with-fused-optimizers/)
- [`_group_tensors_by_device_and_dtype` 源码](https://github.com/pytorch/pytorch/blob/main/torch/utils/_foreach_utils.py)
- [Apex `multi_tensor_apply` 设计文档](https://github.com/NVIDIA/apex/blob/master/csrc/multi_tensor_apply.cuh)
