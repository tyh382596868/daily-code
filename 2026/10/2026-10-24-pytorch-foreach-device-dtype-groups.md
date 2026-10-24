---
date: 2026-10-24
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/utils/_foreach_utils.py
permalink: https://github.com/pytorch/pytorch/blob/97a3cf652accf801225a2a05b5b3ab19b4d65473/torch/utils/_foreach_utils.py#L8-L60
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, foreach, optimizer]
---

# PyTorch foreach 分组：先按设备和 dtype 分桶 / PyTorch Foreach Groups: Bucket by Device and dtype First

> **一句话 / In one line**: foreach 优化不是把任意 tensor 列表直接交给 kernel；PyTorch 先把兼容项按 `(device, dtype)` 分组。 / A foreach fast path does not accept an arbitrary tensor list; PyTorch first groups compatible tensors by `(device, dtype)`.

## 为什么重要 / Why this matters

optimizer 想一次更新很多参数，但 CUDA、XPU 和不同精度的 tensor 不能混在同一个 foreach kernel 里。这个小工具把“能一起跑”的条件变成明确的数据结构。

An optimizer wants to update many parameters at once, but tensors on different devices or with different dtypes cannot share a foreach kernel. This helper makes “can run together” an explicit data structure.

## 代码 / The code

`pytorch/pytorch` — [`_foreach_utils.py`](https://github.com/pytorch/pytorch/blob/97a3cf652accf801225a2a05b5b3ab19b4d65473/torch/utils/_foreach_utils.py#L8-L60)

```python
def _get_foreach_kernels_supported_devices() -> list[str]:
    return ["cuda", "xpu", "mtia", torch._C._get_privateuse1_backend_name()]

@no_grad()
def _group_tensors_by_device_and_dtype(
    tensorlistlist: TensorListList, with_indices: bool = False,
) -> dict[tuple[torch.device, torch.dtype], tuple[TensorListList, Indices]]:
    return torch._C._group_tensors_by_device_and_dtype(tensorlistlist, with_indices)

def _has_foreach_support(tensors: list[Tensor], device: torch.device) -> bool:
    return _device_has_foreach_support(device) and all(
        t is None or type(t) in _foreach_supported_types for t in tensors
    )
```

## 逐行讲解 / What's happening

支持设备表和类型检查先挡住不可能的 fast path。真正的分组在 C++ 端完成，并可返回原始索引：这让优化器可以分桶计算，又把结果写回原来的参数位置。

The device list and type check reject impossible fast paths early. The C++ helper performs the actual grouping and can retain original indices, letting an optimizer compute by bucket and scatter results back correctly.

## 类比 / The analogy

像机场安检把旅客先按航站楼和登机口分队；分好队后每队可以快速一起过闸，不能把不同航班塞进同一条登机通道。

It is airport routing: group travellers by terminal and gate, then move each group quickly. You cannot send incompatible flights through one boarding lane.

## 自己跑一遍 / Try it yourself

```python
import torch
xs = [torch.ones(2), torch.ones(2, dtype=torch.float64)]
groups = {}
for i, x in enumerate(xs): groups.setdefault((x.device, x.dtype), []).append(i)
print(groups)
```

运行 / Run with: `pip install torch && python try.py`

预期输出 / Expected output: 两个不同 `(device, dtype)` key。 / Two distinct `(device, dtype)` keys.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **fused optimizers** / **fused optimizers**: AdamW、SGD 等会在分组后选择 foreach 或 fused 路径。 / AdamW and SGD select foreach or fused paths after grouping.

## 注意事项 / Caveats / when it breaks

- **索引不是可选装饰** / **Indices are not decoration**: 有 `None` 或多个对应 tensor 列表时，散射回写必须保留位置。 / With `None` values or aligned lists, scatter-back needs the original positions.

## 延伸阅读 / Further reading

- [PyTorch foreach utilities](https://github.com/pytorch/pytorch/blob/97a3cf652accf801225a2a05b5b3ab19b4d65473/torch/utils/_foreach_utils.py#L8-L60)
