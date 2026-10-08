---
date: 2026-10-08
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/nn/attention/_utils.py
permalink: https://github.com/pytorch/pytorch/blob/79ef85d9b61ab6dff4da6790dc21aef835b57a58/torch/nn/attention/_utils.py#L12-L47
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, tensor-layout]
---

# PyTorch SDPA layout：输出也要沿着 query 的步幅走 / PyTorch SDPA Layout: Let the Output Follow the Query Strides

> **一句话 / In one line**: `_empty_with_matching_layout` 在 shape 变化时重建 stride 顺序，让 attention 输出尽量保持 query 的内存布局习惯。 / `_empty_with_matching_layout` rebuilds stride order when shape changes, so attention output keeps the query's layout style.

## 为什么重要 / Why this matters

高性能 attention 不只关心值对不对，还关心结果 tensor 的 layout。上游可能给出转置过、channels-last-like、或者带广播步幅的 query；如果输出随便变成默认 contiguous，后面的 kernel 可能少一次优化机会，甚至触发额外 copy。

High-performance attention cares about tensor layout, not just values. The query may arrive transposed, channels-last-like, or with broadcast-style strides. If the output blindly becomes default contiguous, later kernels may lose an optimization path or pay for an extra copy.

## 代码 / The code

`pytorch/pytorch` — [`torch/nn/attention/_utils.py`](https://github.com/pytorch/pytorch/blob/79ef85d9b61ab6dff4da6790dc21aef835b57a58/torch/nn/attention/_utils.py#L12-L47)

```python
def _empty_with_matching_layout(
    query: torch.Tensor, shape: tuple[int, ...]
) -> torch.Tensor:
    """Allocate an output with the query's dimension order."""
    if tuple(query.shape) == shape:
        return torch.empty_like(query)

    fill_order = sorted(
        range(query.dim()),
        key=lambda idx: query.stride()[idx] if query.stride()[idx] else math.inf,
    )
    strides = [0] * len(fill_order)
    stride = 1
    for idx in fill_order:
        strides[idx] = stride
        stride *= shape[idx]
    return torch.empty_strided(shape, strides, dtype=query.dtype, device=query.device)


def _input_requires_grad(*tensors: torch.Tensor) -> bool:
    """Returns True if any of the tensors requires grad"""
    return any(t.requires_grad for t in tensors)


def _postprocess_flash_output(inpt_tensor: torch.Tensor, og_size: int) -> torch.Tensor:
    """Handles the unpad of the last dimension"""
    if inpt_tensor.size(-1) != og_size:
        return inpt_tensor[..., :og_size]
    return inpt_tensor


def _calculate_scale(head_dim_size: int, scale: float | None) -> float:
    """
    For FlashAttention we pad the head dimension to be a multiple of 8 so we need to scale the output
    by the original head size and not the padded.
    """
```

## 逐行讲解 / What's happening

1. **第 16-17 行 / Lines 16-17**:
   - 中文: 如果输出 shape 和 query 完全一样，直接 `empty_like`，这是最保真的 layout 复制。
   - English: If the output shape exactly matches the query, `empty_like` preserves layout as directly as possible.
1. **第 19-22 行 / Lines 19-22**:
   - 中文: 它按 stride 从小到大排序维度；stride 小的维度在内存里变化更快。stride 为 0 的广播维度放到最后处理。
   - English: Dimensions are sorted by stride from small to large; smaller stride means faster movement in memory. Zero-stride broadcast dimensions are handled last.
1. **第 23-28 行 / Lines 23-28**:
   - 中文: 新 shape 下重新累乘 stride，再用 `empty_strided` 分配一个“维度顺序像 query”的空 tensor。
   - English: It rebuilds cumulative strides for the new shape and allocates an `empty_strided` tensor whose dimension order resembles the query.
1. **第 36-47 行 / Lines 36-47**:
   - 中文: 后两个 helper 展示同一个文件的 attention 细节：FlashAttention 可能 pad head dim，后处理和 scale 都要记住原始大小。
   - English: The later helpers show related attention details: FlashAttention may pad head dim, so postprocessing and scaling must remember the original size.

## 类比 / The analogy

像搬家时房间面积变了，但你还想保持动线：厨房还是靠入口，书桌还是靠窗。家具数量变了，摆放顺序别乱。

It is like moving into a differently sized apartment while keeping the same traffic flow: kitchen near the entrance, desk by the window. The dimensions changed, but the layout order should not.

## 自己跑一遍 / Try it yourself

```python
import math

def matching_strides(old_strides, new_shape):
    order = sorted(range(len(old_strides)), key=lambda i: old_strides[i] or math.inf)
    strides = [0] * len(order)
    stride = 1
    for i in order:
        strides[i] = stride
        stride *= new_shape[i]
    return tuple(strides)

print(matching_strides((12, 1, 4), (2, 5, 3)))
print(matching_strides((0, 1, 8), (7, 4, 2)))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
(15, 1, 5)
(8, 1, 4)
```

第一个例子保留“第 1 维最快、第 2 维其次、第 0 维最慢”的顺序；第二个例子把 zero-stride 维度放到最后。

The first example preserves the order "dim 1 fastest, dim 2 next, dim 0 slowest." The second pushes the zero-stride dimension to the end.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **channels-last convolution** / **channels-last convolution**: 输出 layout 常常跟输入或 weight 的高效路径保持一致。 / Output layout often follows the efficient path implied by inputs or weights.
- **FlashAttention unpadding** / **FlashAttention unpadding**: kernel 内部可以 pad，公共 API 返回时要恢复用户期待的尺寸和语义。 / Kernels can pad internally, but public outputs must restore the shape and semantics users expect.

## 注意事项 / Caveats / when it breaks

- **layout 不是语义** / **Layout is not semantics**: stride 顺序只影响存储和性能，不应该改变数值含义。 / Stride order affects storage and performance, not value semantics.
- **zero stride 要小心** / **Zero strides need care**: 广播视图的 stride 不能照搬到新分配输出里，所以这里把它排到最后。 / Broadcast strides should not be copied blindly into a fresh output, so this helper orders them last.

## 延伸阅读 / Further reading

- [PyTorch tensor views](https://pytorch.org/docs/stable/tensor_view.html)
- [PyTorch scaled dot product attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
