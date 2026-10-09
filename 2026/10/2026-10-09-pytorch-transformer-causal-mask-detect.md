---
date: 2026-10-09
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/nn/modules/transformer.py
permalink: https://github.com/pytorch/pytorch/blob/83441a98cd065f2a334a0b8c70bf0c00f0e4cd2e/torch/nn/modules/transformer.py#L1213-L1253
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, attention-mask]
---
# PyTorch causal mask：先尊重 hint，再验证 mask / PyTorch Causal Mask Detection: Trust the Hint, Otherwise Compare the Mask

> **一句话 / In one line**: `is_causal` 明说了就直接信；没说时，才把传入 mask 和标准上三角 causal mask 做一次形状匹配。 / If `is_causal` is explicit, PyTorch trusts it; otherwise it compares the mask to the canonical causal mask.

## 为什么重要 / Why this matters

Transformer attention 有一个很现实的性能分叉：如果系统知道 mask 是 causal，就可以走更专门的 kernel；如果误判，又可能得到错误注意力。这里的函数把 API 语义处理得很谨慎：用户给了 hint 就尊重 hint，没给时才根据实际 mask 推断。

Transformer attention has a performance-sensitive branch: a causal mask can unlock specialized kernels, but a wrong hint can change semantics. This helper keeps the rule explicit. User intent wins when provided; only missing intent triggers mask inspection.

## 代码 / The code

`pytorch/pytorch` — [`torch/nn/modules/transformer.py`](https://github.com/pytorch/pytorch/blob/83441a98cd065f2a334a0b8c70bf0c00f0e4cd2e/torch/nn/modules/transformer.py#L1213-L1253)

```python
def _detect_is_causal_mask(
    mask: Tensor | None,
    is_causal: bool | None = None,
    size: int | None = None,
) -> bool:
    """Return whether the given attention mask is causal.

    Warning:
    If ``is_causal`` is not ``None``, its value will be returned as is.  If a
    user supplies an incorrect ``is_causal`` hint,

    ``is_causal=False`` when the mask is in fact a causal attention.mask
       may lead to reduced performance relative to what would be achievable
       with ``is_causal=True``;
    ``is_causal=True`` when the mask is in fact not a causal attention.mask
       may lead to incorrect and unpredictable execution - in some scenarios,
       a causal mask may be applied based on the hint, in other execution
       scenarios the specified mask may be used.  The choice may not appear
       to be deterministic, in that a number of factors like alignment,
       hardware SKU, etc influence the decision whether to use a mask or
       rely on the hint.
    ``size`` if not None, check whether the mask is a causal mask of the provided size
       Otherwise, checks for any causal mask.
    """
    # Prevent type refinement
    make_causal = is_causal is True

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype
        )

        # Do not use `torch.equal` so we handle batched masks by
        # broadcasting the comparison.
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal
```

## 逐行讲解 / What's happening

1. **第 1238 行 / Line 1238 (`make_causal`)**:
   - 中文: 先把 `is_causal is True` 保存成布尔值，避免后续类型细化影响逻辑。
   - English: The function first records the explicit `True` hint as a boolean, keeping later logic simple.
2. **第 1240-1244 行 / Lines 1240-1244 (canonical mask)**:
   - 中文: 只有 hint 缺失且 mask 存在时，才生成同尺寸的标准 causal mask。
   - English: The canonical causal mask is generated only when the hint is missing and a mask is present.
3. **第 1246-1251 行 / Lines 1246-1251 (comparison)**:
   - 中文: 代码不用 `torch.equal`，而是元素比较后 `.all()`，这样能保持广播语义更明确。
   - English: Instead of `torch.equal`, PyTorch uses elementwise comparison and `.all()`, which keeps broadcasting behavior under control.

## 类比 / The analogy

像机场安检的快速通道。登机牌上明确写了“已预检”，工作人员按这个标记放行；没有标记时，才现场检查行李是否符合快速通道规则。

It is like an airport precheck lane. If the boarding pass clearly says prechecked, the agent follows that signal. If not, they inspect the bag against the fast-lane rules.

## 自己跑一遍 / Try it yourself

```python
def detect_is_causal(mask, is_causal=None):
    make_causal = is_causal is True
    if is_causal is None and mask is not None:
        size = len(mask)
        expected = [[0 if j <= i else float("-inf") for j in range(size)] for i in range(size)]
        make_causal = mask == expected
    return make_causal

causal = [[0, float("-inf"), float("-inf")], [0, 0, float("-inf")], [0, 0, 0]]
print(detect_is_causal(causal))
print(detect_is_causal([[0, 0], [0, 0]]))
print(detect_is_causal([[0, 0], [0, 0]], is_causal=True))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
True
False
True
```

第三行说明 hint 的优先级最高，即使 mask 本身不像 causal mask。

The third line shows that an explicit hint has priority, even when the mask itself does not look causal.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **FlashAttention dispatch** / **FlashAttention dispatch**: kernel 选择经常依赖 mask 语义。 / Kernel dispatch often depends on mask semantics.
- **生成式解码** / **Autoregressive decoding**: causal hint 能让框架避免重复检查每个 step 的 mask。 / Causal hints help frameworks avoid rechecking masks at every generation step.

## 注意事项 / Caveats / when it breaks

- **hint 错了会很危险** / **Wrong hints are dangerous**: 源码注释明确警告，错误的 `is_causal=True` 可能导致不可预测行为。 / The source warns that an incorrect `is_causal=True` can produce unpredictable execution.
- **形状必须匹配** / **Shape must match**: 形状不等时直接判 False。 / A mismatched shape returns False.

## 延伸阅读 / Further reading

- [PyTorch Transformer mask helper](https://github.com/pytorch/pytorch/blob/83441a98cd065f2a334a0b8c70bf0c00f0e4cd2e/torch/nn/modules/transformer.py#L1213-L1253)
- [PyTorch Transformer docs](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)
