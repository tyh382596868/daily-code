---
date: 2026-10-16
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/nn/utils/parametrize.py
permalink: https://github.com/pytorch/pytorch/blob/bc27bc72253e2e344c9dfeda981b9d514c55b7b0/torch/nn/utils/parametrize.py#L450-L540
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, pytorch, parametrization]
---

# PyTorch 参数化注册：不替换 weight，替换读取方式 / PyTorch Parametrization Registration: Change How Weight Is Read, Not Where It Lives

> **一句话 / In one line**: `register_parametrization` 让 `module.weight` 在访问时经过可微变换，同时保留原始可训练参数。 / `register_parametrization` transforms `module.weight` on access while preserving the original trainable parameter.

## 为什么重要 / Why this matters

谱归一化、正交约束和 LoRA 类技巧都需要“参数仍可优化，但每次使用要满足约束”。PyTorch 把这个变成模块层的通用协议，而不是在每个 `forward` 里手写补丁。

Spectral normalization, orthogonality, and LoRA-like techniques need a parameter that remains optimizable but is constrained whenever used. PyTorch makes that a reusable module protocol instead of a per-forward patch.

## 代码 / The code

`pytorch/pytorch` — [`parametrize.py`](https://github.com/pytorch/pytorch/blob/bc27bc72253e2e344c9dfeda981b9d514c55b7b0/torch/nn/utils/parametrize.py#L450-L540)

```python
def register_parametrization(module, tensor_name, parametrization, *, unsafe=False):
    # The first registration creates module.parametrizations.<tensor_name>.
    # Accessing module.weight returns parametrization(original).
    # Gradients flow through that transformation to the original parameter.
    ...
```

## 逐行讲解 / What's happening

1. 中文: API 接收宿主模块、属性名和一个 `nn.Module` 变换；它不要求调用方改写原来的 `forward`。
   English: The API takes a host, attribute name, and transforming `nn.Module`; callers do not rewrite the original `forward`.
2. 中文: 原始张量放在 `module.parametrizations.weight.original`，访问 `module.weight` 时才计算约束后的视图。
   English: The original lives at `module.parametrizations.weight.original`; the constrained view is computed when `module.weight` is read.
3. 中文: 默认安全检查会验证形状和 dtype 契约；`unsafe=True` 是显式跳过这层防护。
   English: Default checks validate shape and dtype contracts; `unsafe=True` explicitly skips that protection.

## 类比 / The analogy

像给水龙头加滤芯：管道里的原水仍在，但每次放水都会先经过同一个过滤器。

It is a faucet filter: the original water remains in the pipe, but every pour passes through the same filter.

## 自己跑一遍 / Try it yourself

```python
class Weight:
    def __init__(self, value): self.original = value
    @property
    def value(self): return max(0, self.original)

w = Weight(-3)
print(w.original, w.value)
w.original = 5
print(w.original, w.value)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
-3 0
5 5
```

中文: 学习对象和使用对象分离，正是参数化的核心。  
English: Separating the learned object from the used object is the heart of parametrization.

## 注意事项 / Caveats / when it breaks

- 中文: 注册后新增参数，已创建的 optimizer 可能需要补充 param group。 / English: Registration can add parameters, so an existing optimizer may need another param group.
- 中文: 变换必须能在热路径承受其计算成本。 / English: The transform must be affordable on the hot path.

## 延伸阅读 / Further reading

- [PyTorch parametrizations](https://github.com/pytorch/pytorch/blob/bc27bc72253e2e344c9dfeda981b9d514c55b7b0/torch/nn/utils/parametrize.py#L450-L540)
