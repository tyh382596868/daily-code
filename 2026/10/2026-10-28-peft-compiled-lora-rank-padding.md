---
date: 2026-10-28
topic: huggingface
source: huggingface
repo: huggingface/peft
file: src/peft/utils/hotswap.py
permalink: https://github.com/huggingface/peft/blob/main/src/peft/utils/hotswap.py
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, peft, lora]
---

# PEFT 编译态 LoRA 热替换：先补齐 rank，再换权重 / PEFT Compiled LoRA Hotswap: Pad Rank Before Swapping

> **一句话 / In one line**: PEFT 把较小 LoRA rank 零填充到固定上限，让已编译模型的张量形状不变。 / PEFT zero-pads smaller LoRA ranks to a fixed ceiling so a compiled model keeps the same shapes.

## 为什么重要 / Why this matters

服务中切换 adapter 若改变矩阵形状，`torch.compile` 很可能重编译；固定形状就能只替换数值。

Changing adapter matrix shapes can trigger `torch.compile` recompilation; fixed shapes allow a value-only swap.

## 代码 / The code

`huggingface/peft` — [`hotswap.py`](https://github.com/huggingface/peft/blob/main/src/peft/utils/hotswap.py)

```python
padded = torch.zeros(target_rank, in_features,
                     device=weight.device, dtype=weight.dtype)
padded[:original_rank, :] = weight
new_layer = torch.nn.Linear(in_features, target_rank,
                            bias=lora_module.bias is not None)
new_layer.weight.data = padded
```

## 逐行讲解 / What's happening

先分配最大 rank 的矩阵，再把原矩阵拷到前几行，新增行全零。零行不改变 LoRA 的数学结果，却让后续 adapter 都拥有同一 shape。

It allocates the maximum-rank matrix, copies the original rows, and leaves added rows zero. Zero rows preserve LoRA’s result while giving every future adapter one shape.

## 类比 / The analogy

像预留一排固定大小的储物柜：小箱子占前几个格，换箱子无需重建整面墙。

It is a fixed bank of lockers: a small box uses the first slots, so swapping boxes does not rebuild the wall.

## 自己跑一遍 / Try it yourself

```python
import torch
w = torch.ones(2, 3); target = 4
padded = torch.zeros(target, 3); padded[:w.size(0)] = w
print(padded.sum().item())  # 6.0
```

运行 / Run with `pip install torch && python try.py`。输出 / Expected output: `6.0`，填充没有改变有效权重和。 / Padding does not alter the effective weight sum.

## 注意事项 / Caveats / when it breaks

- **上限 / Ceiling**：新 adapter 的 rank 不能超过预留 `target_rank`。 / A new adapter cannot exceed the reserved `target_rank`.
