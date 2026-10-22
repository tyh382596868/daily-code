---
date: 2026-10-22
topic: vla
source: vla
repo: NVIDIA/Isaac-GR00T
file: gr00t/data/types.py
permalink: https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/data/types.py
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, action-head]
build_role: action-head-continuous advanced variant
---

# Isaac GR00T 动作有效位：padding 不是动作 / Isaac GR00T Action Validity: Padding Is Not an Action

> **一句话 / In one line**: 多机器人共用动作 head 时，必须把“没有这个关节”显式标成无效。 / A shared action head must explicitly mark joints an embodiment does not have as invalid.

## 为什么重要 / Why this matters

不同机器人动作维度不同。把短动作补到公共宽度很方便，但若 loss 也学习 padding，模型会把虚构关节当成训练目标。

Embodiments have different action dimensions. Padding to a common width is convenient, but letting loss learn padding turns nonexistent joints into targets.

## 代码 / The code

`NVIDIA/Isaac-GR00T` — [`types.py`](https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/data/types.py)

```python
# conceptual batch contract
actions: torch.Tensor       # [B, H, A_max]
action_mask: torch.BoolTensor  # [B, H, A_max], True only for real dimensions
loss = ((prediction - actions).square() * action_mask).sum() / action_mask.sum()
```

## 逐行讲解 / What's happening

1. **公共形状 / Common shape**：`A_max` 让不同 embodiment 能进同一 batch。 / `A_max` makes heterogeneous embodiments batchable.
2. **乘 mask / Multiply the mask**：无效位既不贡献误差，也不贡献梯度。 / Invalid entries contribute neither error nor gradient.

## 类比 / The analogy

像统一答题卡上有些人没有选修题：空格是版式，不是答错的答案。 / Like optional questions on a common answer sheet: blank slots are layout, not wrong answers.

## 在 nanoVLA 中的位置 / Where this lives in your nanoVLA

这是 `action-head-continuous` 的先进变体，输入是 conditioned action prediction 与 embodiment metadata，输出是 masked loss 或可执行动作。没有它，多 embodiment 训练会偏向 padding 最多的机器人；生产系统还要验证 mask、单位和控制频率。

This is an advanced `action-head-continuous` variant. It consumes predictions and embodiment metadata and produces masked loss or executable actions. Production versions also validate masks, units, and control rates.

## 自己跑一遍 / Try it yourself

```python
import torch
p = torch.tensor([[1., 9.]]); y = torch.tensor([[2., 0.]])
m = torch.tensor([[1., 0.]])
print((((p-y)**2*m).sum()/m.sum()).item())
```

运行 / Run with: `pip install torch && python try.py`  
预期输出 / Expected output: `1.0`；padding 的 `9` 没有影响。 / `1.0`; padded `9` has no effect.
