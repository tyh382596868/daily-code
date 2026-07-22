---
date: 2026-07-05
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/optim/adamw.py
permalink: https://github.com/pytorch/pytorch/blob/main/torch/optim/adamw.py
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, pytorch, optimizer, adamw]
---

# PyTorch AdamW：用一个标志把 weight decay 从 Adam 里拆出来 / PyTorch AdamW: One Flag Decouples Weight Decay from Adam

> **一句话 / In one line**: `AdamW` 不是复制一份 Adam，而是复用 Adam 实现并强制 `decoupled_weight_decay=True`。 / `AdamW` does not duplicate Adam; it reuses Adam and forces `decoupled_weight_decay=True`.

## 为什么重要 / Why this matters

优化器很容易变成两份几乎一样的代码。PyTorch 的 `AdamW` 把核心更新逻辑留在 `Adam` / functional `adam` 路径里，只在 wrapper 层锁定“decoupled weight decay”语义。这样 bug fix、foreach/fused/capturable 等后端选项都能共享。

Optimizers are prone to near-duplicate implementations. PyTorch's `AdamW` keeps the core update in the `Adam` / functional `adam` path and locks the "decoupled weight decay" semantics at the wrapper layer. Bug fixes and foreach/fused/capturable backend options are shared.

## 代码 / The code

`pytorch/pytorch` — [`torch/optim/adamw.py`](https://github.com/pytorch/pytorch/blob/main/torch/optim/adamw.py)

```python
class AdamW(Adam):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        betas: tuple[Union[float, Tensor], Union[float, Tensor]] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):
        super().__init__(
            params,
            lr,
            betas,
            eps,
            weight_decay,
            amsgrad,
            foreach=foreach,
            maximize=maximize,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
            decoupled_weight_decay=True,
        )
```

## 逐行讲解 / What's happening

1. **继承 `Adam` / Inherit from `Adam`**: 中文: `AdamW` 没有重写 step 主算法，而是把自己定义成 Adam 的一个语义特例。 / English: `AdamW` does not rewrite the main step algorithm; it defines itself as one semantic variant of Adam.
2. **保留后端参数 / Keep backend knobs**: 中文: `foreach`、`capturable`、`differentiable`、`fused` 原样传下去，所以多张量 kernel 和 CUDA graph 支持不需要再实现一遍。 / English: `foreach`, `capturable`, `differentiable`, and `fused` pass through unchanged, so multi-tensor kernels and CUDA graph support are not reimplemented.
3. **关键只有一行 / The key is one line**: 中文: `decoupled_weight_decay=True` 把权重衰减放到参数更新外侧，而不是混进 Adam 的梯度动量。 / English: `decoupled_weight_decay=True` applies decay outside the Adam gradient moments rather than mixing it into them.

## 类比 / The analogy

像同一台咖啡机做美式和拿铁：水泵、加热、研磨都一样，只是最后加不加奶。`AdamW` 的“加奶开关”就是 decoupled weight decay。

It is like one coffee machine making americano and latte. The pump, heater, and grinder are shared; only the milk step changes. `AdamW`'s milk switch is decoupled weight decay.

## 自己跑一遍 / Try it yourself

```python
import torch

w = torch.tensor([1.0], requires_grad=True)
opt = torch.optim.AdamW([w], lr=0.1, weight_decay=0.1)
(w * 0).sum().backward()
opt.step()
print(round(float(w), 4))
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```text
0.99
```

中文: 梯度是 0，参数仍从 `1.0` 变成 `0.99`，说明 weight decay 是单独作用在参数上的。

English: The gradient is zero, but the parameter still moves from `1.0` to `0.99`, showing that weight decay acts directly on the parameter.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **SGD momentum variants** / **SGD momentum variants**: 中文: 一个 optimizer 通过标志覆盖 momentum、nesterov 等语义。 / English: One optimizer implementation covers momentum and Nesterov variants through flags.
- **PEFT adapter wrappers** / **PEFT adapter wrappers**: 中文: wrapper 改语义，底层 module 仍共享 forward。 / English: The wrapper changes semantics while the base module keeps the shared forward path.

## 注意事项 / Caveats / when it breaks

- **不是 L2 正则的同义词 / Not just L2 regularization**: 中文: AdamW 的 decay 不进入梯度矩估计，这和把 L2 loss 加进目标函数不同。 / English: AdamW decay does not enter the gradient moment estimates, unlike adding an L2 term to the objective.
- **学习率仍然参与 / Learning rate still matters**: 中文: decoupled 不代表独立于 optimizer step；实际缩放仍随 `lr` 走。 / English: Decoupled does not mean independent from the optimizer step; the actual scale still follows `lr`.

## 延伸阅读 / Further reading

- PyTorch `AdamW` source linked above.
- Loshchilov & Hutter, "Decoupled Weight Decay Regularization".
