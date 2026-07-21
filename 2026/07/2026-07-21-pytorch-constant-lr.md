---
date: 2026-07-21
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/optim/lr_scheduler.py
permalink: https://github.com/pytorch/pytorch/blob/main/torch/optim/lr_scheduler.py#L3492-L3600
difficulty: beginner
read_time: ~8 min
tags: [code-of-the-day, pytorch, lr-scheduler, constantlr, warmup]
---

# PyTorch ConstantLR：先打折，到了步数再还原 / PyTorch ConstantLR: Discount First, Restore at the Milestone

> **一句话 / In one line**: `ConstantLR` 在第 0 步把当前学习率乘上 `factor`，中间保持不动，到 `total_iters` 再乘回 `1 / factor`。 / `ConstantLR` multiplies the current learning rate by `factor` at step 0, leaves it unchanged in the middle, then restores it with `1 / factor` at `total_iters`.

## 为什么重要 / Why this matters

很多人以为 warmup scheduler 每一步都会改学习率。`ConstantLR` 更像一个短期开关：先把学习率降到固定比例，训练若干步后一次性恢复。这个设计也解释了为什么它同时实现递推路径和闭式路径。

Many people expect a warmup scheduler to change the learning rate every step. `ConstantLR` is more like a temporary switch: lower the rate to a fixed fraction, keep it there, then restore it once. That also explains why it implements both chainable and closed-form paths.

## 代码 / The code

`pytorch/pytorch` — [`torch/optim/lr_scheduler.py`](https://github.com/pytorch/pytorch/blob/main/torch/optim/lr_scheduler.py#L3492-L3600)

```python
# Simplified teaching slice, not a verbatim copy.
if factor < 0 or factor > 1:
    raise ValueError("factor must be in [0, 1]")

def get_lr(last_epoch, lr, factor, total_iters):
    if last_epoch == 0:
        return lr * factor
    if last_epoch != total_iters:
        return lr
    return lr / factor

def closed_form(base_lr, last_epoch, factor, total_iters):
    active = last_epoch < total_iters
    return base_lr * (factor if active else 1.0)
```

## 逐行讲解 / What's happening

1. **factor 被限制在 0 到 1 / Factor is bounded to 0..1**: 中文: 这个 scheduler 只负责降低或保持学习率。 English: this scheduler only lowers or preserves the learning rate.
2. **第 0 步打折 / Step 0 discounts**: 中文: 第一次 `step()` 把 optimizer 当前 lr 乘上 `factor`。 English: the first `step()` multiplies the optimizer's current lr by `factor`.
3. **中间不动 / Middle steps do nothing**: 中文: 只要没到 `total_iters`，直接返回当前 lr。 English: until `total_iters`, it returns the current lr unchanged.
4. **里程碑还原 / Milestone restores**: 中文: 到点后乘 `1 / factor`，把之前的折扣撤销。 English: at the milestone, multiplying by `1 / factor` cancels the earlier discount.
5. **闭式公式看 base_lr / Closed form uses base_lr**: 中文: 传 epoch 时不用依赖当前 lr，而是从初始 lr 直接算。 English: when an epoch is passed, PyTorch can compute from the base lr directly.

## 类比 / The analogy

像试营业优惠券：开门第一天全场五折，接下来几天保持五折，到活动结束恢复原价。它不是每天线性涨价。

It is like an opening-week coupon: half price on day one, the same half price for a few days, then back to normal. It is not a daily linear ramp.

## 自己跑一遍 / Try it yourself

```python
def constant_lr(base_lr, factor, total_iters, epochs):
    lr = base_lr
    out = []
    for epoch in range(epochs):
        if epoch == 0:
            lr *= factor
        elif epoch == total_iters:
            lr *= 1.0 / factor
        out.append(round(lr, 4))
    return out

print(constant_lr(0.1, factor=0.2, total_iters=3, epochs=6))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[0.02, 0.02, 0.02, 0.1, 0.1, 0.1]
```

## 注意事项 / Caveats / when it breaks

- **factor 不能是 0 再还原 / Zero cannot be restored**: 真正用 PyTorch 时避免把 factor 设成 0，否则还原步没有意义。 / Avoid `factor=0` in real use if you expect a restoration step.
- **它不是 LinearLR / It is not LinearLR**: 想逐步升温应该用线性或余弦 scheduler。 / Use a linear or cosine scheduler for gradual warmup.
- **组合 scheduler 时注意顺序 / Order matters when composing schedulers**: 它会和外部 lr 改动同时作用。 / It composes with other lr changes applied outside it.

## 延伸阅读 / Further reading

- [PyTorch lr_scheduler.py ConstantLR](https://github.com/pytorch/pytorch/blob/main/torch/optim/lr_scheduler.py#L3492-L3600)
- [PyTorch ConstantLR docs](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ConstantLR.html)
