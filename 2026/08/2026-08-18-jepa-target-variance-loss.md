---
date: 2026-08-18
topic: diffusion
source: tracked
repo: facebookresearch/jepa
file: app/vjepa/train.py
permalink: https://github.com/facebookresearch/jepa/blob/51c59d518fc63c08464af6de585f78ac0c7ed4d5/app/vjepa/train.py#L419-L459
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, world-model]
---

# V-JEPA loss：只预测被遮住的特征，还要保持方差 / V-JEPA Loss: Predict Masked Features, Keep Variance Alive

> **一句话 / In one line**: V-JEPA 的训练步先用 frozen target encoder 生成 masked targets，再让 context encoder + predictor 预测这些 targets，并额外惩罚塌缩的 patch 方差。 / V-JEPA first builds masked targets with a frozen target encoder, then predicts them from context tokens, with an extra penalty when patch variance collapses.

## 为什么重要 / Why this matters

世界模型不一定要重建像素。V-JEPA 学的是“被遮住区域在特征空间里应该是什么”，所以 loss 对齐的是语义 embedding；方差正则则防止 predictor 把所有位置都输出成同一个安全平均值。

A world model does not have to reconstruct pixels. V-JEPA learns what masked regions should be in feature space, so the loss matches semantic embeddings. The variance regularizer keeps the predictor from emitting one bland average vector everywhere.

## 代码 / The code

`facebookresearch/jepa` — [`app/vjepa/train.py`](https://github.com/facebookresearch/jepa/blob/51c59d518fc63c08464af6de585f78ac0c7ed4d5/app/vjepa/train.py#L419-L459)

```python
def forward_target(c):
    with torch.no_grad():
        h = target_encoder(c)
        h = F.layer_norm(h, (h.size(-1),))
        h = apply_masks(h, masks_pred, concat=False)
        return h

def forward_context(c, h):
    z = encoder(c, masks_enc)
    z = predictor(z, h, masks_enc, masks_pred)
    return z

def loss_fn(z, h):
    loss = 0.
    for zi, hi in zip(z, h):
        loss += torch.mean(torch.abs(zi - hi)**loss_exp) / loss_exp
    loss /= len(masks_pred)
    return loss

def reg_fn(z):
    return sum([torch.sqrt(zi.var(dim=1) + 0.0001) for zi in z]) / len(z)
```

## 逐行讲解 / What's happening

1. **第 419-429 行 / Lines 419-429**: 中文: target 分支在 `no_grad` 里运行，先编码整段视频，再只取预测 mask 对应的目标 token。 / English: The target branch runs under `no_grad`, encodes the full video, and keeps only the tokens selected by prediction masks.
2. **第 431-438 行 / Lines 431-438**: 中文: context 分支只看 encoder masks 留下的上下文，再由 predictor 补出目标区域。 / English: The context branch sees only the encoder-mask context, and the predictor fills in the target regions.
3. **第 440-446 行 / Lines 440-446**: 中文: 多组 mask 的误差先逐组累加，再除以 mask 组数，避免某一组独占 loss。 / English: Errors from multiple mask groups are accumulated and averaged so one mask group does not dominate.
4. **第 448-459 行 / Lines 448-459**: 中文: `reg_fn` 看预测 token 在 patch 维度上的标准差，低于 1 的部分会被 ReLU 惩罚。 / English: `reg_fn` measures patch-wise prediction standard deviation, and values below 1 are penalized by ReLU.

## 类比 / The analogy

像老师遮住电影分镜的几格，只给学生看上下文，让学生说出被遮住格子的“剧情语义”，不是逐像素画出来；如果学生每格都答“差不多”，方差正则会扣分。

It is like hiding panels from a storyboard and asking the student to describe their semantic content from context, not redraw pixels. If every hidden panel gets the same answer, the variance term marks it down.

## 自己跑一遍 / Try it yourself

```python
def masked_loss(pred_groups, target_groups, loss_exp=2.0):
    losses = []
    for pred, target in zip(pred_groups, target_groups):
        diffs = [abs(p - t) ** loss_exp / loss_exp for p, t in zip(pred, target)]
        losses.append(sum(diffs) / len(diffs))
    return sum(losses) / len(losses)

def variance_penalty(groups):
    penalties = []
    for values in groups:
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        std = (var + 0.0001) ** 0.5
        penalties.append(max(0.0, 1.0 - std))
    return round(sum(penalties) / len(penalties), 3)

print(masked_loss([[1.0, 2.0], [0.0, 1.0]], [[1.5, 1.5], [0.0, 2.0]]))
print(variance_penalty([[1.0, 1.0, 1.0], [0.0, 2.0, 4.0]]))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
0.1875
0.495
```

第一行是 masked prediction loss；第二行显示完全塌缩的一组会产生更高惩罚。

The first line is the masked prediction loss; the second shows that a collapsed group receives a larger penalty.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **BYOL / EMA teacher** / **BYOL-style teachers**: target 网络不反传，给 online 网络提供稳定目标。 / The target network is not backpropagated through and gives the online network a stable objective.
- **representation anti-collapse** / **anti-collapse regularizers**: 方差、协方差或熵约束常用来避免所有样本落到同一点。 / Variance, covariance, or entropy terms are common ways to avoid one-point representations.

## 注意事项 / Caveats / when it breaks

- **target encoder 要慢更新** / **The target encoder needs slow updates**: 如果 target 和 online 同步太快，目标会变得不稳定。 / If target and online networks move together too quickly, the target becomes unstable.
- **方差不是语义质量** / **Variance is not semantic quality**: 它只能防止塌缩，不能保证预测内容正确。 / It prevents collapse but does not by itself prove semantic correctness.

## 延伸阅读 / Further reading

- [V-JEPA source](https://github.com/facebookresearch/jepa/blob/51c59d518fc63c08464af6de585f78ac0c7ed4d5/app/vjepa/train.py#L419-L459)
