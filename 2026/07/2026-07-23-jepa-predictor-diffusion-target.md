---
date: 2026-07-23
topic: diffusion
source: tracked
repo: facebookresearch/jepa
file: src/models/predictor.py
permalink: https://github.com/facebookresearch/jepa/blob/main/src/models/predictor.py#L762-L1103
difficulty: intermediate
read_time: ~9 min
tags: [code-of-the-day, diffusion, jepa, predictor, masked-targets]
---

# JEPA predictor：先给目标 token 加噪，再让上下文去修复 / JEPA Predictor: Noise the Target Tokens, Then Let Context Repair Them

> **一句话 / In one line**: `VisionTransformerPredictor` 把可见 token 投到 predictor 维度，把目标 token 加噪或替换成 mask token，再把两者拼起来预测被遮住的特征。 / `VisionTransformerPredictor` projects visible tokens into predictor space, noises or masks the target tokens, then concatenates both streams to predict hidden features.

## 为什么重要 / Why this matters

JEPA 不是从像素重建图像，而是在特征空间预测不可见区域。这个 predictor 文件展示了世界模型常见的三件事：上下文只来自可见 token，目标位置仍要带位置信息，训练时可以把目标 token 当作一个小扩散问题来加噪。

JEPA does not reconstruct pixels; it predicts hidden regions in feature space. This predictor shows three common world-model moves: context comes only from visible tokens, target positions still carry position embeddings, and target features can be treated as a small diffusion problem during training.

## 代码 / The code

`facebookresearch/jepa` — [`src/models/predictor.py`](https://github.com/facebookresearch/jepa/blob/main/src/models/predictor.py#L993-L1103)

```python
# Simplified teaching slice, not a verbatim copy.
def diffusion(x, beta_range=(0.5, 1.0), steps=1000):
    alphas = []
    alpha = 1.0
    for i in range(steps):
        beta = beta_range[0] + i * (beta_range[1] - beta_range[0]) / steps
        alpha *= 1.0 - beta
        alphas.append(alpha)
    t = sample_integer(0, steps)
    return alphas[t] ** 0.5 * normalize(x) + (1 - alphas[t]) ** 0.5 * noise_like(x)

def forward(context, target, context_mask, target_mask):
    x = predictor_embed(context)
    x = x + apply_masks(pos_embed, context_mask)
    pred = predictor_embed(target)
    pred = diffusion(pred)
    pred = pred + apply_masks(pos_embed, target_mask)
    return transformer(concat([x, pred]))
```

## 逐行讲解 / What's happening

1. **context 先投影 / Project context first**: 中文: encoder 的输出维度不一定等于 predictor 的工作维度，所以先过一层 `Linear`。 English: encoder features may not match the predictor width, so a `Linear` maps them first.
2. **位置只按 mask 取 / Position is gathered by masks**: 中文: 位置表覆盖全图，但只取 context 或 target 对应的位置。 English: the position table spans the full grid, but only the selected context or target positions are gathered.
3. **目标 token 可加噪 / Targets can be noised**: 中文: 不用固定 mask token 时，代码把 target feature 标准化后按随机噪声步混合。 English: when it does not use a fixed mask token, target features are normalized and mixed with noise at a random step.
4. **拼接后统一建模 / Concatenate, then model jointly**: 中文: transformer 看到的是“可见上下文 + 带位置的目标槽位”。 English: the transformer sees "visible context + positioned target slots" as one sequence.

## 类比 / The analogy

像做填空题：题干是 context，空格的位置是 target mask。训练时还把草稿答案弄脏一点，逼学生根据上下文把它修回来，而不是死记某个固定答案。

It is like a fill-in-the-blank exercise: context is the prompt and target masks are the blanks. Training dirties the draft answer so the model must repair it from context instead of memorizing a fixed token.

## 自己跑一遍 / Try it yourself

```python
def masked(values, indices):
    return [values[i] for i in indices]

pos = ["p0", "p1", "p2", "p3", "p4"]
context = ["c0", "c2"]
target = ["t1", "t3"]

context_seq = list(zip(context, masked(pos, [0, 2])))
target_seq = list(zip(target, masked(pos, [1, 3])))
print(context_seq + target_seq)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[('c0', 'p0'), ('c2', 'p2'), ('t1', 'p1'), ('t3', 'p3')]
```

## 注意事项 / Caveats / when it breaks

- **mask 必须对齐 / Masks must align**: context 和 target mask 的 batch 展开规则错了，预测目标会串位。 / if context and target masks are expanded inconsistently, target positions drift.
- **加噪是在特征空间 / Noise lives in feature space**: 这里不是像素扩散，loss 目标仍是 encoder feature。 / this is not pixel diffusion; the target remains encoder features.
- **位置不能省 / Do not omit positions**: 没有 target 位置，模型只知道“有空格”，不知道“哪个空格”。 / without target positions, the model knows there are blanks but not where they are.

## 延伸阅读 / Further reading

- [JEPA predictor.py](https://github.com/facebookresearch/jepa/blob/main/src/models/predictor.py#L993-L1103)

