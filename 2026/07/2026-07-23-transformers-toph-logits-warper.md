---
date: 2026-07-23
topic: huggingface
source: huggingface
repo: huggingface/transformers
file: src/transformers/generation/logits_process.py
permalink: https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py
difficulty: intermediate
read_time: ~8 min
tags: [code-of-the-day, huggingface, transformers, generation, sampling]
---

# Transformers Top-H：用熵决定保留多少候选 token / Transformers Top-H: Use Entropy to Decide How Many Tokens Survive

> **一句话 / In one line**: `TopHLogitsWarper` 先取前 `top_n` 个 token，再按累计熵阈值保留候选，让采样集合随分布置信度自动变大或变小。 / `TopHLogitsWarper` first takes the top `top_n` tokens, then keeps candidates under a cumulative-entropy threshold so the sampling set adapts to confidence.

## 为什么重要 / Why this matters

Top-k 固定保留 k 个，top-p 固定保留概率质量。Top-H 的思路是：如果分布很尖，就少保留；如果分布本来很不确定，就允许更多候选。这对语言模型、动作 token、离散 planner 都是同一个采样问题。

Top-k keeps a fixed count, and top-p keeps a fixed probability mass. Top-H instead asks: if the distribution is sharp, keep fewer tokens; if it is uncertain, allow more. The same sampling problem appears in LMs, action tokens, and discrete planners.

## 代码 / The code

`huggingface/transformers` — [`generation/logits_process.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py)

```python
# Simplified teaching slice.
def top_h_filter(logits, top_h=0.4, top_n=100):
    top_logits, top_ids = topk(logits, k=min(top_n, len(logits)))
    probs = softmax(top_logits)
    entropy = -sum(p * log(p) for p in probs)
    threshold = entropy * top_h

    keep = []
    running = 0.0
    for token_id, p in zip(top_ids, probs):
        running += -p * log(p)
        keep.append(token_id)
        if running > threshold:
            break
    return keep
```

## 逐行讲解 / What's happening

1. **先限制候选上限 / Cap the candidate pool first**: 中文: `top_n` 是效率护栏，避免对完整词表做复杂统计。 English: `top_n` is an efficiency guardrail so the full vocabulary is not analyzed.
2. **熵来自当前分布 / Entropy comes from this distribution**: 中文: 阈值不是常数，而是 `H(p) * top_h`。 English: the threshold is not fixed; it is `H(p) * top_h`.
3. **按概率顺序累积熵 / Accumulate entropy in probability order**: 中文: 从最可信 token 开始，直到累计不确定性足够。 English: it starts from the most likely token and stops when enough uncertainty has been covered.
4. **至少保留第一名 / Always keep the best token**: 中文: 即使阈值很小，也不会把采样集合清空。 English: even with a tiny threshold, the best token survives.

## 类比 / The analogy

像开会决定候选方案。问题很明确时，只让第一第二方案进下一轮；问题很模糊时，多留几个方案，避免过早拍板。

It is like shortlisting options in a meeting. If the answer is obvious, keep only one or two; if the situation is fuzzy, keep more options alive.

## 自己跑一遍 / Try it yourself

```python
import math

probs = [0.70, 0.20, 0.07, 0.03]
entropy = -sum(p * math.log(p) for p in probs)
threshold = entropy * 0.4
running = 0.0
keep = []
for i, p in enumerate(probs):
    running += -p * math.log(p)
    keep.append(i)
    if running > threshold:
        break
print(round(entropy, 3), keep)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
0.863 [0, 1]
```

## 注意事项 / Caveats / when it breaks

- **先排序再算 / Sort before accumulating**: 不按概率排序会把低概率 token 过早纳入候选。 / without sorting, low-probability tokens can enter too early.
- **`top_h` 不是 top-p / `top_h` is not top-p**: 它控制熵比例，不是概率质量比例。 / it controls an entropy fraction, not probability mass.
- **动作 token 也适用 / Action tokens can use it too**: 离散动作头可以用同样逻辑在确定与探索之间切换。 / discrete action heads can use the same confidence-to-exploration tradeoff.

## 延伸阅读 / Further reading

- [Transformers logits_process.py](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py)
