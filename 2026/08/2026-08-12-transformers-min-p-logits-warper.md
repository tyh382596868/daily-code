---
date: 2026-08-12
topic: huggingface
source: huggingface
repo: huggingface/transformers
file: src/transformers/generation/logits_process.py
permalink: https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, huggingface, generation]
---

# Transformers min-p：阈值跟着最强 token 走 / Transformers min-p: Let the Threshold Follow the Strongest Token

> **一句话 / In one line**: `MinPLogitsWarper` 不是保留固定数量或固定累计概率，而是保留概率至少达到 `min_p * top_prob` 的 token。 / `MinPLogitsWarper` does not keep a fixed count or fixed cumulative mass; it keeps tokens whose probability reaches `min_p * top_prob`.

## 为什么重要 / Why this matters

top-k 和 top-p 都是常见采样截断，但它们不直接表达“模型很自信时就少探索”。min-p 的阈值和当前最强 token 的概率绑定：头部越尖，阈值越高；分布越平，阈值越低。这让采样策略能随模型置信度自动收缩或放松。

Top-k and top-p are common truncation rules, but neither directly says "explore less when the model is confident." Min-p ties the threshold to the current top token probability: sharper distributions get a higher cutoff, flatter distributions get a lower cutoff. The sampler adapts to confidence.

## 代码 / The code

`huggingface/transformers` — [`src/transformers/generation/logits_process.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py)

```python
class MinPLogitsWarper(LogitsProcessor):
    def __init__(self, min_p: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        if not (0 <= min_p <= 1.0):
            raise ValueError(f"`min_p` has to be a float in the [0, 1] interval, but is {min_p}")
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
            raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}")

        self.min_p = min_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Convert logits to probabilities
        probs = torch.softmax(scores, dim=-1)

        # Get the probability of the top token for each sequence in the batch
        top_probs = probs.amax(dim=-1, keepdim=True)

        # Calculate the actual min_p threshold by scaling min_p with the top token's probability
        scaled_min_p = self.min_p * top_probs

        # Create a mask for tokens that have a probability less than the scaled min_p
        tokens_to_remove = probs < scaled_min_p

        # Keep at least min_tokens_to_keep tokens (clip k to vocab size if needed, avoids index out of range)
        k = min(self.min_tokens_to_keep, probs.shape[-1])

        sorted_indices = torch.topk(probs, k, dim=-1).indices
        tokens_to_remove.scatter_(-1, sorted_indices, False)

        scores_processed = scores.masked_fill(tokens_to_remove, self.filter_value)
        return scores_processed
```

## 逐行讲解 / What's happening

1. **初始化 / Initialization**:
   - 中文: `min_p` 限定在 `[0, 1]`，`min_tokens_to_keep` 至少为 1，保证极端分布下也不会把全部候选删光。
   - English: `min_p` is constrained to `[0, 1]`, and `min_tokens_to_keep` is at least 1 so an extreme distribution cannot delete every candidate.
2. **softmax / softmax**:
   - 中文: 这里必须转成概率，因为 min-p 的阈值是概率比值，不是 logit 差值。
   - English: The code converts to probabilities because min-p is a probability ratio, not a raw-logit difference.
3. **自适应阈值 / Adaptive threshold**:
   - 中文: `scaled_min_p = min_p * top_probs` 让每个 batch 样本有自己的 cutoff。
   - English: `scaled_min_p = min_p * top_probs` gives each batch row its own cutoff.
4. **强制保底 / Minimum keep**:
   - 中文: `topk` 找到必须保留的 token，再用 `scatter_` 把这些位置从删除 mask 里抹掉。
   - English: `topk` finds tokens that must survive, and `scatter_` clears those positions from the removal mask.

## 类比 / The analogy

像班级考试划线：不是固定保留前 10 名，而是“分数至少达到第一名的 20%”。第一名很高时线也高，大家分差不大时线就低。

It is like setting a classroom cutoff as "at least 20% of the top score" instead of "top 10 students." If the top score is high, the bar rises; if scores are close, the bar relaxes.

## 自己跑一遍 / Try it yourself

```python
def min_p_keep(probs, min_p, min_keep=1):
    top = max(probs)
    remove = [p < min_p * top for p in probs]
    for i in sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:min_keep]:
        remove[i] = False
    return [i for i, r in enumerate(remove) if not r]

print(min_p_keep([0.70, 0.20, 0.07, 0.03], 0.10))
print(min_p_keep([0.30, 0.27, 0.23, 0.20], 0.10))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[0, 1, 2]
[0, 1, 2, 3]
```

同样是 `min_p=0.10`，尖锐分布删掉尾部，平坦分布几乎全保留。

With the same `min_p=0.10`, a sharp distribution drops the tail, while a flatter distribution keeps almost everything.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Top-p sampling** / **Top-p sampling**: 也按概率过滤，但看累计概率质量。 / It also filters by probability, but uses cumulative mass.
- **Epsilon sampling** / **Epsilon sampling**: 阈值固定，不会跟随当前置信度变化。 / The threshold is fixed and does not adapt to the current confidence.

## 注意事项 / Caveats / when it breaks

- **温度顺序会影响结果** / **Temperature order matters**: 先调温度再做 min-p，和反过来不等价。
- **概率很尖时会很保守** / **Sharp distributions become conservative**: 这是设计目标，但创意写作场景可能需要更低的 `min_p`。

## 延伸阅读 / Further reading

- [Transformers logits processors](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py)
