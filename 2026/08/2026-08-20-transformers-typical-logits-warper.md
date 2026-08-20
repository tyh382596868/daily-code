---
date: 2026-08-20
topic: huggingface
source: huggingface
repo: huggingface/transformers
file: src/transformers/generation/logits_process.py
permalink: https://github.com/huggingface/transformers/blob/94f09cfec149050b5355bab7f207ac69e21f1a02/src/transformers/generation/logits_process.py#L780-L865
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, decoding, typical-sampling]
---

# Transformers typical sampling：挑接近熵的 token / Transformers Typical Sampling: Keep Tokens Near Entropy

> **一句话 / In one line**: `TypicalLogitsWarper` 不只看概率大小，而是保留 surprise 接近分布熵的 token，再按累计概率截断。 / `TypicalLogitsWarper` does not keep tokens by probability alone; it keeps tokens whose surprise is close to entropy, then truncates by cumulative mass.

## 为什么重要 / Why this matters

普通 top-p 往往保留最可能的 token。typical sampling 的目标不一样：它问“这个 token 的意外程度是否接近当前分布的平均意外程度”。这会让模型避开过于机械的最高概率续写，也避开太离谱的低概率词。

Plain top-p keeps the most probable tokens. Typical sampling asks a different question: is this token's surprise close to the distribution's average surprise? That can avoid both mechanical highest-probability continuations and wildly unlikely words.

## 代码 / The code

`huggingface/transformers` — [`src/transformers/generation/logits_process.py`](https://github.com/huggingface/transformers/blob/94f09cfec149050b5355bab7f207ac69e21f1a02/src/transformers/generation/logits_process.py#L780-L865)

```python
class TypicalLogitsWarper(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that performs typical decoding. Inspired on how humans use language, it prioritizes tokens
    whose log probability is close to the entropy of the token probability distribution. This means that the most
    likely tokens may be discarded in the process.
    """

    def __init__(self, mass: float = 0.9, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        mass = float(mass)
        if not (mass > 0 and mass < 1):
            raise ValueError(f"`typical_p` has to be a float > 0 and < 1, but is {mass}")
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
            raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}")

        self.filter_value = filter_value
        self.mass = mass
        self.min_tokens_to_keep = min_tokens_to_keep

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # calculate entropy
        normalized = torch.nn.functional.log_softmax(scores, dim=-1)
        p = torch.exp(normalized)
        ent = -(normalized * p).nansum(-1, keepdim=True)

        # shift and sort
        shifted_scores = torch.abs((-normalized) - ent)
        sorted_scores, sorted_indices = torch.sort(shifted_scores, descending=False)
        sorted_logits = scores.gather(-1, sorted_indices)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative mass above the threshold
        last_ind = (cumulative_probs < self.mass).sum(dim=1)
        last_ind.clamp_(max=sorted_scores.shape[-1] - 1)
        sorted_indices_to_remove = sorted_scores > sorted_scores.gather(1, last_ind.view(-1, 1))
        sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

        scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores_processed
```

## 逐行讲解 / What's happening

1. **第 833-842 行 / Lines 833-842 (`__init__`)**:
   - 中文: `mass` 是 typical set 的累计概率预算，`min_tokens_to_keep` 防止筛到空集合。
   - English: `mass` is the probability budget for the typical set, and `min_tokens_to_keep` prevents filtering everything.
2. **第 846-849 行 / Lines 846-849 (entropy)**:
   - 中文: 先把 logits 变成 log probability，再计算当前分布的熵。
   - English: The code converts logits to log probabilities and computes the current distribution entropy.
3. **第 852-855 行 / Lines 852-855 (typical distance)**:
   - 中文: `-log p` 是 surprise；它和熵越近，token 越“典型”。
   - English: `-log p` is surprise. The closer it is to entropy, the more typical the token is.
4. **第 858-865 行 / Lines 858-865 (masking)**:
   - 中文: 按 typical distance 排序后用累计概率截断，再 scatter 回原词表顺序并填成 `-inf`。
   - English: After sorting by typical distance, it truncates by cumulative probability, scatters the mask back to vocab order, and fills removed logits with `-inf`.

## 类比 / The analogy

像选会议发言。不是只选最大声的人，也不是随机找角落里的人，而是找“信息量刚好代表这场讨论”的发言者。

It is like choosing a conference speaker. You do not always pick the loudest person or a random person in the corner; you pick someone whose contribution is representative of the discussion's information level.

## 自己跑一遍 / Try it yourself

```python
import math

probs = [0.55, 0.25, 0.15, 0.05]
entropy = -sum(p * math.log(p) for p in probs)
surprise = [-math.log(p) for p in probs]
ranked = sorted(range(len(probs)), key=lambda i: abs(surprise[i] - entropy))

kept, mass = [], 0.0
for i in ranked:
    kept.append(i); mass += probs[i]
    if mass >= 0.8: break
print("entropy", round(entropy, 3))
print("kept token ids", kept)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
entropy 1.11
kept token ids [1, 0]
```

中文: 最高概率 token `0` 不一定第一个保留，因为它的 surprise 可能低于当前熵太多。

English: The highest-probability token `0` is not necessarily kept first because its surprise may be too far below entropy.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Top-p sampling** / **Top-p sampling**: 也用累计概率截断，但排序依据是概率而不是 typical distance。 / It also truncates by cumulative probability, but sorts by probability rather than typical distance.
- **Entropy-aware filters** / **Entropy-aware filters**: eta sampling 也根据熵动态改变过滤阈值。 / Eta sampling also uses entropy to adapt its filtering threshold.

## 注意事项 / Caveats / when it breaks

- **需要采样模式** / **Sampling is required**: 过滤 logits 之后还要从剩余 token 里采样，贪心解码会削弱它的意义。 / After filtering, you still need sampling; greedy decoding weakens the point.
- **小 `mass` 会很激进** / **Small `mass` is aggressive**: `typical_p` 太低可能删掉自然但稍微偏离熵的 token。 / Too small a `typical_p` may remove natural tokens that are only slightly off the entropy target.

## 延伸阅读 / Further reading

- [Transformers `TypicalLogitsWarper`](https://github.com/huggingface/transformers/blob/94f09cfec149050b5355bab7f207ac69e21f1a02/src/transformers/generation/logits_process.py#L780-L865)
- [Typical Decoding for Natural Language Generation](https://huggingface.co/papers/2202.00666)
