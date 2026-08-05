---
date: 2026-08-05
topic: huggingface
source: huggingface
repo: huggingface/nanoVLM
file: models/utils.py
permalink: https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/models/utils.py#L27-L51
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, nanovlm, sampling, logits-filtering]
---

# nanoVLM top-k/top-p：采样前先把候选 token 削薄 / nanoVLM top-k/top-p: Thin the Candidate Tokens Before Sampling

> **一句话 / In one line**: `top_k_top_p_filtering` 先保留分数最高的 `k` 个 token，再按累计概率 `p` 截断长尾，并把被删 token 的 logit 设成 `-inf`。 / `top_k_top_p_filtering` first keeps the top `k` tokens, then truncates the probability tail by cumulative mass `p`, setting removed logits to `-inf`.

## 为什么重要 / Why this matters

生成不是直接对全词表无约束采样。低概率 token 太多时，模型会偶尔抽到离谱词；只保留 top-1 又会太死板。top-k 和 top-p 是两种常见折中：前者限制候选数量，后者限制候选总概率质量。nanoVLM 把它们写成一个小函数，在进入 `softmax` 和 `multinomial` 前先清理 logits。

Generation is not usually unconstrained sampling over the whole vocabulary. Too many low-probability tokens cause occasional nonsense; top-1 is too rigid. Top-k and top-p are two common compromises: one limits candidate count, the other limits total probability mass. nanoVLM packages both into a small function that cleans logits before `softmax` and `multinomial`.

## 代码 / The code

`huggingface/nanoVLM` — [`models/utils.py`](https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/models/utils.py#L27-L51)

```python
def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float('Inf')):
    """
    Apply top-k and/or nucleus (top-p) filtering to logits.
    """
    top_k = min(top_k, logits.size(-1))  # Safety

    if top_k > 0:
        # Remove all tokens with a probability less than the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, filter_value)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative probability above top_p
        sorted_indices_to_remove = cumulative_probs > top_p

        # Always keep the first token
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)

    return logits
```

## 逐行讲解 / What's happening

1. **第 31 行 / Line 31 (top-k clamp)**:
   - 中文: `top_k` 先被 clamp 到词表大小，避免用户传一个比 vocab 还大的值导致 `torch.topk` 报错。
   - English: `top_k` is clamped to vocabulary size, preventing `torch.topk` from failing when the requested k is larger than the logits dimension.
2. **第 33-36 行 / Lines 33-36 (hard candidate cap)**:
   - 中文: 第 k 大 logit 是阈值，低于这个阈值的 token 全部改成 `-inf`，后续 softmax 概率就会变成 0。
   - English: The kth largest logit becomes the threshold; every token below it is set to `-inf`, so its later softmax probability becomes zero.
3. **第 38-44 行 / Lines 38-44 (nucleus mass)**:
   - 中文: top-p 先按 logit 降序排序，再对 softmax 概率做累计和，超过 `top_p` 的位置进入删除 mask。
   - English: Top-p sorts logits descending, computes cumulative softmax probability, and marks positions beyond the `top_p` mass for removal.
4. **第 46-49 行 / Lines 46-49 (keep one and scatter back)**:
   - 中文: 最高分 token 永远保留；随后用 `scatter` 把排序后的删除 mask 放回原始词表顺序。
   - English: The best token is always kept; `scatter` then maps the sorted removal mask back to original vocabulary order.

## 类比 / The analogy

这像在菜单上点菜。top-k 是“只看评分最高的 5 道菜”，top-p 是“从最推荐开始往下看，直到推荐概率已经覆盖 90%”。最后你只会从这张短菜单里随机点。

It is like ordering from a menu. Top-k says, "only consider the five highest-rated dishes." Top-p says, "start from the most recommended dishes until they cover 90% of the recommendation mass." You then sample from that shorter menu.

## 自己跑一遍 / Try it yourself

```python
import math

logits = [4.0, 3.0, 2.0, 0.0]
top_k = 2
threshold = sorted(logits, reverse=True)[top_k - 1]
filtered = [x if x >= threshold else -math.inf for x in logits]
print(filtered)

exps = [0 if x == -math.inf else math.exp(x) for x in filtered]
total = sum(exps)
print([round(x / total, 3) for x in exps])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[4.0, 3.0, -inf, -inf]
[0.731, 0.269, 0.0, 0.0]
```

中文: 被过滤的 token 没有被删掉维度，而是变成 `-inf`，这样 shape 不变但概率归零。

English: Filtered tokens are not removed from the tensor; they become `-inf`, preserving shape while zeroing their probability.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Transformers logits warpers** / **Transformers logits warpers**: Hugging Face generation 里也会先改 logits，再统一采样。 / Hugging Face generation also modifies logits first, then samples through one common path.
- **mask-based diffusion decoding** / **mask-based diffusion decoding**: 离散扩散模型也常按置信度过滤候选 token。 / Discrete diffusion models often filter candidate tokens by confidence.

## 注意事项 / Caveats / when it breaks

- **top-p 在 top-k 后执行 / top-p runs after top-k**: 先 top-k 会改变 top-p 看到的分布，两个参数不是完全独立。 / Applying top-k first changes the distribution that top-p sees; the two settings are not independent.
- **至少保留一个 token / Keep at least one token**: 最高分 token 被强制保留，避免所有候选都变成 `-inf`。 / The highest-scoring token is forced to remain so the distribution never becomes all `-inf`.

## 延伸阅读 / Further reading

- nanoVLM `top_k_top_p_filtering`: https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/models/utils.py#L27-L51
