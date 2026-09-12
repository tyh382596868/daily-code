---
date: 2026-08-01
topic: huggingface
source: huggingface
repo: huggingface/transformers
file: src/transformers/generation/logits_process.py
permalink: https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L367-L414
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, transformers, generation]
---

# Transformers repetition penalty：重复 token 的分数要按符号处理 / Transformers Repetition Penalty: Penalize Reused Tokens by Score Sign

> **一句话 / In one line**: `RepetitionPenaltyLogitsProcessor` 只改已经出现过的 token，并且负分数乘 penalty、正分数除 penalty。 / `RepetitionPenaltyLogitsProcessor` only changes tokens already generated, multiplying negative scores and dividing positive scores.

## 为什么重要 / Why this matters

生成模型最容易陷入重复短语。这个 processor 的关键不是“把重复 token 一律减掉”，而是按 logit 的正负号处理：已经不想选的 token 变得更不想选，已经想选的 token 被降温。

Generation models easily fall into repeated phrases. The key is not to subtract a constant from repeated tokens; the processor handles score sign: already-unwanted tokens become less likely, while wanted repeated tokens are cooled down.

## 代码 / The code

`huggingface/transformers` — [`src/transformers/generation/logits_process.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L367-L414)

```python
class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, penalty: float):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")
        self.penalty = penalty

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        score = torch.gather(scores, 1, input_ids)

        # if score < 0 then repetition penalty has to be multiplied to reduce the token probabilities
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)

        scores_processed = scores.scatter(1, input_ids, score)
        return scores_processed
```

## 逐行讲解 / What's happening

1. **第 368-371 行 / Lines 368-371 (penalty validation)**:
   - 中文: penalty 必须是正浮点数；小于 1 会鼓励重复，大于 1 会惩罚重复。
   - English: The penalty must be a positive float; below 1 encourages repetition, above 1 penalizes it.
2. **第 374 行 / Line 374 (`gather`)**:
   - 中文: 只取 `input_ids` 里已经出现过的 token 分数，不扫描整个词表做复杂逻辑。
   - English: It gathers scores only for tokens already present in `input_ids`, avoiding complex logic over the full vocabulary.
3. **第 377 行 / Line 377 (`where`)**:
   - 中文: 负分数乘 penalty 会更负；正分数除 penalty 会变小。
   - English: Multiplying a negative score makes it more negative; dividing a positive score makes it smaller.
4. **第 379-380 行 / Lines 379-380 (`scatter`)**:
   - 中文: 处理后的分数写回原 logits 表，其它 token 完全不动。
   - English: The processed scores are scattered back into the original logits table; all other tokens stay untouched.

## 类比 / The analogy

这像会议主持人控制发言：已经讲过的人不是被赶出房间，只是下一轮拿麦克风的优先级被调低。

It is like a meeting moderator: someone who already spoke is not removed from the room, but their priority for the next microphone pass is lowered.

## 自己跑一遍 / Try it yourself

```python
scores = [2.0, -1.0, 0.5, 3.0]
seen = [0, 1, 0]
penalty = 2.0
for token in set(seen):
    s = scores[token]
    scores[token] = s * penalty if s < 0 else s / penalty
print(scores)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[1.0, -2.0, 0.5, 3.0]
```

token `0` 原本很想被选，现在降到一半；token `1` 原本不想被选，现在更不可能。

token `0` was attractive and is cooled down; token `1` was unattractive and becomes even less likely.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **No-repeat ngram** / **No-repeat ngram**: 更强硬，直接把会形成重复 ngram 的 token 置为禁止。 / This is stricter: it forbids tokens that would create repeated n-grams.
- **Top-k/top-p warpers** / **Top-k/top-p warpers**: 先改 logits 分布，再交给采样器抽 token。 / They reshape the logits distribution before the sampler draws a token.

## 注意事项 / Caveats / when it breaks

- **不是去重器** / **Not a deduplicator**: 它降低概率，不保证重复不会发生。 / It lowers probability; it does not guarantee repetition disappears.
- **prompt token 也算** / **Prompt tokens count too**: 输入 prompt 中出现过的 token 也会被 penalty 处理。 / Tokens from the prompt are also considered seen tokens.

## 延伸阅读 / Further reading

- [Transformers repetition penalty source](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L367-L414)

