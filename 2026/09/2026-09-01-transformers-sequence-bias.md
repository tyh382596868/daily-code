---
date: 2026-09-01
topic: huggingface
source: huggingface
repo: huggingface/transformers
file: src/transformers/generation/logits_process.py
permalink: https://github.com/huggingface/transformers/blob/899f55f8e41654a106a414e93a6f5fd8f3b8bf37/src/transformers/generation/logits_process.py#L1276-L1319
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, transformers, generation, logits]
---

# Transformers sequence bias：短语偏好落到最后一个 token / Transformers Sequence Bias: Apply Phrase Preference to the Final Token

> **一句话 / In one line**: `SequenceBiasLogitsProcessor` 只在上下文已经匹配短语前缀时，给能完成该短语的最后一个 token 加分或扣分。 / `SequenceBiasLogitsProcessor` adds or subtracts score from the final token only when the current context already matches the phrase prefix.

## 为什么重要 / Why this matters

生成控制不一定要改模型权重。很多时候你只想让模型更偏向某些词组、远离某些词组，或者把品牌名、工具名、格式片段稍微推一把。Transformers 把这件事放在 logits processor 层：模型给出原始分数后，processor 按当前上下文改分，再交给采样或 beam search。

Generation control does not always require changing model weights. Often you only need to nudge the model toward or away from phrases, names, or format fragments. Transformers places this at the logits-processor layer: the model emits raw scores, the processor adjusts them from the current context, then sampling or beam search continues.

## 代码 / The code

`huggingface/transformers` — [`src/transformers/generation/logits_process.py`](https://github.com/huggingface/transformers/blob/899f55f8e41654a106a414e93a6f5fd8f3b8bf37/src/transformers/generation/logits_process.py#L1276-L1319)

```python
def __init__(self, sequence_bias: list[list[list[int] | float]]):
    # After _convert_list_arguments_into_dict(), becomes dict[tuple[int, ...], float]
    self.sequence_bias: Any = sequence_bias
    self._validate_arguments()
    self._convert_list_arguments_into_dict()

    # Bias variables that will be populated on the first call (for retrocompatibility purposes, the vocabulary size
    # is inferred in the first usage, which inhibits initializing here)
    self.length_1_bias = None
    self.prepared_bias_variables = False

@add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    # 1 - Prepares the bias tensors. This is only needed the first time the logit processor is called.
    if not self.prepared_bias_variables:
        self._prepare_bias_variables(scores)

    # 2 - prepares an empty bias to add
    bias = torch.zeros_like(scores)

    # 3 - include the bias from length = 1
    bias += self.length_1_bias

    # 4 - include the bias from length > 1, after determining which biased sequences may be completed.
    for sequence_ids, sequence_bias in self.sequence_bias.items():
        if len(sequence_ids) == 1:  # the sequence is of length 1, already applied
            continue
        if len(sequence_ids) > input_ids.shape[1]:  # the sequence is longer than the context, ignore
            continue
        prefix_length = len(sequence_ids) - 1
        last_token = sequence_ids[-1]
        matching_rows = torch.eq(
            input_ids[:, -prefix_length:],
            torch.tensor(sequence_ids[:-1], dtype=input_ids.dtype, device=input_ids.device),
        ).prod(dim=1)
        bias[:, last_token] += torch.where(
            matching_rows.bool(),
            torch.tensor(sequence_bias, device=input_ids.device),
            torch.tensor(0.0, device=input_ids.device),
        )

    # 5 - apply the bias to the scores
    scores_processed = scores + bias
    return scores_processed
```

## 逐行讲解 / What's happening

1. **第 1276-1285 行 / Lines 1276-1285 (lazy setup)**:
   - 中文: 用户传进来的列表先校验并转成 dict；真正依赖 vocab size 的张量第一次调用时才建。
   - English: The user-provided list is validated and converted into a dict; tensors that need the vocab size are built lazily on first call.
2. **第 1293-1298 行 / Lines 1293-1298 (base bias tensor)**:
   - 中文: 每次调用都创建和 `scores` 同形状的 bias，单 token bias 可以直接加进去。
   - English: Each call creates a bias tensor shaped like `scores`; single-token biases can be added immediately.
3. **第 1300-1306 行 / Lines 1300-1306 (multi-token phrases)**:
   - 中文: 多 token 短语不会每一步都扣分，只在上下文长度足够、并且差最后一个 token 时才处理。
   - English: Multi-token phrases are not penalized at every step; they are considered only when the context is long enough and the phrase is one token from completion.
4. **第 1307-1315 行 / Lines 1307-1315 (row-wise prefix match)**:
   - 中文: 批里的每一行都拿最近的 prefix 和目标短语前缀比较，匹配的行才改 `last_token` 分数。
   - English: Each batch row compares its recent prefix against the target phrase prefix; only matching rows adjust the score of `last_token`.
5. **第 1317-1319 行 / Lines 1317-1319 (additive logits edit)**:
   - 中文: 最终控制只是 `scores + bias`，所以它能和其他 processor 串联。
   - English: The final control is just `scores + bias`, which composes cleanly with other processors.

## 类比 / The analogy

像输入法联想。你已经打了“San”，系统才会把“Francisco”排得更靠前；如果你打的是“New”，它不会提前给“Francisco”加权。

It is like autocomplete. After you type "San", the system can rank "Francisco" higher; if you typed "New", it does not boost "Francisco" yet.

## 自己跑一遍 / Try it yourself

```python
scores = [{"red": 0.0, "blue": 0.0} for _ in range(2)]
input_ids = [["make"], ["paint"]]
sequence_bias = {("paint", "blue"): 2.5, ("make", "red"): -1.0}

for row, context in enumerate(input_ids):
    for seq, value in sequence_bias.items():
        prefix, last = seq[:-1], seq[-1]
        if tuple(context[-len(prefix):]) == prefix:
            scores[row][last] += value

print(scores)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[{'red': -1.0, 'blue': 0.0}, {'red': 0.0, 'blue': 2.5}]
```

中文: 同一个 `blue` token 只在第二行被加分，因为只有第二行上下文匹配 `("paint",)`。

English: The same `blue` token is boosted only in the second row because only that row matches the `("paint",)` prefix.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **bad words filtering** / **Bad-words filtering**: 中文: 禁词表也可以看成给某些可完成短语的 token 加 `-inf`。 / English: A bad-words list is similar to adding `-inf` to tokens that would complete forbidden phrases.
- **structured decoding** / **Structured decoding**: 中文: JSON、SQL 等格式约束也常在 logits 层按上下文改下一 token 候选。 / English: JSON or SQL constraints often edit next-token candidates at the logits layer based on context.

## 注意事项 / Caveats / when it breaks

- **tokenization 很关键 / Tokenization matters**: 中文: 带空格和不带空格的词可能是不同 token 序列，bias 写错就不会触发。 / English: A word with and without a leading space may tokenize differently, so the wrong bias sequence may never fire.
- **负 bias 会挡住 beam / Negative bias can trap beams**: 中文: 对长短语太早扣分会让 beam search 放弃正确前缀，所以这段代码选择在最后 token 才动手。 / English: Penalizing a long phrase too early can make beam search abandon useful prefixes, so this processor waits until the final token.

## 延伸阅读 / Further reading

- Transformers logits processors: https://github.com/huggingface/transformers/blob/899f55f8e41654a106a414e93a6f5fd8f3b8bf37/src/transformers/generation/logits_process.py
