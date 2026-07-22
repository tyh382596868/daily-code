---
date: 2026-07-10
topic: huggingface
source: huggingface
repo: huggingface/transformers
file: src/transformers/generation/logits_process.py
permalink: https://github.com/huggingface/transformers/blob/0bc355418bb265136a66c2dedc501066ffbc237d/src/transformers/generation/logits_process.py#L1021-L1088
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, huggingface, generation, logits-processor]
---

# Transformers no-repeat ngram：把历史短语变成下一 token 禁止表 / Transformers No-Repeat N-Gram: Turn History into a Next-Token Ban List

> **一句话 / In one line**: `NoRepeatNGramLogitsProcessor` 先把已生成序列索引成 n-gram 字典，再把会重复短语的 token 分数设成 `-inf`。 / `NoRepeatNGramLogitsProcessor` indexes generated text as n-grams, then sets scores for tokens that would repeat a phrase to `-inf`.

## 为什么重要 / Why this matters

生成模型很容易卡在重复短语里。这个 processor 没有改模型，也没有重训，只是在每一步采样前改 logits：如果某个 token 会让最近的 prefix 组成已出现过的 n-gram，它就被禁止。

Generation models can fall into repeated phrases. This processor does not change or retrain the model; it edits logits before each sampling step. If a token would complete an n-gram that already appeared, that token is banned.

## 代码 / The code

`huggingface/transformers` — [`src/transformers/generation/logits_process.py`](https://github.com/huggingface/transformers/blob/0bc355418bb265136a66c2dedc501066ffbc237d/src/transformers/generation/logits_process.py#L1021-L1088)

```python
def _get_ngrams(ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int):
    """
    Assume ngram_size=2 and prev_input_ids=tensor([[40, 2883, 2712, 4346]]). The output of generated ngrams look like
    this {(40,): [2883], (2883,): [2712], (2712,): [4346]}.
    """
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        # Loop through each n-gram of size ngram_size in the list of tokens (gen_tokens)
        for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]
    return generated_ngrams


def _get_generated_ngrams(banned_ngrams, prev_input_ids, ngram_size, cur_len):
    """
    Determines the banned tokens for the current hypothesis based on previously generated n-grams.
    """
    # Before decoding the next token, prevent decoding of ngrams that have already appeared
    start_idx = cur_len + 1 - ngram_size
    ngram_idx = tuple(prev_input_ids[start_idx:cur_len].tolist())
    return banned_ngrams.get(ngram_idx, [])


def _calc_banned_ngram_tokens(
    ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int, cur_len: int
) -> list[Iterable[int]]:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    generated_ngrams = _get_ngrams(ngram_size, prev_input_ids, num_hypos)
    banned_tokens = [
        _get_generated_ngrams(generated_ngrams[hypo_idx], prev_input_ids[hypo_idx], ngram_size, cur_len)
        for hypo_idx in range(num_hypos)
    ]
    return banned_tokens


class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    r"""
    N-grams are groups of "n" consecutive words, characters, or tokens taken from a sequence of text.
    """
```

## 逐行讲解 / What's happening

1. **第 1038-1047 行 / Lines 1038-1047 (`_get_ngrams`)**:
   - 中文: 每个 n-gram 被拆成 `prefix -> possible_next_tokens`，例如 `(New,) -> [York]`。
   - English: Each n-gram becomes `prefix -> possible_next_tokens`, such as `(New,) -> [York]`.
2. **第 1067-1070 行 / Lines 1067-1070 (`_get_generated_ngrams`)**:
   - 中文: 解码下一 token 前，只看当前尾部的 `n-1` 个 token，查它曾经接过什么。
   - English: Before decoding the next token, it looks only at the current trailing `n-1` tokens and asks what followed them before.
3. **第 1077-1085 行 / Lines 1077-1085 (`_calc_banned_ngram_tokens`)**:
   - 中文: 序列太短时不禁用；足够长后，每个 beam/hypothesis 都单独算 banned list。
   - English: Short sequences ban nothing; once long enough, each beam or hypothesis gets its own banned list.
4. **第 1088 行以后 / From line 1088 (`LogitsProcessor`)**:
   - 中文: 真正接入 generate 时，这些 banned token 会被写成 `-inf`，后续 softmax 就选不到。
   - English: In `generate`, these banned tokens are set to `-inf`, so later softmax cannot select them.

## 类比 / The analogy

这像写作时旁边放一张“刚用过的短语卡片”。每次你要写下一个词，先看这张卡片：如果会复读一句旧话，就换一个词。

It is like writing with a card of recently used phrases beside you. Before picking the next word, you check the card; if the word would repeat an old phrase, choose another one.

## 自己跑一遍 / Try it yourself

```python
def banned(tokens, n):
    table = {}
    for gram in zip(*[tokens[i:] for i in range(n)]):
        table.setdefault(tuple(gram[:-1]), []).append(gram[-1])
    prefix = tuple(tokens[-(n - 1):])
    return table.get(prefix, [])

tokens = [1, 2, 3, 1, 2]
print(banned(tokens, 3))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[3]
```

因为历史里已经出现过 `[1, 2, 3]`，当前尾部又是 `[1, 2]`，所以下一个 token `3` 会被禁掉。

Because `[1, 2, 3]` already appeared and the current suffix is `[1, 2]`, token `3` is banned next.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Beam search constraints** / **Beam search constraints**: 也是在每步生成前修改候选 token 集合。 / They also modify candidate token sets before each generation step.
- **Bad words processors** / **Bad words processors**: 同样通过 logits mask 把部分 token 排除。 / They similarly exclude tokens through logits masking.

## 注意事项 / Caveats / when it breaks

- **可能误伤专名** / **It can hurt proper nouns**: 过小的 `ngram_size` 会让 “New York” 这类短语只能出现一次。 / A small `ngram_size` can allow a phrase like "New York" only once.
- **prompt 也算历史** / **The prompt counts too**: decoder-only 模型会把 prompt 中的 n-gram 也纳入禁止表。 / Decoder-only models include prompt n-grams in the ban table.

## 延伸阅读 / Further reading

- [Transformers no-repeat ngram utilities](https://github.com/huggingface/transformers/blob/0bc355418bb265136a66c2dedc501066ffbc237d/src/transformers/generation/logits_process.py#L1021-L1088)
