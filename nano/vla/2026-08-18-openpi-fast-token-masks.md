---
date: 2026-08-18
topic: vla
source: vla
repo: Physical-Intelligence/openpi
file: src/openpi/models/tokenizer.py
permalink: https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/models/tokenizer.py#L64-L139
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, nano-vla, action-tokenizer]
component: action-tokenizer
variant: advanced
---

# openpi FAST token masks：prefix 看全局，action 只算 loss / openpi FAST Token Masks: Prefix Sees Context, Actions Carry Loss

> **一句话 / In one line**: openpi 的 FAST tokenizer 把 task、离散 state 和 action tokens 拼成同一条语言序列，同时生成 token mask、AR mask 和 loss mask。 / openpi's FAST tokenizer packs task text, discretized state, and action tokens into one language sequence while producing token, AR, and loss masks.

## 为什么重要 / Why this matters

VLA 的 action tokenizer 不只是“把动作变 token”。训练时模型还必须知道哪些 token 是上下文、哪些 token 要因果预测、哪些 token 参与 loss。这里的三张 mask 就是 pi0-FAST 把语言建模和动作预测接在一起的合同。

A VLA action tokenizer is not just "turn actions into tokens." During training, the model must know which tokens are context, which are causally predicted, and which contribute to loss. These three masks are the contract that connects language modeling to action prediction in pi0-FAST.

## 代码 / The code

`Physical-Intelligence/openpi` — [`src/openpi/models/tokenizer.py`](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/models/tokenizer.py#L64-L117)

```python
cleaned_text = prompt.lower().strip().replace("_", " ")
discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
state_str = " ".join(map(str, discretized_state))
prefix = f"Task: {cleaned_text}, State: {state_str};\n"
prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

action_tokens = self._fast_tokenizer(actions[None])[0]
action_tokens_in_pg = self._act_tokens_to_paligemma_tokens(action_tokens)

tokens = prefix_tokens + postfix_tokens
token_mask = [True] * len(tokens)
ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)
```

## 逐行讲解 / What's happening

1. **第 67-75 行 / Lines 67-75**: 中文: prompt 先标准化，state 被分到 256 个离散 bin，再一起写进 prefix。 / English: The prompt is normalized, the state is discretized into 256 bins, and both are written into the prefix.
2. **第 77-87 行 / Lines 77-87**: 中文: actions 先经 FAST tokenizer，再映射到 PaliGemma 词表尾部的保留 token 区间。 / English: Actions go through the FAST tokenizer, then are mapped into reserved tokens near the tail of the PaliGemma vocabulary.
3. **第 91-97 行 / Lines 91-97**: 中文: `ar_mask=0` 的 prefix 可作为双向上下文，`ar_mask=1` 的 postfix 是要自回归预测的动作段。 / English: The prefix with `ar_mask=0` acts as bidirectional context, while the postfix with `ar_mask=1` is the autoregressive action span.
4. **第 98-117 行 / Lines 98-117**: 中文: 三张 mask 和 tokens 一起 padding/truncation，保证长度对齐。 / English: The three masks are padded or truncated together with tokens, preserving alignment.

## 在 nanoVLA 中的位置 / Where this fits in nanoVLA

这是 `action-tokenizer` 的高级版本：它同时定义输入格式、动作 token 空间、attention 语义和 loss 语义。一个 from-scratch nanoVLA 可以先实现这个合同，再替换具体的 FAST tokenizer。

This is the advanced `action-tokenizer` layer: it defines input format, action-token space, attention semantics, and loss semantics together. A from-scratch nanoVLA can implement this contract first and swap in a concrete FAST tokenizer later.

## 类比 / The analogy

像一张考试卷：题干和已知条件可以反复看，答案区必须从左到右填写，评分只看答案区。三张 mask 分别管“纸上哪里有字”“哪里要按顺序写”“哪里算分”。

It is like an exam sheet: instructions and givens are visible, the answer area is filled left-to-right, and grading only looks at the answer area. The three masks say what exists, what is causal, and what is scored.

## 自己跑一遍 / Try it yourself

```python
def build_masks(prefix_len, action_len, max_len):
    tokens_len = prefix_len + action_len
    token_mask = [True] * tokens_len
    ar_mask = [0] * prefix_len + [1] * action_len
    loss_mask = [False] * prefix_len + [True] * action_len
    pad = max_len - tokens_len
    return token_mask + [False] * pad, ar_mask + [False] * pad, loss_mask + [False] * pad

for row in build_masks(prefix_len=4, action_len=3, max_len=10):
    print(row)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[True, True, True, True, True, True, True, False, False, False]
[0, 0, 0, 0, 1, 1, 1, False, False, False]
[False, False, False, False, True, True, True, False, False, False]
```

prefix 有 token 但不算 loss；action 既自回归又算 loss。

The prefix has tokens but no loss; the action span is both autoregressive and scored.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **prefix-LM** / **prefix-LMs**: 前缀可双向看，后缀按因果顺序生成。 / Prefix tokens can be read bidirectionally, while suffix tokens are generated causally.
- **instruction tuning** / **instruction tuning**: prompt 不算 loss，只在 answer span 上训练。 / The prompt is masked out of loss; only the answer span is trained.

## 注意事项 / Caveats / when it breaks

- **state bin 假设归一化范围** / **State bins assume normalized range**: 代码假设 state 已在 `[-1, 1]` 附近。 / The code assumes states are already near `[-1, 1]`.
- **截断会截掉动作** / **Truncation can remove actions**: prompt 太长时，动作 token 和 loss mask 可能被截短。 / If the prompt is too long, action tokens and their loss mask can be truncated.

## 延伸阅读 / Further reading

- [openpi source](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/models/tokenizer.py#L64-L139)
