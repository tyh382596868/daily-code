---
date: 2026-07-12
topic: huggingface
source: huggingface
repo: huggingface/trl
file: trl/trainer/dpo_trainer.py
permalink: https://github.com/huggingface/trl/blob/1925183c91852c30088ed8ad53181e7b7147489e/trl/trainer/dpo_trainer.py#L79-L204
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, preference-data]
---

# TRL preference collator：把 chosen/rejected 拼成同一个 batch / TRL Preference Collator: Put Chosen and Rejected in One Batch

> **一句话 / In one line**: DPO 数据整理器把每条样本拆成 chosen 和 rejected 两条序列，再一起 pad，方便一次 forward 比较偏好。 / The DPO collator turns each example into a chosen sequence and a rejected sequence, then pads them together for one preference-comparison forward pass.

## 为什么重要 / Why this matters

偏好训练的 batch 不是普通 `(input, label)`，而是同一个 prompt 下的两个 completion。TRL 的整理器把这对样本变成“前半 batch 是 chosen，后半 batch 是 rejected”的布局，并额外生成 `completion_mask` 标出哪些 token 参与偏好损失。

Preference training is not a normal `(input, label)` batch; each prompt has two completions. TRL lays them out as chosen samples in the first half and rejected samples in the second half, with a `completion_mask` marking which tokens should contribute to the preference loss.

## 代码 / The code

`huggingface/trl` — [`trl/trainer/dpo_trainer.py`](https://github.com/huggingface/trl/blob/1925183c91852c30088ed8ad53181e7b7147489e/trl/trainer/dpo_trainer.py#L79-L204)

```python
def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_chosen_ids = [example["prompt_ids"] + example["chosen_ids"] for example in examples]
    prompt_rejected_ids = [example["prompt_ids"] + example["rejected_ids"] for example in examples]
    chosen_mask = [[0] * len(example["prompt_ids"]) + [1] * len(example["chosen_ids"]) for example in examples]
    rejected_mask = [[0] * len(example["prompt_ids"]) + [1] * len(example["rejected_ids"]) for example in examples]

    if self.max_length is not None:
        if self.truncation_mode == "keep_start":
            sl = slice(None, self.max_length)
        elif self.truncation_mode == "keep_end":
            sl = slice(-self.max_length, None)
        else:
            raise ValueError(
                f"Unsupported truncation mode: {self.truncation_mode}, expected 'keep_start' or 'keep_end'"
            )
        prompt_chosen_ids = [ids[sl] for ids in prompt_chosen_ids]
        prompt_rejected_ids = [ids[sl] for ids in prompt_rejected_ids]
        chosen_mask = [m[sl] for m in chosen_mask]
        rejected_mask = [m[sl] for m in rejected_mask]

    chosen_attention_mask = [[1] * len(ids) for ids in prompt_chosen_ids]
    rejected_attention_mask = [[1] * len(ids) for ids in prompt_rejected_ids]
    input_ids = prompt_chosen_ids + prompt_rejected_ids
    attention_mask = chosen_attention_mask + rejected_attention_mask
    completion_mask = chosen_mask + rejected_mask
```

## 逐行讲解 / What's happening

1. **拼接 prompt 和 completion / Join prompt and completion**:
   - 中文: chosen/rejected 都保留同一个 prompt，让模型在完全相同上下文下比较两个回答。
   - English: Both chosen and rejected keep the same prompt, so the model compares answers under identical context.
2. **构造 completion mask / Build completion masks**:
   - 中文: prompt 区域是 0，completion 区域是 1；损失只应该看回答部分。
   - English: Prompt positions are 0 and completion positions are 1, so the loss can focus on answers.
3. **截断也同步截 mask / Truncate masks too**:
   - 中文: token 被切掉时，mask 必须同步切掉，否则 token 和监督信号会错位。
   - English: When tokens are truncated, masks must be truncated the same way; otherwise supervision shifts out of alignment.
4. **合并 batch / Concatenate the batch**:
   - 中文: `input_ids = chosen + rejected` 让后续 forward 可以一次算完两边 logprob。
   - English: `input_ids = chosen + rejected` lets the trainer compute both sides' log probabilities in one forward pass.

## 类比 / The analogy

像老师批作文：每道题有 A 卷和 B 卷，先把题干复印到两张卷子上，再只用红笔批答案区域。最后把所有 A 卷叠在前面、B 卷叠在后面，方便成对比较。

It is like grading two essays for the same prompt: copy the prompt onto both sheets, mark only the answer region, then stack all A sheets before all B sheets for paired comparison.

## 自己跑一遍 / Try it yourself

```python
examples = [
    {"prompt_ids": [1, 2], "chosen_ids": [3, 4], "rejected_ids": [5]},
    {"prompt_ids": [6], "chosen_ids": [7], "rejected_ids": [8, 9]},
]
chosen = [e["prompt_ids"] + e["chosen_ids"] for e in examples]
rejected = [e["prompt_ids"] + e["rejected_ids"] for e in examples]
masks = (
    [[0] * len(e["prompt_ids"]) + [1] * len(e["chosen_ids"]) for e in examples]
    + [[0] * len(e["prompt_ids"]) + [1] * len(e["rejected_ids"]) for e in examples]
)
print(chosen + rejected)
print(masks)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
[[1, 2, 3, 4], [6, 7], [1, 2, 5], [6, 8, 9]]
[[0, 0, 1, 1], [0, 1], [0, 0, 1], [0, 1, 1]]
```

注意 chosen 和 rejected 的 prompt 是重复的，但 mask 只打开 completion。

Notice that chosen and rejected repeat the prompt, while the mask opens only the completion part.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **DPO/IPO/KTO trainer** / **DPO/IPO/KTO trainers**: 偏好算法都要把成对答案变成可比较的 logprob。 / Preference algorithms need paired answers converted into comparable log probabilities.
- **多模态偏好数据** / **multimodal preference data**: 图像或视频输入也会复用同一上下文，只替换候选回答。 / Image or video prompts reuse the same context while swapping candidate answers.

## 注意事项 / Caveats / when it breaks

- **截断策略会改变训练信号** / **truncation changes the training signal**: `keep_start` 可能切掉答案尾部，`keep_end` 可能切掉 prompt 前缀。 / `keep_start` can drop answer tails, while `keep_end` can drop prompt prefixes.
- **padding 不等于 loss mask** / **padding is not the loss mask**: `attention_mask` 管可见 token，`completion_mask` 管训练 token。 / `attention_mask` controls visible tokens; `completion_mask` controls supervised tokens.

## 延伸阅读 / Further reading

- TRL DPO trainer source — https://github.com/huggingface/trl/blob/1925183c91852c30088ed8ad53181e7b7147489e/trl/trainer/dpo_trainer.py
