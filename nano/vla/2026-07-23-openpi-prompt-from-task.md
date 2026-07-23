---
date: 2026-07-23
topic: vla
source: vla
repo: Physical-Intelligence/openpi
file: src/openpi/transforms.py
permalink: https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/transforms.py
difficulty: beginner
read_time: ~7 min
tags: [code-of-the-day, vla, openpi, data-pipeline, prompt]
---

# openpi PromptFromTask：把 task 字段变成语言条件 / openpi PromptFromTask: Turn the Task Field into Language Conditioning

> **一句话 / In one line**: `PromptFromTask` 这类 transform 把数据集里的任务描述复制到模型约定的 `prompt` 字段，让后续 tokenizer 和 VLM backbone 都走同一条输入契约。 / A transform like `PromptFromTask` copies a dataset task description into the model's agreed `prompt` field so the tokenizer and VLM backbone share one input contract.

## 为什么重要 / Why this matters

VLA 数据集常常有不同字段名：`task`、`language_instruction`、`prompt`、`goal`。训练代码如果到处硬编码字段名，很快会变成不可维护的适配层。openpi 的 transform 思路是先把外部 schema 归一化，再让模型只关心稳定字段。

VLA datasets often use different names: `task`, `language_instruction`, `prompt`, or `goal`. If training code hard-codes those names everywhere, adapters become unmaintainable. openpi-style transforms normalize the external schema first, so the model only sees stable fields.

## 代码 / The code

`Physical-Intelligence/openpi` — [`src/openpi/transforms.py`](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/transforms.py)

```python
# Simplified teaching slice.
class PromptFromTask:
    def __call__(self, data):
        task = data.get("task")
        if task is None:
            task = data.get("language_instruction", "")
        return {**data, "prompt": task}

class TokenizePrompt:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, data):
        return {**data, "prompt_tokens": self.tokenizer(data["prompt"])}
```

## 逐行讲解 / What's happening

1. **读外部字段 / Read external fields**: 中文: transform 允许数据集继续用自己的字段名。 English: the dataset can keep its own naming.
2. **写标准字段 / Write a standard field**: 中文: 后续模型只读 `prompt`，不用知道原始数据从哪来。 English: downstream model code reads only `prompt`.
3. **tokenizer 成为下一环 / Tokenizer becomes the next stage**: 中文: 文本归一化和 token 化分开，便于替换 tokenizer。 English: text normalization and tokenization stay separate, making tokenizers swappable.

## 在 nanoVLA 中的位置 / Where this fits in nanoVLA

这是 `training-step` / data-pipeline 的前置适配层。它不改变图像、状态或动作，但决定语言条件是否能稳定进入 `vlm-backbone-wiring`。生产 VLA 需要把每个数据源都先映射到同一套字段：`observation`、`state`、`action`、`prompt`、`mask`。

This is a pre-model adapter for the `training-step` data pipeline. It does not change images, states, or actions, but it determines whether language conditioning reaches `vlm-backbone-wiring` consistently. A production VLA maps every data source into the same fields: `observation`, `state`, `action`, `prompt`, and `mask`.

## 自己跑一遍 / Try it yourself

```python
def prompt_from_task(row):
    return {**row, "prompt": row.get("task", row.get("language_instruction", ""))}

print(prompt_from_task({"task": "open the drawer"})["prompt"])
print(prompt_from_task({"language_instruction": "close the lid"})["prompt"])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
open the drawer
close the lid
```

## 注意事项 / Caveats / when it breaks

- **空字符串也是条件 / Empty strings are still conditioning**: 静默把缺失任务变成空 prompt，可能让模型学到“无指令”行为。 / silently converting missing tasks to empty prompts can teach a no-instruction behavior.
- **不要在模型里猜字段名 / Do not guess field names in the model**: 字段适配应留在 transform 层。 / schema adaptation belongs in transforms, not the model.
- **多语言任务要统一 / Normalize multilingual tasks**: 训练混合中英文时，prompt 规范会影响 tokenizer 分布。 / mixed-language training makes prompt normalization part of the data contract.

## 延伸阅读 / Further reading

- [openpi transforms.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/transforms.py)

