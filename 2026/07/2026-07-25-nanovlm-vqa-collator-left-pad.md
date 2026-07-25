---
date: 2026-07-25
topic: huggingface
source: huggingface
repo: huggingface/nanoVLM
file: data/collators.py
permalink: https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/data/collators.py#L4-L71
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, huggingface, collator]
---

# nanoVLM VQA Collator：左 padding 文本，保留图像列表 / nanoVLM VQA Collator: Left-Pad Text and Keep Images as a List

> **一句话 / In one line**: 多模态 batch 的文本可以 stack, 图像数量却不一定相同, 所以 collator 只 stack token 张量。 / In a multimodal batch, text can be stacked while image counts may differ, so the collator stacks only token tensors.

## 为什么重要 / Why this matters

VQA 样本通常长度不同, 还可能每条样本带不同数量的图像。这个 collator 做了三个关键决定: 过滤空样本, 按左侧 padding 对齐文本, `labels` 用 `-100` 避免 padding 参与 loss, 而 `images` 保留为 Python list。

VQA samples have different text lengths and may carry different numbers of images. This collator makes three practical choices: drop empty rows, left-pad text, use `-100` label padding so loss ignores it, and keep `images` as a Python list.

## 代码 / The code

`huggingface/nanoVLM` -- [`data/collators.py`](https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/data/collators.py#L4-L71)

```python
class BaseCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _pad_batch(self, batch, max_length):
        batch["input_ids"] = [torch.nn.functional.pad(ids, (max_length - len(ids), 0), value=self.tokenizer.pad_token_id) for ids in batch["input_ids"]]
        batch["labels"]    = [torch.nn.functional.pad(labels, (max_length - len(labels), 0), value=self.tokenizer.pad_token_id) for labels in batch["labels"]]
        batch["attention_mask"] = [torch.nn.functional.pad(attention_mask, (max_length - len(attention_mask), 0), value=0) for attention_mask in batch["attention_mask"]]

    def prepare_batch(self, batch, max_length=None):
        # 1) Handle empty
        if not batch:
            return {"input_ids": [], "labels": [], "attention_mask": [], "images": []}

        # 2) Drop None rows
        batch = [s for s in batch if s is not None]
        if not batch:
            return {"input_ids": [], "labels": [], "attention_mask": [], "images": []}

        # batch is a list of dicts, each containing "input_ids", "attention_mask", "labels", "images"
        # let's convert it to a dict of lists of tensors
        batch = {k: [item[k] for item in batch] for k in batch[0]}

        if max_length is not None:
            batch = self._discard_samples_that_are_too_long(batch, max_length)

        if len(batch["input_ids"]) == 0:
            return batch

        # Pad samples to max length
        if max_length is not None:
            max_len = max_length
        else:
            max_len = max(map(len, batch["input_ids"]))
        self._pad_batch(batch, max_len) #  dictionaries in Python are mutable and passed by reference

        return {
            "input_ids": torch.stack(batch["input_ids"]),
            "attention_mask": torch.stack(batch["attention_mask"]),
            "images": batch["images"],
            "labels": torch.stack(batch["labels"]),
        }

    def _discard_samples_that_are_too_long(self, batch, max_length):
        filtered = [
            (ids, label, attn, img)
            for ids, label, attn, img in zip(batch["input_ids"], batch["labels"], batch["attention_mask"], batch["images"])
            if len(ids) <= max_length
        ]
        if not filtered:
            return {"input_ids": [], "labels": [], "attention_mask": [], "images": []}
        batch_token_ids, batch_labels, batch_attentions, batch_images = zip(*filtered)
        return {"input_ids": list(batch_token_ids), "labels": list(batch_labels), "attention_mask": list(batch_attentions), "images": list(batch_images)}


class VQACollator(BaseCollator):  # Visual Question Answering Collator
    def __init__(self, tokenizer, max_length):
        self.max_length = max_length
        super().__init__(tokenizer)

    def _pad_batch(self, batch, max_length):  # Reimplementing to use -100 as the pad value for labels, so that it's ignored by the loss
        batch["input_ids"] = [torch.nn.functional.pad(ids, (max_length - len(ids), 0), value=self.tokenizer.pad_token_id) for ids in batch["input_ids"]]
        batch["labels"]    = [torch.nn.functional.pad(labels, (max_length - len(labels), 0), value=-100) for labels in batch["labels"]]
        batch["attention_mask"] = [torch.nn.functional.pad(attention_mask, (max_length - len(attention_mask), 0), value=0) for attention_mask in batch["attention_mask"]]

    def __call__(self, batch):
        batch = self.prepare_batch(batch, max_length=self.max_length)
        return batch
```

## 逐行讲解 / What's happening

1. **第 13-21 行 / Lines 13-21 (`空 batch 和 None 行`)**:
   - 中文: 训练数据管线里坏样本直接被过滤, 而不是在模型 forward 里崩。
   - English: Bad samples are filtered in the data pipeline, not inside model forward.
2. **第 25-38 行 / Lines 25-38 (`list-of-dicts 转 dict-of-lists`)**:
   - 中文: 先转置 batch, 再统一决定最大长度。
   - English: It transposes list-of-dicts into dict-of-lists, then chooses the padding length.
3. **第 40-45 行 / Lines 40-45 (`只 stack 文本张量`)**:
   - 中文: `input_ids`、`attention_mask`、`labels` 堆起来, `images` 继续是 list。
   - English: It stacks token tensors but leaves `images` as a list.
4. **第 64-67 行 / Lines 64-67 (`VQA 特化 padding`)**:
   - 中文: labels 的 padding 值改成 `-100`, 正好被 cross entropy ignore。
   - English: The VQA subclass pads labels with `-100`, matching cross-entropy ignore behavior.

## 类比 / The analogy

这像把一组包裹装箱。纸质表格可以垫空白页对齐, 但每个包裹里的实物尺寸不同, 就不要硬塞进同一个模具。

It is like packing shipments. Paper forms can be padded with blank pages to align, but the physical items differ in size, so you do not force them into one rigid mold.

## 自己跑一遍 / Try it yourself

```python
class VQACollator:
    def __init__(self, max_length): self.max_length = max_length
    def __call__(self, batch):
        batch = [x for x in batch if x is not None and len(x["input_ids"]) <= self.max_length]
        def pad(x, value):
            return [value] * (self.max_length - len(x)) + x
        return {
            "input_ids": [pad(x["input_ids"], 0) for x in batch],
            "labels": [pad(x["labels"], -100) for x in batch],
            "images": [x["images"] for x in batch],
        }

b = [{"input_ids": [5, 6], "labels": [7, 8], "images": ["img0"]}]
out = VQACollator(4)(b)
print(out["input_ids"], out["labels"], out["images"])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
[[0, 0, 5, 6]] [[-100, -100, 7, 8]] [['img0']]
```

左 padding 后, 新 token 仍然贴在右侧, 自回归解码更容易复用最后位置。

After left padding, real tokens stay on the right, which is convenient for autoregressive decoding at the last position.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **TRL preference collator**: chosen/rejected 也常先过滤再拼 batch。 / Chosen/rejected examples are also filtered and packed before batching.
- **LLaVA 类数据管线**: 图像常保持 list, 直到模型里按样本展开。 / LLaVA-style pipelines often keep images as lists until the model expands them per sample.

## 注意事项 / Caveats / when it breaks

- **空 batch 要被上游处理**: 返回空列表不等于合法训练 batch。 / An empty return is not a valid training batch unless the caller handles it.
- **左 padding 需要模型支持**: 位置编码和 attention mask 必须和左 padding 一致。 / The model position logic and attention mask must agree with left padding.

## 延伸阅读 / Further reading

- nanoVLM repository: https://github.com/huggingface/nanoVLM
- PyTorch pad: https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
