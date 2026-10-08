---
date: 2026-10-08
topic: huggingface
source: huggingface
repo: huggingface/nanoVLM
file: data/collators.py
permalink: https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/data/collators.py#L1-L71
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, huggingface, collator]
---

# nanoVLM collator：先过滤坏样本，再左填充成 batch / nanoVLM Collator: Filter Bad Samples, Then Left-Pad a Batch

> **一句话 / In one line**: 这个 collator 把变长多模态样本整理成稳定 batch，并用 `-100` 保护 VQA label padding 不进 loss。 / This collator turns variable-length multimodal samples into a stable batch and uses `-100` so VQA label padding is ignored by the loss.

## 为什么重要 / Why this matters

VLM 训练 batch 里常见三种麻烦：样本可能是 `None`，文本长度不一致，某些样本超过最大长度。这个 collator 把这些脏边界集中处理：空 batch 直接返回，坏行丢掉，超长样本过滤，然后对 token、mask、label 做一致 padding。

VLM training batches often have three annoyances: samples may be `None`, text lengths vary, and some samples exceed the max length. This collator centralizes those boundaries: empty batches return early, bad rows are dropped, overlong samples are filtered, then tokens, masks, and labels are padded consistently.

## 代码 / The code

`huggingface/nanoVLM` — [`data/collators.py`](https://github.com/huggingface/nanoVLM/blob/4e0c0961846135c2217f95e54cb4c2d66eb55e42/data/collators.py#L1-L71)

```python
import torch


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

1. **第 13-21 行 / Lines 13-21**:
   - 中文: 空 batch 和全 `None` batch 都走统一空结构返回，后续 trainer 不需要猜 key 是否存在。
   - English: Empty batches and all-`None` batches return the same empty structure, so the trainer does not guess which keys exist.
1. **第 23-31 行 / Lines 23-31**:
   - 中文: list-of-dicts 先转成 dict-of-lists；如果设了 `max_length`，超长样本被整行过滤，图像也跟着同步过滤。
   - English: The list of dicts becomes a dict of lists. When `max_length` is set, overlong rows are filtered together with their images.
1. **第 33-45 行 / Lines 33-45**:
   - 中文: batch 内最大长度来自固定上限或当前 batch，padding 后 token/mask/label 才能 `torch.stack`。
   - English: Max length comes from a fixed cap or the current batch; only after padding can tokens, masks, and labels be stacked.
1. **第 64-67 行 / Lines 64-67**:
   - 中文: VQA 版本重写 label padding 值为 `-100`，这是 PyTorch cross entropy 的 ignore index 常用约定。
   - English: The VQA variant overrides label padding to `-100`, the common ignore index for PyTorch cross entropy.

## 类比 / The analogy

像把不同高度的书装进同一个快递盒：太高的书先拿出来，矮书用填充物垫齐，标签贴在不会被扫描计费的位置。

It is like packing books of different heights into one shipping box: remove books that are too tall, pad the shorter ones, and put filler labels where the scanner will ignore them.

## 自己跑一遍 / Try it yourself

```python
def left_pad(xs, width, value):
    return [value] * (width - len(xs)) + xs

batch = [None, {"ids": [4, 5], "labels": [9, 8]}, {"ids": [1, 2, 3, 4], "labels": [7, 7, 7, 7]}]
max_len = 3
rows = [x for x in batch if x and len(x["ids"]) <= max_len]
print([left_pad(r["ids"], max_len, 0) for r in rows])
print([left_pad(r["labels"], max_len, -100) for r in rows])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[[0, 4, 5]]
[[-100, 9, 8]]
```

长度为 4 的样本被过滤，label 的 padding 使用 `-100`，不会参与 loss。

The length-4 sample is filtered out, and label padding uses `-100`, so it will not contribute to loss.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Transformers data collators** / **Transformers data collators**: NLP batch 通常把 padding、mask 和 label ignore index 集中在 collator。 / NLP batches usually centralize padding, masks, and label ignore index inside the collator.
- **OpenVLA action collator** / **OpenVLA action collator**: 多模态动作训练也需要同步过滤文本、图像和动作字段。 / Multimodal action training also needs synchronized filtering across text, images, and actions.

## 注意事项 / Caveats / when it breaks

- **全过滤要被上游接受** / **All-filtered batches need upstream handling**: 如果返回空 batch，trainer 必须能跳过这一步。 / If an empty batch is returned, the trainer must know how to skip that step.
- **左填充要匹配模型假设** / **Left padding must match model assumptions**: 有些 decoder 或 position-id 逻辑更偏好右填充，不能只看 collator。 / Some decoders or position-id code prefer right padding, so the collator cannot be chosen in isolation.

## 延伸阅读 / Further reading

- [nanoVLM repository](https://github.com/huggingface/nanoVLM)
- [PyTorch cross entropy ignore_index](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
