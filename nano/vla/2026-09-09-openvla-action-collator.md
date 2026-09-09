---
date: 2026-09-09
topic: vla
source: vla
repo: openvla/openvla
file: prismatic/util/data_utils.py
permalink: https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/prismatic/util/data_utils.py#L94-L142
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, training-step, collator, multimodal]
build_role: training-step advanced variant
---

# OpenVLA Action Collator：把变长语言和图像装进稳定 batch / OpenVLA Action Collator: Pack Variable-Length Language and Images into a Stable Batch

> **一句话 / In one line**: `PaddedCollatorForActionPrediction` 把变长 token、标签和多相机输入整理成动作预测训练 step 可以直接消费的 batch。 / `PaddedCollatorForActionPrediction` turns variable-length tokens, labels, and multi-camera inputs into a batch that the action-prediction training step can consume directly.

## 为什么重要 / Why this matters

中文：VLA 的样本同时有语言、图像、状态和动作。语言长度天然不同，图像有时是单 tensor、有时是多相机字典，而动作标签还需要用 `IGNORE_INDEX` 屏蔽 padding。这个 collator 把这些不规则输入压成一个稳定契约，让模型 forward 不必在每个 batch 里重新处理形状分支。

English: A VLA example combines language, images, state, and action labels. Language lengths vary, image inputs may be one tensor or a multi-camera dictionary, and padded action labels must be ignored by the loss. This collator turns those irregular examples into a stable model-facing contract.

## 代码 / The code

`openvla/openvla` — [`prismatic/util/data_utils.py`](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/prismatic/util/data_utils.py#L94-L142)

```python
@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        # For now, we only support Tokenizers with `padding_side = "right"` during training
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {
                k: torch.stack([pixel_values[idx][k] for idx in range(len(input_ids))]) for k in pixel_values[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        output = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output
```

## 逐行讲解 / What's happening

1. **第 101-107 行 / Lines 101-107**:
   - 中文: 先把每个 instance 的 token、label 和图像取出来；`dataset_name` 可选，但一旦存在就按样本保留。
   - English: The collator extracts tokens, labels, and images first. `dataset_name` is optional, but when present it is preserved per example.
2. **第 109-119 行 / Lines 109-119**:
   - 中文: 只支持 right padding，`input_ids` 用 pad token，`labels` 用 `IGNORE_INDEX`，再按 `model_max_length` 截断并生成 attention mask。
   - English: Training currently assumes right padding. Input IDs use the tokenizer’s pad token, labels use `IGNORE_INDEX`, both are truncated to the model limit, and the attention mask is derived from input IDs.
3. **第 121-132 行 / Lines 121-132**:
   - 中文: VLA 训练拒绝没有图像的样本，然后兼容单 tensor 和 `{camera_name: tensor}` 两种图像组织方式。
   - English: VLA training rejects image-less examples, then supports both a single image tensor and a `{camera_name: tensor}` multi-camera mapping.
4. **第 134-142 行 / Lines 134-142**:
   - 中文: 输出字段固定；数据集名字如果存在就附加，方便后续按数据源取 normalization 统计。
   - English: The output fields are stable, with dataset names added only when supplied so downstream code can select per-dataset normalization statistics.

## 类比 / The analogy

中文：像给不同长度的工具箱配同样大的运输托盘。短工具箱用 padding 填空，标签上的“空位”被标成不计分；多相机输入则像托盘上固定编号的几个格子。

English: Imagine placing toolboxes of different lengths on identical shipping pallets. Empty space is padded and marked as ignored for scoring, while multi-camera inputs occupy fixed, named slots on the pallet.

## 在 nanoVLA 中的位置 / Where this lives in your nanoVLA

中文：这是 `training-step` 的 advanced variant，位置就在 dataset transform 和 VLM/action head 的 `forward` 之间。上游给它的是每条样本的变长 token、action labels 和 camera tensors；下游训练 step 直接消费 `input_ids`、`attention_mask`、`labels` 和 `pixel_values`。省掉它，padding、相机字典和 loss mask 会泄漏到模型内部，后续换 tokenizer 或换相机数量都会变得危险。生产版还应补齐 left padding、混合精度 dtype、非连续 tensor、空 batch、action/state 的显式 schema 校验，以及跨数据集 normalization 的版本管理。

English: This is an advanced `training-step` variant between dataset transforms and the VLM/action head’s `forward`. It receives variable-length tokens, action labels, and camera tensors; the training step consumes the stable `input_ids`, `attention_mask`, `labels`, and `pixel_values` contract. Without it, padding, camera layout, and loss masking leak into the model and make tokenizer or embodiment changes fragile. Production code should add left-padding support, dtype checks, non-contiguous tensor handling, empty-batch validation, explicit action/state schemas, and versioned normalization metadata.

## 自己跑一遍 / Try it yourself

```python
IGNORE_INDEX = -100

def collate(instances, pad_id=0, max_len=5):
    width = min(max(len(x["input_ids"]) for x in instances), max_len)
    ids, labels = [], []
    for x in instances:
        ids.append(x["input_ids"][:width] + [pad_id] * (width - len(x["input_ids"])))
        labels.append(x["labels"][:width] + [IGNORE_INDEX] * (width - len(x["labels"])))
    mask = [[token != pad_id for token in row] for row in ids]
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}

print(collate([
    {"input_ids": [7, 8], "labels": [70, 80]},
    {"input_ids": [9, 10, 11], "labels": [90, 100, 110]},
]))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
{'input_ids': [[7, 8, 0], [9, 10, 11]], 'labels': [[70, 80, -100], [90, 100, 110]], 'attention_mask': [[True, True, False], [True, True, True]]}
```

中文：padding 的值可以进入 batch，但不能进入 loss；这就是 `input_ids` 和 `labels` 使用不同 padding value 的原因。

English: Padding may exist in the batch, but it must not contribute to the loss. That is why `input_ids` and `labels` use different padding values.

## 注意事项 / Caveats / when it breaks

- **所有样本都必须有视觉输入** / **Every sample must have visual input**: 这是 VLA 的显式 contract，不适合直接复用到 language-only 数据。
- **`pixel_values` 的结构必须整批一致** / **`pixel_values` must have one consistent structure per batch**: 不能把 tensor 样本和 dict 样本混在同一个 batch。
- **截断会同步影响 labels** / **Truncation must stay synchronized with labels**: 只截 `input_ids` 会让 token 和监督错位。

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **nanoVLM projector** / **nanoVLM projector**: 视觉输入先被规整成固定 token 宽度，再进入语言模型空间。
- **LeRobot processors** / **LeRobot processors**: 另一种做法是在 collator 之前把 state/action schema 和 normalization 固化。
- **openpi transforms** / **openpi transforms**: 把输入输出变换拆成可组合 pipeline，collator 只接收已经满足 schema 的数据。

## 延伸阅读 / Further reading

- [OpenVLA data_utils.py](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/prismatic/util/data_utils.py)
- [PyTorch pad_sequence](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html)
