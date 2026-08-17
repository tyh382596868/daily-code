---
date: 2026-08-17
topic: robotics
source: trending
repo: XiaomiRobotics/Xiaomi-Robotics-1
file: xr1/mibot/data/collate/custom_collate.py
permalink: https://github.com/XiaomiRobotics/Xiaomi-Robotics-1/blob/556cca33963a2b36d835a40374c3b4c8eef68401/xr1/mibot/data/collate/custom_collate.py#L16-L111
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, robotics, collate]
---

# Xiaomi XR1 CustomCollate：把多轮图文动作样本拼成一条训练序列 / Xiaomi XR1 CustomCollate: Pack Multimodal Action Samples into One Training Sequence

> **一句话 / In one line**: `CustomCollate` 用 Qwen3-VL processor 编码图文消息，重建 3D position ids，再把多个样本串成 varlen attention 需要的连续序列。 / `CustomCollate` encodes multimodal chat messages with a Qwen3-VL processor, rebuilds 3D position ids, then concatenates samples into the continuous sequence expected by varlen attention.

## 为什么重要 / Why this matters

VLA 训练的 collate 不只是 `default_collate`。图像 token、文字 token、状态 token 和动作 target 要在同一条序列里对齐，还要记录每个样本的边界。这里展示了一个生产味很强的 batch 组装方式。

VLA collation is not just `default_collate`. Image tokens, text tokens, state tokens, and action targets must align in one sequence, while sample boundaries are still recorded. This is a production-flavored way to assemble such batches.

## 代码 / The code

`XiaomiRobotics/Xiaomi-Robotics-1` — [`xr1/mibot/data/collate/custom_collate.py`](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1/blob/556cca33963a2b36d835a40374c3b4c8eef68401/xr1/mibot/data/collate/custom_collate.py#L16-L111)

```python
class CustomCollate:
    def __init__(self) -> None:
        self.max_length = int(os.environ.get("MAX_LENGTH", 4096))
        special_tokens = {"score": "<score>", "state": "<state>"}
        special_tokens.update({f"a_{index}": f"<a_{index}>" for index in range(60)})
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-VL-4B-Instruct", use_fast=True, extra_special_tokens=special_tokens
        )
        token_ids = self.processor.tokenizer.convert_tokens_to_ids(
            ["<score>", "<state>", "<a_0>", "<a_59>"]
        )
        if token_ids != [SCORE_ID, STATE_ID, STATE_ID + 1, STATE_ID + 60]:
            raise ValueError(f"unexpected action token ids: {token_ids}")

    def _position_ids(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = inputs["input_ids"]
        grids = inputs.get("image_grid_thw")
        if grids is None:
            return torch.arange(input_ids.shape[1]).view(1, 1, -1).expand(3, 1, -1)
        tokens = input_ids[0].tolist()
        positions, start = [], 0
        for grid in grids:
            image_start = tokens.index(IMAGE_ID, start)
            text_length = image_start - start
            base = positions[-1].max() + 1 if positions else 0
            positions.append(torch.arange(text_length).view(1, -1).expand(3, -1) + base)
            time, height, width = (int(value) for value in grid)
            height //= self.processor.image_processor.merge_size
            width //= self.processor.image_processor.merge_size
            temporal = torch.arange(time).view(-1, 1).expand(-1, height * width).flatten()
            rows = torch.arange(height).view(1, -1, 1).expand(time, -1, width).flatten()
            columns = torch.arange(width).view(1, 1, -1).expand(time, height, -1).flatten()
            positions.append(torch.stack([temporal, rows, columns]) + text_length + base)
            start = image_start + time * height * width
        if start < len(tokens):
            base = positions[-1].max() + 1
            positions.append(torch.arange(len(tokens) - start).view(1, -1).expand(3, -1) + base)
        return torch.cat(positions, dim=1).view(3, 1, -1)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        selected = []
        total_length = 0
        for item in batch:
            inputs = self.processor.apply_chat_template(
                item["messages"], tokenize=True, return_dict=True, return_tensors="pt", do_resize=False
            )
            length = inputs["input_ids"].shape[1]
            if total_length + length > self.max_length:
                continue
            state = (inputs["input_ids"][0] == STATE_ID).nonzero(as_tuple=False)
            im_starts = (inputs["input_ids"][0] == IM_START_ID).nonzero(as_tuple=False).flatten()
            im_starts = im_starts[im_starts < state[0, 0]] if state.numel() else im_starts[:0]
            if im_starts.numel() == 0:
                raise ValueError("cannot locate the action-conditioning turn")
            inputs["position_ids"] = self._position_ids(inputs)
            payload = {key: value for key, value in item.items() if key != "messages"}
            selected.append((inputs, payload, int(im_starts[-1])))
            total_length += length

        lengths = [inputs["input_ids"].shape[1] for inputs, _, _ in selected]
        offsets = [0] + list(itertools.accumulate(lengths))
        result = {
            "input_ids": torch.cat([inputs["input_ids"] for inputs, _, _ in selected], dim=1),
            "position_ids": torch.cat([inputs["position_ids"] for inputs, _, _ in selected], dim=2),
            "cu_seq_lens_q": torch.tensor(offsets, dtype=torch.int32),
            "cu_seq_lens_k": torch.tensor(offsets, dtype=torch.int32),
            "action_segments": torch.tensor(list(zip(offsets[:-1], offsets[1:])), dtype=torch.long),
            "action_vlm_condition_segments": torch.tensor(
                [[offsets[index], offsets[index] + end] for index, (_, _, end) in enumerate(selected)],
                dtype=torch.long,
            ),
        }
        return result
```

## 逐行讲解 / What's happening

1. **第 18-28 行 / Lines 18-28**: 中文: collator 注册 `<state>` 和 60 个动作 token，并检查 token id 连续性。 / English: The collator registers `<state>` plus 60 action tokens and checks that their token ids are contiguous.
2. **第 30-55 行 / Lines 30-55**: 中文: `_position_ids` 为文本 token 和图像网格 token 分别构造三轴位置。 / English: `_position_ids` builds three-axis positions for text tokens and image-grid tokens.
3. **第 57-83 行 / Lines 57-83**: 中文: batch 内样本按总 token 长度筛选，超过 `MAX_LENGTH` 的样本被跳过。 / English: Samples are selected under a total token budget; items that would exceed `MAX_LENGTH` are skipped.
4. **第 85-99 行 / Lines 85-99**: 中文: 多个样本沿序列维拼接，同时生成 `cu_seq_lens_*` 和 action segment 边界。 / English: Multiple samples are concatenated along the sequence dimension while `cu_seq_lens_*` and action segment boundaries are recorded.

## 类比 / The analogy

像把几段电影胶片接成一卷长片：画面和字幕都要接上，但剪辑师还要保留每一段的起止帧，否则放映机不知道哪里是一段样本。

It is like splicing several film reels into one long reel: frames and captions are joined, but the editor still keeps start and end frames for each segment so the projector knows the sample boundaries.

## 自己跑一遍 / Try it yourself

```python
import itertools

lengths = [5, 3, 4]
max_length = 9
selected, total = [], 0
for i, length in enumerate(lengths):
    if total + length > max_length:
        continue
    selected.append((i, length, 2 if i == 0 else 1))
    total += length

offsets = [0] + list(itertools.accumulate(length for _, length, _ in selected))
print(offsets)
print(list(zip(offsets[:-1], offsets[1:])))
print([[offsets[i], offsets[i] + end] for i, (_, _, end) in enumerate(selected)])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[0, 5, 8]
[(0, 5), (5, 8)]
[[0, 2], [5, 6]]
```

第三个样本会超过总长度预算，所以 batch 只保留前两个，同时保留它们在长序列中的边界。

The third sample would exceed the total length budget, so the batch keeps the first two and records their boundaries inside the long sequence.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **FlashAttention varlen** / **FlashAttention varlen**: `cu_seq_lens` 是把多段序列打包成一条长序列的常见接口。 / `cu_seq_lens` is a common interface for packing many sequences into one long sequence.
- **VLA action supervision** / **VLA action supervision**: action target 和 action mask 常在 collate 阶段对齐，因为模型 forward 只想看到规则 batch。 / Action targets and masks are often aligned during collation because model forward prefers regular batches.

## 注意事项 / Caveats / when it breaks

- **跳样本会改变 batch 组成** / **Skipping changes the batch**: 超长样本被 `continue` 掉，训练吞吐和样本分布会受 `MAX_LENGTH` 影响。 / Overlong samples are skipped, so throughput and sample distribution depend on `MAX_LENGTH`.
- **特殊 token id 是硬合同** / **Special token ids are a hard contract**: tokenizer 更新后如果 id 不连续，这段代码会直接报错。 / If a tokenizer update changes the expected id layout, this code fails fast.

## 延伸阅读 / Further reading

- [Xiaomi-Robotics-1 source](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1/blob/556cca33963a2b36d835a40374c3b4c8eef68401/xr1/mibot/data/collate/custom_collate.py#L16-L111)
