---
date: 2026-07-24
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/datasets/lerobot_dataset.py
permalink: https://github.com/huggingface/lerobot/blob/a0eb860d1e4d566157dd0eba11b5c86f9ea7d07a/src/lerobot/datasets/lerobot_dataset.py#L481-L510
difficulty: beginner
read_time: ~8 min
tags: [code-of-the-day, vla, lerobot, dataset, observation-window]
build_role: short-term-observation-memory advanced variant
---

# LeRobot Dataset slice：窗口读取复用单帧路径 / LeRobot Dataset Slice: Window Reads Reuse the Single-Frame Path

> **一句话 / In one line**: `LeRobotDataset.__getitem__` 看到 `slice` 时递归调用单帧 `self[item_idx]`，让窗口读取自动继承 delta-timestamp、视频解码和 transform 逻辑。 / `LeRobotDataset.__getitem__` handles a `slice` by recursively calling the single-frame `self[item_idx]`, so window reads inherit delta-timestamp, video decoding, and transform logic.

## 为什么重要 / Why this matters

VLA 训练经常需要连续窗口：历史观测、未来动作 chunk、重叠 rollout。最危险的做法是为 slice 写第二套数据读取逻辑，因为它很容易漏掉 padding mask、视频 decode 或 image transform。LeRobot 的改动很小，但设计清楚：slice 只是多个 scalar read。

VLA training often needs contiguous windows: observation history, future action chunks, or overlapping rollouts. The dangerous approach is a second data-reading path for slices, because it can miss padding masks, video decoding, or image transforms. LeRobot's change is tiny but clear: a slice is just repeated scalar reads.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/datasets/lerobot_dataset.py`](https://github.com/huggingface/lerobot/blob/a0eb860d1e4d566157dd0eba11b5c86f9ea7d07a/src/lerobot/datasets/lerobot_dataset.py#L481-L510)

```python
    def __getitem__(self, idx: int | slice) -> dict | list[dict]:
        """Return one frame or a slice of frames, with all transforms applied.

        Loads the frame from the underlying HF dataset, expands delta-timestamp
        windows, decodes video frames, and applies image transforms. Delegates
        the core logic to :class:`DatasetReader`.

        Args:
            idx: Integer index or slice into the possibly episode-filtered dataset.

        Returns:
            A frame dictionary for an integer index, or a list of frame
            dictionaries for a slice.

        Raises:
            RuntimeError: If the dataset is currently being recorded and
                :meth:`finalize` has not been called yet.
        """
        if self.writer is not None and not self._is_finalized:
            raise RuntimeError(
                "Cannot read from a dataset that is being recorded. Call finalize() first, then access items."
            )
        if isinstance(idx, slice):
            return [self[item_idx] for item_idx in range(*idx.indices(len(self)))]

        reader = self._ensure_reader()
        if reader.hf_dataset is None:
            # One-shot load after finalize()
            reader.load_and_activate()
        return reader.get_item(idx)
```

## 逐行讲解 / What's happening

1. **第 481 行 / Line 481 (`int | slice`)**: 中文: API 明确承认两种访问模式：单帧和窗口。 English: the API explicitly supports two access modes: one frame and a window.
2. **第 498-501 行 / Lines 498-501 (recording guard)**: 中文: 未 finalize 的在线写入数据集不能读，避免读到半成品 episode。 English: an unfinalized recording dataset cannot be read, preventing half-written episodes.
3. **第 502-503 行 / Lines 502-503 (`slice`)**: 中文: `idx.indices(len(self))` 统一处理负数、反向步长和越界。 English: `idx.indices(len(self))` normalizes negatives, reverse steps, and out-of-range bounds.
4. **第 505-510 行 / Lines 505-510 (`reader.get_item`)**: 中文: 单帧路径仍由 `DatasetReader` 负责，所以 slice 不复制复杂逻辑。 English: the scalar path remains owned by `DatasetReader`, so slice support duplicates no complex logic.

## 类比 / The analogy

像图书馆借一排书，不新开一套“批量借书系统”，而是按书号逐本走同一个扫码流程。这样每本书的借出记录、逾期规则和权限检查都一致。

It is like borrowing a row of books from a library by scanning each book through the same checkout flow. Every book keeps the same record, due-date rule, and permission check.

## 在 nanoVLA 中的位置 / Where this fits in nanoVLA

这是 `short-term-observation-memory` / data-window 层。它位于原始 episode 存储之后、batch collator 和模型 transform 之前。上游给它帧索引或 slice，下游拿到已经展开好的历史观测和 padding 标记；如果省掉这层，模型训练很容易把 episode 开头的缺失历史当成真实观测。

This belongs to the `short-term-observation-memory` / data-window layer. It sits after raw episode storage and before the batch collator and model transforms. Upstream provides frame indices or slices; downstream receives expanded history and padding flags. Without this layer, training can mistake missing history at episode starts for real observations.

## 自己跑一遍 / Try it yourself

```python
class ToyDataset:
    def __init__(self, values):
        self.values = values
    def __len__(self):
        return len(self.values)
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        prev_i = max(0, idx - 1)
        return {"state": [self.values[prev_i], self.values[idx]], "is_pad": idx == 0}

d = ToyDataset([10, 20, 30])
print(d[0])
print(d[:2])
print(d[2::-1])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
{'state': [10, 10], 'is_pad': True}
[{'state': [10, 10], 'is_pad': True}, {'state': [10, 20], 'is_pad': False}]
[{'state': [20, 30], 'is_pad': False}, {'state': [10, 20], 'is_pad': False}, {'state': [10, 10], 'is_pad': True}]
```

中文: slice 没有实现自己的 padding 规则，它复用了单帧路径，所以行为一致。 English: the slice path has no separate padding rule; it reuses scalar reads, so behavior stays consistent.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LeRobot delta_timestamps** / **LeRobot delta_timestamps**: 历史观测窗口也是在单帧读取时展开。 / observation history windows are expanded during scalar frame reads.
- **ACT temporal ensemble** / **ACT temporal ensemble**: 推理阶段的重叠 action chunk 也可以被看成多个单步预测的组合。 / overlapping action chunks at inference can also be treated as repeated single-step predictions.

## 注意事项 / Caveats / when it breaks

- **返回 list 而不是 batch tensor / Returns a list, not a batch tensor**: collator 仍要负责堆叠和 padding。 / the collator still owns stacking and padding.
- **长 slice 会重复开销 / Long slices repeat scalar costs**: 很长窗口可能需要更批量化的 reader。 / very long windows may need a more batched reader.
- **录制中不可读 / Cannot read while recording**: online dataset 要先 `finalize()`，否则 offset 和视频文件可能不完整。 / online datasets must call `finalize()` first because offsets and video files may be incomplete.

## 延伸阅读 / Further reading

- [LeRobot dataset slice commit](https://github.com/huggingface/lerobot/commit/a0eb860d1e4d566157dd0eba11b5c86f9ea7d07a)

