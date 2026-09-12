---
date: 2026-09-06
topic: vla
source: vla
repo: NVIDIA/Isaac-GR00T
file: gr00t/data/state_action/state_action_processor.py
permalink: https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/gr00t/data/state_action/state_action_processor.py#L2597-L2681
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, vla, action-chunking, relative-action, absolute-action]
build_role: action-chunking advanced variant
---

# Isaac-GR00T ActionChunk：绝对动作和相对动作一键互转 / Isaac-GR00T ActionChunk: Convert Absolute and Relative Actions Both Ways

> **一句话 / In one line**: 这个 processor 先按 action type 选 chunk 结构，再用 reference frame 把相对动作和绝对动作互相转换。 / This processor first picks a chunk structure by action type, then uses a reference frame to convert relative and absolute actions in both directions.

## 为什么重要 / Why this matters

中文：chunked action 不是单纯把动作切块。真实机器人还要知道这段 chunk 是相对当前末端位姿的偏移，还是绝对目标。GR00T 把 chunk 形式、参考坐标系和动作格式转换放在同一个 processor 里，模型头和环境接口才不会互相猜。

English: Chunked actions are more than just sliced trajectories. A real robot also needs to know whether a chunk is an offset from the current end-effector pose or an absolute target. GR00T keeps chunk form, reference frame, and action-format conversion in one processor so the model head and environment interface do not have to guess each other’s semantics.

## 代码 / The code

`NVIDIA/Isaac-GR00T` — [`gr00t/data/state_action/state_action_processor.py`](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/gr00t/data/state_action/state_action_processor.py#L2597-L2681)

```python
def _convert_to_relative_action(self, action, action_format, reference_state):
    if self.action_type == ActionType.EEF:
        chunk = EndEffectorActionChunk.from_array(action)
        reference_frame = EndEffectorPose.from_action_format(
            reference_state, action_format
        )
    else:
        chunk = JointActionChunk([JointPose(m) for m in action])
        reference_frame = JointPose(reference_state)
    return chunk.relative_chunking(reference_frame=reference_frame).to(action_format)


def _convert_to_absolute_action(self, action, action_format, reference_state):
    if self.action_type == ActionType.EEF:
        chunk = EndEffectorActionChunk.from_array(action)
        reference_frame = EndEffectorPose.from_action_format(
            reference_state, action_format
        )
    else:
        chunk = JointActionChunk([JointPose(m) for m in action])
        reference_frame = JointPose(reference_state)
    return chunk.to_absolute_chunking(reference_frame=reference_frame).to(action_format)
```

## 逐行讲解 / What's happening

1. **第 2597-2637 行 / Lines 2597-2637**:
   - 中文: 先按 `ActionType` 分支，再把数组包装成 EEF 或 joint chunk。
   - English: The code branches on `ActionType` first, then wraps the array as either an EEF or joint chunk.
2. **第 2639-2660 行 / Lines 2639-2660**:
   - 中文: 相对动作不是“减一个向量”这么粗糙，而是先建 reference frame，再做 chunk-level relative transform。
   - English: Relative action conversion is not a simple vector subtraction; it builds a reference frame first and then applies a chunk-level relative transform.
3. **第 2661-2681 行 / Lines 2661-2681**:
   - 中文: 绝对动作路径对称地反向转换，最后再回到指定 action_format。
   - English: The absolute-action path mirrors the same structure in reverse, then converts back to the requested action format.

## 类比 / The analogy

中文：像把一串地图坐标和“相对这里往前走几步”的指令互相翻译，翻译前先确认当前原点在哪。

English: It is like translating between GPS coordinates and “walk forward a few steps from here,” but you first have to agree on where “here” is.

## 在 nanoVLA 中的位置 / Where this lives in your nanoVLA

中文：这是 `action-chunking` 的 advanced variant。它放在 nanoVLA 的动作头和机器人执行器之间：上游模型吐出 chunk，下游执行器需要知道这些 chunk 是相对量还是绝对量。没有这一层，你会把坐标系转换散进 policy、dataset 和 env 三处，最后谁都不敢改接口。生产版通常还会补时间对齐、关节限幅、速度/加速度约束和失败回滚。

English: This is an advanced variant of `action-chunking`. In your nanoVLA it sits between the action head and the robot executor: the upstream model emits chunks, and the downstream executor needs to know whether those chunks are relative or absolute. Without this layer, coordinate transforms leak into the policy, dataset, and environment separately, and nobody wants to touch the interface. A production version also adds time alignment, joint limits, velocity/acceleration constraints, and failure rollback.

## 自己跑一遍 / Try it yourself

```python
def to_relative(x, ref):
    return [round(v - r, 2) for v, r in zip(x, ref)]

def to_absolute(x, ref):
    return [round(v + r, 2) for v, r in zip(x, ref)]

ref = [10.0, 20.0, 30.0]
abs_action = [11.5, 21.0, 28.5]
rel_action = to_relative(abs_action, ref)
print(rel_action)
print(to_absolute(rel_action, ref))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[1.5, 1.0, -1.5]
[11.5, 21.0, 28.5]
```

中文：chunking 的难点不是切段，而是切完后还能明确每段相对哪个参考系。

English: The hard part of chunking is not slicing; it is keeping the reference frame of each slice explicit.
