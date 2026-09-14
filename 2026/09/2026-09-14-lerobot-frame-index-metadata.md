---
date: 2026-09-14
topic: robotics
source: tracked
repo: huggingface/lerobot
file: src/lerobot/processor/converters.py
permalink: https://github.com/huggingface/lerobot/blob/8c894413c0967d83624d17a440afd70754fddb01/src/lerobot/processor/converters.py#L166-L194
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, lerobot, metadata, processor-conversion, frame-index]
---

# LeRobot frame_index：让处理器不丢时间坐标 / LeRobot frame_index: Keep Time Coordinates Through Processors

> **一句话 / In one line**: LeRobot 把 `frame_index` 纳入 complementary-data 契约，让 batch 和 transition 来回转换时保住每一帧的身份。 / LeRobot adds `frame_index` to the complementary-data contract so batch/transition conversion preserves each frame's identity.

## 为什么重要 / Why this matters

中文：机器人数据流里，state 和 action 不是悬空的数值，它们属于某个 episode、某个时间戳、某一帧。如果 processor 在 batch 转 transition 的过程中只保留模型输入，`frame_index` 这类元数据就会悄悄消失，后面的对齐、回放和诊断都会变得不可信。这个改动很小，但它把“模型要算什么”和“这份数据来自哪里”分成了两条清晰的通道。

English: Robot batches are not just tensors. Every state and action belongs to an episode, a timestamp, and a frame. If a processor keeps only model inputs while converting a batch into a transition, metadata such as `frame_index` disappears silently, making alignment, replay, and debugging unreliable. This small contract change keeps model data and provenance on separate, explicit paths.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/processor/converters.py`](https://github.com/huggingface/lerobot/blob/8c894413c0967d83624d17a440afd70754fddb01/src/lerobot/processor/converters.py#L166-L194)

```python
_COMPLEMENTARY_KEYS = (
    "task",
    "index",
    "task_index",
    "episode_index",
    "frame_index",
    "timestamp",
    "language_persistent",
    "language_events",
    MESSAGES_RENDERED,
    "message_streams",
    "target_message_indices",
    # Text-generation request keys: carried into complementary_data so a prompt-formatting
    # processor step can read the kind and rewrite QUERY_TEXT.
    QUERY_KIND,
    QUERY_TEXT,
)


def _extract_complementary_data(batch: dict[str, Any]) -> dict[str, Any]:
    """Extract complementary data from a batch dictionary.

    Includes padding flags (any key containing ``_is_pad``) plus the fixed
    set of metadata / language keys defined in ``_COMPLEMENTARY_KEYS`` —
    each only when present in ``batch``.
    """
    pad_keys = {k: v for k, v in batch.items() if "_is_pad" in k}
    extras = {k: batch[k] for k in _COMPLEMENTARY_KEYS if k in batch}
    return {**pad_keys, **extras}
```

## 逐行讲解 / What's happening

1. **第 166-182 行 / Lines 166-182 (`_COMPLEMENTARY_KEYS`)**:
   - 中文：这是一个显式白名单。`frame_index` 和 `episode_index` 一样，被定义为处理器之间必须携带的上下文，而不是随手丢进 observation 的普通字段。
   - English: This is an explicit allowlist. Like `episode_index`, `frame_index` is declared as processor context rather than an incidental observation field.
2. **第 185-191 行 / Lines 185-191 (`_extract_complementary_data`)**:
   - 中文：函数的文档把两类数据说清楚：任何包含 `_is_pad` 的 padding 标志，以及白名单中的元数据和语言键。
   - English: The docstring separates two classes of data: padding flags discovered by suffix, and metadata/language keys selected by the allowlist.
3. **第 192 行 / Line 192 (`pad_keys`)**:
   - 中文：padding 标志采用约定式发现，适合一组数量会扩展的字段，不需要每新增一个相机或模态就改白名单。
   - English: Padding flags are discovered by convention, which scales when new cameras or modalities add more padded fields.
4. **第 193-194 行 / Lines 193-194 (`extras` and the merge)**:
   - 中文：`if k in batch` 让缺失元数据保持合法；最后合并出的字典只包含当前 batch 真正提供的补充信息。
   - English: `if k in batch` keeps missing metadata valid, and the final dictionary contains only complementary values that this batch actually supplied.
5. **从修复到不变量 / From fix to invariant**:
   - 中文：`batch_to_transition` 应该把这里的结果放进 `COMPLEMENTARY_DATA`，`transition_to_batch` 再原样取回。于是“有 `frame_index` 的输入，往返后仍有同一个 `frame_index`”成为可测试的不变量。
   - English: `batch_to_transition` can place this result in `COMPLEMENTARY_DATA`, and `transition_to_batch` can restore it. The testable invariant becomes: a batch with `frame_index` has the same metadata after a round trip.

## 类比 / The analogy

中文：把一卷胶片送进剪辑台。画面帧是主要内容，但每格胶片旁边还贴着时间码和镜头编号。剪辑台可以改变画面格式，却不能把编号撕掉，否则你无法知道动作发生在整段轨迹的哪个位置。

English: Think of a film reel going through an editing desk. The image is the main payload, but every frame also carries a timecode and shot number. The editor may change the image format, but must keep the labels or the frame can no longer be placed back into the trajectory.

## 自己跑一遍 / Try it yourself

```python
COMPLEMENTARY_KEYS = ("episode_index", "frame_index", "timestamp")


def extract(batch):
    pads = {k: v for k, v in batch.items() if "_is_pad" in k}
    extras = {k: batch[k] for k in COMPLEMENTARY_KEYS if k in batch}
    return {**pads, **extras}


batch = {"state": [0.1, 0.2], "episode_index": 4, "frame_index": 17, "image_is_pad": False}
print(extract(batch))
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
{'image_is_pad': False, 'episode_index': 4, 'frame_index': 17}
```

中文：注意 `state` 没有进入结果，因为它是主数据；`frame_index` 则作为补充上下文被保留下来。这个边界正是 processor pipeline 能稳定组合的原因。

English: Notice that `state` stays out because it is the main payload, while `frame_index` survives as context. That boundary is what lets processors compose without losing provenance.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LeRobot episode_index** / **LeRobot episode_index**: episode identity follows the same complementary-data path. / episode identity travels through the same complementary-data path.
- **RLDS step metadata** / **RLDS step metadata**: trajectory readers keep step keys beside observations and actions. / Trajectory readers keep step keys beside observations and actions.
- **Distributed tracing context** / **Distributed tracing context**: request IDs are carried beside business payloads so downstream services can correlate work. / Request IDs travel beside business payloads so downstream services can correlate work.

## 注意事项 / Caveats / when it breaks

- **白名单要及时更新** / **Keep the allowlist current**: 新增的 episode、frame 或语言字段若不加入列表，转换会静默丢数据。 / New episode, frame, or language fields disappear silently unless added to the list.
- **缺失和空值不是一回事** / **Missing is not the same as empty**: `if k in batch` 保留了空列表或零值；下游不要把它们误判成“没有字段”。 / `if k in batch` preserves empty lists and zero values; downstream code must not confuse them with absent keys.
- **容器身份要谨慎** / **Be careful with container identity**: 当前逻辑传递原对象；若 processor 需要隔离修改，应明确复制策略。 / The current path passes the original object; processors that mutate values need an explicit copy policy.

## 延伸阅读 / Further reading

- [LeRobot converters.py](https://github.com/huggingface/lerobot/blob/8c894413c0967d83624d17a440afd70754fddb01/src/lerobot/processor/converters.py)
- [LeRobot frame_index regression test](https://github.com/huggingface/lerobot/blob/8c894413c0967d83624d17a440afd70754fddb01/tests/processor/test_converters.py#L246-L255)
