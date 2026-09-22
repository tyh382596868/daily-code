---
date: 2026-09-22
topic: huggingface
source: huggingface
repo: huggingface/datasets
file: src/datasets/iterable_dataset.py
permalink: https://github.com/huggingface/datasets/blob/111c1ad2145c3c59c3ca92bec89a7d5d23035861/src/datasets/iterable_dataset.py#L479-L607
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, huggingface, datasets, streaming, arrow, resumability]
---

# Datasets RebatchedArrow：流式 chunk 也能拼成固定 batch / Datasets RebatchedArrow: Turn Streaming Chunks into Fixed Batches

> **一句话 / In one line**: `RebatchedArrowExamplesIterable` 把大小不一的 Arrow chunk 拼成固定 batch，遇到跨界 chunk 就切开，并把跳过位置写进 state dict。 / `RebatchedArrowExamplesIterable` merges uneven Arrow chunks into fixed-size batches, slices chunks that cross a boundary, and records resume offsets in its state dict.

## 为什么重要 / Why this matters

流式数据的上游通常决定 chunk 大小，训练器却希望拿到统一的 batch。简单地把 chunk 直接 yield 出去会让 batch shape 抖动；简单地丢掉尾部又会丢数据。这个类把“流式读取”和“固定批次”接起来，还考虑了中断恢复。

Streaming sources often decide their own chunk sizes while trainers want fixed-size batches. Yielding chunks directly creates unstable batch shapes; dropping every tail loses data. This iterable bridges streaming reads and fixed batches while keeping interruption recovery in mind.

关键状态有三个：已经跳过了多少 chunk、上一个 chunk 已经裁掉多少行、已经发出了多少 batch。尤其是 `cropped_chunk_length`，它表示上次 yield 时把一个 chunk 切成两半，下一次恢复必须从剩余半段继续。

Three pieces of state matter: how many chunks were skipped, how many rows were cropped from the previous chunk, and how many batches were emitted. `cropped_chunk_length` is especially important: after a chunk is split across a batch boundary, resume must start from its remainder.

## 代码 / The code

`huggingface/datasets` — [`src/datasets/iterable_dataset.py`](https://github.com/huggingface/datasets/blob/111c1ad2145c3c59c3ca92bec89a7d5d23035861/src/datasets/iterable_dataset.py#L479-L607)

```python
class RebatchedArrowExamplesIterable(_BaseExamplesIterable):
    def __init__(
        self,
        ex_iterable: _BaseExamplesIterable,
        batch_size: Optional[int],
        drop_last_batch: bool = False,
        force_convert_to_arrow: bool = False,
    ):
        super().__init__()
        self.ex_iterable = ex_iterable
        self.batch_size = batch_size
        self.drop_last_batch = drop_last_batch
        self.force_convert_to_arrow = force_convert_to_arrow

    def _init_state_dict(self) -> dict:
        self._state_dict = {
            "examples_iterable": self.ex_iterable._init_state_dict(),
            "previous_state": None,
            "batch_idx": 0,
            "num_chunks_since_previous_state": 0,
            "cropped_chunk_length": 0,
            "type": self.__class__.__name__,
        }
        return self._state_dict

    def _iter_arrow(self) -> Iterator[tuple[Key, pa.Table]]:
        if self._state_dict and self._state_dict["previous_state"]:
            self.ex_iterable.load_state_dict(self._state_dict["previous_state"])
        if self.ex_iterable.iter_arrow:
            iterator = self.ex_iterable.iter_arrow()
        elif self.force_convert_to_arrow:
            iterator = _convert_to_arrow(self.ex_iterable, batch_size=1)
        else:
            raise RuntimeError("underlying iterable does not provide iter_arrow()")

        if self.batch_size is None or self.batch_size <= 0:
            if self._state_dict and self._state_dict["batch_idx"] > 0:
                return
            all_pa_table = pa.concat_tables([pa_table for _, pa_table in iterator])
            if self._state_dict:
                self._state_dict["batch_idx"] = 1
            yield "all", all_pa_table
            return

        keys_buffer = []
        chunks_buffer = []
        chunks_buffer_size = 0
        num_chunks_to_skip = self._state_dict["num_chunks_since_previous_state"] if self._state_dict else 0
        chunk_length_to_crop = self._state_dict["cropped_chunk_length"] if self._state_dict else 0

        if self._state_dict:
            previous_state = self.ex_iterable.state_dict()
            self._state_dict["previous_state"] = previous_state

        for key, pa_table in iterator:
            for num_chunks_since_previous_state, chunk in enumerate(pa_table.to_reader(max_chunksize=self.batch_size)):
                if num_chunks_to_skip > 1:
                    num_chunks_to_skip -= 1
                    continue
                elif num_chunks_to_skip == 1 and chunk_length_to_crop == 0:
                    num_chunks_to_skip -= 1
                    continue
                elif num_chunks_to_skip == 1 and chunk_length_to_crop > 0:
                    chunk = chunk.slice(chunk_length_to_crop, len(chunk) - chunk_length_to_crop)
                    num_chunks_to_skip = 0
                    chunk_length_to_crop = 0
                if len(chunk) == 0:
                    continue

                if chunks_buffer_size + len(chunk) < self.batch_size:
                    keys_buffer.append(key)
                    chunks_buffer.append(chunk)
                    chunks_buffer_size += len(chunk)
                    continue
                elif chunks_buffer_size + len(chunk) == self.batch_size:
                    keys_buffer.append(key)
                    chunks_buffer.append(chunk)
                    new_key = "_".join(str(_key) for _key in keys_buffer)
                    if self._state_dict:
                        self._state_dict["batch_idx"] += 1
                        self._state_dict["num_chunks_since_previous_state"] += len(chunks_buffer)
                        self._state_dict["cropped_chunk_length"] = 0
                    yield new_key, pa.Table.from_batches(chunks_buffer)
                    keys_buffer, chunks_buffer, chunks_buffer_size = [], [], 0
                else:
                    cropped_chunk_length = self.batch_size - chunks_buffer_size
                    keys_buffer.append(f"{key}[:{cropped_chunk_length}]")
                    chunks_buffer.append(chunk.slice(0, cropped_chunk_length))
                    new_key = "_".join(str(_key) for _key in keys_buffer)
                    if self._state_dict:
                        self._state_dict["batch_idx"] += 1
                        self._state_dict["cropped_chunk_length"] = cropped_chunk_length
                    yield new_key, pa.Table.from_batches(chunks_buffer)
                    keys_buffer = [f"{key}[{cropped_chunk_length}:]"]
                    chunks_buffer = [chunk.slice(cropped_chunk_length, len(chunk) - cropped_chunk_length)]
                    chunks_buffer_size = len(chunk) - cropped_chunk_length

        if not self.drop_last_batch and chunks_buffer:
            new_key = "_".join(str(_key) for _key in keys_buffer)
            if self._state_dict:
                self._state_dict["batch_idx"] += 1
                self._state_dict["cropped_chunk_length"] = 0
            yield new_key, pa.Table.from_batches(chunks_buffer)
```

## 逐行讲解 / What's happening

1. **第 505-514 行 / Lines 505-514 (resume state)**:
   - 中文: state 同时保存底层 iterable 的状态和 re-batch 自己的边界状态；两层都恢复，才能继续读到同一行。
   - English: The state stores both the wrapped iterable's position and the re-batcher's boundary state. Both layers must resume together to reach the same row.
2. **第 547-559 行 / Lines 547-559 (skip and crop)**:
   - 中文: 如果恢复点落在已经被切过的 chunk 里，先 `slice` 掉已消费的前缀；空 chunk 则直接跳过。
   - English: If resume lands inside a previously split chunk, `slice` removes the consumed prefix first; empty chunks are skipped.
3. **第 562-575 行 / Lines 562-575 (exact fit)**:
   - 中文: 小于 batch size 就先缓存；刚好等于时 yield 一个 Arrow table，并把 buffer 清空。
   - English: Smaller chunks are buffered; an exact fit yields one Arrow table and resets the buffers.
4. **第 582-597 行 / Lines 582-597 (spillover)**:
   - 中文: 大 chunk 被切成“填满当前 batch 的前半段”和“留给下一 batch 的后半段”。`cropped_chunk_length` 记录的是前半段长度。
   - English: An oversized chunk is split into the prefix that fills the current batch and the remainder for the next one. `cropped_chunk_length` records the prefix length.
5. **第 600-607 行 / Lines 600-607 (tail policy)**:
   - 中文: `drop_last_batch=False` 时保留最后不满的 batch；训练配置可以显式选择丢掉它。
   - English: With `drop_last_batch=False`, the final partial batch is kept; training configuration can explicitly choose to drop it.

## 类比 / The analogy

这像把不同尺寸的散货装进固定容量的集装箱。箱子没装满就继续装；刚好装满就封箱；一袋货太大时，先切出能填满当前箱子的部分，剩下的贴上“下一箱”标签。

It is like packing loose cargo into fixed-capacity containers. Keep filling an unfinished container, seal an exact fit, and when one shipment crosses the boundary, put the prefix in the current container and label the remainder for the next one.

## 自己跑一遍 / Try it yourself

```python
def rebatch(chunks, size):
    buffer = []
    for chunk in chunks:
        while chunk:
            take = min(size - len(buffer), len(chunk))
            buffer += chunk[:take]
            chunk = chunk[take:]
            if len(buffer) == size:
                yield buffer
                buffer = []
    if buffer:
        yield buffer

print(list(rebatch([[0, 1], [2, 3, 4], [5]], 3)))
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
[[0, 1, 2], [3, 4, 5]]
```

中文: 示例里一个输入 chunk 被拆到了两个 batch；真实实现还要把这个拆分点序列化。 / English: One input chunk crosses two output batches here; the real implementation additionally serializes that split point.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **DataLoader batch collation** / **DataLoader batch collation**: 中文: collate 也把不规则样本整理成稳定结构，但通常不负责流式 resume。 / English: Collation also normalizes irregular samples, but usually does not own streaming resume.
- **Arrow record batches** / **Arrow record batches**: 中文: Arrow table 可以零拷贝切片，适合在 batch 边界上做轻量裁剪。 / English: Arrow tables can be sliced cheaply, which makes them useful at batch boundaries.
- **Kafka consumer offsets** / **Kafka consumer offsets**: 中文: 消费者 offset 与 batch 内部的 crop state 是同一种“恢复到精确位置”的问题。 / English: Consumer offsets and intra-batch crop state solve the same “resume at an exact position” problem.

## 注意事项 / Caveats / when it breaks

- **state 必须和底层 iterator 成对保存** / **State must pair with the wrapped iterator**: 中文: 只保存 `batch_idx` 不够，底层 shard 位置不对仍会重复或跳过数据。 / English: Saving only `batch_idx` is insufficient; a wrong underlying shard position still duplicates or skips data.
- **Arrow schema 必须兼容** / **Arrow schemas must agree**: 中文: `pa.Table.from_batches` 要求列结构兼容，多个来源拼接前要先统一 features。 / English: `pa.Table.from_batches` expects compatible column structure, so features must be normalized across sources.
- **无界 batch 很危险** / **Unbounded batches are risky**: 中文: `batch_size=None` 会把所有 table 拼到内存里，只适合小数据或明确的全量操作。 / English: `batch_size=None` concatenates every table in memory and is suitable only for small or deliberately whole-dataset operations.

## 延伸阅读 / Further reading

- [Datasets `RebatchedArrowExamplesIterable`](https://github.com/huggingface/datasets/blob/111c1ad2145c3c59c3ca92bec89a7d5d23035861/src/datasets/iterable_dataset.py#L479-L607)
- [Hugging Face streaming datasets](https://huggingface.co/docs/datasets/stream)
- [Apache Arrow record batches](https://arrow.apache.org/docs/python/generated/pyarrow.RecordBatch.html)
