---
date: 2026-08-07
topic: huggingface
source: huggingface
repo: huggingface/datasets
file: src/datasets/iterable_dataset.py
permalink: https://github.com/huggingface/datasets/blob/main/src/datasets/iterable_dataset.py#L1866-L2071
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, datasets, streaming]
---

# Datasets skip/take：流式切片也要可恢复 / Datasets skip/take: Streaming Slices Must Be Resumable

> **一句话 / In one line**: `SkipExamplesIterable` 和 `TakeExamplesIterable` 把“跳过/截取多少条”写进 state dict。 / `SkipExamplesIterable` and `TakeExamplesIterable` record skipped/taken counts in a state dict.

## 为什么重要 / Why this matters

流式 dataset 不能随机寻址。`skip(1000)` 和 `take(1000)` 看起来像数组切片，其实必须边迭代边计数；训练中断后恢复时，计数还要回到正确位置。

A streaming dataset cannot seek like an array. `skip(1000)` and `take(1000)` look like slicing, but they must count while iterating; after interruption, resume must restore the right position.

## 代码 / The code

`huggingface/datasets` — [`src/datasets/iterable_dataset.py`](https://github.com/huggingface/datasets/blob/main/src/datasets/iterable_dataset.py#L1866-L2071)

```python
def _init_state_dict(self):
    self._state_dict = {"skipped": 0, "examples_iterable": ...}

def __iter__(self):
    skipped = self._state_dict["skipped"] if self._state_dict else 0
    for item in self.ex_iterable:
        if skipped + 1 <= self.n:
            skipped += 1
            self._state_dict["skipped"] = skipped
        else:
            yield item
```

## 逐行讲解 / What's happening

1. **第 1866-1881 行 / Lines 1866-1881 (wrapper)**:
   - 中文: `SkipExamplesIterable` 包住底层 iterable，只额外保存 `n` 和 sharding/shuffling 策略。
   - English: `SkipExamplesIterable` wraps the source iterable and stores only `n` plus sharding/shuffling policy.
2. **第 1893-1908 行 / Lines 1893-1908 (resume counter)**:
   - 中文: `skipped` 写入 `_state_dict`，恢复时不会重新跳错数量。
   - English: `skipped` is written into `_state_dict`, so resume does not skip the wrong number.
3. **第 1945-1953 行 / Lines 1945-1953 (sharding)**:
   - 中文: 分 shard 时，skip 数量也按 shard 拆分，避免每个 worker 都跳全量。
   - English: When sharding, the skip count is split by shard so every worker does not skip the full amount.
4. **第 2017-2071 行 / Lines 2017-2071 (take)**:
   - 中文: `TakeExamplesIterable` 是镜像逻辑：达到 `n` 后停止，并记录 `taken`。
   - English: `TakeExamplesIterable` mirrors the logic: stop at `n` and record `taken`.

## 类比 / The analogy

像排队取号：流式队伍不能直接跳到第 1000 个人，只能数过去；如果中途停电，计数器必须保存下来。

It is like counting people in a line: a streaming queue cannot jump to person 1000; it must count. If power fails, the counter must be saved.

## 自己跑一遍 / Try it yourself

```python
def stream():
    for i in range(10):
        yield i

def skip_take(xs, skip, take):
    out = []
    for i, x in enumerate(xs):
        if i < skip:
            continue
        if len(out) >= take:
            break
        out.append(x)
    return out

print(skip_take(stream(), 3, 4))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[3, 4, 5, 6]
```

数组切片的外观背后，是逐条计数的流式状态机。

Behind the array-slice appearance is a per-example streaming state machine.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Streaming shuffle buffers** / **Streaming shuffle buffers**: 也用小状态近似数组操作。 / They also use small state to approximate array operations.
- **Distributed dataloading** / **Distributed dataloading**: rank 切片必须和 skip/take 顺序一致。 / Rank slicing must agree with skip/take order.

## 注意事项 / Caveats / when it breaks

- **skip/take 会冻结数据源顺序** / **skip/take can freeze source order**: 后续 shuffle 数据源可能被禁止，避免跳到错误 shard。 / Later data-source shuffling can be disallowed to avoid skipping the wrong shard.
- **Arrow batch 要切表** / **Arrow batches need table slicing**: batch 模式不是一条条 yield，而是要按 offset 切 `pa_table`。 / Batch mode slices `pa_table` by offset instead of yielding examples one by one.

## 延伸阅读 / Further reading

- [HF Datasets iterable slicing](https://github.com/huggingface/datasets/blob/main/src/datasets/iterable_dataset.py#L1866-L2071)

