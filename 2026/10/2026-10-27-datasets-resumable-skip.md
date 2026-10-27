---
date: 2026-10-27
topic: huggingface
source: huggingface
repo: huggingface/datasets
file: src/datasets/iterable_dataset.py
permalink: https://github.com/huggingface/datasets/blob/ef61228a5756dcebfff8ee245d89c72822fb6ae6/src/datasets/iterable_dataset.py#L2148-L2213
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, streaming]
---

# Datasets skip：跳过也要写进状态 / Datasets Skip: Skipping Must Be Checkpointed Too

> **一句话 / In one line**: `SkipExamplesIterable` 把已跳数量放入 state，使流式恢复不会重新产出旧样本。 / `SkipExamplesIterable` stores the skip count so a resumed stream does not replay old samples.

## 为什么重要 / Why this matters

流式数据没有可随机访问的索引；训练中断后，仅靠重新执行 `skip(n)` 会与已消费进度脱节。

Streams have no random-access index. After interruption, re-running `skip(n)` alone loses the exact consumption position.

## 代码 / The code

`huggingface/datasets` — [`src/datasets/iterable_dataset.py`](https://github.com/huggingface/datasets/blob/ef61228a5756dcebfff8ee245d89c72822fb6ae6/src/datasets/iterable_dataset.py#L2148-L2213)

```python
def _init_state_dict(self):
    self._state_dict = {"skipped": 0,
        "examples_iterable": self.ex_iterable._init_state_dict(), "type": self.__class__.__name__}
    return self._state_dict

def __iter__(self):
    skipped = self._state_dict["skipped"] if self._state_dict else 0
    for key_example in self.ex_iterable:
        if skipped + 1 <= self.n:
            skipped += 1
            if self._state_dict: self._state_dict["skipped"] = skipped
        else:
            yield key_example
```

## 逐行讲解 / What's happening

状态同时保存包装器的计数和底层 iterable 的状态；每跳过一个样本立即记账。

State keeps both the wrapper count and the child iterable state; each skipped item is recorded immediately.

## 类比 / The analogy

像有书签的有声书：书签不只说“第十章”，还要标明这一章已听到哪里。

It is an audiobook bookmark that records the position inside the chapter, not merely the chapter number.

## 自己跑一遍 / Try it yourself

```python
state = {"skipped": 0}; n = 2
for x in range(5):
    if state["skipped"] < n: state["skipped"] += 1
    else: print(x)
print(state)
```

运行 / Run with `python try.py`。输出 / Expected output: `2 3 4` 与 `{'skipped': 2}`。

## 注意事项 / Caveats / when it breaks

- **上游可恢复性 / Upstream resumability**：只有底层 source 也能恢复时，完整状态才有意义。 / Full recovery requires the underlying source to be resumable too.
