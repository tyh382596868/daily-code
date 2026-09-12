---
date: 2026-07-28
topic: huggingface
source: huggingface
repo: huggingface/datasets
file: src/datasets/iterable_dataset.py
permalink: https://github.com/huggingface/datasets/blob/b305031c4ec070e5e657049abaa70659883958e2/src/datasets/iterable_dataset.py#L1901-L2005
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, datasets, streaming, shuffle-buffer]
---

# Datasets shuffle buffer：流式数据也能近似打乱 / Datasets Shuffle Buffer: Approximate Shuffle for Streams

> **一句话 / In one line**: `BufferShuffledExamplesIterable` 用固定大小内存池做 reservoir-like shuffle，让无限流也能被稳定、可恢复地打乱。 / `BufferShuffledExamplesIterable` uses a fixed-size memory pool for reservoir-like shuffling, making even streams shuffleable in a stable, resumable way.

## 为什么重要 / Why this matters

普通数据集可以一次性随机置换索引，但流式数据不能把所有样本装进内存。HF datasets 这段实现保留一个 `mem_buffer`：池满后随机吐出一个旧样本，再用新样本补位，最后把剩余池子洗牌输出。

Regular datasets can randomly permute all indices, but streaming datasets cannot load every sample into memory. This implementation keeps a `mem_buffer`: once full, it randomly yields one old sample, replaces it with a new sample, and shuffles the remaining buffer at the end.

## 代码 / The code

`huggingface/datasets` — [`src/datasets/iterable_dataset.py`](https://github.com/huggingface/datasets/blob/b305031c4ec070e5e657049abaa70659883958e2/src/datasets/iterable_dataset.py#L1901-L2005)

```python
class BufferShuffledExamplesIterable(_BaseExamplesIterable):
    def __init__(self, ex_iterable: _BaseExamplesIterable, buffer_size: int, generator: np.random.Generator):
        super().__init__()
        self.ex_iterable = ex_iterable
        self.buffer_size = buffer_size
        self.generator = generator

    @staticmethod
    def _iter_random_indices(rng: np.random.Generator, buffer_size: int, random_batch_size=1000) -> Iterator[int]:
        while True:
            yield from (int(i) for i in rng.integers(0, buffer_size, size=random_batch_size))

    def __iter__(self):
        buffer_size = self.buffer_size
        rng = deepcopy(self.generator)
        indices_iterator = self._iter_random_indices(rng, buffer_size)
        # this is the shuffle buffer that we keep in memory
        mem_buffer = []
        for x in self.ex_iterable:
            if len(mem_buffer) == buffer_size:  # if the buffer is full, pick and example from it
                i = next(indices_iterator)
                yield mem_buffer[i]
                mem_buffer[i] = x  # replace the picked example by a new one
            else:  # otherwise, keep filling the buffer
                mem_buffer.append(x)
        # when we run out of examples, we shuffle the remaining examples in the buffer and yield them
        rng.shuffle(mem_buffer)
        yield from mem_buffer

    def shuffle_data_sources(self, generator: np.random.Generator) -> "BufferShuffledExamplesIterable":
        """Shuffle the wrapped examples iterable as well as the shuffling buffer."""
        return BufferShuffledExamplesIterable(
            self.ex_iterable.shuffle_data_sources(generator), buffer_size=self.buffer_size, generator=self.generator
        )

    def shard_data_sources(self, num_shards: int, index: int, contiguous=True) -> "BufferShuffledExamplesIterable":
        """Keep only the requested shard."""
        return BufferShuffledExamplesIterable(
            self.ex_iterable.shard_data_sources(num_shards, index, contiguous=contiguous),
            buffer_size=self.buffer_size,
            generator=self.generator,
        )
```

## 逐行讲解 / What's happening

1. **第 1901-1906 行 / Lines 1901-1906 (`wrapper`)**:
   - 中文: 这个 iterable 包住上游 iterable，只新增 buffer size 和随机数生成器。
   - English: This iterable wraps an upstream iterable and only adds buffer size plus a random generator.
2. **第 1943-1947 行 / Lines 1943-1947 (`random indices`)**:
   - 中文: 随机索引批量生成，避免每个样本都调用一次 RNG API。
   - English: Random indices are generated in batches to avoid one RNG API call per sample.
3. **第 1948-1963 行 / Lines 1948-1963 (`stream shuffle`)**:
   - 中文: buffer 未满先填；满了就随机弹出旧样本，再把新样本放进同一个槽。
   - English: The buffer fills first; once full, it randomly emits an old sample and places the new one into that slot.
4. **第 1982-1994 行 / Lines 1982-1994 (`compose`)**:
   - 中文: shuffle 和 shard 都继续包裹上游 iterable，所以变换可以组合。
   - English: Shuffle and shard continue wrapping the upstream iterable, so transformations remain composable.

## 类比 / The analogy

这像洗一条不断进门的传送带：你桌上最多放 100 张牌，每来一张新牌就随机拿走桌上一张，再把新牌补进去。

It is like shuffling a conveyor belt: your table holds at most 100 cards; when a new card arrives, you randomly remove one card from the table and put the new card in its place.

## 自己跑一遍 / Try it yourself

```python
import random

def buffer_shuffle(xs, size, seed=0):
    rng = random.Random(seed)
    buf = []
    for x in xs:
        if len(buf) == size:
            i = rng.randrange(size)
            yield buf[i]
            buf[i] = x
        else:
            buf.append(x)
    rng.shuffle(buf)
    yield from buf

print(list(buffer_shuffle(range(10), 3, seed=7)))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
[1, 0, 3, 2, 4, 7, 6, 9, 5, 8]
```

输出不是全局完美洗牌，但只用 3 个槽就让局部顺序明显随机化了。

The result is not a perfect global permutation, but three slots are enough to make local order visibly randomized.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **PyTorch DataLoader prefetch** / **PyTorch DataLoader prefetch**: 中文: 也用有限队列在吞吐和内存之间折中。 / English: It also uses bounded queues to trade memory for throughput.
- **Streaming replay buffers** / **Streaming replay buffers**: 中文: RL 经验回放也常用固定容量池近似随机采样。 / English: RL replay buffers also use fixed-capacity pools for approximate random sampling.

## 注意事项 / Caveats / when it breaks

- **buffer 太小** / **Tiny buffers**: 中文: buffer size 越小，越接近原始顺序。 / English: The smaller the buffer, the closer the output stays to original order.
- **恢复语义** / **Resume semantics**: 中文: state dict 不保存 buffer 内容，恢复时会重新填池。 / English: The state dict does not store buffer contents, so resume refills the pool.

## 延伸阅读 / Further reading

- [huggingface/datasets source](https://github.com/huggingface/datasets/blob/b305031c4ec070e5e657049abaa70659883958e2/src/datasets/iterable_dataset.py#L1901-L2005)
