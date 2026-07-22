---
date: 2026-07-08
topic: huggingface
source: huggingface
repo: huggingface/datasets
file: src/datasets/iterable_dataset.py
permalink: https://github.com/huggingface/datasets/blob/41adfd0f9ee9ba3a6b4f719d5b551c5b19ae45e2/src/datasets/iterable_dataset.py#L1205-L1278
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, datasets, streaming]
---

# Datasets 随机轮转数据源：流式混合也要可恢复 / Datasets Random Source Cycling: Streaming Mixtures Must Be Resumable

> **一句话 / In one line**: `RandomlyCyclingMultiSourcesExamplesIterable` 不直接拼接数据源，而是用可保存状态的随机 index 流选择下一个 source。 / `RandomlyCyclingMultiSourcesExamplesIterable` does not concatenate sources; it samples a resumable stream of source indices.

## 为什么重要 / Why this matters

流式训练经常要混合多个数据源，比如一部分 web 文本、一部分代码、一部分高质量指令数据。简单 `random.choice()` 能跑，但 checkpoint 后很难从同一个位置恢复。HF datasets 这段代码把随机数状态和 batch 内 offset 都放进 state dict，所以 iterable dataset 可以暂停、保存、再继续。

Streaming training often mixes multiple sources, such as web text, code, and high-quality instruction data. A plain `random.choice()` works until you need to resume from a checkpoint. This HF datasets code stores both RNG state and the offset inside the generated random batch, making the iterable dataset restartable.

## 代码 / The code

`huggingface/datasets` — [`src/datasets/iterable_dataset.py`](https://github.com/huggingface/datasets/blob/41adfd0f9ee9ba3a6b4f719d5b551c5b19ae45e2/src/datasets/iterable_dataset.py#L1205-L1278)

```python
class RandomlyCyclingMultiSourcesExamplesIterable(CyclingMultiSourcesExamplesIterable):
    def __init__(
        self,
        ex_iterables: list[_BaseExamplesIterable],
        generator: np.random.Generator,
        probabilities: Optional[list[float]] = None,
        stopping_strategy: Literal[
            "first_exhausted", "all_exhausted", "all_exhausted_without_replacement"
        ] = "first_exhausted",
    ):
        super().__init__(ex_iterables, stopping_strategy)
        self.generator = deepcopy(generator)
        self.probabilities = probabilities

    def _get_indices_iterator(self):
        rng = deepcopy(self.generator)
        num_sources = len(self.ex_iterables)
        random_batch_size = 1000
        index_offset = self._state_dict["bit_generator_index_offset"] if self._state_dict else 0
        if self._state_dict:
            rng.bit_generator.state = self._state_dict["bit_generator_state"]
        if self.probabilities is None:
            while True:
                for i in islice(rng.integers(0, num_sources, size=random_batch_size), index_offset, None):
                    index_offset = (index_offset + 1) % random_batch_size
                    if self._state_dict:
                        self._state_dict["bit_generator_index_offset"] = index_offset
                        if index_offset == 0:
                            self._state_dict["bit_generator_state"] = rng.bit_generator.state
                    yield int(i)
        else:
            while True:
                for i in islice(
                    rng.choice(num_sources, size=random_batch_size, p=self.probabilities), index_offset, None
                ):
                    index_offset = (index_offset + 1) % random_batch_size
                    if self._state_dict:
                        self._state_dict["bit_generator_index_offset"] = index_offset
                        if index_offset == 0:
                            self._state_dict["bit_generator_state"] = rng.bit_generator.state
                    yield int(i)

    def _init_state_dict(self) -> dict:
        for ex_iterable in self.ex_iterables:
            ex_iterable._init_state_dict()
        self._state_dict = {
            "bit_generator_state": self.generator.bit_generator.state,
            "bit_generator_index_offset": 0,
            "previous_states": [None] * len(self.ex_iterables),
            "is_exhausted": [False] * len(self.ex_iterables),
            "type": self.__class__.__name__,
        }
        return self._state_dict
```

## 逐行讲解 / What's happening

1. **第 1215 行 / Line 1215 (`deepcopy(generator)`)**:
   - 中文: 构造时复制 RNG，避免外部继续使用同一个 generator 改变 dataset 的采样轨迹。
   - English: The constructor copies the RNG so outside code cannot advance the dataset's sampling path.
2. **第 1221-1225 行 / Lines 1221-1225 (恢复 RNG)**:
   - 中文: 如果已有 state dict，就把 bit generator 状态恢复到保存点，并从 batch 内 offset 继续。
   - English: If a state dict exists, it restores the bit generator state and resumes at the saved offset inside the random batch.
3. **第 1227-1234 行 / Lines 1227-1234 (`rng.integers`)**:
   - 中文: 没有概率时，批量生成 1000 个 source index，减少每条样本都调用 RNG 的开销。
   - English: Without probabilities, it generates 1000 source indices at a time, reducing per-example RNG overhead.
4. **第 1249-1258 行 / Lines 1249-1258 (`_init_state_dict`)**:
   - 中文: state 里不仅有 RNG，还有每个子 iterable 的状态和 exhausted 标记。
   - English: The state stores not only RNG state, but also each child iterable's state and exhaustion flags.

## 类比 / The analogy

这像一个电台 DJ 在三张歌单之间随机切歌。真正要保存的不只是“下一首从哪张歌单来”，还包括随机抽签机器当前状态，以及每张歌单已经播到第几首。

This is like a radio DJ randomly switching among three playlists. To resume the show, saving only the next playlist is not enough; you also need the lottery machine state and each playlist's current position.

## 自己跑一遍 / Try it yourself

```python
import random

state = random.Random(7).getstate()
rng = random.Random()
rng.setstate(state)
indices = [rng.randrange(3) for _ in range(6)]
saved = rng.getstate()
more = [rng.randrange(3) for _ in range(3)]

rng2 = random.Random()
rng2.setstate(saved)
print(indices)
print(more)
print([rng2.randrange(3) for _ in range(3)])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[1, 0, 1, 2, 0, 0]
[2, 0, 1]
[2, 0, 1]
```

恢复后的第三行和原来的第二行一致，这就是 iterable dataset checkpoint 想要的性质。

The third line after restore matches the original second line. That is the property iterable dataset checkpointing needs.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **PyTorch DataLoader workers** / **PyTorch DataLoader workers**: worker seed 必须可控，否则多进程读取会产生不可复现的数据顺序。 / Worker seeds must be controlled, or multi-process loading becomes non-reproducible.
- **RL replay sampling** / **RL replay sampling**: 经验回放的 sampler 也需要保存 RNG 才能精确恢复训练。 / Replay-buffer samplers also need RNG state for exact training resume.

## 注意事项 / Caveats / when it breaks

- **子数据源也要可恢复** / **Child sources must also resume**: 只保存顶层 RNG 不够，每个 `ex_iterable` 也要有自己的 state。 / Saving only the top-level RNG is not enough; every child iterable needs its own state.
- **概率改变会改变轨迹** / **Changing probabilities changes the path**: checkpoint 后修改 `probabilities` 会让后续样本顺序失去可比性。 / Modifying `probabilities` after checkpointing invalidates the subsequent sample path.

## 延伸阅读 / Further reading

- [HF datasets iterable dataset source cycling](https://github.com/huggingface/datasets/blob/41adfd0f9ee9ba3a6b4f719d5b551c5b19ae45e2/src/datasets/iterable_dataset.py#L1205-L1278)
