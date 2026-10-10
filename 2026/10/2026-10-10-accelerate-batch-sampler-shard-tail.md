---
date: 2026-10-10
topic: huggingface
source: huggingface
repo: huggingface/accelerate
file: src/accelerate/data_loader.py
permalink: https://github.com/huggingface/accelerate/blob/4ebc726849a2d7f03c16198847c22954e224dd87/src/accelerate/data_loader.py#L191-L211
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, distributed-dataloader]
---
# Accelerate BatchSamplerShard：尾 batch 也要公平补齐 / Accelerate BatchSamplerShard: Make the Tail Batch Fair Too

> **一句话 / In one line**: 分布式 batch split 不只是切片；最后一个短 batch 还要决定丢弃、局部给出，还是从开头补齐。 / Distributed batch splitting is more than slicing; the short tail batch must be dropped, yielded partially, or padded from the beginning.

## 为什么重要 / Why this matters

多进程训练里，每个 rank 最好拿到形状稳定的 batch。Accelerate 对完整 batch 直接按 rank 切片；最后一个短 batch 则由 `drop_last` 和 `even_batches` 决定是否补齐。

In multi-process training, each rank usually wants stable batch shapes. Accelerate slices full batches by rank; the final short batch is handled according to `drop_last` and `even_batches`.

## 代码 / The code

`huggingface/accelerate` — [`src/accelerate/data_loader.py`](https://github.com/huggingface/accelerate/blob/4ebc726849a2d7f03c16198847c22954e224dd87/src/accelerate/data_loader.py#L191-L211)

```python
    def _iter_with_split(self):
        initial_data = []
        batch_length = self.batch_sampler.batch_size // self.num_processes
        for idx, batch in enumerate(self.batch_sampler):
            if idx == 0:
                initial_data = batch
            if len(batch) == self.batch_size:
                # If the batch is full, we yield the part of it this process is responsible of.
                yield batch[batch_length * self.process_index : batch_length * (self.process_index + 1)]

        # If drop_last is True of the last batch was full, iteration is over, otherwise...
        if not self.drop_last and len(initial_data) > 0 and len(batch) < self.batch_size:
            if not self.even_batches:
                if len(batch) > batch_length * self.process_index:
                    yield batch[batch_length * self.process_index : batch_length * (self.process_index + 1)]
            else:
                # For degenerate cases where the dataset has less than num_process * batch_size samples
                while len(initial_data) < self.batch_size:
                    initial_data += initial_data
                batch = batch + initial_data
                yield batch[batch_length * self.process_index : batch_length * (self.process_index + 1)]
```

## 逐行讲解 / What's happening

1. **入口 / Entry**:
   - 中文: 先看函数签名和输入，它定义了这个模块承担的边界职责。
   - English: Start from the signature and inputs; they define the module boundary.
2. **核心状态 / Core state**:
   - 中文: 中间变量保存的是工程约束，例如 cache、rank、bin 或时间线。
   - English: The intermediate variables encode engineering constraints such as cache, rank, bins, or timeline state.
3. **返回值 / Return value**:
   - 中文: 输出不是孤立结果，而是给下游模块继续消费的契约。
   - English: The output is not an isolated result; it is the contract consumed downstream.

## 类比 / The analogy

像把一盘寿司分给多张桌子。整盘好分；最后剩三块时，要么有人少吃，要么从第一盘再补几块保证每桌一样多。

It is like splitting sushi across tables. Full trays divide cleanly; the final pieces either leave some tables short or get padded from the first tray.

## 自己跑一遍 / Try it yourself

```python
def shard(batches, rank, world, batch_size):
    part = batch_size // world; initial = []
    for i, batch in enumerate(batches):
        if i == 0: initial = list(batch)
        if len(batch) == batch_size:
            yield batch[part*rank:part*(rank+1)]
    while len(initial) < batch_size:
        initial += initial
    batch = batch + initial
    yield batch[part*rank:part*(rank+1)]
print(list(shard([[0,1,2,3],[4,5]], 1, 2, 4)))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
[[2, 3], [0, 1]]
```

这个小例子保留了源码里最重要的控制结构，方便你先写最小版再回到工程实现。

The small example keeps the most important control structure from the source, so you can write the minimal version before returning to the production implementation.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **训练/推理边界** / **Training/inference boundaries**: 这类代码常把数学步骤转换成工程契约。 / This kind of code turns a mathematical step into an engineering contract.
- **nano 系统实现** / **Nano-system implementation**: 从这些片段抽象出的接口可以直接变成你自己的最小模块。 / The interface abstracted from these snippets can become your own minimal module.

## 注意事项 / Caveats / when it breaks

- **边界条件** / **Boundary cases**: 空输入、短 batch、短视频或越界动作通常最容易出 bug。 / Empty inputs, short batches, short videos, or out-of-range actions are the easiest places to break.
- **契约要写测试** / **Test the contract**: 这些函数依赖调用方遵守 shape、顺序和状态生命周期。 / These functions rely on callers respecting shapes, ordering, and state lifetime.

## 延伸阅读 / Further reading

- [huggingface/accelerate source](https://github.com/huggingface/accelerate/blob/4ebc726849a2d7f03c16198847c22954e224dd87/src/accelerate/data_loader.py#L191-L211)
