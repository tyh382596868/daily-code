---
date: 2026-10-10
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/_dynamo/polyfills/itertools.py
permalink: https://github.com/pytorch/pytorch/blob/ce6f6706bf4969173539c16f5e282fd0bf9a8d41/torch/_dynamo/polyfills/itertools.py#L296-L311
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, dynamo-polyfill]
---
# PyTorch Dynamo tee：多个迭代器共用一条惰性链 / PyTorch Dynamo tee: Several Iterators Share One Lazy Chain

> **一句话 / In one line**: `tee` 不复制整个 iterable，而是让多个迭代器共享一条按需扩展的 linked list。 / `tee` does not copy the whole iterable; it lets several iterators share a lazily extended linked list.

## 为什么重要 / Why this matters

Dynamo polyfill 把 Python 迭代语义写成可替换、可分析的普通 Python。这里的关键是懒加载：只有某个分支需要新值，底层 iterator 才前进一步。

Dynamo polyfills express Python iteration semantics in replaceable, analyzable Python. The key here is laziness: the source iterator advances only when one branch needs a new value.

## 代码 / The code

`pytorch/pytorch` — [`torch/_dynamo/polyfills/itertools.py`](https://github.com/pytorch/pytorch/blob/ce6f6706bf4969173539c16f5e282fd0bf9a8d41/torch/_dynamo/polyfills/itertools.py#L296-L311)

```python
def tee(iterable: Iterable[_T], n: int = 2, /) -> tuple[Iterator[_T], ...]:
    iterator = iter(iterable)
    shared_link = [None, None]

    def _tee(link) -> Iterator[_T]:  # type: ignore[no-untyped-def]
        try:
            while True:
                if link[1] is None:
                    link[0] = next(iterator)
                    link[1] = [None, None]
                value, link = link
                yield value
        except StopIteration:
            return

    return tuple(_tee(shared_link) for _ in range(n))
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

像几个人共读一卷收据纸。纸卷只吐一次，每个人手里有自己的指针，慢的人沿着已经吐出的纸继续读。

It is like several people reading one receipt roll. The roll prints each item once, while every reader keeps a pointer into the printed strip.

## 自己跑一遍 / Try it yourself

```python
def mini_tee(iterable, n=2):
    it = iter(iterable); shared = [None, None]
    def child(link):
        while True:
            if link[1] is None:
                link[0] = next(it); link[1] = [None, None]
            value, link = link
            yield value
    return tuple(child(shared) for _ in range(n))
a, b = mini_tee([10, 20, 30])
print(next(a), next(a))
print(next(b), next(b), next(b))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
10 20
10 20 30
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

- [pytorch/pytorch source](https://github.com/pytorch/pytorch/blob/ce6f6706bf4969173539c16f5e282fd0bf9a8d41/torch/_dynamo/polyfills/itertools.py#L296-L311)
