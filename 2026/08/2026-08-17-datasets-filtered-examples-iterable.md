---
date: 2026-08-17
topic: huggingface
source: huggingface
repo: huggingface/datasets
file: src/datasets/iterable_dataset.py
permalink: https://github.com/huggingface/datasets/blob/836b82e0544cabf6474b25ade131b4d21e570373/src/datasets/iterable_dataset.py#L1812-L1856
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, streaming]
---

# Datasets FilteredExamplesIterable：filter 先变成 mask 再丢列 / Datasets FilteredExamplesIterable: Turn Filter into a Mask, Then Drop the Column

> **一句话 / In one line**: `FilteredExamplesIterable` 复用 map 管线，先给样本加一个临时布尔列，再按这个列过滤并删除它。 / `FilteredExamplesIterable` reuses the map pipeline by adding a temporary boolean column, filtering on it, then dropping that column.

## 为什么重要 / Why this matters

流式 dataset 不能先把全量数据读进内存再过滤。这里的设计把“用户函数返回 bool”转换成普通 map 输出的一列，这样 Python 样本路径和 Arrow batch 路径都能共享同一个 lazy pipeline。

A streaming dataset cannot load everything into memory before filtering. This design turns a user predicate into a normal mapped boolean column, letting both Python example iteration and Arrow batch iteration share the same lazy pipeline.

## 代码 / The code

`huggingface/datasets` — [`src/datasets/iterable_dataset.py`](https://github.com/huggingface/datasets/blob/836b82e0544cabf6474b25ade131b4d21e570373/src/datasets/iterable_dataset.py#L1812-L1856)

```python
class FilteredExamplesIterable(MappedExamplesIterable):
    mask_column_name = "===MASK==="

    def __init__(
        self,
        ex_iterable: _BaseExamplesIterable,
        function: Callable,
        with_indices: bool = False,
        input_columns: Optional[list[str]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        fn_kwargs: Optional[dict] = None,
        formatting: Optional["FormattingConfig"] = None,
    ):
        self.mask_function = function
        if ex_iterable.is_typed:
            features = Features({**ex_iterable.features, self.mask_column_name: Value("bool")})
        else:
            features = None
        super().__init__(
            ex_iterable=ex_iterable,
            function=partial(
                async_add_mask if inspect.iscoroutinefunction(function) else add_mask,
                function,
                mask_column_name=self.mask_column_name,
            ),
            with_indices=with_indices,
            input_columns=input_columns,
            batched=batched,
            batch_size=batch_size,
            fn_kwargs=fn_kwargs,
            formatting=formatting,
            features=features,
        )

    def _iter(self):
        for key, example in super()._iter():
            example = dict(example)
            if example.pop(self.mask_column_name):
                yield key, example

    def _iter_arrow(self, max_chunksize: Optional[int] = None):
        for key, pa_table in super()._iter_arrow(max_chunksize=max_chunksize):
            mask = pa_table[self.mask_column_name]
            yield key, pa_table.drop(self.mask_column_name).filter(mask)
```

## 逐行讲解 / What's happening

1. **第 1812-1813 行 / Lines 1812-1813**: 中文: filter 不是独立重写一条管线，而是继承 `MappedExamplesIterable`。临时列名故意很怪，降低和用户列撞名的概率。 / English: Filtering is built on top of `MappedExamplesIterable`. The odd temporary column name reduces collisions with user columns.
2. **第 1827-1830 行 / Lines 1827-1830**: 中文: 如果底层 iterable 有 schema，就把 mask 列也加进 `Features`，保证 Arrow 类型知道它是 bool。 / English: If the wrapped iterable has a schema, the mask column is added to `Features` so Arrow knows it is boolean.
3. **第 1831-1845 行 / Lines 1831-1845**: 中文: 用户传入的 predicate 被包成 `add_mask` 或 `async_add_mask`，所以 async filter 也走同一个结构。 / English: The user predicate is wrapped as `add_mask` or `async_add_mask`, so async filters follow the same structure.
4. **第 1847-1856 行 / Lines 1847-1856**: 中文: Python 路径用 `pop` 删除临时列，Arrow 路径用 `drop(...).filter(mask)` 批量过滤。 / English: The Python path removes the temporary key with `pop`, while the Arrow path drops the column and filters in batch.

## 类比 / The analogy

像在仓库包裹上贴一张临时“通过安检”贴纸。出库时只放行有贴纸的包裹，最后把贴纸撕掉，客户不会看到内部流程标记。

Think of adding a temporary “passed inspection” sticker to warehouse packages. Shipping only lets stickered packages through, then removes the sticker so customers never see the internal marker.

## 自己跑一遍 / Try it yourself

```python
MASK = "===MASK==="

def add_mask(row):
    row = dict(row)
    row[MASK] = row["score"] >= 0.5
    return row

rows = [{"id": 1, "score": 0.8}, {"id": 2, "score": 0.1}, {"id": 3, "score": 0.6}]
out = []
for row in map(add_mask, rows):
    if row.pop(MASK):
        out.append(row)
print(out)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[{'id': 1, 'score': 0.8}, {'id': 3, 'score': 0.6}]
```

filter 的中间结果是一列数据，而不是一个隐藏控制流；这让流式、批式和 async 路径更容易统一。

The filter result becomes a data column rather than hidden control flow, which makes streaming, batched, and async paths easier to unify.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **SQL WHERE mask** / **SQL WHERE masks**: 查询优化器也常把谓词变成可组合的布尔表达式。 / Query engines also turn predicates into composable boolean expressions.
- **训练样本权重** / **Training sample weights**: 数据管线常先生成权重或 mask，再在后续阶段消费它。 / Data pipelines often generate weights or masks first, then consume them downstream.

## 注意事项 / Caveats / when it breaks

- **临时列仍可能撞名** / **The temporary column can still collide**: 用户如果真的有 `===MASK===` 列，就会发生语义混淆。 / A user column named `===MASK===` would still cause semantic confusion.
- **predicate 要可重放** / **The predicate should be replayable**: 流式恢复和 sharding 会重新执行 predicate，带副作用的函数会很难调试。 / Streaming resume and sharding may replay the predicate, so side effects are hard to debug.

## 延伸阅读 / Further reading

- [Datasets source](https://github.com/huggingface/datasets/blob/836b82e0544cabf6474b25ade131b4d21e570373/src/datasets/iterable_dataset.py#L1812-L1856)
