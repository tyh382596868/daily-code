---
date: 2026-08-04
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/utils/data/_utils/collate.py
permalink: https://github.com/pytorch/pytorch/blob/51e7462a808d5c9e968e1dc8ae9621a25ce3785c/torch/utils/data/_utils/collate.py#L118-L243
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, dataloader, collate]
---

# PyTorch collate：递归拼 batch 前先查类型注册表 / PyTorch collate: Check the Type Registry Before Recursing a Batch

> **一句话 / In one line**: `collate` 用类型注册表处理叶子节点，再递归保留 dict、namedtuple 和 sequence 的结构。 / `collate` handles leaf nodes through a type registry, then recursively preserves mappings, namedtuples, and sequences.

## 为什么重要 / Why this matters

`DataLoader` 不是只会把 tensor `stack` 起来。真实样本经常是 `{"image": tensor, "meta": {"id": ...}}` 这种嵌套结构。`collate` 的价值在于：叶子节点按类型合并，容器结构尽量原样返回，让模型收到的是“同形状的 batch 化样本树”。

`DataLoader` does more than `stack` tensors. Real samples often look like nested trees such as `{"image": tensor, "meta": {"id": ...}}`. `collate` keeps that structure: merge leaves by type, preserve containers when possible, and return a batched tree with the same shape.

## 代码 / The code

`pytorch/pytorch` — [`torch/utils/data/_utils/collate.py`](https://github.com/pytorch/pytorch/blob/51e7462a808d5c9e968e1dc8ae9621a25ce3785c/torch/utils/data/_utils/collate.py#L118-L243)

```python
def collate(
    batch,
    *,
    collate_fn_map: dict[type | tuple[type, ...], Callable] | None = None,
):
    r"""
    General collate function that handles collection type of element within each batch.
    """
    elem = batch[0]
    elem_type = type(elem)

    if collate_fn_map is not None:
        if elem_type in collate_fn_map:
            return collate_fn_map[elem_type](batch, collate_fn_map=collate_fn_map)

        for collate_type in collate_fn_map:
            if isinstance(elem, collate_type):
                return collate_fn_map[collate_type](
                    batch, collate_fn_map=collate_fn_map
                )

    if isinstance(elem, collections.abc.Mapping):
        try:
            if isinstance(elem, collections.abc.MutableMapping):
                clone = copy.copy(elem)
                clone.update(
                    {
                        key: collate(
                            [d[key] for d in batch], collate_fn_map=collate_fn_map
                        )
                        for key in elem
                    }
                )
                return clone
            else:
                return elem_type(
                    {
                        key: collate(
                            [d[key] for d in batch], collate_fn_map=collate_fn_map
                        )
                        for key in elem
                    }
                )
        except TypeError:
            return {
                key: collate([d[key] for d in batch], collate_fn_map=collate_fn_map)
                for key in elem
            }
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(
            *(
                collate(samples, collate_fn_map=collate_fn_map)
                for samples in zip(*batch, strict=False)
            )
        )
    elif isinstance(elem, collections.abc.Sequence):
        it = iter(batch)
        elem_size = len(next(it))

        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = list(zip(*batch, strict=False))

        if isinstance(elem, tuple):
            return [
                collate(samples, collate_fn_map=collate_fn_map)
                for samples in transposed
            ]
        else:
            try:
                if isinstance(elem, collections.abc.MutableSequence):
                    clone = copy.copy(elem)  # type: ignore[arg-type]
                    for i, samples in enumerate(transposed):
                        clone[i] = collate(samples, collate_fn_map=collate_fn_map)
                    return clone
                else:
                    return elem_type(
                        [
                            collate(samples, collate_fn_map=collate_fn_map)
                            for samples in transposed
                        ]
                    )
            except TypeError:
                return [
                    collate(samples, collate_fn_map=collate_fn_map)
                    for samples in transposed
                ]

    raise TypeError(default_collate_err_msg_format.format(elem_type))
```

## 逐行讲解 / What's happening

1. **第 150-161 行 / Lines 150-161 (registry first)**:
   - 中文: 先精确匹配 `elem_type`，再按插入顺序尝试 `isinstance`，所以自定义类型可以覆盖默认行为。
   - English: It first checks the exact element type, then tries `isinstance` in insertion order, so custom types can override default behavior.
2. **第 163-194 行 / Lines 163-194 (mapping recursion)**:
   - 中文: dict 类样本按 key 拆开，每个 key 的值递归 collate；能复制原类型就复制，不能就退回普通 dict。
   - English: Mapping samples are split by key and each key's values are collated recursively; PyTorch preserves the original mapping type when possible and falls back to `dict`.
3. **第 202-241 行 / Lines 202-241 (sequence transpose)**:
   - 中文: list/tuple 会先检查长度一致，再 `zip(*batch)`，把“样本维”转成“字段维”。
   - English: Lists and tuples must have equal lengths; `zip(*batch)` transposes from sample-major to field-major form.

## 类比 / The analogy

这像把几份外卖订单合并装袋：主食放主食袋，饮料放饮料袋，小票按原订单夹回去。你不是把所有东西倒进一个桶，而是保留订单结构。

It is like merging several takeout orders: entrees go into one bag, drinks into another, receipts stay attached to the original structure. You do not dump everything into one bucket.

## 自己跑一遍 / Try it yourself

```python
def collate(batch):
    elem = batch[0]
    if isinstance(elem, dict):
        return {k: collate([x[k] for x in batch]) for k in elem}
    if isinstance(elem, list):
        if len({len(x) for x in batch}) != 1:
            raise RuntimeError("lists must have equal size")
        return [collate(items) for items in zip(*batch)]
    return list(batch)

samples = [{"x": [1, 2], "id": "a"}, {"x": [3, 4], "id": "b"}]
print(collate(samples))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
{'x': [[1, 3], [2, 4]], 'id': ['a', 'b']}
```

中文: 结构没丢，只是每个叶子都多了 batch 维。
English: The structure remains; each leaf simply gains a batch dimension.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **JAX pytrees** / **JAX pytrees**: 也是先保留树结构，再处理叶子。 / They also preserve tree structure while transforming leaves.
- **Hugging Face data collators** / **Hugging Face data collators**: tokenizer batch 通常也是对字段递归合并。 / Tokenizer batches usually merge fields recursively as well.

## 注意事项 / Caveats / when it breaks

- **sequence 长度必须一致** / **Sequence lengths must match**: ragged list 不能自动 stack，需要自定义 padding collator。 / Ragged lists need a custom padding collator.
- **tuple 返回 list 是兼容行为** / **Tuple becomes list for compatibility**: 普通 tuple 不会保留 tuple 类型，这是历史兼容。 / Plain tuples become lists for backward compatibility.

## 延伸阅读 / Further reading

- PyTorch `collate.py`: https://github.com/pytorch/pytorch/blob/51e7462a808d5c9e968e1dc8ae9621a25ce3785c/torch/utils/data/_utils/collate.py#L118-L243
