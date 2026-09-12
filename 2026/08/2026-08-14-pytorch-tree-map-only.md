---
date: 2026-08-14
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/utils/_pytree.py
permalink: https://github.com/pytorch/pytorch/blob/2cba5882bd7a8bb9656da89f5c43a40a69fb26ce/torch/utils/_pytree.py#L1543-L1760
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, pytree]
---

# PyTorch tree_map_only：只改你关心的叶子 / PyTorch tree_map_only: Transform Only the Leaves You Care About

> **一句话 / In one line**: `tree_map_only` 用 `map_only` 包一层 predicate，让 pytree 映射只作用在指定类型或条件的叶子上。 / `tree_map_only` wraps a predicate with `map_only`, so a pytree map only transforms leaves matching a type or condition.

## 为什么重要 / Why this matters

PyTorch API 经常要处理嵌套 batch：里面可能有 tensor、数字、字符串、metadata。`tree_map_only` 让你“只把 tensor 搬到 GPU”或“只改浮点叶子”，不用手写递归，也不用误伤非 tensor 元数据。

PyTorch APIs often receive nested batches containing tensors, numbers, strings, and metadata. `tree_map_only` lets you move only tensors to a device or modify only floating leaves without writing recursive code or touching unrelated metadata.

## 代码 / The code

`pytorch/pytorch` — [`torch/utils/_pytree.py`](https://github.com/pytorch/pytorch/blob/2cba5882bd7a8bb9656da89f5c43a40a69fb26ce/torch/utils/_pytree.py#L1543-L1760)

```python
def tree_map(
    func: Callable[..., Any],
    tree: PyTree,
    *rests: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree:
    leaves, treespec = tree_flatten(tree, is_leaf=is_leaf)
    flat_args = [leaves] + [treespec.flatten_up_to(r) for r in rests]
    return treespec.unflatten(map(func, *flat_args))


def tree_map_(
    func: Callable[..., Any],
    tree: PyTree,
    *rests: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree:
    leaves, treespec = tree_flatten(tree, is_leaf=is_leaf)
    flat_args = [leaves] + [treespec.flatten_up_to(r) for r in rests]
    deque(map(func, *flat_args), maxlen=0)  # consume and exhaust the iterable
    return tree


def map_only(
    type_or_types_or_pred: TypeAny | Callable[[Any], bool], /
) -> MapOnlyFn[FnAny[Any]]:
    if isinstance(type_or_types_or_pred, (type, tuple, types.UnionType)):

        def pred(x: Any) -> bool:
            return isinstance(x, type_or_types_or_pred)  # type: ignore[arg-type]

    elif callable(type_or_types_or_pred):
        pred = type_or_types_or_pred  # type: ignore[assignment]
    else:
        raise TypeError("Argument must be a type, a tuple of types, or a callable.")

    def wrapper(func: Callable[[T], Any]) -> Callable[[Any], Any]:
        @functools.wraps(func)
        def wrapped(x: T) -> Any:
            if pred(x):
                return func(x)
            return x

        return wrapped

    return wrapper


def tree_map_only(
    type_or_types_or_pred: TypeAny | Callable[[Any], bool],
    /,
    func: FnAny[Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
) -> PyTree:
    return tree_map(map_only(type_or_types_or_pred)(func), tree, is_leaf=is_leaf)
```

## 逐行讲解 / What's happening

1. **第 1582-1584 行 / Lines 1582-1584**: 中文: `tree_map` 先 flatten 第一棵树，再用同一个 `treespec` 对其他树做前缀匹配，最后 unflatten 回原结构。 / English: `tree_map` flattens the first tree, aligns the rest with that `treespec`, then reconstructs the original structure.
2. **第 1615-1618 行 / Lines 1615-1618**: 中文: in-place 版本故意消耗 `map` 迭代器，只保留副作用并返回原树。 / English: The in-place form exhausts the `map` iterator for side effects and returns the original tree.
3. **第 1681-1689 行 / Lines 1681-1689**: 中文: `map_only` 接受类型、类型元组、union 或 predicate，统一转成 `pred(x)`。 / English: `map_only` accepts types, tuples, unions, or predicates and normalizes them into `pred(x)`.
4. **第 1691-1760 行 / Lines 1691-1760**: 中文: 不匹配的叶子原样返回，`tree_map_only` 只是把这个包装函数交给普通 `tree_map`。 / English: Non-matching leaves pass through unchanged; `tree_map_only` just feeds the wrapped function into regular `tree_map`.

## 类比 / The analogy

像整理一个混合抽屉：你只想给所有电池贴标签，钥匙、发票、螺丝都留在原地。`tree_map_only` 就是那个“只碰电池”的手套。

It is like sorting a drawer where you only label the batteries while leaving keys, receipts, and screws alone. `tree_map_only` is the glove that only touches batteries.

## 自己跑一遍 / Try it yourself

```python
def map_only(pred, func):
    return lambda x: func(x) if pred(x) else x

def tree_map(func, tree):
    if isinstance(tree, dict):
        return {k: tree_map(func, v) for k, v in tree.items()}
    if isinstance(tree, tuple):
        return tuple(tree_map(func, v) for v in tree)
    return func(tree)

batch = {"x": (1, "keep"), "meta": "robot"}
print(tree_map(map_only(lambda x: isinstance(x, int), lambda x: x + 10), batch))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
{'x': (11, 'keep'), 'meta': 'robot'}
```

整数叶子被改了，字符串 metadata 没动；这就是 typed pytree transform 的价值。

Only integer leaves changed while string metadata stayed intact; that is the value of typed pytree transforms.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Accelerate device moves** / **Accelerate device moves**: 分布式库经常递归处理 batch，但只对 tensor 做 device 操作。 / Distributed libraries often recurse through batches while applying device operations only to tensors.
- **DataLoader collate** / **DataLoader collate**: collate 也是先识别容器类型，再递归处理叶子。 / Collation also identifies container types first, then recursively handles leaves.

## 注意事项 / Caveats / when it breaks

- **结构来自第一棵树** / **Structure comes from the first tree**: 多输入 `tree_map` 要求后续树能被第一棵树的 `treespec` 对齐。 / Multi-input `tree_map` requires later trees to align with the first tree's `treespec`.
- **predicate 别太宽** / **Do not make the predicate too broad**: `callable` 或父类 predicate 写宽了，可能改到本来应保留的对象。 / A broad predicate can transform objects that should have stayed as metadata.

## 延伸阅读 / Further reading

- [pytorch/pytorch source](https://github.com/pytorch/pytorch/blob/2cba5882bd7a8bb9656da89f5c43a40a69fb26ce/torch/utils/_pytree.py#L1543-L1760)
