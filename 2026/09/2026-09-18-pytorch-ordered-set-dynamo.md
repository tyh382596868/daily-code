---
date: 2026-09-18
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/_dynamo/variables/sets.py
permalink: https://github.com/pytorch/pytorch/blob/39b52fbc3366d6e969c835c232f8bc854124ae43/torch/_dynamo/variables/sets.py#L874-L991
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, pytorch, torchdynamo, ordered-set, guards]
---

# PyTorch Dynamo OrderedSet：集合也有可观察的顺序 / PyTorch Dynamo OrderedSet: A Set Can Have Observable Order

> **一句话 / In one line**: TorchDynamo 把 `OrderedSet` 从普通 `set` 中拆出来建模，因为迭代、`pop`、反向遍历和重建都暴露了插入顺序。 / TorchDynamo models `OrderedSet` separately from `set` because iteration, `pop`, reverse iteration, and reconstruction expose insertion order.

## 为什么重要 / Why this matters

编译器最容易犯的错误，是把“数学上相似”的对象当成“语义上一样”的对象。普通 `set` 只承诺成员关系；`torch.utils._ordered_set.OrderedSet` 还承诺顺序。因此，Dynamo 如果把它直接当成 `SetVariable`，图里的常量、guard、`pop()` 结果和 Python eager 行为都会错位。

Compilers often fail by treating mathematically similar objects as semantically identical. A regular `set` promises membership, while `OrderedSet` also promises order. If Dynamo reused `SetVariable` blindly, compiled constants, guards, `pop()`, and eager Python behavior would diverge.

这段代码的关键设计是“共享大部分集合操作，但覆盖所有顺序敏感的地方”。`BaseSetVariable` 提供公共集合逻辑，`OrderedSetVariable` 作为 sibling 保留自己的 `debug_repr`、重建、反向迭代、`pop` 和运算顺序。

The design shares most set operations while overriding every order-sensitive edge. `BaseSetVariable` supplies common logic, and `OrderedSetVariable` remains a sibling that owns ordered repr, reconstruction, reverse iteration, `pop`, and operator behavior.

## 代码 / The code

`pytorch/pytorch` — [`torch/_dynamo/variables/sets.py`](https://github.com/pytorch/pytorch/blob/39b52fbc3366d6e969c835c232f8bc854124ae43/torch/_dynamo/variables/sets.py#L874-L991)

```python
class OrderedSetVariable(BaseSetVariable):
    """``torch.utils._ordered_set.OrderedSet``, an insertion-ordered set.

    OrderedSet is a pure-Python ``collections.abc.MutableSet``, not a subclass
    of ``set`` (see #192874), so this is a sibling of SetVariable under
    BaseSetVariable rather than a subclass of it. Its named-method surface is
    the same as ``set``'s, mutators included, so the ``set`` implementations
    are registered in ``tp_methods`` below. What OrderedSet does differently
    is overridden here: insertion order is observable (iteration, ``repr``,
    ``pop`` and reconstruction all follow it), and the in-place operators are
    ``MutableSet``'s, which accept any iterable and mutate in place.
    """

    _cpython_type = OrderedSet

    def method_flags_type(self) -> type:
        return set

    def _get_internal_dict(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        from .dicts import ConstDictVariable

        return ConstDictVariable(self.items, mutation_type=ValueMutationNew())

    tp_members = {"_dict": Member(_get_internal_dict, readonly_setter)}

    def install_set_contains_guard(
        self, tx: "InstructionTranslatorBase", args: list[VariableTracker]
    ) -> None:
        pass

    def tp_iter_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        from .iter import SetIterator

        return SetIterator(self.items)

    def debug_repr(self) -> str:
        if not self.items:
            return "OrderedSet([])"
        else:
            items: list[str] = []
            for k in self.items:
                key_str = _item_debug_repr(k.vt)
                items.append(key_str)
            return "OrderedSet([" + ", ".join(items) + "])"

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> "VariableTracker":
        items = ", ".join(tracked_repr(tx, item.vt) for item in self.items)
        return VariableTracker.build(tx, f"{self.python_type_name()}([{items}])")

    def as_python_constant(self) -> OrderedSet[Any]:
        return OrderedSet([k.vt.as_python_constant() for k in self.items])

    def python_type(self) -> type[OrderedSet[Any]]:
        return OrderedSet

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(
            lambda: codegen.load_import_from("torch.utils._ordered_set", "OrderedSet")
        )
        codegen.foreach([x.vt for x in self.items])
        codegen.append_output(create_instruction("BUILD_LIST", arg=len(self.items)))
        codegen.extend_output(create_call_function(1, False))

    def is_hashable(self) -> bool:
        return False

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        raise_type_error(tx, f"unhashable type: '{self.python_type_name()}'")

    def sq_contains_impl(
        self, tx: "InstructionTranslatorBase", item: VariableTracker
    ) -> VariableTracker:
        if not is_hashable(item):
            raise_type_error(tx, f"unhashable type: '{item.python_type_name()}'")
        return VariableTracker.build(tx, item in self)

    def reversed_(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        keys: list[VariableTracker] = [k.vt for k in reversed(list(self.items))]
        return variables.ListIteratorVariable(keys, mutation_type=ValueMutationNew())

    def pop(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if not self.items:
            raise_observed_exception(KeyError, tx, args=["pop from an empty set"])
        self.should_reconstruct_all = True
        tx.output.side_effects.mutation(self)
        key, _ = self.items.popitem()
        return key.vt
```

## 逐行讲解 / What's happening

1. **第 874-885 行 / Lines 874-885 (the class contract)**:
   - 中文: 注释先写出边界：它是 `MutableSet`，但不是 Python 内建 `set` 的子类，所以不能靠 `isinstance(..., SetVariable)` 复用全部语义。
   - English: The class comment establishes the boundary: it is a `MutableSet`, but not a subclass of builtin `set`, so all semantics cannot come from `SetVariable`.
2. **第 894-902 行 / Lines 894-902 (the backing dict)**:
   - 中文: `OrderedSet` 的真实存储是 `_dict`。把它暴露给 Dynamo，可以让 `elem in self._dict` 这样的 Python 实现继续被 trace，而不是无条件 graph break。
   - English: `OrderedSet` is backed by `_dict`. Exposing that member lets Dynamo trace Python implementations such as `elem in self._dict` instead of graph-breaking.
3. **第 904-914 行 / Lines 904-914 (guard override)**:
   - 中文: 普通 set 的 membership guard 不适用，因为 `set.__contains__` 和 `_dict` 的 key-order guard 不是同一件事，所以这里故意 no-op。
   - English: Regular set membership guards do not apply: `set.__contains__` is not the same source as a dict-backed key-order guard, so this method intentionally does nothing.
4. **第 916-948 行 / Lines 916-948 (ordered iteration and reconstruction)**:
   - 中文: 所有遍历都从 `self.items` 这个有序 dict 视图产生；重建时也按原顺序 `BUILD_LIST` 后重新构造 `OrderedSet`。
   - English: Iteration reads from the ordered `self.items` mapping, and reconstruction builds a list in that same order before constructing a new `OrderedSet`.
5. **第 967-991 行 / Lines 967-991 (reverse and pop)**:
   - 中文: `reversed()` 反转插入顺序，`pop()` 走 `dict.popitem()`，因此弹出的是最后插入的元素，而不是普通 set 的任意元素。
   - English: `reversed()` flips insertion order, while `pop()` uses `dict.popitem()`, making the newest element observable instead of returning an arbitrary set member.

## 类比 / The analogy

普通 `set` 像一袋没有编号的螺丝：你只关心某颗螺丝在不在。`OrderedSet` 像一叠按放入顺序排好的文件：不仅要知道文件在不在，还要能从最后一份开始倒着取。编译器必须把这两个容器当成不同的合同。

A regular `set` is a bag of unlabelled screws: only membership matters. An `OrderedSet` is a stack of files arranged by insertion time: you must preserve the order and be able to take the newest file first. The compiler needs two different contracts.

## 自己跑一遍 / Try it yourself

```python
class OrderedSet:
    def __init__(self, values=()):
        self._dict = dict.fromkeys(values)

    def add(self, value):
        self._dict[value] = None

    def pop(self):
        return self._dict.popitem()[0]

    def __iter__(self):
        return iter(self._dict)

    def __reversed__(self):
        return reversed(list(self._dict))

items = OrderedSet(["a", "b", "c"])
print(list(items))
print(list(reversed(items)))
print(items.pop(), list(items))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
['a', 'b', 'c']
['c', 'b', 'a']
c ['a', 'b']
```

中文: 这就是 Dynamo 不能简单复用普通 set 追踪逻辑的可观察差异。  
English: This observable difference is exactly why Dynamo cannot blindly reuse ordinary set tracking.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **TorchDynamo `SetVariable`** / **TorchDynamo `SetVariable`**: 中文: 共享 `BaseSetVariable` 的集合运算，但不保留插入顺序。 / English: It shares `BaseSetVariable` operations without promising insertion order.
- **Python dict-backed ordered containers** / **Python dict-backed ordered containers**: 中文: 利用 dict key order 同时获得 membership 和稳定遍历。 / English: They use dict key order to get both membership and stable iteration.
- **FX graph constants** / **FX graph constants**: 中文: 常量重建时也必须保留原对象的可观察行为，而不只是值集合。 / English: Constant reconstruction must preserve observable behavior, not merely the set of values.

## 注意事项 / Caveats / when it breaks

- **guard 必须匹配真实存储** / **Guards must match storage**: 中文: guard 如果读错来源，会出现编译图错误复用或不必要重编译。 / English: A guard attached to the wrong source can either reuse an invalid graph or trigger needless recompilation.
- **顺序会扩大编译合同** / **Order expands the compiler contract**: 中文: 迭代、repr、pop 都可能让原本“无序”的内部实现变成可观察行为。 / English: Iteration, repr, and pop can make what looks like an unordered internal detail observable.
- **纯 Python 类型需要额外追踪** / **Pure-Python types need extra tracing**: 中文: `OrderedSet` 没有 C-level flags，Dynamo 需要手动补 method flags 和成员映射。 / English: Because `OrderedSet` is pure Python, Dynamo supplies method flags and member mappings explicitly.

## 延伸阅读 / Further reading

- [PyTorch OrderedSet implementation](https://github.com/pytorch/pytorch/blob/39b52fbc3366d6e969c835c232f8bc854124ae43/torch/_dynamo/variables/sets.py)
- [TorchDynamo overview](https://pytorch.org/docs/stable/torch.compiler_dynamo_overview.html)
- [Python collections.abc MutableSet](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableSet)
