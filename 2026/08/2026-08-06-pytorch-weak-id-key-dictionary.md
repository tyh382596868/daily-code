---
date: 2026-08-06
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/utils/weak.py
permalink: https://github.com/pytorch/pytorch/blob/7c70074e45d780864847433666fa071f5dda4237/torch/utils/weak.py#L153-L241
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, weakref]
---

# PyTorch WeakIdKeyDictionary：按身份弱引用对象 / PyTorch WeakIdKeyDictionary: Weak Keys by Object Identity

> **一句话 / In one line**: 这个小字典用 weakref 记住对象身份，让内部缓存不会把 Tensor 或模块意外续命。 / This small dictionary keys by weak object identity so internal caches do not accidentally keep tensors or modules alive.

## 为什么重要 / Why this matters

框架内部经常需要给 Python 对象挂缓存，但普通 dict 会强引用 key，缓存本身就可能导致对象无法释放。`WeakIdKeyDictionary` 选择“按身份比较 + 弱引用释放”，适合张量、模块、编译跟踪等生命周期敏感路径。

Framework internals often attach caches to Python objects, but a normal dict strongly references its keys. `WeakIdKeyDictionary` combines identity-based keys with weak-reference cleanup, which is useful for tensors, modules, and compiler tracking state.

## 代码 / The code

`pytorch/pytorch` — [`torch/utils/weak.py`](https://github.com/pytorch/pytorch/blob/7c70074e45d780864847433666fa071f5dda4237/torch/utils/weak.py#L153-L241)

```python
class WeakIdKeyDictionary(MutableMapping):
    def __init__(self, dict=None, ref_type=WeakIdRef) -> None:  # CHANGED
        self.data = {}

        self.ref_type = ref_type  # CHANGED

        def remove(k, selfref=ref(self)) -> None:
            self = selfref()
            if self is not None:
                if self._iterating:
                    self._pending_removals.append(k)
                else:
                    try:
                        del self.data[k]
                    except KeyError:
                        pass

        self._remove = remove
        self._pending_removals = []
        self._iterating = set()
        self._dirty_len = False
        if dict is not None:
            self.update(dict)

    def _commit_removals(self) -> None:
        pop = self._pending_removals.pop
        d = self.data
        while True:
            try:
                key = pop()
            except IndexError:
                return

            try:
                del d[key]
            except KeyError:
                pass

    def _scrub_removals(self) -> None:
        d = self.data
        self._pending_removals = [k for k in self._pending_removals if k in d]
        self._dirty_len = False

    def __delitem__(self, key) -> None:
        self._dirty_len = True
        del self.data[self.ref_type(key)]  # CHANGED

    def __getitem__(self, key):
        return self.data[self.ref_type(key)]  # CHANGED

    def __len__(self) -> int:
        if self._dirty_len and self._pending_removals:
            self._scrub_removals()
        return len(self.data) - len(self._pending_removals)

    def __setitem__(self, key, value) -> None:
        self.data[self.ref_type(key, self._remove)] = value  # CHANGED

    def copy(self):
        new = WeakIdKeyDictionary()
        with _IterationGuard(self):
            for key, value in self.data.items():
                o = key()
                if o is not None:
                    new[o] = value
        return new

    __copy__ = copy

    def __deepcopy__(self, memo):
        from copy import deepcopy

        new = self.__class__()
        with _IterationGuard(self):
            for key, value in self.data.items():
                o = key()
                if o is not None:
                    new[o] = deepcopy(value, memo)
        return new
```

## 逐行讲解 / What's happening

1. **第 153-158 行 / Lines 153-158 (storage)**:
   - 中文: 真正的数据放在 `self.data`，key 不是原对象，而是 `WeakIdRef` 包装后的弱引用。
   - English: Actual entries live in `self.data`; keys are not the original objects but weak references wrapped by `WeakIdRef`.
2. **第 159-170 行 / Lines 159-170 (callback removal)**:
   - 中文: key 对象被回收时，weakref callback 会删掉缓存项；如果正在迭代，就先加入待删除列表。
   - English: When the key object is collected, the weakref callback removes the cache entry; during iteration it defers deletion.
3. **第 201-219 行 / Lines 201-219 (mapping API)**:
   - 中文: `__getitem__`、`__setitem__`、`__delitem__` 都把用户传入的对象重新包装成同一种 ref。
   - English: `__getitem__`, `__setitem__`, and `__delitem__` wrap the user object into the same ref type before touching the backing dict.
4. **第 221-241 行 / Lines 221-241 (copy)**:
   - 中文: 拷贝时先调用 `key()` 取回活对象，死对象直接跳过。
   - English: Copying dereferences each weak key with `key()` and skips entries whose objects are already gone.

## 类比 / The analogy

它像健身房的临时储物柜：会员离开后，柜号记录自动失效；你不能因为登记了柜号，就让会员永远留在健身房。

It is like a temporary gym locker: once a member leaves, the locker record expires. Recording the locker number must not keep the member in the building.

## 自己跑一遍 / Try it yourself

```python
import weakref, gc

class Box: pass

store = {}
def remember(obj, value):
    store[weakref.ref(obj, lambda r: store.pop(r, None))] = value

box = Box()
remember(box, "cached")
print(len(store))
del box
gc.collect()
print(len(store))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
1
0
```

缓存没有强行留住 `box`，这就是弱引用 key 的核心价值。

The cache does not keep `box` alive; that is the core value of weak-reference keys.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Dynamo 编译缓存** / **Dynamo compile caches**: 跟踪对象状态时不能把用户对象生命周期拉长。 / Tracking object state must not extend user-object lifetimes.
- **模块 hook 注册表** / **Module hook registries**: hook 和对象引用关系必须避免循环持有。 / Hook registries must avoid accidental reference cycles.

## 注意事项 / Caveats / when it breaks

- **只能 key 可弱引用对象** / **Only weak-referenceable objects work**: `int`、`tuple` 这类内建值不能直接弱引用。 / Built-in values like `int` or `tuple` cannot be weak-referenced directly.
- **迭代期间延迟删除** / **Deletion is deferred while iterating**: 长度和底层 dict 可能短时间不同步，所以代码要维护 `_pending_removals`。 / Length and backing storage can briefly diverge, so `_pending_removals` is needed.

## 延伸阅读 / Further reading

- [Python weakref documentation](https://docs.python.org/3/library/weakref.html)
- [PyTorch weak utilities](https://github.com/pytorch/pytorch/blob/7c70074e45d780864847433666fa071f5dda4237/torch/utils/weak.py)
