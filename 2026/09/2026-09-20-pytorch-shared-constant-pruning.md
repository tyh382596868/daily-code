---
date: 2026-09-20
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/_dynamo/guards.py
permalink: https://github.com/pytorch/pytorch/blob/50231cc1f690bf1d8794717042fc6baeff2d5993/torch/_dynamo/guards.py#L4210-L4231
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, pytorch, torch-dynamo, guards, serialization]
---

# PyTorch Dynamo shared constants：按 id 清理前先问“它是不是全局共享” / PyTorch Dynamo Shared Constants: Ask Whether an Object Is Globally Shared Before Pruning by ID

> **一句话 / In one line**: Dynamo 的 guard-state pruning 不能把 `None`、dtype 或可导入 class 当成普通对象删除，因为它们会被全进程共享的 `id()` 一起命中。 / Dynamo guard-state pruning must not delete `None`, dtypes, or importable classes like ordinary objects, because their process-wide `id()` is shared by unrelated references.

## 为什么重要 / Why this matters

保存编译产物时，Dynamo 会把一部分运行时对象从 guard state 里裁掉，再在加载时重建。裁剪键是 `id()`，所以“这个对象是否还需要”不是局部问题：如果删掉的是 `torch.float32`，tensor reducer、代码常量和别的模块属性可能都指向同一个对象。

When serializing compiled artifacts, Dynamo prunes some runtime objects from guard state and reconstructs them during loading. The pruning key is `id()`, so whether an object is needed is not local: deleting `torch.float32` can affect tensor reducer payloads, code constants, and unrelated module attributes that point to the same object.

这次修复把风险集中到 `_is_shared_constant`。它既识别 `FunctionPicklerBase._is_literal` 覆盖的精确类型，也额外保护空 tuple 和可按全名解析的 class；但 `<locals>` class 仍然允许裁剪，因为它不能被稳定地按模块路径 pickle。

The fix centralizes the risk in `_is_shared_constant`. It recognizes the exact literal types covered by `FunctionPicklerBase._is_literal`, adds empty tuples and importable classes, but still allows `<locals>` classes to be pruned because they cannot be reliably pickled by module path.

## 代码 / The code

`pytorch/pytorch` — [`torch/_dynamo/guards.py`](https://github.com/pytorch/pytorch/blob/50231cc1f690bf1d8794717042fc6baeff2d5993/torch/_dynamo/guards.py#L4210-L4231)

```python
def _is_shared_constant(value: Any) -> bool:
    """Whether pruning ``value`` by id would poison unrelated references to it.

    Pruning is keyed by ``id()``, and a literal such as ``torch.float32`` or
    ``Ellipsis`` is one object process-wide, so registering an unguarded
    reference as missing would turn EVERY other reference -- the dtype inside
    every tensor's reducer payload, a code object's constant -- into the
    sentinel. FunctionPicklerBase._is_literal names exactly those values (by
    exact type, so an IntEnum member or a str subclass is still pruned). The
    empty tuple is the one container CPython shares the same way (an empty
    frozenset is not); it matters for the module attribute loop, since a pytree
    leaf is never a tuple. A class is one object too (torch.Tensor is the pytype
    of every tensor payload); that matters for the local-scope leaf loop, since
    the module loop skips every callable. A class that pickle cannot find by
    name (a <locals> class) stays prunable: pickling it by reference would fail
    the dump, and the artifact would import its module at load.
    """
    if type(value) is tuple and not value:
        return True
    if inspect.isclass(value) and FunctionPicklerBase._fqn_resolves(value):
        return True
    return FunctionPicklerBase._is_literal(value)
```

调用点也很有教育意义：module attribute loop 在 `torch/_dynamo/guards.py:4657-4666` 跳过 shared constant，local-scope leaf loop 在 `torch/_dynamo/guards.py:4848-4862` 使用同一个判断。一个 helper 统一了两个对象图入口。

The call sites are instructive too: the module-attribute loop skips shared constants at `torch/_dynamo/guards.py:4657-4666`, while the local-scope leaf loop uses the same predicate at `torch/_dynamo/guards.py:4848-4862`. One helper now governs both object-graph entry points.

## 逐行讲解 / What's happening

1. **按精确类型识别 literal / Exact-type literal detection**:
   - 中文: `_is_literal` 保护的是进程级共享的字面量；精确类型检查避免把自定义 subclass 错当成可共享 singleton。
   - English: `_is_literal` protects process-shared literals; exact-type checks avoid treating custom subclasses as the same shared singleton.
2. **空 tuple 特判 / Empty tuple special case**:
   - 中文: CPython 会复用空 tuple；即使 pytree 不把 tuple 当 leaf，module attribute loop 仍可能遇到它。
   - English: CPython reuses the empty tuple; even though pytree does not expose tuples as leaves, the module-attribute loop can still see one.
3. **class 的可解析性 / Class resolvability**:
   - 中文: 可导入 class 的 identity 也可能被很多 payload 共享，因此只有 `_fqn_resolves` 成功时才保护。
   - English: Importable class identities can also be shared by many payloads, so they are protected only when `_fqn_resolves` succeeds.
4. **为什么 `<locals>` 仍可裁剪 / Why `<locals>` classes remain prunable**:
   - 中文: 局部 class 没有稳定的模块全名；强行保存引用会让 pickle dump 失败，加载时还会导入错误模块。
   - English: A local class has no stable module-qualified name; forcing a reference would fail the pickle dump and could import the wrong module at load time.
5. **使用同一 predicate / One predicate at both edges**:
   - 中文: guard tree 和 local scope 是两条不同的遍历路径，共用 helper 才不会一边修好、另一边仍把 singleton 放进 missing values。
   - English: The guard tree and local scope are different traversal paths; sharing the helper prevents fixing one path while the other still registers singletons as missing.

## 类比 / The analogy

这像整理办公室的失物招领。普通纸箱可以按“箱子 id”搬走，但公司总钥匙、公共打印机和墙上的时钟不能因为某个房间暂时没用就删除；它们的身份被整栋楼共享。

It is like cleaning an office lost-and-found. Ordinary boxes can be removed by a box ID, but the master key, shared printer, and wall clock cannot disappear because one room does not need them; the whole building shares their identity.

## 自己跑一遍 / Try it yourself

```python
import torch

def shared(value):
    if type(value) is tuple and not value:
        return True
    return value in (None, Ellipsis, torch.float32)

for value in (None, Ellipsis, torch.float32, (), [1], "local"):
    print(repr(value), shared(value))
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
None True
Ellipsis True
torch.float32 True
() True
[1] False
local False
```

中文: 真实实现还会处理可导入 class 和 exact-type literal；这个小实验先展示“按 identity 清理”的危险边界。
English: The real implementation also handles importable classes and exact-type literals; this small experiment shows the dangerous boundary of identity-based pruning.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **pickle reducers** / **pickle reducers**: 中文: reducer payload 里保存 dtype、device、class 等共享对象，恢复时依赖它们的 identity 或可导入路径。 / English: Reducer payloads store shared dtypes, devices, and classes whose identity or import path matters during restore.
- **interned strings and sentinels** / **interned strings and sentinels**: 中文: 单例 sentinel 的删除不能只看当前引用，因为其它结构可能复用了同一对象。 / English: Removing a singleton sentinel cannot depend only on the current reference because other structures may reuse that object.
- **compiler constant pools** / **compiler constant pools**: 中文: 常量池用共享对象节省空间，但也要求 GC、序列化和重写逻辑尊重共享关系。 / English: Constant pools save space through sharing, but GC, serialization, and rewriting must preserve those sharing relationships.

## 注意事项 / Caveats / when it breaks

- **identity 与 equality 不同** / **Identity is not equality**: 中文: `value in (...)` 只是教学示意；真实逻辑必须严格控制类型与解析性。 / English: `value in (...)` is only a teaching simplification; production logic must control exact types and resolvability.
- **共享 class 仍可能有动态状态** / **Shared classes can still carry dynamic state**: 中文: 保护 class identity 不等于保护 class 的实例属性，实例仍需按 guard tree 处理。 / English: Protecting a class identity does not protect its instance attributes; instances still need normal guard-tree handling.
- **修复要覆盖全部遍历入口** / **Cover every traversal entry**: 中文: 只在 module loop 加判断会留下 local-scope 或 reducer 路径的回归。 / English: Adding the check only to the module loop leaves regressions through local-scope or reducer paths.

## 延伸阅读 / Further reading

- [PyTorch shared-constant helper](https://github.com/pytorch/pytorch/blob/50231cc1f690bf1d8794717042fc6baeff2d5993/torch/_dynamo/guards.py#L4210-L4231)
- [Python pickle documentation](https://docs.python.org/3/library/pickle.html)
- [TorchDynamo serialization package](https://github.com/pytorch/pytorch/blob/50231cc1f690bf1d8794717042fc6baeff2d5993/torch/_dynamo/package.py)
