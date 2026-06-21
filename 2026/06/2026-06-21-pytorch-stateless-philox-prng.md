---
date: 2026-06-21
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/func/_random.py
permalink: https://github.com/pytorch/pytorch/blob/dc3fad579e528b145f60cb08bc9f2d2a30dca1d4/torch/func/_random.py#L1-L140
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, prng, stateless, philox, vmap, functional]
---

# PyTorch 终于有 JAX 风格的无状态随机数了:Philox key / split / fold_in / normal_ / PyTorch ships JAX-style stateless PRNG: Philox key / split / fold_in / normal_

> **一句话 / In one line**: `torch.func._random` 引入了 JAX 式的无状态 PRNG:一个 key 是 `uint64[seed, offset]` 张量,`split()` 生成子 key,`fold_in(key, i)` 关联层号或循环索引,全部都不碰全局 RNG 状态。/ `torch.func._random` introduces JAX-style stateless PRNG: a key is a `uint64[seed, offset]` tensor, `split()` derives independent child keys, `fold_in(key, i)` binds a key to a layer index or loop counter — none of this touches global RNG state.

## 为什么重要 / Why this matters

PyTorch 传统的 `torch.randn(...)` 依赖全局 RNG 状态——这在 `vmap` 和 `torch.compile` 里是个麻烦:并行执行的多条路径会竞争同一个 RNG 状态,导致随机性不可组合、不可重现。JAX 用"key 是值,不是状态"解决了这个问题。

`torch.func._random` 现在把同样的心智模型带进 PyTorch:key 是一个普通的 `uint64` 张量 `[seed, offset]`;`split(key, n)` 调用底层 Philox 计数器 RNG 派生出 `n` 个互相独立的子 key;`fold_in(key, i)` 可以把循环变量或层号"嵌入" key,而不需要手动 split。`normal_(key, result)` 就地用 Philox 填充 `result`,输出完全由 key 决定——同一个 key 永远产生同一批随机数。

PyTorch's classical `torch.randn(...)` relies on a global RNG state — a headache inside `vmap` and `torch.compile`, where multiple parallel execution paths race on the same RNG and lose reproducibility. JAX solved this with "key is a value, not state".

`torch.func._random` brings that same mental model to PyTorch: a key is a plain `uint64` tensor `[seed, offset]`; `split(key, n)` calls the underlying Philox counter RNG to derive `n` independent child keys; `fold_in(key, i)` folds a loop variable or layer index into a key without a manual split. `normal_(key, result)` fills `result` in-place using Philox — output is fully determined by the key, so the same key always produces the same numbers.

## 代码 / The code

`pytorch/pytorch` — [`torch/func/_random.py`](https://github.com/pytorch/pytorch/blob/dc3fad579e528b145f60cb08bc9f2d2a30dca1d4/torch/func/_random.py#L1-L140)

```python
def key(seed: int, impl: str = "philox4x32-10",
        device: torch.device | None = None) -> torch.Tensor:
    if impl != "philox4x32-10":
        raise NotImplementedError(f"key() does not support PRNG impl '{impl}'")
    # key = (seed, offset) — offset 从 0 开始,每次采样后内部自增
    return torch.tensor([seed, 0], dtype=torch.uint64, device=device)


def split(key: torch.Tensor, num: int = 2) -> torch.Tensor:
    # 调用 Philox 派生算法:给定父 key → 生成 num 个独立子 key
    return torch.ops.aten._philox_key_split(key, num)


def fold_in(key: torch.Tensor, data: int) -> torch.Tensor:
    # 等价于 split(key, data + 1)[data],但只派生一个 key,更高效
    return torch.ops.aten._philox_key_fold_in(key, data)


def normal_(key: torch.Tensor, result: torch.Tensor,
            *, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    # 就地填充:result 由 key 完全决定
    return torch.ops.aten._philox_normal_(result, key, mean, std)


def normal(key: torch.Tensor, *shape, mean=0.0, std=1.0,
           dtype=None) -> torch.Tensor:
    if len(shape) == 1 and isinstance(shape[0], Sequence):
        shape = tuple(shape[0])
    if dtype is None:
        dtype = torch.float32
    result = torch.empty(shape, dtype=dtype, device=key.device)
    return normal_(key, result, mean=mean, std=std)
```

## 逐行讲解 / What's happening

1. **`key(seed)` — key 的本质是 `uint64[2]`**
   - 中文: `(seed, 0)` — 第一个分量是用户提供的种子,第二个分量是计数器偏移量(Philox counter-based RNG 的 `offset`),初始为 0。底层 Philox 算法会把 `(seed, offset)` 当作 128 位计数器用于加密散列生成伪随机数。
   - English: `(seed, 0)` — the first element is the user seed, the second is a counter offset (the `offset` inside Philox), starting at 0. The underlying Philox algorithm treats the combined `(seed, offset)` as a 128-bit counter fed into an AES-like key-schedule to produce pseudorandom output.

2. **`split(key, num)` — 派生子 key,不共享状态**
   - 中文: 调用 `torch.ops.aten._philox_key_split`。这不是简单的 seed+i 递增——Philox 保证不同 key 的输出流在统计意义上是独立的。关键点:这个函数没有副作用,连续两次对同一个 key 调用 `split` 得到完全相同的子 key 数组。支持 batched key:如果 `key.shape == (*batch, K)`,输出是 `(num, *batch, K)`。
   - English: Calls `torch.ops.aten._philox_key_split`. This is not a simple seed+i increment — Philox guarantees that streams from different keys are statistically independent. Key property: no side effects. Calling `split` twice on the same key returns the exact same child keys. Supports batched keys: if `key.shape == (*batch, K)`, the result is `(num, *batch, K)`.

3. **`fold_in(key, data)` — 用整数给 key 打标签**
   - 中文: 在按层编号初始化权重、或在 `vmap` 的内层循环里区分不同 agent 的 key 时非常有用。等价于 `split(key, data+1)[data]`,但只需计算一个子 key 而不是 `data+1` 个——在 data 大的时候更高效。
   - English: Useful for layer-index weight initialization or for distinguishing per-agent keys inside a `vmap` inner loop. Equivalent to `split(key, data+1)[data]`, but computes only one child key instead of `data+1` — more efficient when `data` is large.

4. **`normal_(key, result)` — 确定性就地填充**
   - 中文: 注意参数顺序:`aten._philox_normal_(result, key, mean, std)` — 底层 aten op 把 `result` 放第一位(修改目标),`key` 放第二位。Python 层 wrapper `normal_` 把顺序调成"key 先"更符合 JAX 习惯。返回 `result` 本身,方便链式调用。
   - English: Note the argument swap at the aten layer: `aten._philox_normal_(result, key, ...)` puts `result` first (the mutation target) and `key` second. The Python-level wrapper `normal_` swaps them to put `key` first, matching JAX conventions. Returns `result` itself to allow chaining.

## 类比 / The analogy

想象每个 key 是一把带编号的钥匙。`split` 是锁匠从一把主钥匙刻出若干把子钥匙——任何一把子钥匙都打不开其他子钥匙对应的锁,但你用同一把主钥匙再刻一次,得到的还是完全一样的子钥匙集合。`fold_in(key, 5)` 是"在钥匙上刻了个数字 5",变成了专门开第 5 扇门的子钥匙——同一把主钥匙 + 同一个数字永远得到同一把子钥匙。`torch.randn()` 的全局 RNG 则像一个公共钥匙圈:任何人用一次都会让整个圈子状态改变,其他人就算用同一把圈上的钥匙也得到不同结果。

Think of each key as a numbered master key. `split` is a locksmith cutting child keys from a master — no child key opens any other child key's lock, but cutting from the same master again produces the exact same child set. `fold_in(key, 5)` is engraving "5" on the key, making it the dedicated key for door number 5 — same master + same number always yields the same child key. PyTorch's global `torch.randn()` is like a shared key ring: anyone using it advances the ring's state, so other users of the same key get different results.

## 自己跑一遍 / Try it yourself

```python
import torch
from torch.func import _random as R

# 1. 创建 key
k = R.key(42)
print("key:", k)                      # tensor([42,  0], dtype=torch.uint64)

# 2. split 派生两个子 key
k1, k2 = R.split(k, 2)
print("k1:", k1, "k2:", k2)          # 不同,但确定性

# 3. fold_in:按层号给 key 打标签
layer_keys = [R.fold_in(k, i) for i in range(4)]
print("layer key 2:", layer_keys[2])

# 4. 同一个 key,同样的随机数
x = R.normal(k1, (3,))
y = R.normal(k1, (3,))
print("equal?", torch.allclose(x, y))  # True

# 5. 不同 key,不同随机数
z = R.normal(k2, (3,))
print("different?", not torch.allclose(x, z))  # True
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
key: tensor([42,  0], dtype=torch.uint64)
k1: tensor([...], dtype=torch.uint64)  k2: tensor([...], dtype=torch.uint64)
equal? True
different? True
```

中文: 第 4 步证明了"无状态"的核心含义——同一个 key 调两次 `normal` 永远给同样结果,因为输出只取决于 key,不取决于调用顺序。

English: Step 4 proves the "stateless" core property — calling `normal` twice with the same key always returns identical tensors, because the output depends only on the key, never on call order.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **JAX `jax.random`** / **JAX 随机数**: 所有 JAX 随机 API 都是无状态的:key 是一个 `uint32[2]`,`jax.random.split(key)` 返回子 key 数组。PyTorch 的 `torch.func._random` 是对这套 API 的直接移植。/ All JAX random APIs are stateless: key is `uint32[2]`, `jax.random.split` returns child key arrays. PyTorch's `torch.func._random` is a direct port of this design.
- **`torch.compile` + `vmap` 内部** / **inside `torch.compile` + `vmap`**: `vmap` 会沿 batch 维度向量化函数——如果函数里有 `torch.randn`,不同 batch 项会竞争同一全局 RNG。用 batched key + `split` 可以让每个 batch 项得到独立的随机流,确保结果与顺序无关。/ `vmap` vectorises functions across a batch dimension. If the function calls `torch.randn`, different batch elements race on the same global RNG. Using batched keys + `split` gives each element its own independent stream, making results independent of execution order.
- **Linen / Flax 的 `self.make_rng`** / **Flax's `self.make_rng`**: Flax 用 `fold_in(key, module_hash)` 给每个子模块生成专用 key——和 `torch.func._random.fold_in(key, layer_idx)` 的用途完全一致。/ Flax uses `fold_in(key, module_hash)` to give each submodule a dedicated key — exactly the same use-case as `torch.func._random.fold_in(key, layer_idx)`.

## 注意事项 / Caveats / when it breaks

- **仍在 `torch.func._random` 命名空间** / **Still under `torch.func._random`**: 前缀 `_` 表示 experimental API,接口可能在没有 deprecation warning 的情况下变更。不要在生产代码里 hardcode 这个路径。/ The `_` prefix marks this as experimental — the interface may change without a deprecation warning. Don't hardcode this import path in production code.
- **只支持 `philox4x32-10`** / **Only `philox4x32-10` is implemented**: 如果你需要 `xoshiro` 或其他 RNG 算法,目前只能自己实现。/ If you need `xoshiro` or other RNG algorithms, you'll need to implement them yourself for now.
- **非 vmap 场景不是必须用** / **Not required outside vmap**: 在普通顺序代码里,全局 `torch.manual_seed()` + `torch.randn()` 仍然是更简单的选择——无状态 PRNG 的优势主要体现在并行/函数式变换里。/ In ordinary sequential code, global `torch.manual_seed()` + `torch.randn()` is simpler and sufficient. The stateless API's advantage is primarily in parallel/functional transform contexts.

## 延伸阅读 / Further reading

- Philox counter-based RNG 原论文: [Random123 paper (SC 2011)](https://dl.acm.org/doi/10.1145/2063384.2063405)
- JAX 无状态随机数设计文档: [jax.random documentation](https://jax.readthedocs.io/en/latest/jax.random.html)
- PyTorch `torch.func` 总览: [Functorch docs](https://pytorch.org/docs/stable/func.html)
