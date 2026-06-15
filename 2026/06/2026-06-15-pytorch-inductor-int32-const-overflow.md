---
date: 2026-06-15
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/_inductor/codegen/simd_kernel_features.py
permalink: https://github.com/pytorch/pytorch/blob/57d0ff82a28071bbf3aa9c5ed6321ae9903eea51/torch/_inductor/codegen/simd_kernel_features.py#L133-L179
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, pytorch, inductor, compiler, sympy, int32-overflow]
---

# PyTorch Inductor 现在会检查"地址表达式里的常数项"是否会溢出 int32 / PyTorch Inductor now inspects the *constant term* of fused address expressions for int32 overflow

> **一句话 / In one line**: 把所有自由符号置零、提取出地址表达式的常数偏移、判断它是否在 `[-2³¹, 2³¹-1]` 之外——一旦超出就强制 index dtype 升到 int64。 / Substitute every free symbol with 0, extract the constant offset of the address expression, and if it falls outside `[-2³¹, 2³¹-1]`, promote `index_dtype` to int64.

## 为什么重要 / Why this matters

Inductor 决定一个融合 kernel 用 int32 还是 int64 做寻址,本来只看 `total_numel` 和每个 buffer 的 `storage_size` 是否落在 int32 范围。但有一类常见的 bug 它原来抓不到:**slice_scatter / mm.backward 这种"拼接型"算子,融合后的最终地址表达式会带一个非常大的常数项**。形如 `base + x0 + 310*x1`,即使 `x0`、`x1` 都在 int32 内,`base` 单独就已经超过 2³¹ 了——融合到一个 Triton kernel 里就出现 `Scalar -2779057358 is out of range for type int32`,直接报死。这个 PR 用了一个非常简洁的 sympy 技巧把这种"常数项越界"挑出来了。

Inductor used to decide int32 vs int64 indexing by checking `total_numel` and each buffer's `storage_size` against INT32_MAX. But a whole class of bugs slipped through: **slice_scatter / mm.backward fusions whose final address expression carries a large *constant* offset.** Take a fused expression like `base + x0 + 310*x1` — even when each symbolic part fits int32, `base` alone may exceed 2³¹. Triton compilation then raises `Scalar -2779057358 is out of range for type int32`. This PR uses a tight sympy trick to catch exactly that case.

## 代码 / The code

`pytorch/pytorch` — [`torch/_inductor/codegen/simd_kernel_features.py`](https://github.com/pytorch/pytorch/blob/57d0ff82a28071bbf3aa9c5ed6321ae9903eea51/torch/_inductor/codegen/simd_kernel_features.py#L133-L179)

```python
def select_index_dtype(self) -> torch.dtype:
    # Gather all used buffer names
    buffer_names: OrderedSet[str] = OrderedSet()
    for node in self.scheduler_nodes():
        buffer_names.update(node.get_buffer_names())
        buffer_names.update(node.used_buffer_names())
    buffers = [V.graph.get_buffer(name) for name in buffer_names]

    # In theory we can separately check xnumel and rnumel are <= int_max
    # but some indexers do use the full linear index so we need to be
    # conservative here.
    total_numel = self.numel * self.reduction_numel

    from .simd import SIMDScheduling

    if SIMDScheduling.can_use_32bit_indexing(total_numel, buffers):
        # Fused address expressions may carry a constant term that
        # overflows int32 even when numel/storage_size are within range.
        if self.any_index_expr_const_overflows_int32():
            return torch.int64
        return torch.int32
    return torch.int64

@cache_on_self
def any_index_expr_const_overflows_int32(self) -> bool:
    """Return True if any MemoryDep index has a constant term outside
    [-2**31, 2**31 - 1]."""
    int32_max = sympy.Integer(2**31 - 1)
    int32_min = sympy.Integer(-(2**31))
    for node in self.scheduler_nodes():
        for dep in itertools.chain(node.read_writes.reads, node.read_writes.writes):
            if not isinstance(dep, MemoryDep):
                continue
            index = dep.index
            if not isinstance(index, sympy.Expr):
                continue
            try:
                const_part = index.subs(
                    {s: sympy.Integer(0) for s in index.free_symbols}
                )
                if not isinstance(const_part, sympy.Expr):
                    continue
                if const_part > int32_max or const_part < int32_min:
                    return True
            except (ZeroDivisionError, TypeError, ValueError):
                continue
    return False
```

## 逐行讲解 / What's happening

1. **第 148 行 / Line 148 (`can_use_32bit_indexing(total_numel, buffers)`)**:
   - 中文: 这是旧的 32-bit 索引判定——只看 `total_numel`(`xnumel * rnumel`)和每个 buffer 的 `storage_size` 加起来是否能塞进 int32。
   - English: This is the old 32-bit indexing gate — just `total_numel` (`xnumel * rnumel`) plus each buffer's `storage_size`, all summed and compared to INT32_MAX.

2. **第 149-152 行 / Lines 149-152 (the new guard)**:
   - 中文: 即使旧条件说"int32 够用",新条件追加一道:再扫一遍所有 MemoryDep,如果哪个地址表达式的"纯常数部分"超出 int32 范围,就升级到 int64。
   - English: Even if the old check says "int32 is fine", a second gate is appended — scan every MemoryDep; if any index expression's pure constant part is outside int32, promote to int64.

3. **第 160-161 行 / Lines 160-161 (`int32_max`, `int32_min`)**:
   - 中文: 注意这里用的是 **非对称** 区间 `[-2**31, 2**31 - 1]`,不是 `Abs(c) <= 2**31 - 1`。原因:int32 是 32 位补码,负数能表示到 `-2**31`,但正数只能到 `2**31 - 1`——这就是为什么 `INT_MIN == -INT_MAX - 1`。如果用 `Abs(c)` 就会把 `-2**31` 误判成越界。
   - English: The range is **asymmetric** — `[-2**31, 2**31 - 1]`, not `Abs(c) <= 2**31 - 1`. Two's complement: negatives reach `-2**31`, positives only `2**31 - 1`. That's why `INT_MIN == -INT_MAX - 1`. Using `Abs(c)` would wrongly flag the legal value `-2**31`.

4. **第 170-172 行 / Lines 170-172 (`index.subs({s: 0 for s in index.free_symbols})`)**:
   - 中文: 这是整个 patch 的灵魂——把表达式里 *所有* 自由符号都置 0,sympy 自然把剩下的部分化简成纯常数。比如 `base + x0 + 310*x1`,把 `x0=0, x1=0` 代进去就剩下 `base`。这种"代入零取常数"的技巧是 sympy 里提取多项式常数项最简单可靠的方法,不用自己写 polynomial canonicalize。
   - English: The heart of the patch — substitute *every* free symbol with 0, and sympy reduces the rest to a pure constant. `base + x0 + 310*x1` with `x0=0, x1=0` collapses to `base`. The "substitute-zero" idiom is the simplest, most reliable way in sympy to extract a polynomial's constant term — you don't need to roll your own canonicalization pass.

5. **第 169 行 + 177-178 行 / Line 169 + 177-178 (the `try ... except`)**:
   - 中文: 包了一层异常吞噬——`ZeroDivisionError / TypeError / ValueError`。为什么需要?因为 Inductor 的 index 表达式里可能混进 floor division、mod 之类的非线性算子,代入 0 之后可能 `1/0` 或类型不匹配。这种边界 case 不应该让整个判定崩溃,而是退让回旧逻辑(只看 numel)。
   - English: A `try ... except` swallows `ZeroDivisionError / TypeError / ValueError`. Inductor's index expressions can contain floor division, mod, and other non-linear ops; substituting 0 might trigger `1/0` or a type mismatch. The fallback is to skip this dep (don't promote) — the patch refuses to crash the whole compilation just because a corner case threw.

6. **第 156 行 / Line 156 (`@cache_on_self`)**:
   - 中文: 这个扫描函数被 `cache_on_self` 包了,意思是同一个 `SIMDKernelFeatures` 实例上只算一次。融合调度可能多次调用 `select_index_dtype`,缓存能保证扫所有 MemoryDep 这种 O(节点数 × dep 数)的工作只做一遍。
   - English: `cache_on_self` memoizes the scan per-instance. Fusion may call `select_index_dtype` multiple times; the cache keeps the O(nodes × deps) MemoryDep traversal from happening repeatedly.

## 类比 / The analogy

想象你帮一个超大书店做"书架坐标"系统:每本书的位置是 `(走廊号, 行号, 列号)` 用一个公式合成一个全局编号。你之前只检查"走廊数、行数、列数都没超过 65535"——以为整个全局编号肯定在 32-bit 范围内。但某一天你新开了一片书架,起始编号定在 30 亿("跟原来的旧书架隔开,避免编号冲突"),结果客户搜索时编号溢出,系统返回"找不到这本书"。这个 patch 加的就是一道:**不仅看每个分量的取值范围,还要看"起始偏移"——也就是把走廊号、行号、列号都填 0 后剩下的那个数——是不是已经爆了**。

Imagine a huge bookstore that gives every book a global ID computed from `(aisle, row, column)`. The old check verified each of `aisle, row, column` is `< 65535`, assuming the resulting global ID will fit a 32-bit int. Then a new wing opens with its starting offset at 3 billion ("to keep IDs disjoint from the old wing"), and customer searches start returning "book not found" because the global ID overflows. This patch adds the missing check: **inspect not just each component's range, but also the *starting offset* — what you get when every component is zero — and make sure even that fits.**

## 自己跑一遍 / Try it yourself

```python
import sympy

x0, x1 = sympy.symbols("x0 x1")

INT32_MAX = sympy.Integer(2**31 - 1)
INT32_MIN = sympy.Integer(-(2**31))

# Three example fused address expressions.
exprs = [
    x0 + 310 * x1,                          # purely symbolic, const_part = 0
    3_000_000_000 + x0 + 310 * x1,          # large positive constant (~3e9)
    -2**31 + x0,                            # exactly INT32_MIN — borderline, still OK
    -2**31 - 1 + x0,                        # one past INT32_MIN, must promote
]

for e in exprs:
    const_part = e.subs({s: sympy.Integer(0) for s in e.free_symbols})
    overflows = const_part > INT32_MAX or const_part < INT32_MIN
    print(f"{str(e):>35s}   const_part={const_part:>14d}   overflows={overflows}")
```

运行 / Run with:
```bash
pip install sympy
python try.py
```

预期输出 / Expected output:
```
                       x0 + 310*x1   const_part=             0   overflows=False
        x0 + 310*x1 + 3000000000     const_part=    3000000000   overflows=True
                  x0 - 2147483648    const_part=   -2147483648   overflows=False
                  x0 - 2147483649    const_part=   -2147483649   overflows=True
```

中文一两句:看第三行—— `-2³¹` 是 *合法* 的 int32 值,所以不算溢出;第四行差一个,就越界了。这就是为什么 patch 里要分开比较 `> int32_max` 和 `< int32_min`,而不是用对称的 `Abs(c) > 2**31 - 1`。

In English: row 3 is the asymmetry tell — `-2**31` *is* a legal int32, so it must not be flagged. Row 4 is one off and must be flagged. That's exactly why the patch compares against `int32_max` and `int32_min` separately, rather than `Abs(c) > 2**31 - 1`.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **CUDA / Triton kernel addressing in general** / **CUDA / Triton 地址计算**: 任何用 32-bit 整数索引大张量的 kernel 都吃这一套。 cuBLAS / cuDNN 里 size-product 接近 2³¹ 的张量永远要用 int64 stride。 / Any kernel that uses 32-bit integer addressing for large tensors faces this. cuBLAS / cuDNN switch to int64 strides once size product approaches 2³¹.
- **LLVM ScalarEvolution** / **LLVM ScalarEvolution**: 编译器在做循环展开 / 向量化前也要分析归纳变量的范围,LLVM 的 `SCEV` 类有完全类似的 "extract constant part, check overflow" 流程。 / LLVM's `ScalarEvolution` does the same constant-overflow check before deciding to vectorize a loop.
- **XLA's bit-width inference** / **XLA 的 bit-width 推断**: XLA HLO 的 layout / stride 推断会对 `dynamic_slice` 类算子做类似的"常数偏移上溢"检查。 / XLA HLO's layout / stride inference does similar overflow checks for `dynamic_slice`-family ops.

## 注意事项 / Caveats / when it breaks

- **`free_symbols` 一定要全部代换 / replace *all* free symbols**: 漏掉一个 `xnumel` 之类的就会拿到一个还有符号的表达式,不能跟整数比大小。 patch 里的 `isinstance(const_part, sympy.Expr)` 是一道保险——如果某个奇怪的 dep 让 `subs` 之后还剩符号,直接 skip。 / Missing one free symbol leaves a residual symbol that you can't compare against an integer. The `isinstance(const_part, sympy.Expr)` guard skips that dep instead of crashing.
- **常数项是必要不充分条件 / Constant overflow is sufficient but not necessary**: 这个 patch 只抓"纯常数已经爆了"的情况。还存在另一类:常数没爆,但 `stride * symbol` 在最大取值下爆了。`can_use_32bit_indexing` 里已经处理了 numel,但理论上严格的检查需要符号区间分析(symbol range analysis),复杂度高得多。 / The patch only catches "constant alone overflows". A stricter check would do symbolic range analysis on `stride * symbol_max`, which is significantly more expensive.
- **`@cache_on_self` 的生命周期 / cache lifetime**: 缓存绑在 `SIMDKernelFeatures` 实例上,只要 fusion 阶段没有重新构造它,缓存就有效。如果上层逻辑哪天换成"每次 query 都新建 features",这个缓存就退化成无效装饰器。 / The cache is bound to the `SIMDKernelFeatures` instance; if upstream switches to rebuilding it per query, the decorator becomes a no-op.

## 延伸阅读 / Further reading

- [PR #186060 — full diff and discussion](https://github.com/pytorch/pytorch/pull/186060)
- [Repro test in `test/inductor/test_cuda_repro.py`](https://github.com/pytorch/pytorch/blob/57d0ff82a28071bbf3aa9c5ed6321ae9903eea51/test/inductor/test_cuda_repro.py)
- [sympy `subs` documentation](https://docs.sympy.org/latest/modules/core.html#sympy.core.basic.Basic.subs)
- [LLVM ScalarEvolution overview](https://llvm.org/docs/Passes.html#scalar-evolution) — for a much larger version of the same analysis.
