---
date: 2026-06-01
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/_higher_order_ops/associative_scan.py
permalink: https://github.com/pytorch/pytorch/blob/b169d39b8ed8cbe3a6499379faa51d33f8a66b81/torch/_higher_order_ops/associative_scan.py#L304-L403
difficulty: advanced
read_time: ~12 min
tags: [code-of-the-day, pytorch, higher-order-op, parallel-scan, state-space-model]
---

# PyTorch 把 Blelloch 并行扫描装进 100 行 HOP / PyTorch packs Blelloch parallel scan into a 100-line HOP

> **一句话 / In one line**: 把"前缀和"递归成"两两合并 → 递归 → 交错回去",`generic_associative_scan` 用 100 行 Python 写出了 Mamba/S5 训练背后的 O(log N) 扫描原语 / Recursively pair, scan halves, interleave — `generic_associative_scan` is the 100-line reference implementation of the O(log N) parallel-scan primitive that makes Mamba / S5 fast.

## 为什么重要 / Why this matters

线性时间 RNN / 状态空间模型(Mamba、RWKV、S5、minGRU)能在 GPU 上跑得比 attention 还快,靠的不是模型本身有多巧,而是它们的循环可以被改写成 **associative scan**,然后用 Blelloch 并行扫描算法在 O(log N) 深度内算完。这个文件是 PyTorch 新的 `torch.associative_scan` HOP 的 eager 参考实现——读一遍你就同时理解了(1)Blelloch 算法的递归结构,(2)PyTorch HOP 系统是怎么把"用户写的二元函数"塞进 IR 的,(3)为什么 SSM 训练能跑那么快。

Linear-time RNNs / state-space models (Mamba, RWKV, S5, minGRU) outrun attention not because their recurrence is clever, but because the recurrence can be rewritten as an **associative scan** and computed in O(log N) depth using the Blelloch parallel-scan algorithm. This file is the eager reference implementation behind PyTorch's new `torch.associative_scan` HOP — read it and you simultaneously understand (1) the recursive shape of Blelloch's algorithm, (2) how PyTorch's higher-order-op system threads a user-defined binary function into the IR, and (3) why SSM training is so fast.

## 代码 / The code

`pytorch/pytorch` — [`torch/_higher_order_ops/associative_scan.py`](https://github.com/pytorch/pytorch/blob/b169d39b8ed8cbe3a6499379faa51d33f8a66b81/torch/_higher_order_ops/associative_scan.py#L304-L403)

```python
def generic_associative_scan(operator, leaves, dim=0, additional_inputs=()):
    r"""
    This function performs the associative_scan operation.
    The algorithm works by recursively collecting neighbours of ``leaves`` and subsequently
    applying the ``operator`` on all pairs in parallel along ``dim``.
    The results of the recursive calls are later combined.
    """

    def call_operator(*args):
        return pytree.tree_leaves(operator(*args))

    def _scan(elems):
        """Perform the actual recursive scan on ``elems``."""
        num_elems = elems[0].shape[dim]

        if num_elems < 2:
            return elems

        reduced_elems = call_operator(
            *[aten.slice(elem, dim, 0, -1, 2) for elem in elems],
            *[aten.slice(elem, dim, 1, None, 2) for elem in elems],
            *additional_inputs,
        )

        # Recursively compute scan for partially reduced tensors.
        odd_elems = _scan(reduced_elems)

        if num_elems % 2 == 0:
            even_elems = call_operator(
                *[aten.slice(e, dim, 0, -1) for e in odd_elems],
                *[aten.slice(e, dim, 2, None, 2) for e in elems],
                *additional_inputs,
            )
        else:
            even_elems = call_operator(
                *odd_elems,
                *[aten.slice(e, dim, 2, None, 2) for e in elems],
                *additional_inputs,
            )

        # The first element of a scan is the same as the first element
        # of the original `elems`.
        even_elems = [
            torch.cat([aten.slice(elem, dim, 0, 1), result], dim=dim)
            if result.shape.numel() > 0 and elem.shape[dim] > 0
            else result
            if result.shape.numel() > 0
            else aten.slice(elem, dim, 0, 1)
            for (elem, result) in zip(elems, even_elems)
        ]

        return list(
            safe_map(functools.partial(_interleave, dim=dim), even_elems, odd_elems)
        )

    scans = _scan(leaves)
    return scans
```

## 逐行讲解 / What's happening

举个 `[0, 1, 2, 3]` + 加法的例子,目标是 `[0, 1, 3, 6]`(累加和)。

Take `[0, 1, 2, 3]` with `+` as the operator; we want `[0, 1, 3, 6]` (prefix sum).

1. **`reduced_elems = op(elems[0::2], elems[1::2])`**:
   - 中文: 把数组**偶数位置**(`[0, 2]`)和**奇数位置**(`[1, 3]`)两两喂给 operator,得到 `[0+1, 2+3] = [1, 5]`。这一步把长度对半砍,**所有 pair 是并行算的**——这就是 Blelloch 之所以叫"并行"扫描的关键。
   - English: feed the **even** positions (`[0, 2]`) and the **odd** positions (`[1, 3]`) pair-wise into the operator → `[0+1, 2+3] = [1, 5]`. This halves the length, and **all pairs run in parallel** — that's where the "parallel" in parallel scan comes from.

2. **`odd_elems = _scan(reduced_elems)`**:
   - 中文: 递归地对长度减半的数组再扫一遍,得到 `[1, 6]`。这一步对应"奇数下标的最终结果"——`odd_elems[0]=1` 是原数组前 2 个的和,`odd_elems[1]=6` 是原数组前 4 个的和。
   - English: recursively scan the halved array → `[1, 6]`. These are "odd-indexed final results" — `odd_elems[0]=1` is the sum of the first 2 originals, `odd_elems[1]=6` is the sum of the first 4.

3. **偶数下标(第 30 行起)/ even indices (line 30 onwards)**:
   - 中文: 偶数位置(下标 0, 2, ...)的最终结果 = "前一个奇数位置的结果" + "原数组对应的元素"。对长度为偶数的情况:`even_elems[i] = odd_elems[i-1] + elems[2*i]`,实现里就是 `op(odd_elems[:-1], elems[2::2])`,得到 `[1+2] = [3]`。然后把 `elems[0]=0` 拼到最前面,得到 `even_elems = [0, 3]`。
   - English: each even-indexed final = previous odd result + original element. For even length: `even_elems[i] = odd_elems[i-1] + elems[2*i]`, i.e. `op(odd_elems[:-1], elems[2::2])` → `[1+2] = [3]`. Then prepend `elems[0]=0` since the very first element of any prefix-sum is itself → `even_elems = [0, 3]`.

4. **`_interleave(even_elems, odd_elems)`**:
   - 中文: 把 `even=[0, 3]` 和 `odd=[1, 6]` 交错拼成 `[0, 1, 3, 6]`——完成!整个 `_scan` 的深度是 `O(log N)`,因为每次递归把长度对半砍。
   - English: interleave `even=[0, 3]` with `odd=[1, 6]` → `[0, 1, 3, 6]` — done! Total recursion depth is `O(log N)` because each call halves the length.

5. **`pytree.tree_leaves(operator(*args))`**:
   - 中文: 用户传进来的 `operator` 可能返回 pytree(比如 SSM 里 `combine` 返回 `(A, B)` 二元组)。`tree_leaves` 把它打平成 list of tensors,所以 `_scan` 内部统一处理 list,递归出去再让外层包回 pytree。这就是 HOP 系统支持任意结构化输入的标准做法。
   - English: the user's `operator` may return a pytree (e.g. an SSM's `combine` returns the pair `(A, B)`). `tree_leaves` flattens it to a list of tensors so `_scan` always operates on lists. The outer caller unflattens later. This is the standard HOP-system trick for supporting structured inputs.

6. **`additional_inputs`**:
   - 中文: 用户的 `operator` 可能闭包了外面的张量(比如 SSM 里的固定 `A` 矩阵)。HOP 不允许闭包,所以外层把这些"提取出来"的张量塞到 `additional_inputs` 里,每次递归调用都原样透传给 `operator`。
   - English: the user's `operator` may close over outer tensors (e.g. an SSM's fixed `A` matrix). HOPs don't allow closures, so the framework "lifts" those tensors into `additional_inputs` and re-threads them through every recursive `operator` call.

## 类比 / The analogy

想象一队 16 个士兵报"我前面的人数",一对一传话要 16 步。Blelloch 算法的做法:先让站在偶数位的人和右边邻居握手报和(8 个握手并行,1 步);递归对这 8 个和做同样的事;最后再让每个偶数位的人从他左边那位拿到"截止到我前面的总和",加上自己。整个过程 `log2(16) = 4` 步,而且每一步内部所有人同时动作。GPU 上的并行扫描就是把"士兵"换成"线程",把"握手"换成"shared-memory load+add"。

Picture 16 soldiers each reporting "how many people are to my left". One-at-a-time it takes 16 steps. Blelloch's trick: every even-positioned soldier shakes hands with their right neighbour to get a pair-sum (8 handshakes, all in parallel, 1 step); recurse on those 8 sums; finally every even-positioned soldier grabs "left-cumulative-up-to-me" from their left and adds their own count. Total: `log2(16) = 4` steps, with everyone moving simultaneously inside each step. GPUs swap "soldier" for "thread" and "handshake" for "shared-memory load+add" — same algorithm.

## 自己跑一遍 / Try it yourself

```python
# scan_demo.py — needs torch 2.6+
import torch
from torch._higher_order_ops.associative_scan import associative_scan

# 1) Plain prefix sum — same as cumsum
x = torch.arange(8, dtype=torch.float32)
print("cumsum:    ", associative_scan(lambda a, b: a + b, x, dim=0))
print("reference: ", x.cumsum(0))

# 2) Linear recurrence as an associative op:
#    h[t] = a[t] * h[t-1] + b[t]
#    combine((A1,B1), (A2,B2)) = (A1*A2, A2*B1 + B2)  -- this is associative!
def combine(left, right):
    A1, B1 = left
    A2, B2 = right
    return A2 * A1, A2 * B1 + B2

T = 6
a = torch.full((T,), 0.9)        # decay
b = torch.arange(1.0, T + 1.0)   # inputs
A, B = associative_scan(combine, (a, b), dim=0)
print("Mamba-style h[t]:", B)

# Verify with a sequential RNN
h, hs = 0.0, []
for t in range(T):
    h = a[t] * h + b[t]
    hs.append(h.item())
print("sequential h[t]:", hs)
```

运行 / Run with:
```bash
pip install --upgrade torch
python scan_demo.py
```

预期输出 / Expected output:
```
cumsum:     tensor([ 0.,  1.,  3.,  6., 10., 15., 21., 28.])
reference:  tensor([ 0.,  1.,  3.,  6., 10., 15., 21., 28.])
Mamba-style h[t]: tensor([1.0000, 2.9000, 5.6100, 9.0490, 13.1441, 17.8297])
sequential h[t]: [1.0, 2.9, 5.61, 9.049000000000001, 13.144100000000002, 17.829690000000003]
```

中文: 第二段是关键——`(A, B) ⊕ (C, D) = (CA, CB + D)` 这条结合律就是 Mamba/S5 把线性循环扫成并行的灵魂。同样一段循环,Blelloch 在 `log2(T)` 深度内算完。

English: the second snippet is the punchline — `(A, B) ⊕ (C, D) = (CA, CB + D)` is the associative combiner that lets Mamba / S5 turn a linear recurrence into a parallel scan. Same recurrence, but Blelloch computes it in `log2(T)` depth.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Mamba / Mamba-2** / **Mamba and Mamba-2**: 中文: 上面 demo 里的 `combine` 就是它们 selective-SSM 的本体,只是 `a, b` 变成依赖输入的张量。CUDA kernel 里手写的并行扫描和这里 `_scan` 是一一对应。
- **JAX `jax.lax.associative_scan`** / **JAX's `lax.associative_scan`**: 完全相同的 API,完全相同的算法,源码也几乎对得上——`generic_associative_scan` 的注释直接说自己是 JAX 版本的端口。
- **CUDA `cub::BlockScan` / `thrust::exclusive_scan`**: NVIDIA 工业级 GPU 扫描,内部就是 Blelloch + Kogge-Stone 的混合;PyTorch 这份是 eager 参考,真正在 Inductor 编译时会下推到这些 kernel。
- **`torch.cumsum` / `cumprod` / `logcumsumexp`**: 它们都是 `associative_scan` 的特例,只是 operator 固定。

## 注意事项 / Caveats / when it breaks

- **operator 必须满足结合律 / operator MUST be associative**: 中文: `(a + b) + c == a + (b + c)`,否则结果错且没有报错——是用户的责任。最常见的踩坑是浮点数加法**不严格结合**,所以并行扫描结果和顺序版会有最低 bit 的差。
- **`pure & pointwise`**: 中文: 不能有 side effect,不能用全局状态。HOP 框架会假设 operator 是纯函数才敢并行调用。
- **不支持 cross-dim 操作 / no cross-dim ops inside operator**: 因为 operator 在每个 `dim` 切片上被独立调用。要做 `dim=-1` 的 softmax 之类需要先 reshape。
- **eager 版很慢 / eager fallback is slow**: 中文: 这份 Python 实现是"语义参考",真要快得 `torch.compile` 一下让 Inductor 把它降到 CUDA。
- **递归深度 = `log2(seq_len)`**: 序列长度是 2^31 的话递归深度 31,Python 没问题但 IR 节点数会爆炸,生产 SSM 里通常先做 chunked scan(每 chunk 内并行扫,chunk 之间顺序拼)。

## 延伸阅读 / Further reading

- [PyTorch RFC: `torch.associative_scan`](https://github.com/pytorch/pytorch/issues/95408)
- [Blelloch (1990): *Prefix Sums and Their Applications*](https://www.cs.cmu.edu/~scandal/papers/CMU-CS-90-190.html) — 原始论文,讲并行 scan
- [Mamba paper §3.2 *Parallel Scan*](https://arxiv.org/abs/2312.00752) — 直接用这条结合律
- [JAX `jax.lax.associative_scan` source](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/control_flow/loops.py) — 算法的孪生版本
