---
date: 2026-05-28
topic: infrastructure
source: tracked
repo: Dao-AILab/flash-attention
file: flash_attn/utils/distributed.py
permalink: https://github.com/Dao-AILab/flash-attention/blob/59f01d6e1a1655a148ed4b22b5d4fbb9da2c2cf0/flash_attn/utils/distributed.py#L19-L104
difficulty: intermediate
read_time: ~12 min
tags: [code-of-the-day, infrastructure, sequence-parallel, autograd, collectives, distributed]
---

# 把 NCCL 通信原语包成可微分算子:flash-attention 30 行实现序列并行 / Wrapping NCCL collectives as differentiable ops: flash-attention's 30-line sequence-parallel primitives

> **一句话 / In one line**: 序列并行训练需要让 `all_gather` / `reduce_scatter` / `all_reduce` 参与反向传播,而 PyTorch 原生 c10d 算子是不可微的——flash-attention 用 3 个 `torch.autograd.Function`,把每个 collective 的反向定义为它的"对偶 collective",就把一切串了起来。 / Sequence parallel training needs `all_gather` / `reduce_scatter` / `all_reduce` to participate in autograd, but PyTorch's raw c10d ops are non-differentiable. flash-attention defines three `torch.autograd.Function`s where each collective's backward is its *dual* collective, and the whole machinery falls out.

## 为什么重要 / Why this matters

序列并行(sequence parallelism, SP)是 Megatron / Llama / Mistral 训练时常用的策略:在 LayerNorm / Dropout 这类逐 token 操作之前把 batch 沿 sequence 维切到不同 rank,做完之后再 gather 回来——目的是把激活的内存压力从 O(b·s·h) 降到 O(b·s/p·h)。问题是:从 `x_local` 到 `x_full = all_gather(x_local)` 这一步,前向要做 NCCL 的 `_all_gather_base`,而 PyTorch 的 c10d 直接调用是**没有 autograd 信息**的——`backward` 跟过来一看根本不知道梯度怎么往回传。flash-attention 用最小代价解决了这个问题:每个 collective 写一个 `torch.autograd.Function`,前向就是 NCCL 调用,反向写另一个对偶的 NCCL 调用。配对关系也非常优雅:`all_gather` 的反向是 `reduce_scatter`(因为聚合的梯度等于按 rank 切回去再 sum),`reduce_scatter` 的反向是 `all_gather`(对称),`all_reduce` 的反向是恒等(每个 rank 已经看到求和后的梯度)。这段 30 行代码是 Megatron 风格 SP 的最小核心,读完之后你就能自己手写一个序列并行的 LayerNorm。

Sequence parallelism (SP) is a staple of Megatron / Llama / Mistral training: before per-token ops like LayerNorm or Dropout, you shard the batch along the sequence dimension across ranks, then gather back afterwards — cutting activation memory from O(b·s·h) to O(b·s/p·h). The friction: stepping from `x_local` to `x_full = all_gather(x_local)` requires a NCCL `_all_gather_base`, and PyTorch's c10d ops are **non-differentiable when invoked directly** — autograd has no clue how the gradient should flow backwards. flash-attention solves it with the minimum possible code: one `torch.autograd.Function` per collective, where the forward calls the NCCL primitive and the backward calls the *dual* primitive. The pairings are themselves the lesson: `all_gather`'s backward is `reduce_scatter` (because the gradient of "concatenate-everyone's-shard" is "split-then-sum"), `reduce_scatter`'s backward is `all_gather` (symmetry), and `all_reduce`'s backward is identity (every rank already received the summed gradient). These 30 lines are the minimal core of Megatron-style SP — internalize them and you can roll your own sequence-parallel LayerNorm by hand.

## 代码 / The code

`Dao-AILab/flash-attention` — [`flash_attn/utils/distributed.py`](https://github.com/Dao-AILab/flash-attention/blob/59f01d6e1a1655a148ed4b22b5d4fbb9da2c2cf0/flash_attn/utils/distributed.py#L19-L104)

```python
# Raw operation, does not support autograd, but does support async
def all_gather_raw(input_: Tensor, process_group: ProcessGroup, async_op: bool = False):
    world_size = torch.distributed.get_world_size(process_group)
    output = torch.empty(
        world_size * input_.shape[0], *input_.shape[1:], dtype=input_.dtype, device=input_.device
    )
    handle = torch.distributed.all_gather_into_tensor(
        output, input_.contiguous(), group=process_group, async_op=async_op
    )
    return output, handle


# Raw operation, does not support autograd, but does support async
def reduce_scatter_raw(input_: Tensor, process_group: ProcessGroup, async_op: bool = False):
    world_size = torch.distributed.get_world_size(process_group)
    assert input_.shape[0] % world_size == 0
    output = torch.empty(
        input_.shape[0] // world_size, *input_.shape[1:], dtype=input_.dtype, device=input_.device
    )
    handle = torch.distributed.reduce_scatter_tensor(
        output, input_.contiguous(), group=process_group, async_op=async_op
    )
    return output, handle


# Raw operation, does not support autograd, but does support async
def all_reduce_raw(input_: Tensor, process_group: ProcessGroup, async_op: bool = False):
    input_ = input_.contiguous()
    handle = torch.distributed.all_reduce(input_, group=process_group, async_op=async_op)
    return input_, handle


class AllGatherFunc(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatenate."""

    @staticmethod
    def forward(ctx, input_: Tensor, process_group: ProcessGroup) -> Tensor:
        ctx.process_group = process_group
        output, _ = all_gather_raw(input_, process_group)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        grad_input, _ = reduce_scatter_raw(grad_output, ctx.process_group)
        return grad_input, None


# Supports autograd, but does not support async
all_gather = AllGatherFunc.apply


class ReduceScatterFunc(torch.autograd.Function):
    """Reduce scatter the input from the sequence parallel region and concatenate."""

    @staticmethod
    def forward(ctx, input_: Tensor, process_group: ProcessGroup) -> Tensor:
        ctx.process_group = process_group
        output, _ = reduce_scatter_raw(input_, process_group)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        grad_input, _ = all_gather_raw(grad_output, ctx.process_group)
        return grad_input, None


# Supports autograd, but does not support async
reduce_scatter = ReduceScatterFunc.apply


class AllReduceFunc(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatenate."""

    @staticmethod
    def forward(ctx, input_: Tensor, process_group: ProcessGroup) -> Tensor:
        ctx.process_group = process_group
        output, _ = all_reduce_raw(input_, process_group)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return grad_output, None


# Supports autograd, but does not support async
all_reduce = AllReduceFunc.apply
```

## 逐行讲解 / What's happening

1. **`all_gather_raw` (lines 19-28)**:
   - 中文: 分配一个 `world_size * shape[0]` 的输出 buffer,然后调用 `all_gather_into_tensor`(即新版的 `_all_gather_base`)把每个 rank 的本地 shard 顺序拼到一起。返回 `(output, handle)`,`handle` 给 `async_op=True` 时用——同步调用就忽略它。**关键点**:这一层不接 autograd,谁直接调用谁就拿不到反向。
   - English: Allocate an output buffer of shape `(world_size * shape[0], ...)`, then call `all_gather_into_tensor` (the modern name for `_all_gather_base`) to concatenate every rank's local shard in rank order. Returns `(output, handle)` — the handle matters only when `async_op=True`. **Crucial**: this layer is not autograd-aware; calling it directly gives you no backward.

2. **`reduce_scatter_raw` (lines 32-41)**:
   - 中文: `all_gather` 的"对偶"。输入要满足 `shape[0] % world_size == 0`,因为它会按 rank 切成 `world_size` 份,逐 chunk 做 `sum`,然后给每个 rank 返回其中一份。也就是说 `reduce_scatter` 一步把"sum + split"合并成了一个 NCCL 调用,这才是它存在的意义——分两步做会比这慢一倍。
   - English: The "dual" of `all_gather`. Input must satisfy `shape[0] % world_size == 0` because it splits the input into `world_size` chunks, sums each chunk across ranks, and hands each rank exactly one of those sums. The point of `reduce_scatter` is fusing "sum + split" into a single NCCL call — doing it in two steps is roughly 2× slower.

3. **`all_reduce_raw` (lines 45-48)**:
   - 中文: 最简单的一种——每个 rank 把自己那份加起来,所有 rank 都看到同一个总和。不改 shape,所以也不需要分配新 buffer。
   - English: The simplest one — every rank ends up with the sum across ranks. Shape doesn't change, so no new buffer needed.

4. **`AllGatherFunc.backward` (line 62) — `reduce_scatter_raw(grad_output, ...)`**:
   - 中文: 这是全篇最值得记住的一行。前向把 `[shape[0]/p]` 聚合成 `[shape[0]]`,所以反向需要把 `[shape[0]]` 的 grad 拆成 `p` 份再 sum——这恰好就是 `reduce_scatter`。直觉:"聚合的转置就是按 rank 求和后切片"。
   - English: This is the single line worth memorizing. Forward turns `[shape[0]/p]` into `[shape[0]]`; backward must turn the `[shape[0]]` gradient into `p` pieces and sum them — and that's exactly `reduce_scatter`. Intuition: "the transpose of a gather is a sum-then-split."

5. **`ReduceScatterFunc.backward` (line 81) — `all_gather_raw(grad_output, ...)`**:
   - 中文: 完全对称。前向是 sum-then-split,反向就是 gather。这也是数学上 `reduce_scatter` 和 `all_gather` 互为伴随算子的必然结果。
   - English: Perfectly symmetric. Forward = sum-then-split, backward = gather. This isn't a coincidence — `reduce_scatter` and `all_gather` are mathematical adjoints of each other.

6. **`AllReduceFunc.backward` (line 100) — `return grad_output, None`**:
   - 中文: 看起来奇怪:为什么 `all_reduce` 的反向是恒等?因为前向每个 rank 都得到了完全一样的 sum,所以当 loss 对这个 sum 的梯度沿不同路径传回来时,每个 rank 拿到的 `grad_output` 就已经是"我自己那一份对 sum 的影响"。直接原样返回即可——不要再 `all_reduce` 一次,那会把梯度乘以 `world_size`(经典 bug)。
   - English: Looks suspicious — why is `all_reduce`'s backward identity? Because forward gave every rank the same sum, so when the loss's gradient w.r.t. that sum flows backwards, each rank's `grad_output` is already "my contribution to the sum." Just return it as-is — **don't `all_reduce` again**, that would multiply the gradient by `world_size` (the canonical bug here).

7. **`all_gather = AllGatherFunc.apply` etc. (lines 67, 86, 104)**:
   - 中文: 一行别名,把 `.apply` 暴露成普通函数,用起来就跟 `torch.distributed.all_gather` 一样自然。
   - English: One-line aliases that expose `.apply` as a normal callable, so usage feels identical to `torch.distributed.all_gather`.

## 类比 / The analogy

想象一家公司有 4 个分公司,每天总部要把"今天 4 个分公司加起来卖了多少"算出来。`all_reduce` 就是"每个分公司发邮件给所有人,大家最后都看到 4 份数据并自己加起来"——总和算两次也是同一个数,所以反向"问总公司的总销售对我们分公司的贡献"时,我们分公司知道答案就是"我那一份"。`all_gather` 是"每个分公司把自己的日报抄送给所有人",反向问"总报告每页对我的贡献"时,就要把对应那一页发回去——也就是 `reduce_scatter`。整个 SP 系统就是这些"前向广播 / 反向收回"的对偶关系搭出来的。

Imagine a company with 4 regional offices. `all_reduce` is "every office emails everyone, then each office adds up the four numbers" — everyone ends up with the same total, so when you ask "what did my office contribute to the total?" the answer is just "my number." `all_gather` is "every office mails its full daily report to all the others," so when you ask "how does each page of the combined report depend on me?" you only care about your page and ship that back — exactly `reduce_scatter`. The whole sequence-parallel pipeline is built out of these forward-broadcast / backward-collect duals.

## 自己跑一遍 / Try it yourself

```python
# try_sp_autograd.py — run with 2 GPUs:
#   torchrun --nproc_per_node=2 try_sp_autograd.py
import os, torch, torch.distributed as dist

torch.distributed.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)


class AllGatherFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pg):
        ctx.pg = pg
        out = torch.empty(world * x.shape[0], *x.shape[1:], dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x.contiguous(), group=pg)
        return out

    @staticmethod
    def backward(ctx, g):
        gi = torch.empty(g.shape[0] // world, *g.shape[1:], dtype=g.dtype, device=g.device)
        dist.reduce_scatter_tensor(gi, g.contiguous(), group=ctx.pg)
        return gi, None


x = torch.full((2,), float(rank + 1), device="cuda", requires_grad=True)  # rank0=[1,1], rank1=[2,2]
y = AllGatherFunc.apply(x, dist.group.WORLD)                              # both ranks: [1,1,2,2]
loss = (y * torch.tensor([10., 20., 30., 40.], device="cuda")).sum()      # weighted sum
loss.backward()
print(f"rank {rank}: y={y.tolist()}, x.grad={x.grad.tolist()}")
```

运行 / Run with:
```bash
pip install torch
torchrun --nproc_per_node=2 try_sp_autograd.py
```

预期输出 / Expected output:
```
rank 0: y=[1.0, 1.0, 2.0, 2.0], x.grad=[10.0, 20.0]
rank 1: y=[1.0, 1.0, 2.0, 2.0], x.grad=[30.0, 40.0]
```

注意 rank 0 拿到了梯度 `[10, 20]`(对应 `y[0:2]` 的权重),rank 1 拿到了 `[30, 40]`(对应 `y[2:4]`)。这正是 `reduce_scatter` 在 backward 里把全局梯度切回各自 shard 的效果——如果你把 `backward` 改成 `return g, None`(即恒等),每个 rank 会拿到 `[10+30, 20+40] = [40, 60]`,数值会错一倍。

Note that rank 0 receives gradient `[10, 20]` (matching the weights for `y[0:2]`) and rank 1 receives `[30, 40]` (matching `y[2:4]`). That's exactly `reduce_scatter` carving the global gradient back into per-rank shards. If you replace `backward` with the identity `return g, None`, both ranks would get `[10+30, 20+40] = [40, 60]` — numerically wrong by a factor that depends on world size.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Megatron-LM 的 `_CopyToModelParallelRegion` / `_GatherFromModelParallelRegion`** / **Megatron's tensor-parallel collectives**: 同样的"前向 collective + 反向对偶 collective"配方,只不过沿 hidden-dim 切而不是 sequence-dim 切 / Same "forward collective + dual-collective backward" recipe, just sharded along the hidden dim instead of the sequence dim.
- **`torch.distributed.nn.functional`** / **`torch.distributed.nn.functional`**: PyTorch 官方现在提供了 `all_gather`, `reduce_scatter` 等带 autograd 的版本,实现思路一模一样 / PyTorch officially ships autograd-aware versions of the same primitives — internally implemented exactly the same way.
- **DeepSpeed `_AllToAll` for MoE** / **DeepSpeed's MoE `_AllToAll`**: `all_to_all` 的反向是它自己(自伴随),所以包成 autograd.Function 时 backward 还是 `all_to_all` / `all_to_all` is self-adjoint, so its autograd.Function backward is another `all_to_all`.
- **JAX 的 `pjit` + `psum` + `pmean`** / **JAX's `pjit` + `psum`/`pmean`**: JAX 用 effect system 自动求这些对偶 collective,但底下的数学完全一样 / JAX derives these duals automatically via its effect system, but the underlying math is identical.

## 注意事项 / Caveats / when it breaks

- **梯度方向反过来写就是 2× bug** / **Flipping forward/backward causes a 2× bug**: 不少新手会把 `all_reduce` 的反向也写成 `all_reduce`,结果梯度被乘了 `world_size`,loss 看起来"训得快"但模型最后崩了。原因如上文 `AllReduceFunc.backward` 注释 / A classic bug: writing `all_reduce` in the backward of `all_reduce` multiplies the gradient by `world_size`. Loss looks like it "drops fast" then training diverges.
- **`.contiguous()` 必须显式调用** / **Must explicitly `.contiguous()`**: NCCL 要求连续内存,而 PyTorch 切片后经常是 non-contiguous;`all_gather_raw` 显式 `input_.contiguous()` 是必要的防御 / NCCL requires contiguous memory; PyTorch slices are often non-contiguous, so the explicit `input_.contiguous()` inside `all_gather_raw` is mandatory defense.
- **`ProcessGroup` 不会被 ctx.save_for_backward** / **`ProcessGroup` can't go into `save_for_backward`**: 因为它不是 tensor。所以代码直接挂在 `ctx.process_group` 上,而不是 `ctx.save_for_backward(...)` / Because it's not a tensor. The code attaches it as a plain attribute `ctx.process_group` rather than saving it.
- **`backward` 返回值的个数必须与 `forward` 的输入个数一致** / **`backward` must return exactly one gradient per `forward` input**: `forward(input_, process_group)` 有 2 个输入,所以 `backward` 返回 `(grad_input, None)`——`None` 表示"不对 process_group 求梯度" / `forward` takes `(input_, process_group)`, so `backward` returns `(grad_input, None)` — `None` says "no gradient for process_group."
- **async 模式下 backward 时机要小心** / **Be careful with `async_op` in backward**: 这里反向没用 async,因为 SP 的 backward 通常需要立刻拿到结果传给前一层;如果用了 async 一定要 `.wait()` / The backwards here don't use `async_op`, because SP backward usually needs the result immediately for the upstream layer; if you do use async, you must `.wait()`.

## 延伸阅读 / Further reading

- [Megatron-LM paper §3 — "Tensor parallelism"](https://arxiv.org/abs/1909.08053) — 序列并行 / 张量并行的鼻祖论文
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198) — 序列并行的正式提出,讲清了 `all_gather` ↔ `reduce_scatter` 的内存账
- [PyTorch RFC: Functional Collectives](https://github.com/pytorch/pytorch/issues/93173) — 官方 autograd-aware collective 的设计文档
- [flash-attention `ColumnParallelLinear` source](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/fused_dense.py) — 这套 `all_gather` / `reduce_scatter` 在真正 SP MLP 里是怎么用的
