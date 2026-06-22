---
date: 2026-06-22
topic: infrastructure
source: tracked
repo: Dao-AILab/flash-attention
file: flash_attn/ops/fused_dense.py
permalink: https://github.com/Dao-AILab/flash-attention/blob/940cd9680f3315f2f06b43ab5bea2c2cf2d96806/flash_attn/ops/fused_dense.py#L166-L248
difficulty: intermediate
read_time: ~12 min
tags: [code-of-the-day, infrastructure, tensor-parallelism, distributed-training, linear-layer]
---

# Tensor 并行的两块积木：ColumnParallelLinear + RowParallelLinear / Tensor Parallelism's Two Building Blocks: ColumnParallelLinear + RowParallelLinear

> **一句话 / In one line**: 把一个 Linear 层沿"输出维度"切成 N 份（Column），再把下一个 Linear 沿"输入维度"切（Row），两者合体就是 Megatron 风格的张量并行。 / Split one Linear along its output dimension (Column) and the next along its input dimension (Row); chained together, they form the Megatron-style tensor-parallel sandwich used in every serious LLM training stack.

## 为什么重要 / Why this matters

训练 70B 以上的大模型时，单卡放不下一层的权重。Tensor Parallelism（TP）把每个线性层的权重切片到多张 GPU 上，让每张 GPU 只持有 `W[:, local_cols]` 或 `W[local_rows, :]`，然后通过集合通信把结果拼回来。flash-attention 的 `fused_dense.py` 把这个模式写成了可复用的 `nn.Linear` 子类：`ColumnParallelLinear`（切输出维）+ `RowParallelLinear`（切输入维）。它们成对出现，形成一个"全聚合 → 矩阵乘 → 减散"的三步流水，整个 forward 只需要两次集合通信（而非每个矩阵乘一次）。

When training 70B+ models, a single GPU can't hold the weights of even one layer. Tensor Parallelism (TP) shards each Linear's weight across N GPUs so each holds only `W[:, local_cols]` or `W[local_rows, :]`, then uses collective operations to reconstruct the result. Flash-attention's `fused_dense.py` packages this as drop-in `nn.Linear` subclasses: `ColumnParallelLinear` (splits output dim) + `RowParallelLinear` (splits input dim). They always appear as a pair forming a three-step pipeline — all-gather → matmul → reduce-scatter — requiring only two collectives per MLP or attention projection.

## 代码 / The code

`Dao-AILab/flash-attention` — [`flash_attn/ops/fused_dense.py`](https://github.com/Dao-AILab/flash-attention/blob/940cd9680f3315f2f06b43ab5bea2c2cf2d96806/flash_attn/ops/fused_dense.py#L166-L248)

```python
class ColumnParallelLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        process_group: ProcessGroup,
        bias: bool = True,
        sequence_parallel=True,
        multiple_of=1,
        device=None,
        dtype=None,
    ) -> None:
        world_size = torch.distributed.get_world_size(process_group)
        if out_features % multiple_of:
            raise ValueError(f"out_features ({out_features}) must be a multiple of {multiple_of}")
        multiple = out_features // multiple_of
        # We want to split @multiple across world_size, but it could be an uneven split
        div = multiple // world_size
        mod = multiple % world_size
        # The first @mod ranks get @div + 1 copies, the rest get @div copies
        local_multiple = div + int(torch.distributed.get_rank(process_group) < mod)
        super().__init__(
            in_features, local_multiple * multiple_of, bias=bias, device=device, dtype=dtype
        )
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel

    def forward(self, x):
        # If self.sequence_parallel is True, we're doing Tensor Parallel with sequence parallelism:
        # we do an all_gather of x before doing the matmul.
        # If not, then the input is already gathered.
        return fused_dense_func(
            x,
            self.weight,
            self.bias,
            process_group=self.process_group,
            sequence_parallel=self.sequence_parallel,
        )


class RowParallelLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        process_group: ProcessGroup,
        bias: bool = True,
        sequence_parallel=True,
        multiple_of=1,
        device=None,
        dtype=None,
    ) -> None:
        world_size = torch.distributed.get_world_size(process_group)
        rank = torch.distributed.get_rank(process_group)
        if in_features % multiple_of:
            raise ValueError(f"in_features ({in_features}) must be a multiple of {multiple_of}")
        multiple = in_features // multiple_of
        # We want to split @multiple across world_size, but it could be an uneven split
        div = multiple // world_size
        mod = multiple % world_size
        # The first @mod ranks get @div + 1 copies, the rest get @div copies
        local_multiple = div + int(torch.distributed.get_rank(process_group) < mod)
        # Only rank 0 will have bias
        super().__init__(
            local_multiple * multiple_of,
            out_features,
            bias=bias and rank == 0,
            device=device,
            dtype=dtype,
        )
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel

    def forward(self, x):
        """
        We're doing Tensor Parallel with sequence parallelism: we do the matmul and then
        a reduce_scatter of the result.
        """
        out = fused_dense_func(x, self.weight, self.bias)
        reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
        return reduce_fn(out, self.process_group)
```

## 逐行讲解 / What's happening

1. **`multiple = out_features // multiple_of` + `div / mod / local_multiple`（ColumnParallelLinear.__init__）**:
   - 中文: 这三行解决"world_size 不能整除 out_features"的问题。先把 `out_features` 除以 `multiple_of`（对齐到 head_dim 或其他倍数），再做带余数除法。前 `mod` 个 rank 各多分一份（`div+1`），剩下的分 `div` 份。公式 `div + int(rank < mod)` 是一个对 rank 单调不增的分配方案，数学上正确且无 if-else。
   - English: These three lines handle uneven splits when `world_size` doesn't divide `out_features`. After normalizing by `multiple_of`, integer-divide: the first `mod` ranks each get `div+1` slices, the rest get `div`. The formula `div + int(rank < mod)` is a branchless monotone assignment — mathematically clean, no conditionals.

2. **`super().__init__(in_features, local_multiple * multiple_of, ...)`（ColumnParallelLinear）**:
   - 中文: 直接继承 `nn.Linear` 并把 `out_features` 设成本 rank 的局部份额。GPU 的权重张量形状是 `(local_out, in_features)`，不是完整的 `(out_features, in_features)`。
   - English: Inherits `nn.Linear` with the local output size. The GPU's weight tensor is `(local_out, in_features)`, not the full `(out_features, in_features)` — PyTorch handles all other bookkeeping automatically.

3. **`fused_dense_func(x, self.weight, self.bias, process_group=..., sequence_parallel=True)`（ColumnParallelLinear.forward）**:
   - 中文: `fused_dense_func` 内部会先 all-gather 输入 x（因为 sequence_parallel=True 时，每张 GPU 只有序列的一个分片），然后做局部矩阵乘，输出形状 `(B, S, local_out)`。不需要再做 reduce，因为各 rank 的输出拼起来就是完整结果。
   - English: When `sequence_parallel=True`, `fused_dense_func` all-gathers `x` first (each GPU holds only a sequence shard), then performs the local matmul. Output is `(B, S, local_out)` per GPU — no reduce needed because the per-rank outputs concatenate to form the full output.

4. **`bias=bias and rank == 0`（RowParallelLinear.__init__）**:
   - 中文: 偏置只分配给 rank 0。因为 RowParallelLinear 的最后一步是 all-reduce（把各 rank 的局部输出加起来），如果每个 rank 都有偏置，最终结果会把偏置加了 world_size 次。只让 rank 0 持有 bias，all-reduce 后偏置自然只加了一次。
   - English: Bias lives only on rank 0. The final step in `RowParallelLinear.forward` is `all_reduce` (summing partial matmul outputs across ranks). If every rank had a bias, the reduce would accumulate it `world_size` times. Assigning it only to rank 0 means the bias appears exactly once after the reduce.

5. **`reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce`（RowParallelLinear.forward）**:
   - 中文: 当 sequence_parallel=True，用 reduce-scatter（每个 rank 得到汇总结果的一个序列分片）而不是 all-reduce（所有 rank 都得到完整结果）。reduce-scatter 的通信量和 all-reduce 相同，但输出已经是"切好的"状态，可以直接送入下一个 ColumnParallelLinear 的 all-gather，形成 ring 结构。
   - English: With `sequence_parallel=True`, `reduce_scatter` scatters the summed output so each rank holds only its sequence shard — bandwidth identical to all-reduce but the output is already sharded for the next layer's `ColumnParallelLinear` all-gather, forming a ring that costs the same total bytes as one all-reduce per layer pair.

## 类比 / The analogy

想象一家工厂把一条长生产线（矩阵乘法）拆成两段：第一段（Column）有 N 台机器并排，每台加工输入物料的全部但只生产自己那份产品；第二段（Row）有 N 台机器，每台接收上游所有机器送来的一小部分原料、各自加工，最后把所有机器的结果混合。关键是：第一段生产完毕后不需要汇总，第二段混合时只需要一次集合。两段合起来只开两次"跨厂运输"，就完成了一整条完整的生产线。

Think of a factory that breaks one long assembly line (matrix multiply) into two stages. Stage 1 (Column): N machines run in parallel — each gets all the raw material but produces only its share of the product. Stage 2 (Row): N machines each receive a small portion from every Stage 1 machine, process it, then pool all results. The key is that Stage 1 needs no inter-machine sync, and Stage 2 pools results exactly once. The two-stage factory runs two inter-plant shipments total — just as efficient as one single machine, but now N workers run simultaneously.

## 自己跑一遍 / Try it yourself

```python
import os, torch, torch.distributed as dist

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29500")
dist.init_process_group("gloo", rank=0, world_size=1)

in_f, out_f = 64, 128
pg = dist.new_group([0])

col = torch.nn.Linear(in_f, out_f // 1)  # rank 0 owns all outputs (world_size=1)
row = torch.nn.Linear(out_f, 32)

x = torch.randn(2, 10, in_f)
h = col(x)        # (2, 10, 128)
y = row(h)        # (2, 10, 32)
print("col weight:", col.weight.shape)  # torch.Size([128, 64])
print("row weight:", row.weight.shape)  # torch.Size([32, 128])
print("output:    ", y.shape)           # torch.Size([2, 10, 32])

dist.destroy_process_group()
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
col weight: torch.Size([128, 64])
row weight: torch.Size([32, 128])
output:     torch.Size([2, 10, 32])
```

注意 world_size=1 时退化为普通 Linear，形状不变。真正跑多卡时，每个 rank 的 `col.weight` 只有 `(128/N, 64)` 行——这就是 TP 的核心：N 张卡各持有 1/N 的权重，推理时没有一张卡需要完整权重。

With `world_size=1` this degrades to a plain `nn.Linear`. On N GPUs, each rank's `col.weight` would be `(128/N, 64)` — the key insight of TP: no single GPU ever needs to hold the full weight matrix.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Megatron-LM** (`megatron/core/tensor_parallel/layers.py`): 原始来源，flash-attention 的实现是直接参考 Megatron 的设计，但写成了更轻量的纯 PyTorch 子类。 / The original source; flash-attention's implementation is a lighter pure-PyTorch rewrite of Megatron's design.
- **DeepSeek-V3** (`model.py`): MoE 的每个专家 FFN 正是 ColumnParallel → RowParallel 的结构，只是加了专家路由（先 top-K 派送，再 Column/Row 计算）。 / Each expert FFN in DeepSeek's MoE is exactly this Column→Row pattern, wrapped in a top-K expert router.
- **vLLM 的 `linear.py`**: vLLM 也有自己的 `ColumnParallelLinear` / `RowParallelLinear`，逻辑相同，但加了 quantization hook。 / vLLM's `linear.py` has the same classes with an added quantization hook for GPTQ/AWQ/FP8.
- **PyTorch `torch.distributed.tensor`**: `DTensor` + `distribute_module` 是 PyTorch 原生的 TP 方案，用声明式 placement 描述同样的 Column/Row 切分，无需手写 all-gather/reduce-scatter。 / PyTorch's `DTensor` + `distribute_module` is the native TP API expressing the same Column/Row sharding declaratively without hand-written collectives.

## 注意事项 / Caveats / when it breaks

- **`multiple_of` 必须整除 `out_features` / `multiple_of` must divide `out_features`**: TP 切分通常以 head_dim 为单位（确保 attention head 不被割裂），所以传入 `multiple_of=head_dim`。忘掉这个会导致 `ValueError`。 / TP sharding is typically head-dim aligned (to keep attention heads intact), so pass `multiple_of=head_dim`. Omitting this causes a `ValueError` at init time.
- **bias 只在 rank 0 / Bias only on rank 0**: `RowParallelLinear` 的 bias 只活在 rank 0 上。如果你用 `model.state_dict()` 合并权重再 split，要注意 bias 的 shard 规则与 weight 不同。 / The bias lives only on rank 0. When merging/splitting state dicts, bias sharding rules differ from weight sharding.
- **sequence_parallel=False 时用 all_reduce，通信量翻倍 / `sequence_parallel=False` uses all_reduce, doubling communication**: 如果你不用 sequence parallelism，把 `sequence_parallel=False`，forward 里的 `reduce_scatter` 会换成 `all_reduce`，每张卡都持有完整激活，占用更多显存。 / Without sequence parallelism, set `sequence_parallel=False`; the `reduce_scatter` becomes `all_reduce`, giving every GPU the full activation at higher memory cost.

## 延伸阅读 / Further reading

- [Megatron-LM 论文 (Shoeybi et al. 2019)](https://arxiv.org/abs/1909.08053) — Tensor Parallelism 的原始设计，Column/Row 这对命名就来自这篇论文。
- [flash-attention `fused_dense.py` 完整文件](https://github.com/Dao-AILab/flash-attention/blob/940cd9680f3315f2f06b43ab5bea2c2cf2d96806/flash_attn/ops/fused_dense.py) — 上面还有 `FusedMLP` 和 `ParallelFusedMLP`，把 Column+Row 组合成一个模块。
- [PyTorch DTensor 官方教程](https://pytorch.org/tutorials/intermediate/TP_tutorial.html) — 用 `DTensor` 声明式写 TP，比手写 all-gather / reduce-scatter 更现代。
