---
date: 2026-06-24
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/distributed/tensor/parallel/style.py
permalink: https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/parallel/style.py
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, pytorch, tensor-parallelism, dtensor, distributed, sharding]
---

# ColwiseParallel / RowwiseParallel：DTensor 张量并行里的 Shard(0) / Shard(1) 命名之谜 / ColwiseParallel / RowwiseParallel: The Shard(0) / Shard(1) Naming Puzzle in DTensor Tensor Parallelism

> **一句话 / In one line**: `ColwiseParallel` 把 weight 存成 `Shard(0)`，`RowwiseParallel` 存成 `Shard(1)`——听起来反的，但因为 `nn.Linear` 计算 `x @ W.T`，所以两者名称和语义完全自洽。/ `ColwiseParallel` stores weight as `Shard(0)` while `RowwiseParallel` stores it as `Shard(1)` — counterintuitive until you remember that `nn.Linear` computes `x @ W.T`, making both names exactly right.

## 为什么重要 / Why this matters

张量并行（TP）是训练千亿参数模型的核心技术：把单个矩阵乘法切开，让多个 GPU 各负责一个分片，最后 AllReduce / AllGather 拼回完整输出。PyTorch 原生 DTensor 体系让这件事只需要一个装饰器，但 `ColwiseParallel` 和 `RowwiseParallel` 的 Shard 编号让很多人看不懂。本篇把这个命名彻底讲清楚，顺便展示 DTensor 如何把通信细节完全隐藏起来。

Tensor Parallelism (TP) is the backbone of training 100B+ parameter models: split individual matrix multiplications across GPUs, each computing a shard, then AllReduce/AllGather to reconstruct the full output. PyTorch's native DTensor abstraction wraps all of this behind a single decorator — but the Shard numbering in `ColwiseParallel` vs `RowwiseParallel` trips up nearly everyone who reads it. This note untangles the naming once and for all, and shows how DTensor completely hides the communication.

## 代码 / The code

`pytorch/pytorch` — [`torch/distributed/tensor/parallel/style.py`](https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/parallel/style.py)

```python
class ColwiseParallel(ParallelStyle):
    """
    Partition a compatible nn.Module in a column-wise fashion.
    Currently supports nn.Linear and nn.Embedding.
    Users can compose it with RowwiseParallel to achieve the
    well-known Megatron-LM Column and Row Parallel Linear layers.
    """

    def _partition_linear_fn(self, name, module, device_mesh):
        # colwise shard weight/bias to Shard(0), weight be Shard(0)
        # means Colwise as Linear is input * weight^T + bias,
        # where weight would become Shard(1)
        for name, param in module.named_parameters():
            dist_param = nn.Parameter(
                distribute_tensor(param, device_mesh, [Shard(0)],
                                  tensor_type=DTensor)
            )
            module.register_parameter(name, dist_param)

    def _partition_embedding_fn(self, name, module, device_mesh):
        # Embedding: shard along the output dimension (embedding_dim)
        for name, param in module.named_parameters():
            dist_param = nn.Parameter(
                distribute_tensor(param, device_mesh, [Shard(1)],
                                  tensor_type=DTensor)
            )
            module.register_parameter(name, dist_param)

    @staticmethod
    def _prepare_input_fn(input_layouts, desired_input_layouts, mod, inputs, device_mesh):
        # ColwiseParallel: input is Replicate (each device has the full input)
        return _PrepareModuleInputFn(input_layouts, desired_input_layouts)(mod, inputs, device_mesh)

    def _apply(self, module, device_mesh):
        if isinstance(module, nn.Linear):
            return self._apply_style(
                module, device_mesh,
                col_fn=self._partition_linear_fn,
                # output will be Shard(0) — each rank holds partial output rows
                # caller may AllGather or keep sharded for the next RowwiseParallel
                output_layouts=Shard(0),
            )
        elif isinstance(module, nn.Embedding):
            return self._apply_style(
                module, device_mesh,
                col_fn=self._partition_embedding_fn,
                output_layouts=Shard(1),
            )
        raise NotImplementedError(f"ColwiseParallel not supported for {type(module)}")


class RowwiseParallel(ParallelStyle):
    """
    Partition a compatible nn.Module in a row-wise fashion.
    Currently supports nn.Linear and nn.Embedding.
    """

    def _partition_linear_fn(self, name, module, device_mesh):
        # Rowwise shard weight to Shard(1), weight be Shard(1)
        # means Rowwise as nn.Linear is input * weight^T + bias,
        # where weight would become Shard(0)
        module.register_parameter(
            "weight",
            nn.Parameter(
                distribute_tensor(module.weight, device_mesh, [Shard(1)],
                                  tensor_type=DTensor)
            ),
        )
        if module.bias is not None:
            # bias: only one device applies it, so replicate
            module.register_parameter(
                "bias",
                nn.Parameter(
                    distribute_tensor(module.bias, device_mesh, [Replicate()],
                                      tensor_type=DTensor)
                ),
            )

    def _apply(self, module, device_mesh):
        if isinstance(module, nn.Linear):
            return self._apply_style(
                module, device_mesh,
                row_fn=self._partition_linear_fn,
                # input is expected Shard(1) from prior ColwiseParallel
                # output is _Partial — DTensor triggers AllReduce automatically
                output_layouts=Replicate(),
            )
        raise NotImplementedError(f"RowwiseParallel not supported for {type(module)}")


# ---- Usage: parallelize an MLP ----
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel

parallelize_module(
    mlp,
    device_mesh,
    parallelize_plan={
        "fc1": ColwiseParallel(),   # W shape (out, in) → shard on dim 0 → each rank: (out/N, in)
        "fc2": RowwiseParallel(),   # W shape (out, in) → shard on dim 1 → each rank: (out, in/N)
    },
)
# DTensor injects the AllReduce on fc2's output automatically.
```

## 逐行讲解 / What's happening

1. **为什么 `ColwiseParallel` 用 `Shard(0)`**:
   - 中文: `nn.Linear` 的权重 W 形状是 `(out_features, in_features)`。`Shard(0)` 把行切开，每个 rank 拥有 `(out/N, in)` 的分片。`forward` 计算 `x @ W.T`，W.T 变成 `(in, out/N)`——这正是列并行：每个 rank 负责输出向量的一段列。
   - English: `nn.Linear`'s weight W has shape `(out_features, in_features)`. `Shard(0)` splits along rows, giving each rank `(out/N, in)`. The forward pass computes `x @ W.T`, making W.T become `(in, out/N)` — this is exactly column parallelism: each rank owns a slice of output columns.

2. **为什么 `RowwiseParallel` 用 `Shard(1)`**:
   - 中文: `Shard(1)` 把列切开，每个 rank 拥有 `(out, in/N)` 的分片。W.T 变成 `(in/N, out)`——这是行并行：每个 rank 处理输入向量的一段行，最后 AllReduce 求和。
   - English: `Shard(1)` splits along columns, giving each rank `(out, in/N)`. W.T becomes `(in/N, out)` — row parallelism: each rank processes a slice of input rows, then AllReduce sums the partial outputs.

3. **`distribute_tensor(param, device_mesh, [Shard(0)], tensor_type=DTensor)`**:
   - 中文: 把普通 tensor 转成 DTensor，标注其在 device_mesh 上的分布规格（placement）。这是 DTensor 的核心原语。
   - English: Converts a plain tensor into a `DTensor` annotated with its placement spec on the device mesh. This is the fundamental DTensor primitive.

4. **bias 在 RowwiseParallel 里用 `Replicate()`**:
   - 中文: bias 加法在输出维度上，每个 rank 都要完整地加，所以 replicate 即可（不需要切开）。ColwiseParallel 的 bias 随 weight 一起 `Shard(0)`——每个 rank 加自己那段列的 bias。
   - English: The bias is added across the full output dimension, so every rank needs the whole bias — `Replicate()`. In ColwiseParallel, bias shards along dim 0 alongside the weight, so each rank adds just its own slice of bias.

5. **`output_layouts=Replicate()` 在 RowwiseParallel 里触发 AllReduce**:
   - 中文: 每个 rank 做了部分矩阵乘（partial dot-product），结果是 `_Partial`。DTensor 看到目标 layout 是 `Replicate`，自动插入 AllReduce 来合并所有 rank 的部分和。这一切对用户代码透明。
   - English: Each rank computed a partial dot-product (`_Partial` placement). Seeing the desired output layout is `Replicate`, DTensor automatically inserts an AllReduce to sum all ranks' partial results. This is transparent to user code.

6. **`parallelize_module` 的 `parallelize_plan`**:
   - 中文: 字典的键是子模块的名字，值是要应用的 style。可以嵌套、可以混搭。不需要手工写 AllReduce 或 AllGather。
   - English: Dict keys are submodule names, values are the style to apply. Nesting and mixing are supported. No manual AllReduce/AllGather needed.

## 类比 / The analogy

把 W 想成一本菜单，分配给 4 个厨师：
- **列并行（Column）**：每个厨师拿到菜单的 1/4 列（负责做不同的菜），顾客点餐后 4 个厨师同时出餐，端到桌上拼成完整一桌菜——对应 ColwiseParallel，输出在 Shard(0)，上游可以 AllGather。
- **行并行（Row）**：每个厨师拿到食材的 1/4 行（负责处理不同食材），最后把大家做的半成品加在一起——对应 RowwiseParallel，输出 AllReduce 求和。

Think of W as a recipe book distributed to 4 chefs:
- **Column parallel**: each chef gets 1/4 of the output columns (different dishes). They cook in parallel and the dishes are assembled at the table — `ColwiseParallel`: outputs stay sharded, upstream can AllGather them.
- **Row parallel**: each chef handles 1/4 of the input ingredients, then everyone's partial results are summed together — `RowwiseParallel`: outputs are AllReduced.

## 自己跑一遍 / Try it yourself

```python
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel

# Run: torchrun --nproc_per_node=2 try.py
dist.init_process_group("nccl")
rank = dist.get_rank()
device = torch.device(f"cuda:{rank}")
mesh = init_device_mesh("cuda", (dist.get_world_size(),))

class MLP(nn.Module):
    def __init__(self): super().__init__(); self.fc1 = nn.Linear(16, 32); self.fc2 = nn.Linear(32, 16)
    def forward(self, x): return self.fc2(torch.relu(self.fc1(x)))

mlp = MLP().to(device)
parallelize_module(mlp, mesh, {"fc1": ColwiseParallel(), "fc2": RowwiseParallel()})

x = torch.randn(4, 16, device=device)
y = mlp(x)  # AllReduce injected automatically by DTensor
if rank == 0:
    print(f"output shape: {y.shape}, fc1.weight placement: {mlp.fc1.weight.placements}")
dist.destroy_process_group()
```

运行 / Run with:
```bash
pip install torch
torchrun --nproc_per_node=2 try.py
```

预期输出 / Expected output:
```
output shape: torch.Size([4, 16])
fc1.weight placement: (Shard(dim=0),)
```

中文：`fc1.weight.placements` 显示 `Shard(dim=0)` 而不是 `Shard(dim=1)`，但我们叫它"列并行"——因为数学上 W.T 的列才是被切开的。注意 `fc2` 的 AllReduce 完全自动，y 是完整的普通 tensor。

`fc1.weight.placements` shows `Shard(dim=0)`, not `Shard(dim=1)`, yet we call it column-parallel — because mathematically it's the columns of W.T that are sharded. Notice `fc2`'s AllReduce is fully automatic; `y` is a plain replicated tensor.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Megatron-LM `ColumnParallelLinear` / `RowParallelLinear`** / **Megatron-LM**: 上游参考实现，手动插 AllReduce，DTensor 把这层抽象掉了 / the upstream reference implementation — manual AllReduce; DTensor abstracts that away.
- **`Dao-AILab/flash-attention` `ColumnParallelLinear`**（昨天的笔记）/ **(yesterday's note)**: 同一个模式，但在 Triton kernel 层面手写通信 / the same pattern but with hand-written Triton-level communication.
- **`SequenceParallel`（同文件）** / **`SequenceParallel` (same file)**: 沿序列维度切，和 ColwiseParallel 配合用于 attention 层 / shards along the sequence dimension; pairs with ColwiseParallel for attention layers.

## 注意事项 / Caveats / when it breaks

- **`Shard(0)` 命名不等于行切分！** / **`Shard(0)` does NOT mean "shard the rows of the effective matrix"**: 是 W 的存储维度 0（行），但数学计算用的是 W.T 的维度 0（列）——方向是反的。看代码注释一定要分清"存储维度"和"数学维度"。
- **input 必须 Replicate，否则 DTensor 会报错** / **Input must be Replicate; otherwise DTensor will error**: ColwiseParallel 期望每个 rank 都有完整输入；如果上层输出也是 Shard，需要先 `PrepareModuleInput` 做 AllGather。
- **bias 在跨设备 LoRA + TP 组合时要小心** / **Bias handling gets tricky with LoRA + TP**: LoRA 的 A/B 矩阵和基础层的 TP sharding 可能冲突，需要分别配置 placement。

## 延伸阅读 / Further reading

- [PyTorch DTensor docs: Tensor Parallelism](https://pytorch.org/docs/stable/distributed.tensor.parallel.html)
- [Megatron-LM: Efficient Large-Scale Language Model Training (Shoeybi et al., 2019)](https://arxiv.org/abs/1909.08053)
- [PyTorch DTensor RFC (GitHub)](https://github.com/pytorch/pytorch/issues/88838)
