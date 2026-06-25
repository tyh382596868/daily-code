---
date: 2026-06-25
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/distributed/pipelining/microbatch.py
permalink: https://github.com/pytorch/pytorch/blob/3e3ec43fcf3b5ee3bfc39ece6ea47219463b1f42/torch/distributed/pipelining/microbatch.py#L240-L351
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, pipeline-parallelism, distributed, pytree, microbatch]
---

# Pipeline 并行的微批次拆分：pytree-aware 的 split_args_kwargs_into_chunks / Pipeline-Parallel Microbatch Splitting: pytree-aware split_args_kwargs_into_chunks

> **一句话 / In one line**: `torch.distributed.pipelining` 通过 pytree flatten/unflatten 对任意嵌套的 args/kwargs 做 chunk 拆分，用户无需手动切片批次就能用上流水线并行。 / `torch.distributed.pipelining` uses pytree flatten/unflatten to chunk arbitrarily-nested args/kwargs, letting users write pipeline-parallel code without manually slicing batches.

## 为什么重要 / Why this matters

流水线并行的核心操作是把一个大批次切成多个微批次（microbatch），依次送进流水线的各个 stage。难点在于：用户的 `forward` 函数接受各种各样的参数——可能是 `(input_ids, attention_mask)`，也可能是 `({"images": ..., "text": ...}, labels)`，嵌套深度不一，有些参数需要按 batch 维度切片，有些（如 `temperature`）需要在所有微批次中复制。

`split_args_kwargs_into_chunks` 通过 `TensorChunkSpec`（指定沿哪个维度切）和 `_Replicate`（在所有 chunk 中复制）两个规格对象，配合 `torch.utils._pytree.tree_map`，把所有的拆分逻辑统一成一套算法：先 flatten 嵌套结构、对每个叶子节点按规格拆分或复制，再 rotate 维度顺序（让 chunk 成为外层维度），最后 unflatten 还原结构。调用方只需要传一个大批次，pipeline stage 自动看到正确尺寸的微批次。

Pipeline parallelism requires splitting one big batch into microbatches and feeding them through stages in sequence. The challenge: user `forward` functions take arbitrary argument shapes — `(input_ids, mask)`, `({"images": ..., "text": ...}, labels)`, nested lists, etc. Some arguments should be sliced along the batch dim; others (like `temperature` scalars) should be replicated in every chunk. `split_args_kwargs_into_chunks` solves this uniformly with pytree traversal: flatten the nested arg structure, apply per-leaf `TensorChunkSpec` (shard) or `_Replicate` (broadcast) policies, rotate so chunks become the outer dimension, then unflatten to reconstruct the original nested shape. The pipeline framework calls this once; user code never needs to write a custom slicer.

## 代码 / The code

`pytorch/pytorch` — [`torch/distributed/pipelining/microbatch.py`](https://github.com/pytorch/pytorch/blob/3e3ec43fcf3b5ee3bfc39ece6ea47219463b1f42/torch/distributed/pipelining/microbatch.py#L240-L351)

```python
def split_args_kwargs_into_chunks(
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None,
    chunks: int,
    args_chunk_spec: tuple[TensorChunkSpec, ...] | None = None,
    kwargs_chunk_spec: dict[str, TensorChunkSpec] | None = None,
) -> tuple[list[tuple], list[dict]]:
    """
    Given a sequence of args and kwargs, split them into a number of chunks
    according to their respective chunking specs.

    # Steps (from the inline comment):
    # 1. pytree.tree_flatten each arg and its spec into a 1d array of values.
    #    args = ([A, [B, C]], D)  →  flat_args = ([A, B, C], D)
    # 2. Shard or replicate per leaf policy (chunks=2):
    #    flat_args = ([[A,A], [B,B], [C_1,C_2]], [D,D])
    # 3. Rotate: chunks become outer dimension:
    #    args_chunks = [([A, B, C_1], D), ([A, B, C_2], D)]
    # 4. Unflatten each chunk back to original nested shape:
    #    args_chunks = [([A, [B, C_1]], D), ([A, [B, C_2]], D)]
    """
    if kwargs is None:
        kwargs = {}

    # Default spec: Tensors (and BlockMasks) are chunked on dim 0; everything
    # else is replicated in every microbatch.
    def default_spec(v):
        if isinstance(v, torch.Tensor | BlockMask):
            return TensorChunkSpec(DEFAULT_CHUNK_DIM)
        else:
            return _Replicate()

    if args_chunk_spec is None:
        args_chunk_spec = tree_map(
            default_spec, args, is_leaf=lambda v: isinstance(v, BlockMask)
        )
    if kwargs_chunk_spec is None:
        kwargs_chunk_spec = tree_map(
            default_spec, kwargs, is_leaf=lambda v: isinstance(v, BlockMask)
        )

    # Delegate the actual shard/replicate work to _shard_dict_of_args,
    # treating positional args as a dict keyed by index.
    args_split_dict = _shard_dict_of_args(
        dict(enumerate(args)),
        dict(enumerate(args_chunk_spec)),
        chunks,
    )
    real_num_chunks = len(args_split_dict)

    kwargs_split = _shard_dict_of_args(
        kwargs,
        kwargs_chunk_spec,
        real_num_chunks,
    )

    # Edge case: kwargs may yield fewer chunks than args (e.g. when args has
    # no tensors and kwargs drives the actual chunk count).
    if len(kwargs_split) < real_num_chunks:
        real_num_chunks = len(kwargs_split)
        args_split_dict = _shard_dict_of_args(
            dict(enumerate(args)),
            dict(enumerate(args_chunk_spec)),
            real_num_chunks,
        )

    if len(args_split_dict) != len(kwargs_split):
        raise RuntimeError(
            "args and kwargs are split into different number of chunks: "
            f"{len(args_split_dict)}, {len(kwargs_split)}"
        )

    # Convert back from dict-of-index to tuple
    args_split = [
        tuple(chunk_args[i] for i in range(len(chunk_args)))
        for chunk_args in args_split_dict
    ]

    return args_split, kwargs_split
```

## 逐行讲解 / What's happening

1. **`default_spec` 函数（L298-304）：推断每个参数叶子节点的拆分策略 / Inferring split policy per leaf**
   - 中文：如果叶子是 `torch.Tensor` 或 `BlockMask`（FlexAttention 的遮罩对象），就默认沿 `DEFAULT_CHUNK_DIM=0`（batch 维度）切片；否则（整数、字符串、Python 对象等）复制到每个 chunk。这个 `_Replicate` 哨兵对象会在 `_shard_dict_of_args` 内被识别，对相应值做广播而不是切片。
   - English: If a leaf is a `Tensor` or `BlockMask`, shard it along dim 0 (the batch dimension). Everything else gets a `_Replicate` sentinel, which tells `_shard_dict_of_args` to broadcast the value into every microbatch unchanged. This handles scalars, strings, nested config dicts — all the non-tensor arguments users routinely pass.

2. **`tree_map(default_spec, args, ...)` 与 `args_chunk_spec`（L306-314）：规格树和参数树同构 / Spec tree mirrors arg tree**
   - 中文：`tree_map` 遍历 `args` 的完整 pytree 结构，对每个叶子调用 `default_spec`，生成一棵和 `args` 形状完全一致的"规格树"。`is_leaf` 参数用于告知 `tree_map` 把 `BlockMask` 当叶子处理（而不是递归进它的内部）。如果用户已经提供了 `args_chunk_spec`，这步跳过——用户可以对特定参数指定非零维度的切分策略，比如 `TensorChunkSpec(split_dim=1)` 来按序列维度切。
   - English: `tree_map` walks the pytree structure of `args`, calls `default_spec` on every leaf, and returns a spec tree with the same nested shape. The `is_leaf` hook prevents `tree_map` from recursing inside `BlockMask` objects. If the user supplies `args_chunk_spec` directly, this step is skipped — they can specify per-argument split dimensions (e.g. `TensorChunkSpec(split_dim=1)` to chunk along the sequence axis instead of the batch axis).

3. **`_shard_dict_of_args`（L316-327）：实际的 flatten → shard/replicate → rotate**
   - 中文：这个内部函数（代码不在本范围内，但逻辑是核心的）实现了注释里描述的四步骤。它先把 args 和 spec 各自 pytree-flatten 为一维列表，对每个叶子按规格切分（`torch.chunk` 或 list-replicate），然后旋转维度（从"叶子 × chunk"变成"chunk × 叶子"），最后 unflatten 还原嵌套形状。返回值是 `list[dict]`，每个 dict 对应一个微批次的（index→value）映射。
   - English: This helper (defined elsewhere in the file) does the four-step transformation described in the docstring: pytree-flatten args and their spec into flat lists, shard or replicate each leaf, rotate dimensions so chunks are outer, then unflatten back to the original nested shape. It returns a `list[dict]` where each dict maps argument index to the microbatch value for that chunk.

4. **`real_num_chunks` 修正（L329-338）：处理 kwargs 驱动 chunk 数量的边界情况 / Edge case: kwargs drives chunk count**
   - 中文：当 `args` 全是标量（没有 Tensor）时，`_shard_dict_of_args(args, ...)` 可能会返回用户指定的 `chunks` 个复制体，但实际的批次大小由 `kwargs` 里的 Tensor 决定。这个修正让两侧的 chunk 数量协商一致。
   - English: When all positional args are scalars (no tensors), `_shard_dict_of_args` might yield the requested `chunks` replicas rather than the actual batch size. The reconciliation step re-shards args using the chunk count inferred from kwargs tensors, ensuring both sides agree.

## 类比 / The analogy

想象你在装配流水线上打包快递。一个大包裹（batch）需要被拆成 N 个小包裹（microbatch）分批次处理。你有各种物品：有些是分批的（可切割的 Tensor，比如一打鸡蛋 → 每份3个），有些是公共的（不可切割的配件，比如一张操作说明书 → 每份都要放一份）。`TensorChunkSpec` 就是"这件物品按某维度切分"的标签，`_Replicate` 是"这件物品每份都要"的标签。`tree_map` 负责在整个包裹树里给每件物品贴标签，`_shard_dict_of_args` 负责真正执行切割和复制。最终你得到 N 个结构完全一样的小包裹，可以送进流水线的 N 个 stage。

Think of it as sorting a large shipment into microbatch packages on an assembly line. The big batch is a mixed box containing sliceable items (tensors — e.g. a dozen eggs, split 3 per package) and non-sliceable items (scalars — e.g. an instruction sheet, one copy per package). `TensorChunkSpec` is the "split-along-dim" label; `_Replicate` is the "put-one-in-every-package" label. `tree_map` walks the whole nested box structure and labels every item. `_shard_dict_of_args` does the physical splitting and copying. The result is N identically shaped small packages, each self-contained and ready for a pipeline stage.

## 自己跑一遍 / Try it yourself

```python
import torch
from torch.distributed.pipelining.microbatch import (
    TensorChunkSpec, split_args_kwargs_into_chunks
)

# Simulate: args = (input_ids, labels), kwargs = {"temperature": 1.0}
# input_ids and labels are tensors to be chunked; temperature is a scalar to replicate
input_ids = torch.arange(16).view(4, 4)  # batch=4, seq=4
labels    = torch.arange(4)               # batch=4
args  = (input_ids, labels)
kwargs = {"temperature": 1.0}

args_chunks, kwargs_chunks = split_args_kwargs_into_chunks(args, kwargs, chunks=2)

print(f"Number of microbatches: {len(args_chunks)}")
for i, (a, kw) in enumerate(zip(args_chunks, kwargs_chunks)):
    print(f"  chunk {i}: input_ids shape={a[0].shape}, labels={a[1]}, temp={kw['temperature']}")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
Number of microbatches: 2
  chunk 0: input_ids shape=torch.Size([2, 4]), labels=tensor([0, 1]), temp=1.0
  chunk 1: input_ids shape=torch.Size([2, 4]), labels=tensor([2, 3]), temp=1.0
```

注意 `temperature=1.0` 在两个微批次中都出现了（复制），而 `input_ids` 和 `labels` 按 batch 维度各取了一半（切分）。

`temperature` appears in both chunks (replicated) while `input_ids` and `labels` are each halved along the batch dimension. That's the `_Replicate` vs `TensorChunkSpec` policy difference in action.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **`torch.distributed.pipelining.PipelineStage`** / **PipelineStage**: 调用 `split_args_kwargs_into_chunks` 后把每个 chunk 依次发给下游 stage，配合 `merge_chunks`（`L354`）在最后一个 stage 重新聚合输出。
- **`torchgpipe` / `fairscale.nn.Pipe`** / **torchgpipe / fairscale Pipe**: 早期的 PyTorch pipeline 实现使用了类似的 chunk 概念，但没有 pytree-aware 的 spec 机制，需要用户手动切分 tensor。
- **JAX `pjit` / `jax.vmap` over microbatches** / **JAX pjit / vmap over microbatches**: JAX 通过 `in_shardings` 指定每个参数的分片策略，在理念上与 `TensorChunkSpec` 非常相似——也是"参数树 + 规格树同构"。

## 注意事项 / Caveats / when it breaks

- **`BlockMask` 是特殊的叶子节点** / **BlockMask is a special leaf**: `is_leaf=lambda v: isinstance(v, BlockMask)` 防止 `tree_map` 拆解 BlockMask 的内部结构（它是一个命名元组），否则每个字段会被单独当叶子处理，产生错误的规格。
- **`real_num_chunks` 可能小于 `chunks`** / **`real_num_chunks` can be less than `chunks`**: 当最小的 Tensor batch 维度 < 请求的 `chunks` 数时，PyTorch 会按实际可切分的块数返回。不要假设 `len(args_chunks) == chunks`。
- **不支持非 tensor 参数的切片** / **No per-leaf dim for non-tensor args**: `_Replicate` 只有广播语义，没有"按某个非 Tensor 属性分配到不同 stage"的能力。如果你需要不同 stage 用不同的配置，需要在外部手动构造 `kwargs_chunk_spec`。

## 延伸阅读 / Further reading

- [`torch.distributed.pipelining` 官方教程](https://pytorch.org/docs/stable/distributed.pipelining.html)
- [GPipe 原始论文（pipeline parallelism 的基础）](https://arxiv.org/abs/1811.06965)
- [`TensorChunkSpec` 源码定义（同文件 L1-120）](https://github.com/pytorch/pytorch/blob/3e3ec43fcf3b5ee3bfc39ece6ea47219463b1f42/torch/distributed/pipelining/microbatch.py#L1-L120)
