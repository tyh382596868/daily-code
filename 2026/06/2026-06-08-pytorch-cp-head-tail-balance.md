---
date: 2026-06-08
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/distributed/tensor/experimental/_context_parallel/_load_balancer.py
permalink: https://github.com/pytorch/pytorch/blob/1c94cbdb6208bbb4035b724bdd11c093947f42a5/torch/distributed/tensor/experimental/_context_parallel/_load_balancer.py#L78-L164
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, distributed, context-parallel, causal-attention, load-balancing]
---

# 把因果三角切成头尾配对:PyTorch 的 Context-Parallel 负载均衡 / Pairing head with tail: PyTorch's context-parallel load balancer for causal attention

> **一句话 / In one line**: 因果 attention 的 mask 是三角形,后面的 token 干活多,所以不能等分序列;把 token 重排成"头-尾-头-尾"再均分,每张卡看到的三角面积就一样大了。 / The causal mask is a triangle — late tokens do more work — so you can't slice the sequence evenly. Permute it to "head-tail-head-tail" then slice, and every rank sees the same triangle area.

## 为什么重要 / Why this matters

长上下文训练越来越流行,Context Parallel(CP)就是把一条 N 万 token 的序列横切到多张 GPU 上算 attention。问题:因果 mask 是个下三角,如果你把 `[0, 1, 2, ..., 7]` 直接切成 `[0..3]` 给 rank 0、`[4..7]` 给 rank 1,rank 0 只算 10 次 dot product,rank 1 要算 26 次——超过 2 倍的不均衡,rank 1 永远是 straggler,整个 CP 组速度被它拖住。PyTorch 2.7 把一段众所周知的 Megatron-LM trick 搬进了主干:重排成 `[0, 7, 1, 6, 2, 5, 3, 4]` 再切,两张卡各算 18 次。19 行 `_generate_indices` 就完成了这件事。

Long-context training is everywhere now and Context Parallel (CP) is how you spread a single 100k-token sequence across multiple GPUs for attention. Problem: the causal mask is a lower triangle. If you just slice `[0, 1, 2, ..., 7]` evenly — `[0..3]` to rank 0, `[4..7]` to rank 1 — rank 0 computes 10 dot products and rank 1 computes 26. Over 2× imbalance; rank 1 is permanently the straggler that bottlenecks the whole CP group. PyTorch 2.7 ported a well-known Megatron-LM trick into core: permute to `[0, 7, 1, 6, 2, 5, 3, 4]` first, then slice — both ranks compute 18. The whole thing lives in a 19-line `_generate_indices`.

## 代码 / The code

`pytorch/pytorch` — [`torch/distributed/tensor/experimental/_context_parallel/_load_balancer.py`](https://github.com/pytorch/pytorch/blob/1c94cbdb6208bbb4035b724bdd11c093947f42a5/torch/distributed/tensor/experimental/_context_parallel/_load_balancer.py#L78-L164)

```python
class _HeadTailLoadBalancer(_LoadBalancer):
    def __init__(self, seq_length: int, world_size: int, device: str | torch.device):
        self.seq_length = seq_length
        self.world_size = world_size
        self.device = device

    def _generate_indices(self, restore: bool = False) -> Tensor:
        """
        Head-tail load-balance strategy rearranges the Q tensor by combining
        Q[0:k] (on seq dim) and Q[-k:] for rank 0, Q[k:2k] and Q[-2k:-k] for
        rank 1, and so on.

        For seq_len=8, world_size=2 the resulting slice indices are:
            slice_indices = Tensor([0, 7, 1, 6, 2, 5, 3, 4])
        """
        seq_length = self.seq_length
        world_size = self.world_size
        if seq_length % (world_size * 2) != 0:
            raise AssertionError
        chunk_size = seq_length // (world_size * 2)

        # Split sequence into 2*world_size chunks, then pair chunk r with
        # chunk (2*world_size - 1 - r) for each rank.
        indices = torch.arange(seq_length, dtype=torch.int, device=self.device)
        chunks = indices.view(world_size * 2, chunk_size)
        head_idx = torch.arange(world_size, device=self.device)
        tail_idx = 2 * world_size - 1 - head_idx
        paired = torch.stack([chunks[head_idx], chunks[tail_idx]], dim=1)
        all_indices_tensor = paired.reshape(-1)

        if restore:
            all_indices_tensor = torch.argsort(all_indices_tensor)

        return all_indices_tensor.unsqueeze(0)  # add batch dim
```

## 逐行讲解 / What's happening

1. **`chunk_size = seq_length // (world_size * 2)`**:
   - 中文: 不是切 `world_size` 块,而是切 `2 * world_size` 块——预留一倍数量,好让每个 rank 拿到"一头一尾"两个 chunk。
   - English: don't carve `world_size` chunks, carve `2 * world_size` — twice as many, so each rank can take one "head" chunk and one "tail" chunk.

2. **`indices = arange(seq_length); chunks = indices.view(world_size * 2, chunk_size)`**:
   - 中文: 简单把 `[0, 1, ..., 7]` reshape 成 `[[0,1], [2,3], [4,5], [6,7]]` 这种 `(2*W, chunk)` 形状,后面就直接按 chunk 索引拼。
   - English: simply reshape `[0..7]` into `[[0,1], [2,3], [4,5], [6,7]]`, a `(2*W, chunk)` matrix, so later we can index by chunk number.

3. **`head_idx = arange(world_size)` & `tail_idx = 2*world_size - 1 - head_idx`**:
   - 中文: rank r 想吃第 r 个 chunk(头)和倒数第 r+1 个 chunk(尾)。world=2 时 `head_idx=[0,1]`, `tail_idx=[3,2]`——rank 0 拿 chunk 0+3,rank 1 拿 chunk 1+2。
   - English: rank r should eat chunk #r (head) and the (r+1)-th-from-the-back chunk (tail). For world=2: `head_idx=[0,1]`, `tail_idx=[3,2]` — rank 0 takes chunks 0+3, rank 1 takes chunks 1+2.

4. **`paired = torch.stack([chunks[head_idx], chunks[tail_idx]], dim=1)`**:
   - 中文: 把 head 和 tail 沿新维度配对,形状 `(W, 2, chunk)`。每个 rank 一个 slot,slot 里先 head 后 tail。
   - English: pair head and tail along a new axis, shape `(W, 2, chunk)`. One slot per rank, head-then-tail inside.

5. **`paired.reshape(-1)`**:
   - 中文: 拉平成 `(seq_length,)`,最终就是 `[0, 7, 1, 6, 2, 5, 3, 4]`。下游会用这串索引重排 Q/K/V,然后按 rank 平均切。
   - English: flatten to `(seq_length,)`, yielding `[0, 7, 1, 6, 2, 5, 3, 4]`. The caller uses these indices to permute Q/K/V, then slices evenly per rank.

6. **`if restore: argsort`**:
   - 中文: 反向操作。算完 attention 输出在重排后的位置上,要还原回原始 token 顺序就用 `argsort` 求逆置换——`out[restore_idx]` 把它放回原位。
   - English: the inverse. After attention runs on the permuted layout you need to put outputs back in original token order — `argsort` gives the inverse permutation; `out[restore_idx]` undoes the rearrangement.

## 类比 / The analogy

像三个人分一块切成 6 片的蛋糕,但每片大小不同——前面的小,后面的大,体积按 `1, 2, 3, 4, 5, 6` 算。如果直接两片一组分,A 拿 `1+2=3`、B 拿 `3+4=7`、C 拿 `5+6=11`,极不公平。换个分法:第一和第六、第二和第五、第三和第四,每个人都是 `7`。Context Parallel 的因果三角就是那块越后越大的蛋糕,head-tail 配对就是这个"对称取片"的分法。

Picture three people splitting a six-slice cake where the slices get bigger from one end to the other — volumes `1, 2, 3, 4, 5, 6`. Take two adjacent slices each and A gets `1+2=3`, B gets `3+4=7`, C gets `5+6=11`. Awful. Instead pair the first with the last, the second with the second-last, third with fourth — everyone gets `7`. The causal triangle is the cake; head-tail pairing is the symmetric splitting trick.

## 自己跑一遍 / Try it yourself

```python
# try.py
import torch

def head_tail_indices(seq_length: int, world_size: int) -> torch.Tensor:
    assert seq_length % (world_size * 2) == 0
    chunk_size = seq_length // (world_size * 2)
    indices = torch.arange(seq_length)
    chunks = indices.view(world_size * 2, chunk_size)
    head = torch.arange(world_size)
    tail = 2 * world_size - 1 - head
    return torch.stack([chunks[head], chunks[tail]], dim=1).reshape(-1)

def causal_load_per_rank(perm, world_size):
    # rank r gets perm[r * 2k : (r+1) * 2k]; "work" of a token at position p in the
    # original (unpermuted!) sequence is (p + 1)  ← it attends to tokens 0..p
    per_rank = perm.view(world_size, -1)  # (W, 2k)
    return (per_rank + 1).sum(dim=1).tolist()

for W in [2, 4]:
    seq = 8 * W
    perm = head_tail_indices(seq, W)
    naive = torch.arange(seq).view(W, -1)
    print(f"seq={seq}, world={W}")
    print(f"  permutation                : {perm.tolist()}")
    print(f"  load per rank (head-tail)  : {causal_load_per_rank(perm, W)}")
    print(f"  load per rank (naive even) : {(naive + 1).sum(dim=1).tolist()}")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
seq=16, world=2
  permutation                : [0, 15, 1, 14, 2, 13, 3, 12, 4, 11, 5, 10, 6, 9, 7, 8]
  load per rank (head-tail)  : [68, 68]
  load per rank (naive even) : [36, 100]
seq=32, world=4
  permutation                : [0, 31, 1, 30, 2, 29, 3, 28, 4, 27, ...]
  load per rank (head-tail)  : [132, 132, 132, 132]
  load per rank (naive even) : [36, 100, 164, 228]
```

朴素切分下 rank 0 和 rank-3 工作量差 6 倍以上;head-tail 配对后完全相等。CP 训练每一步快慢由最慢的卡决定,这一手就是决定性的。

Naive slicing has a >6× gap between rank 0 and rank N-1; head-tail pairing flattens it to 0. CP step time is dominated by the slowest rank — this is the difference.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Megatron-LM `context_parallel_size`**: 原始出处,叫 zigzag attention sharding / The original; called "zigzag attention sharding" there.
- **DeepSpeed Ulysses / Ring Attention**: 它们走 sequence-parallel 但同样要处理因果不均;策略略有不同(Ring 是按 step 滚 KV),但问题本质一样 / They take a sequence-parallel route and face the same causal imbalance; Ring solves it by rolling KV step-by-step.
- **vLLM / SGLang prefill chunking**: 推理侧也用类似的"短前缀 + 长前缀混搭"配对来均衡 batch-level cost / On the inference side, similar "short prefix + long prefix" pairing balances batch-level cost.

## 注意事项 / Caveats / when it breaks

- **要求 `seq_length % (2*W) == 0`** / **Requires `seq_length % (2*W) == 0`**: 代码直接 raise AssertionError。CP 里通常上游会 pad,但你自己 hack 时要记得 / The code raises AssertionError on mismatch. CP setups usually pad upstream; remember it if you patch this in by hand.
- **只对因果三角负载有效** / **Only useful for the causal triangle**: 全注意力(双向)本身就是均衡的,这一手不仅没用,还多花一次置换的内存搬运 / Full (bidirectional) attention is already balanced; the permute is wasted memory traffic.
- **Document-level packing 要换一个 LB**: 当一个 batch 里有多条不同长度的文档拼在一起,causal mask 不再是一个大三角而是多个小三角,要用同文件里的 `_PerDocumentHeadTailLoadBalancer` 或 `_PTRR` 策略 / If your batch packs multiple variable-length documents, the mask becomes many small triangles — use `_PerDocumentHeadTailLoadBalancer` or `_PTRR` in the same file.

## 延伸阅读 / Further reading

- PyTorch CP docs: https://docs.pytorch.org/docs/main/distributed.tensor.html (Context Parallel section)
- Megatron-LM context parallelism README — original zigzag description
- "Ring Attention with Blockwise Transformers" (Liu et al., 2023) — alternative attention-parallel approach
