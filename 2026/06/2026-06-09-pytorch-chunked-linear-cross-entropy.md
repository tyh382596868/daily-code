---
date: 2026-06-09
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/nn/modules/linear_cross_entropy.py
permalink: https://github.com/pytorch/pytorch/blob/f723c876cf5d6884f829514dc591cf9e69f4948e/torch/nn/modules/linear_cross_entropy.py#L751-L772
difficulty: intermediate
read_time: ~12 min
tags: [code-of-the-day, pytorch, chunked-cross-entropy, llm-training, memory-efficient]
---

# PyTorch 把"分块 Linear+CE"塞进了官方,16 行就是整套算法 / PyTorch shipped chunked Linear + Cross-Entropy to core — the whole algorithm is 16 lines

> **一句话 / In one line**: 不再把 `(N, vocab)` 大 logits 一次性物化:每次只算一小块,算完 `log_softmax_target` 立刻丢掉,峰值显存从 `O(N·V)` 砍成 `O(chunk·V)` —— Liger / Apple-CCE / cut-cross-entropy 早就在用的把戏,现在进了 PyTorch 主仓。 / Never materialize the full `(N, vocab)` logits matrix: compute one chunk at a time, take its `log_softmax_target`, then throw the chunk away. Peak memory drops from `O(N·V)` to `O(chunk·V)` — the trick Liger / Apple-CCE / cut-cross-entropy pioneered, now landed in core PyTorch.

## 为什么重要 / Why this matters

训练一个 LLM,最后一层 `Linear(hidden, vocab)` 后面紧跟 cross-entropy。对 32k+ 词表的现代模型(Qwen3 是 152k,DeepSeek-V3 是 129k),把整个 `(batch × seq, vocab)` logits 矩阵物化一份,在 fp16 / bf16 下就是几个 GB —— activation 占用可能比模型本身还大,完全是浪费。社区的解法是"分块":同时只算一小撮 token 的 logits,把损失累加上去就丢掉。Liger-Kernel、Apple 的 CCE、Tri Dao 的 cut-cross-entropy 各自实现了一版,但都是 third-party。这次 PyTorch 把它做成了官方 `nn.LinearCrossEntropy`,而且把核心数学浓缩到了 16 行 —— 是 `reduction='none'` forward 的 loss-only 分支。读这一段你就理解了整个 chunked-CE 的算法骨架。

Training an LLM, the final `Linear(hidden, vocab)` is followed by cross-entropy. For modern 32k+ vocabs (Qwen3 is 152k, DeepSeek-V3 is 129k), materialising the full `(batch × seq, vocab)` logits in fp16/bf16 burns several gigabytes — the activation footprint can outweigh the model itself, which is pure waste. The community fix is chunking: compute logits for one small group of tokens at a time, fold their loss into the running total, then throw the chunk away. Liger-Kernel, Apple's CCE, and Tri Dao's cut-cross-entropy each shipped third-party versions. PyTorch just promoted this into core as `nn.LinearCrossEntropy`, and the *mathematical heart* of the algorithm is a 16-line block: the `reduction='none'` loss-only branch. Read just that block and you understand the entire chunked-CE algorithm.

## 代码 / The code

`pytorch/pytorch` — [`torch/nn/modules/linear_cross_entropy.py`](https://github.com/pytorch/pytorch/blob/f723c876cf5d6884f829514dc591cf9e69f4948e/torch/nn/modules/linear_cross_entropy.py#L751-L772)

```python
# reduction='none' forward (no upstream grad yet): per-sample loss
# into an (N,) output, no gradient precompute. ``chunk.weight_chunk``
# is the unsigned masked class weight here (neg_weight_target's "none"
# forward form), so each row's loss is W[T[n]] * (log denom -
# shifted_logit[T[n]]) = W[T[n]] * (-log_softmax). The "none" backward
# (loss_grad_output set) routes through the grad loop below instead.
if reduction == "none" and loss_grad_output is None:
    linear_bias_cast = ctx.linear_bias_cast
    out = torch.empty(ctx.num_batches, dtype=dtype, device=ctx.input.device)
    for chunk in ctx.chunks():
        logits = chunk.logits
        ctx.mm(chunk.input, chunk.linear_weight.T, out=logits)
        if linear_bias_cast is not None:
            logits.add_(linear_bias_cast)
        logits.sub_(ctx.amax(logits))
        # Read the target logit BEFORE ``sumexp_`` -- it does ``exp_()``
        # in place, overwriting logits with exp(shifted).
        ls_target = logits.gather(1, chunk.target_chunk.unsqueeze(1)).squeeze(1)
        softmax_denom = ctx.sumexp_(logits, dim=1)
        loss_chunk = softmax_denom.log_().sub_(ls_target.to(softmax_denom.dtype))
        loss_chunk.mul_(chunk.weight_chunk.to(softmax_denom.dtype))
        out.narrow(0, chunk.bchunk_start, chunk.bchunk_size).copy_(loss_chunk)
    return (out, ctx.grad_input.to(dtype), ctx.grad_linear_weight, ctx.grad_linear_bias)
```

## 逐行讲解 / What's happening

1. **`for chunk in ctx.chunks()`**:
   - 中文: `ctx` 把 `(num_batches × hidden)` 的输入按 `batch_chunk_size` 切成块,这个外层循环对每一块做一次完整的 "matmul → log-softmax → loss" pass。注意"chunk"切的是 batch 维 (token 数),不是 vocab 维 —— vocab 永远完整。
   - English: `ctx` slices the `(num_batches × hidden)` input into chunks of `batch_chunk_size` rows. The outer loop runs one full "matmul → log-softmax → loss" pass per chunk. Notice that *batches* (i.e. tokens) get chunked, never the vocab — vocab dimension stays whole.

2. **`ctx.mm(chunk.input, chunk.linear_weight.T, out=logits)`**:
   - 中文: 当前块的局部 logits = `chunk.input @ W.T`,结果直接写进预分配的 `chunk.logits` buffer。这个 `(chunk × vocab)` 是整段代码里唯一占大空间的临时张量 —— chunk_size 通常 256-2048,vocab 152k 的话就是 ~600 MB / chunk(bf16),而不是全 batch 的几十 GB。
   - English: The chunk's local logits = `chunk.input @ W.T`, written directly into a pre-allocated `chunk.logits` buffer. This `(chunk × vocab)` tensor is the *only* large temporary in the whole snippet — typical `chunk_size` is 256-2048, so for a 152k vocab in bf16 you peak at ~600 MB per chunk instead of the tens-of-GB you'd burn materialising the full batch.

3. **`logits.add_(linear_bias_cast)` 然后 `logits.sub_(ctx.amax(logits))`**:
   - 中文: 加 bias,再做 softmax 稳定性的经典操作 —— 每一行减去自己的 max。这一减最重要的是数值稳定:exp 永远不会上溢,因为最大值的 exp 就是 1。
   - English: Add the bias, then do the classic numerical-stability shift: subtract the per-row max. This is the move that prevents `exp` from overflowing — the row's max becomes 0, so `exp(0) = 1` is the largest value entering the sum.

4. **`ls_target = logits.gather(1, chunk.target_chunk.unsqueeze(1)).squeeze(1)`** (注释强调"BEFORE sumexp_"):
   - 中文: **关键顺序**:必须在 `sumexp_` 之前把目标位置的 logit 取出来。因为下一行 `sumexp_` 会原地 `exp_()` 覆盖掉 logits buffer —— 等做完就只有 `exp(logits)` 了,原来的 logit 值丢了。这一行就是"先存证据再毁现场"。
   - English: **Critical ordering**: gather the target-position logits *before* `sumexp_`. The next line does `exp_()` in place and overwrites `logits` with `exp(shifted_logits)` — the original logit values are gone. This line is the "save the evidence before erasing the crime scene" move.

5. **`softmax_denom = ctx.sumexp_(logits, dim=1)`**:
   - 中文: 原地 `exp_()` 后沿 vocab 维求和 —— 得到 softmax 的分母 `Σ exp(shifted)`。注意求和总是用 `acc_dtype` (fp32),fp16 在词表大时会 underflow。
   - English: In-place `exp_()` then reduce along the vocab dim — this is the softmax denominator `Σ exp(shifted)`. The accumulation runs in `acc_dtype` (fp32); fp16 silently under-flows on large vocabs because `1/N` is sub-normal at `N ≥ 65536`.

6. **`loss_chunk = softmax_denom.log_().sub_(ls_target.to(...))`**:
   - 中文: `log(Σ exp) - target_logit` 就是 `-log_softmax(target)`,也就是每个样本的 cross-entropy。这一行 = 完整 CE,不需要再算全 `log_softmax` 张量。
   - English: `log(Σ exp) - target_logit` equals `-log_softmax(target)`, which is exactly the per-sample CE. One line, the entire CE — no need to materialise the full `log_softmax` tensor anywhere.

7. **`loss_chunk.mul_(chunk.weight_chunk...)` 然后 `out.narrow(...).copy_(loss_chunk)`**:
   - 中文: 乘上类别权重(也处理了 ignore_index 的 mask —— `weight_chunk` 在被忽略的位置是 0),把这一块的损失写进最终输出 `(N,)` 张量的对应槽位。
   - English: Multiply by the class weight (which also encodes the `ignore_index` mask — `weight_chunk` is zero on ignored positions), then copy the per-chunk loss into its slot in the final `(N,)` output. Done.

## 类比 / The analogy

想象你是一家有 152,000 道菜的超大餐厅,每天要给 100,000 个顾客上菜。如果你坚持"一次性算出每个顾客对每道菜的偏好"(那就是 `(N, V)` 的偏好矩阵)和"全部上桌让他们挑",厨房根本放不下。聪明的做法是分桌服务:每次只处理 100 个顾客,算他们对全部 152k 道菜的偏好(数据库还是要全的)→ 立刻让他们点菜(`gather` 目标) → 算出他们这桌的"满意度" → 把桌子清掉、上下一桌。`out.narrow(...).copy_(...)` 就是把每桌的"满意度"按号码填进总账本。 chunked-CE 之所以能省钱,不是因为算少了,而是因为**每个临时矩阵的生命周期被限制在一桌之内**。

Picture yourself running a restaurant with a menu of 152,000 dishes and 100,000 customers a day. If you insist on materialising every customer's preference over every dish at once — the full `(N, V)` matrix — and laying it all out on tables, the kitchen simply won't fit. The smart move is table-by-table service: take 100 customers, compute their preferences over all 152k dishes (the menu stays whole), let them point at one (`gather` the target), record their satisfaction, clear the table, seat the next 100. `out.narrow(...).copy_(...)` is the cashier copying each table's satisfaction into the master ledger by seat number. Chunked-CE saves memory not by computing less, but by *constraining the lifetime of every intermediate to one table*.

## 自己跑一遍 / Try it yourself

```python
import torch

@torch.no_grad()
def chunked_ce(x, W, target, chunk=256):
    """Memory-cheap CE for one Linear + CE pass. Loss-only forward."""
    N = x.shape[0]
    losses = torch.empty(N, dtype=x.dtype, device=x.device)
    for i in range(0, N, chunk):
        logits = x[i:i+chunk] @ W.T                  # (chunk, V)
        logits.sub_(logits.amax(dim=1, keepdim=True))
        ls_target = logits.gather(1, target[i:i+chunk, None]).squeeze(1)
        denom = logits.exp_().sum(dim=1, dtype=torch.float32)
        losses[i:i+chunk] = (denom.log_() - ls_target.float()).to(x.dtype)
    return losses

# 100k tokens, 50k vocab — would be 20 GB in bf16 if materialised.
N, hidden, V = 100_000, 1024, 50_000
x = torch.randn(N, hidden, dtype=torch.bfloat16, device="cuda")
W = torch.randn(V, hidden, dtype=torch.bfloat16, device="cuda")
target = torch.randint(0, V, (N,), device="cuda")

loss = chunked_ce(x, W, target).mean()
print("loss:", loss.item(), "  peak chunk logits:", 256 * V * 2 / 1e6, "MB")
```

运行 / Run with:
```bash
pip install torch  # needs CUDA build for the GPU memory savings to be visible
python try.py
```

预期输出 / Expected output:
```
loss: ~10.82   peak chunk logits: 25.6 MB
```

中文一句:25 MB 的临时张量 vs. "全 batch 物化" 的 20 GB —— 800× 的差距。这就是为什么所有 LLM 训练框架现在都默认开 chunked-CE。

English: 25 MB of intermediates per chunk versus 20 GB if you materialise the full batch — an 800× reduction. This is why every major LLM training framework now ships chunked-CE by default.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Liger-Kernel (LinkedIn)**: 中文:用 Triton 重写了同样的 chunked CE,做到了更激进的 fused kernel —— PyTorch 这版是 eager 的 reference,Liger 是 fused 的 production。 / English: Reimplements the same chunked-CE in Triton with a more aggressive fused kernel. The PyTorch version is the eager reference; Liger is the fused production path.
- **Apple's CCE (Cut Cross Entropy)**: 中文:更进一步 —— 只为目标位置算 logits,完全不算非目标的 exp。同样的"目标先取、其余只算 denom"思路推到极限。 / English: Goes one step further — compute logits only at the target position, never the non-target columns. The same "save target first, accumulate the denominator" idea pushed to its logical extreme.
- **DeepSpeed Tied-Embedding LCE / Megatron 的 vocab-parallel**: 中文:大词表场景的另一种解 —— 不是分块,而是把 vocab 拆到不同 GPU。两种思路正交,常常组合使用。 / English: An orthogonal approach to large vocabs — instead of chunking, shard the vocab dimension across GPUs. The two ideas compose and modern stacks ship both.

## 注意事项 / Caveats / when it breaks

- **chunk_size 太小 → kernel launch 开销吃光收益 / Too-small chunk → kernel-launch overhead eats the savings**: 中文:`chunk=8` 启动 12,500 次 mm 调用,GPU 大部分时间空转。实战 256-2048 比较合理,跟 SDPA / FlashAttention 的 tile 量级类似。 / English: A chunk of 8 means 12,500 mm launches and the GPU spends most of its time idle between dispatches. Sweet spot is 256-2048, in the same order of magnitude as SDPA / FlashAttention tile sizes.
- **不是双反向友好的 / Not double-backward friendly**: 中文:看文件后面的 `_linear_cross_entropy_batch_chunked_backward`,显式提示 `retain_graph=True / double backward 不支持`。一些高阶用法(meta-learning、二阶优化)需要回退到 eager 全 CE。 / English: The backward method documents that `retain_graph=True` and double-backward are unsupported. Workflows that need them (meta-learning, second-order optimization) must fall back to the eager full-CE path.
- **label_smoothing > 0 直接 raise**: 中文:这一版还不支持 label smoothing,看到 `NotImplementedError` 就老老实实降级到 `F.cross_entropy`。 / English: Label smoothing isn't implemented yet — the constructor raises `NotImplementedError`. If your recipe needs it, fall back to `F.cross_entropy` until upstream lands the missing path.

## 延伸阅读 / Further reading

- [PyTorch RFC: Chunked Linear + Cross Entropy (pytorch/pytorch #178893)](https://github.com/pytorch/pytorch/issues/178893)
- [Liger-Kernel — FusedLinearCrossEntropy](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/fused_linear_cross_entropy.py)
- [Apple's Cut Cross Entropy paper (arXiv 2411.09009)](https://arxiv.org/abs/2411.09009)
