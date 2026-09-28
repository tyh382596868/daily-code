---
date: 2026-09-28
topic: huggingface
source: huggingface
repo: huggingface/trl
file: trl/kernels/chunked_logprob.py
permalink: https://github.com/huggingface/trl/blob/a7c34f363a8716473a0f15378621a3358b994417/trl/kernels/chunked_logprob.py#L145-L285
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, chunked-logprob]
---

# TRL chunked logprob：不落地完整词表 logits / TRL Chunked Logprob: Avoid Materializing Full-Vocabulary Logits

> **一句话 / In one line**: RLHF 只需要目标 token 的 logprob 和熵时，可以按词表分块流式累积，而不是生成 `[N,V]` 大矩阵。 / When RLHF needs target-token logprobs and entropy, it can stream over vocabulary chunks instead of materializing an `[N,V]` matrix.

## 为什么重要 / Why this matters

偏好学习和 GRPO/RLHF 经常要算大词表上的 logprob。完整 logits 会把显存吃掉，尤其是长序列乘大词表。TRL 这里把词表切块，用 online logsumexp 累积归一化项，并在 backward 里重算 tile 来省内存。

Preference learning and GRPO/RLHF repeatedly compute logprobs over huge vocabularies. Full logits can dominate memory, especially for long sequences. TRL chunks the vocabulary, accumulates online logsumexp statistics, and recomputes tiles during backward to save memory.

## 代码 / The code

`huggingface/trl` — [`trl/kernels/chunked_logprob.py`](https://github.com/huggingface/trl/blob/a7c34f363a8716473a0f15378621a3358b994417/trl/kernels/chunked_logprob.py#L145-L285)

```python
class ChunkedLogProbFunction(torch.autograd.Function):
    """
    Per-token log-probabilities and entropy of `hidden @ weight.T`, without materializing the `[N, V]` logits.

    The projection runs in cuBLAS on `[TOKEN_CHUNK_SIZE, chunk_size]` tiles; a Triton kernel folds each tile into
    online-logsumexp statistics in one pass. The backward recomputes each tile, turns it into the logits gradient in
    place, and accumulates the gradient GEMMs in fp32.
    """

    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,  # [N, H]
        weight: torch.Tensor,  # [V, H]
        bias: torch.Tensor | None,  # [V]
        targets: torch.Tensor,  # [N]
        temperature: float,
        chunk_size: int,
        final_logit_softcapping: float | None = None,
        logit_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # entropy is often computed for logging only (no grad required); without this, autograd would
        # materialize its incoming gradient as zeros and backward would waste compute on a no-op term
        ctx.set_materialize_grads(False)
        device = hidden.device
        N = hidden.shape[0]
        vocab = weight.shape[0]
        compute_dtype = (
            torch.get_autocast_dtype(device.type) if torch.is_autocast_enabled(device.type) else hidden.dtype
        )
        kernel_args = {
            "logit_scale": logit_scale,
            "softcap": final_logit_softcapping or 1.0,
            "inv_t": 1 / temperature,
            "HAS_SOFTCAP": final_logit_softcapping is not None,
        }

        running_max = torch.full((N,), float("-inf"), device=device, dtype=torch.float32)
        sum_exp = torch.zeros((N,), device=device, dtype=torch.float32)
        x_sum_exp = torch.zeros((N,), device=device, dtype=torch.float32)
        target_logit = torch.zeros((N,), device=device, dtype=torch.float32)
        mm_buf = torch.empty((min(N, TOKEN_CHUNK_SIZE), chunk_size), device=device, dtype=compute_dtype)

        for token_start in range(0, N, TOKEN_CHUNK_SIZE):
            token_end = min(token_start + TOKEN_CHUNK_SIZE, N)
            n = token_end - token_start
            hidden_chunk = hidden[token_start:token_end].to(compute_dtype)
            for start in range(0, vocab, chunk_size):
                end = min(start + chunk_size, vocab)
                tile = mm_buf[:n, : end - start]
                torch.mm(hidden_chunk, weight[start:end].to(compute_dtype).t(), out=tile)
                if bias is not None:
                    tile.add_(bias[start:end].to(compute_dtype))
                _forward_kernel[(n,)](
                    tile,
                    tile.stride(0),
                    targets[token_start:token_end],
                    running_max[token_start:token_end],
                    sum_exp[token_start:token_end],
                    x_sum_exp[token_start:token_end],
                    target_logit[token_start:token_end],
                    start,
                    end - start,
                    BLOCK_SIZE=_BLOCK_SIZE,
                    **kernel_args,
                )

        log_z = running_max + torch.log(sum_exp)
        logprobs = target_logit - log_z
        entropy = log_z - x_sum_exp / sum_exp

        ctx.save_for_backward(hidden, weight, bias, targets, log_z, entropy)
        ctx.compute_dtype = compute_dtype
        ctx.chunk_size = chunk_size
        ctx.kernel_args = kernel_args
        return logprobs, entropy

    @staticmethod
    def backward(ctx, grad_logprobs: torch.Tensor | None, grad_entropy: torch.Tensor | None):  # type: ignore
        # `trl.trainer.utils` imports this module, so import from it at call time
        from ..trainer.utils import maybe_gather_lm_head_ctx

        hidden, weight, bias, targets, log_z, entropy = ctx.saved_tensors
        compute_dtype = ctx.compute_dtype
        chunk_size = ctx.chunk_size
        needs_hidden_grad, needs_weight_grad, needs_bias_grad = ctx.needs_input_grad[:3]
        N = hidden.shape[0]
        with maybe_gather_lm_head_ctx(weight, bias):
            vocab = weight.shape[0]
            # Always accumulate in fp32, even when the inputs are not
            grad_hidden = (
                torch.zeros(hidden.shape, device=hidden.device, dtype=torch.float32) if needs_hidden_grad else None
            )
            grad_weight = (
                torch.zeros(weight.shape, device=weight.device, dtype=torch.float32) if needs_weight_grad else None
            )
            grad_bias = torch.zeros(bias.shape, device=bias.device, dtype=torch.float32) if needs_bias_grad else None
            mm_buf = torch.empty((min(N, TOKEN_CHUNK_SIZE), chunk_size), device=hidden.device, dtype=compute_dtype)
            grad_logprobs = grad_logprobs.float().contiguous() if grad_logprobs is not None else None
            grad_entropy = grad_entropy.float().contiguous() if grad_entropy is not None else None

            for token_start in range(0, N, TOKEN_CHUNK_SIZE):
                token_end = min(token_start + TOKEN_CHUNK_SIZE, N)
                n = token_end - token_start
                hidden_chunk = hidden[token_start:token_end].to(compute_dtype)
                for start in range(0, vocab, chunk_size):
                    end = min(start + chunk_size, vocab)
                    w_chunk = weight[start:end].to(compute_dtype)
                    tile = mm_buf[:n, : end - start]
                    torch.mm(hidden_chunk, w_chunk.t(), out=tile)
                    if bias is not None:
                        tile.add_(bias[start:end].to(compute_dtype))
                    _backward_kernel[(n, triton.cdiv(end - start, _BLOCK_SIZE))](
                        tile,
                        tile.stride(0),
                        targets[token_start:token_end],
                        log_z[token_start:token_end],
                        entropy[token_start:token_end],
                        grad_logprobs[token_start:token_end] if grad_logprobs is not None else log_z,
                        grad_entropy[token_start:token_end] if grad_entropy is not None else log_z,
                        start,
                        end - start,
                        HAS_LOGPROB_GRAD=grad_logprobs is not None,
                        HAS_ENTROPY_GRAD=grad_entropy is not None,
                        BLOCK_SIZE=_BLOCK_SIZE,
                        **ctx.kernel_args,
                    )
                    if grad_hidden is not None:
                        _addmm_fp32(grad_hidden[token_start:token_end], tile, w_chunk)
                    if grad_weight is not None:
                        _addmm_fp32(grad_weight[start:end], tile.t(), hidden_chunk)
                    if grad_bias is not None:
                        grad_bias[start:end] += tile.sum(dim=0, dtype=torch.float32)

        return (
            grad_hidden.to(hidden.dtype) if grad_hidden is not None else None,
            grad_weight.to(weight.dtype) if grad_weight is not None else None,
            grad_bias.to(bias.dtype) if grad_bias is not None else None,
            None,
            None,
            None,
```

## 逐行讲解 / What's happening

1. **第 166-168 行 / Lines 166-168**:
   - 中文: 不需要 entropy 梯度时避免 materialized zero grad，少做一次无用 backward。
   - English: When entropy is logging-only, it avoids materializing a zero gradient and wasting backward work.
1. **第 182-186 行 / Lines 182-186**:
   - 中文: 每个 token 只保存 running max、sum exp、目标 logit 等小向量，而不是全词表 logits。
   - English: Each token keeps small statistics such as running max, sum exp, and target logit instead of full logits.
1. **第 188-210 行 / Lines 188-210**:
   - 中文: 外层按 token chunk，内层按 vocab chunk，GEMM tile 立刻交给 Triton kernel 折叠进统计量。
   - English: The outer loop chunks tokens, the inner loop chunks vocabulary, and each GEMM tile is folded into statistics immediately.
1. **第 246-277 行 / Lines 246-277**:
   - 中文: 反向传播重算同样的 tile，用计算换显存，再把梯度累加到 hidden、weight 和 bias。
   - English: Backward recomputes the same tiles, trading compute for memory, then accumulates gradients for hidden, weight, and bias.

## 类比 / The analogy

像查电话簿找一个号码：你不用把整本电话簿复印到桌上，只要一页页扫，同时记住目前需要的统计信息。

It is like searching a phone book. You do not photocopy the whole book onto your desk; you scan page by page while keeping the statistics you need.

## 自己跑一遍 / Try it yourself

```python
import math
logits=[1.0, 3.0, 2.0, 0.0]
target=1
running=-math.inf; total=0.0
for chunk in [logits[:2], logits[2:]]:
    m=max(running, max(chunk))
    total=total*math.exp(running-m)+sum(math.exp(x-m) for x in chunk)
    running=m
log_z=running+math.log(total)
print(round(logits[target]-log_z, 4))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
-0.4402
```

分块之后得到的 logsumexp 和一次性全量计算等价，只是峰值内存低得多。

The chunked logsumexp matches the full computation while using much lower peak memory.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **PyTorch linear cross entropy** / **PyTorch has similar fused/chunked ideas for linear plus CE.**
- **vLLM paged attention** / **Paged attention also avoids materializing one huge dense object.**
- **FlashAttention** / **FlashAttention streams tiles and keeps online softmax stats.**

## 注意事项 / Caveats / when it breaks

- **重算会增加计算** / **Memory drops, but backward does extra GEMMs.**
- **kernel 依赖硬件** / **The fast path assumes Triton/GPU support.**
- **数值稳定靠 running max** / **Removing the online max update would make large logits unstable.**

## 延伸阅读 / Further reading

- TRL source permalink above
- FlashAttention paper for online softmax intuition
