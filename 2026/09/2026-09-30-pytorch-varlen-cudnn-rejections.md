---
date: 2026-09-30
topic: pytorch
source: pytorch
repo: pytorch/pytorch
file: torch/nn/attention/varlen.py
permalink: https://github.com/pytorch/pytorch/blob/e9cbdb93e7527e23377e8e3aca8055f0858d8c7f/torch/nn/attention/varlen.py#L91-L170
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, pytorch, attention-backend]
---

# PyTorch varlen attention：先收集拒绝理由，再选择后端 / PyTorch Varlen Attention: Collect Rejection Reasons Before Choosing a Backend

> **一句话 / In one line**: cuDNN 能不能跑变长 attention，不靠一个大 if 猜，而是逐条列出不满足的约束。 / Whether cuDNN can run variable-length attention is decided by a list of failed constraints, not by one opaque if statement.

## 为什么重要 / Why this matters

变长 attention 同时碰到 CUDA、cuDNN 版本、head dimension、causal mask、paged KV cache 等约束。把“拒绝理由”收集成列表，不只方便 fallback，也让错误信息和调试路径变得可解释。

Variable-length attention sits at the intersection of CUDA, cuDNN versions, head dimensions, causal masking, and paged KV caches. Returning a list of rejection reasons makes fallback explainable instead of mysterious.

## 代码 / The code

`pytorch/pytorch` — [`torch/nn/attention/varlen.py`](https://github.com/pytorch/pytorch/blob/e9cbdb93e7527e23377e8e3aca8055f0858d8c7f/torch/nn/attention/varlen.py#L91-L170)

```python
def _cudnn_supports_head_dims(
    query: torch.Tensor, value: torch.Tensor, needs_backward: bool
) -> bool:
    """Return whether cuDNN supports this varlen head-dimension combination."""
    dims = (query.shape[-1], value.shape[-1])
    if dims[0] <= 128 and dims[1] <= 128:
        return True
    if not query.is_cuda:
        return False
    cudnn_version, major_cap = _cudnn_version_and_major_capability(query.device.index)
    if cudnn_version is None or major_cap != 10:
        return False
    if not needs_backward:
        return cudnn_version >= 92400 and dims in _CUDNN_SM100_FORWARD_LARGE_HEAD_DIMS
    return cudnn_version >= 91900 and dims in _CUDNN_SM100_BACKWARD_LARGE_HEAD_DIMS


def _cudnn_rejection_reasons(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seq_q: torch.Tensor,
    cu_seq_k: torch.Tensor | None,
    max_q: int,
    window_size: list[int],
    seqused_k: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    num_splits: int | None = None,
) -> list[str]:
    """Return the constraints preventing cuDNN varlen attention."""
    reasons = []
    if not query.is_cuda:
        reasons.append("query must be on CUDA")
    elif not _should_use_cudnn(query.device.index):
        reasons.append("cuDNN >= 9.18 on SM90 or SM100 is required")
    # cuDNN 9.24 is the oldest release validated for max_q <= 128 and for the
    # bottom-right causal alignment that KV caches require.
    cudnn_version = _cudnn_version(query)
    if max_q <= 128 and cudnn_version < 92400:
        reasons.append("max_q <= 128 requires cuDNN >= 9.24")
    elif max_q == 1 and query.is_cuda and _cudnn_decode_disabled(query.device.index):
        reasons.append("decode (max_q == 1) is disabled for this cuDNN version")
    if query.dtype not in (torch.float16, torch.bfloat16):
        reasons.append("query dtype must be float16 or bfloat16")
    if query.shape[-1] % 8 != 0 or value.shape[-1] % 8 != 0:
        reasons.append("query and value head dimensions must be divisible by 8")
    needs_backward = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (query, key, value)
    )
    if not _cudnn_supports_head_dims(query, value, needs_backward):
        phase = "backward" if needs_backward else "forward"
        reasons.append(
            f"query/value head dimensions {(query.shape[-1], value.shape[-1])} "
            f"are unsupported for cuDNN varlen {phase}"
        )
    if window_size == [-1, 0]:
        # cuDNN aligns the causal diagonal to the bottom right of a KV cache,
        # matching Flash for any lengths, but to the top left of packed
        # sequences, which only matches Flash when Q and K share lengths.
        if seqused_k is not None:
            if cudnn_version < 92400:
                reasons.append(
                    "causal attention with a KV cache requires cuDNN >= 9.24"
                )
        elif cu_seq_q is not cu_seq_k:
            reasons.append(
                "causal attention requires seqused_k or the same cu_seq tensor "
                "for Q and K"
            )
    elif window_size != [-1, -1]:
        reasons.append("window_size must be (-1, -1) or causal (-1, 0)")
    if num_splits is not None:
        reasons.append("num_splits is not supported")
    if block_table is not None:
        page_size = key.size(1)
        if page_size <= 0 or page_size & (page_size - 1):
            reasons.append("paged KV requires a power-of-two page size")
        if seqused_k is None:
            reasons.append("block_table requires seqused_k")
    return reasons
```

## 逐行讲解 / What's happening

1. **第 91-105 行 / Lines 91-105**:
   - 中文: `_cudnn_supports_head_dims` 先处理常见小 head，再给 SM100 大 head dimension 开版本白名单。
   - English: `_cudnn_supports_head_dims` accepts common small heads first, then uses version-gated allowlists for large SM100 head dimensions.
1. **第 108-125 行 / Lines 108-125**:
   - 中文: `_cudnn_rejection_reasons` 从空列表开始，把每个不满足的硬条件追加进去，而不是提前返回。
   - English: `_cudnn_rejection_reasons` starts with an empty list and appends every violated hard condition instead of returning early.
1. **第 126-145 行 / Lines 126-145**:
   - 中文: decode、小 query、dtype、head dim 和 backward 支持分别检查，错误信息能直接告诉用户差的是哪一项。
   - English: Decode mode, small queries, dtype, head dimensions, and backward support are checked separately, so the message points to the exact missing condition.
1. **第 146-170 行 / Lines 146-170**:
   - 中文: causal alignment 和 paged KV cache 是最容易混的地方；代码明确要求 `seqused_k`、共享 `cu_seq` 或 power-of-two page size。
   - English: Causal alignment and paged KV are easy to mix up, so the code spells out the `seqused_k`, shared-`cu_seq`, and power-of-two page constraints.

## 类比 / The analogy

像体检报告：医生不会只写“不适合手术”，而是列出血压、血糖、心电图哪些没过线。这样下一步是调整药物还是换方案，一眼能看出来。

It is like a medical checklist. Instead of saying “not eligible,” the report lists blood pressure, glucose, and ECG issues, making the next action obvious.

## 自己跑一遍 / Try it yourself

```python
def rejection(dtype='fp16', cuda=True, head=160, causal=True, has_lengths=False):
    reasons = []
    if not cuda: reasons.append('query must be on CUDA')
    if dtype not in ('fp16', 'bf16'): reasons.append('dtype must be fp16 or bf16')
    if head % 8: reasons.append('head dimension must be divisible by 8')
    if head > 128: reasons.append('large head needs a newer cuDNN allowlist')
    if causal and not has_lengths: reasons.append('causal packed mode needs lengths')
    return reasons

print(rejection(head=128, has_lengths=True))
print(rejection(dtype='fp32', cuda=False, head=160))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[]
['query must be on CUDA', 'dtype must be fp16 or bf16', 'large head needs a newer cuDNN allowlist', 'causal packed mode needs lengths']
```

空列表就是“可以尝试这个后端”；非空列表既是 fallback 条件，也是调试报告。

An empty list means “try this backend.” A non-empty list is both a fallback trigger and a debugging report.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **vLLM fusion gates** / **vLLM fusion gates**: kernel fusion 通常先跑一组 capability checks，再选择 fused 或 unfused path。 / Kernel fusion often runs capability checks before selecting fused or unfused paths.
- **Diffusers scheduler validation** / **Diffusers scheduler validation**: scheduler 参数经常逐项验证，给出可读的拒绝原因。 / Scheduler options are often validated one by one to produce readable rejection reasons.

## 注意事项 / Caveats / when it breaks

- **不要把 reason 当性能模型** / **Reasons are not a performance model**: 通过所有约束只说明“能跑”，不保证最快。 / Passing all constraints means runnable, not necessarily fastest.
- **版本条件要跟实际 kernel 对齐** / **Version checks must track kernels**: cuDNN 白名单如果滞后，会错失可用后端；如果超前，会触发运行时失败。 / Stale cuDNN allowlists can miss working backends; overly optimistic ones can fail at runtime.

## 延伸阅读 / Further reading

- [PyTorch scaled dot product attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [cuDNN frontend documentation](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/)
