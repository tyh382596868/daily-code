# PyTorch

Notes tagged `pytorch`, newest first. Daily teaching points sourced from
[`pytorch/pytorch`](https://github.com/pytorch/pytorch) — fused optimizers, autograd
internals, distributed, `torch.compile`, and other framework internals.

| Date | Title | File |
|------|-------|------|
| 2026-06-08 | [PyTorch 把 FP8 attention 写进了官方:一个 154 行的 SDPA 量化 wrapper / PyTorch shipped FP8 attention to core: a 154-line quantized SDPA wrapper](../2026/06/2026-06-08-pytorch-fp8-sdpa-experimental.md) | `torch/nn/attention/experimental/_scaled_dot_product_attention_quantized.py` |
| 2026-06-08 | [把因果三角切成头尾配对:PyTorch 的 Context-Parallel 负载均衡 / Pairing head with tail: PyTorch's context-parallel load balancer for causal attention](../2026/06/2026-06-08-pytorch-cp-head-tail-balance.md) | `torch/distributed/tensor/experimental/_context_parallel/_load_balancer.py` |
| 2026-06-08 | [PyTorch 把 vLLM 的 paged KV cache 写进了官方:80 行的 page table 分配器 / PyTorch shipped vLLM-style paged KV cache: an 80-line page-table allocator](../2026/06/2026-06-08-pytorch-paged-attention-reserve.md) | `torch/nn/attention/experimental/_paged_attention.py` |
| 2026-06-07 | [PyTorch 怎么让 FA3、FA4 这种外部后端"插进"SDPA 调度器:一个 137 行的注册表 / How PyTorch lets external backends (FA3, FA4) plug into the SDPA dispatcher: a 137-line registry](../2026/06/2026-06-07-pytorch-flash-attention-registry.md) | `torch/nn/attention/_registry.py` |
| 2026-06-05 | [`is_causal=True` 在 KV-cache 解码里是错的:upper-left 和 lower-right 的差别 / `is_causal=True` is wrong for KV-cache decoding: the upper-left vs. lower-right story](../2026/06/2026-06-05-pytorch-causal-bias-upper-lower.md) | `torch/nn/attention/bias.py` |
| 2026-06-04 | [PyTorch's EMA is one fused lerp_ over the whole parameter list](../2026/06/2026-06-04-pytorch-ema-foreach-lerp.md) | `torch/optim/swa_utils.py` |
| 2026-06-03 | [PyTorch 把 vLLM 的 paged KV cache 写进了核心 attention API / PyTorch's new `varlen_attn` brings paged KV-cache, GQA and split-KV into core attention](../2026/06/2026-06-03-pytorch-varlen-attn-paged.md) | `torch/nn/attention/varlen.py` |
| 2026-06-02 | [一份 70 行的 multi-tensor 模板:`_get_total_norm` / The 70-line multi-tensor template: `_get_total_norm`](../2026/06/2026-06-02-pytorch-foreach-total-norm.md) | `torch/nn/utils/clip_grad.py` |
| 2026-06-01 | [PyTorch 把 Blelloch 并行扫描装进 100 行 HOP / PyTorch packs Blelloch parallel scan into a 100-line HOP](../2026/06/2026-06-01-pytorch-associative-scan-blelloch.md) | `torch/_higher_order_ops/associative_scan.py` |
| 2026-06-01 | [PagedAttention 进 PyTorch 主仓:80 行就把 vLLM 的"虚拟内存"搬了进来 / PagedAttention lands in core PyTorch: vLLM's "virtual memory" trick in 80 lines](../2026/06/2026-06-01-pytorch-paged-attention-allocator.md) | `torch/nn/attention/experimental/_paged_attention.py` |
| 2026-05-31 | [PyTorch 把"变长序列 attention"做成一个公开算子 / PyTorch turned variable-length attention into a public op](../2026/05/2026-05-31-pytorch-varlen-attn-dispatch.md) | `torch/nn/attention/varlen.py` |
| 2026-05-29 | [Adafactor: a row vector and a column vector replace the full second-moment matrix](../2026/05/2026-05-29-pytorch-adafactor-rank1-factorization.md) | `torch/optim/_adafactor.py` |
| 2026-05-28 | [Composable activation checkpointing with forward hooks and a generator](../2026/05/2026-05-28-pytorch-composable-activation-checkpoint.md) | `torch/distributed/_composable/checkpoint_activation.py` |
| 2026-05-27 | [Muon's Newton-Schulz orthogonalization in 5 bf16 matmuls](../2026/05/2026-05-27-pytorch-muon-newton-schulz.md) | `torch/optim/_muon.py` |
