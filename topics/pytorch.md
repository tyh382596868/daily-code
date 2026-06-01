# PyTorch

Notes tagged `pytorch`, newest first. Daily teaching points sourced from
[`pytorch/pytorch`](https://github.com/pytorch/pytorch) — fused optimizers, autograd
internals, distributed, `torch.compile`, and other framework internals.

| Date | Title | File |
|------|-------|------|
| 2026-06-01 | [PagedAttention lands in core PyTorch: vLLM's "virtual memory" trick in 80 lines](../2026/06/2026-06-01-pytorch-paged-attention-allocator.md) | `torch/nn/attention/experimental/_paged_attention.py` |
| 2026-05-29 | [Adafactor: a row vector and a column vector replace the full second-moment matrix](../2026/05/2026-05-29-pytorch-adafactor-rank1-factorization.md) | `torch/optim/_adafactor.py` |
| 2026-05-28 | [Composable activation checkpointing with forward hooks and a generator](../2026/05/2026-05-28-pytorch-composable-activation-checkpoint.md) | `torch/distributed/_composable/checkpoint_activation.py` |
| 2026-05-27 | [Muon's Newton-Schulz orthogonalization in 5 bf16 matmuls](../2026/05/2026-05-27-pytorch-muon-newton-schulz.md) | `torch/optim/_muon.py` |
