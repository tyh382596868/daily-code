# PyTorch

Notes tagged `pytorch`, newest first. Daily teaching points sourced from
[`pytorch/pytorch`](https://github.com/pytorch/pytorch) — fused optimizers, autograd
internals, distributed, `torch.compile`, and other framework internals.

| Date | Title | File |
|------|-------|------|
| 2026-06-05 | [`is_causal=True` is wrong for KV-cache decoding: upper-left vs. lower-right](../2026/06/2026-06-05-pytorch-causal-bias-upper-lower.md) | `torch/nn/attention/bias.py` |
| 2026-05-29 | [Adafactor: a row vector and a column vector replace the full second-moment matrix](../2026/05/2026-05-29-pytorch-adafactor-rank1-factorization.md) | `torch/optim/_adafactor.py` |
| 2026-05-28 | [Composable activation checkpointing with forward hooks and a generator](../2026/05/2026-05-28-pytorch-composable-activation-checkpoint.md) | `torch/distributed/_composable/checkpoint_activation.py` |
| 2026-05-27 | [Muon's Newton-Schulz orthogonalization in 5 bf16 matmuls](../2026/05/2026-05-27-pytorch-muon-newton-schulz.md) | `torch/optim/_muon.py` |
