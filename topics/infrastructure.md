# Infrastructure

Notes tagged `infrastructure`, newest first. Covers serving systems, CUDA kernels, training optimizers, distributed training, and tooling.

| Date | Title | Repo |
|------|-------|------|
| 2026-06-08 | [把 DC 电机的扭矩-速度曲线压成 45 行 Warp kernel / Squeezing a DC motor's torque-speed curve into 45 lines of NVIDIA Warp](../2026/06/2026-06-08-newton-dc-motor-warp-kernel.md) | [newton-physics/newton](https://github.com/newton-physics/newton) |
| 2026-06-08 | [DoRA 的 forward 就是一句话:用 (magnitude / \|\|W + BA\|\|) 重新归一化每一列 / DoRA's whole forward is "renormalize each column by magnitude / \|\|W + BA\|\|"](../2026/06/2026-06-08-torchtune-dora-magnitude-direction.md) | [pytorch/torchtune](https://github.com/pytorch/torchtune) |
| 2026-05-28 | [DyT (Dynamic Tanh): a "normalization" Triton kernel that does no reduction](../2026/05/2026-05-28-liger-kernel-dyt-triton.md) | [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) |
| 2026-05-28 | [Wrapping NCCL collectives as differentiable ops: flash-attention's 30-line sequence-parallel primitives](../2026/05/2026-05-28-flash-attention-sequence-parallel-autograd.md) | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) |
| 2026-05-26 | [StreamingLLM in 30 lines](../2026/05/2026-05-26-kvpress-streaming-llm.md) | [NVIDIA/kvpress](https://github.com/NVIDIA/kvpress) |
| 2026-05-26 | [vLLM's intrusive doubly-linked list for KV cache LRU](../2026/05/2026-05-26-vllm-intrusive-lru-queue.md) | [vllm-project/vllm](https://github.com/vllm-project/vllm) |
