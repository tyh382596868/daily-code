# Infrastructure

Notes tagged `infrastructure`, newest first. Covers serving systems, CUDA kernels, training optimizers, distributed training, and tooling.

| Date | Title | Repo |
|------|-------|------|
| 2026-06-11 | [vLLM 的 KV-cache "高水位线":只对新进队的请求收一笔押金,治好抢占抖动 / vLLM's KV-cache "watermark": charge admission rent only on newly-admitted requests to stop preemption thrash](../2026/06/2026-06-11-vllm-kv-cache-watermark.md) | [vllm-project/vllm](https://github.com/vllm-project/vllm) |
| 2026-06-11 | [LightX2V 把 Wan 视频模型的 FFN 从 7 次 kernel 启动压到 3 次:MXFP8 融合的完整教学版 / LightX2V fuses Wan video model FFN from 7 kernel launches down to 3: a textbook walk-through of MXFP8 fusion](../2026/06/2026-06-11-lightx2v-mxfp8-ffn-fuse.md) | [ModelTC/LightX2V](https://github.com/ModelTC/LightX2V) |
| 2026-06-08 | [把 DC 电机的扭矩-速度曲线压成 45 行 Warp kernel / Squeezing a DC motor's torque-speed curve into 45 lines of NVIDIA Warp](../2026/06/2026-06-08-newton-dc-motor-warp-kernel.md) | [newton-physics/newton](https://github.com/newton-physics/newton) |
| 2026-06-08 | [DoRA 的 forward 就是一句话:用 (magnitude / \|\|W + BA\|\|) 重新归一化每一列 / DoRA's whole forward is "renormalize each column by magnitude / \|\|W + BA\|\|"](../2026/06/2026-06-08-torchtune-dora-magnitude-direction.md) | [pytorch/torchtune](https://github.com/pytorch/torchtune) |
| 2026-05-28 | [DyT (Dynamic Tanh): a "normalization" Triton kernel that does no reduction](../2026/05/2026-05-28-liger-kernel-dyt-triton.md) | [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) |
| 2026-05-28 | [Wrapping NCCL collectives as differentiable ops: flash-attention's 30-line sequence-parallel primitives](../2026/05/2026-05-28-flash-attention-sequence-parallel-autograd.md) | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) |
| 2026-05-26 | [StreamingLLM in 30 lines](../2026/05/2026-05-26-kvpress-streaming-llm.md) | [NVIDIA/kvpress](https://github.com/NVIDIA/kvpress) |
| 2026-05-26 | [vLLM's intrusive doubly-linked list for KV cache LRU](../2026/05/2026-05-26-vllm-intrusive-lru-queue.md) | [vllm-project/vllm](https://github.com/vllm-project/vllm) |
