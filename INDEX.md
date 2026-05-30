# Index

Full archive of daily code notes, newest first.

<!-- auto-updated by daily-code-teach -->

## Archive

| Date | Topic | Title | Source |
|------|-------|-------|--------|
| 2026-05-29 | vla | [VLA action survey: how every repo adds action (decision tree + comparison)](nano/vla/README-action-survey.md) | survey doc |
| 2026-05-29 | vla | [Async inference: split slow inference and fast control into client-server](nano/vla/2026-05-29-lerobot-async-inference.md) | [huggingface/lerobot](https://github.com/huggingface/lerobot) (vla, inference-loop) |
| 2026-05-29 | vla | [Fine-tuning a 7B VLA to a new robot is a handful of PEFT lines](nano/vla/2026-05-29-openvla-lora-finetune.md) | [openvla/openvla](https://github.com/openvla/openvla) (vla, fine-tune-lora) |
| 2026-05-29 | vla | [Action chunking: predict a chunk, drip-feed it via a queue](nano/vla/2026-05-29-act-action-chunking.md) | [huggingface/lerobot](https://github.com/huggingface/lerobot) (vla, action-chunking) |
| 2026-05-29 | vla | [Continuous action head: flow matching emits real-valued trajectories](nano/vla/2026-05-29-groot-flow-matching-action-head.md) | [huggingface/lerobot](https://github.com/huggingface/lerobot) (vla, action-head-continuous) |
| 2026-05-29 | vla | [One Conv2d is the entire patch embedding](nano/vla/2026-05-29-nanovlm-patch-embed.md) | [huggingface/nanoVLM](https://github.com/huggingface/nanoVLM) (vla, vision-encoder) |
| 2026-05-29 | wam | [dreamzero appends action and state as register tokens inside the video sequence](nano/wam/2026-05-29-dreamzero-action-registers.md) | [dreamzero0/dreamzero](https://github.com/dreamzero0/dreamzero) (wam, action-register-tokens) |
| 2026-05-29 | wam | [FastWAM spins up a full second DiT just for actions](nano/wam/2026-05-29-fastwam-action-dit.md) | [yuantianyuan01/FastWAM](https://github.com/yuantianyuan01/FastWAM) (wam, parallel-action-dit) |
| 2026-05-29 | wam | [lingbot-va's action stack: two Linears and a deepcopy](nano/wam/2026-05-29-lingbot-action-embedder.md) | [Robbyant/lingbot-va](https://github.com/Robbyant/lingbot-va) (wam, action-encoder-projector) |
| 2026-05-29 | wam | [Resample's feat_cache lets a 3D VAE process video of arbitrary length, one chunk at a time](nano/wam/2026-05-29-wan21-resample-streaming-cache.md) | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) (wam, temporal-compression) |
| 2026-05-29 | wam | [60 lines of denoise loop is the entire WAM "generate"](nano/wam/2026-05-29-wan21-denoise-loop.md) | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) (wam, sampler-inference) |
| 2026-05-29 | wam | [Two main lines of a training step: add noise, then weight the loss](nano/wam/2026-05-29-lingbot-add-noise-loss.md) | [Robbyant/lingbot-va](https://github.com/Robbyant/lingbot-va) (wam, training-loop) |
| 2026-05-29 | wam | [CFG is two forwards and one weighted sum](nano/wam/2026-05-29-wan21-classifier-free-guidance.md) | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) (wam, classifier-free-guidance) |
| 2026-05-29 | wam | [Text conditioning is 25 lines of cross-attention](nano/wam/2026-05-29-wan21-text-cross-attention.md) | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) (wam, text-conditioning) |
| 2026-05-29 | wam | [Splitting RoPE three ways: frame, height, width](nano/wam/2026-05-29-wan21-3d-rope.md) | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) (wam, patchify-positional) |
| 2026-05-29 | wam | [One line of padding turns nn.Conv3d into a causal 3D conv](nano/wam/2026-05-29-wan21-vae-causal-conv3d.md) | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) (wam, vae-encoder-decoder) |
| 2026-05-29 | diffusion | [Distillation gradient = subtract two score networks (DMD in 70 lines)](2026/05/2026-05-29-causal-forcing-dmd-gradient.md) | [thu-ml/Causal-Forcing](https://github.com/thu-ml/Causal-Forcing) (trending) |
| 2026-05-29 | wam | [Seven mask predicates compose into one FlexAttention BlockMask for video + action](nano/wam/2026-05-29-lingbot-flex-mask-compose.md) | [Robbyant/lingbot-va](https://github.com/Robbyant/lingbot-va) (wam) |
| 2026-05-29 | vla | [SmolVLA's VLM + slim action expert: deep-copy the config, shrink it, rewire cross-attention](nano/vla/2026-05-29-smolvla-vlm-with-expert.md) | [huggingface/lerobot](https://github.com/huggingface/lerobot) (vla) |
| 2026-05-29 | huggingface | [nanoVLM trades 256 image tokens for 64 fat tokens via pixel shuffle](2026/05/2026-05-29-nanovlm-pixel-shuffle-projector.md) | [huggingface/nanoVLM](https://github.com/huggingface/nanoVLM) (huggingface) |
| 2026-05-29 | pytorch | [Adafactor: a row vector and a column vector replace the full second-moment matrix](2026/05/2026-05-29-pytorch-adafactor-rank1-factorization.md) | [pytorch/pytorch](https://github.com/pytorch/pytorch) (pytorch) |
| 2026-05-29 | robotics | [One Linear layer for every robot body (CategorySpecificLinear)](2026/05/2026-05-29-isaac-groot-category-specific-linear.md) | [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) (tracked) |
| 2026-05-28 | wam | [A complete rectified-flow scheduler in 90 lines](nano/wam/2026-05-28-dreamzero-flow-match-scheduler.md) | [dreamzero0/dreamzero](https://github.com/dreamzero0/dreamzero) (wam) |
| 2026-05-28 | vla | [OpenVLA's training step: 40 lines that supervise a robot policy as if it were an LLM](nano/vla/2026-05-28-openvla-training-step.md) | [openvla/openvla](https://github.com/openvla/openvla) (vla) |
| 2026-05-28 | infrastructure | [DyT (Dynamic Tanh): a "normalization" Triton kernel that does no reduction](2026/05/2026-05-28-liger-kernel-dyt-triton.md) | [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) (trending) |
| 2026-05-28 | huggingface | [A closure that re-ties weights after FSDP2 silently breaks them](2026/05/2026-05-28-accelerate-fsdp2-weight-retie.md) | [huggingface/accelerate](https://github.com/huggingface/accelerate) (huggingface) |
| 2026-05-28 | pytorch | [Composable activation checkpointing with forward hooks and a generator](2026/05/2026-05-28-pytorch-composable-activation-checkpoint.md) | [pytorch/pytorch](https://github.com/pytorch/pytorch) (pytorch) |
| 2026-05-28 | infrastructure | [Wrapping NCCL collectives as differentiable ops: flash-attention's 30-line sequence-parallel primitives](2026/05/2026-05-28-flash-attention-sequence-parallel-autograd.md) | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) (tracked) |
| 2026-05-27 | huggingface | [PEFT's LoRA forward: one line of addition is the whole algorithm](2026/05/2026-05-27-huggingface-peft-lora-forward.md) | [huggingface/peft](https://github.com/huggingface/peft) (huggingface) |
| 2026-05-27 | pytorch | [Muon's Newton-Schulz orthogonalization in 5 bf16 matmuls](2026/05/2026-05-27-pytorch-muon-newton-schulz.md) | [pytorch/pytorch](https://github.com/pytorch/pytorch) (pytorch) |
| 2026-05-27 | diffusion | [MPPI's softmax update: the heart of model-based planning in 30 lines](2026/05/2026-05-27-stable-worldmodel-mppi.md) | [galilai-group/stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) (trending) |
| 2026-05-27 | diffusion | [SIGReg: single-GPU isotropic-Gaussian regularization via random projections](2026/05/2026-05-27-le-wm-sigreg.md) | [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm) (tracked) |
| 2026-05-26 | robotics | [ReinFlow: rectified flow refactored into PyTorch modules](2026/05/2026-05-26-reinflow-rectified-flow.md) | [ReinFlow/ReinFlow](https://github.com/ReinFlow/ReinFlow) (trending) |
| 2026-05-26 | robotics | [π₀'s flow matching loss in 25 lines](2026/05/2026-05-26-openpi-flow-matching-loss.md) | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) (tracked) |
| 2026-05-26 | infrastructure | [StreamingLLM in 30 lines](2026/05/2026-05-26-kvpress-streaming-llm.md) | [NVIDIA/kvpress](https://github.com/NVIDIA/kvpress) (trending) |
| 2026-05-26 | infrastructure | [vLLM's intrusive doubly-linked list for KV cache LRU](2026/05/2026-05-26-vllm-intrusive-lru-queue.md) | [vllm-project/vllm](https://github.com/vllm-project/vllm) (tracked) |
| 2026-05-25 | diffusion | [CEM planning inside a learned world model](2026/05/2026-05-25-nano-world-model-cem-planner.md) | [simchowitzlabpublic/nano-world-model](https://github.com/simchowitzlabpublic/nano-world-model) (trending) |
| 2026-05-25 | diffusion | [DiT's adaLN-Zero block](2026/05/2026-05-25-dit-adaln-zero-block.md) | [facebookresearch/DiT](https://github.com/facebookresearch/DiT) (tracked) |
| 2026-05-10 | robotics | [OpenVLA action tokenizer](2026/05/2026-05-10-openvla-action-tokenizer-example.md) | [openvla/openvla](https://github.com/openvla/openvla) (tracked) |
