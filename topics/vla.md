# VLA — build your own

Notes tagged `vla`, newest first. Daily teaching points sourced from
[openvla](https://github.com/openvla/openvla),
[openvla-oft](https://github.com/openvla/openvla-oft),
[lerobot](https://github.com/huggingface/lerobot),
[openpi](https://github.com/Physical-Intelligence/openpi),
[Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T),
and [starVLA](https://github.com/starVLA/starVLA).

Each entry teaches **one component** of a Vision-Language-Action model and
maps it explicitly to its role in a from-scratch `nanoVLA` / production VLA build.
Components covered include: vision / observation encoder, action tokenizer & head,
VLM backbone wiring, training loop & loss, action chunking, fine-tune scripts,
inference loop.

> 📋 **Action survey**: how every VLA / WAM repo wires actions, with a selection
> decision tree — [README-action-survey.md](../nano/vla/README-action-survey.md).

| Date | Component | Title | Repo |
|------|-----------|-------|------|
| 2026-05-29 | inference loop (async) | [Async inference: split slow inference and fast control into client-server](../nano/vla/2026-05-29-lerobot-async-inference.md) | [huggingface/lerobot](https://github.com/huggingface/lerobot) |
| 2026-05-29 | fine-tune / LoRA | [Fine-tuning a 7B VLA to a new robot is a handful of PEFT lines](../nano/vla/2026-05-29-openvla-lora-finetune.md) | [openvla/openvla](https://github.com/openvla/openvla) |
| 2026-05-29 | action chunking | [Action chunking: predict a chunk, drip-feed it via a queue](../nano/vla/2026-05-29-act-action-chunking.md) | [huggingface/lerobot](https://github.com/huggingface/lerobot) |
| 2026-05-29 | continuous action head | [Continuous action head: flow matching emits real-valued trajectories](../nano/vla/2026-05-29-groot-flow-matching-action-head.md) | [huggingface/lerobot](https://github.com/huggingface/lerobot) |
| 2026-05-29 | vision encoder | [One Conv2d is the entire patch embedding](../nano/vla/2026-05-29-nanovlm-patch-embed.md) | [huggingface/nanoVLM](https://github.com/huggingface/nanoVLM) |
| 2026-05-29 | VLM backbone wiring + action expert head | [SmolVLA's VLM + slim action expert: deep-copy the config, shrink it, rewire cross-attention](../nano/vla/2026-05-29-smolvla-vlm-with-expert.md) | [huggingface/lerobot](https://github.com/huggingface/lerobot) |
| 2026-05-28 | training step + L1 metric | [OpenVLA's training step: 40 lines that supervise a robot policy as if it were an LLM](../nano/vla/2026-05-28-openvla-training-step.md) | [openvla/openvla](https://github.com/openvla/openvla) |

<!-- entries auto-appended by daily-code-teach, newest first -->
