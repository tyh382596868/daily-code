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

| Date | Component | Title | Repo |
|------|-----------|-------|------|
| 2026-05-29 | VLM backbone wiring + action expert head | [SmolVLA's VLM + slim action expert: deep-copy the config, shrink it, rewire cross-attention](../2026/05/2026-05-29-vla-smolvla-vlm-with-expert.md) | [huggingface/lerobot](https://github.com/huggingface/lerobot) |
| 2026-05-28 | training step + L1 metric | [OpenVLA's training step: 40 lines that supervise a robot policy as if it were an LLM](../2026/05/2026-05-28-vla-openvla-training-step.md) | [openvla/openvla](https://github.com/openvla/openvla) |

<!-- entries auto-appended by daily-code-teach, newest first -->
