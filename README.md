# Daily Code

Six curated code teaching points every day:
1. one from the **tracked** repos (rotates over robotics / diffusion / infrastructure),
2. one from **PyTorch** (`pytorch/pytorch` internals),
3. one from a **Hugging Face** main library (`transformers` / `diffusers` / `accelerate` / `peft` / `trl` / `datasets` / `tokenizers` / `nanoVLM`),
4. one **VLA** component (rotates over `openvla` / `openvla-oft` / `lerobot` / `openpi` / `Isaac-GR00T` / `starVLA`) — chosen as a building block for a from-scratch `nanoVLA` / production VLA,
5. one **World Action Model** component (rotates over `dreamzero` / `lingbot-va` / `FastWAM` / `Wan2.1` / `Open-Sora`) — chosen as a building block for a from-scratch `nanoWAM` / production WAM,
6. one **trending** project freshly discovered from GitHub.

Each entry is short (~10 minutes to read), self-contained, and follows a fixed template:
real code → step-by-step walkthrough → vivid analogy → minimal runnable example.
VLA and WAM entries additionally include a **"在 nanoVLA / nanoWAM 中的位置"** section
that maps the component to its role in a from-scratch implementation.

## Latest

<!-- auto-updated by daily-code-teach -->
- **2026-05-29** · vla · 📋 [VLA action survey: how every repo adds action (decision tree)](nano/vla/README-action-survey.md) — survey doc
- **2026-05-29** · vla · [Async inference: split slow inference and fast control into client-server](nano/vla/2026-05-29-lerobot-async-inference.md) — `huggingface/lerobot` (inference-loop)
- **2026-05-29** · vla · [Fine-tuning a 7B VLA to a new robot is a handful of PEFT lines](nano/vla/2026-05-29-openvla-lora-finetune.md) — `openvla/openvla` (fine-tune-lora)
- **2026-05-29** · vla · [Action chunking: predict a chunk, drip-feed it via a queue](nano/vla/2026-05-29-act-action-chunking.md) — `huggingface/lerobot` (action-chunking)
- **2026-05-29** · vla · [Continuous action head: flow matching emits real-valued trajectories](nano/vla/2026-05-29-groot-flow-matching-action-head.md) — `huggingface/lerobot` (action-head-continuous)
- **2026-05-29** · vla · [One Conv2d is the entire patch embedding](nano/vla/2026-05-29-nanovlm-patch-embed.md) — `huggingface/nanoVLM` (vision-encoder)
- **2026-05-29** · wam · [dreamzero appends action and state as register tokens inside the video sequence](nano/wam/2026-05-29-dreamzero-action-registers.md) — `dreamzero0/dreamzero` (action-register-tokens)
- **2026-05-29** · wam · [FastWAM spins up a full second DiT just for actions](nano/wam/2026-05-29-fastwam-action-dit.md) — `yuantianyuan01/FastWAM` (parallel-action-dit)
- **2026-05-29** · wam · [lingbot-va's action stack: two Linears and a deepcopy](nano/wam/2026-05-29-lingbot-action-embedder.md) — `Robbyant/lingbot-va` (action-encoder-projector)
- **2026-05-29** · wam · [Resample's feat_cache lets a 3D VAE process video of arbitrary length, one chunk at a time](nano/wam/2026-05-29-wan21-resample-streaming-cache.md) — `Wan-Video/Wan2.1` (temporal-compression)
- **2026-05-29** · wam · [60 lines of denoise loop is the entire WAM "generate"](nano/wam/2026-05-29-wan21-denoise-loop.md) — `Wan-Video/Wan2.1` (sampler-inference)
- **2026-05-29** · wam · [Two main lines of a training step: add noise, then weight the loss](nano/wam/2026-05-29-lingbot-add-noise-loss.md) — `Robbyant/lingbot-va` (training-loop)
- **2026-05-29** · wam · [CFG is two forwards and one weighted sum](nano/wam/2026-05-29-wan21-classifier-free-guidance.md) — `Wan-Video/Wan2.1` (classifier-free-guidance)
- **2026-05-29** · wam · [Text conditioning is 25 lines of cross-attention](nano/wam/2026-05-29-wan21-text-cross-attention.md) — `Wan-Video/Wan2.1` (text-conditioning)
- **2026-05-29** · wam · [Splitting RoPE three ways: frame, height, width](nano/wam/2026-05-29-wan21-3d-rope.md) — `Wan-Video/Wan2.1` (patchify-positional)
- **2026-05-29** · wam · [One line of padding turns nn.Conv3d into a causal 3D conv](nano/wam/2026-05-29-wan21-vae-causal-conv3d.md) — `Wan-Video/Wan2.1` (vae-encoder-decoder)
- **2026-05-29** · diffusion · [Distillation gradient = subtract two score networks (DMD in 70 lines)](2026/05/2026-05-29-causal-forcing-dmd-gradient.md) — `thu-ml/Causal-Forcing`
- **2026-05-29** · wam · [Seven mask predicates compose into one FlexAttention BlockMask for video + action](nano/wam/2026-05-29-lingbot-flex-mask-compose.md) — `Robbyant/lingbot-va`
- **2026-05-29** · vla · [SmolVLA's VLM + slim action expert: deep-copy the config, shrink it, rewire cross-attention](nano/vla/2026-05-29-smolvla-vlm-with-expert.md) — `huggingface/lerobot`
- **2026-05-29** · huggingface · [nanoVLM trades 256 image tokens for 64 fat tokens via pixel shuffle](2026/05/2026-05-29-nanovlm-pixel-shuffle-projector.md) — `huggingface/nanoVLM`
- **2026-05-29** · pytorch · [Adafactor: a row vector and a column vector replace the full second-moment matrix](2026/05/2026-05-29-pytorch-adafactor-rank1-factorization.md) — `pytorch/pytorch`
- **2026-05-29** · robotics · [One Linear layer for every robot body (CategorySpecificLinear)](2026/05/2026-05-29-isaac-groot-category-specific-linear.md) — `NVIDIA/Isaac-GR00T`
- **2026-05-28** · wam · [A complete rectified-flow scheduler in 90 lines](nano/wam/2026-05-28-dreamzero-flow-match-scheduler.md) — `dreamzero0/dreamzero`
- **2026-05-28** · vla · [OpenVLA's training step: 40 lines that supervise a robot policy as if it were an LLM](nano/vla/2026-05-28-openvla-training-step.md) — `openvla/openvla`

## Topics

- [Robotics](topics/robotics.md) — VLA, manipulation, locomotion
- [Diffusion / World Model](topics/diffusion.md) — generative models, video diffusion
- [Infrastructure](topics/infrastructure.md) — serving, kernels, training systems
- [PyTorch](topics/pytorch.md) — framework internals (optimizers, autograd, distributed, compile)
- [Hugging Face](topics/huggingface.md) — transformers, diffusers, accelerate, peft, trl, datasets, tokenizers
- [VLA — build your own](topics/vla.md) — components for `nanoVLA` and production VLA
- [WAM — build your own](topics/wam.md) — components for `nanoWAM` and production WAM

## nano/ — curriculum-driven build series

The `vla` and `wam` tracks follow a dependency-ordered curriculum (see
[`.config/nano-curriculum.json`](.config/nano-curriculum.json)) for building
`nanoVLA` / `nanoWAM` from scratch. Each day picks the next uncovered component
whose dependencies are satisfied.

- [`nano/vla/`](nano/vla/) — nano VLA components, dated, flat
- [`nano/wam/`](nano/wam/) — nano WAM components, dated, flat

## Full archive

See [INDEX.md](INDEX.md).

## How it works

Generated by the `daily-code` Claude Code skill (see `.claude/skills/daily-code/SKILL.md`).
Each day the skill:

1. **Scans** the repos listed in [`.config/tracked-repos.json`](.config/tracked-repos.json),
   picks one with recent activity matching today's topic in the rotation
2. **Picks** one teaching point from `pytorch/pytorch`, one from a rotating Hugging Face
   main library, one from a rotating VLA repo, and one from a rotating WAM repo
3. **Discovers** one trending project from GitHub matching the configured query
4. **Writes** six teaching notes (tracked + pytorch + huggingface + vla + wam + trending)
   and indexes them

## Customize what's tracked

Edit [`.config/tracked-repos.json`](.config/tracked-repos.json). Add or remove repos, change
the trending query, or adjust the topic rotation.

## Run

```
daily code
```

That's it. The skill chains fetch + teach automatically.
