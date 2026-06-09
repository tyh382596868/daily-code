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
- **2026-06-08** · vla · [pi0-FAST 把 state / action / language 全塞进同一条 token 流,靠 PaliGemma 的 prefix-LM mask 完成融合 / pi0-FAST stuffs state / action / language into one token stream and lets PaliGemma's prefix-LM mask do the fusion](nano/vla/2026-06-08-pi0fast-multimodal-prefix-lm-fusion.md) — `huggingface/lerobot` (vlm-backbone-wiring)
- **2026-06-08** · vla · [pi0-FAST 怎么知道 action 该停了:训练埋两个 stop signal,JAX 和 PyTorch 各用一个 / How pi0-FAST knows when actions should stop: training plants two stop signals, JAX and PyTorch each pick a different one](nano/vla/2026-06-08-pi0fast-stop-signals-decode-loop.md) — `huggingface/lerobot` (inference-loop)
- **2026-06-08** · vla · [OpenVLA-OFT 把 LLaMA 退化成"位置查询编码器":action 位置全塞零,L1 head 一次出 8 步 / OpenVLA-OFT turns LLaMA into a "position-only query encoder": zero the action embeddings, let the L1 head emit 8 steps at once](nano/vla/2026-06-08-openvla-oft-zero-action-l1-head.md) — `openvla/openvla-oft` (action-head-continuous)
- **2026-06-08** · vla · [把要预测的位置塞 placeholder,让 attention 从 context 单向"填空" — 这条设计线从 BERT 走到了 OFT / Put placeholders at positions to be predicted, let attention "fill in" from context — a design lineage from BERT (2018) to OpenVLA-OFT (2025)](nano/vla/2026-06-08-openvla-oft-placeholder-attention-lineage.md) — `openvla/openvla-oft` (vlm-backbone-wiring)
- **2026-06-08** · vla · [OpenVLA 没有"融合模块":vision 钉前缀,action 钉后缀,32 层 causal attention 自己融 / OpenVLA has no "fusion module": vision pinned at prefix, action pinned at suffix, 32 layers of causal attention do the rest](nano/vla/2026-06-08-openvla-multimodal-fusion-causal-mask.md) — `openvla/openvla` (vlm-backbone-wiring)
- **2026-06-08** · vla · [OpenVLA 的训练目标就是标准 LM 的 next-token prediction,只是 labels 多了一行 mask / OpenVLA's training target is just standard LM next-token prediction — only one line of label masking restricts the loss to the 7 action positions](nano/vla/2026-06-08-openvla-next-token-prediction-target.md) — `openvla/openvla` (training-step)
- **2026-06-08** · infrastructure · [把 DC 电机的扭矩-速度曲线压成 45 行 Warp kernel / Squeezing a DC motor's torque-speed curve into 45 lines of NVIDIA Warp](2026/06/2026-06-08-newton-dc-motor-warp-kernel.md) — `newton-physics/newton` (trending)
- **2026-06-08** · wam · [一份 14 行的 EmbedND:文本 / 图像 / 视频共用一个 RoPE 模块 / 14 lines of EmbedND: text, image, and video share one RoPE module](nano/wam/2026-06-08-open-sora-embed-nd-generalized-rope.md) — `hpcaitech/Open-Sora` (patchify-positional)
- **2026-06-08** · vla · [整个 GR00T 的训练步骤就 6 行干净的 flow-matching / The whole GR00T training step is six clean lines of flow matching](nano/vla/2026-06-08-lerobot-groot-flow-matching-action-head.md) — `huggingface/lerobot` (action-head-continuous)
- **2026-06-08** · huggingface · [把困扰 Llama 移植半年的 RoPE 重排压成两行 view + transpose / The two-line `view` + `transpose` that fixes Llama's RoPE port nightmare](2026/06/2026-06-08-transformers-permute-for-rope.md) — `huggingface/transformers`
- **2026-06-08** · pytorch · [PyTorch 把 FP8 attention 写进了官方:一个 154 行的 SDPA 量化 wrapper / PyTorch shipped FP8 attention to core: a 154-line quantized SDPA wrapper](2026/06/2026-06-08-pytorch-fp8-sdpa-experimental.md) — `pytorch/pytorch`
- **2026-06-08** · infrastructure · [DoRA 的 forward 就是一句话:用 (magnitude / \|\|W + BA\|\|) 重新归一化每一列 / DoRA's whole forward is "renormalize each column by magnitude / \|\|W + BA\|\|"](2026/06/2026-06-08-torchtune-dora-magnitude-direction.md) — `pytorch/torchtune` (tracked)
- **2026-06-08** · wam · [没有学习参数也能 ×8 上采样:Open-Sora 的 3D pixel-shuffle / Upsample 8× with zero learnable params: Open-Sora's 3D pixel-shuffle](nano/wam/2026-06-08-open-sora-pixel-shuffle-3d.md) — `hpcaitech/Open-Sora` (vae-encoder-decoder)
- **2026-06-08** · wam · [Wan2.1 的 WanAttentionBlock:DiT block 的生产级长相 / Wan2.1's WanAttentionBlock: what a production-grade DiT block actually looks like](nano/wam/2026-06-08-wan21-attention-block-production.md) — `Wan-Video/Wan2.1` (dit-block)
- **2026-06-08** · vla · [用一层 Conv2d 把图片切成 token:nanoVLM 的视觉编码器 / One Conv2d turns pixels into tokens: nanoVLM's vision encoder](nano/vla/2026-06-08-nanovlm-vit-patch-embed.md) — `huggingface/nanoVLM` (vision-encoder)
- **2026-06-08** · vla · [37 行的 ViTPatchEmbeddings:一个 Conv2d 就是整个"图像分块" / 37 lines of ViTPatchEmbeddings: one Conv2d *is* the entire "patchify" step](nano/vla/2026-06-08-nanovlm-vit-patch-embeddings.md) — `huggingface/nanoVLM` (vision-encoder)
- **2026-06-08** · trending · [flashdreams 的 BlockKVCache:[sink | rolling window] 用 4 步协议讲清楚 / flashdreams's BlockKVCache: [sink | rolling window] explained as a 4-step protocol](2026/06/2026-06-08-flashdreams-block-kvcache.md) — `NVIDIA/flashdreams` (trending)
- **2026-06-08** · trending · [28 行 MiniPointNet:把 64 个点塞进一个 token / 28-line MiniPointNet: cram 64 points into one token](2026/06/2026-06-08-humanego-mini-pointnet.md) — `TX-Leo/HumanEgo` (trending)
- **2026-06-08** · huggingface · [第一个 block 的残差几乎没变?那就跳过剩下所有 block / If the first block's residual barely moved, skip every other block](2026/06/2026-06-08-diffusers-first-block-cache.md) — `huggingface/diffusers`
- **2026-06-08** · huggingface · [一个 step 调两次模型:Diffusers 的 Heun 二阶 flow-match 采样 / Two model calls per step: diffusers' Heun 2nd-order flow-match sampler](2026/06/2026-06-08-diffusers-flow-match-heun-step.md) — `huggingface/diffusers`
- **2026-06-08** · pytorch · [把因果三角切成头尾配对:PyTorch 的 Context-Parallel 负载均衡 / Pairing head with tail: PyTorch's context-parallel load balancer for causal attention](2026/06/2026-06-08-pytorch-cp-head-tail-balance.md) — `pytorch/pytorch`
- **2026-06-08** · pytorch · [PyTorch 把 vLLM 的 paged KV cache 写进了官方:80 行的 page table 分配器 / PyTorch shipped vLLM-style paged KV cache: an 80-line page-table allocator](2026/06/2026-06-08-pytorch-paged-attention-reserve.md) — `pytorch/pytorch`
- **2026-06-08** · tracked · [DINOv3 的 fp8 Linear:一个 65 行的可微分 fp8 矩阵乘法 / DINOv3's fp8 Linear: a differentiable fp8 matmul in 65 lines](2026/06/2026-06-08-dinov3-fp8-linear-autograd.md) — `facebookresearch/dinov3`
- **2026-06-08** · tracked · [DINOv3 的 Gram Loss:不蒸特征,蒸"特征之间的关系" / DINOv3's Gram Loss: don't distill features — distill the *relationships between* features](2026/06/2026-06-08-dinov3-gram-loss.md) — `facebookresearch/dinov3`
- **2026-06-07** · wam · [Flux / SD3 的双流 DiT 块:图像和文本各自一套 QKV,只在 attention 那一步合体 / Flux / SD3's dual-stream DiT block: image and text get their own QKV, and they only meet at attention](nano/wam/2026-06-07-open-sora-mmdit-double-stream.md) — `hpcaitech/Open-Sora` (dit-block)
- **2026-06-07** · vla · [一根 stride 等于 patch 的 Conv2d:VLA 视觉编码器的整个入口就这么简单 / One Conv2d with stride = patch size: the entire entry point of a VLA's vision encoder](nano/vla/2026-06-07-nanovlm-vit-patch-embeddings.md) — `huggingface/nanoVLM` (vision-encoder)
- **2026-06-07** · trending · [把交互式 world model 推理装进 CUDA graph:warmup → capture → replay,每帧延迟省下几十毫秒 / Wrapping interactive world-model inference in a CUDA graph: warmup → capture → replay shaves tens of milliseconds per frame](2026/06/2026-06-07-flashdreams-cuda-graph-wrapper.md) — `NVIDIA/flashdreams` (trending)
- **2026-06-07** · huggingface · [diffusers 怎么用一根 CUDA stream + pinned CPU 镜像把 30 GB 模型塞进 24 GB GPU / How diffusers fits a 30 GB diffusion model on a 24 GB GPU with one CUDA stream and pinned CPU mirrors](2026/06/2026-06-07-diffusers-group-offloading.md) — `huggingface/diffusers`
- **2026-06-07** · pytorch · [PyTorch 怎么让 FA3、FA4 这种外部后端"插进"SDPA 调度器:一个 137 行的注册表 / How PyTorch lets external backends (FA3, FA4) plug into the SDPA dispatcher: a 137-line registry](2026/06/2026-06-07-pytorch-flash-attention-registry.md) — `pytorch/pytorch`
- **2026-06-07** · tracked · [V-JEPA 的 3D 块掩码:140 行 torch 就能逼模型学"视频物理" / V-JEPA's 3D block masking: 140 lines of plain torch that force the model to learn video physics](2026/06/2026-06-07-jepa-multiblock3d-masking.md) — `facebookresearch/jepa`
- **2026-06-05** · wam · [同一个 DiT 骨架,两种条件注入方式:GR00T 的 cross-attn 变体 / Same DiT skeleton, two conditioning strategies: GR00T's cross-attention variant](nano/wam/2026-06-05-isaac-groot-dit-cross-attn.md) — `NVIDIA/Isaac-GR00T` (dit-block)
- **2026-06-05** · vla · [一颗 Conv2d 就是 patch embed:从零搭一个能给 VLA 用的 ViT / One Conv2d is your patch embed: a ViT from scratch ready to feed a VLA](nano/vla/2026-06-05-nanovlm-vit-from-scratch.md) — `huggingface/nanoVLM` (vision-encoder)

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
