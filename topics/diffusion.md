# Diffusion / World Model

Notes tagged `diffusion`, newest first. Covers diffusion samplers, video diffusion, world model architectures, latent dynamics.

| Date | Title | Repo |
|------|-------|------|
| 2026-06-24 | [DPM-Solver++(2M)：用历史预测做二阶修正的视频扩散采样器 / DPM-Solver++(2M): 2nd-Order Multistep Correction via History Tracking in Video Diffusion](../2026/06/2026-06-24-cogvideo-dpmpp2m-sampler.md) | THUDM/CogVideo |
| 2026-06-24 | [ActionIDConstraintLogitsProcessor：一个 mask 把 LLM 的输出空间限制到动作词表 / ActionIDConstraintLogitsProcessor: One Mask That Constrains an LLM's Output Space to the Action Vocabulary](../2026/06/2026-06-24-ud-vla-action-constraint-logits.md) | OpenHelix-Team/Unified-Diffusion-VLA |
| 2026-06-21 | [VEnhancer 的 SDEdit 内核:60 行把"加噪→去噪"全讲完 / VEnhancer's SDEdit core: forward + reverse diffusion in 60 lines](../2026/06/2026-06-21-venhancer-sdedit-gaussian-diffusion.md) | Vchitect/VEnhancer |
| 2026-06-21 | [IC-LoRA 40 行:把参考视频 token 原封不动地拼进去,让 DiT 自己学对齐 / IC-LoRA in 40 lines: append clean reference tokens and let the DiT learn alignment on its own](../2026/06/2026-06-21-ltx2-ic-lora-reference-conditioning.md) | Lightricks/LTX-2 |
| 2026-06-13 | [100 行写完一个 JEPA 世界模型 —— 完整的 encode → predict → rollout 合约 / 100 lines for a complete JEPA world model — the full encode → predict → rollout contract](../2026/06/2026-06-13-le-wm-jepa-rollout.md) | lucas-maes/le-wm |
| 2026-06-13 | [NVIDIA flashdreams 的 `initialize_cache`:一次性 encode + 流式 VAE,这是交互式 AR 视频生成的设计核心 / NVIDIA flashdreams' `initialize_cache`: one-shot encode + streaming VAE — the design core of interactive AR video generation](../2026/06/2026-06-13-flashdreams-interactive-ar-cache.md) | NVIDIA/flashdreams |
| 2026-06-10 | [用世界模型当"想象器":dino_wm 的 71 行 CEM planner / Using the world model as an imagination engine: dino_wm's 71-line CEM planner](../2026/06/2026-06-10-dino-wm-cem-planner.md) | gaoyuezhou/dino_wm |
| 2026-06-10 | [Helios 的 attention 派发器:在 GPU 动物园里活下来的 167 行 / Helios's attention dispatcher: 167 lines that survive the GPU zoo](../2026/06/2026-06-10-helios-attention-dispatch.md) | PKU-YuanGroup/Helios |
| 2026-06-08 | [flashdreams 的 BlockKVCache:[sink | rolling window] 用 4 步协议讲清楚 / flashdreams's BlockKVCache: [sink | rolling window] explained as a 4-step protocol](../2026/06/2026-06-08-flashdreams-block-kvcache.md) | NVIDIA/flashdreams |
| 2026-06-08 | [28 行 MiniPointNet:把 64 个点塞进一个 token / 28-line MiniPointNet: cram 64 points into one token](../2026/06/2026-06-08-humanego-mini-pointnet.md) | TX-Leo/HumanEgo |
| 2026-06-08 | [DINOv3 的 fp8 Linear:一个 65 行的可微分 fp8 矩阵乘法 / DINOv3's fp8 Linear: a differentiable fp8 matmul in 65 lines](../2026/06/2026-06-08-dinov3-fp8-linear-autograd.md) | facebookresearch/dinov3 |
| 2026-06-08 | [DINOv3 的 Gram Loss:不蒸特征,蒸"特征之间的关系" / DINOv3's Gram Loss: don't distill features — distill the *relationships between* features](../2026/06/2026-06-08-dinov3-gram-loss.md) | facebookresearch/dinov3 |
| 2026-06-07 | [把交互式 world model 推理装进 CUDA graph:warmup → capture → replay,每帧延迟省下几十毫秒 / Wrapping interactive world-model inference in a CUDA graph: warmup → capture → replay shaves tens of milliseconds per frame](../2026/06/2026-06-07-flashdreams-cuda-graph-wrapper.md) | NVIDIA/flashdreams |
| 2026-06-07 | [V-JEPA 的 3D 块掩码:140 行 torch 就能逼模型学"视频物理" / V-JEPA's 3D block masking: 140 lines of plain torch that force the model to learn video physics](../2026/06/2026-06-07-jepa-multiblock3d-masking.md) | facebookresearch/jepa |
| 2026-06-05 | [Self-Forcing:只在一个步、只在最后几帧反向传播 / Self-Forcing: backprop through one step, only on the last few frames](../2026/06/2026-06-05-minwm-self-forcing.md) | shengshu-ai/minWM |
| 2026-06-05 | [V-JEPA 的 3D 块状 mask:乘起来取交集、补集留给 encoder / V-JEPA's 3D block mask: multiply for intersection, complement feeds the encoder](../2026/06/2026-06-05-jepa-multiblock3d-mask.md) | facebookresearch/jepa |
| 2026-06-04 | [DINOv3 distills the pairwise patch-similarity matrix, not the features](../2026/06/2026-06-04-dinov3-gram-loss.md) | facebookresearch/dinov3 |
| 2026-06-03 | [Self-Forcing 三件套:让自回归视频世界模型能 backprop 的 83 行 / Self-Forcing's three helpers: 83 lines that make autoregressive video world models trainable](../2026/06/2026-06-03-minwm-self-forcing.md) | shengshu-ai/minWM |
| 2026-06-03 | [DINOv3 的 RoPE 在训练时随机抖动坐标:一份免费的"位置数据增广" / DINOv3's RoPE randomizes its own coordinates at train time — free positional-encoding augmentation](../2026/06/2026-06-03-dinov3-rope-coord-augment.md) | facebookresearch/dinov3 |
| 2026-06-02 | [80 行展示「扩散 LLM 出图」的完整流水线 / 80 lines that demonstrate a full "diffusion-LLM image generation" pipeline](../2026/06/2026-06-02-llada-uni-two-phase-t2i.md) | inclusionAI/LLaDA2.0-Uni |
| 2026-06-02 | [90 行 2:4 稀疏的可热切 nn.Linear / 90 lines of hot-swappable 2:4 sparse nn.Linear](../2026/06/2026-06-02-dinov3-sparse-linear-w24.md) | facebookresearch/dinov3 |
| 2026-06-01 | [minWM 把 CM 蒸馏压成三个函数:抽 pair、教师跑 CFG+Euler、MSE / minWM compresses consistency distillation into three functions: sample a pair, teacher does CFG+Euler, student MSE](../2026/06/2026-06-01-minwm-consistency-distillation.md) | shengshu-ai/minWM |
| 2026-06-01 | [Self-Forcing 蒸馏的两个支柱:随机退出步 + 软梯度掩码 / The two pillars of Self-Forcing distillation: random exit step + soft gradient mask](../2026/06/2026-06-01-minwm-self-forcing.md) | shengshu-ai/minWM |
| 2026-06-01 | [用 80 行写一个 FP8 Linear:dinov3 的训练级 e4m3 量化 / 80 lines to ship an FP8 Linear: dinov3's training-grade e4m3 quantization](../2026/06/2026-06-01-dinov3-fp8-linear-autograd.md) | facebookresearch/dinov3 |
| 2026-06-01 | [V-JEPA 的"管子"遮罩:同一片像素在所有帧上一起被挡 / V-JEPA's "tube" mask: occlude the same pixels across every frame](../2026/06/2026-06-01-jepa-tube-mask.md) | facebookresearch/jepa |
| 2026-05-31 | [Self-Forcing 的全部代码就 80 行 / Self-Forcing is 80 lines](../2026/05/2026-05-31-minwm-self-forcing.md) | shengshu-ai/minWM |
| 2026-05-31 | [DINO 世界模型的 26 行自回归 rollout / DINO-WM's 26-line autoregressive rollout](../2026/05/2026-05-31-dino-wm-latent-rollout.md) | gaoyuezhou/dino_wm |
| 2026-05-29 | [Distillation gradient = subtract two score networks (DMD in 70 lines)](../2026/05/2026-05-29-causal-forcing-dmd-gradient.md) | thu-ml/Causal-Forcing |
| 2026-05-27 | [MPPI's softmax update: the heart of model-based planning in 30 lines](../2026/05/2026-05-27-stable-worldmodel-mppi.md) | galilai-group/stable-worldmodel |
| 2026-05-27 | [SIGReg: single-GPU isotropic-Gaussian regularization via random projections](../2026/05/2026-05-27-le-wm-sigreg.md) | lucas-maes/le-wm |
| 2026-05-25 | [CEM planning inside a learned world model](../2026/05/2026-05-25-nano-world-model-cem-planner.md) | simchowitzlabpublic/nano-world-model |
| 2026-05-25 | [DiT's adaLN-Zero block](../2026/05/2026-05-25-dit-adaln-zero-block.md) | facebookresearch/DiT |
