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
- **2026-07-01** · infrastructure · [nanoGPT 推理优化：只给最后一个 token 做 lm_head / nanoGPT Inference Optimization: Run lm_head Only on the Last Token](2026/07/2026-07-01-nanogpt-last-token-logits.md) — `karpathy/nanoGPT` (tracked)
- **2026-07-01** · pytorch · [PyTorch 子模块手术：用点路径精准替换一层 / PyTorch Submodule Surgery: Replace a Layer by Dotted Path](2026/07/2026-07-01-pytorch-submodule-surgery.md) — `pytorch/pytorch`
- **2026-07-01** · huggingface · [TRL 的 PEFT adapter EMA teacher：不用复制整模型的自蒸馏 / TRL PEFT Adapter EMA Teacher: Self-Distillation Without Copying the Whole Model](2026/07/2026-07-01-trl-peft-adapter-ema-teacher.md) — `huggingface/trl`
- **2026-07-01** · vla · [LeRobot action queue：把重叠 action chunk 合成连续控制流 / LeRobot Action Queue: Merge Overlapping Action Chunks into a Continuous Control Stream](nano/vla/2026-07-01-lerobot-action-queue-merge.md) — `huggingface/lerobot` (inference-loop cross-repo)
- **2026-07-01** · wam · [DreamZero action/state RoPE：把控制 token 接进视频坐标系 / DreamZero Action/State RoPE: Splice Control Tokens into the Video Coordinate System](nano/wam/2026-07-01-dreamzero-action-state-rope.md) — `dreamzero0/dreamzero` (action-conditioning cross-repo)
- **2026-07-01** · infrastructure · [LMCache 的 CacheEngineKey：KV chunk 的分布式门牌号 / LMCache CacheEngineKey: A Distributed Address for KV Chunks](2026/07/2026-07-01-lmcache-cache-engine-key.md) — `LMCache/LMCache` (trending)
- **2026-06-30** · diffusion · [Wan2.1 的 3D RoPE 自注意力：把时间、高度、宽度拆成三把尺 / Wan2.1 3D RoPE Self-Attention: Three Rulers for Time, Height, and Width](2026/06/2026-06-30-wan21-rope-self-attention.md) — `Wan-Video/Wan2.1` (tracked)
- **2026-06-30** · pytorch · [PyTorch 梯度裁剪的第二步：只缩小，不放大 / PyTorch Gradient Clipping Step Two: Scale Down, Never Up](2026/06/2026-06-30-pytorch-clip-grads-with-norm.md) — `pytorch/pytorch`
- **2026-06-30** · huggingface · [LoRA-GA 初始化：用一次梯度 SVD 给 adapter 指方向 / LoRA-GA Init: Use One Gradient SVD to Aim the Adapter](2026/06/2026-06-30-peft-loraga-svd-init.md) — `huggingface/peft`
- **2026-06-30** · vla · [LeRobot Diffusion Policy：从噪声动作轨迹反推可执行 chunk / LeRobot Diffusion Policy: Denoise a Noisy Action Trajectory into an Executable Chunk](nano/vla/2026-06-30-lerobot-diffusion-action-sampler.md) — `huggingface/lerobot` (action-head-continuous cross-repo)
- **2026-06-30** · wam · [nanoWAM 的 3D 坐标层：RoPE 不只是一维位置 / nanoWAM's 3D Coordinate Layer: RoPE Is Not Just One-Dimensional Position](nano/wam/2026-06-30-wan21-rope-grid-apply.md) — `Wan-Video/Wan2.1` (patchify-positional cross-repo)
- **2026-06-30** · diffusion · [DiffSynth 的 WanVideoPipeline：把视频生成拆成可插拔单元 / DiffSynth's WanVideoPipeline: Video Generation as Pluggable Units](2026/06/2026-06-30-diffsynth-wan-video-pipeline-units.md) — `modelscope/DiffSynth-Studio` (trending)
- **2026-06-26** · robotics · [StarVLA 双流协同训练：一个 step 里两条 backward / StarVLA Dual-Stream Cotrain: Two Backward Passes in One Step](2026/06/2026-06-26-starvla-cotrain.md) — `starVLA/starVLA` (tracked)
- **2026-06-26** · pytorch · [`torch.func.vmap`：把任意函数向量化，批量维度随心所欲 / `torch.func.vmap`: Vectorize Any Function, Put the Batch Dimension Anywhere](2026/06/2026-06-26-pytorch-vmap.md) — `pytorch/pytorch`
- **2026-06-26** · huggingface · [`split_dataset_by_node`：一个 API 背后藏着两种完全不同的分布式数据切分策略 / `split_dataset_by_node`: One API, Two Fundamentally Different Distribution Strategies Hidden Inside](2026/06/2026-06-26-datasets-split-by-node.md) — `huggingface/datasets`
- **2026-06-26** · vla · [ACT 的 CVAE 动作编码器：用重参数化技巧压缩动作序列，推理时 latent 直接置零 / ACT's CVAE Action Encoder: Compress Action Chunks via Reparameterization, Set Latent to Zero at Inference](nano/vla/2026-06-26-lerobot-act-cvae.md) — `huggingface/lerobot` (action-chunking cross-repo)
- **2026-06-26** · wam · [DiT 训练循环五步法：VAE 编码 → 随机时间步 → 扩散损失 → backward → EMA 更新 / DiT's 5-Step Training Loop: VAE Encode → Random Timestep → Diffusion Loss → Backward → EMA Update](nano/wam/2026-06-26-dit-training-loop.md) — `facebookresearch/DiT` (training-loop cross-repo)
- **2026-06-26** · robotics · [FastVideo 的视频稀疏注意力序列并行：把 gate_compress 打包进 all-to-all，省掉一次通信 / FastVideo VSA Sequence Parallelism: Bundle gate_compress into the all-to-all and Save One Communication Round](2026/06/2026-06-26-fastvideo-distributed-vsa.md) — `hao-ai-lab/FastVideo` (trending)
- **2026-06-25** · infrastructure · [DeepSeek-V3 MLA 的 absorb 技巧：KV 缓存压缩 70× / DeepSeek-V3 MLA's absorb Trick: 70× KV-Cache Compression](2026/06/2026-06-25-deepseek-v3-mla-absorb.md) — `deepseek-ai/DeepSeek-V3` (tracked)
- **2026-06-25** · pytorch · [Pipeline 并行的微批次拆分：pytree-aware 的 split_args_kwargs_into_chunks / Pipeline-Parallel Microbatch Splitting: pytree-aware split_args_kwargs_into_chunks](2026/06/2026-06-25-pytorch-pipeline-microbatch.md) — `pytorch/pytorch`
- **2026-06-25** · huggingface · [nanoVLM GQA：prefill/decode 统一路径 + `is_causal` 精确谓词 / nanoVLM GQA: Unified Prefill/Decode Path + Precise `is_causal` Predicate](2026/06/2026-06-25-nanovlm-gqa-kvcache.md) — `huggingface/nanoVLM`
- **2026-06-25** · vla · [SmolVLA 的流匹配训练步：Beta 分布时间采样 + VLM 骨干输出 MSE 损失 / SmolVLA's Flow-Matching Training Step: Beta-Distributed Time Sampling + VLM Backbone MSE Loss](nano/vla/2026-06-25-smolvla-flow-matching-step.md) — `huggingface/lerobot` (training-step cross-repo)
- **2026-06-25** · wam · [Wan2.1 WanModel：一个 Linear 产生 6 个 adaLN 调制向量 + 逐视频变长 patchify / Wan2.1 WanModel: One Linear Produces 6 adaLN Modulation Vectors + Per-Video Variable-Length Patchify](nano/wam/2026-06-25-wan21-adaln-modulation.md) — `Wan-Video/Wan2.1` (dit-block cross-repo)
- **2026-06-25** · infrastructure · [Psi0 的 SD3 风格联合自注意力：动作 token 和 VLA token 共享一个注意力块 / Psi0's SD3-Style Joint Self-Attention: Action Tokens and VLA Tokens Share One Attention Block](2026/06/2026-06-25-psi0-joint-vla-attention.md) — `physical-superintelligence-lab/Psi0` (trending)
- **2026-06-24** · diffusion · [DPM-Solver++(2M)：用历史预测做二阶修正的视频扩散采样器 / DPM-Solver++(2M): 2nd-Order Multistep Correction via History Tracking in Video Diffusion](2026/06/2026-06-24-cogvideo-dpmpp2m-sampler.md) — `THUDM/CogVideo` (tracked)
- **2026-06-24** · pytorch · [ColwiseParallel / RowwiseParallel：DTensor 张量并行里的 Shard(0) / Shard(1) 命名之谜 / ColwiseParallel / RowwiseParallel: The Shard(0) / Shard(1) Naming Puzzle in DTensor Tensor Parallelism](2026/06/2026-06-24-pytorch-colwise-rowwise-tensor-parallel.md) — `pytorch/pytorch`
- **2026-06-24** · huggingface · [AlignDevicesHook：70B 模型如何用 meta device 实现零显存加载 / AlignDevicesHook: How 70B Models Load with Zero GPU Memory via the Meta Device](2026/06/2026-06-24-accelerate-align-devices-hook.md) — `huggingface/accelerate`
- **2026-06-24** · vla · [Isaac-GR00T ActionChunk：relative vs delta 归一化 + SLERP 旋转插值 / Isaac-GR00T ActionChunk: Relative vs Delta Normalization + SLERP Rotation Interpolation](nano/vla/2026-06-24-groot-action-chunk-delta-relative.md) — `NVIDIA/Isaac-GR00T` (action-chunking cross-repo)
- **2026-06-24** · wam · [FlowDPMSolverMultistepScheduler：阶数自动升级的 ODE 求解器 / FlowDPMSolverMultistepScheduler: An Auto-Order-Escalating ODE Solver for Flow Matching](nano/wam/2026-06-24-wan21-flow-dpm-solver.md) — `Wan-Video/Wan2.1` (sampler-inference cross-repo)
- **2026-06-24** · diffusion · [ActionIDConstraintLogitsProcessor：一个 mask 把 LLM 的输出空间限制到动作词表 / ActionIDConstraintLogitsProcessor: One Mask That Constrains an LLM's Output Space to the Action Vocabulary](2026/06/2026-06-24-ud-vla-action-constraint-logits.md) — `OpenHelix-Team/Unified-Diffusion-VLA` (trending)
- **2026-06-23** · robotics · [可组合 VLA 数据变换管道：冻结 dataclass + z-score / 分位数归一化 / Composable VLA Data-Transform Pipeline: Frozen Dataclass + Z-score / Quantile Normalization](2026/06/2026-06-23-openpi-normalize-transform.md) — `Physical-Intelligence/openpi` (tracked)
- **2026-06-23** · pytorch · [Python `@overload` 精确类型收窄：`model[2]` → `Module`，`model[1:3]` → `Sequential` / Python `@overload` for Precise Type Narrowing: `model[2]` → `Module`, `model[1:3]` → `Sequential`](2026/06/2026-06-23-pytorch-sequential-overload-typing.md) — `pytorch/pytorch`
- **2026-06-23** · huggingface · [Krea2TextFusion：融合文本编码器所有隐藏层输出的跨层注意力 / Krea2TextFusion: Fusing All Text-Encoder Hidden-Layer Outputs via Cross-Attention](2026/06/2026-06-23-diffusers-krea2-text-fusion.md) — `huggingface/diffusers`
- **2026-06-23** · vla · [Wall-X ActionHead：DOF 掩码 + Beta 分布流匹配，跨机器人泛化的动作头 / Wall-X ActionHead: DOF Masking + Beta-Distribution Flow Matching for Cross-Embodiment VLA](nano/vla/2026-06-23-wall-x-dof-masked-flow-matching.md) — `huggingface/lerobot` (action-head-continuous cross-repo)
- **2026-06-23** · wam · [8 行广播构建视频帧分组因果注意力掩码 / Build Video-Frame-Grouped Causal Attention Masks in 8 Lines of Broadcasting](nano/wam/2026-06-23-fastwam-group-causal-attn-mask.md) — `yuantianyuan01/FastWAM` (dit-block cross-repo)
- **2026-06-23** · robotics · [MemoryVLA CogMemBank：ICLR 2026 认知记忆库 — 跨时间 Transformer 检索 + Gate 融合 + ToMe 整合 / MemoryVLA CogMemBank: ICLR 2026 Cognitive Memory Bank — Cross-Transformer Retrieval + GateFusion + ToMe Consolidation](2026/06/2026-06-23-memoryvla-cogmembank.md) — `shihao1895/MemoryVLA` (trending)
- **2026-06-22** · infrastructure · [Tensor 并行的两块积木：ColumnParallelLinear + RowParallelLinear / Tensor Parallelism's Two Building Blocks: ColumnParallelLinear + RowParallelLinear](2026/06/2026-06-22-flash-attn-tensor-parallel-linear.md) — `Dao-AILab/flash-attention` (tracked)
- **2026-06-22** · pytorch · [优化器藏进反向传播：`_apply_optimizer_in_backward` / The Optimizer Hidden Inside Backward: `_apply_optimizer_in_backward`](2026/06/2026-06-22-pytorch-optimizer-in-backward.md) — `pytorch/pytorch`
- **2026-06-22** · huggingface · [Transformers 的连续批处理三步核心：前缀缓存 + token 预算分割 + paged-attention 块分配 / Transformers Continuous Batching Core: Prefix Cache + Token-Budget Split + Paged-Attention Block Allocation](2026/06/2026-06-22-transformers-continuous-batching-scheduler.md) — `huggingface/transformers`
- **2026-06-22** · vla · [生产级 VLA 推理流水线：gRPC 策略服务器的 5 步内核 / Production VLA Inference Pipeline: the 5-Step Core of the gRPC Policy Server](nano/vla/2026-06-22-lerobot-policy-server-inference-pipeline.md) — `huggingface/lerobot` (inference-loop cross-repo)
- **2026-06-22** · wam · [时序 Tile VAE：无限长视频的 O(1) 显存编码 / Temporal-Tiled VAE: O(1) Memory Encoding for Arbitrarily Long Videos](nano/wam/2026-06-22-open-sora-hunyuanvae-temporal-tiling.md) — `hpcaitech/Open-Sora` (vae-encoder-decoder cross-repo)
- **2026-06-22** · infrastructure · [VLA 专属 Triton 注意力：前缀 + 后缀双区域 Softmax / VLA-Specific Triton Attention: Prefix + Suffix Two-Region Softmax](2026/06/2026-06-22-fluxvla-triton-prefix-suffix-softmax.md) — `FluxVLA/FluxVLA` (trending)
- **2026-06-21** · infrastructure · [nanoGPT 的优化器配置：一行判断分出"要衰减"和"不衰减" / nanoGPT's Optimizer Setup: One-Line Rule to Split Decay vs. No-Decay](2026/06/2026-06-21-nanogpt-configure-optimizers.md) — `karpathy/nanoGPT` (tracked)
- **2026-06-21** · pytorch · [`torch.func.grad`：把函数变成它自己的梯度函数 / `torch.func.grad`: Turn Any Function Into Its Own Gradient Function](2026/06/2026-06-21-pytorch-func-grad-transform.md) — `pytorch/pytorch`
- **2026-06-21** · huggingface · [VeRA：全模型共享一对冻结随机矩阵，每层只训练两个缩放向量 / VeRA: One Frozen Random Matrix Pair for the Whole Model, Two Scale Vectors Per Layer](2026/06/2026-06-21-peft-vera-shared-random-matrices.md) — `huggingface/peft`
- **2026-06-21** · vla · [SAC 高斯动作头：预测 (μ, σ)、重参数化采样、Tanh 压缩到 [-1,1] / SAC Gaussian Action Head: Predict (μ, σ), Reparameterize-Sample, Tanh-Squash to [-1,1]](nano/vla/2026-06-21-lerobot-gaussian-actor-sac.md) — `huggingface/lerobot` (action-head-continuous)
- **2026-06-21** · wam · [Mixture of Transformers：动作 expert 和视频 expert 在每一层共享注意力池 / Mixture of Transformers: Action Expert and Video Expert Share One Attention Pool at Every Layer](nano/wam/2026-06-21-fastwam-mot-mixed-attention.md) — `yuantianyuan01/FastWAM` (action-conditioning)
- **2026-06-21** · robotics · [NeRF 风格的 3D 位置嵌入：让 VLM 真正知道每个视觉 patch 在空间中的位置 / NeRF-Style 3D Position Embeddings: Giving the VLM Actual Spatial Awareness](2026/06/2026-06-21-spatialvla-ego3d-position-embedding.md) — `SpatialVLA/SpatialVLA` (trending)
- **2026-06-21** · diffusion · [VEnhancer 的 SDEdit 内核:60 行把"加噪→去噪"全讲完 / VEnhancer's SDEdit core: forward + reverse diffusion in 60 lines](2026/06/2026-06-21-venhancer-sdedit-gaussian-diffusion.md) — `Vchitect/VEnhancer` (tracked)
- **2026-06-21** · pytorch · [PyTorch 终于有 JAX 风格的无状态随机数了:Philox key / split / fold_in / normal_ / PyTorch ships JAX-style stateless PRNG: Philox key / split / fold_in / normal_](2026/06/2026-06-21-pytorch-stateless-philox-prng.md) — `pytorch/pytorch`
- **2026-06-21** · huggingface · [30 行 multimodal fusion:一个布尔 mask 把图像 embedding 缝进 token 流 / 30 lines of multimodal fusion: one boolean mask splices image embeddings into the token stream](2026/06/2026-06-21-nanovlm-mask-indexed-image-token-splice.md) — `huggingface/nanoVLM`
- **2026-06-21** · vla · [层级特征 × DiT 动作头:starVLA 怎么让 early DiT 看底层像素、late DiT 看高层语义 / Layer-wise features × DiT action head: how starVLA lets early DiT layers read low-level pixels while late layers read high-level semantics](nano/vla/2026-06-21-starvla-layerwise-flow-matching-action-head.md) — `starVLA/starVLA` (action-head-continuous)
- **2026-06-21** · wam · [Wan2.1 的 I2V 秘密:CLIP 的第 31 层特征,而不是 CLS pooling / Wan2.1's I2V secret: CLIP's block-31 features, not CLS pooling](nano/wam/2026-06-21-wan21-clip-spatial-tokens-i2v.md) — `Wan-Video/Wan2.1` (text-conditioning advanced I2V)
- **2026-06-21** · diffusion · [IC-LoRA 40 行:把参考视频 token 原封不动地拼进去,让 DiT 自己学对齐 / IC-LoRA in 40 lines: append clean reference tokens and let the DiT learn alignment on its own](2026/06/2026-06-21-ltx2-ic-lora-reference-conditioning.md) — `Lightricks/LTX-2` (trending)
- **2026-06-15** · robotics · [一个 meta device 怎么悄悄毁掉了 VLA 的 flow-matching 时间采样器 / How `torch.device("meta")` silently destroyed a VLA's flow-matching time sampler](2026/06/2026-06-15-isaac-groot-n1d7-meta-device-beta-fix.md) — `NVIDIA/Isaac-GR00T` (tracked)
- **2026-06-15** · pytorch · [PyTorch Inductor 现在会检查"地址表达式里的常数项"是否会溢出 int32 / PyTorch Inductor now inspects the *constant term* of fused address expressions for int32 overflow](2026/06/2026-06-15-pytorch-inductor-int32-const-overflow.md) — `pytorch/pytorch`
- **2026-06-15** · huggingface · [把 RwLock::read() 从"每个 pre-token 一次"降到"每次 encode 一次": tokenizers 在 88 核机器上跑赢 158% / Amortizing `RwLock::read()` from per-pre-token to per-call: +158% throughput on 88-thread aarch64](2026/06/2026-06-15-tokenizers-rwlock-tokenize-in-pretokenized.md) — `huggingface/tokenizers`
- **2026-06-15** · vla · [一份 80 行的 flow-matching action head:训练 + 推理一次讲完 / 80 lines of flow-matching action head: train and infer end-to-end](nano/vla/2026-06-15-vla-jepa-flow-matching-action-head.md) — `huggingface/lerobot` (action-head-continuous)
- **2026-06-15** · wam · [Action 不在 cross-attn 里,直接和帧 token 并排坐:VLA-JEPA 的 ActionConditionedVideoPredictor / Actions don't go into cross-attn — they sit alongside frame tokens: VLA-JEPA's `ActionConditionedVideoPredictor`](nano/wam/2026-06-15-vla-jepa-action-block-causal-mask.md) — `huggingface/lerobot` (action-conditioning)
- **2026-06-15** · robotics · [NVIDIA Lyra 的 Gumbel-Softmax 直通 top-k:200 行代码搞定"训练时可微、推理时硬 top-k" / NVIDIA Lyra's Gumbel-Softmax straight-through top-k: 200 lines that switch between differentiable training and hard inference](2026/06/2026-06-15-lyra-gumbel-softmax-token-pruning.md) — `nv-tlabs/lyra` (trending)
- **2026-06-14** · infrastructure · [DeepSeek-V3 的 ue8m0 act_quant:一句 ceil(log2(s)) 就把 FP8 scale 变成 Blackwell 原生 / DeepSeek-V3's ue8m0 act_quant: one ceil(log2(s)) makes the FP8 scale Blackwell-native](2026/06/2026-06-14-deepseek-v3-ue8m0-act-quant.md) — `deepseek-ai/DeepSeek-V3` (tracked)
- **2026-06-14** · pytorch · [PyTorch 把 Muon 写进官方了:40 行 Newton-Schulz 把梯度矩阵正交化 / PyTorch officially ships Muon — 40 lines of Newton-Schulz that orthogonalises the gradient matrix](2026/06/2026-06-14-pytorch-muon-newton-schulz.md) — `pytorch/pytorch`
- **2026-06-14** · huggingface · [TRL 把 RLHF 的"训练权重 → vLLM 推理"压成 56 行,FSDP1 和 FSDP2 各走一条路 / TRL squeezes RLHF's "shipping training weights into vLLM" into 56 lines — FSDP1 and FSDP2 take different routes](2026/06/2026-06-14-trl-fsdp-vllm-weight-sync.md) — `huggingface/trl`
- **2026-06-14** · vla · [GR00T 用的 Eagle2.5 投影器:NHWC 三步乱舞把视觉 token 砍掉四分之三 / GR00T's Eagle2.5 projector: three NHWC permutations chop 75 % of the vision tokens](nano/vla/2026-06-14-groot-eagle25-pixel-shuffle-projector.md) — `huggingface/lerobot` (modality-projector)
- **2026-06-14** · wam · [Open-Sora 把"T5 文本编码"做成 46 行,顺手解决了 TP 的对齐难题 / Open-Sora's 46-line "T5 text encoder" sneaks in a tensor-parallel-friendly alignment trick](nano/wam/2026-06-14-open-sora-hf-embedder-text-conditioning.md) — `hpcaitech/Open-Sora` (text-conditioning)
- **2026-06-14** · infrastructure · [Astra(ICLR 2026)的"动作专家混合":80 行让一个 DiT 同时驱动游戏、车、机械臂 / Astra's (ICLR 2026) Mixture of Action Experts: 80 lines let one DiT drive games, cars, and manipulators](2026/06/2026-06-14-astra-multimodal-action-moe.md) — `EternalEvan/Astra` (trending)
- **2026-06-13** · diffusion · [100 行写完一个 JEPA 世界模型 —— 完整的 encode → predict → rollout 合约 / 100 lines for a complete JEPA world model — the full encode → predict → rollout contract](2026/06/2026-06-13-le-wm-jepa-rollout.md) — `lucas-maes/le-wm` (tracked)
- **2026-06-13** · pytorch · [PyTorch 终于把"自带 FA + FA3 + FA4 + 你家自定义 attention"做成了插件 / PyTorch ships a real plugin system for "built-in FA + FA3 + FA4 + your custom attention"](2026/06/2026-06-13-pytorch-flash-attention-registry.md) — `pytorch/pytorch`
- **2026-06-13** · huggingface · [一句数学恒等式 = 加载期外科手术:Transformers 把 Conv3d patch-embed 在 load 时换成 Linear / A math identity becomes a load-time surgery: Transformers swaps Conv3d patch-embed for Linear at checkpoint load](2026/06/2026-06-13-transformers-conv3d-linear-fusion.md) — `huggingface/transformers`
- **2026-06-13** · vla · [pi0 PyTorch 的 6 行 flow-matching loss + 整个训练 step / pi0 PyTorch's 6-line flow-matching loss + complete training step](nano/vla/2026-06-13-openpi-pi0-pytorch-flow-matching-loss.md) — `Physical-Intelligence/openpi` (training-step)
- **2026-06-13** · wam · [FastWAM 的 50 行训练 while-loop:HF accelerate 让"梯度累积 + 多卡同步"在零分支语句下完成 / FastWAM's 50-line training while-loop: HF accelerate makes gradient accumulation + multi-process sync work without a single if-branch](nano/wam/2026-06-13-fastwam-accelerate-training-loop.md) — `yuantianyuan01/FastWAM` (training-loop)
- **2026-06-13** · diffusion · [NVIDIA flashdreams 的 `initialize_cache`:一次性 encode + 流式 VAE,这是交互式 AR 视频生成的设计核心 / NVIDIA flashdreams' `initialize_cache`: one-shot encode + streaming VAE — the design core of interactive AR video generation](2026/06/2026-06-13-flashdreams-interactive-ar-cache.md) — `NVIDIA/flashdreams` (trending)
- **2026-06-12** · robotics · [53 行的 FiLM 残差块 —— Diffusion Policy 的"条件注入"全在这 / 53 lines of FiLM residual block — the whole "conditional injection" of Diffusion Policy lives here](2026/06/2026-06-12-diffusion-policy-film-residual-block.md) — `real-stanford/diffusion_policy` (tracked)
- **2026-06-12** · pytorch · [数学恒等式当编译器优化:PyTorch Inductor 让 ConvTranspose2d 直接借用 backward-input 的 Triton kernel / Math identity as a compiler optimization: PyTorch Inductor lets ConvTranspose2d reuse the backward-input Triton kernel](2026/06/2026-06-12-pytorch-convtranspose-reuses-bwd-template.md) — `pytorch/pytorch`
- **2026-06-12** · huggingface · [HF datasets 接 Apache Iceberg:一场"提取可序列化视图"的精彩外科手术 / HF datasets meets Apache Iceberg: a clean "extract a picklable view" surgical operation](2026/06/2026-06-12-hf-datasets-iceberg-picklability.md) — `huggingface/datasets`
- **2026-06-12** · vla · [同一份 `infer()` 跑 JAX 和 PyTorch 两套 VLA:openpi 的 80 行统一推理器 / One `infer()` for both JAX and PyTorch VLAs: openpi's 80-line unified rollout wrapper](nano/vla/2026-06-12-openpi-policy-unified-jax-pytorch-infer.md) — `Physical-Intelligence/openpi` (inference-loop)
- **2026-06-12** · wam · [一份 130 行的完整 FlowMatchScheduler:把"训练时哪些 timestep 更重要"也包进去了 / A complete 130-line FlowMatchScheduler that also bundles "which timesteps matter more at training time"](nano/wam/2026-06-12-lingbot-flow-match-scheduler-with-training-weight.md) — `Robbyant/lingbot-va` (noise-scheduler)
- **2026-06-12** · robotics · [把"人手当机器人末端执行器"那篇 VITRA:50 行代码搞定异构动作的 masked diffusion loss / VITRA — the "human hand as a robot end-effector" paper: 50 lines handle masked diffusion loss for a heterogeneous action vector](2026/06/2026-06-12-vitra-masked-multi-component-diffusion-loss.md) — `microsoft/VITRA` (trending)
- **2026-06-11** · infrastructure · [vLLM 的 KV-cache "高水位线":只对新进队的请求收一笔押金,治好抢占抖动 / vLLM's KV-cache "watermark": charge admission rent only on newly-admitted requests to stop preemption thrash](2026/06/2026-06-11-vllm-kv-cache-watermark.md) — `vllm-project/vllm` (tracked)
- **2026-06-11** · pytorch · [PyTorch Inductor 把"全图拓扑排序"换成了"局部 BFS 搬运" — 18000 个节点中只动几十个 / PyTorch Inductor replaces a whole-graph topo sort with two tiny BFS helpers — out of 18 000 nodes it now moves only a few dozen](2026/06/2026-06-11-pytorch-surgical-fx-move.md) — `pytorch/pytorch`
- **2026-06-11** · huggingface · [Accelerate 把"动态 batch size"塞进了多卡 sharding:一招"循环填回初始 batch"让所有进程同步收尾 / Accelerate retrofits dynamic batch sizes into multi-process sharding via the classic "ring back to initial batches" trick](2026/06/2026-06-11-accelerate-dynamic-batch-sampler.md) — `huggingface/accelerate`
- **2026-06-11** · vla · [pi0-FAST 的 action tokenizer:把"连续动作"装进 PaliGemma 词表的尾部空槽 / pi0-FAST's action tokenizer: stuffing continuous actions into PaliGemma's reserved vocab tail](nano/vla/2026-06-11-lerobot-fast-action-tokenizer-paligemma.md) — `huggingface/lerobot` (action-tokenizer)
- **2026-06-11** · wam · [让 3D VAE "听不到未来":Open-Sora 的三个因果原语 / Making a 3D VAE deaf to the future: Open-Sora's three causal primitives](nano/wam/2026-06-11-open-sora-causal-3d-vae.md) — `hpcaitech/Open-Sora` (temporal-compression)
- **2026-06-11** · infrastructure · [LightX2V 把 Wan 视频模型的 FFN 从 7 次 kernel 启动压到 3 次:MXFP8 融合的完整教学版 / LightX2V fuses Wan video model FFN from 7 kernel launches down to 3: a textbook walk-through of MXFP8 fusion](2026/06/2026-06-11-lightx2v-mxfp8-ffn-fuse.md) — `ModelTC/LightX2V` (trending)
- **2026-06-10** · vla · [MEM 短期视觉记忆完整实现:用 4 招把"多帧观测压成单帧 token" / Implementing MEM's short-term visual memory: four moves that compress multi-frame observations into single-frame tokens](nano/vla/2026-06-10-mem-short-term-video-memory.md) — `physical-intelligence/mem` (short-term-observation-memory)
- **2026-06-10** · diffusion · [用世界模型当"想象器":dino_wm 的 71 行 CEM planner / Using the world model as an imagination engine: dino_wm's 71-line CEM planner](2026/06/2026-06-10-dino-wm-cem-planner.md) — `gaoyuezhou/dino_wm` (tracked)
- **2026-06-10** · pytorch · [PyTorch 把"一张 GPU 切成多张"写进了官方:92 行的 Green Context wrapper / PyTorch shipped "slice one GPU into many" to core — a 92-line Green Context wrapper](2026/06/2026-06-10-pytorch-cuda-green-contexts.md) — `pytorch/pytorch`
- **2026-06-09** · vla · [五种 VLA 的同一道题:image / language / state / action 怎么变成 action — OpenVLA、OFT、pi0-FAST、pi0、GR00T 全对照 / Five VLAs, one problem: turning image / language / state / action into actions — OpenVLA, OFT, pi0-FAST, pi0, GR00T side by side](nano/vla/2026-06-09-vla-five-models-multimodal-synthesis.md) — `Physical-Intelligence/openpi` (vlm-backbone-wiring)
- **2026-06-09** · vla · [GR00T-N1.7 数据流完整拆解:image / language / state / action 如何经 cross-attention DiT 变成 action / GR00T-N1.7 end-to-end data flow: image / language / state / action turning into action via cross-attention DiT](nano/vla/2026-06-09-groot-cross-attention-multimodal-fusion.md) — `NVIDIA/Isaac-GR00T` (vlm-backbone-wiring)
- **2026-06-09** · vla · [pi0 完整数据流:image / language / state / action 四模态如何流到最终 action / pi0 end-to-end data flow: how image / language / state / action turn into final action](nano/vla/2026-06-09-pi0-flow-matching-multimodal-fusion.md) — `Physical-Intelligence/openpi` (action-head-continuous)
- **2026-06-09** · vla · [Real-Time Chunking:让动作 chunk 之间的接缝消失的 130 行 / Real-Time Chunking: 130 lines that make the seams between action chunks disappear](nano/vla/2026-06-09-lerobot-rtc-action-chunking.md) — `huggingface/lerobot` (action-chunking)
- **2026-06-09** · wam · [Wan2.1 的 noise scheduler:同一根 flow-matching 时间轴,被 `time_shift` 重塑成"分辨率自适应" / Wan2.1's noise scheduler: one flow-matching time axis, reshaped by `time_shift` into a resolution-aware schedule](nano/wam/2026-06-09-wan21-flow-match-dynamic-shift.md) — `Wan-Video/Wan2.1` (noise-scheduler)
- **2026-06-09** · huggingface · [TRL 的 GOLD trainer:用"字节偏移"对齐两种不同的 tokenizer / TRL's GOLD trainer aligns two different tokenizers via byte offsets](2026/06/2026-06-09-trl-byte-offset-cross-tokenizer.md) — `huggingface/trl`
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
