# Hugging Face

Notes tagged `huggingface`, newest first. Daily teaching points from the HF main
libraries — `transformers`, `diffusers`, `accelerate`, `datasets`, `peft`, `trl`,
`tokenizers`, `nanoVLM`.

| Date | Title | Repo |
|------|-------|------|
| 2026-07-22 | [TRL GRPO：训练器里把生成后端也变成可插拔模块 / TRL GRPO: Make the Generation Backend Pluggable Inside the Trainer](../2026/07/2026-07-22-trl-grpo-vllm-generation.md) | huggingface/trl |
| 2026-07-21 | [Transformers DynamicLayer：KV cache 默认就是沿时间维追加 / Transformers DynamicLayer: The Default KV Cache Appends Along Time](../2026/07/2026-07-21-transformers-dynamic-cache-layer.md) | huggingface/transformers |
| 2026-07-20 | [PEFT INC LoRA：只给支持的量化 Linear 换适配层 / PEFT INC LoRA: Dispatch Only Supported Quantized Linear Layers](../2026/07/2026-07-20-peft-inc-lora-dispatch.md) | huggingface/peft |
| 2026-07-19 | [Datasets map：流式样本也能攒成 batch 再变换 / Datasets map: Streamed Examples Can Still Be Batched Before Transform](../2026/07/2026-07-19-datasets-mapped-examples-iterable.md) | huggingface/datasets |
| 2026-07-17 | [Diffusers FlowMatch：每个 token 可以有自己的 timestep / Diffusers FlowMatch: Each Token Can Carry Its Own Timestep](../2026/07/2026-07-17-diffusers-flowmatch-per-token-timestep.md) | huggingface/diffusers |
| 2026-07-16 | [Accelerate GradientAccumulationPlugin：累积梯度也要定义同步合同 / Accelerate GradientAccumulationPlugin: Gradient Accumulation Needs a Sync Contract](../2026/07/2026-07-16-accelerate-gradient-accumulation-plugin.md) | huggingface/accelerate |
| 2026-07-15 | [Transformers DETR：无 mask 时用 arange 保住 dtype / Transformers DETR: Use arange Without a Mask to Preserve dtype](../2026/07/2026-07-15-transformers-detr-sine-position-embedding.md) | huggingface/transformers |
| 2026-07-14 | [nanoVLM generate：多模态预填一次，后面只解码新 token / nanoVLM generate: Multimodal Prefill Once, Then Decode New Tokens](../2026/07/2026-07-14-nanovlm-multimodal-prefill-decode.md) | huggingface/nanoVLM |
| 2026-07-13 | [tokenizers BPE trainer：用堆维护下一次最值得合并的 pair / tokenizers BPE Trainer: Use a Heap for the Next Best Pair](../2026/07/2026-07-13-tokenizers-bpe-trainer-heap.md) | huggingface/tokenizers |
| 2026-07-12 | [TRL preference collator：把 chosen/rejected 拼成同一个 batch / TRL Preference Collator: Put Chosen and Rejected in One Batch](../2026/07/2026-07-12-trl-preference-collator.md) | huggingface/trl |
| 2026-07-10 | [Transformers no-repeat ngram：把历史短语变成下一 token 禁止表 / Transformers No-Repeat N-Gram: Turn History into a Next-Token Ban List](../2026/07/2026-07-10-transformers-no-repeat-ngram.md) | huggingface/transformers |
| 2026-07-09 | [PEFT PSOFT OrthLayer：用 skew 矩阵生成可训练正交旋转 / PEFT PSOFT OrthLayer: Train an Orthogonal Rotation from a Skew Matrix](../2026/07/2026-07-09-peft-psoft-orth-layer.md) | huggingface/peft |
| 2026-07-08 | [Datasets 随机轮转数据源：流式混合也要可恢复 / Datasets Random Source Cycling: Streaming Mixtures Must Be Resumable](../2026/07/2026-07-08-datasets-randomly-cycling-sources.md) | huggingface/datasets |
| 2026-07-07 | [Accelerate tied parameters：先找共享权重，再把断掉的引用接回去 / Accelerate Tied Parameters: Find Shared Weights, Then Retie Broken References](../2026/07/2026-07-07-accelerate-find-retie-tied-parameters.md) | huggingface/accelerate |
| 2026-07-06 | [Diffusers zero-terminal SNR：把最后一步真的推到纯噪声 / Diffusers Zero-Terminal SNR: Make the Last Step Truly Pure Noise](../2026/07/2026-07-06-diffusers-zero-terminal-snr.md) | huggingface/diffusers |
| 2026-07-05 | [Transformers DynamicLayer：KV cache 就是沿时间维拼接 / Transformers DynamicLayer: A KV Cache Is Concatenation Along Time](../2026/07/2026-07-05-transformers-dynamic-kv-layer.md) | huggingface/transformers |
| 2026-07-03 | [nanoVLM projector：用 pixel shuffle 少传视觉 token / nanoVLM Projector: Use Pixel Shuffle to Send Fewer Vision Tokens](../2026/07/2026-07-03-nanovlm-modality-projector.md) | huggingface/nanoVLM |
| 2026-07-02 | [tokenizers 并行开关：同一个 iterator 可串行也可并行 / tokenizers Parallelism Switch: One Iterator, Serial or Parallel](../2026/07/2026-07-02-tokenizers-maybe-parallel-iterator.md) | huggingface/tokenizers |
| 2026-07-01 | [TRL 的 PEFT adapter EMA teacher：不用复制整模型的自蒸馏 / TRL PEFT Adapter EMA Teacher: Self-Distillation Without Copying the Whole Model](../2026/07/2026-07-01-trl-peft-adapter-ema-teacher.md) | huggingface/trl |
| 2026-06-30 | [LoRA-GA 初始化：用一次梯度 SVD 给 adapter 指方向 / LoRA-GA Init: Use One Gradient SVD to Aim the Adapter](../2026/06/2026-06-30-peft-loraga-svd-init.md) | huggingface/peft |
| 2026-06-26 | [`split_dataset_by_node`：一个 API 背后藏着两种完全不同的分布式数据切分策略 / `split_dataset_by_node`: One API, Two Fundamentally Different Distribution Strategies Hidden Inside](../2026/06/2026-06-26-datasets-split-by-node.md) | huggingface/datasets |
| 2026-06-25 | [nanoVLM GQA：prefill/decode 统一路径 + `is_causal` 精确谓词 / nanoVLM GQA: Unified Prefill/Decode Path + Precise `is_causal` Predicate](../2026/06/2026-06-25-nanovlm-gqa-kvcache.md) | huggingface/nanoVLM |
| 2026-06-24 | [AlignDevicesHook：70B 模型如何用 meta device 实现零显存加载 / AlignDevicesHook: How 70B Models Load with Zero GPU Memory via the Meta Device](../2026/06/2026-06-24-accelerate-align-devices-hook.md) | huggingface/accelerate |
| 2026-06-23 | [Krea2TextFusion：融合文本编码器所有隐藏层输出的跨层注意力 / Krea2TextFusion: Fusing All Text-Encoder Hidden-Layer Outputs via Cross-Attention](../2026/06/2026-06-23-diffusers-krea2-text-fusion.md) | huggingface/diffusers |
| 2026-06-22 | [Transformers 的连续批处理三步核心：前缀缓存 + token 预算分割 + paged-attention 块分配 / Transformers Continuous Batching Core: Prefix Cache + Token-Budget Split + Paged-Attention Block Allocation](../2026/06/2026-06-22-transformers-continuous-batching-scheduler.md) | huggingface/transformers |
| 2026-06-21 | [VeRA：全模型共享一对冻结随机矩阵，每层只训练两个缩放向量 / VeRA: One Frozen Random Matrix Pair for the Whole Model, Two Scale Vectors Per Layer](../2026/06/2026-06-21-peft-vera-shared-random-matrices.md) | huggingface/peft |
| 2026-06-21 | [30 行 multimodal fusion:一个布尔 mask 把图像 embedding 缝进 token 流 / 30 lines of multimodal fusion: one boolean mask splices image embeddings into the token stream](../2026/06/2026-06-21-nanovlm-mask-indexed-image-token-splice.md) | huggingface/nanoVLM |
| 2026-06-15 | [把 RwLock::read() 从"每个 pre-token 一次"降到"每次 encode 一次": tokenizers 在 88 核机器上跑赢 158% / Amortizing `RwLock::read()` from per-pre-token to per-call: +158% throughput on 88-thread aarch64](../2026/06/2026-06-15-tokenizers-rwlock-tokenize-in-pretokenized.md) | huggingface/tokenizers |
| 2026-06-14 | [TRL 把 RLHF 的"训练权重 → vLLM 推理"压成 56 行,FSDP1 和 FSDP2 各走一条路 / TRL squeezes RLHF's "shipping training weights into vLLM" into 56 lines — FSDP1 and FSDP2 take different routes](../2026/06/2026-06-14-trl-fsdp-vllm-weight-sync.md) | huggingface/trl |
| 2026-06-13 | [一句数学恒等式 = 加载期外科手术:Transformers 把 Conv3d patch-embed 在 load 时换成 Linear / A math identity becomes a load-time surgery: Transformers swaps Conv3d patch-embed for Linear at checkpoint load](../2026/06/2026-06-13-transformers-conv3d-linear-fusion.md) | huggingface/transformers |
| 2026-06-12 | [HF datasets 接 Apache Iceberg:一场"提取可序列化视图"的精彩外科手术 / HF datasets meets Apache Iceberg: a clean "extract a picklable view" surgical operation](../2026/06/2026-06-12-hf-datasets-iceberg-picklability.md) | huggingface/datasets |
| 2026-06-11 | [Accelerate 把"动态 batch size"塞进了多卡 sharding:一招"循环填回初始 batch"让所有进程同步收尾 / Accelerate retrofits dynamic batch sizes into multi-process sharding via the classic "ring back to initial batches" trick](../2026/06/2026-06-11-accelerate-dynamic-batch-sampler.md) | huggingface/accelerate |
| 2026-06-10 | [训练完才动手:PEFT 把"切除 LoRA 入侵维度"做成了一个 140 行的后处理 / After training, then surgery: PEFT ships "remove LoRA intruder dimensions" as a 140-line post-hoc step](../2026/06/2026-06-10-peft-intruder-dimension.md) | huggingface/peft |
| 2026-06-09 | [TRL 的 GOLD trainer:用"字节偏移"对齐两种不同的 tokenizer / TRL's GOLD trainer aligns two different tokenizers via byte offsets](../2026/06/2026-06-09-trl-byte-offset-cross-tokenizer.md) | huggingface/trl |
| 2026-06-08 | [把困扰 Llama 移植半年的 RoPE 重排压成两行 view + transpose / The two-line `view` + `transpose` that fixes Llama's RoPE port nightmare](../2026/06/2026-06-08-transformers-permute-for-rope.md) | huggingface/transformers |
| 2026-06-08 | [第一个 block 的残差几乎没变?那就跳过剩下所有 block / If the first block's residual barely moved, skip every other block](../2026/06/2026-06-08-diffusers-first-block-cache.md) | huggingface/diffusers |
| 2026-06-08 | [一个 step 调两次模型:Diffusers 的 Heun 二阶 flow-match 采样 / Two model calls per step: diffusers' Heun 2nd-order flow-match sampler](../2026/06/2026-06-08-diffusers-flow-match-heun-step.md) | huggingface/diffusers |
| 2026-06-07 | [diffusers 怎么用一根 CUDA stream + pinned CPU 镜像把 30 GB 模型塞进 24 GB GPU / How diffusers fits a 30 GB diffusion model on a 24 GB GPU with one CUDA stream and pinned CPU mirrors](../2026/06/2026-06-07-diffusers-group-offloading.md) | huggingface/diffusers |
| 2026-06-05 | [APG:把 CFG 的更新拆成平行 + 正交分量,只压缩平行那块 / APG: split the CFG update into parallel + orthogonal components, shrink only the parallel one](../2026/06/2026-06-05-diffusers-apg-projected-guidance.md) | huggingface/diffusers |
| 2026-06-04 | [diffusers ships a commit-by-confidence scheduler for masked-diffusion LMs](../2026/06/2026-06-04-diffusers-block-refinement-scheduler.md) | huggingface/diffusers |
| 2026-06-03 | [First-Block Cache: 拿首块残差当"风向标",其余 DiT 块直接跳过 / First-Block Cache: use the first DiT block's residual as a weathervane, skip every block in between](../2026/06/2026-06-03-diffusers-first-block-cache.md) | huggingface/diffusers |
| 2026-06-02 | [两个 timestep embedding + 一个 gate = 任意步长扩散 / Two timestep embeddings + one gate = any-step diffusion](../2026/06/2026-06-02-diffusers-anyflow-dual-timestep.md) | huggingface/diffusers |
| 2026-06-01 | [First Block Cache:跑完第一块,如果"看起来差不多"就跳过剩下的整 DiT / First Block Cache: run block 0, and if it "looks similar enough" skip the rest of the DiT](../2026/06/2026-06-01-diffusers-first-block-cache.md) | huggingface/diffusers |
| 2026-06-01 | [FlowMapEulerDiscreteScheduler:把"任意步采样"塞进一个 Euler 实现里 / FlowMapEulerDiscreteScheduler: any-step sampling, in one Euler `step`](../2026/06/2026-06-01-diffusers-flow-map-euler.md) | huggingface/diffusers |
| 2026-05-31 | [Flow Matching Euler 的核心其实只有一行 / The heart of a flow-matching Euler step is one line](../2026/05/2026-05-31-diffusers-flow-match-euler-step.md) | huggingface/diffusers |
| 2026-05-29 | [nanoVLM trades 256 image tokens for 64 fat tokens via pixel shuffle](../2026/05/2026-05-29-nanovlm-pixel-shuffle-projector.md) | huggingface/nanoVLM |
| 2026-05-28 | [A closure that re-ties weights after FSDP2 silently breaks them](../2026/05/2026-05-28-accelerate-fsdp2-weight-retie.md) | huggingface/accelerate |
| 2026-05-27 | [PEFT's LoRA forward: one line of addition is the whole algorithm](../2026/05/2026-05-27-huggingface-peft-lora-forward.md) | huggingface/peft |
