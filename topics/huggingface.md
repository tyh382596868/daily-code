# Hugging Face

Notes tagged `huggingface`, newest first. Daily teaching points from the HF main
libraries — `transformers`, `diffusers`, `accelerate`, `datasets`, `peft`, `trl`,
`tokenizers`, `nanoVLM`.

| Date | Title | Repo |
|------|-------|------|
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
