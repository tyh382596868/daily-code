# Hugging Face

Notes tagged `huggingface`, newest first. Daily teaching points from the HF main
libraries — `transformers`, `diffusers`, `accelerate`, `datasets`, `peft`, `trl`,
`tokenizers`, `nanoVLM`.

| Date | Title | Repo |
|------|-------|------|
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
