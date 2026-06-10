# WAM — build your own

Notes tagged `wam`, newest first. Daily teaching points sourced from
[dreamzero](https://github.com/dreamzero0/dreamzero),
[lingbot-va](https://github.com/Robbyant/lingbot-va),
[FastWAM](https://github.com/yuantianyuan01/FastWAM),
[Wan2.1](https://github.com/Wan-Video/Wan2.1),
and [Open-Sora](https://github.com/hpcaitech/Open-Sora).

Each entry teaches **one component** of a World Action Model and maps it
explicitly to its role in a from-scratch `nanoWAM` / production WAM build.

| Date | Component | Title | Repo |
|------|-----------|-------|------|
| 2026-06-10 | classifier-free-guidance (cross-repo) | [28 行 DiT LabelEmbedder:CFG 的"教学版"实现 / DiT's 28-line LabelEmbedder: the textbook implementation of classifier-free guidance](../nano/wam/2026-06-10-dit-label-embedder-cfg.md) | [facebookresearch/DiT](https://github.com/facebookresearch/DiT) |
| 2026-06-09 | noise-scheduler | [Wan2.1 的 noise scheduler:同一根 flow-matching 时间轴,被 `time_shift` 重塑成"分辨率自适应" / Wan2.1's noise scheduler: one flow-matching time axis, reshaped by `time_shift` into a resolution-aware schedule](../nano/wam/2026-06-09-wan21-flow-match-dynamic-shift.md) | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) |
| 2026-06-08 | patchify-positional | [一份 14 行的 EmbedND:文本 / 图像 / 视频共用一个 RoPE 模块 / 14 lines of EmbedND: text, image, and video share one RoPE module](../nano/wam/2026-06-08-open-sora-embed-nd-generalized-rope.md) | [hpcaitech/Open-Sora](https://github.com/hpcaitech/Open-Sora) |
| 2026-06-08 | vae-encoder-decoder | [没有学习参数也能 ×8 上采样:Open-Sora 的 3D pixel-shuffle / Upsample 8× with zero learnable params: Open-Sora's 3D pixel-shuffle](../nano/wam/2026-06-08-open-sora-pixel-shuffle-3d.md) | [hpcaitech/Open-Sora](https://github.com/hpcaitech/Open-Sora) |
| 2026-06-08 | dit-block | [Wan2.1 的 WanAttentionBlock:DiT block 的生产级长相 / Wan2.1's WanAttentionBlock: what a production-grade DiT block actually looks like](../nano/wam/2026-06-08-wan21-attention-block-production.md) | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) |
| 2026-06-07 | dit-block | [Flux / SD3 的双流 DiT 块:图像和文本各自一套 QKV,只在 attention 那一步合体 / Flux / SD3's dual-stream DiT block: image and text get their own QKV, and they only meet at attention](../nano/wam/2026-06-07-open-sora-mmdit-double-stream.md) | [hpcaitech/Open-Sora](https://github.com/hpcaitech/Open-Sora) |
| 2026-06-05 | dit-block | [同一个 DiT 骨架,两种条件注入方式:GR00T 的 cross-attn 变体 / Same DiT skeleton, two conditioning strategies: GR00T's cross-attention variant](../nano/wam/2026-06-05-isaac-groot-dit-cross-attn.md) | [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) |
| 2026-06-04 | action-conditioning | [GR00T fuses action and flow-time into one small MLP](../nano/wam/2026-06-04-isaac-groot-action-encoder.md) | [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) |
| 2026-06-03 | sampler-inference | [FastWAM 的"动作 sampler 跳过 video forward":WAM 推理的 prompt-prefill 类比 / FastWAM's action sampler skips the video expert each step — WAM-inference's answer to "prompt prefill + decode"](../nano/wam/2026-06-03-fastwam-action-prefill-cache.md) | [yuantianyuan01/FastWAM](https://github.com/yuantianyuan01/FastWAM) |
| 2026-06-02 | nano | [一次 attention 调用,两条互不共享权重的流 / One attention call, two streams that never share weights](../nano/wam/2026-06-02-open-sora-mmdit-double-stream.md) | [hpcaitech/Open-Sora](https://github.com/hpcaitech/Open-Sora) |
| 2026-06-01 | dit-block | [DoubleStreamBlock:两条流、各自做归一化和投影、然后只共享一次 attention / DoubleStreamBlock: two streams, separate norms and projections, fused by one shared attention](../nano/wam/2026-06-01-open-sora-double-stream-block.md) | [hpcaitech/Open-Sora](https://github.com/hpcaitech/Open-Sora) |
| 2026-06-01 | dit-block | [把 DiT 一分为二:Open-Sora 的 MMDiT DoubleStreamBlock / Splitting DiT in two: Open-Sora's MMDiT DoubleStreamBlock](../nano/wam/2026-06-01-open-sora-mmdit-double-stream.md) | [hpcaitech/Open-Sora](https://github.com/hpcaitech/Open-Sora) |
| 2026-05-31 | DiT-block | [MM-DiT 把"文字"和"图像"做成两条对等的 stream / MM-DiT treats text and image as two peer streams in one attention](../nano/wam/2026-05-31-opensora-mmdit-double-stream.md) | [hpcaitech/Open-Sora](https://github.com/hpcaitech/Open-Sora) |
| 2026-05-29 | temporal-compression | [Resample's feat_cache lets a 3D VAE process video of arbitrary length, one chunk at a time](../nano/wam/2026-05-29-wan21-resample-streaming-cache.md) | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) |
| 2026-05-29 | sampler-inference | [60 lines of denoise loop is the entire WAM "generate"](../nano/wam/2026-05-29-wan21-denoise-loop.md) | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) |
| 2026-05-29 | training-loop | [Two main lines of a training step: add noise, then weight the loss](../nano/wam/2026-05-29-lingbot-add-noise-loss.md) | [Robbyant/lingbot-va](https://github.com/Robbyant/lingbot-va) |
| 2026-05-29 | classifier-free-guidance | [CFG is two forwards and one weighted sum](../nano/wam/2026-05-29-wan21-classifier-free-guidance.md) | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) |
| 2026-05-29 | text-conditioning | [Text conditioning is 25 lines of cross-attention](../nano/wam/2026-05-29-wan21-text-cross-attention.md) | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) |
| 2026-05-29 | patchify-positional | [Splitting RoPE three ways: frame, height, width](../nano/wam/2026-05-29-wan21-3d-rope.md) | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) |
| 2026-05-29 | vae-encoder-decoder | [One line of padding turns nn.Conv3d into a causal 3D conv](../nano/wam/2026-05-29-wan21-vae-causal-conv3d.md) | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) |
| 2026-05-29 | — | [Seven mask predicates compose into one FlexAttention BlockMask for video + action](../nano/wam/2026-05-29-lingbot-flex-mask-compose.md) | [Robbyant/lingbot-va](https://github.com/Robbyant/lingbot-va) |
| 2026-05-28 | — | [A complete rectified-flow scheduler in 90 lines](../nano/wam/2026-05-28-dreamzero-flow-match-scheduler.md) | [dreamzero0/dreamzero](https://github.com/dreamzero0/dreamzero) |
