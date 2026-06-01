# WAM — build your own

Notes tagged `wam`, newest first. Daily teaching points sourced from
[dreamzero](https://github.com/dreamzero0/dreamzero),
[lingbot-va](https://github.com/Robbyant/lingbot-va),
[FastWAM](https://github.com/yuantianyuan01/FastWAM),
[Wan2.1](https://github.com/Wan-Video/Wan2.1),
and [Open-Sora](https://github.com/hpcaitech/Open-Sora).

Each entry teaches **one component** of a World Action Model and maps it
explicitly to its role in a from-scratch `nanoWAM` / production WAM build.
Components covered include: VAE / latent encoder, DiT-style backbone block,
noise scheduler / flow matcher, conditioning logic (text / image / action),
classifier-free guidance, training loop, inference sampler, temporal compression,
action-frame fusion.

| Date | Component | Title | Repo |
|------|-----------|-------|------|
| 2026-06-01 | DiT block (MMDiT cross-repo variant) | [Splitting DiT in two: Open-Sora's MMDiT DoubleStreamBlock](../nano/wam/2026-06-01-open-sora-mmdit-double-stream.md) | [hpcaitech/Open-Sora](https://github.com/hpcaitech/Open-Sora) |
| 2026-05-29 | temporal compression | [Resample's feat_cache lets a 3D VAE process video of arbitrary length, one chunk at a time](../nano/wam/2026-05-29-wan21-resample-streaming-cache.md) | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) |
| 2026-05-29 | sampler / inference loop | [60 lines of denoise loop is the entire WAM "generate"](../nano/wam/2026-05-29-wan21-denoise-loop.md) | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) |
| 2026-05-29 | training loop | [Two main lines of a training step: add noise, then weight the loss](../nano/wam/2026-05-29-lingbot-add-noise-loss.md) | [Robbyant/lingbot-va](https://github.com/Robbyant/lingbot-va) |
| 2026-05-29 | classifier-free guidance | [CFG is two forwards and one weighted sum](../nano/wam/2026-05-29-wan21-classifier-free-guidance.md) | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) |
| 2026-05-29 | text conditioning (cross-attn) | [Text conditioning is 25 lines of cross-attention](../nano/wam/2026-05-29-wan21-text-cross-attention.md) | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) |
| 2026-05-29 | patchify + 3D RoPE | [Splitting RoPE three ways: frame, height, width](../nano/wam/2026-05-29-wan21-3d-rope.md) | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) |
| 2026-05-29 | VAE / latent encoder-decoder | [One line of padding turns nn.Conv3d into a causal 3D conv](../nano/wam/2026-05-29-wan21-vae-causal-conv3d.md) | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) |
| 2026-05-29 | attention masking (action-frame fusion) | [Seven mask predicates compose into one FlexAttention BlockMask for video + action](../nano/wam/2026-05-29-lingbot-flex-mask-compose.md) | [Robbyant/lingbot-va](https://github.com/Robbyant/lingbot-va) |
| 2026-05-28 | noise scheduler / flow matcher | [A complete rectified-flow scheduler in 90 lines](../nano/wam/2026-05-28-dreamzero-flow-match-scheduler.md) | [dreamzero0/dreamzero](https://github.com/dreamzero0/dreamzero) |

<!-- entries auto-appended by daily-code-teach, newest first -->
