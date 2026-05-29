# WAM — build your own

Notes tagged `wam`, newest first. Daily teaching points sourced from
[dreamzero](https://github.com/dreamzero0/dreamzero),
[lingbot-va](https://github.com/Robbyant/lingbot-va),
[FastWAM](https://github.com/yuantianyuan01/FastWAM),
[Wan2.1](https://github.com/Wan-Video/Wan2.1),
and [Open-Sora](https://github.com/PKU-YuanGroup/Open-Sora).

Each entry teaches **one component** of a World Action Model and maps it
explicitly to its role in a from-scratch `nanoWAM` / production WAM build.
Components covered include: VAE / latent encoder, DiT-style backbone block,
noise scheduler / flow matcher, conditioning logic (text / image / action),
classifier-free guidance, training loop, inference sampler, temporal compression,
action-frame fusion.

| Date | Component | Title | Repo |
|------|-----------|-------|------|
| 2026-05-29 | attention masking (action-frame fusion) | [Seven mask predicates compose into one FlexAttention BlockMask for video + action](../2026/05/2026-05-29-wam-lingbot-flex-mask-compose.md) | [Robbyant/lingbot-va](https://github.com/Robbyant/lingbot-va) |
| 2026-05-28 | noise scheduler / flow matcher | [A complete rectified-flow scheduler in 90 lines](../2026/05/2026-05-28-wam-dreamzero-flow-match-scheduler.md) | [dreamzero0/dreamzero](https://github.com/dreamzero0/dreamzero) |

<!-- entries auto-appended by daily-code-teach, newest first -->
