# nanoWAM PROGRESS  ⭐ Single Source of Truth

> **每次新对话第一件事:Claude 必须读这份文件,跟着 "下一步要做什么" 接着干。**
> **每次会话结束:Claude 必须更新这份文件,然后 commit + push 到 `nanowam` 分支(不 PR、不 merge)。**

Last updated: 2026-06-15 (created)
Current stage: **0 — Sanity check**
Current branch: `nanowam`

---

## 🛑 HARD RULES — 永远不破

1. **`nanowam` 分支永远不 PR、不 merge 进 `main`。** daily-code 主线(日更教学笔记 + 自动合 main)和 nanoWAM 工程线(这个分支)互相隔离。Claude 不许调 `mcp__github__create_pull_request` / `mcp__github__merge_pull_request` 把 nanowam 的内容推上 main。
2. **`nanowam/` 子目录必须自包含。** 不引用 daily-code 仓库里 `nanowam/` 之外任何文件、不写 `../2026/...` 之类的相对路径、不依赖 `INDEX.md` / `topics/` / `.config/`。判定标准:`cp -r nanowam/ /tmp/new-repo/ && cd /tmp/new-repo && python -c "import nanowam"` 后整套东西仍然能用。
3. **会话结束只在 `nanowam` 分支 `git push origin nanowam`**。不动 main、不动 claude/* 分支。
4. **不要在 `nanowam/` 之外建任何文件。** 所有产物(代码、文档、配置、脚本、笔记)都在 `nanowam/` 里。

---

## Why this file exists

User 在网页跟 Claude 对话,本地 GPU 服务器上跑训练的事得 user 自己做。
切 chat / 换账号 / 清 history 都会让 Claude 失忆。
**解决方案:把所有"全局规划 + 当前进度 + 关键决策 + 已知 bug"全写进仓库本身。**
新 Claude `git pull` 一下读这份文件,就能精确接续。

---

## Bootstrap protocol — 任何新对话都按这个走

新 Claude 上来必须:

1. `git fetch origin && git checkout nanowam && git pull`
2. 读 `nanowam/PROGRESS.md` (本文件) — current stage、剩余 TODO、已知 bug、上次承接点
3. 读 `nanowam/ARCHITECTURE.md` — 设计大图
4. 读 `nanowam/decisions/*.md` — 所有关键决策(为什么 Wan2.1、为什么分阶段、为什么 noisy_cond_prob=0.5 …)
5. 读 `nanowam/stage{N}_*/README.md` 当中 `N = current stage` 的那一份 — 当前阶段的验收标准
6. **然后**才回答 user 的新消息

会话过程中要做的:
- 改了代码 → 改文件 → user 跑 → 把 user 反馈结果(loss 曲线、报错、显存数字)记到 `nanowam/stage{N}_*/NOTES.md`
- 做了新决定 → 在 `nanowam/decisions/` 加一份 `NNN-<topic>.md`
- 完成一项 TODO → 在本文件的 "Stage N TODO" 节里打勾 `[x]`

会话结束前 Claude 必须:
- 更新本文件的 `## Current stage` 节(进度 + 卡点 + 下一步)
- `git add nanowam/ && git commit -m "wip: <stage N> <topic>" && git push origin nanowam`
- 在最后一条回复里告诉 user push 了什么、要 user 下一步做什么(跑哪个脚本、看哪个文件)

---

## High-level plan (永远不会改的总目标)

复刻并简化 LingBot-VA(自家代码 = `/tmp/daily_code_cache/lingbot_va/`)的核心结构,用 **Wan-AI/Wan2.1-T2V-1.3B** 做主干,做一个能跑闭环的 **nanoWAM**。

5 个阶段,**每阶段独立可跑、独立验证**,跑通且 loss 收敛后才进下一阶段:

| 阶段 | 目标 | 验收标准 | 状态 |
|---|---|---|---|
| **0. sanity** | Wan2.1-T2V-1.3B 能 load + VAE encode/decode 重构得对 | 重构 PSNR > 30 dB,T2V 推理出一段视频 | 🟡 in progress |
| **1. video_only** | 自己 robot 数据上微调纯 video,T2V 收敛 | video MSE loss 单调下降,fix prompt 出图视觉一致 | ⬜ not started |
| **2. first_frame** | 加首帧 latent 替换 + first_frame_causal mask | 给一张图,生成的视频开头跟图一致,后续帧连贯 | ⬜ not started |
| **3. action** | 加 action token 序列 + 双 head,简单 causal mask | video loss 和 action loss 都下降,action 维度量级合理 | ⬜ not started |
| **4. teacher_forcing** | 加 cond 拷贝 + FlexAttn 三规则 mask + diffusion forcing | mask 矩阵 unit test 通过,loss 不退化 | ⬜ not started |
| **5. inference** | KV cache + 半步去噪 + 闭环 server | 单 chunk 推理 < 100ms,接真相机能闭环 | ⬜ not started |

⚠ 阶段次序非常重要。**不要跳级**。loss 跑歪了往往是上一阶段没真收敛,而不是当前阶段代码错。

---

## Current stage:  Stage 0 — Sanity check

### Stage 0 TODO

- [ ] User 在本地服务器上 `git clone` 仓库,checkout `nanowam` 分支
- [ ] User: **建 conda 环境用 Python 3.11**(不要用 base 的 3.13),按 `nanowam/stage0_sanity/SETUP.md` 装 lingbot 同款版本(torch==2.9.0+cu126, diffusers==0.36.0, transformers==4.55.2, flash_attn)
- [ ] User: 从 HF 下载 `Wan-AI/Wan2.1-T2V-1.3B`,记本地路径填进 `nanowam/configs/wan21_1_3B.yaml`
- [ ] User: 跑 `nanowam/stage0_sanity/00_env_check.py`,**把 stdout 贴回 chat**。期望走路径 A(diffusers)
- [ ] User: 跑 `nanowam/stage0_sanity/01_load_check.py`,验证 DiT / VAE / T5 能都 load 上,记录显存占用
- [ ] User: 跑 `nanowam/stage0_sanity/02_vae_roundtrip.py`,验证 VAE 重构 PSNR > 30 dB
- [ ] User: 跑 `nanowam/stage0_sanity/03_t2v_inference.py`,用官方 prompt 生成一段视频,看是否合理
- [ ] User 把 3 个脚本的 stdout / 显存数字 / 视频截图贴回 chat,Claude 把结果记到 `nanowam/stage0_sanity/NOTES.md`
- [ ] Claude 根据上面结果决定 stage 1 的 batch_size / chunk_size / image resolution
- [ ] Stage 0 关闭,切到 stage 1

### Stage 0 已知卡点 / 注意事项

- Wan2.1 VAE 的 z_dim 是 **16**(不是 Wan2.2 的 48),lingbot 默认值是 48 不能直接搬过来用 — 见 `decisions/002-wan21-vs-wan22.md`
- Wan2.1-T2V-1.3B 的 T5 是 `umt5-xxl`,光 T5 自己就 ~4.7B,T5 用 fp16/bf16 加载就好,不参与训练
- 如果 user 是 24G 显存的 4090,batch=1 + F=16 + 720p 大概率 OOM,要降到 480p 或 F=8

### Stage 0 承接点 — 下次对话 user 应该报告什么

User 跑完上面 3 个脚本后,贴出:
1. 每个组件的显存占用(单独 load DiT、VAE、T5 各占多少)
2. VAE 重构 PSNR
3. T2V 推理一段视频(给个截图 / 一段描述就行)
4. 服务器 GPU 型号和显存大小

然后 Claude 据此定 stage 1 的训练配置。

---

## File map (常用文件位置,新 Claude 速查)

```
nanowam/
├── PROGRESS.md          ⭐ 本文件
├── ARCHITECTURE.md      数据流大图 + 设计原则
├── decisions/           关键决策日志(append-only)
│   ├── 001-pick-wan21-t2v-1.3b.md
│   ├── 002-wan21-vs-wan22-vae.md
│   └── ...
├── shared/              跨阶段共享 utilities
│   ├── vae_wrapper.py
│   ├── text_encoder.py
│   ├── data_lerobot.py
│   └── flow_matching.py
├── configs/
│   └── wan21_1_3B.yaml  Wan2.1-T2V-1.3B 路径 + 训练超参
├── stage{N}_*/
│   ├── README.md        当前阶段的任务、验收标准、参考代码出处
│   ├── NOTES.md         user 跑出来的实际数字、loss 曲线、报错
│   ├── train.py / xx.py 这一阶段的核心代码
│   └── config.yaml      阶段专属配置覆盖
└── scripts/
    └── babysit.sh       (optional) 远程 ssh 看 loss
```

参考代码出处(都是 user 本地已有的或可 clone 的):

| 出处 | 用在 nanoWAM 哪里 |
|---|---|
| `lingbot_va/wan_va/modules/model.py` | 整体架构、`_input_embed`、AdaLN block(stage 1+) |
| `lingbot_va/wan_va/train.py` | `_add_noise` 双拷贝、训练 loop(stage 4) |
| `lingbot_va/wan_va/wan_va_server.py` | KV cache + 闭环推理(stage 5) |
| `fastwam/src/fastwam/models/wan22/wan_video_dit.py:473-507` | `build_video_to_video_mask` 三种模式(stage 2) |
| `fastwam/src/fastwam/models/wan22/fastwam.py:340-468` | `first_frame_latents` 替换机制(stage 2) |
| `Wan-Video/Wan2.1` 官方仓库 | DiT 权重格式、VAE、T5 接口 |

---

## Decisions log (一句话索引,详细看 decisions/*.md)

| ID | 决策 | 日期 |
|---|---|---|
| 001 | 主干用 Wan2.1-T2V-1.3B,不用 Wan2.2-5B(更小、家族兼容) | 2026-06-15 |
| 002 | VAE 用 Wan2.1 自带 z_dim=16,不用 Wan2.2 z_dim=48 | 2026-06-15 |
| 003 | 分 5 阶段串行,不一上来就堆完整 lingbot 架构 | 2026-06-15 |
| 004 | Stage 1-3 用简单 causal mask,Stage 4 才上 FlexAttn | 2026-06-15 |
| 005 | Action 用 lingbot 的 5D `[B, dim, F, τ, 1]` 壳子接口(复用 video 管线) | 2026-06-15 |

---

## Open questions (待 user 决定)

- [ ] User 用什么 robot 数据集?LIBERO / RoboTwin / 自录 / DROID?(决定 dataloader 怎么写)
- [ ] User 的 robot action 维度 `action_dim` 是多少?(决定 `action_embedder` 输入)
- [ ] User 想最终部署在什么硬件?(决定 stage 5 是否一定要做 FDM-grounded async)
- [ ] User 是否需要 video 也作为最终输出(true WAM),还是只要 action 输出就行(那 stage 1-2 都可以裁简)?

---

## 长期 TODO (远期,等阶段 5 跑通再考虑)

- Mixed-t cond 训练(lingbot 的 noisy_cond_prob=0.5)
- FDM-grounded async inference(Algorithm 2 Branch B)
- Variable chunk size training
- 数据并行 / FSDP 训大模型
- Eval pipeline(成功率、动作平滑度)
