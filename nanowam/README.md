# nanoWAM

Karpathy nanoGPT 风格的"从头实现 World-Action Model"工程项目,基于 **Wan-AI/Wan2.1-T2V-1.3B** 主干。

## 状态

🚧 工程刚启动,**当前在 Stage 0(sanity check)**。所有进度看 [`PROGRESS.md`](PROGRESS.md)。

## 开发协议

- **`nanowam/` 子目录自包含**,以后会迁到独立仓库;不引用本仓库 `nanowam/` 之外任何文件
- **只在 `nanowam` 分支开发**,不 PR 不 merge 到 main
- 任何新 Claude 对话开始前必须读 `PROGRESS.md` → `ARCHITECTURE.md` → `decisions/*.md` → `stage{N}_*/README.md`

## 目录

```
PROGRESS.md           ⭐ 当前进度、TODO、承接点(单一真相源)
ARCHITECTURE.md       数据流大图、设计原则
README.md             本文件
decisions/            关键决策日志(append-only)
shared/               跨阶段共享代码(VAE wrap、T5、data loader、flow matching)
configs/              wan21_1_3B.yaml 主干配置
stage0_sanity/        load 检查、VAE 重构、T2V 推理
stage1_video_only/    纯 video fine-tune
stage2_first_frame/   加首帧 latent 替换
stage3_action/        加 action token + 双 head
stage4_teacher_forcing/  加 cond 拷贝 + FlexAttn 三规则
stage5_inference/     KV cache + 闭环 server
scripts/              babysit / monitoring
```

## 借鉴的现成代码

User 本地 `/tmp/daily_code_cache/` 下应该有以下仓库的 clone(daily-code 主线 fetch 步骤会建出来):

- `lingbot_va/` — 主要参考(整体架构 + 双拷贝 + FlexAttn mask)
- `fastwam/` — `first_frame_latents` 替换、`video_attention_mask_mode` 三种 mask
- `dreamzero/` — Wan2.1 接 robot 数据的实战配置
- `wan2_1/` — 官方 DiT/VAE/T5 接口

迁仓后这些路径要 user 在新机器上重新建,**不能在代码里硬编码 `/tmp/daily_code_cache`**;应该通过 config yaml 注入。

## 怎么跟 Claude 协作

User 在网页跟 Claude 对话,本地 GPU 服务器跑训练的事 user 自己做。每次会话:

1. Claude:`git pull` + 读 PROGRESS.md → 完全 catch up
2. User 描述本次要做什么 / 反馈上次脚本跑出来的结果
3. Claude 改代码 / 写文档 / 给下一步指令
4. User 跑 / 贴结果
5. Claude 把结果记到 `stage{N}_*/NOTES.md`、更新 PROGRESS.md、push 到 `nanowam` 分支

Claude 永远不能跳过第 1 步("我记得我们上次……")。
