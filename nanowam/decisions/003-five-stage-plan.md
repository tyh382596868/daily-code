# Decision 003: Five-stage incremental build plan

Date: 2026-06-15
Status: ✅ accepted

## Context

要从零搭一个 LingBot-VA 等价的 nanoWAM。一上来堆完整架构(双拷贝 + FlexAttn 三规则 + 双 head + KV cache + FDM-grounded async)风险极高 —— 一旦 loss 不下降,根本不知道是 mask 错了、是 action init 错了、是 video 不收敛、还是 cond 拷贝逻辑错。

## Decision

分 5 个阶段串行,每阶段独立可跑、独立验证,跑通且 loss 收敛后才进下一阶段:

| 阶段 | 加什么 | 不加什么 | 验收 |
|---|---|---|---|
| 0 sanity | 验证 Wan2.1 能 load + VAE 重构 + T2V 推理 | 不训练任何东西 | PSNR > 30, T2V 出图合理 |
| 1 video_only | robot 数据上做纯 video flow-matching fine-tune | action / 首帧 / cond / FlexAttn | video MSE 单调下降 |
| 2 first_frame | 首帧 latent 替换 + first_frame_causal mask | action / cond / FlexAttn | 给图能生连贯续帧 |
| 3 action | 加 action token + 双 head + 简单 causal | cond 拷贝 / FlexAttn | video & action loss 都下降 |
| 4 teacher_forcing | cond 拷贝 + FlexAttn 三规则 + diffusion forcing | KV cache | mask 单测过,loss 不退化 |
| 5 inference | KV cache + 半步去噪 + 闭环 server | async / FDM-grounded | 单 chunk < 100ms,闭环跑通 |

## Rationale

1. **debugging 颗粒度**:每阶段只新增一件事,任何 loss 不收敛都能定位到"刚加的那个"
2. **可观测**:每阶段都有自己的验收标准 + NOTES.md 记录实际数字
3. **Karpathy 风格**:像 nanoGPT 那样,每个 stage 是独立可跑的脚本,不一上来抽 `BaseTrainer`,等所有 stage 跑通了再 refactor
4. **风险递减**:Stage 1 跑通后,user 已经有一个能用的 video fine-tuning 框架,即使后面 4 阶段失败也不会前功尽弃

## Trade-offs accepted

- **代码会有少量重复**:每个 stage 自己的 `train.py` 都会重复一些样板代码。接受。等 5 个 stage 都跑通之后才 refactor 出共享 utilities,不是开始就抽
- **总耗时长**:5 个阶段串行 ≈ 1 个月。但每阶段都有可交付物,中途叫停损失最小
- **从 stage 3 到 stage 4 的 jump 较大**(加 cond + FlexAttn 两件事),stage 4 内部可能要再切分

## What doesn't fit this plan

如果 user 后来决定走不同的路线(比如改用 dreamzero 的 block-wise KV cat 而不是 lingbot 的 FlexAttn mask),stage 4 设计要重新做。stage 1-3 不受影响。

## References

- nanoGPT 风格:Karpathy 把 transformer 从最简到完整版分了 train.py、model.py、prepare.py 几个独立可跑文件
- LingBot-VA paper §3.3:teacher forcing + noisy history augmentation 是分开两个 concept,stage 3→4 的拆分对应这个
