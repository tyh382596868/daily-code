# Decision 001: Use Wan-AI/Wan2.1-T2V-1.3B as the video DiT backbone

Date: 2026-06-15
Status: ✅ accepted

## Context

要做 nanoWAM,需要选一个视频生成模型当主干。候选档位:

| 档位 | 模型 | 参数 | 备注 |
|---|---|---:|---|
| 真 nano | facebookresearch/DiT-XL/2 | 675M | 图像 DiT,要自己扩时空 |
| 真 nano | Latte | ~600M | DiT-style video,小数据集预训练 |
| 轻量 | **Wan2.1-T2V-1.3B** | 1.3B | Wan 家族最小 |
| 轻量 | Open-Sora-Plan small | 220M-1.3B | 同等量级,但代码迁移成本高 |
| 中等 | CogVideoX-2B | 2B | 文生视频,diffusers 集成好 |
| 大 | Wan2.2-TI2V-5B | 5B | lingbot/fastwam/dreamzero 在用 |

## Decision

选 **`Wan-AI/Wan2.1-T2V-1.3B`**。

## Rationale

1. **代码迁移成本最低**:lingbot 的 `WanTransformer3DModel` 架构(`model.py:599-613`)就是 Wan 家族通用 DiT 模板。从 Wan2.2-5B 迁到 Wan2.1-1.3B 主要改 yaml 数字(`in_channels=16`、调小 `num_layers/heads/head_dim`),代码逻辑不动。
2. **VAE 跟着小**:Wan2.1 VAE z_dim=16(Wan2.2 是 48),latent token 数小 3 倍,训练显存省。
3. **预训练对得上**:Wan 家族都是 T2V/I2V flow-matching,直接 fine-tune 进 WAM 设定不用换 loss 公式。
4. **生态印证**:dreamzero 同时支持 Wan2.1-I2V-14B 和 Wan2.2-TI2V-5B(`wan_flow_matching_action_tf.py:23-24`),证明 Wan2.1 在 robotic WAM 上是被验证过的家族。

## "T2V 没有 I2V 接口" 不是问题

虽然 1.3B 是纯 T2V,**WAM 的首帧条件不依赖底座的 I2V 接口**,而是上层包装:

- lingbot 推理(`wan_va_server.py:291-295`):`noisy_latents[:, :, 0:1] = latent_cond[:, :, 0:1]` + `timesteps[0:1] *= 0`,把首帧位置直接覆盖成 clean latent,根本没调底座 I2V 通道
- fastwam 训练(`fastwam.py:340-343, 467-468`):`first_frame_latents = input_latents[:, :, 0:1]` + 训练时把首帧位置替换成 clean latent。**没用底座 CLIP image encoder 通道**

所以 Wan2.1-T2V-1.3B 完全够用,而且更干净(没有用不上的 CLIP image encoder 通道)。

## Trade-offs accepted

- **没有图像高层语义通道**:如果将来需要,自己接一个独立 CLIP encoder 拼到 text context 末尾即可(dreamzero `wan_video_dit.py:1790-1794` 就是这么做的)。这跟底座是不是 I2V 无关。
- **1.3B 比 Wan2.2-5B 表现略差**:接受。nanoWAM 目标是跑通 + 教学清晰,不是 SOTA。

## References

- 本仓库对话:user 上一次明确选择 "Wan2.1-T2V-1.3B"
- HF: https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B
- fastwam 配置参考:`/tmp/daily_code_cache/fastwam/configs/model/fastwam.yaml`
- lingbot 架构参考:`/tmp/daily_code_cache/lingbot_va/wan_va/modules/model.py:595-650`
