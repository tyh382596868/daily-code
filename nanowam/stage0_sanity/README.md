# Stage 0 — Sanity Check

> 目标:在 user 的 GPU 服务器上验证 Wan2.1-T2V-1.3B 能 load 并跑通最基础的 inference,为后续 stage 定基线。**不训练任何东西。**

## 入口脚本(按顺序跑)

| 脚本 | 做什么 | 通过标准 |
|---|---|---|
| `01_load_check.py` | 分别 load DiT、VAE、T5,打印每个的显存占用 | 三个都能 load 上,加在一起 < user GPU 显存 |
| `02_vae_roundtrip.py` | 拿一段 8 帧视频,VAE encode → decode,算 PSNR | PSNR > 30 dB(否则 VAE 配错) |
| `03_t2v_inference.py` | 给一个 prompt,跑官方 T2V 流程出 16 帧视频 | 输出视频画面合理(不是全黑/纯噪声) |

跑法:

```bash
cd nanowam/stage0_sanity
# 先填好 ../configs/wan21_1_3B.yaml 里的 model_path
python 01_load_check.py
python 02_vae_roundtrip.py
python 03_t2v_inference.py
```

## 完成后 user 要贴回 chat 的内容

1. **`01_load_check.py` 的 stdout**(显存占用数字)
2. **`02_vae_roundtrip.py` 的 PSNR 数字**
3. **`03_t2v_inference.py` 的输出视频路径**(以及一句话描述生成内容)
4. **GPU 型号 + 显存大小**(`nvidia-smi`)

把这些贴出来,下一次会话 Claude 把它写进 `NOTES.md`,然后据此设 Stage 1 的训练超参(batch、F、resolution)。

## 已知卡点

- **Wan2.1-T2V-1.3B 加载方式**:用 `diffusers` 的 `WanPipeline.from_pretrained` 还是 Wan 官方仓库的脚本?当前脚本骨架按 `diffusers` 思路写(更通用),如果失败再切官方。
- **z_dim=16**:VAE 加 decoder 之前一定确认 `vae.config.z_dim == 16`,不要错装成 Wan2.2 的 48。
- **T5 (umt5-xxl) 显存**:bf16 大概 9 GB,fp32 18 GB,**默认 bf16**。
- **官方权重格式**:Wan2.1-T2V-1.3B 可能是 `.safetensors` 也可能是 `.pth`,脚本要兼容。

## 别做什么

- 不训练任何东西(这是 stage 1 的事)
- 不实现 action / 首帧 / cond / FlexAttn(那是 stage 2-4 的事)
- 不写优化器、不写 dataloader、不写 trainer 类
- VAE 重构 PSNR 没过 30 就停,不要继续往下走

## 完成后

更新 `PROGRESS.md` 的 "Stage 0 TODO" 全部打勾,把 "Current stage" 改成 `1 — Video-only fine-tune`,把 `stage1_video_only/README.md` 写出来,然后 push。
