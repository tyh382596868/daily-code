# Stage 0 NOTES

> User 跑完 3 个脚本后,Claude 把结果记到这里。

## Environment

- GPU: <TODO 待 user 填,例如 "RTX 4090 24GB" 或 "H100 80GB">
- Driver / CUDA: <TODO>
- PyTorch: <TODO>
- diffusers version: <TODO>

## Results

### 01_load_check.py

- VAE 显存: <TODO> MB
- T5  显存: <TODO> MB
- DiT 显存: <TODO> MB
- DiT 参数: <TODO> M
- DiT architecture 数字(填回 configs/wan21_1_3B.yaml):
  - num_attention_heads = <TODO>
  - attention_head_dim  = <TODO>
  - num_layers          = <TODO>
  - ffn_dim             = <TODO>

### 02_vae_roundtrip.py

- latent shape: <TODO>
- PSNR: <TODO> dB
- 结论: <PASS / MARGINAL / FAIL>

### 03_t2v_inference.py

- inference 耗时: <TODO> s
- 峰值显存:     <TODO> MB
- 输出视频路径: stage0_sanity/out_t2v_smoke.mp4
- 视觉质量(主观): <TODO 一句话描述>

## Decisions implied for Stage 1

- Stage 1 batch_size: <TODO,等数据回来后定>
- Stage 1 num_frames: <TODO>
- Stage 1 resolution: <TODO>
- Stage 1 grad checkpointing: <TODO,看 DiT 显存>
