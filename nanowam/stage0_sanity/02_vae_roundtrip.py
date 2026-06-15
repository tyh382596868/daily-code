"""
Stage 0 — script 02: VAE round-trip check.

目标:拿一段 8 帧合成视频(渐变色条),过 Wan2.1 VAE encode 再 decode,算 PSNR。

通过标准:
  - PSNR > 30 dB
  - 否则 VAE 没装对(常见原因:把 Wan2.2 VAE 当 Wan2.1 用了)

用法:
  python 02_vae_roundtrip.py [--config ../configs/wan21_1_3B.yaml]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """两个 [-1, 1] 范围的 tensor,算 PSNR (dB)."""
    a = a.float().clamp(-1, 1)
    b = b.float().clamp(-1, 1)
    mse = (a - b).pow(2).mean().item()
    if mse < 1e-12:
        return 100.0
    # signal range = 2 (from -1 to 1)
    return 10 * torch.log10(torch.tensor(4.0 / mse)).item()


def make_synthetic_video(F: int = 8, H: int = 256, W: int = 320, device="cuda") -> torch.Tensor:
    """造一段简单可视化的合成视频:RGB 渐变 + 帧间小漂移。
    Returns: [1, 3, F, H, W] in [-1, 1].
    """
    f_idx = torch.arange(F, device=device).float().view(F, 1, 1, 1)  # (F,1,1,1)
    y = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1)
    x = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W)
    # R channel: 水平梯度;G: 垂直梯度;B: 时间方向漂移
    r = x.expand(F, 1, H, W)
    g = y.expand(F, 1, H, W)
    b = (f_idx / max(F - 1, 1) * 2 - 1).expand(F, 1, H, W)
    video = torch.cat([r, g, b], dim=1)              # (F, 3, H, W)
    video = video.permute(1, 0, 2, 3).unsqueeze(0)   # (1, 3, F, H, W)
    return video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).parent.parent / "configs" / "wan21_1_3B.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_path = cfg["model_path"]
    device = torch.device(cfg["device"])
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg["dtype"]]

    F = cfg["data"]["num_frames"]
    H = cfg["data"]["height"]
    W = cfg["data"]["width"]
    assert (F - 1) % cfg["vae"]["temporal_compression"] == 0, \
        f"num_frames - 1 必须能整除 {cfg['vae']['temporal_compression']}(VAE 时间压缩比),当前 F-1={F-1}"
    assert H % cfg["vae"]["spatial_compression"] == 0 and W % cfg["vae"]["spatial_compression"] == 0, \
        f"H,W 必须是 {cfg['vae']['spatial_compression']} 的倍数,当前 H={H} W={W}"

    print(f"==== VAE round-trip check ====")
    print(f"video shape = (1, 3, {F}, {H}, {W})")

    from diffusers import AutoencoderKLWan  # type: ignore
    vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype).to(device).eval()

    video = make_synthetic_video(F, H, W, device).to(dtype)
    print(f"input video range = [{video.min().item():.3f}, {video.max().item():.3f}]")

    with torch.no_grad():
        # encode
        enc = vae.encode(video).latent_dist
        latent = enc.sample()                                    # [1, z_dim, F', H', W']
        print(f"latent shape = {tuple(latent.shape)}")
        # decode
        decoded = vae.decode(latent).sample                      # [1, 3, F, H, W]
        print(f"decoded shape = {tuple(decoded.shape)}")

    metric = psnr(video, decoded)
    print(f"\nPSNR = {metric:.2f} dB")

    if metric > 30:
        print("✅ PASS — VAE 装对了,可以进 stage 1")
    elif metric > 20:
        print("⚠ MARGINAL — PSNR 偏低,确认 VAE 加载方式 / z_dim 设置")
    else:
        print("❌ FAIL — PSNR 太低,极大概率 VAE 配置错(Wan2.2 装成 Wan2.1?dtype 错?)")
        print("    把这个数字 + 上面所有 shape 贴回 chat,Claude debug")


if __name__ == "__main__":
    main()
