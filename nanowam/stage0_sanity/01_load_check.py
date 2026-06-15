"""
Stage 0 — script 01: load check.

目标:分别 load Wan2.1-T2V-1.3B 的三个子模块(DiT、VAE、T5),打印每个的:
  - 参数量
  - 加载后 GPU 显存增量(MB)

通过标准:
  - 三个都能成功 load
  - 三个加起来 < user GPU 显存
  - 把数字贴回 chat,Claude 写进 NOTES.md 并据此设 stage 1 训练超参

用法:
  cd nanowam/stage0_sanity
  python 01_load_check.py [--config ../configs/wan21_1_3B.yaml]

注意:
  - Wan2.1-T2V-1.3B 的具体加载接口要 user 在跑的时候根据实际 HF 仓库的 README 调整
  - 当前脚本写的是"按 diffusers 标准布局"的假设,如果失败 user 反馈给 Claude,Claude 切到官方 wan2_1 仓库的脚本
  - **这只是骨架**,真实可能要改 import 路径和类名
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import yaml


def gpu_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.memory_allocated(device) / 1024 / 1024


def count_params(module: torch.nn.Module) -> tuple[int, int]:
    """Return (total, trainable) param counts."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).parent.parent / "configs" / "wan21_1_3B.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_path = cfg["model_path"]
    dtype_str = cfg["dtype"]
    device = torch.device(cfg["device"])

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"model_path not found: {model_path}\n"
            f"先去 HF 下载: hf snapshot-download {cfg['model_repo']} --local-dir {model_path}\n"
            f"或者改 configs/wan21_1_3B.yaml 的 model_path 字段"
        )

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_str]

    print(f"==== nanoWAM Stage 0 — load check ====")
    print(f"device       = {device}")
    print(f"dtype        = {dtype}")
    print(f"model_path   = {model_path}")
    print()

    mem_start = gpu_memory_mb(device)
    print(f"[mem] baseline before loading: {mem_start:.0f} MB")

    # ============================================================
    # 1. VAE
    # ============================================================
    print("\n--- loading VAE ---")
    # TODO(user/claude): Wan2.1 的 VAE 类应该是 diffusers.AutoencoderKLWan 或类似
    # 如果 diffusers 还没集成,user 反馈,Claude 切到 wan2_1 官方仓库的 WanVAE 类
    try:
        from diffusers import AutoencoderKLWan  # type: ignore
        vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype).to(device).eval()
    except ImportError:
        raise RuntimeError(
            "diffusers.AutoencoderKLWan 不可用。\n"
            "  方案 1: pip install -U diffusers (>=0.32?)\n"
            "  方案 2: 改用 wan2_1 官方仓库的 WanVAE 类(改本脚本 import)\n"
            "  user 把报错贴回 chat,Claude 决定走哪条"
        )
    total, _ = count_params(vae)
    mem_after_vae = gpu_memory_mb(device)
    print(f"VAE params = {total/1e6:.1f} M")
    print(f"[mem] after VAE  : {mem_after_vae:.0f} MB  (+{mem_after_vae - mem_start:.0f})")
    print(f"VAE z_dim = {getattr(vae.config, 'z_dim', getattr(vae.config, 'latent_channels', '?'))}")
    assert (
        getattr(vae.config, "z_dim", getattr(vae.config, "latent_channels", None)) == cfg["vae"]["z_dim"]
    ), f"VAE z_dim 不是 {cfg['vae']['z_dim']},检查是不是装成了 Wan2.2 VAE"

    # ============================================================
    # 2. T5 text encoder (umt5-xxl)
    # ============================================================
    print("\n--- loading T5 text encoder ---")
    # Wan2.1 用 umt5-xxl,通常通过 transformers.T5EncoderModel 加载
    from transformers import T5EncoderModel, AutoTokenizer  # type: ignore
    t5 = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=dtype).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    total, _ = count_params(t5)
    mem_after_t5 = gpu_memory_mb(device)
    print(f"T5  params = {total/1e6:.1f} M")
    print(f"[mem] after T5   : {mem_after_t5:.0f} MB  (+{mem_after_t5 - mem_after_vae:.0f})")

    # ============================================================
    # 3. DiT transformer
    # ============================================================
    print("\n--- loading DiT ---")
    try:
        from diffusers import WanTransformer3DModel  # type: ignore
        dit = WanTransformer3DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=dtype).to(device).eval()
    except ImportError:
        raise RuntimeError(
            "diffusers.WanTransformer3DModel 不可用。同 VAE 处理:升级 diffusers 或切官方类。"
        )
    total, _ = count_params(dit)
    mem_after_dit = gpu_memory_mb(device)
    print(f"DiT params = {total/1e6:.1f} M  (期望 ~1.3B)")
    print(f"[mem] after DiT  : {mem_after_dit:.0f} MB  (+{mem_after_dit - mem_after_t5:.0f})")

    # ============================================================
    # 把 DiT 的 architecture 关键参数打印出来,user 把这些数字填进 wan21_1_3B.yaml
    # ============================================================
    print("\n==== DiT architecture (填进 configs/wan21_1_3B.yaml 的 dit 段)====")
    for k in [
        "num_attention_heads", "attention_head_dim", "num_layers",
        "ffn_dim", "in_channels", "out_channels", "patch_size", "text_dim",
    ]:
        v = getattr(dit.config, k, "<missing>")
        print(f"  dit.{k}: {v}")

    print("\n==== 完成 ✅ ====")
    print(f"总显存占用: {mem_after_dit:.0f} MB")
    print("把上面打印的所有数字贴回 chat,Claude 写进 stage0_sanity/NOTES.md。")


if __name__ == "__main__":
    main()
