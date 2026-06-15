"""
Stage 0 — script 00: 环境探测,决定后续脚本走 diffusers 路线还是官方 wan 路线。

Run this FIRST. 把 stdout 贴回 chat,Claude 据此决定 stage 0 后续脚本走哪条:

  - 走 diffusers 路径:01/02/03 用 diffusers 的 AutoencoderKLWan / WanTransformer3DModel / WanPipeline
  - 走官方 wan 路径 :01/02/03 用 wan.WanT2V + wan.configs.WAN_CONFIGS
  - 都不行 :让 user 升级 diffusers 或 clone https://github.com/Wan-Video/Wan2.1

Usage:
  python 00_env_check.py
"""

from __future__ import annotations

import importlib
import sys


def check_import(module_name: str, item: str | None = None) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(module_name)
        ver = getattr(mod, "__version__", "?")
        if item is not None:
            if not hasattr(mod, item):
                return False, f"{module_name} {ver} 但没有 {item}"
            return True, f"{module_name} {ver}, has {item}"
        return True, f"{module_name} {ver}"
    except ImportError as e:
        return False, f"{module_name}: not installed ({e})"


def main():
    print("==== nanoWAM Stage 0 env check ====")
    print(f"python = {sys.version.split()[0]}")
    print()

    # 基础依赖
    print("--- 基础依赖 ---")
    for mod in ["torch", "transformers", "accelerate", "einops", "yaml", "PIL"]:
        ok, msg = check_import(mod)
        print(f"  {'✓' if ok else '✗'} {msg}")

    # PyTorch + CUDA
    try:
        import torch
        print(f"\n  torch.cuda.is_available = {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  cuda     = {torch.version.cuda}")
            print(f"  device   = {torch.cuda.get_device_name(0)}")
            mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  GPU mem  = {mem:.1f} GB")
    except Exception as e:
        print(f"  torch CUDA 探测失败: {e}")

    # ============================================================
    # 路径 A: diffusers 集成
    # ============================================================
    print("\n--- 路径 A: diffusers 集成(lingbot/fastwam 同款,首选)---")
    diffusers_ok = True
    for cls in ["WanPipeline", "WanTransformer3DModel", "AutoencoderKLWan"]:
        ok, msg = check_import("diffusers", cls)
        print(f"  {'✓' if ok else '✗'} diffusers.{cls} {'✓' if ok else f'— {msg}'}")
        diffusers_ok &= ok

    # ============================================================
    # 路径 B: 官方 wan 包
    # ============================================================
    print("\n--- 路径 B: 官方 wan 包(备选)---")
    wan_ok = True
    ok, msg = check_import("wan")
    print(f"  {'✓' if ok else '✗'} {msg}")
    wan_ok &= ok
    if ok:
        for item in ["WanT2V", "configs"]:
            ok2, msg2 = check_import("wan", item)
            print(f"  {'✓' if ok2 else '✗'} wan.{item} {'✓' if ok2 else f'— {msg2}'}")
            wan_ok &= ok2
        try:
            from wan.configs import WAN_CONFIGS  # type: ignore
            available = list(WAN_CONFIGS.keys())
            print(f"  WAN_CONFIGS 可用键: {available}")
            if "t2v-1.3B" in available:
                print("  ✓ 't2v-1.3B' 配置存在")
            else:
                print("  ⚠ 没有 't2v-1.3B' 配置")
        except Exception as e:
            print(f"  ⚠ 读 WAN_CONFIGS 失败: {e}")

    # ============================================================
    # 决定
    # ============================================================
    print("\n==== 推荐路径 ====")
    if diffusers_ok:
        print("✅ 走路径 A(diffusers)。可以直接跑后续 01/02/03 脚本。")
    elif wan_ok:
        print("⚠ diffusers 不够新,走路径 B(官方 wan 包)。")
        print("  user 把这份输出贴回 chat,Claude 改 01/02/03 用 wan.WanT2V 接口。")
    else:
        print("❌ 两条路都不通。选一个动作:")
        print("  方案 1: pip install -U 'diffusers>=0.32'  (推荐,跟 lingbot 同款)")
        print("  方案 2: git clone https://github.com/Wan-Video/Wan2.1 && cd Wan2.1 && pip install -e .")
        print("         (然后 'import wan' 应该能用,后续 Claude 改脚本走路径 B)")


if __name__ == "__main__":
    main()
