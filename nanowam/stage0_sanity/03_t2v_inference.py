"""
Stage 0 — script 03: Wan2.1-T2V-1.3B inference smoke test.

目标:给一个 prompt,跑完整 T2V 流程,出一段 16 帧视频。

通过标准:
  - 没有崩
  - 输出 mp4 / gif 画面合理(不是全黑 / 纯噪声 / 全灰)
  - 记录单次推理耗时 + 显存峰值

用法:
  python 03_t2v_inference.py [--prompt "..."] [--config ../configs/wan21_1_3B.yaml]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import yaml


DEFAULT_PROMPT = (
    "A robotic arm slowly picks up a red block from a wooden table, "
    "studio lighting, photorealistic, top-down view."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).parent.parent / "configs" / "wan21_1_3B.yaml")
    ap.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    ap.add_argument("--num_frames", type=int, default=16, help="生成视频帧数")
    ap.add_argument("--steps", type=int, default=25, help="去噪步数")
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "out_t2v_smoke.mp4")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_path = cfg["model_path"]
    device = torch.device(cfg["device"])
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg["dtype"]]

    print(f"==== Wan2.1-T2V-1.3B smoke inference ====")
    print(f"prompt    = {args.prompt}")
    print(f"frames    = {args.num_frames}, steps = {args.steps}, guidance = {args.guidance}")
    print(f"output    = {args.out}")

    try:
        from diffusers import WanPipeline  # type: ignore
    except ImportError:
        raise RuntimeError(
            "diffusers.WanPipeline 不可用。\n"
            "  方案 1: 升级 diffusers\n"
            "  方案 2: 改用 Wan 官方仓库 (/tmp/daily_code_cache/wan2_1) 的推理脚本\n"
            "  user 把报错贴回 chat"
        )

    pipe = WanPipeline.from_pretrained(model_path, torch_dtype=dtype)
    pipe.to(device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.time()
    with torch.inference_mode():
        out = pipe(
            prompt=args.prompt,
            num_frames=args.num_frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            height=cfg["data"]["height"],
            width=cfg["data"]["width"],
        )
    dt = time.time() - t0

    frames = out.frames[0]  # list[PIL.Image] or numpy array depending on pipeline
    print(f"\ninference took {dt:.1f} s")
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        print(f"peak GPU mem  = {peak:.0f} MB")

    # save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        from diffusers.utils import export_to_video  # type: ignore
        export_to_video(frames, str(args.out), fps=cfg["data"]["fps"])
        print(f"✅ saved video to {args.out}")
    except Exception as e:
        print(f"⚠ video export failed: {e}")
        print(f"  把第一帧保存成 PNG 给你看一眼能不能用")
        try:
            frames[0].save(args.out.with_suffix(".png"))
            print(f"  saved first frame -> {args.out.with_suffix('.png')}")
        except Exception as e2:
            print(f"  连 PNG 都存不了: {e2}")
            print(f"  frames type: {type(frames)}, len: {len(frames) if hasattr(frames, '__len__') else '?'}")

    print("\n==== 完成 ✅ ====")
    print("把推理耗时 + 显存峰值 + 视频路径贴回 chat,Claude 写进 NOTES.md。")


if __name__ == "__main__":
    main()
