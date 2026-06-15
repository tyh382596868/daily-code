# nanoWAM SETUP

⚠ Python 版本和包版本和 lingbot-va 的官方推荐对齐,不要瞎改。任何"版本能不能换成 X"的问题,先确认跑过再说。

## 版本基线(直接抄 lingbot-va `pyproject.toml` + `README.md`)

| 项 | 版本 | 不能动的理由 |
|---|---|---|
| Python | **3.11**(也可 3.10/3.12) | lingbot 字面允许 3.10-3.13;flash_attn 在 3.13 prebuilt wheel 经常缺,**实测最稳 3.11** |
| torch | 2.9.0 (cu126) | lingbot 官方 |
| torchvision | 0.24.0 | |
| torchaudio | 2.9.0 | |
| diffusers | **0.36.0** | Wan 集成版本(0.32 之前没有,0.36 是 lingbot 验证过的) |
| transformers | 4.55.2 | |
| numpy | 1.26.4 (<2) | 兼容性 |
| flash_attn | latest, `--no-build-isolation` | lingbot DiT 用到 |
| CUDA | 12.6 | torch 2.9 wheel 对应的 |

## 完整安装(从空 conda base 开始)

```bash
# 1. 建环境
conda create -n nanowam python=3.11 -y
conda activate nanowam

# 2. 先确认机器 CUDA(torch wheel 必须对得上)
nvidia-smi | head -4     # 找 "CUDA Version: X.Y"
# 如果是 12.6+ 走下面;12.4 把 cu126 换 cu124;12.1 整体降到 torch 2.5

# 3. PyTorch (CUDA 12.6 wheel,lingbot 同款)
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/cu126

# 4. ML 栈
pip install \
    diffusers==0.36.0 \
    transformers==4.55.2 \
    accelerate \
    "numpy==1.26.4" \
    einops easydict pyyaml \
    opencv-python pillow imageio "imageio[ffmpeg]" \
    "huggingface_hub[cli]" \
    ftfy safetensors

# 5. flash_attn(可能源码编译,几十分钟)
pip install flash-attn --no-build-isolation

# 6. 验证
cd nanowam/stage0_sanity
python 00_env_check.py
```

## 下载 Wan2.1-T2V-1.3B 权重(跟环境安装并行)

```bash
# 推荐放持久盘,不要放 /tmp
hf download Wan-AI/Wan2.1-T2V-1.3B \
    --local-dir /PATH/TO/persistent/models/Wan2.1-T2V-1.3B

# 然后改 nanowam/configs/wan21_1_3B.yaml 的 model_path 字段
```

## 装完该看到的 00 输出

```
✓ torch 2.9.0
✓ torch.cuda.is_available = True
✓ diffusers.WanPipeline ✓
✓ diffusers.WanTransformer3DModel ✓
✓ diffusers.AutoencoderKLWan ✓
==== 推荐路径 ====
✅ 走路径 A(diffusers)。
```

如果 `WanPipeline` 还是 ✗,说明 diffusers 0.36 没集成或者类名变了,
**贴 00 输出回 chat 让 Claude 切路径 B**(走官方 wan 包)。

## 常见踩坑

| 症状 | 原因 | 解法 |
|---|---|---|
| `pip install flash-attn` 卡几十分钟 | 没 prebuilt wheel,在源码编译 | 正常,等;或换 Python 3.11 (wheel 更全) |
| `flash-attn` 编译报 `nvcc not found` | 没装 cuda toolkit | `conda install -c nvidia cuda-toolkit=12.6` |
| `torch` 装完 `torch.cuda.is_available() = False` | wheel CUDA 版本跟驱动不匹配 | 看 `nvidia-smi` 的 CUDA Version,换对应 `cu1XX` wheel |
| `ImportError: numpy` 相关 | numpy 2.x 不兼容 | 严格 `numpy==1.26.4` |
| `diffusers.WanPipeline` 不存在 | diffusers < 0.36 | `pip install -U diffusers==0.36.0` |

## 为什么不用 Python 3.13

User 第一次跑 `00_env_check.py` 用的是 conda base 的 Python 3.13.12。理论上 lingbot `pyproject.toml` 允许 3.13,但:

- `flash_attn` 的 prebuilt wheel 对 Python 3.13 覆盖差,大概率要源码编译(慢)
- 大量科学计算包(opencv-python、imageio 等)对 3.13 的兼容性还不完整

**结论**:新建 conda env 用 Python 3.11,跟 lingbot 实际跑通过的环境对齐。
