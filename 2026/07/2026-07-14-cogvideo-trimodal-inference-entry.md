---
date: 2026-07-14
topic: diffusion
source: tracked
repo: zai-org/CogVideo
file: inference/cli_demo.py
permalink: https://github.com/zai-org/CogVideo/blob/7a1af7154511e0ce4e4be8d62faa8c5e5a3532d2/inference/cli_demo.py#L47-L199
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, video-generation, inference-pipeline]
---

# CogVideoX 推理入口：一个函数分发 T2V/I2V/V2V / CogVideoX Inference Entry: One Function Dispatches T2V/I2V/V2V

> **一句话 / In one line**: CogVideoX 的 demo 不是三套脚本，而是在同一个 `generate_video` 里按生成类型选择 pipeline、输入媒体和默认分辨率。 / CogVideoX's demo is not three scripts; one `generate_video` chooses the pipeline, media input, and default resolution from the generation type.

## 为什么重要 / Why this matters

视频扩散产品通常很快就会长出三条路径：纯文本生成、图生视频、视频续写。如果每条路径都复制一份推理代码，scheduler、LoRA、offload、VAE tiling 迟早会漂移。CogVideoX 这段把差异压在 pipeline 选择和最后一次调用参数里，公共设置只写一遍。

Video diffusion products quickly grow three paths: text-to-video, image-to-video, and video-to-video. If each path copies inference code, scheduler settings, LoRA loading, offload, and VAE tiling drift. This CogVideoX entry keeps the differences at pipeline selection and the final call, while shared setup appears once.

## 代码 / The code

`zai-org/CogVideo` — [`inference/cli_demo.py`](https://github.com/zai-org/CogVideo/blob/7a1af7154511e0ce4e4be8d62faa8c5e5a3532d2/inference/cli_demo.py#L47-L199)

```python
RESOLUTION_MAP = {
    "cogvideox1.5-5b-i2v": (768, 1360),
    "cogvideox1.5-5b": (768, 1360),
    "cogvideox-5b-i2v": (480, 720),
    "cogvideox-5b": (480, 720),
    "cogvideox-2b": (480, 720),
}

def generate_video(..., generate_type: str = Literal["t2v", "i2v", "v2v"], ...):
    image = None
    video = None

    model_name = model_path.split("/")[-1].lower()
    desired_resolution = RESOLUTION_MAP[model_name]
    if width is None or height is None:
        height, width = desired_resolution
    elif (height, width) != desired_resolution:
        if generate_type != "i2v":
            height, width = desired_resolution

    if generate_type == "i2v":
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
        image = load_image(image=image_or_video_path)
    elif generate_type == "t2v":
        pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
    else:
        pipe = CogVideoXVideoToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
        video = load_video(image_or_video_path)

    if lora_path:
        pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
        pipe.fuse_lora(components=["transformer"], lora_scale=1.0)

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
```

## 逐行讲解 / What's happening

1. **分辨率表 / Resolution table**:
   - 中文: `RESOLUTION_MAP` 把模型名映射到推荐 `(height, width)`，避免用户给 T2V 模型传一个不支持的尺寸。
   - English: `RESOLUTION_MAP` maps model names to recommended `(height, width)`, preventing unsupported sizes for T2V models.
2. **三路 pipeline / Three pipelines**:
   - 中文: `generate_type` 决定用 `CogVideoXPipeline`、`CogVideoXImageToVideoPipeline` 还是 `CogVideoXVideoToVideoPipeline`。
   - English: `generate_type` selects `CogVideoXPipeline`, `CogVideoXImageToVideoPipeline`, or `CogVideoXVideoToVideoPipeline`.
3. **LoRA 融合 / LoRA fusion**:
   - 中文: adapter 先加载，再 `fuse_lora` 到 transformer，推理时少一层动态 adapter 调用。
   - English: The adapter is loaded and then fused into the transformer, removing the dynamic adapter path during inference.
4. **公共推理配置 / Shared inference setup**:
   - 中文: DPM scheduler、CPU offload、VAE slicing/tiling 对三种生成方式共用。
   - English: The DPM scheduler, CPU offload, and VAE slicing/tiling are shared across all three generation modes.

## 类比 / The analogy

这像一个火车站有三种乘客入口：只带票、带自行车、带大件行李。检票和站台调度是同一套，区别只在入口检查和装载方式。

It is like a train station with three entrances: ticket only, bicycle, and bulky luggage. Ticketing and platform dispatch are shared; only the entrance checks and loading differ.

## 自己跑一遍 / Try it yourself

```python
def choose(generate_type, custom_size=False):
    pipe = {"t2v": "text", "i2v": "image", "v2v": "video"}[generate_type]
    size = "custom" if generate_type == "i2v" and custom_size else "recommended"
    return pipe, size, ["DPM", "cpu_offload", "vae_tiling"]

for mode in ["t2v", "i2v", "v2v"]:
    print(mode, choose(mode, custom_size=True))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
t2v ('text', 'recommended', ['DPM', 'cpu_offload', 'vae_tiling'])
i2v ('image', 'custom', ['DPM', 'cpu_offload', 'vae_tiling'])
v2v ('video', 'recommended', ['DPM', 'cpu_offload', 'vae_tiling'])
```

关键点是公共推理配置和任务差异分开管理。

The key point is that shared inference settings and task-specific differences are managed separately.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Diffusers pipelines** / **Diffusers pipelines**: 多数图生、文生、编辑 pipeline 共享 scheduler 和 VAE 配置，只在输入编码处不同。 / Many image, text, and edit pipelines share scheduler and VAE settings, differing mostly at input encoding.
- **Wan2.1 T2V/I2V** / **Wan2.1 T2V/I2V**: 也把文本、图像和视频条件压进统一采样循环。 / It similarly compresses text, image, and video conditioning into one sampling loop.

## 注意事项 / Caveats / when it breaks

- **模型名必须在表里 / Model names must be in the table**: 新模型没有加入 `RESOLUTION_MAP` 会直接查表失败。 / A new model missing from `RESOLUTION_MAP` fails before inference.
- **offload 是显存换时间 / Offload trades memory for time**: `enable_sequential_cpu_offload` 省显存，但会增加 CPU/GPU 搬运。 / `enable_sequential_cpu_offload` saves VRAM but adds CPU/GPU transfers.

## 延伸阅读 / Further reading

- [CogVideoX CLI demo](https://github.com/zai-org/CogVideo/blob/7a1af7154511e0ce4e4be8d62faa8c5e5a3532d2/inference/cli_demo.py#L47-L199)
