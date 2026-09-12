---
date: 2026-08-30
topic: diffusion
source: tracked
repo: zai-org/CogVideo
file: inference/cli_demo.py
permalink: https://github.com/zai-org/CogVideo/blob/7a1af7154511e0ce4e4be8d62faa8c5e5a3532d2/inference/cli_demo.py#L98-L150
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, video-generation, pipeline-dispatch]
---

# CogVideo pipeline dispatch：一种入口，三种生成模式 / CogVideo Pipeline Dispatch: One Entry, Three Generation Modes

> **一句话 / In one line**: CogVideoX 的 CLI 先根据模型和生成类型选择分辨率、pipeline、LoRA、scheduler 和 offload 策略，再统一进入采样。 / CogVideoX's CLI first selects resolution, pipeline, LoRA, scheduler, and offload policy from the model and generation type, then enters one shared sampling path.

## 为什么重要 / Why this matters

视频生成 demo 很容易变成三份几乎一样的脚本：text-to-video 一份，image-to-video 一份，video-to-video 再一份。这段入口代码把差异集中到装配阶段，让后面的推理流程少关心模式分支。

Video-generation demos can easily become three near-duplicate scripts: one for text-to-video, one for image-to-video, and another for video-to-video. This entry point keeps the differences in the assembly phase, so the later inference path can stay mostly shared.

## 代码 / The code

`zai-org/CogVideo` — [`inference/cli_demo.py`](https://github.com/zai-org/CogVideo/blob/7a1af7154511e0ce4e4be8d62faa8c5e5a3532d2/inference/cli_demo.py#L98-L150)

```python
model_name = model_path.split("/")[-1].lower()
desired_resolution = RESOLUTION_MAP[model_name]
if width is None or height is None:
    height, width = desired_resolution
    logging.info(
        f"\033[1mUsing default resolution {desired_resolution} for {model_name}\033[0m"
    )
elif (height, width) != desired_resolution:
    if generate_type == "i2v":
        # For i2v models, use user-defined width and height
        logging.warning(
            f"\033[1;31mThe width({width}) and height({height}) are not recommended for {model_name}. The best resolution is {desired_resolution}.\033[0m"
        )
    else:
        # Otherwise, use the recommended width and height
        logging.warning(
            f"\033[1;31m{model_name} is not supported for custom resolution. Setting back to default resolution {desired_resolution}.\033[0m"
        )
        height, width = desired_resolution

if generate_type == "i2v":
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
    image = load_image(image=image_or_video_path)
elif generate_type == "t2v":
    pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
else:
    pipe = CogVideoXVideoToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
    video = load_video(image_or_video_path)

# If you're using with lora, add this code
if lora_path:
    pipe.load_lora_weights(
        lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1"
    )
    pipe.fuse_lora(components=["transformer"], lora_scale=1.0)

# 2. Set Scheduler.
# Can be changed to `CogVideoXDPMScheduler` or `CogVideoXDDIMScheduler`.
# We recommend using `CogVideoXDDIMScheduler` for CogVideoX-2B.
# using `CogVideoXDPMScheduler` for CogVideoX-5B / CogVideoX-5B-I2V.

# pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
pipe.scheduler = CogVideoXDPMScheduler.from_config(
    pipe.scheduler.config, timestep_spacing="trailing"
)

# 3. Enable CPU offload for the model.
# turn off if you have multiple GPUs or enough GPU memory(such as H100) and it will cost less time in inference
# and enable to("cuda")
# pipe.to("cuda")

# pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()
```

## 逐行讲解 / What's happening

1. **第 98-116 行 / Lines 98-116**:
   - 中文: 先从模型名查推荐分辨率；如果用户给了不匹配的尺寸，只有 `i2v` 保留自定义尺寸，其它模式回退到模型默认值。
   - English: The code first looks up the recommended resolution from the model name; when user dimensions do not match, only `i2v` keeps the custom size while other modes fall back to the model default.
2. **第 118-125 行 / Lines 118-125**:
   - 中文: `generate_type` 决定具体 pipeline 类，也决定是否需要加载输入图片或输入视频。
   - English: `generate_type` selects the concrete pipeline class and determines whether an input image or video must be loaded.
3. **第 127-132 行 / Lines 127-132**:
   - 中文: LoRA 权重被加载到 pipeline 后直接 fuse 到 transformer，推理时少一层 adapter 调用。
   - English: LoRA weights are loaded into the pipeline and fused into the transformer, avoiding an extra adapter path during inference.
4. **第 139-150 行 / Lines 139-150**:
   - 中文: scheduler 统一换成 trailing timestep 的 DPM 版本，并打开 sequential CPU offload 来降低显存压力。
   - English: The scheduler is replaced with a trailing-timestep DPM variant, and sequential CPU offload is enabled to reduce GPU memory pressure.

## 类比 / The analogy

这像一个电影院前台：你可以买普通电影票、IMAX 票或带餐套票，但前台先把影厅、座位、餐品和入场路线配好，进场后你看到的是同一套放映流程。

It is like a movie-theater counter: you may buy a regular ticket, an IMAX ticket, or a meal bundle, but the counter first assigns the room, seat, food, and entry route. Once inside, the projection flow is the same.

## 自己跑一遍 / Try it yourself

```python
def choose(generate_type, custom_size=False):
    resolution = (480, 720)
    if custom_size and generate_type != "i2v":
        custom_size = False
    pipe = {"t2v": "TextPipe", "i2v": "ImagePipe", "v2v": "VideoPipe"}[generate_type]
    loaded = {"t2v": None, "i2v": "image", "v2v": "video"}[generate_type]
    return pipe, loaded, "custom" if custom_size else resolution

for mode in ["t2v", "i2v", "v2v"]:
    print(mode, choose(mode, custom_size=True))
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
t2v ('TextPipe', None, (480, 720))
i2v ('ImagePipe', 'image', 'custom')
v2v ('VideoPipe', 'video', (480, 720))
```

注意 `i2v` 保留了自定义尺寸，其它模式统一回到模型推荐尺寸。

Notice that `i2v` keeps the custom size, while the other modes return to the model's recommended size.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Diffusers pipelines** / **Diffusers pipelines**: 同一个模型族常把 T2I、I2I、inpainting 拆成不同 pipeline，但共享 scheduler 和组件装配。 / A model family often splits T2I, I2I, and inpainting into different pipelines while sharing scheduler and component assembly.
- **Wan2.1 sampling entry** / **Wan2.1 sampling entry**: 推理入口先装配 text/image context，再进入 timestep 循环。 / The inference entry assembles text and image context before entering the timestep loop.
- **Open-Sora demos** / **Open-Sora demos**: 配置、模型加载和采样器选择先发生，实际采样路径保持窄接口。 / Config, model loading, and sampler choice happen first, keeping the actual sampling path narrow.

## 注意事项 / Caveats / when it breaks

- **模型名必须在表里** / **Model names must be mapped**: `RESOLUTION_MAP[model_name]` 假设模型名已经登记，否则会直接失败。 / `RESOLUTION_MAP[model_name]` assumes the model name is registered and otherwise fails immediately.
- **LoRA fuse 不是可逆开关** / **LoRA fuse is not a free toggle**: fuse 后要切换 adapter 或 scale 会更麻烦。 / After fusing, switching adapters or scales becomes less convenient.
- **offload 换速度换显存** / **Offload trades speed for memory**: sequential CPU offload 省显存，但每步搬运会拖慢推理。 / Sequential CPU offload saves memory, but moving modules each step slows inference.

## 延伸阅读 / Further reading

- CogVideo CLI demo: https://github.com/zai-org/CogVideo/blob/7a1af7154511e0ce4e4be8d62faa8c5e5a3532d2/inference/cli_demo.py
- Diffusers CogVideoX docs: https://huggingface.co/docs/diffusers/main/en/api/pipelines/cogvideox
