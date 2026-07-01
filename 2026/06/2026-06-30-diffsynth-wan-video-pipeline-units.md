---
date: 2026-06-30
topic: diffusion
source: trending
repo: modelscope/DiffSynth-Studio
file: diffsynth/pipelines/wan_video.py
permalink: https://github.com/modelscope/DiffSynth-Studio/blob/be8eee932d4680df465ec61f97487a2e0be1c93f/diffsynth/pipelines/wan_video.py#L25-L181
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, trending, video-diffusion, pipeline]
---

# DiffSynth 的 WanVideoPipeline：把视频生成拆成可插拔单元 / DiffSynth's WanVideoPipeline: Video Generation as Pluggable Units

> **一句话 / In one line**: `WanVideoPipeline` 不是把所有逻辑写进一个巨大 `__call__`，而是注册一串 `PipelineUnit`。 / `WanVideoPipeline` does not put everything into one giant `__call__`; it registers a chain of `PipelineUnit`s.

## 为什么重要 / Why this matters

现代视频生成管线功能很多：文生视频、图生视频、首尾帧、音频驱动、ControlNet、VACE、TeaCache、sequence parallel。DiffSynth 的做法是把这些能力拆成一组 unit，主 pipeline 只负责装配模型和按顺序执行单元。

Modern video generation pipelines have many modes: text-to-video, image-to-video, first-last-frame, audio-driven generation, ControlNet, VACE, TeaCache, and sequence parallelism. DiffSynth splits those features into units, while the main pipeline focuses on model assembly and ordered execution.

## 代码 / The code

`modelscope/DiffSynth-Studio` — [`diffsynth/pipelines/wan_video.py`](https://github.com/modelscope/DiffSynth-Studio/blob/be8eee932d4680df465ec61f97487a2e0be1c93f/diffsynth/pipelines/wan_video.py#L25-L181)

```python
class WanVideoPipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.scheduler = FlowMatchScheduler("Wan")
        self.tokenizer: HuggingfaceTokenizer = None
        self.audio_processor: Wav2Vec2Processor = None
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.dit2: WanModel = None
        self.vae: WanVideoVAE = None
        self.motion_controller: WanMotionControllerModel = None
        self.vace: VaceWanModel = None
        self.vace2: VaceWanModel = None
        self.vap: MotWanModel = None
        self.animate_adapter: WanAnimateAdapter = None
        self.audio_encoder: WanS2VAudioEncoder = None
        self.in_iteration_models = ("dit", "motion_controller", "vace", "animate_adapter", "vap")
        self.in_iteration_models_2 = ("dit2", "motion_controller", "vace2", "animate_adapter", "vap")
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_S2V(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_ImageEmbedderVAE(),
            WanVideoUnit_ImageEmbedderCLIP(),
            WanVideoUnit_ImageEmbedderFused(),
            WanVideoUnit_FunControl(),
            WanVideoUnit_FunReference(),
            WanVideoUnit_FunCameraControl(),
            WanVideoUnit_SpeedControl(),
            WanVideoUnit_VACE(),
            WanVideoUnit_AnimateVideoSplit(),
            WanVideoUnit_AnimatePoseLatents(),
            WanVideoUnit_AnimateFacePixelValues(),
            WanVideoUnit_AnimateInpaint(),
            WanVideoUnit_VAP(),
            WanVideoUnit_UnifiedSequenceParallel(),
            WanVideoUnit_TeaCache(),
            WanVideoUnit_CfgMerger(),
            WanVideoUnit_LongCatVideo(),
            WanVideoUnit_WanToDance_ProcessInputs(),
            WanVideoUnit_WanToDance_RefImageEmbedder(),
            WanVideoUnit_WanToDance_ImageKeyframesEmbedder(),
        ]
        self.post_units = [
            WanVideoPostUnit_S2V(),
        ]
        self.model_fn = model_fn_wan_video
        self.compilable_models = ["dit", "dit2"]
```

## 逐行讲解 / What's happening

1. **第 29-32 行 / Lines 29-32**:
   - 中文: pipeline 声明空间和时间维度的整除约束，提前把 shape 合法性变成框架属性。
   - English: The pipeline declares spatial and temporal divisibility constraints as framework-level properties.
2. **第 34-48 行 / Lines 34-48**:
   - 中文: 所有可能用到的模型槽位先显式列出，后续 `from_pretrained` 再填充。
   - English: All possible model slots are listed up front; `from_pretrained` fills them later.
3. **第 51-77 行 / Lines 51-77**:
   - 中文: 每个功能是一个 unit，开关功能时主要是让 unit 决定是否处理输入。
   - English: Each feature is a unit; enabling a feature mostly means letting the relevant unit process the inputs.
4. **第 81-82 行 / Lines 81-82**:
   - 中文: 只有 DiT 主体被标记为可 compile，避免编译一堆控制流胶水。
   - English: Only the DiT modules are marked compilable, avoiding compilation of control-flow glue.

## 类比 / The analogy

这像一条厨房流水线：洗菜、切菜、炒菜、装盘是独立工位。菜单变复杂时，不需要把厨师训练成一整套硬编码流程，只要加一个新工位。

It is like a kitchen line: washing, cutting, cooking, and plating are separate stations. When the menu grows, you add a station instead of hard-coding one huge chef routine.

## 自己跑一遍 / Try it yourself

```python
class Unit:
    def __init__(self, name): self.name = name
    def __call__(self, state):
        state.append(self.name)
        return state

units = [Unit("shape"), Unit("noise"), Unit("prompt"), Unit("cfg")]
state = []
for unit in units:
    state = unit(state)
print(" -> ".join(state))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
shape -> noise -> prompt -> cfg
```

这个小例子展示的是架构，不是扩散数学：复杂 pipeline 可以先做成可组合步骤。

The toy example demonstrates architecture rather than diffusion math: a complex pipeline can start as composable steps.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Diffusers pipelines** / **Diffusers pipelines**: 文本编码、latent 准备、scheduler step、decode 也是固定阶段。 / Text encoding, latent preparation, scheduler steps, and decoding are also staged.
- **LeRobot policies** / **LeRobot policies**: 观测预处理、模型 forward、动作后处理也常被拆成小单元。 / Observation preprocessing, model forward, and action postprocessing are often split into small units.

## 注意事项 / Caveats / when it breaks

- **unit 顺序就是隐式合约** / **Unit order is an implicit contract**: Prompt 必须先于 CFG，noise 必须先于 denoise。 / Prompt must precede CFG, and noise must precede denoising.
- **状态字典会膨胀** / **The state dictionary can grow large**: unit 多了以后，需要清晰命名中间字段。 / With many units, intermediate field names need discipline.

## 延伸阅读 / Further reading

- [DiffSynth-Studio Wan pipeline](https://github.com/modelscope/DiffSynth-Studio/blob/be8eee932d4680df465ec61f97487a2e0be1c93f/diffsynth/pipelines/wan_video.py)

