---
date: 2026-09-18
topic: huggingface
source: huggingface
repo: huggingface/diffusers
file: src/diffusers/modular_pipelines/wan/modular_blocks_wan22.py
permalink: https://github.com/huggingface/diffusers/blob/7221eef4573574925b67a69e9fc1482bf093e569/src/diffusers/modular_pipelines/wan/modular_blocks_wan22.py#L43-L104
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, diffusers, wan22, modular-pipeline]
---

# Diffusers Wan2.2 modular block：把 denoise 变成可组合流水线 / Diffusers Wan2.2 Modular Block: Turn Denoising into a Composable Pipeline

> **一句话 / In one line**: Diffusers 用一个声明式 `SequentialPipelineBlocks` 把文本输入、时间步、latent 准备和 Wan2.2 去噪串成一个有明确输出合同的 block。 / Diffusers uses a declarative `SequentialPipelineBlocks` to compose text input, timestep setup, latent preparation, and Wan2.2 denoising behind an explicit output contract.

## 为什么重要 / Why this matters

大型生成 pipeline 很容易变成一整个几百行的“上帝类”：文本编码、scheduler、CFG、transformer、VAE 全部交织在一起。这样做的第一个版本很快，但当 I2V、VACE、蒸馏模型或不同 guider 出现时，复制粘贴会让维护成本爆炸。

Large generation pipelines easily turn into giant classes where text encoding, schedulers, CFG, transformers, and VAEs are intertwined. That is fast for a first implementation, but I2V, VACE, distilled models, and alternate guiders make copy-paste maintenance painful.

Wan2.2 的 modular block 选择把“阶段”作为一等公民。`Wan22CoreDenoiseStep` 不自己实现每一行 denoise 数学，而是声明它依赖哪些子 block、它们按什么顺序运行、最后输出什么。真正的行为位于可替换的 step 类里。

Wan2.2 makes pipeline stages first-class. `Wan22CoreDenoiseStep` does not reimplement every denoising equation; it declares which blocks it needs, their order, and the final output. The replaceable step classes contain the actual behavior.

## 代码 / The code

`huggingface/diffusers` — [`src/diffusers/modular_pipelines/wan/modular_blocks_wan22.py`](https://github.com/huggingface/diffusers/blob/7221eef4573574925b67a69e9fc1482bf093e569/src/diffusers/modular_pipelines/wan/modular_blocks_wan22.py#L43-L104)

```python
class Wan22CoreDenoiseStep(SequentialPipelineBlocks):
    """
    denoise block that takes encoded conditions and runs the denoising process.

      Components:
          transformer (`WanTransformer3DModel`) scheduler (`UniPCMultistepScheduler`) guider (`ClassifierFreeGuidance`)
          guider_2 (`ClassifierFreeGuidance`) transformer_2 (`WanTransformer3DModel`)

      Configs:
          boundary_ratio (default: 0.875): The boundary ratio to divide the denoising loop into high noise and low
          noise stages.

      Inputs:
          num_videos_per_prompt (`None`, *optional*, defaults to 1):
              TODO: Add description.
          prompt_embeds (`Tensor`):
              Pre-generated text embeddings. Can be generated from text_encoder step.
          negative_prompt_embeds (`Tensor`, *optional*):
              Pre-generated negative text embeddings. Can be generated from text_encoder step.
          num_inference_steps (`None`, *optional*, defaults to 50):
              TODO: Add description.
          timesteps (`None`, *optional*):
              TODO: Add description.
          sigmas (`None`, *optional*):
              TODO: Add description.
          height (`int`, *optional*):
              TODO: Add description.
          width (`int`, *optional*):
              TODO: Add description.
          num_frames (`int`, *optional*):
              TODO: Add description.
          latents (`Tensor | NoneType`, *optional*):
              TODO: Add description.
          generator (`None`, *optional*):
              TODO: Add description.
          attention_kwargs (`None`, *optional*):
              TODO: Add description.
          **denoiser_input_fields (`None`, *optional*):
              conditional model inputs for the denoiser: e.g. prompt_embeds, negative_prompt_embeds, etc.

      Outputs:
          latents (`Tensor`):
              Denoised latents.
    """

    model_name = "wan"
    block_classes = [
        WanTextInputStep,
        WanSetTimestepsStep,
        WanPrepareLatentsStep,
        Wan22DenoiseStep,
    ]
    block_names = ["input", "set_timesteps", "prepare_latents", "denoise"]

    @property
    def description(self):
        return "denoise block that takes encoded conditions and runs the denoising process."

    @property
    def outputs(self):
        return [OutputParam.template("latents")]
```

## 逐行讲解 / What's happening

1. **第 43 行 / Line 43 (`SequentialPipelineBlocks`)**:
   - 中文: 这个基类表示“按顺序运行一组 pipeline block”，所以组合逻辑从模型实现里抽出来了。
   - English: The base class means “run a sequence of pipeline blocks,” separating composition from model implementation.
2. **第 47-53 行 / Lines 47-53 (components and boundary ratio)**:
   - 中文: 文档把两套 transformer、scheduler 和 guider 写进组件合同，`boundary_ratio` 说明 Wan2.2 会按噪声阶段切换 denoiser。
   - English: The component contract names two transformers, schedulers, and guiders, while `boundary_ratio` documents the high-noise/low-noise handoff.
3. **第 55-81 行 / Lines 55-81 (inputs)**:
   - 中文: 这一段不是装饰性 docstring；它把 prompt embedding、negative embedding、timesteps、分辨率和 latent 输入集中声明，方便 modular pipeline 做校验和组合。
   - English: This is more than documentation: it centralizes prompt embeddings, negative embeddings, timesteps, resolution, and latent inputs so the modular pipeline can validate and compose them.
4. **第 88-95 行 / Lines 88-95 (`block_classes` and `block_names`)**:
   - 中文: 这是最关键的“流水线骨架”：先读条件，再设时间表，再准备 latent，最后进入 Wan2.2 denoise。
   - English: This is the pipeline skeleton: read conditions, set the schedule, prepare latents, then enter the Wan2.2 denoiser.
5. **第 97-104 行 / Lines 97-104 (description and outputs)**:
   - 中文: 输出合同只有 `latents`，说明这个 block 可以作为更大 text-to-video pipeline 的中间阶段，而不是直接绑定 VAE 输出。
   - English: The block exposes only `latents`, so it can sit inside a larger text-to-video pipeline without being coupled to final VAE decoding.

## 类比 / The analogy

这像一家餐厅的出餐线：点单、排队取号、备料、烹饪是四个工位。总厨不需要在菜单里重写每个工位的动作，只要声明工位顺序和最后交付的是“装好盘的菜”。换菜单时，可以替换某个工位而不拆掉整条线。

This is like a restaurant line with four stations: take the order, assign a ticket, prepare ingredients, and cook. The head chef does not rewrite every station in the menu; it declares the order and the final deliverable. A menu variant can replace one station without rebuilding the whole line.

## 自己跑一遍 / Try it yourself

```python
class Block:
    def __init__(self, name):
        self.name = name

    def __call__(self, state):
        state.append(self.name)
        return state

class Pipeline:
    def __init__(self, blocks):
        self.blocks = blocks

    def __call__(self):
        state = []
        for block in self.blocks:
            state = block(state)
        return {"latents": state}

pipeline = Pipeline([Block("input"), Block("timesteps"), Block("latents"), Block("denoise")])
print(pipeline())
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
{'latents': ['input', 'timesteps', 'latents', 'denoise']}
```

中文: 小例子里每个 block 只负责自己的阶段，pipeline 只负责顺序和输出合同。  
English: Each toy block owns one stage; the pipeline owns ordering and the output contract.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Diffusers modular pipeline** / **Diffusers modular pipeline**: 中文: 通过 `ComponentSpec`、`InputParam` 和 block state，把大 pipeline 拆成可以重用的积木。 / English: `ComponentSpec`, `InputParam`, and block state turn a large pipeline into reusable pieces.
- **PyTorch `nn.Sequential`** / **PyTorch `nn.Sequential`**: 中文: 共享“按顺序传递 state”的思想，但 modular pipeline 还额外声明组件和输入输出。 / English: It shares sequential state passing, while modular pipelines add explicit component and input/output contracts.
- **Wan2.2 dual denoiser** / **Wan2.2 dual denoiser**: 中文: high-noise 与 low-noise 阶段可在同一 blockset 里切换不同 transformer。 / English: High-noise and low-noise stages can select different transformers within one blockset.

## 注意事项 / Caveats / when it breaks

- **声明不等于实现** / **Declaration is not execution**: 中文: `block_classes` 只描述组成，真正的 timestep、CFG 和 transformer 调用仍在子 block 中。 / English: `block_classes` describes composition; timestep math, CFG, and transformer calls still live in child blocks.
- **输入合同必须稳定** / **Input contracts must stay stable**: 中文: 新 block 如果修改字段名，组合 pipeline 的连接点就会断。 / English: Renaming fields in a new block can break every composition point that depends on the contract.
- **阶段拆分要有边界** / **Stages need useful boundaries**: 中文: 拆得太细会让 state 管理变复杂，拆得太粗又回到巨型 pipeline。 / English: Splitting too finely complicates state management; splitting too coarsely recreates the monolithic pipeline.

## 延伸阅读 / Further reading

- [Diffusers Wan2.2 modular blocks](https://github.com/huggingface/diffusers/blob/7221eef4573574925b67a69e9fc1482bf093e569/src/diffusers/modular_pipelines/wan/modular_blocks_wan22.py)
- [Diffusers modular pipelines guide](https://huggingface.co/docs/diffusers/main/en/api/modular_pipelines/overview)
- [Wan2.2 denoise block](https://github.com/huggingface/diffusers/blob/7221eef4573574925b67a69e9fc1482bf093e569/src/diffusers/modular_pipelines/wan/denoise.py)
