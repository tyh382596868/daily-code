---
date: 2026-07-22
topic: vla
source: vla
repo: Physical-Intelligence/openpi
file: src/openpi/models_pytorch/gemma_pytorch.py
permalink: https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/models_pytorch/gemma_pytorch.py#L9-L90
difficulty: advanced
read_time: ~9 min
tags: [code-of-the-day, vla, vlm-backbone-wiring, action-expert]
build_role: vlm-backbone-wiring advanced variant
---

# openpi PaliGemmaWithExpert：VLM 负责前缀，Gemma expert 负责动作后缀 / openpi PaliGemmaWithExpert: The VLM Owns the Prefix, the Gemma Expert Owns the Action Suffix

> **一句话 / In one line**: openpi 把 PaliGemma 和一个独立 Gemma action expert 包进同一模块，用配置保证两条 token 流能在后续层里对齐。 / openpi wraps PaliGemma and a separate Gemma action expert into one module, using configuration to keep the two token streams aligned for later layers.

## 为什么重要 / Why this matters

VLA 不只是“图像进 LM”。生产级动作模型经常需要一条视觉语言前缀流和一条动作后缀流：前者理解场景和指令，后者专门承载带噪动作、时间步和动作 expert 层。这个类的价值在于把两套模型的维度、精度和入口函数放在一个清晰边界里。

A VLA is not just "image into an LM." Production action models often need a vision-language prefix stream and an action suffix stream: the first understands scene and instruction, the second carries noisy actions, timesteps, and action-expert layers. This class puts the two models' dimensions, precision choices, and entry points behind one clear boundary.

## 代码 / The code

`Physical-Intelligence/openpi` — [`src/openpi/models_pytorch/gemma_pytorch.py`](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/models_pytorch/gemma_pytorch.py#L9-L90)

```python
# Simplified teaching slice, preserving the module wiring.
class PaliGemmaWithExpertModel(nn.Module):
    def __init__(self, vlm_config, action_expert_config, precision="bfloat16"):
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.vision_config.projection_dim = 2048

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            hidden_size=action_expert_config.width,
            num_hidden_layers=action_expert_config.depth,
            vocab_size=257152,
            torch_dtype="float32",
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)

    def embed_image(self, image):
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens):
        return self.paligemma.language_model.embed_tokens(tokens)
```

## 逐行讲解 / What's happening

1. **先造 PaliGemma 配置 / First build the PaliGemma config**: 中文: VLM 的 hidden size、层数、head 参数来自 openpi 自己的配置，而不是硬编码 checkpoint。 English: the VLM hidden size, depth, and head parameters come from openpi's config rather than a hard-coded checkpoint.
2. **再造 Gemma expert 配置 / Then build the Gemma expert config**: 中文: action expert 是另一套 Gemma，它和语言模型共享某些结构约束，但服务于动作后缀。 English: the action expert is another Gemma that shares structural constraints with the language model but serves the action suffix.
3. **`embed_tokens = None` 很关键 / `embed_tokens = None` is intentional**: 中文: expert 不负责把词 id 变 embedding；动作后缀通常已由动作投影器生成 embedding。 English: the expert does not turn token ids into embeddings; action suffix embeddings are usually produced by an action projector.
4. **精度选择集中处理 / Precision is handled centrally**: 中文: 大部分权重可以 bf16，但视觉 patch embedding 和 norm 常保留 fp32。 English: most weights can be bf16, while visual patch embeddings and norms often stay fp32.
5. **入口函数分开 / Entry points stay separate**: 中文: 图像和语言 token 各自先变成 embedding，后面再按 prefix/suffix 规则拼接。 English: image and language tokens first become embeddings separately, then later join under prefix/suffix rules.

## 类比 / The analogy

像一辆双驾驶舱工程车。前舱看路、读地图、听指令；后舱只控制机械臂动作。两边必须用同一套仪表刻度，否则后舱看不懂前舱传来的坐标。

It is like a two-cabin engineering vehicle. The front cabin watches the road, reads the map, and hears instructions; the rear cabin controls the arm. Both cabins need compatible gauges, or the rear cabin cannot interpret coordinates from the front.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

这是 `vlm-backbone-wiring` 的 advanced variant。你的 nanoVLA 可以先用一个小 ViT/MLP 做 `embed_image`，一个 tiny LM 做 prefix，再用一个小 action expert 处理 `[noisy_action, timestep]` suffix。上游是视觉编码器、语言 tokenizer、动作投影器；下游是连续动作 head 或 flow-matching loss。如果省掉 expert，动作 token 会和语言 token 完全共享容量，复杂控制任务更容易互相干扰。

This is an advanced variant of `vlm-backbone-wiring`. In a nanoVLA, you can start with a small ViT/MLP for `embed_image`, a tiny LM for the prefix, and a small action expert for the `[noisy_action, timestep]` suffix. Upstream are the vision encoder, language tokenizer, and action projector; downstream are the continuous action head or flow-matching loss. Without the expert, action tokens fully share language capacity, which can cause interference on harder control tasks.

## 自己跑一遍 / Try it yourself

```python
class TinyVLA:
    def embed_image(self, img): return [f"vision:{x}" for x in img]
    def embed_language(self, toks): return [f"text:{t}" for t in toks]
    def embed_action(self, actions): return [f"action:{a}" for a in actions]
    def make_streams(self, img, toks, actions):
        prefix = self.embed_image(img) + self.embed_language(toks)
        suffix = self.embed_action(actions)
        return prefix, suffix

prefix, suffix = TinyVLA().make_streams(["cam0"], ["pick", "cup"], [0.1, 0.2])
print(prefix)
print(suffix)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
['vision:cam0', 'text:pick', 'text:cup']
['action:0.1', 'action:0.2']
```

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LeRobot SmolVLA** / **LeRobot SmolVLA**: 中文: 同样把 VLM 和 action expert 作为两条互相注意的流。 / English: it similarly keeps VLM and action expert streams that attend to each other.
- **OpenVLA-OFT** / **OpenVLA-OFT**: 中文: 用 action 位置 placeholder 让 LM 输出连续动作 head。 / English: it uses action-position placeholders so the LM can feed a continuous action head.
- **GR00T** / **GR00T**: 中文: 图像、语言、状态和动作都先变成 token，再进入扩散/flow action head。 / English: image, language, state, and action become tokens before entering a diffusion or flow action head.

## 注意事项 / Caveats / when it breaks

- **维度必须对齐 / Dimensions must align**: prefix 和 suffix 后续要互相 attention，head dim、KV heads、hidden size 不能随便配。 / Prefix and suffix streams later attend to each other, so head dim, KV heads, and hidden sizes cannot be arbitrary.
- **不要让 expert 自己查词表 / Do not let the expert own token lookup**: action suffix 通常不是离散文本 token，直接用 embedding 更清晰。 / The action suffix is usually not discrete text tokens, so feeding embeddings directly is cleaner.
- **fp32 保留点要验证 / Validate fp32 islands**: 视觉 embedding 和 norm 保留 fp32 可以稳，但会增加显存。 / Keeping visual embeddings and norms in fp32 can stabilize training but costs memory.

## 延伸阅读 / Further reading

- [openpi PaliGemmaWithExpertModel](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/models_pytorch/gemma_pytorch.py#L9-L90)

