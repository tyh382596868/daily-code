---
date: 2026-08-21
topic: diffusion
source: trending
repo: Tencent-Hunyuan/HY-World-2.0
file: hyworld2/panogen/pipeline.py
permalink: https://github.com/Tencent-Hunyuan/HY-World-2.0/blob/df9988efb87bfc0f4947eb3889411cf957478b06/hyworld2/panogen/pipeline.py#L191-L292
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, world-model, panorama, pipeline]
---

# HY-World panorama pipeline：把指令、图像和缓存参数一次交清 / HY-World Panorama Pipeline: Hand Off Prompt, Image, and Cache Knobs Together

> **一句话 / In one line**: `HunyuanPanoPipeline.forward` 规范化 panorama prompt，检查输入图像，然后把尺寸、扩散步数、推理任务和 Taylor cache 参数统一转交给 `generate_image`。 / `HunyuanPanoPipeline.forward` normalizes the panorama prompt, validates the input image, then forwards size, diffusion steps, task mode, and Taylor-cache knobs into `generate_image`.

## 为什么重要 / Why this matters

世界生成 pipeline 容易把控制参数散落在 CLI、模型和后处理里。HY-World 把 panorama 任务的输入契约集中在一个 forward：prompt 必须带全景指令，图像路径必须存在，扩散步数和 cache 策略必须一起进入模型调用。这样调试时能清楚知道“这次生成到底用了什么条件”。

World-generation pipelines often scatter controls across CLI, model calls, and post-processing. HY-World centralizes the panorama task contract in one forward path: the prompt must include the panorama instruction, the image path must exist, and diffusion/cache settings travel together into the model call.

## 代码 / The code

`Tencent-Hunyuan/HY-World-2.0` — [`hyworld2/panogen/pipeline.py`](https://github.com/Tencent-Hunyuan/HY-World-2.0/blob/df9988efb87bfc0f4947eb3889411cf957478b06/hyworld2/panogen/pipeline.py#L191-L292)

```python
def forward(
    self,
    image,
    *,
    prompt=PANO_INSTRUCTION,
    seed=None,
    height=960,
    width=1952,
    use_system_prompt="en_unified",
    system_prompt=None,
    bot_task="think_recaption",
    diff_infer_steps=50,
    verbose=2,
    max_new_tokens=2048,
    infer_align_image_size=False,
    blend_width=32,
    use_taylor_cache=False,
    taylor_cache_interval=5,
    taylor_cache_order=2,
    taylor_cache_enable_first_enhance=False,
    taylor_cache_first_enhance_steps=3,
    taylor_cache_enable_tailing_enhance=False,
    taylor_cache_tailing_enhance_steps=1,
    taylor_cache_low_freqs_order=2,
    taylor_cache_high_freqs_order=2,
):
    image = str(image)
    if not Path(image).exists():
        raise ValueError(f"Input image does not exist: {image}")

    # Ensure the panorama instruction is always present
    if PANO_INSTRUCTION not in prompt:
        prompt = f"{PANO_INSTRUCTION} {prompt}".strip()
        print(f"[Info] Panorama instruction prepended. Final prompt: {prompt}")

    cot_text, samples = self.model.generate_image(
        prompt=prompt,
        image=[image],
        seed=seed,
        image_size=[height, width],
        use_system_prompt=use_system_prompt,
        system_prompt=system_prompt,
        bot_task=bot_task,
        diff_infer_steps=diff_infer_steps,
        verbose=verbose,
        max_new_tokens=max_new_tokens,
        infer_align_image_size=infer_align_image_size,
        use_taylor_cache=use_taylor_cache,
        taylor_cache_interval=taylor_cache_interval,
        taylor_cache_order=taylor_cache_order,
        taylor_cache_enable_first_enhance=taylor_cache_enable_first_enhance,
        taylor_cache_first_enhance_steps=taylor_cache_first_enhance_steps,
        taylor_cache_enable_tailing_enhance=taylor_cache_enable_tailing_enhance,
        taylor_cache_tailing_enhance_steps=taylor_cache_tailing_enhance_steps,
        taylor_cache_low_freqs_order=taylor_cache_low_freqs_order,
        taylor_cache_high_freqs_order=taylor_cache_high_freqs_order,
    )
```

## 逐行讲解 / What's happening

1. **第 191-217 行 / Lines 191-217 (pipeline contract)**:
   - 中文: forward 参数同时覆盖 prompt、输出尺寸、扩散步数、任务模式和 Taylor cache。
   - English: The forward signature covers prompt, output size, diffusion steps, task mode, and Taylor-cache controls.
2. **第 250-257 行 / Lines 250-257 (input normalization)**:
   - 中文: 图像路径先转字符串并检查存在；prompt 缺少全景指令时自动补上。
   - English: The image path is stringified and checked; the panorama instruction is prepended if missing.
3. **第 267-288 行 / Lines 267-288 (single model handoff)**:
   - 中文: 所有关键推理参数一次性传给 `generate_image`，避免 pipeline 外层和模型内层状态不一致。
   - English: All important inference settings are passed into `generate_image` together, avoiding outer/inner state drift.
4. **第 290-292 行 / Lines 290-292 (reasoning trace)**:
   - 中文: 如果模型返回 chain-of-thought/recaption 文本，pipeline 会把它作为调试线索打印。
   - English: If the model returns reasoning or recaption text, the pipeline prints it as a debugging trace.

## 类比 / The analogy

像给施工队一张完整工单：地点、尺寸、工艺、加速选项都写在同一页，队伍不用在多个聊天记录里拼条件。

It is like giving a construction crew one complete work order: location, dimensions, process, and acceleration options are all on the same page.

## 自己跑一遍 / Try it yourself

```python
PANO_INSTRUCTION = "Generate a panorama."

def normalize_prompt(prompt):
    if PANO_INSTRUCTION not in prompt:
        prompt = f"{PANO_INSTRUCTION} {prompt}".strip()
    return prompt

payload = {
    "prompt": normalize_prompt("sunlit kitchen"),
    "image_size": [960, 1952],
    "diff_infer_steps": 50,
    "use_taylor_cache": False,
}
print(payload)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
{'prompt': 'Generate a panorama. sunlit kitchen', 'image_size': [960, 1952], 'diff_infer_steps': 50, 'use_taylor_cache': False}
```

中文: toy 版本只演示 pipeline 把 prompt 和生成参数整理成一次模型调用的输入。

English: The toy version only shows the pipeline organizing prompt and generation settings into one model-call payload.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Diffusers pipelines** / **Diffusers pipelines**: pipeline 负责输入检查、默认参数和 scheduler/model 的调用顺序。
- **world-model rollout scripts** / **world-model rollout scripts**: rollout 配置应集中记录 seed、horizon、step count 和缓存策略。

## 注意事项 / Caveats / when it breaks

- **pipeline 不是模型本体** / **The pipeline is not the model core**: 这里主要是契约和调度，真正生成逻辑在 `generate_image`。
- **cache 参数要随实验记录** / **Cache settings belong in experiment logs**: Taylor cache 会影响速度和可能的输出差异，不能只记录 seed。

## 延伸阅读 / Further reading

- [HY-World panorama pipeline](https://github.com/Tencent-Hunyuan/HY-World-2.0/blob/df9988efb87bfc0f4947eb3889411cf957478b06/hyworld2/panogen/pipeline.py#L191-L292)
- [HY-World 2.0 repository](https://github.com/Tencent-Hunyuan/HY-World-2.0)
