---
date: 2026-05-29
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/text2video.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/text2video.py#L240-L246
difficulty: beginner
read_time: ~8 min
tags: [code-of-the-day, wam, classifier-free-guidance, cfg, sampling]
build_role: Classifier-free guidance — training dropout + inference combine that strengthens prompt adherence
---

# CFG 就是两次 forward 一行加权 / CFG is two forwards and one weighted sum

> **一句话 / In one line**: Classifier-free guidance 在推理时跑两次模型(一次有 prompt、一次"空" prompt),然后用 `noise_pred = uncond + scale * (cond - uncond)` 把"有 prompt 比无 prompt 多出来的方向"放大,几乎所有 text-to-video 模型都靠这一行决定 prompt 服从度。 / At inference, CFG runs the model twice (with prompt and with the null prompt) and combines as `noise_pred = uncond + scale * (cond - uncond)` — amplifying the direction the conditioned prediction adds over the unconditional one. Every text-to-video model relies on this single line for prompt fidelity.

## 为什么重要 / Why this matters

Diffusion 模型如果只学 `p(x | prompt)`,推理时会"听半句话"—— 模型不太敢偏离训练分布,prompt 弱时基本忽视 prompt。CFG 的思路:让同一个模型同时学 `p(x | prompt)` 和 `p(x)`(无条件版本),推理时取两者方向的**外推**:`cond + (scale-1) * (cond - uncond)`,放大 prompt 带来的"额外信号"。代价是推理变成 2 倍 forward,但 prompt 跟随度立刻可控,scale 越大越严格服从 prompt(但太大会过饱和)。Wan2.1 这 7 行就是工业级实现 —— 用一个 `context_null` 跑 unconditional pass,再一行算 guidance。在 nanoWAM 里这是必装件,因为不带 CFG 的 video diffusion 在 SOTA prompt benchmark 上分数会差 30%+。

A pure conditional diffusion model `p(x | prompt)` is too cautious — it stays near the unconditional manifold and underuses the prompt. CFG trains one model that can produce *both* conditioned and unconditioned scores; at inference it extrapolates the conditional direction past the unconditional baseline: `cond + (scale - 1) * (cond - uncond)`. The cost is 2× forward passes per step; the reward is a tunable knob on prompt adherence. Wan2.1's seven-line implementation is the industrial template — one `context_null` for the unconditional run, one line for the guidance combine. nanoWAM should treat CFG as a hard requirement; ablating it loses ~30% on prompt-following benchmarks.

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/text2video.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/text2video.py#L230-L246)

```python
# context: encoded prompt   |  context_null: encoded ""/negative prompt
arg_c    = {'context': context,      'seq_len': seq_len}
arg_null = {'context': context_null, 'seq_len': seq_len}

for _, t in enumerate(tqdm(timesteps)):
    latent_model_input = latents
    timestep = torch.stack([t])

    self.model.to(self.device)

    # === CFG: two forward passes per step ===
    noise_pred_cond   = self.model(latent_model_input, t=timestep, **arg_c)[0]
    noise_pred_uncond = self.model(latent_model_input, t=timestep, **arg_null)[0]

    # === The one-line CFG combine (DDPM/score interpretation: extrapolation) ===
    noise_pred = noise_pred_uncond + guide_scale * (
        noise_pred_cond - noise_pred_uncond)

    # scheduler.step uses the guided noise_pred to denoise x_t → x_{t-1}
    temp_x0 = sample_scheduler.step(noise_pred.unsqueeze(0), t,
                                    latents[0].unsqueeze(0))[0]
    latents = [temp_x0.squeeze(0)]
```

## 逐行讲解 / What's happening

1. **`context` vs `context_null` / Two prompts per step**:
   - 中文:`context` 是 T5 编出来的真实 prompt embedding;`context_null` 是"空字符串"或者 negative prompt 编出来的 unconditional 版本(通常预先编一次,整轮采样里不变)。
   - English: `context` is the T5-encoded user prompt. `context_null` is the encoded empty string (or a negative prompt) — usually encoded once before sampling and reused for every step.

2. **两次 forward / Two forwards per timestep**:
   - 中文:**注意**两次 forward 用的是**同一个** `latent_model_input` 和**同一个** `timestep`,只换 context。这意味着模型本身就能"接受空 context"—— 训练时是怎么教会的?见下文的训练侧 dropout。
   - English: both forwards use the **same** `latent_model_input` and **same** `timestep`. Only the `context` differs. The model itself must be trained to accept null context — that's done during training via prompt dropout (covered below).

3. **The combine line / The combine line**:
   - 中文:`noise_pred_uncond + scale * (cond - uncond)` 重写一下就是 `cond + (scale - 1) * (cond - uncond)` —— 在 cond 方向上往 uncond 的反方向**外推**。`scale = 1` 等价于纯 conditional,没有 CFG;`scale = 0` 等价于纯 unconditional;实际值常用 5-9.
   - English: rewrite to `cond + (scale - 1) * (cond - uncond)` and it's clearly an *extrapolation* from `uncond` toward `cond`. `scale = 1` = pure conditional (no CFG), `scale = 0` = pure unconditional, real-world values are 5-9.

4. **scheduler.step 用 guided noise / Scheduler consumes guided noise**:
   - 中文:`noise_pred` 喂进 scheduler.step,scheduler 用它算 `x_{t-1}`(对 Wan 这种 flow-matching 模型,`scheduler.step` 就是 `x = x + dt * v` 的欧拉一步)。从 scheduler 的角度看,CFG 完全透明 —— 它只接收一个 noise pred。
   - English: `noise_pred` enters `scheduler.step`, which produces `x_{t-1}` (for Wan's flow-matching scheduler, that's `x = x + dt * v` Euler). The scheduler is CFG-agnostic — it just consumes whatever noise prediction we hand it.

5. **训练侧:dropout 让模型同时学 cond + uncond / Training side: dropout teaches both**:
   - 中文:推理这 7 行只有意义当训练阶段以概率 `p_drop`(通常 0.1)把 `context` 替换成 `context_null`。这样同一组参数既学到 `p(x|c)` 也学到 `p(x|∅)`。Wan2.1 的训练脚本里那一行就是 `if random() < 0.1: context = context_null`。
   - English: the inference code only works because training drops `context` to `context_null` with probability ~0.1 each step. The single weight set then represents both `p(x | c)` and `p(x | ∅)`. Wan2.1's training loop has a one-line `if random() < 0.1: context = context_null` — the entire CFG capability hinges on it.

## 类比 / The analogy

像两个版本的菜谱叠加:一份是"做饭"(uncond,可能做任何菜),一份是"做番茄炒蛋"(cond)。CFG 是说:看看具体指令比泛泛指令多让你做了哪些动作("加番茄"、"打蛋"……),然后把这些"额外动作"放大 5 倍,**确保**最终成品是番茄炒蛋而不是泛泛的菜。`scale` 是放大倍数 —— 放大太大会变成"番茄炒蛋成品糊化",所以工程上有个甜区(5-9)。

Imagine two recipes laid over each other: a vague "make dinner" (uncond) and a specific "make tomato-egg stir-fry" (cond). CFG amplifies the *extras* the specific recipe adds — "add tomato", "beat eggs" — by a factor `scale`, ensuring the dish actually tastes like tomato-egg rather than generic food. Push the scale too high and the dish becomes a tomato-egg caricature; the sweet spot in practice is 5-9.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-WAM

中文:在 nanoWAM 里 CFG 横跨两个文件 —— `nano/wam/train.py` 里训练循环加一行 prompt dropout,`nano/wam/sample.py` 里推理循环加两次 forward + 一次 combine。它的"上游"是文本编码器(给 `context_null` 一份编码),"下游"是 scheduler.step。如果你直接省掉 CFG:模型仍然能生成视频,但 prompt 会被严重忽视 —— prompt benchmark(VBench、EvalCrafter)会差 20-30%。生产实现还要补:(1) **negative prompt**(用 "blurry, low quality" 这种短语代替空字符串,推开 specific 模式)、(2) **CFG scheduling**(只在中间 timesteps 用 CFG,早晚都不用,减少 artifacts)、(3) **dynamic CFG**(根据 timestep 调整 scale)。这些都是这 7 行的扩展,核心数学不变。

English: in nanoWAM, CFG lives in two files — one line of prompt dropout in `nano/wam/train.py` and two forwards plus a combine in `nano/wam/sample.py`. Upstream is the text encoder (which produces both `context` and a once-cached `context_null`); downstream is `scheduler.step`. Omit CFG and you still get videos, but VBench/EvalCrafter scores drop 20-30% on prompt adherence. Production extensions: (1) **negative prompt** — replace the empty string with "blurry, low quality", actively pushing away from bad modes; (2) **interval CFG** — apply CFG only in middle timesteps, none at extremes, to reduce over-saturation; (3) **dynamic CFG** — schedule `scale` over t. All variations of this same seven-line core.

## 自己跑一遍 / Try it yourself

```python
# pip install torch
# 1-D toy: target is N(+5, 1), unconditional is N(0, 1). CFG should push samples toward +5.
import torch

# pretend "scores" — log-derivative of a Gaussian: s(x; mu) = -(x - mu)
def score(x, mu, sigma=1.0):
    return -(x - mu) / sigma**2

x_t = torch.zeros(8)                       # 8 noisy samples sitting at 0
cond_score   = score(x_t, mu=+5.0)         # model with prompt
uncond_score = score(x_t, mu= 0.0)         # model without prompt

for scale in [1.0, 3.0, 7.0]:
    guided = uncond_score + scale * (cond_score - uncond_score)
    x_new  = x_t + 0.2 * guided            # one Euler step
    print(f"scale={scale}: x_t={x_t[0]:.2f} -> x_t+dt={x_new[0]:.2f}")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
scale=1.0: x_t=0.00 -> x_t+dt=1.00
scale=3.0: x_t=0.00 -> x_t+dt=3.00
scale=7.0: x_t=0.00 -> x_t+dt=7.00
```

中文:scale 越大,单步把 `x_t` 推得越接近 prompt 目标(5.0);scale=7 已经过冲了 —— 这正是过饱和的来源,工程上要在 prompt 跟随度和过饱和之间做权衡。

English: higher `scale` drives `x_t` toward the conditional target faster; `scale=7` already overshoots — the classic over-saturation problem. In practice you trade prompt fidelity against artefacts.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Stable Diffusion / SDXL** / **Stable Diffusion / SDXL**: 中文 — 一模一样的两 forward + 一线性组合,只是 context 来自 CLIP 而非 T5。 / English — identical two-forward + linear combine, with CLIP-encoded context.
- **Imagen, PixArt-α, Sora** / **Imagen, PixArt-α, Sora**: 中文 — 全是同一个 trick,这是文本到视觉生成模型的通用件。 / English — all use the same trick; it's a universal text-to-visual generation primitive.
- **AudioLDM / MusicGen** / **AudioLDM / MusicGen**: 中文 — 音频生成也是同样的 CFG,context 换成 text 或音乐风格 embedding。 / English — audio generation runs the same CFG, swapping in audio-style or text embeddings.
- **昨天的 Causal-Forcing DMD** / **Yesterday's Causal-Forcing DMD**: 中文 — 训练蒸馏时,`real_score` 和 `fake_score` 各自内部都用 CFG(`real_guidance_scale`、`fake_guidance_scale`)—— CFG 已经下沉到了蒸馏目标里。 / English — DMD distillation internally uses CFG on both the real and fake score networks (`real_guidance_scale`, `fake_guidance_scale`). CFG has soaked into the distillation objective itself.

## 注意事项 / Caveats / when it breaks

- **训练没做 dropout = CFG 无效** / **No dropout in training = CFG does nothing**: 中文 — 如果训练没用 10% 概率把 context 换成 null,推理时 `context_null` 对模型没有意义,uncond pass 直接吐垃圾。先做训练,再做 CFG。 / English — if you never train with null context (10% dropout), the unconditional forward returns garbage and CFG amplifies that garbage. Fix training first.
- **scale 太大过饱和** / **High scale = saturation**: 中文 — scale > 12 在大多数模型上会让颜色饱和、动作僵硬。工程甜区 5-9。 / English — `scale > 12` typically produces over-saturated colours and stiff motion. The sweet spot is 5-9.
- **2× compute** / **2× compute at inference**: 中文 — 推理时间直接翻倍。蒸馏方法(如昨天的 Causal-Forcing / DMD)就是为了把 CFG 内化进单 forward 来省这倍计算。 / English — inference cost literally doubles. Distillation methods like yesterday's Causal-Forcing/DMD exist specifically to fold CFG into a single forward pass.
- **context_null 应该是空 prompt 而非全 0** / **`context_null` is the encoded empty prompt, not zeros**: 中文 — 直接 `torch.zeros` 不行,模型没见过这种"空 embedding"。要真用 T5 编码空字符串。 / English — don't shortcut by passing `torch.zeros`. The model has never seen "zero embedding" during training. Encode the empty string (or your negative prompt) through the actual text encoder.

## 延伸阅读 / Further reading

- [Classifier-Free Diffusion Guidance (Ho & Salimans, 2022)](https://arxiv.org/abs/2207.12598)
- [Wan2.1 inference script — full sampling loop](https://github.com/Wan-Video/Wan2.1/blob/main/wan/text2video.py)
- [Negative-prompt practical guide (HF)](https://huggingface.co/docs/diffusers/using-diffusers/img2img#negative-prompts)
- [Interval CFG — On Distillation of Guided Diffusion Models (Meng et al., 2023)](https://arxiv.org/abs/2210.03142)
