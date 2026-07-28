---
date: 2026-07-28
topic: wam
source: wam
repo: yuantianyuan01/FastWAM
file: src/fastwam/models/wan22/fastwam.py
permalink: https://github.com/yuantianyuan01/FastWAM/blob/45d8e1458921d83f8ad6cf9ce993d371208dabd0/src/fastwam/models/wan22/fastwam.py#L448-L568
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, training-loop, action-conditioning]
build_role: training-loop advanced variant, dual video/action denoising loss
---

# FastWAM training_loss：视频和动作各自加噪再一起预测 / FastWAM training_loss: Noise Video and Action Separately, Predict Them Together

> **一句话 / In one line**: FastWAM 训练时给 video latents 和 action 序列使用两套 scheduler、两套噪声目标，再通过 MoT 混合注意力联合回归。 / During training, FastWAM uses separate schedulers and noise targets for video latents and action sequences, then jointly regresses them through MoT mixed attention.

## 为什么重要 / Why this matters

WAM 不只是视频扩散模型加一个 action 条件。这里动作本身也是要去噪的预测对象：视频 expert 和 action expert 先各自预处理 token，再进同一个 MoT 注意力层，最后分别算 video loss 和 action loss。

A WAM is not just a video diffusion model with an action condition. Here the action itself is also a denoising target: the video expert and action expert preprocess their tokens separately, pass through one MoT attention module, and then compute separate video and action losses.

## 代码 / The code

`yuantianyuan01/FastWAM` — [`src/fastwam/models/wan22/fastwam.py`](https://github.com/yuantianyuan01/FastWAM/blob/45d8e1458921d83f8ad6cf9ce993d371208dabd0/src/fastwam/models/wan22/fastwam.py#L448-L568)

```python
def training_loss(self, sample, tiled: bool = False):
    inputs = self.build_inputs(sample, tiled=tiled)
    input_latents = inputs["input_latents"]
    batch_size = input_latents.shape[0]
    context = inputs["context"]
    context_mask = inputs["context_mask"]
    action = inputs["action"]
    action_is_pad = inputs["action_is_pad"]
    image_is_pad = inputs["image_is_pad"]

    noise_video = torch.randn_like(input_latents)
    timestep_video = self.train_video_scheduler.sample_training_t(
        batch_size=batch_size,
        device=self.device,
        dtype=input_latents.dtype,
    )
    latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
    target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

    noise_action = torch.randn_like(action)
    timestep_action = self.train_action_scheduler.sample_training_t(
        batch_size=batch_size,
        device=self.device,
        dtype=action.dtype,
    )
    noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
    target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

    video_pre = self.video_expert.pre_dit(
        x=latents,
        timestep=timestep_video,
        context=context,
        context_mask=context_mask,
        action=action,
        fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
    )

    action_pre = self.action_expert.pre_dit(
        action_tokens=noisy_action,
        timestep=timestep_action,
        context=context,
        context_mask=context_mask,
    )

    attention_mask = self._build_mot_attention_mask(
        video_seq_len=video_pre["tokens"].shape[1],
        action_seq_len=action_pre["tokens"].shape[1],
        video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
        device=video_pre["tokens"].device,
    )
    tokens_out = self.mot(
        embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
        attention_mask=attention_mask,
        freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
        context_all={
            "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
            "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
        },
        t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
    )

    pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
    pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
```

## 逐行讲解 / What's happening

1. **第 448-456 行 / Lines 448-456 (`build_inputs`)**:
   - 中文: 输入先被整理成 video latent、context、action 和 padding mask。
   - English: Inputs are normalized into video latents, context, action, and padding masks.
2. **第 458-477 行 / Lines 458-477 (`dual noise`)**:
   - 中文: 视频和动作各采一个 timestep，各自生成 noisy input 和 training target。
   - English: Video and action each sample a timestep and build their own noisy input plus training target.
3. **第 479-493 行 / Lines 479-493 (`two experts`)**:
   - 中文: video expert 处理 latent token，action expert 处理 noisy action token。
   - English: The video expert processes latent tokens, while the action expert processes noisy action tokens.
4. **第 498-528 行 / Lines 498-528 (`MoT`)**:
   - 中文: 两路 token 带着各自 RoPE、context 和时间调制进入同一个混合注意力模块。
   - English: Both token streams enter one mixed-attention module with their own RoPE, context, and time modulation.
5. **第 539-563 行 / Lines 539-563 (`weighted losses`)**:
   - 中文: 视频和动作 loss 分别按 scheduler 权重缩放，再用 lambda 合成总损失。
   - English: Video and action losses are separately weighted by scheduler weights and combined with lambdas.

## 类比 / The analogy

这像双语同传训练：一边听视频语言，一边听动作语言，两边各有噪声和答案，但同一个会议室里的讨论让它们互相对齐。

It is like training simultaneous interpreters for two languages: the video stream and action stream each have noise and targets, but sharing one meeting room lets them align with each other.

## 在 nanoWAM 中的位置 / Where this lives in your nanoWAM

在 nanoWAM 里，这属于 `training-loop` 的高级版本，依赖 `noise-scheduler`、`action-conditioning`、`dit-block` 和 `output-head`。输入是视频 latent、语言 context 和动作序列；输出是 video/action 两个 denoising loss。省掉动作侧去噪会把 action 降级成普通条件，生产级还要补 padding mask、不同 horizon 的对齐和 loss 权重调参。

In nanoWAM this is an advanced `training-loop` variant that depends on `noise-scheduler`, `action-conditioning`, `dit-block`, and `output-head`. It takes video latents, language context, and action sequences, then emits two denoising losses. If action-side denoising is removed, action becomes a passive condition. A production version also needs padding masks, horizon alignment, and loss-weight tuning.

## 自己跑一遍 / Try it yourself

```python
def add_noise(x, noise, t):
    return [(1 - t) * a + t * n for a, n in zip(x, noise)]

def target(x, noise):
    return [n - a for a, n in zip(x, noise)]

video = [1.0, 2.0]
action = [0.2, -0.4]
video_noise = [0.0, 4.0]
action_noise = [1.0, 1.0]

noisy_video = add_noise(video, video_noise, 0.25)
noisy_action = add_noise(action, action_noise, 0.75)
print(noisy_video, target(video, video_noise))
print(noisy_action, target(action, action_noise))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
[0.75, 2.5] [-1.0, 2.0]
[0.8, 0.65] [0.8, 1.4]
```

视频和动作可以在同一个 batch 里使用不同噪声强度，这就是双 scheduler 的核心。

Video and action can use different noise strengths in the same batch; that is the core of the dual-scheduler setup.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **DreamZero action/video timesteps** / **DreamZero action/video timesteps**: 中文: 也把视频噪声和动作噪声解耦。 / English: It also decouples video noise from action noise.
- **GR00T flow action head** / **GR00T flow action head**: 中文: 动作本身作为连续流匹配目标训练。 / English: The action itself is trained as a continuous flow-matching target.

## 注意事项 / Caveats / when it breaks

- **时间轴对齐** / **Timeline alignment**: 中文: action horizon 必须和视频 transition 数能整除，否则监督关系不清楚。 / English: Action horizon must align with video transitions, or supervision becomes ambiguous.
- **loss 权重** / **Loss weights**: 中文: `loss_lambda_video/action` 不平衡会让模型只学其中一路。 / English: Imbalanced `loss_lambda_video/action` can make the model learn only one stream.

## 延伸阅读 / Further reading

- [yuantianyuan01/FastWAM source](https://github.com/yuantianyuan01/FastWAM/blob/45d8e1458921d83f8ad6cf9ce993d371208dabd0/src/fastwam/models/wan22/fastwam.py#L448-L568)
