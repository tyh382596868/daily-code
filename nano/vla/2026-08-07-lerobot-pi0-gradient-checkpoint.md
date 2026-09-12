---
date: 2026-08-07
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/policies/pi0/modeling_pi0.py
permalink: https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi0/modeling_pi0.py#L402-L471
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, training-step, checkpointing]
build_role: training-step advanced variant, memory-saving checkpoint wrapper around image/language embedding
---

# LeRobot pi0 checkpoint wrapper：显存优化要包在组件边界上 / LeRobot pi0 Checkpoint Wrapper: Put Memory Savings on Component Boundaries

> **一句话 / In one line**: pi0 用 `_apply_checkpoint` 统一包住图像和语言 embedding，训练时省显存，推理时直跑。 / pi0 uses `_apply_checkpoint` around image and language embedding so training saves memory while inference runs directly.

## 为什么重要 / Why this matters

VLA 的前缀通常包含多张相机图像和语言 token，embedding 阶段会产生大量 activation。把 checkpointing 藏在每个调用点里会让代码散；LeRobot 先做一个小 wrapper，再让 image/lang embedding 共用它。

A VLA prefix often contains multiple camera images plus language tokens, producing many activations. Hiding checkpointing at every call site makes the code noisy; LeRobot builds one wrapper and reuses it for image and language embedding.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/pi0/modeling_pi0.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi0/modeling_pi0.py#L402-L471)

```python
def _apply_checkpoint(self, func, *args, **kwargs):
    if self.gradient_checkpointing_enabled and self.training:
        return torch.utils.checkpoint.checkpoint(
            func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
        )
    return func(*args, **kwargs)

img_emb = self._apply_checkpoint(image_embed_func, img)
lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)
```

## 逐行讲解 / What's happening

1. **第 402-417 行 / Lines 402-417 (enable flags)**:
   - 中文: 开关不只存在 pi0 外壳，也会传到 language model、vision tower 和 action expert。
   - English: The switch is not only on the pi0 wrapper; it is propagated to the language model, vision tower, and action expert.
2. **第 428-434 行 / Lines 428-434 (single wrapper)**:
   - 中文: 只有训练且开关打开时才 checkpoint；推理路径完全不变。
   - English: Checkpointing is used only in training with the flag enabled; inference stays unchanged.
3. **第 456-461 行 / Lines 456-461 (image boundary)**:
   - 中文: 每个相机图像 embedding 都通过同一个 wrapper。
   - English: Each camera image embedding passes through the same wrapper.
4. **第 468-471 行 / Lines 468-471 (language boundary)**:
   - 中文: 语言 token embedding 也复用同一条显存策略。
   - English: Language-token embedding reuses the same memory policy.

## 类比 / The analogy

像给每个入口装同一种门禁：图像门和文字门都按同一规则放行，不需要每扇门写一份流程。

It is like installing the same access control at every entrance: image and text doors follow one rule instead of carrying separate procedures.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

在 nanoVLA 里，它属于 `training-step` 和 `vlm-backbone-wiring` 之间的工程层。模型结构不变，只改变训练时保存哪些 activation。这样可以先把功能写清楚，再用一个开关决定是否牺牲计算换显存。

In a nanoVLA, this is an engineering layer between `training-step` and `vlm-backbone-wiring`. The model structure stays the same; only training activation storage changes. You can keep the functional path clear and decide with one switch whether to trade compute for memory.

## 自己跑一遍 / Try it yourself

```python
training = True
enabled = True

def apply_checkpoint(name):
    return f"checkpoint({name})" if training and enabled else name

print(apply_checkpoint("image_embed"))
training = False
print(apply_checkpoint("image_embed"))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
checkpoint(image_embed)
image_embed
```

训练和推理共用同一个调用点，但显存策略不同。

Training and inference share one call site, but use different memory policies.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Transformers gradient checkpointing** / **Transformers gradient checkpointing**: 通常也是一个全局开关下发到每层。 / A global switch is often propagated into every layer.
- **torch.compile toggles** / **torch.compile toggles**: 编译策略也常包在模型边界，而不是散在业务逻辑里。 / Compilation policy is also often placed at model boundaries instead of business logic.

## 注意事项 / Caveats / when it breaks

- **RNG 保存被关闭** / **RNG preservation is disabled**: `preserve_rng_state=False` 更快，但含 dropout 的段落要确认可接受。 / `preserve_rng_state=False` is faster, but dropout-heavy segments need review.
- **只在训练启用** / **Training only**: 推理 checkpoint 没有意义，只会增加开销。 / Checkpointing during inference adds cost without memory benefit.

## 延伸阅读 / Further reading

- [LeRobot pi0 modeling](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi0/modeling_pi0.py#L402-L471)

