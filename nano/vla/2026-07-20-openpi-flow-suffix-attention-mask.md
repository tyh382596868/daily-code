---
date: 2026-07-20
topic: vla
source: vla
repo: Physical-Intelligence/openpi
file: src/openpi/models/pi0.py
permalink: https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/models/pi0.py#L176-L241
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, vla, openpi, flow-matching, attention-mask]
build_role: action-head-continuous advanced variant
---

# openpi pi0：prefix 进缓存，suffix 带噪动作迭代去噪 / openpi pi0: Cache the Prefix, Denoise Noisy Action Suffixes

> **一句话 / In one line**: pi0 先把图像/语言 prefix 写进 KV cache，每个采样步只重算 noisy action suffix，并拼出 suffix 看 prefix 的 attention mask。 / pi0 pre-fills image/language prefix into KV cache, then recomputes only the noisy action suffix at each sampling step with a mask that lets suffix tokens attend to the prefix.

## 为什么重要 / Why this matters

连续动作 VLA 的推理瓶颈常在“同一段观测要重复参与多步去噪”。openpi 把观测 prefix 和动作 suffix 拆开：prefix 只算一次，suffix 每个 ODE/Euler 步更新。这是把 flow-matching action head 做成可部署推理循环的关键。

Continuous-action VLA inference often wastes compute by reprocessing the same observation during every denoising step. openpi separates observation prefix from action suffix: prefill the prefix once, update the suffix at every ODE/Euler step. That is the key to making a flow-matching action head deployable.

## 代码 / The code

`Physical-Intelligence/openpi` — [`src/openpi/models/pi0.py`](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/models/pi0.py#L176-L241)

```python
# first fill KV cache with a forward pass of the prefix
prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
positions = jnp.cumsum(prefix_mask, axis=1) - 1
_, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

def step(carry):
    x_t, time = carry
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
        observation, x_t, jnp.broadcast_to(time, batch_size)
    )
    suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
    full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
    assert full_attn_mask.shape == (
        batch_size,
        suffix_tokens.shape[1],
        prefix_tokens.shape[1] + suffix_tokens.shape[1],
    )
    positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

    (prefix_out, suffix_out), _ = self.PaliGemma.llm(
        [None, suffix_tokens],
        mask=full_attn_mask,
        positions=positions,
        kv_cache=kv_cache,
        adarms_cond=[None, adarms_cond],
    )
    assert prefix_out is None
    v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

    return x_t + dt * v_t, time + dt
```

## 逐行讲解 / What's happening

1. **prefix 只预填一次 / Prefill prefix once**: 中文: 图像、文本、state 等观测 token 先进入 `kv_cache`。 English: image, text, and state observation tokens are written into `kv_cache` first.
2. **suffix 每步重算 / Recompute suffix each step**: 中文: `x_t` 是当前 noisy action，时间步不同，suffix embedding 也不同。 English: `x_t` is the current noisy action; different times produce different suffix embeddings.
3. **suffix 内部 mask / Mask inside the suffix**: 中文: `make_attn_mask` 控制动作 token 之间能否互相看。 English: `make_attn_mask` controls how action tokens attend to each other.
4. **suffix 看 prefix / Suffix attends to prefix**: 中文: `einops.repeat(prefix_mask, "b p -> b s p")` 把 prefix 可见性复制给每个 suffix query。 English: `einops.repeat(prefix_mask, "b p -> b s p")` copies prefix visibility for every suffix query.
5. **位置接在 prefix 后面 / Positions continue after prefix**: 中文: suffix position 从 prefix 有效长度之后开始。 English: suffix positions start after the valid prefix length.
6. **只读 suffix 输出 / Read only suffix outputs**: 中文: prefix 用 cache，不再产生输出；动作速度 `v_t` 来自最后的 action horizon。 English: the prefix is cached and produces no output; action velocity `v_t` comes from the final action horizon.

## 类比 / The analogy

像翻译会议记录。会议背景材料先读进脑子，之后每次只重写“下一段动作草稿”；草稿可以引用背景材料，但不用每次把背景材料重读一遍。

It is like translating meeting notes. You read the background material once, then rewrite only the next action draft each time; the draft can reference the background without rereading it.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

这是 `action-head-continuous` 的推理侧高级形态，依赖前面的 `vlm-backbone-wiring` 和 prefix/suffix token schema。上游是 observation encoder 和 noisy action sampler，下游是 ODE 更新器。nanoVLA 可以先不用 KV cache，直接全序列 forward；生产版本需要像这里一样缓存 prefix，否则多步采样会把延迟放大十倍。

This is an inference-side advanced form of `action-head-continuous`, depending on `vlm-backbone-wiring` and a prefix/suffix token schema. Upstream are the observation encoder and noisy-action sampler; downstream is the ODE updater. A nanoVLA can start with full-sequence forwards, but production needs prefix caching like this or multi-step sampling multiplies latency.

## 自己跑一遍 / Try it yourself

```python
prefix_mask = [[1, 1, 0]]
suffix_mask = [[1, 1]]

def repeat_prefix(prefix, suffix_len):
    return [prefix[:] for _ in range(suffix_len)]

def suffix_causal(n):
    return [[j <= i for j in range(n)] for i in range(n)]

full = [p + s for p, s in zip(repeat_prefix(prefix_mask[0], 2), suffix_causal(2))]
print(full)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[[1, 1, 0, True, False], [1, 1, 0, True, True]]
```

每个 suffix query 都能看有效 prefix token，同时 suffix 内部仍保持自己的因果结构。

Each suffix query can see valid prefix tokens while preserving its own causal structure.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LeRobot pi0 cache clone** / **LeRobot pi0 cache clone**: prefix cache 可复用，但每次 rollout 要隔离可变状态。 / Prefix cache is reusable, but mutable state must be isolated per rollout.
- **FAST action tokenizer** / **FAST action tokenizer**: 离散动作也把观测 prompt 和动作 token 分成 prefix/suffix。 / Discrete actions also split observation prompt and action tokens into prefix and suffix.

## 注意事项 / Caveats / when it breaks

- **mask shape 必须精确 / Mask shape must be exact**: suffix query 的 key 长度是 `prefix_len + suffix_len`。 / A suffix query keys over `prefix_len + suffix_len` tokens.
- **position 不能从 0 重来 / Positions must not restart at zero**: suffix position 要接在 prefix 后，否则 RoPE/position 会冲突。 / Suffix positions must continue after prefix or positional encodings collide.
- **cache 是观测条件的一部分 / Cache belongs to the observation**: 换了图像或语言指令就必须重新 prefill。 / Change the image or instruction and the prefix must be prefilled again.

## 延伸阅读 / Further reading

- [openpi pi0.py](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/models/pi0.py#L176-L241)
- [openpi repository](https://github.com/Physical-Intelligence/openpi)
