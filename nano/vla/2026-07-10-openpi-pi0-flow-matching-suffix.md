---
date: 2026-07-10
topic: vla
source: vla
repo: Physical-Intelligence/openpi
file: src/openpi/models/pi0.py
permalink: https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/models/pi0.py#L141-L214
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, flow-matching, action-head]
build_role: action-head-continuous advanced variant
---

# openpi pi0 suffix：把 noisy action 和时间步接进同一次 forward / openpi pi0 Suffix: Feed Noisy Actions and Time into One Forward Pass

> **一句话 / In one line**: pi0 把 noisy actions 编成 suffix token，用 flow-matching 目标 `noise - actions` 监督 action head。 / pi0 encodes noisy actions as suffix tokens and supervises the action head with the flow-matching target `noise - actions`.

## 为什么重要 / Why this matters

VLA 的动作头不是单独的小 MLP。这里动作、状态、图像和语言一起进 PaliGemma：prefix 负责观察和指令，suffix 负责带时间步的动作去噪。训练目标不是直接回归动作，而是回归从当前 noisy action 指向噪声的速度场。

A VLA action head is not just a separate small MLP. Here actions, state, images, and language enter PaliGemma together: the prefix carries observation and instruction, while the suffix carries time-conditioned noisy actions. The target is not the action itself, but the velocity field from current noisy action toward noise.

## 代码 / The code

`Physical-Intelligence/openpi` — [`src/openpi/models/pi0.py`](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/models/pi0.py#L141-L214)

```python
def embed_suffix(
    self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
) -> tuple[
    at.Float[at.Array, "b s emb"],
    at.Bool[at.Array, "b s"],
    at.Bool[at.Array, " s"],
    at.Float[at.Array, "b emb"] | None,
]:
    input_mask = []
    ar_mask = []
    tokens = []
    if not self.pi05:
        # add a single state token
        state_token = self.state_proj(obs.state)[:, None, :]
        tokens.append(state_token)
        input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
        # image/language inputs do not attend to state or actions
        ar_mask += [True]

    action_tokens = self.action_in_proj(noisy_actions)
    # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
    time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
    if self.pi05:
        # time MLP (for adaRMS)
        time_emb = self.time_mlp_in(time_emb)
        time_emb = nnx.swish(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        time_emb = nnx.swish(time_emb)
        action_expert_tokens = action_tokens
        adarms_cond = time_emb
    else:
        # mix timestep + action information using an MLP (no adaRMS)
        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
        action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
        action_time_tokens = self.action_time_mlp_in(action_time_tokens)
        action_time_tokens = nnx.swish(action_time_tokens)
        action_time_tokens = self.action_time_mlp_out(action_time_tokens)
        action_expert_tokens = action_time_tokens
        adarms_cond = None
    tokens.append(action_expert_tokens)
    input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
    # image/language/state inputs do not attend to action tokens
    ar_mask += [True] + ([False] * (self.action_horizon - 1))
    tokens = jnp.concatenate(tokens, axis=1)
    input_mask = jnp.concatenate(input_mask, axis=1)
    ar_mask = jnp.array(ar_mask)
    return tokens, input_mask, ar_mask, adarms_cond

def compute_loss(self, rng, observation, actions, *, train: bool = False):
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
    x_t = time[..., None, None] * noise + (1 - time[..., None, None]) * actions
    u_t = noise - actions
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask = make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1
    (prefix_out, suffix_out), _ = self.PaliGemma.llm(
        [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
    )
    v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
    return jnp.mean(jnp.square(v_t - u_t), axis=-1)
```

## 逐行讲解 / What's happening

1. **第 151-157 行 / Lines 151-157 (state token)**:
   - 中文: 旧版 pi0 把当前 state 加成一个 suffix token，给 action expert 一个低维机器人状态入口。
   - English: Older pi0 adds current state as a suffix token, giving the action expert a low-dimensional robot-state input.
2. **第 159-177 行 / Lines 159-177 (action plus time)**:
   - 中文: noisy action 先投影，再和 sin-cos 时间步融合；pi0.5 则把时间步改成 adaRMS 条件。
   - English: Noisy actions are projected first, then fused with sin-cos time; pi0.5 moves the time signal into adaRMS conditioning.
3. **第 195-200 行 / Lines 195-200 (`x_t`, `u_t`)**:
   - 中文: `x_t` 是动作和噪声的线性插值，`u_t` 是 flow matching 要预测的速度。
   - English: `x_t` is a linear interpolation between action and noise, and `u_t` is the flow-matching velocity target.
4. **第 202-214 行 / Lines 202-214 (one forward pass)**:
   - 中文: prefix 和 suffix 一起进 LLM，最后只取 action horizon 的 suffix 输出做动作速度预测。
   - English: Prefix and suffix enter the LLM together, and only the action-horizon suffix outputs are projected into action velocity.

## 类比 / The analogy

这像给机器人一张“当前动作草稿”和一个“修改进度条”。模型不是直接写最终答案，而是学每个进度条位置该把草稿往哪个方向改。

It is like giving the robot a draft action and an editing progress bar. The model does not write the final answer directly; it learns which direction to move the draft at each progress value.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

这是 `action-head-continuous` 的高级版本，依赖前面的 `vlm-backbone-wiring` 和 action horizon 表示。上游提供图像/语言 prefix、state 和目标 action；这里把 noisy action 变成可被 transformer 读取的 suffix；下游采样器用同一个 head 从噪声一步步积分回动作。

This is an advanced `action-head-continuous` component, depending on earlier `vlm-backbone-wiring` and action-horizon representation. Upstream provides image/language prefix, state, and target actions; this block turns noisy actions into transformer-readable suffix tokens; downstream sampling integrates from noise back to actions with the same head.

## 自己跑一遍 / Try it yourself

```python
actions = [0.0, 1.0, 2.0]
noise = [3.0, 3.0, 3.0]
time = 0.25
x_t = [time * n + (1 - time) * a for a, n in zip(actions, noise)]
u_t = [n - a for a, n in zip(actions, noise)]
print([round(x, 2) for x in x_t])
print(u_t)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[0.75, 1.5, 2.25]
[3.0, 2.0, 1.0]
```

第一行是带噪动作，第二行是模型要学的速度方向。

The first line is the noisy action, and the second is the velocity direction the model learns.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LeRobot DiffusionPolicy** / **LeRobot DiffusionPolicy**: 也从 noisy action chunk 预测去噪方向。 / It also predicts denoising directions from noisy action chunks.
- **GR00T flow-matching heads** / **GR00T flow-matching heads**: 同样把时间步作为动作 head 的条件。 / They also condition action heads on diffusion or flow time.

## 注意事项 / Caveats / when it breaks

- **mask 形状必须和 token 拼接一致** / **Mask shape must match token packing**: prefix/suffix mask、positions 和 token 长度任何一个错位都会污染 attention。 / Any mismatch among prefix/suffix masks, positions, and token lengths corrupts attention.
- **Beta 采样不是随便选的** / **The beta schedule is intentional**: `Beta(1.5, 1)` 会改变训练时看到的噪声时间分布。 / `Beta(1.5, 1)` changes which noise levels the model sees during training.

## 延伸阅读 / Further reading

- [openpi `Pi0.embed_suffix` and `compute_loss`](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/models/pi0.py#L141-L214)
