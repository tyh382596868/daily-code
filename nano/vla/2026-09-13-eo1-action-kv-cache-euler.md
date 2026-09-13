---
date: 2026-09-13
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/policies/eo1/modeling_eo1.py
permalink: https://github.com/huggingface/lerobot/blob/b6ec0060779550c0a157ae34feb89e0cf86012a8/src/lerobot/policies/eo1/modeling_eo1.py#L557-L661
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, vla, inference-loop, action-chunking, flow-matching, kv-cache]
build_role: inference-loop advanced variant
---

# EO-1 动作采样：前缀只算一次，动作块反复去噪 / EO-1 Action Sampling: Cache the Prefix, Denoise Only the Action Chunk

> **一句话 / In one line**: EO-1 把语言、图像和 state 前缀 prefill 到 KV cache，然后只对 action span 做 Euler flow-matching 迭代。 / EO-1 prefills the language, image, and state prefix into a KV cache, then iterates Euler flow matching only over the action span.

## 为什么重要 / Why this matters

中文：VLA 的动作采样通常要跑多个 denoise step。如果每一步都重新计算整段视觉语言上下文，推理成本会随着上下文长度重复增长。EO-1 先定位连续的 action token chunk，固定前缀只跑一次，再让每个 timestep 只更新动作 embedding 和对应的 cache suffix。

English: VLA action sampling often takes multiple denoising steps. Recomputing the entire vision-language context at every step makes cost grow with the context length repeatedly. EO-1 locates a contiguous action-token chunk, runs the fixed prefix once, then updates only action embeddings and the cache suffix at each timestep.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/eo1/modeling_eo1.py`](https://github.com/huggingface/lerobot/blob/b6ec0060779550c0a157ae34feb89e0cf86012a8/src/lerobot/policies/eo1/modeling_eo1.py#L557-L661)

```python
@torch.no_grad()
def sample_actions(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    pixel_values: torch.Tensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    mm_token_type_ids: torch.IntTensor | None = None,
    states: torch.Tensor | None = None,
    *,
    state_token_id: int,
    action_token_id: int,
    **kwargs,
) -> Tensor:
    """Sample actions from the model."""
    if states is None:
        raise ValueError("states are required for EO1 action sampling.")
    if mm_token_type_ids is None:
        raise ValueError("mm_token_type_ids are required for EO1 action sampling.")

    # 1. Resolve the left-padded rollout prompt and locate the action span.
    chunk_size = self.config.chunk_size

    inputs_embeds = self.embed_prefix(
        input_ids,
        states=states,
        state_token_id=state_token_id,
        action_token_id=action_token_id,
    ).clone()
    _, action_placeholder_mask = self.get_placeholder_mask(
        input_ids,
        inputs_embeds,
        state_token_id=state_token_id,
        action_token_id=action_token_id,
    )
    action_mask = action_placeholder_mask[..., 0]
    token_counts = action_mask.sum(dim=1)
    if not torch.all(token_counts == chunk_size):
        raise ValueError(
            f"Each sample must contain exactly {chunk_size} action tokens, got {token_counts.tolist()}."
        )
    if action_mask.ne(action_mask[:1]).any():
        raise ValueError(
            "Batch inference expects all samples to share the same action token mask after left padding."
        )
    act_start = int(action_mask[0].to(torch.int64).argmax().item())
    act_end = act_start + self.config.chunk_size
    if not torch.all(action_mask[:, act_start:act_end]):
        raise ValueError("Action tokens must form a contiguous chunk of length chunk_size.")
    act_slice = slice(act_start, act_end)

    # 2. Encode the fixed prefix once and cache its KV state.
    batch_size = input_ids.shape[0]
    device = inputs_embeds.device
    attention_mask = attention_mask.to(device)
    mm_token_type_ids = mm_token_type_ids.to(device)
    position_ids, _ = self.vlm_backbone.model.get_rope_index(
        input_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
        mm_token_type_ids=mm_token_type_ids,
    )
    position_ids = position_ids.to(device)

    outputs = self.vlm_backbone.model(
        input_ids=input_ids[:, :act_start],
        attention_mask=attention_mask[:, :act_start],
        position_ids=position_ids[..., :act_start],
        inputs_embeds=inputs_embeds[:, :act_start],
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        mm_token_type_ids=mm_token_type_ids[:, :act_start],
        use_cache=True,
        return_dict=True,
    )

    x_t = self.sample_noise(
        (batch_size, chunk_size, self.config.max_action_dim),
        device,
    ).to(dtype=self.action_in_proj.weight.dtype)
    past_key_values = outputs.past_key_values

    # 3. Denoise only the action chunk while keeping the prefix cache invariant.
    def denoise_fn(input_x_t, current_timestep):
        action_time_embs = self.embed_suffix(current_timestep, input_x_t)
        inputs_embeds[:, act_slice] = action_time_embs.to(inputs_embeds.dtype)

        # Keep the prefix KV cache invariant across denoising steps.
        past_key_values.crop(act_start)
        outputs = self.vlm_backbone.model(
            attention_mask=attention_mask[:, :act_end],
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds[:, act_slice],
            position_ids=position_ids[..., act_slice],
            use_cache=True,
            return_dict=True,
        )
        with self.flow_head_autocast_context():
            hidden_states = outputs.last_hidden_state[:, :chunk_size]
            hidden_states = hidden_states.to(dtype=self.action_out_proj.dtype)
            v_t = self.action_out_proj(hidden_states)
        return v_t.reshape(input_x_t.shape).to(input_x_t.dtype)

    x_t = euler_integrate(denoise_fn, x_t, self.config.num_denoise_steps)
    return x_t
```

## 逐行讲解 / What's happening

1. **第 572-575 行 / Lines 572-575 (required inputs)**:
   - 中文：动作采样不是纯文本生成，`states` 和多模态 token type 是模型契约的一部分，缺失时立即失败。
   - English: Action sampling is not plain text generation. `states` and multimodal token types are part of the model contract and fail fast when absent.
2. **第 577-606 行 / Lines 577-606 (action span validation)**:
   - 中文：批量推理要求每个样本的 action mask 一致、数量等于 `chunk_size`，并且连续。这样后面的 slice 才能在 batch 内共享。
   - English: Batched inference requires identical action masks, exactly `chunk_size` action tokens, and contiguity. That makes one shared slice valid for the whole batch.
3. **第 608-637 行 / Lines 608-637 (prefix prefill)**:
   - 中文：先计算 RoPE 位置，再把 `:act_start` 的固定前缀送入 backbone 并开启 `use_cache=True`。随机动作轨迹从正确的 dtype 开始。
   - English: EO-1 computes RoPE positions, runs the fixed `:act_start` prefix with `use_cache=True`, and initializes the noisy action trajectory in the model's input dtype.
4. **第 639-645 行 / Lines 639-645 (cache invariant)**:
   - 中文：每次 denoise 前只改 action slice；`past_key_values.crop(act_start)` 丢掉上一次 action suffix，保留前缀 KV 不变。
   - English: Each denoising call changes only the action slice. `past_key_values.crop(act_start)` removes the previous action suffix while preserving the prefix KV.
5. **第 646-661 行 / Lines 646-661 (action-only Euler step)**:
   - 中文：backbone 只接收 action embeddings 和前缀长度的 attention mask，action output head 预测速度场，最后由 `euler_integrate` 把噪声轨迹推进到可执行动作。
   - English: The backbone receives action embeddings plus a prefix-sized attention mask; the action head predicts a velocity field, and `euler_integrate` moves the noisy trajectory toward executable actions.

## 类比 / The analogy

中文：像看一场已经搭好的舞台。灯光、布景和剧本前半段只检查一次并留在后台；每个去噪 step 只替换舞台中央的一小段动作排练，不必把整座剧院重新搭一遍。

English: Imagine a stage that has already been set. Lights, scenery, and the first half of the script are checked once and kept backstage; each denoising step rehearses only the action segment in the center instead of rebuilding the theater.

## 在 nanoVLA 中的位置 / Where this lives in your nanoVLA

中文：这是 `inference-loop` 的 advanced variant，位于多模态 prefix encoder 和连续 action head 之间。输入是带 state/image/language 条件的 prefix、连续 noisy action chunk 和 flow timestep；输出是 `[B, chunk_size, action_dim]`。实现 nanoVLA 时可以先做无 cache 的正确版本，再把固定 prefix 的 KV cache、action suffix crop、Euler solver 拆成独立策略。生产版还要处理 left padding、不同 batch 的 action span、cache mutation、action horizon mask、dtype autocast，以及采样失败时的回退路径。

English: This is an advanced `inference-loop` component between the multimodal prefix encoder and the continuous action head. It consumes a state/image/language-conditioned prefix, a noisy continuous action chunk, and a flow timestep, then returns `[B, chunk_size, action_dim]`. In nanoVLA, start with a correct uncached path, then isolate prefix KV caching, suffix cropping, and the Euler solver. Production code also needs left padding, per-batch action spans, cache mutation, horizon masks, autocast, and fallback behavior.

## 自己跑一遍 / Try it yourself

```python
def sample(prefix, noisy, steps):
    calls = {"prefix": 0, "action": 0}
    calls["prefix"] += 1
    cached_prefix = tuple(prefix)
    x = list(noisy)
    for step in range(steps):
        calls["action"] += 1
        velocity = [value - target for value, target in zip(x, cached_prefix[-len(x):])]
        x = [value - velocity_i / steps for value, velocity_i in zip(x, velocity)]
    return x, calls


print(sample([10, 20, 30], [6, 8, 9], 3))
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
([8.814814814814815, 16.444444444444443, 23.77777777777778], {'prefix': 1, 'action': 3})
```

中文：prefix 只计算一次，action step 执行三次。真实 EO-1 的 `cached_prefix` 是多层 KV，而 `velocity` 来自 backbone 加 action projection。

English: The prefix is computed once while the action step runs three times. EO-1's `cached_prefix` is a multi-layer KV cache, and its `velocity` comes from the backbone plus action projection.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **openpi flow suffix** / **openpi flow suffix**: prefix prefill 后，只重复计算 noisy action suffix。 / Prefill the prefix, then repeatedly compute only the noisy action suffix.
- **nanoVLM generation** / **nanoVLM generation**: prompt KV cache 后逐 token decode。 / Cache prompt KV, then decode one token at a time.
- **FastWAM action cache** / **FastWAM action cache**: 视频 expert 的 K/V 在动作去噪期间复用。 / Reuse video-expert K/V while action denoising proceeds.

## 注意事项 / Caveats / when it breaks

- **action mask 必须连续** / **The action mask must be contiguous**: 非连续 placeholder 不能直接用一个 slice 表达。
- **cache 是可变状态** / **The cache is mutable state**: 不 crop 或错误复用 suffix 会把上一步动作泄漏到下一步。
- **前缀和 action dtype 要对齐** / **Prefix and action dtypes must align**: 混合精度下要显式 cast embedding 和 action head 输入。
- **Euler 不是唯一 solver** / **Euler is not the only solver**: 换 solver 时仍要保持 timestep 参数化和 velocity 定义一致。

## 延伸阅读 / Further reading

- [LeRobot EO-1 action sampling](https://github.com/huggingface/lerobot/blob/b6ec0060779550c0a157ae34feb89e0cf86012a8/src/lerobot/policies/eo1/modeling_eo1.py)
- [OpenPI flow matching](https://github.com/Physical-Intelligence/openpi)
