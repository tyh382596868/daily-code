---
date: 2026-09-20
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/policies/eo1/modeling_eo1.py
permalink: https://github.com/huggingface/lerobot/blob/5aa74557f84c54d4b458f8b9643c5aa2982acfed/src/lerobot/policies/eo1/modeling_eo1.py#L426-L555
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, vla, eo1, flow-matching, multimodal-training]
build_role: training-step advanced variant
---

# EO-1 双监督训练：动作 flow loss 和文本 loss 共用一次 backbone / EO-1 Dual-Supervision Training: One Backbone, Action Flow Loss and Text Loss

> **一句话 / In one line**: EO-1 把动作 placeholder、带噪 action chunk 和文本 token 放进同一条多模态序列，一次 Qwen backbone forward 同时得到 flow-matching loss 与 text cross entropy。 / EO-1 places action placeholders, a noisy action chunk, and text tokens in one multimodal sequence, then gets flow-matching loss and text cross entropy from one Qwen backbone forward.

## 为什么重要 / Why this matters

VLA 训练经常面临一个结构选择：视觉语言主干只做理解，再接一个动作 head；或者让主干同时承担语言建模和动作条件建模。EO-1 的 `forward` 选择第二条路，但没有为动作和文本各跑一遍 backbone，而是先把动作时间步与 noisy action embedding scatter 进 placeholder，再共享 hidden states。

VLA training often faces a structural choice: use the vision-language backbone only for understanding and attach an action head, or make the backbone model both language and action conditioning. EO-1 chooses the second path without running the backbone twice: it scatters timestep and noisy-action embeddings into placeholders, then shares the hidden states.

这段代码特别适合学习训练 step 的真实边界：只有包含 action placeholder 的 row 才采样 `time`、`noise` 和 `u_t`；padding action 可以从 attention 和 loss 两处分别屏蔽；文本 loss 采用 shifted labels；动作 loss 则从 action token hidden states 投影到 velocity，再和 `noise - active_action` 做 MSE。

This is a useful view of a real training-step boundary: only rows with action placeholders sample `time`, `noise`, and `u_t`; padded actions can be masked in both attention and loss; text uses shifted labels; action hidden states are projected to velocity and compared with `noise - active_action` using MSE.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/eo1/modeling_eo1.py`](https://github.com/huggingface/lerobot/blob/5aa74557f84c54d4b458f8b9643c5aa2982acfed/src/lerobot/policies/eo1/modeling_eo1.py#L426-L555)

```python
def forward(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.LongTensor | None = None,
    pixel_values: torch.FloatTensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    mm_token_type_ids: torch.IntTensor | None = None,
    states: torch.FloatTensor | None = None,
    action: torch.FloatTensor | None = None,
    action_is_pad: torch.BoolTensor | None = None,
    text_labels: torch.LongTensor | None = None,
    *,
    state_token_id: int,
    action_token_id: int,
    **kwargs,
) -> EO1Output:
    """Run EO1 flow and sparse assistant-token supervision in one sequence."""
    inputs_embeds = self.embed_prefix(
        input_ids,
        states=states,
        state_token_id=state_token_id,
        action_token_id=action_token_id,
    )
    _, action_mask = self.get_placeholder_mask(
        input_ids,
        inputs_embeds,
        state_token_id=state_token_id,
        action_token_id=action_token_id,
    )
    action_token_mask = action_mask[..., 0]
    action_rows = action_token_mask.any(dim=-1)
    u_t = None
    if action_rows.any():
        active_action = action[action_rows]
        time = self.sample_time(active_action.shape[0], inputs_embeds.device)
        noise = self.sample_noise(active_action.shape, inputs_embeds.device)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * active_action
        u_t = noise - active_action
        action_time_embs = self.embed_suffix(time, x_t)
        expected_tokens = int(action_token_mask.sum().item())
        if expected_tokens != action_time_embs.shape[0] * action_time_embs.shape[1]:
            raise ValueError("EO-1 requires one action placeholder per supervised horizon step.")
        inputs_embeds = inputs_embeds.masked_scatter(
            action_mask,
            action_time_embs.to(inputs_embeds.device, inputs_embeds.dtype),
        )

    active_action_is_pad = None
    if action_rows.any() and not self.config.supervise_padding_actions:
        active_action_is_pad = action_is_pad[action_rows].to(
            device=inputs_embeds.device, dtype=torch.bool
        )
        action_padding_mask = torch.zeros_like(action_token_mask)
        action_padding_mask = action_padding_mask.masked_scatter(
            action_token_mask,
            active_action_is_pad.reshape(-1),
        )
        attention_mask = attention_mask.masked_fill(action_padding_mask, 0)

    hidden_states = self._apply_checkpoint(
        vlm_forward_func,
        input_ids,
        attention_mask,
        inputs_embeds,
        pixel_values,
        image_grid_thw,
        mm_token_type_ids,
    )
    text_loss = None
    if text_labels is not None:
        shifted_labels = text_labels[:, 1:].contiguous()
        text_mask = shifted_labels != -100
        if text_mask.any():
            text_hidden = hidden_states[:, :-1][text_mask]
            text_hidden = text_hidden.to(self.vlm_backbone.lm_head.weight.dtype)
            text_logits = self.vlm_backbone.lm_head(text_hidden).float()
            text_loss = F.cross_entropy(
                text_logits, shifted_labels[text_mask].to(text_logits.device)
            )

    flow_loss = None
    if action_rows.any():
        assert u_t is not None
        v_t = self._apply_checkpoint(
            action_out_proj_func,
            hidden_states[action_token_mask],
        )
        v_t = v_t.reshape(u_t.shape).to(dtype=u_t.dtype)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        if not self.config.supervise_padding_action_dims:
            original_action_dim = self.config.output_features[ACTION].shape[0]
            losses = losses[..., :original_action_dim]
        if not self.config.supervise_padding_actions:
            losses = losses[~active_action_is_pad]
        flow_loss = losses.mean()

    losses = [value for value in (flow_loss, text_loss) if value is not None]
    return EO1Output(
        loss=sum(losses) if losses else None,
        flow_loss=flow_loss,
        text_loss=text_loss,
    )
```

## 逐行讲解 / What's happening

1. **先建 prefix / Build the prefix first**:
   - 中文: `embed_prefix` 把 state token、action placeholder、图像和文本对齐到统一 embedding 序列。
   - English: `embed_prefix` aligns state tokens, action placeholders, images, and text into one embedding sequence.
2. **只对 active rows 采样 / Sample only active rows**:
   - 中文: `action_rows` 避免纯文本样本无意义地采样 action noise，也让一个 batch 混合语言和控制样本。
   - English: `action_rows` avoids meaningless action-noise sampling for text-only rows and allows mixed language/control batches.
3. **flow-matching 构造 / Flow-matching construction**:
   - 中文: `x_t = t noise + (1-t) action`，监督速度是 `u_t = noise - action`；`embed_suffix` 把二者变成 action token embedding。
   - English: `x_t = t noise + (1-t) action`, with target velocity `u_t = noise - action`; `embed_suffix` turns them into action-token embeddings.
4. **scatter 而不是拼接 / Scatter rather than concatenate**:
   - 中文: `masked_scatter` 精确替换 placeholder 位置，保持多模态序列的原有布局和 attention mask。
   - English: `masked_scatter` replaces exactly the placeholder positions while preserving the multimodal sequence layout and attention mask.
5. **padding 两次处理 / Handle padding twice**:
   - 中文: padding action 先从 backbone attention 中移除，再从最终 MSE 中移除；否则模型会为不存在的动作维度学习。
   - English: Padded actions are removed from backbone attention and from final MSE; otherwise the model learns nonexistent action dimensions.
6. **共享 hidden states / Share hidden states**:
   - 中文: 一次 Qwen forward 后，文本 token 走 lm head，动作 token 走 `action_out_proj`，两个监督信号最后相加。
   - English: After one Qwen forward, text tokens use the LM head and action tokens use `action_out_proj`; the two supervision signals are summed.

## 类比 / The analogy

这像一个双语教练同时看一场训练：他先让学生把“我想做什么”和“我要说什么”写在同一张白板上，只讲解一次，再分别检查语言答案和动作轨迹。共享讲解时间，两个评分表各自保留。

It is like a bilingual coach watching one training session: the student writes “what I should say” and “what I should do” on one board, the coach explains once, then grades language and movement with separate score sheets. The explanation is shared; the objectives stay separate.

## 在 nanoVLA 中的位置 / Where this lives in your nanoVLA

中文: 这是 `training-step` / loss assembly 层，位于 batch collator 和 optimizer 之间。上游提供 tokenized prompt、图像、state、action chunk、padding mask 与 text labels；这里决定哪些 row 进入 flow matching，替换 action placeholder，并让一次 VLM forward 同时产出 `flow_loss` 和 `text_loss`；下游 optimizer 对总 loss 反向传播。做 from-scratch nanoVLA 时，可以先实现单一 action loss，再加入 sparse text loss；真正生产化还要明确 action/text loss 权重、混合精度、gradient checkpoint、有效 action 维度和分布式 reduction。

English: This is the `training-step` / loss-assembly layer between the batch collator and optimizer. Upstream provides tokenized prompts, images, state, action chunks, padding masks, and text labels; this layer chooses flow-matching rows, replaces action placeholders, and produces `flow_loss` plus `text_loss` from one VLM forward; downstream code backpropagates the total. For a from-scratch nanoVLA, start with action loss alone, then add sparse text loss. Production code must define loss weights, mixed precision, checkpointing, valid action dimensions, and distributed reduction.

## 自己跑一遍 / Try it yourself

```python
import torch

action = torch.tensor([[[1.0, 0.0], [0.5, 0.5]]])
noise = torch.zeros_like(action)
t = torch.tensor([[0.25]])
x_t = t[:, :, None] * noise + (1 - t[:, :, None]) * action
target_velocity = noise - action
prediction = torch.zeros_like(target_velocity)
flow_loss = torch.nn.functional.mse_loss(prediction, target_velocity)
print(x_t, target_velocity, float(flow_loss))
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
tensor([[[0.7500, 0.0000],
         [0.3750, 0.3750]]]) tensor([[[-1.0000,  0.0000],
         [-0.5000, -0.5000]]]) 0.375
```

中文: 这是 EO-1 flow 分支的最小数学核心；真实模型只是在它前面加了 token embedding 和 backbone。
English: This is the minimal mathematical core of EO-1's flow branch; the real model adds token embeddings and a backbone around it.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **multitask language models** / **multitask language models**: 中文: 一个 backbone 共享表征，多个 head 分别负责 token、value 或动作目标。 / English: One backbone shares representations while separate heads handle token, value, or action targets.
- **diffusion policy training** / **diffusion policy training**: 中文: 时间步和 noisy action 是训练输入，velocity 或 noise 是监督目标。 / English: Timestep and noisy action are inputs, while velocity or noise is the supervision target.
- **packed multimodal sequences** / **packed multimodal sequences**: 中文: placeholder + mask 让不同 modality 在同一序列中保持位置契约。 / English: Placeholders and masks preserve positional contracts for different modalities in one sequence.

## 注意事项 / Caveats / when it breaks

- **action placeholder 数量必须相等** / **Placeholder count must match**: 中文: `expected_tokens` 检查的是结构契约，不匹配时应立刻报错。 / English: `expected_tokens` checks a structural contract and should fail immediately on mismatch.
- **loss 权重不一定该相加** / **Losses may need weights**: 中文: 代码当前直接相加；真实数据若文本 token 远多于 action token，通常需要显式权重或归一化。 / English: The code adds losses directly; real data may need explicit weighting or normalization when text tokens vastly outnumber action tokens.
- **padding 语义要前后一致** / **Padding semantics must agree**: 中文: attention mask、action dim mask 和 loss reduction 必须使用同一套有效长度定义。 / English: Attention masks, action-dimension masks, and loss reduction must share the same definition of validity.

## 延伸阅读 / Further reading

- [LeRobot EO-1 forward](https://github.com/huggingface/lerobot/blob/5aa74557f84c54d4b458f8b9643c5aa2982acfed/src/lerobot/policies/eo1/modeling_eo1.py#L426-L555)
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [LeRobot policies](https://github.com/huggingface/lerobot/tree/5aa74557f84c54d4b458f8b9643c5aa2982acfed/src/lerobot/policies)
