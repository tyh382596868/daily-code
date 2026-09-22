---
date: 2026-09-22
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/policies/groot/groot_n1_7.py
permalink: https://github.com/huggingface/lerobot/blob/9a6bb61043bac8c14353fcb6ea513b7473c118e3/src/lerobot/policies/groot/groot_n1_7.py#L562-L631
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, vla, groot, backbone-wiring, flow-matching]
build_role: VLM backbone + action expert wiring
---

# GR00T N1.7 接线：VLM token 如何喂给连续动作 head / GR00T N1.7 Wiring: Feeding VLM Tokens into a Continuous Action Head

> **一句话 / In one line**: GR00T N1.7 把 VLM 的视觉语言 token、机器人 state 和带噪 action token 接到同一个 action DiT，再只对动作尾部计算 flow-matching loss。 / GR00T N1.7 feeds VLM tokens, robot state, and noisy action tokens into one action DiT, then computes flow-matching loss only on the action suffix.

## 为什么重要 / Why this matters

“VLM + action head” 真正难的地方不在于把两个模块实例化，而在于契约：VLM 输出什么 shape，state 以什么历史长度进入，action 的噪声时间步如何对齐，action expert 怎样 cross-attend 到 backbone，最后哪些位置参与 loss。这个 `forward` 把整条训练路径压在 65 行里。

The hard part of “VLM plus action head” is not constructing two modules. It is the contract: the VLM feature shape, state-history length, action noise time, cross-attention interface, and the exact positions that contribute to loss. This `forward` compresses that training path into about 65 lines.

这里还有两个生产级细节。第一，`embodiment_id` 会进入 state/action encoder，让一个 head 适配多个机器人。第二，`action_mask` 在最后才乘到逐元素 MSE 上，使 padding 或无效动作维度不污染梯度。

Two production details stand out. First, `embodiment_id` conditions the state and action encoders so one head can serve multiple robots. Second, `action_mask` is applied to elementwise MSE at the end, keeping padding or invalid action dimensions out of the gradient.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/groot/groot_n1_7.py`](https://github.com/huggingface/lerobot/blob/9a6bb61043bac8c14353fcb6ea513b7473c118e3/src/lerobot/policies/groot/groot_n1_7.py#L562-L631)

```python
def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
    backbone_features = self.vlln(backbone_output["backbone_features"])
    backbone_output["backbone_features"] = self.vl_self_attention(backbone_features)
    return backbone_output

def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
    self.set_frozen_modules_to_eval_mode()
    backbone_output = self.process_backbone_output(backbone_output)
    vl_embeds = backbone_output.backbone_features
    device = vl_embeds.device
    embodiment_id = action_input.embodiment_id

    if action_input.state.shape[1] != self.config.state_history_length:
        raise ValueError("state history length does not match GR00T N1.7 config.")
    state = action_input.state.view(action_input.state.shape[0], 1, -1)
    state_features = self.state_encoder(state, embodiment_id)

    if self.training and self.state_dropout_prob > 0:
        do_dropout = (
            torch.rand(state_features.shape[0], device=state_features.device) < self.state_dropout_prob
        )
        state_features = state_features * (1 - do_dropout[:, None, None].to(dtype=state_features.dtype))

    actions = action_input.action
    noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
    t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
    t = t[:, None, None]
    noisy_trajectory = (1 - t) * noise + t * actions
    velocity = actions - noise
    t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
    action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

    if self.config.add_pos_embed:
        pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
        action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

    sa_embs = torch.cat((state_features, action_features), dim=1)
    model_output, _ = self.model(
        hidden_states=sa_embs,
        encoder_hidden_states=vl_embeds,
        encoder_attention_mask=backbone_output.backbone_attention_mask,
        timestep=t_discretized,
        return_all_hidden_states=True,
    )

    pred = self.action_decoder(model_output, embodiment_id)
    pred_actions = pred[:, -actions.shape[1] :]
    action_mask = action_input.action_mask
    action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
    loss = action_loss.sum() / (action_mask.sum() + 1e-6)
    return BatchFeature(
        data={
            "loss": loss,
            "action_loss": action_loss,
            "action_mask": action_mask,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }
    )
```

## 逐行讲解 / What's happening

1. **第 562-565 行 / Lines 562-565 (backbone boundary)**:
   - 中文: 先对 VLM features 做可选 LayerNorm 和轻量 self-attention，再交给动作 head；这让 backbone/action 的维度契约集中在一处。
   - English: VLM features first pass through optional LayerNorm and lightweight self-attention. The backbone/action dimension contract is centralized here.
2. **第 574-583 行 / Lines 574-583 (state and dropout)**:
   - 中文: state history 长度不匹配就立即报错；训练时随机丢 state 是一种条件 dropout，让模型不至于过度依赖 proprioception。
   - English: A mismatched state-history length fails early. Training-time state dropout prevents the model from relying too heavily on proprioception.
3. **第 585-592 行 / Lines 585-592 (flow target)**:
   - 中文: `noisy_trajectory` 是 noise 与真实 action 的插值，`velocity = actions - noise` 是 flow-matching 目标；同一个 `t` 同时喂给 action encoder 和 DiT。
   - English: `noisy_trajectory` interpolates noise and data, while `velocity = actions - noise` is the flow-matching target. The same `t` conditions both the action encoder and the DiT.
4. **第 598-616 行 / Lines 598-616 (cross-attention)**:
   - 中文: state token 和 action token 在 self-attention 序列里拼接，VLM token 作为 `encoder_hidden_states` 被 action expert 查询。
   - English: State and action tokens share the DiT self-attention sequence, while VLM tokens arrive as `encoder_hidden_states` for cross-attention.
5. **第 618-631 行 / Lines 618-631 (masked suffix loss)**:
   - 中文: decoder 输出可能包含 state 位置，但只取最后的 action 长度；mask 后再归一化，避免 padding 改变 loss 尺度。
   - English: The decoder may produce positions corresponding to state tokens, so only the final action horizon is kept. Masking happens before normalization to keep padding from changing loss scale.

## 类比 / The analogy

把 VLM 想成看过现场的领班，state 是机器人当前姿势，noisy action 是被打乱的施工计划。action DiT 不需要重新看完整现场，只要查询领班的摘要，再把当前姿势和待修正的计划排在桌上，最后只检查动作部分。

Think of the VLM as a foreman who has inspected the scene, the state as the robot's current posture, and the noisy action as a scrambled work plan. The action DiT queries the foreman's summary, lays posture and plan on one table, and grades only the action portion.

## 在 nanoVLA 中的位置 / Where this lives in your nanoVLA

这是 `vlm-backbone-wiring` 组件，依赖前面的 `modality-projector`：视觉 token 已经被投影到语言模型空间后，才有资格作为 `encoder_hidden_states` 进入 action expert。输入是 VLM token、attention mask、state history、embodiment id 和 action chunk；输出是 action loss 或预测 action。

This is the `vlm-backbone-wiring` component and depends on `modality-projector`: visual tokens must already live in the language-model space before they can become `encoder_hidden_states` for the action expert. Inputs are VLM tokens, an attention mask, state history, an embodiment ID, and an action chunk; outputs are an action loss or predicted actions.

从零实现时，可以先用一个小 Transformer 代替 VLM，再实现 `state_encoder`、`action_encoder` 和一个 cross-attention DiT。省略这层接线，VLM 和动作模型只能各自工作，语言与视觉上下文不会进入动作预测。生产版本还需要处理多相机 mask、不同 embodiment 的 action width、冻结模块的 eval 行为、mixed precision 和推理缓存。

For a from-scratch implementation, start with a small Transformer in place of the VLM, then add `state_encoder`, `action_encoder`, and a cross-attention DiT. Without this wiring, the VLM and action model remain separate and language/vision context never reaches control. Production adds multi-camera masks, per-embodiment action widths, frozen-module eval behavior, mixed precision, and inference caching.

## 自己跑一遍 / Try it yourself

```python
import torch

B, T, D = 2, 4, 3
vlm = torch.randn(B, 6, 8)
state = torch.randn(B, 1, 8)
action = torch.ones(B, T, 8)
noise = torch.zeros_like(action)
t = torch.tensor([0.25, 0.75])[:, None, None]
noisy = (1 - t) * noise + t * action
tokens = torch.cat([state, noisy], dim=1)
target = action - noise
pred = torch.zeros_like(action)
mask = torch.tensor([[[1.0] * D + [0.0] * (8 - D)] * T] * B)
loss = (((pred - target) ** 2) * mask).sum() / (mask.sum() + 1e-6)
print(tokens.shape, vlm.shape, round(loss.item(), 4))
```

运行 / Run with:

```bash
pip install torch
python try.py
```

预期输出 / Expected output:

```text
torch.Size([2, 5, 8]) torch.Size([2, 6, 8]) 1.0
```

中文: 这里用全一 action、全零 noise 和零预测，让 masked MSE 稳定为 1；真正的 action expert 要学习 `action - noise`。 / English: Unit actions, zero noise, and a zero predictor make the masked MSE stably equal to 1; a real action expert must learn `action - noise`.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **SmolVLA prefix embedding** / **SmolVLA prefix embedding**: 中文: 把图像、语言和 state 排成统一 prefix，再交给后续动作模块。 / English: It arranges image, language, and state into a shared prefix before the action module.
- **openpi suffix embedding** / **openpi suffix embedding**: 中文: 把 action、state、time 做成 suffix token，与 VLM prefix 形成另一种接线。 / English: It builds action, state, and time as suffix tokens, offering a different wiring around the VLM prefix.
- **StarVLA QwenGR00T** / **StarVLA QwenGR00T**: 中文: 同样把 Qwen hidden states 交给 flow-matching action head，但把 framework 配置和多模型 registry 放得更显式。 / English: It sends Qwen hidden states into a flow-matching action head while making framework configuration and model registries more explicit.

## 注意事项 / Caveats / when it breaks

- **VLM mask 不能丢** / **Do not drop the VLM mask**: 中文: padding token 进入 cross-attention 会把无效上下文当成视觉证据。 / English: Letting padding tokens into cross-attention turns invalid context into apparent visual evidence.
- **flow time 要统一** / **Keep flow time consistent**: 中文: action encoder、DiT 和 loss target 必须共享同一个时间语义。 / English: The action encoder, DiT, and target must share one time semantics.
- **embodiment 不是装饰字段** / **Embodiment is not metadata**: 中文: 不同机器人 action width 或 state layout 不同，不能只把 id 记录下来却不进入 encoder。 / English: Robots differ in action width and state layout; recording an ID without conditioning the encoder is not enough.

## 延伸阅读 / Further reading

- [LeRobot GR00T N1.7 action head](https://github.com/huggingface/lerobot/blob/9a6bb61043bac8c14353fcb6ea513b7473c118e3/src/lerobot/policies/groot/groot_n1_7.py#L562-L631)
- [LeRobot policy documentation](https://github.com/huggingface/lerobot/tree/9a6bb61043bac8c14353fcb6ea513b7473c118e3/src/lerobot/policies)
- [Flow matching for generative modeling](https://arxiv.org/abs/2210.02747)
