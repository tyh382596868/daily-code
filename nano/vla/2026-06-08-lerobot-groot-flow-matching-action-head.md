---
date: 2026-06-08
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/policies/groot/action_head/flow_matching_action_head.py
permalink: https://github.com/huggingface/lerobot/blob/49755a3d9e7d43ae93092de8324e75348955afab/src/lerobot/policies/groot/action_head/flow_matching_action_head.py#L268-L345
difficulty: advanced
read_time: ~12 min
tags: [code-of-the-day, vla, action-head, flow-matching, groot]
build_role: continuous action head (flow-matching) — the module that turns VL embeddings + state + noisy action chunk + flow-time into a velocity prediction
---

# 整个 GR00T 的训练步骤就 6 行干净的 flow-matching / The whole GR00T training step is six clean lines of flow matching

> **一句话 / In one line**: 采一个 `t ∈ (0,1]`,把 `noisy = (1-t)·noise + t·action`,模型预测 `velocity = action - noise`,MSE 回归——这就是 GR00T 整个 continuous action head 的训练目标。 / Sample `t ∈ (0,1]`, set `noisy = (1-t)·noise + t·action`, ask the model to regress `velocity = action - noise`, take MSE — that's the entire training objective for GR00T's continuous action head.

## 为什么重要 / Why this matters

VLA 模型的 action head 有三大流派:**离散 token**(OpenVLA)、**diffusion**(π₀)、**flow-matching**(GR00T / SmolVLA)。Flow-matching 在 robotics 里赢的是稳定性和推理速度——只要 4-10 步 Euler 就能从噪声走到动作,远比 1000 步 DDPM 快。lerobot 这版 `FlowmatchingActionHead.forward()` 把整套训练目标压缩到 6 行真正干净的代码,周围全是 wiring(状态编码 / 位置编码 / 注意力 mask)。读懂这 6 行,你就读懂了 GR00T-N1、SmolVLA、Pi0、Pi-VLA 一整族当代 robotics policy 在「怎么算 loss」这一关键步上的全部数学。这也是 nanoVLA 课程里 `action-head-continuous` 这一栏的标准答案。

VLA action heads come in three flavors: **discrete tokens** (OpenVLA), **diffusion** (π₀), and **flow-matching** (GR00T / SmolVLA). In robotics, flow-matching wins on stability and inference latency — 4-10 Euler steps go from noise to action, vs. 1000 DDPM steps. lerobot's `FlowmatchingActionHead.forward()` compresses the full training objective into six clean lines, surrounded by wiring (state encoder, positional embeddings, attention masks). Read those six lines and you've read the loss math behind GR00T-N1, SmolVLA, Pi0, Pi-VLA, and every other modern robotics policy in this family. It's also the textbook answer for the `action-head-continuous` slot in the nanoVLA curriculum.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/groot/action_head/flow_matching_action_head.py`](https://github.com/huggingface/lerobot/blob/49755a3d9e7d43ae93092de8324e75348955afab/src/lerobot/policies/groot/action_head/flow_matching_action_head.py#L268-L345)

```python
def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
    # Set frozen modules to eval
    self.set_frozen_modules_to_eval_mode()

    backbone_output = self.process_backbone_output(backbone_output)

    # Get vision and language embeddings.
    vl_embs = backbone_output.backbone_features
    device = vl_embs.device

    # Get embodiment ID.
    embodiment_id = action_input.embodiment_id

    # Embed state.
    state_features = self.state_encoder(action_input.state, embodiment_id)

    # Embed noised action trajectory.
    actions = action_input.action
    noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
    t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
    t = t[:, None, None]  # shape (B,1,1) for broadcast

    noisy_trajectory = (1 - t) * noise + t * actions
    velocity = actions - noise

    # Convert (continuous) t -> discrete if needed
    t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
    action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

    # Maybe add position embedding.
    if self.config.add_pos_embed:
        pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
        pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
        action_features = action_features + pos_embs

    # Join vision, language, state and action embedding along sequence dimension.
    future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
    sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

    vl_attn_mask = backbone_output.backbone_attention_mask

    model_output = self.model(
        hidden_states=sa_embs,
        encoder_hidden_states=vl_embs,
        encoder_attention_mask=vl_attn_mask,
        timestep=t_discretized,
        return_all_hidden_states=False,
    )
    pred = self.action_decoder(model_output, embodiment_id)
    pred_actions = pred[:, -actions.shape[1] :]

    # Slice out only the action portion of pred and target.
    action_mask = action_input.action_mask
    loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
    loss = loss.sum() / action_mask.sum()
    return BatchFeature(data={"loss": loss})
```

## 逐行讲解 / What's happening

把这个 forward 拆成五个语义阶段:

Five semantic stages:

### Stage 1 — 拿到 VL embeddings / Get VL embeddings (lines 1-7)

- 中文: `backbone_output` 是 VLM (Eagle / Qwen / SigLIP+LLM) 跑过 (图像 + 语言指令) 之后的 hidden states。 `process_backbone_output` 跑了一次 LN + self-attention 把它对 action head 这边做了一次 re-tokenize。
- English: `backbone_output` is hidden states from the VLM (Eagle / Qwen / SigLIP+LLM) after consuming (image + language instruction). `process_backbone_output` applies a LayerNorm + self-attention pass that re-tokenizes them into something the action head can cross-attend to.

### Stage 2 — 采样 flow time + 构造 noisy trajectory / Sample flow-time, build noisy trajectory (lines 8-13) ★

这是论文里**最重要的 6 行**: / This is the **most important six lines** in the paper:

```python
noise = torch.randn(actions.shape, ...)
t = self.sample_time(actions.shape[0], ...)        # ~ Beta(1.5, 1) shifted into (0, 1]
t = t[:, None, None]

noisy_trajectory = (1 - t) * noise + t * actions   # convex combination
velocity = actions - noise                          # target velocity field
```

- 中文: Rectified flow 的核心:把 `noise → action` 想成一条直线,`t` 是这条直线上的位置,`noisy_trajectory` 是直线上的点,`velocity = action - noise` 是直线方向(常数,不依赖 t)。模型只学这个 *方向*,推理时多步 Euler 沿着方向积分就能从 noise 走到 action。
- English: This is the heart of rectified flow. Think of `noise → action` as a straight line in action space. `t` is your position along the line, `noisy_trajectory` is the point at that position, and `velocity = action - noise` is the direction (constant — does not depend on `t`!). The model is trained to predict that *direction* given the point. At inference, you run Euler integration along the predicted directions and walk from noise to action.

- 中文: `sample_time` 用 Beta(1.5, 1) shifted,而不是 uniform[0,1]——这是 SD3 论文里的 logit-normal 替代品,经验上对训练稳定性有帮助 (在低噪声端 t 接近 0 的样本被压缩,因为那里梯度方差大)。
- English: `sample_time` uses a shifted Beta(1.5, 1) instead of `Uniform[0, 1]`. This is SD3's logit-normal alternative — empirically it stabilizes training by down-weighting `t ≈ 0` samples where the gradient variance is highest.

### Stage 3 — Embedding 拼接 / Compose the token sequence (lines 14-20)

- 中文: `(state_features, future_tokens, action_features)` 沿 seq 维 concat。`future_tokens` 是可学习的「placeholder token」,数量 = action horizon——它们的作用是给 DiT 提供一个固定大小的输出 buffer,decoder 从这里抽 action。
- English: Concatenate `(state_features, future_tokens, action_features)` along the sequence dim. `future_tokens` are *learnable* placeholder tokens, count = action horizon. Their role is to give the DiT a fixed-size output buffer the decoder reads from. The actual noisy action embeddings ride alongside, also as tokens.

### Stage 4 — Cross-attention DiT forward / The DiT cross-attends to VL (lines 21-26)

- 中文: 把 `sa_embs` 作为 query 序列,`vl_embs` 作为 key/value 序列,加上 `timestep=t_discretized` 通过 AdaLN 注入 flow time——这就是一个标准 DiT block (`FlowmatchingActionHead.model`)。 `t_discretized` 把连续 `t` 离散化进 1000 个 bucket,因为 AdaLN 喜欢离散 token 输入。
- English: `sa_embs` becomes the query sequence; `vl_embs` becomes K/V; `t_discretized` is fed in as an AdaLN conditioning token. That's a vanilla DiT block — `FlowmatchingActionHead.model`. The discretization (continuous `t` → one of 1000 buckets) is a small efficiency win: AdaLN works on embedded tokens, so a `nn.Embedding(1000)` is cheaper than re-computing a sinusoidal embedding every step.

### Stage 5 — 计算 velocity-regression loss / Velocity-regression MSE (lines 27-31)

- 中文: 解码器吐出预测 velocity,只取最后 `actions.shape[1]` 个 token (这些对应 action positions)。 `action_mask` 用来排除 padding 的 future steps (不是所有 horizon 长度都满的)。loss 就是简单的 masked MSE。
- English: The decoder outputs predicted velocity. Slice the trailing `actions.shape[1]` tokens (the positions that correspond to the action chunk). `action_mask` excludes padded future steps (not every horizon is full-length). Loss is masked MSE.

## 类比 / The analogy

想象你站在体育馆这头,目标是走到球门(`actions`),起点是一个随机被推到的位置(`noise`)。教练在体育馆里告诉你**方向**(`velocity = actions - noise`)。训练时,教练把你随机扔在「起点到球门」的连线上的任何一个点(`(1-t)·noise + t·action`),不告诉你 t 是多少,你必须根据当前位置 + 视觉/语言指令(VL embeds)预测应该走的方向。推理时,教练把你扔在起点(噪声),你按预测的方向走一小步,看看新位置,再预测,再走——4-10 步就到了。

Picture a gym. You start at a random spot on the floor (the noise) and your goal is the goal-line (the action). The coach tells you only the *direction* (`velocity = actions - noise`). During training the coach drops you at a random point along the line from start to goal (`(1-t)·noise + t·action`) without telling you what `t` is — you must predict the right direction from your current position plus a visual + language instruction. At inference, the coach drops you at the start (pure noise), you take a small step in the predicted direction, re-evaluate, step again — 4 to 10 steps and you're at the goal.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-VLA

这是 `nano_vla.action-head-continuous` 这一栏的标杆实现。前置依赖:`vlm-backbone-wiring`(VLM 必须能吐出 `vl_embs` 给 action head cross-attend)。下游消费者:`inference-loop`(streaming obs → action),它会调 `get_action()` 跑 Euler 积分。

在你的 nanoVLA 里,这个 forward 对应一个非常薄的模块:

输入:
- `vl_embs: (B, T_vl, D)` —— 从 VLM backbone 来
- `state: (B, D_state)` —— 机器人当前关节角 / 末端位姿
- `action_chunk: (B, H, D_action)` —— ground truth action,horizon=H

输出:`loss` (scalar) —— 一个 MSE。

最小实现需要:**(1) state encoder MLP**,**(2) action encoder MLP**(吃 `(noisy_action, t)`),**(3) 一个小 DiT block**(给 `sa_embs` 做 self-attention,cross-attend `vl_embs`),**(4) action decoder linear**(回到 `D_action`)。不需要 embodiment_id(那是 GR00T 用来跨机器人共享 backbone 的奢侈品)。如果省掉这个组件,你的 nanoVLA 就只能输出 token-discrete action(像 OpenVLA),不能生成精细的连续控制。

生产级 VLA 还需要加上:LayerNorm 在 input/output、AdaLN 注入 t 而不是直接 concat、`Beta(1.5, 1) timestep schedule`(SD3 / GR00T 同款)、`expand_batch` 复制多份 noise 算 mean loss 降方差(GR00T 用的细节,文件里被我省略了)。

This is the gold-standard implementation of the `nano_vla.action-head-continuous` slot. Prerequisites: `vlm-backbone-wiring` (the VLM must produce `vl_embs` for the action head to cross-attend). Downstream consumer: `inference-loop` (streaming obs → action), which calls `get_action()` to run Euler integration.

In your nanoVLA, this `forward` corresponds to one thin module:

Inputs:
- `vl_embs: (B, T_vl, D)` — from the VLM backbone
- `state: (B, D_state)` — current joint angles or end-effector pose
- `action_chunk: (B, H, D_action)` — ground-truth action chunk of length `H`

Output: `loss` (scalar) — a single MSE.

The minimal implementation needs **(1)** a state encoder MLP, **(2)** an action encoder MLP that ingests `(noisy_action, t)`, **(3)** a small DiT block (self-attend on `sa_embs`, cross-attend on `vl_embs`), and **(4)** an action decoder linear back to `D_action`. You do *not* need `embodiment_id` — that's a GR00T-only luxury for sharing one backbone across many robots. Omit this component and your nanoVLA can only emit token-discrete actions (like OpenVLA), losing fine continuous control.

For a production VLA, add: LayerNorm at input/output, AdaLN conditioning on `t` (instead of concatenating), the `Beta(1.5, 1)` timestep schedule (same as SD3 / GR00T), and `expand_batch` to draw multiple noise samples per (vl_embs, action) pair and average the loss to reduce variance (a GR00T detail I trimmed from the snippet above).

## 自己跑一遍 / Try it yourself

```python
# pip install torch
import torch
import torch.nn as nn
import torch.nn.functional as F

class NanoFlowHead(nn.Module):
    """The skeleton of GR00T's action head, stripped to one transformer block."""
    def __init__(self, d=128, H=16, action_dim=7, n_heads=4, T_bucket=1000):
        super().__init__()
        self.H, self.action_dim = H, action_dim
        self.state_enc = nn.Linear(action_dim, d)
        self.action_enc = nn.Linear(action_dim, d)
        self.t_emb = nn.Embedding(T_bucket, d)
        self.pos = nn.Embedding(H, d)
        self.block = nn.TransformerDecoderLayer(d, n_heads, batch_first=True)
        self.decoder = nn.Linear(d, action_dim)

    def forward(self, vl_embs, state, action_chunk):
        B = action_chunk.shape[0]
        noise = torch.randn_like(action_chunk)
        t = torch.rand(B, device=action_chunk.device)              # uniform for the demo
        t_b = (t * 1000).long().clamp(max=999)
        noisy = (1 - t[:, None, None]) * noise + t[:, None, None] * action_chunk
        velocity = action_chunk - noise

        a = self.action_enc(noisy) + self.pos.weight.unsqueeze(0) + self.t_emb(t_b).unsqueeze(1)
        s = self.state_enc(state).unsqueeze(1)
        toks = torch.cat([s, a], dim=1)                            # (B, 1+H, d)
        h = self.block(toks, vl_embs)                              # cross-attend
        pred = self.decoder(h[:, -self.H:])                        # action positions
        return F.mse_loss(pred, velocity)

# Toy training step
B, T_vl, d, H, A = 4, 10, 128, 16, 7
head = NanoFlowHead(d=d, H=H, action_dim=A)
vl = torch.randn(B, T_vl, d)
state = torch.randn(B, A)
acts = torch.randn(B, H, A)
loss = head(vl, state, acts)
loss.backward()
print(f"loss = {loss.item():.4f}")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
loss = ~1.0
```

中文:注意这个 forward 里完全没有 sigmoid / softmax / 离散化——velocity 是一个连续向量,loss 是简单 MSE。这是 flow-matching 和 diffusion 共同的优势:训练目标就是回归。

English: Notice no sigmoid, no softmax, no discretization anywhere in this `forward` — velocity is a plain continuous vector and the loss is a plain MSE. That's the shared appeal of flow-matching and diffusion policies: training is just regression.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **`huggingface/lerobot` 的 SmolVLA action expert** / **lerobot's SmolVLA action expert**: 同一套 flow-matching loss,但 backbone 换成轻量 SmolVLM,用更小的 action expert。 / Same flow-matching loss, but with a SmolVLM backbone and a smaller action expert.
- **Physical-Intelligence/openpi 的 π₀** / **Physical-Intelligence/openpi's π₀**: 公式完全一样;π₀ 论文里把这套 loss 叫做 "flow-matching action expert objective"。 / The exact same equations; the π₀ paper calls this the "flow-matching action expert objective".
- **`Robbyant/lingbot-va`** / **`Robbyant/lingbot-va`**: WAM 用同样的 `(1-t)·noise + t·x` parameterization,但目标是视频 latent 而不是 action chunk。 / Uses the same `(1-t)·noise + t·x` parameterization, but the target is video latents instead of an action chunk.
- **`facebookresearch/DiT`** / **`facebookresearch/DiT`**: 这个 `t_discretized → AdaLN` 模式直接借自 DiT;只是 GR00T 把 noisy 「图像」换成 noisy 「action 序列」。 / The `t_discretized → AdaLN` pattern is borrowed directly from DiT — GR00T just swapped noisy images for noisy action sequences.
- **对比 `huggingface/lerobot/policies/gaussian_actor`** / **vs. `gaussian_actor` in the same folder**: 同一个 action head slot,但用 Gaussian regression 而不是 flow-matching——结构一模一样,forward 的内容换成 `mean, std = decoder(h); loss = -log_prob(action_chunk, mean, std)`。 / Same slot, but Gaussian regression instead of flow matching — the wiring is identical, only the forward's last 4 lines change to `mean, std = decoder(h); loss = -log_prob(action_chunk, mean, std)`.

## 注意事项 / Caveats / when it breaks

- **`velocity` 不依赖于 `t`** / **`velocity` is independent of `t`**: 这是 rectified flow 的关键性质。如果你看到代码里 velocity 居然带 `t`,那大概率是普通 diffusion 改写过来的,不是真正的 RF。 / This is the defining property of rectified flow. If you see velocity depending on `t`, that's vanilla diffusion in disguise — not RF.
- **`sample_time` 用 Beta 不是 Uniform** / **`sample_time` uses Beta, not Uniform**: GR00T 默认 `Beta(1.5, 1)`,SD3 用 logit-normal——两者目的相同:压制 t→0 高方差区间的梯度。 / GR00T defaults to `Beta(1.5, 1)`, SD3 uses logit-normal — both shrink the high-variance gradient region near `t = 0`.
- **action_mask 必须乘进 loss** / **`action_mask` MUST be multiplied into the loss**: 不然 padded future steps 的随机噪声会主导梯度。 / Otherwise the random noise in padded future steps dominates the gradient.
- **推理时是 `get_action()`,不是这个 `forward()`** / **Inference uses `get_action()`, not this `forward()`**: 这个 forward 算 loss;`get_action()`(line 347+)从纯噪声开始,跑 N 步 Euler 积分得到最终 action。两个函数共用 `action_encoder` / `model` / `action_decoder`。 / This `forward` computes loss; `get_action()` (line 347+) starts from pure noise and runs `N` Euler steps to integrate to the final action. Both share the same `action_encoder` / `model` / `action_decoder`.

## 延伸阅读 / Further reading

- [GR00T-N1 技术报告](https://research.nvidia.com/labs/gear/gr00t/)
- [Lipman et al. — Flow Matching for Generative Modeling (2023)](https://arxiv.org/abs/2210.02747)
- [Stable Diffusion 3 paper — the timestep schedule trick](https://arxiv.org/abs/2403.03206)
- [SmolVLA in lerobot — the lightweight cousin](https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/smolvla)
