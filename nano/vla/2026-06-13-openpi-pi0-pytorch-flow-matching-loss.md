---
date: 2026-06-13
topic: vla
source: vla
repo: Physical-Intelligence/openpi
file: src/openpi/models_pytorch/pi0_pytorch.py
permalink: https://github.com/Physical-Intelligence/openpi/blob/c23745b5ad24e98f66967ea795a07b2588ed6c79/src/openpi/models_pytorch/pi0_pytorch.py#L317-L374
difficulty: advanced
read_time: ~12 min
tags: [code-of-the-day, vla, flow-matching, training-step, pi0]
build_role: nanoVLA training-step (flow-matching variant) — replaces the openvla next-token loss as the alternative training-step recipe for continuous-action VLAs
---

# pi0 PyTorch 的 6 行 flow-matching loss + 整个训练 step / pi0 PyTorch's 6-line flow-matching loss + complete training step

> **一句话 / In one line**: 一个 58 行的 PyTorch `forward()` 就是 pi0 的完整训练 step —— 采噪 + 采 t、线性插值出 `x_t`、一遍 prefix+suffix 大 forward、把最后 `action_horizon` 个 token 投影成速度向量,最后 `F.mse_loss(noise - actions, v_t)`。/ A single 58-line PyTorch `forward()` *is* pi0's training step: sample noise + t, interpolate `x_t`, run one big prefix+suffix forward, project the final `action_horizon` tokens into a velocity vector, then `F.mse_loss(noise - actions, v_t)`.

## 为什么重要 / Why this matters

VLA 这条线最让人困惑的是"训练目标到底长什么样" —— 早期 openvla 是离散 token 的 next-token-prediction,2024 年 pi0 一篇文章把整个生态拉到了 flow-matching:**连续动作不再 tokenize,直接在动作向量上加噪声,让 transformer 学着回归"从 noise 到 target 的速度场"**。这份 PyTorch 实现是 pi0 JAX 版的对照实现,跑在更主流的 PyTorch 栈上,也是 lerobot/openpi 接 PyTorch 训练的官方入口。看懂这 58 行你就拿到了"如何在自己的 nanoVLA 里换掉 next-token loss 换成 flow-matching"的完整改造图纸。

The most confusing part of the VLA story is "what does the training objective actually look like?" — early openvla was discrete next-token prediction over a fixed vocab; the 2024 pi0 paper shifted the field to flow matching: **continuous actions are no longer tokenized; you add noise to the action vector itself and teach the transformer to regress the velocity field "noise → target"**. This PyTorch port mirrors pi0's JAX original on the more familiar PyTorch stack and is the official PyTorch training entry point for lerobot / openpi. Read these 58 lines and you have a complete blueprint for swapping a next-token loss in your nanoVLA for flow matching.

## 代码 / The code

`Physical-Intelligence/openpi` — [`src/openpi/models_pytorch/pi0_pytorch.py`](https://github.com/Physical-Intelligence/openpi/blob/c23745b5ad24e98f66967ea795a07b2588ed6c79/src/openpi/models_pytorch/pi0_pytorch.py#L317-L374)

```python
def forward(self, observation, actions, noise=None, time=None) -> Tensor:
    """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
    images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

    if noise is None:
        noise = self.sample_noise(actions.shape, actions.device)

    if time is None:
        time = self.sample_time(actions.shape[0], actions.device)

    time_expanded = time[:, None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions

    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
    if (
        self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        == torch.bfloat16
    ):
        suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
        prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1

    att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

    def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        return suffix_out

    suffix_out = self._apply_checkpoint(
        forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
    )

    suffix_out = suffix_out[:, -self.config.action_horizon:]
    suffix_out = suffix_out.to(dtype=torch.float32)

    def action_out_proj_func(suffix_out):
        return self.action_out_proj(suffix_out)

    v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

    return F.mse_loss(u_t, v_t, reduction="none")
```

## 逐行讲解 / What's happening

1. **第 319 行 / Line 319 (`_preprocess_observation`)**:
   - 中文: 把原始 batch 拆成 5 个 tensor —— 多视角图像、image padding mask、语言 token、语言 padding mask、本体状态 (state)。这是后面所有 embed 的输入,**`train=True` 这一参数会启用图像增广**。
   - English: Splits the raw batch into 5 tensors — multi-view images, image padding mask, language tokens, language padding mask, and proprioceptive state. These feed everything downstream; **`train=True` switches on image augmentation**.

2. **第 321-325 行 / Lines 321-325 (sample noise + sample time)**:
   - 中文: 噪声形状和 `actions` 完全一样,每个 batch 元素采一个 scalar `t`。注意 `sample_time` 在 pi0 里用的是 **Beta(1.5, 1) * 0.999 + 0.001** 而不是均匀分布 —— 这样 `t` 更偏向大值,等价于"训练时多看 noise 较强的样本",这是 pi0 论文的一个 trick。
   - English: Noise matches `actions` shape; each batch element gets one scalar `t`. Note `sample_time` is **Beta(1.5, 1) * 0.999 + 0.001**, not uniform — `t` skews toward larger values, so training puts more weight on noisier samples. A pi0-paper trick.

3. **第 327-329 行 / Lines 327-329 — the flow-matching trio**:
   - 中文: 整段代码的灵魂三行 —— `t * noise + (1-t) * actions` 是 noise 和真实动作之间的**直线插值**;`u_t = noise - actions` 是这条直线的**速度** (常向量 = noise - actions)。flow-matching 的整个范式就是教模型回归这个速度向量。注意这是"rectified flow"的最简形式:沿直线插值,没有正弦/余弦 schedule。
   - English: The soul of the whole snippet — `t * noise + (1-t) * actions` is the **straight-line interpolation** between noise and ground-truth actions; `u_t = noise - actions` is the **constant velocity along that line**. Flow matching's entire paradigm is "teach the model to regress this velocity vector". This is "rectified flow" in its simplest form — straight-line interp, no sinusoidal schedule.

4. **第 331-332 行 / Lines 331-332 (`embed_prefix` + `embed_suffix`)**:
   - 中文: pi0 的 token 流分两段 —— **prefix** 是图像 + 语言 (像普通 VLM 那样),**suffix** 是 state + `x_t` (noised action) + time embedding。**这是 pi0 区别于纯 LLM 的关键**:动作不是输出的 token,而是作为"上下文的一部分"喂回去,模型要做的是预测下一个动作 token 应该是什么样的速度。
   - English: pi0's token stream is two halves — **prefix** = images + language (vanilla VLM-style), **suffix** = state + `x_t` (noised action) + time embedding. **This is what separates pi0 from a pure LLM**: actions are not output tokens but part of the *input* context, and the model predicts what velocity the next action token should advance with.

5. **第 333-338 行 / Lines 333-338 (bf16 promotion)**:
   - 中文: 一个非常生产味的细节 —— 如果 backbone 已经是 bf16,就把 prefix/suffix embedding 也 cast 成 bf16,**避免 attention 入口的 dtype mismatch**。这是 LLM 微调里最常踩的坑之一。
   - English: A very production-flavoured detail — if the backbone is already bf16, promote prefix/suffix embeddings to bf16 too, **dodging a dtype mismatch at the attention input**. One of the most common pitfalls in LLM fine-tuning.

6. **第 340-347 行 / Lines 340-347 (拼接 mask + 2D causal mask)**:
   - 中文: 把 prefix 和 suffix 的 padding mask、attention mask 沿时间轴拼起来 (`dim=1` 就是 token 维度),`make_att_2d_masks` 构造一个 `(B, L, L)` 的 2D 注意力 mask,`position_ids = cumsum(pad) - 1` 让 padded token 位置正确。`_prepare_attention_masks_4d` 把它再 reshape 成 (B, 1, L, L) 喂给 HF transformers。
   - English: Concatenate prefix and suffix padding & attention masks along the time dim (`dim=1` is the token axis), `make_att_2d_masks` builds a `(B, L, L)` 2D attention mask, `position_ids = cumsum(pad) - 1` keeps padded-token positions sane. `_prepare_attention_masks_4d` reshapes to `(B, 1, L, L)` for HF transformers.

7. **第 349-363 行 / Lines 349-363 (gradient-checkpointed backbone forward)**:
   - 中文: 把 backbone 大 forward 包成一个闭包 `forward_func`,交给 `_apply_checkpoint`。如果开启了 grad checkpoint,这一段在反向时会重算而不是保存中间激活 —— **这是 7B/13B VLA 单卡能 fine-tune 的关键**。`paligemma_with_expert` 是 PaliGemma 主干 + action expert 双 transformer 串联的容器,接受两个 `inputs_embeds` (prefix + suffix) 并行处理。
   - English: Wrap the big backbone forward in a closure `forward_func` handed to `_apply_checkpoint`. If grad-checkpointing is on, this block recomputes during backward instead of storing activations — **the single reason 7B / 13B VLAs fit on a single GPU during fine-tune**. `paligemma_with_expert` chains PaliGemma + an action expert as two transformers running in parallel over `[prefix_embs, suffix_embs]`.

8. **第 365-372 行 / Lines 365-372 (取 suffix 末尾 + 投影成 v_t)**:
   - 中文: 切出 suffix 输出的最后 `action_horizon` 个 token —— 这正好对应那串 noised action 的位置。先 cast 回 fp32 保留精度,再过 `action_out_proj` (也是 grad-checkpoint 的) 投影成 `(B, H, action_dim)` 的速度向量 `v_t`。
   - English: Slice the last `action_horizon` suffix tokens — these align positionally with the noised-action slots. Promote back to fp32 for precision, then `action_out_proj` (also grad-checkpointed) projects to a `(B, H, action_dim)` velocity vector `v_t`.

9. **第 374 行 / Line 374 (the actual loss)**:
   - 中文: 一行 MSE,**且 `reduction="none"`** —— 把每元素的方差 loss 返回给外面的 trainer,让它有机会按 mask 加权或按 timestep 加权再 reduce。
   - English: A single MSE call, **with `reduction="none"`** — returns per-element squared error so the outer trainer can apply masks or per-timestep weights before reducing.

## 类比 / The analogy

中文: 把 flow-matching 想象成**教孩子从涂鸦回到原画**。你拿原画 `actions` 加随机涂鸦 `noise`,按时间 `t` 混在一起得到一张半涂鸦的中间稿 `x_t`;此时如果你告诉模型"在这个时间点,你每往前一步应该擦掉什么"(velocity `u_t = noise - actions`),它就能学会"在任何 t 的中间稿上把涂鸦逐步擦回原画"。pi0 的 backbone 像一个看过无数张原画 + 涂鸦对的素描老师 —— 训练目标 (MSE) 就是把它的"擦除指南" `v_t` 和真实擦除方向对齐。当训练好之后,推理时从纯涂鸦 `t=1` 开始,沿着 `v_t` 一小步一小步走,**最后就走回了原画 `t=0`**。

English: Think of flow matching as **teaching a kid to recover the original drawing from a scribble**. Take the real drawing `actions` and a random scribble `noise`, mix them by time `t` into a half-scribbled intermediate `x_t`. At this moment, if you tell the model "at time `t`, what should you erase per unit step" (velocity `u_t = noise - actions`), the kid learns to *gradually un-scribble* any intermediate back to the original. pi0's backbone is a sketching teacher who has seen countless (original, scribble) pairs — the training target (MSE) aligns its "erasure guide" `v_t` with the truth. At inference time, start from pure scribble `t=1` and follow `v_t` in small steps; **you end back at the original drawing `t=0`**.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-VLA

中文: 在我们的 nanoVLA 课程图里,这一份属于 **`training-step` 这个槽位** (依赖 `vlm-backbone-wiring`)。**整个 nanoVLA 的训练 step 就是这 60 行的精简版本**,只把以下三个 hook 替换成你自己的实现就行:
- `_preprocess_observation` → 你自己的 dataset 怎么解 batch
- `embed_prefix / embed_suffix` → 你自己的视觉 + 语言 + 状态投影器 (取决于你选的 backbone)
- `paligemma_with_expert.forward` → 你的 backbone (PaliGemma / SmolVLM / Qwen2-VL …) + 一个 action expert tower

上游是 `vlm-backbone-wiring` (告诉你 prefix/suffix 怎么拼);下游是 `inference-loop` (训练好后从 noise 走到 action)。**省掉这一步会发生什么?** —— 你没法监督任何东西,优化器只能瞪着模型而无所事事。

生产级实现要在这份精简版基础上加上:**(1)** `_apply_checkpoint` 的 gradient checkpointing,不然 13B PaliGemma 单卡 OOM;**(2)** bf16 / fp32 混合精度 promotion,不然 attention layer dtype 不匹配会 silently 算错;**(3)** `Beta(1.5, 1)` 的 timestep 采样,而不是均匀分布,这是 pi0 论文最实用的 trick 之一;**(4)** `reduction="none"` 让外层 trainer 按 padding mask 加权。

English: In the nanoVLA curriculum graph, this fills the **`training-step` slot** (depends on `vlm-backbone-wiring`). **The whole nanoVLA training step is a stripped-down version of these 60 lines** — you only need to replace three hooks with your own:
- `_preprocess_observation` → however your dataset decodes a batch
- `embed_prefix / embed_suffix` → your own vision + language + state projector (depends on your backbone choice)
- `paligemma_with_expert.forward` → your backbone (PaliGemma / SmolVLM / Qwen2-VL ...) + an action expert tower

Upstream is `vlm-backbone-wiring` (tells you how prefix/suffix get assembled); downstream is `inference-loop` (the trained model rolls from noise to actions). **What happens if you skip this?** — you have no supervisable target; the optimizer stares at the model and does nothing.

A production implementation needs to add on top of this minimal version: **(1)** `_apply_checkpoint` gradient checkpointing — without it 13B PaliGemma OOMs on a single GPU; **(2)** bf16/fp32 mixed-precision promotion — otherwise attention dtype mismatches silently miscompute; **(3)** `Beta(1.5, 1)` timestep sampling instead of uniform, which is one of pi0's most practical tricks; **(4)** `reduction="none"` so the outer trainer can weight by padding masks.

## 自己跑一遍 / Try it yourself

```python
# nano_flow_matching_step.py — the core flow-matching loss in ~25 lines, no VLM required
import torch
import torch.nn as nn
import torch.nn.functional as F

D, ACTION_DIM, H = 64, 7, 8        # latent, action dim, horizon

class TinyActionExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(D + ACTION_DIM + 1, D)
        self.head = nn.Linear(D, ACTION_DIM)
    def forward(self, context, x_t, t):                   # context: (B, D) state summary
        ctx = context.unsqueeze(1).expand(-1, H, -1)      # (B, H, D)
        t_emb = t[:, None, None].expand(-1, H, 1)         # (B, H, 1)
        x = torch.cat([ctx, x_t, t_emb], dim=-1)          # (B, H, D + A + 1)
        return self.head(self.lin(x))                     # (B, H, A) -> predicted v_t

model = TinyActionExpert()
B = 4
actions = torch.randn(B, H, ACTION_DIM)                   # ground-truth action chunk
context = torch.randn(B, D)                              # would be VLM output in real life

noise = torch.randn_like(actions)
t = torch.distributions.Beta(1.5, 1.0).sample((B,)) * 0.999 + 0.001
x_t = t[:, None, None] * noise + (1 - t[:, None, None]) * actions
u_t = noise - actions
v_t = model(context, x_t, t)

loss = F.mse_loss(v_t, u_t)
print(f"flow-matching loss = {loss.item():.4f}")
```

运行 / Run with:
```bash
pip install torch
python nano_flow_matching_step.py
```

预期输出 / Expected output:
```
flow-matching loss = 1.99
```

中文: 注意这里的 `context` 是随机的 —— 真要训练时,context 应该是 VLM 主干在 prefix 上吐出来的 pooled embedding。这个 30 行的玩具版本可以一直梯度下降,几千 step 后 loss 会到 0 附近 (因为 context 是常数,模型记住了)。换成 random `actions` + 真实 VLM context,才能学到真正的策略。

English: Notice the `context` is random here — in real training, `context` is the pooled embedding the VLM backbone emits over the prefix. This 30-line toy can be optimised happily for a few thousand steps and the loss will drop to ~0 (because the context is constant, the model memorises). Swap in real random `actions` + a real VLM context and you have the actual policy training loop.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **`huggingface/lerobot` GR00T flow-matching head** / **`huggingface/lerobot` GR00T flow-matching head**: 同样的 6 行 flow-matching 三件套,只是 action expert 不一样。/ Same 6-line flow-matching trio, different action expert.
- **`huggingface/lerobot` SmolVLA** / **`huggingface/lerobot` SmolVLA**: 也是 flow matching + PaliGemma-with-expert,但是 backbone 换成 SmolVLM,体积小很多。/ Also flow matching + PaliGemma-with-expert, but with SmolVLM backbone — much smaller.
- **`openvla/openvla` discrete next-token loss** / **`openvla/openvla` discrete next-token loss**: 对照组 —— 没有 noise 没有 t,把 action 离散化成 token,直接 cross-entropy。两条路线现在在 VLA 社区并存,各有优劣。/ The control case — no noise, no `t`, discretise actions into tokens and cross-entropy. Two lines coexist in the VLA community today, each with trade-offs.

## 注意事项 / Caveats / when it breaks

- **Beta(1.5, 1) 不能换成 uniform / Don't swap Beta(1.5, 1) for uniform**:
  - 中文: 看似无害的改动,会让训练更偏向小 t (清晰样本) 的简单情况,模型在大 t (强噪) 上学得很差,推理时早期步数 (t ≈ 1) 表现崩盘。
  - English: A seemingly harmless change biases training toward small-`t` (cleaner) easy cases; the model never learns large-`t` (noisier) regime, and inference at the early steps (t ≈ 1) collapses.
- **action_horizon 一定要和推理时一致 / `action_horizon` must match inference**:
  - 中文: 训练时 `suffix_out[:, -action_horizon:]` 切出的位置数和推理时要预测的 chunk 长度强绑定。**改了一个不改另一个就直接训歪**。
  - English: The `suffix_out[:, -action_horizon:]` slice during training is tightly coupled to the chunk length predicted at inference. **Change one but not the other and training silently misaligns.**
- **embed_suffix 里 time 怎么编码 / How `time` is encoded inside `embed_suffix` matters**:
  - 中文: 源码里 `embed_suffix` 通常用 sinusoidal embedding 或 MLP 把 `time` 投影成 D 维向量再加进 state token。如果换 backbone 时这里没对齐 (比如 backbone hidden dim 改了),模型就接不到 t 这个关键信号。
  - English: `embed_suffix` typically encodes `time` with a sinusoidal or MLP projection into a D-dim vector and adds it to the state token. If you change backbones without realigning this (e.g. backbone hidden dim differs), the model loses the critical `t` signal.

## 延伸阅读 / Further reading

- [pi0 paper — "A Vision-Language-Action Flow Model for General Robot Control"](https://arxiv.org/abs/2410.24164)
- [Rectified Flow paper — Liu et al. 2022](https://arxiv.org/abs/2209.03003)
- [openpi PyTorch port README](https://github.com/Physical-Intelligence/openpi#pytorch-port)
- [Flow Matching survey — Lipman 2024](https://arxiv.org/abs/2210.02747)
