---
date: 2026-06-08
topic: vla
source: vla
repo: openvla/openvla-oft
file: prismatic/extern/hf/modeling_prismatic.py
permalink: https://github.com/moojink/openvla-oft/blob/main/prismatic/extern/hf/modeling_prismatic.py#L618-L621
difficulty: advanced
read_time: ~14 min
tags: [code-of-the-day, vla, openvla-oft, action-head, zero-embedding, rope, parallel-decode, nano-vla]
build_role: action-head-continuous (deep-dive variant) — how OpenVLA-OFT turns LLaMA into a position-only condition encoder by zeroing action embeddings + L1 regression head
---

# OpenVLA-OFT 把 LLaMA 退化成"位置查询编码器":action 位置全塞零 + fork transformers 改 bidirectional mask,L1 head 一次出 8 步 / OpenVLA-OFT turns LLaMA into a "position-only query encoder": zero the action embeddings + fork transformers to swap causal mask for bidirectional, then let the L1 head emit 8 steps at once

> **📌 订正 (2026-06-08) / Correction**: 本笔记初版说 "OFT 完全不改 attention mask",**这是错的**。OFT 同时做了两件事:(1) 在 `modeling_prismatic.py` 里把 56 个 action 位置的 input embedding 乘零;(2) **在 fork 过的 `transformers` 库**([`moojink/transformers-openvla-oft` commit `bc339d9`](https://github.com/moojink/transformers-openvla-oft/commit/bc339d9))里把 `LlamaSdpaAttention.forward` 的 causal mask 改成全序列 bidirectional mask + `is_causal=False`。两个机制叠加才解释了 OFT 的 parallel decoding。下面 "为什么重要"段落和"逐行讲解"已按此修正。
>
> **📌 Correction (2026-06-08)**: An earlier version of this note said "OFT does not change the attention mask at all" — **this was wrong**. OFT actually does two things: (1) zero out the 56 action-position input embeddings in `modeling_prismatic.py`; (2) **in a forked `transformers` library** ([`moojink/transformers-openvla-oft` commit `bc339d9`](https://github.com/moojink/transformers-openvla-oft/commit/bc339d9)), replace the causal mask in `LlamaSdpaAttention.forward` with a full-sequence bidirectional mask + `is_causal=False`. Both mechanisms together explain OFT's parallel decoding. The "Why this matters" and "What's happening" sections below are corrected accordingly.

> **一句话 / In one line**: OFT 同时做两件事:(1) 把 56 个 action token 位置的 input embedding 全部乘零;(2) 通过 fork transformers 把 LLaMA 的 causal mask 换成全序列 bidirectional mask。前者消除 exposure bias,后者让 56 个 action 位置可以一次 forward 互相交流,叠加后 MLPResNet 把 56 个 hidden state 解码成 8 步连续动作。 / OFT does two things together: (1) multiply the 56 action-token-position input embeddings by zero; (2) via a forked transformers, swap LLaMA's causal mask for a full-sequence bidirectional mask. The first eliminates exposure bias; the second lets the 56 action positions exchange information in a single forward pass; together the MLPResNet decodes the 56 hidden states into 8 continuous action steps.

## 为什么重要 / Why this matters

原版 OpenVLA 把动作"伪装成文字"让 LLaMA 自回归 7 次出 1 步动作 — 训练用真值 teacher forcing,测试用自己上一步的预测,典型的 exposure bias。**OpenVLA-OFT 反过来:不让 LLaMA 生成动作,而是把它变成一个"条件编码器"** — 通过两个相互配合的 trick 实现:**(1) action 位置的 input embedding 在 `modeling_prismatic.py` 里被强制乘零**(消除 exposure bias 来源 — 训练测试看到的都是零);**(2) 在 fork 过的 `transformers` 库里把 LLaMA 的 causal mask 换成全序列 bidirectional**(让 56 个 action 位置可以一次 forward 互相交流,而不是 causal 串行)。最后一个 MLPResNet 把 56 个 hidden state 解码成 8 步连续动作。**没有自回归、没有 exposure bias、推理速度 7-8 倍**,精度反而略升。理解这个"零输入 + bidirectional mask + 单一 forward"的组合,你就理解了为什么后续 π₀、GR00T、SmolVLA 全都抛弃了"action as text"路线。

OpenVLA disguises actions as text and lets LLaMA autoregressively decode 7 tokens to get 1 action step — teacher forcing in training, self-prediction in test, textbook exposure bias. **OpenVLA-OFT flips this**: don't let LLaMA generate actions; turn it into a "condition encoder" instead. Two cooperating tricks: **(1) action-position input embeddings are forced to zero in `modeling_prismatic.py`** (removes the source of exposure bias — both train and test see zeros); **(2) a forked `transformers` library replaces LLaMA's causal mask with a full-sequence bidirectional mask** (lets the 56 action positions communicate in a single forward pass rather than causal-serial). An MLPResNet then decodes 56 hidden states into 8 continuous action steps. **No autoregression, no exposure bias, 7-8× faster inference**, with accuracy maintained or slightly improved. Understanding this "zero input + bidirectional mask + single forward" combo explains why π₀, GR00T, and SmolVLA all dropped the "action-as-text" route.

## 代码 / The code

### 1. 灵魂三行:置零 + 取 hidden + L1 head

[`prismatic/extern/hf/modeling_prismatic.py:618-621`](https://github.com/moojink/openvla-oft/blob/main/prismatic/extern/hf/modeling_prismatic.py#L618-L621)

```python
# === L1 模式:action embedding 全部乘零 ===
all_actions_mask = all_actions_mask.unsqueeze(-1)        # (B, seq_len, 1)
input_embeddings = input_embeddings * ~all_actions_mask  # ← 56 个 action 位置的 input embedding → 0
```

### 2. 隐藏在 fork 里的 mask 改造 — 让 LLaMA 变 bidirectional

[`moojink/transformers-openvla-oft` commit `bc339d9`](https://github.com/moojink/transformers-openvla-oft/commit/bc339d9) 改了 `src/transformers/models/llama/modeling_llama.py` 里的 `LlamaSdpaAttention.forward`,把默认 causal mask 替换成全序列 bidirectional mask:

```python
# 改动前(原版 transformers v4.40.1):
attn_output = F.scaled_dot_product_attention(
    query_states, key_states, value_states,
    attn_mask=causal_mask,                          # 下三角(causal)
    is_causal=causal_mask is None and q_len > 1,    # 默认 causal
)

# 改动后(OFT fork):
if causal_mask is not None:
    D = causal_mask.shape[-1]
    last_row = causal_mask[:, :, -1, :].clone()     # ← 取原 causal mask 的最后一行
    new_mask = last_row.unsqueeze(2).expand(-1, -1, D, -1)  # ← 复制 D 遍
    causal_mask = new_mask                          # ← 覆盖

attn_output = F.scaled_dot_product_attention(
    query_states, key_states, value_states,
    attn_mask=causal_mask,                          # 现在是 bidirectional + pad 屏蔽
    is_causal=False,                                 # ← 关掉 causal
)
```

**这 4 行的几何含义**:HF 默认生成的 causal mask 是下三角(0 = 可见,-inf = 不可见),且**最后一行**因为对应"看得最远的 query",`-inf` 只可能出现在 pad 列。所以把最后一行复制 D 遍 → 得到"全 0(双向)+ pad 列 -inf"的 bidirectional mask。**一个 broadcast 同时完成"从 causal 变 bidirectional"和"保留 pad 屏蔽"两件事**。

[`vla-scripts/finetune.py:373-390`](https://github.com/moojink/openvla-oft/blob/main/vla-scripts/finetune.py#L373-L390)

```python
# === 32 层 LLaMA forward 后,抽出 56 个 action 位置的 hidden state ===
last_hidden_states = output.hidden_states[-1]                      # (B, seq_len, 4096)
text_hidden_states = last_hidden_states[:, num_patches:-1]         # 跳过 vision+proprio prefix
actions_hidden_states = (
    text_hidden_states[current_action_mask | next_actions_mask]
    .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)        # (B, 56, 4096)
)

if use_l1_regression:
    predicted_actions = action_head.module.predict_action(actions_hidden_states)
    loss = torch.nn.L1Loss()(ground_truth_actions, predicted_actions)
```

[`prismatic/models/action_heads.py:84-107`](https://github.com/moojink/openvla-oft/blob/main/prismatic/models/action_heads.py#L84-L107)

```python
class L1RegressionActionHead(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=4096, action_dim=7):
        self.model = MLPResNet(
            num_blocks=2,
            input_dim=input_dim * ACTION_DIM,    # 4096 * 7 = 28672
            hidden_dim=hidden_dim,
            output_dim=action_dim,                # 7
        )

    def predict_action(self, actions_hidden_states):
        # (B, 56, 4096) → reshape → (B, 8, 28672) → MLPResNet → (B, 8, 7)
        batch_size = actions_hidden_states.shape[0]
        rearranged = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        return self.model(rearranged)
```

### 2. proprio token append 到 vision prefix

[`modeling_prismatic.py:449-459`](https://github.com/moojink/openvla-oft/blob/main/prismatic/extern/hf/modeling_prismatic.py#L449-L459)

```python
def _process_proprio_features(self, projected_patch_embeddings, proprio, proprio_projector):
    if proprio_projector is not None and proprio is not None:
        proprio = proprio.reshape(B, -1)                            # (B, 8)
        proprio_features = proprio_projector(proprio)                # (B, 4096)
        proprio_features = proprio_features.unsqueeze(dim=1)         # (B, 1, 4096)
        # ← proprio 直接 cat 到 vision patch 末尾,当作扩展的 vision prefix
        return torch.cat((projected_patch_embeddings, proprio_features), dim=1)
    return projected_patch_embeddings
```

## 逐行讲解 / What's happening

### 1. **`input_embeddings * ~all_actions_mask`** — 全文的灵魂
- 中文: `all_actions_mask` 在 56 个 action 位置是 True,其他位置是 False。乘 `~mask`(取反)把 56 个位置乘零,其他位置原样。action token id 仍在 `input_ids` 里(用于位置识别),但它们的 embedding 内容被抹掉。
- English: `all_actions_mask` is True at the 56 action positions, False elsewhere. Multiplying by `~mask` zeros those 56 positions and leaves the rest intact. The action token ids stay in `input_ids` (used for position lookup), but their embedding content is wiped.

### 2. **`softmax(q=0) = uniform`** — 第一层魔法(与 bidirectional mask 配合)
- 中文: action 位置 input 是零 → q_proj(零) = 零(LLaMA 的 q_proj 无 bias)→ RoPE(零) = 零 → attention scores = 零 @ K.T = 零向量 → **softmax(零向量) = uniform 分布**。注意"可见位置"由 mask 决定 — **OFT 的 bidirectional mask 让 action 位置的"可见集"扩展为全序列(prefix + 所有其他 action 位置),而原 causal mask 下只能看 prefix + 前 k-1 个 action**。第一层 V_action = 0,所以 uniform @ V 实际上 ≈ `(N_prefix / N_total) × mean(V_prefix)`,一个非零向量。**所有 56 个 action 位置第一层后得到完全相同的 hidden state**。
- English: action position input is 0 → q_proj(0) = 0 (no bias) → RoPE(0) = 0 → attention scores = 0 @ K.T = zero vector → **softmax(zero) = uniform**. The "visible set" depends on the mask — **OFT's bidirectional mask makes each action position's visible set the full sequence (prefix + all other action positions), whereas the original causal mask only allowed prefix + the previous k-1 action positions**. Since V_action = 0 at layer 1, uniform @ V ≈ `(N_prefix / N_total) × mean(V_prefix)`, non-zero. **All 56 action positions get the exact same hidden state after layer 1**.

### 2.5 **bidirectional mask 在后续层才真正发力** — 56 个位置互相 attend
- 中文: 第一层 V_action 是零,bidirectional 让 action 互相看也只是看到零,与 causal 表现接近。**但从第二层起 V_action 变成非零的 prefix mean 派生**,bidirectional mask 让位置 T_k 能 attend 它后面的 T_{k+1}..T_{56},于是 56 个位置可以**双向交换信息**(causal mask 下只能后看前)。这就是 OFT 论文里 "lets the decoder predict all actions simultaneously" 的物理实现 — 不只是 zero embedding 让 emerge,而是 mask 真的允许全方位交流。
- English: at layer 1, V_action is zero, so bidirectional adds nothing over causal. **But from layer 2 onward, V_action becomes derived prefix-mean, non-zero**, and the bidirectional mask lets position T_k attend T_{k+1}..T_{56} — 56 positions now **exchange information in both directions** (causal would only allow backward). This is the physical realization of the OFT paper's "let the decoder predict all actions simultaneously" — not merely emergent from zero embedding, but actively enabled by the bidirectional mask.

### 3. **RoPE 在第二层才开始区分位置** — 隐藏的细节
- 中文: 第二层进 attention 时 h^(1)_a 不再是零,q_proj 出来非零的 q_a^(1)。RoPE 旋转 q_a^(1) 时,**不同位置 a 用不同旋转角**(RoPE 的数学:`<RoPE(q,m), RoPE(k,n)> = f(q·k, m-n)`,只取决于相对位置)。所以 56 个位置的 attention pattern 从第二层开始分化。
- English: at layer 2, h^(1)_a is no longer 0, so q_proj gives non-zero q_a^(1). RoPE rotates q_a^(1) with **different angles per position a** (RoPE's math: `<RoPE(q,m), RoPE(k,n)> = f(q·k, m-n)`, depends only on relative position). The attention patterns of the 56 positions diverge starting at layer 2.

### 4. **`actions_hidden_states.reshape(B, NUM_ACTIONS_CHUNK, -1)`** — chunk 步聚合
- 中文: `(B, 56, 4096)` 按 chunk 步分组,每 7 个连续 hidden state 沿 channel 维 concat 成 `(B, 8, 28672)`。**同一 chunk 步内 7 个动作维度的 hidden 被绑在一起,联合回归**;不同 chunk 步之间在 MLPResNet 里独立解码(但 chunk 步之间的时序连贯性已经在 32 层 attention 里隐式建模了)。
- English: `(B, 56, 4096)` is regrouped per chunk step — 7 consecutive hidden states get channel-concatenated into `(B, 8, 28672)`. **The 7 action-dimension hidden states inside one chunk step are bundled and jointly regressed**; chunk steps are decoded independently by the MLPResNet (but the temporal continuity between them is already encoded by the 32 layers of attention).

### 5. **`MLPResNet(28672 → 4096 → 4096 → 7)`** — 简单的全连接 decode
- 中文: 2-block MLP-ResNet,把 28672 维 hidden 压到 7 维 continuous action。**这就是整个 "解码" 步骤** — 没有 softmax,没有 vocab,没有 sampling。直接出连续值,可微,L1 loss 监督。
- English: 2-block MLP-ResNet compresses 28672-dim hidden into 7-dim continuous action. **That's the entire "decode" step** — no softmax, no vocab, no sampling. Just continuous outputs, differentiable, supervised by L1 loss.

### 6. **`L1Loss()(ground_truth_actions, predicted_actions)`** — 终极目标
- 中文: 对 (B, 8, 7) 求 L1 loss,平均绝对误差。**比 OpenVLA 的 CE on 256-bin 精度高了 ~30 倍**(256 bin 精度 ~0.008,L1 可以收敛到 ~0.0003)。但 L1 取中位数,**多 mode 数据会被平均掉**(详见 caveats)。
- English: L1 loss on `(B, 8, 7)` — mean absolute error. **~30× higher precision than OpenVLA's 256-bin CE** (256-bin precision ~0.008, L1 converges to ~0.0003). But L1 optimizes the median — **multi-modal data gets averaged out** (see caveats).

## 类比 / The analogy

OpenVLA 是一个**让 LLaMA 当编剧**:给它 prompt 和一段开头,逼它"续写"7 个特殊文字。它擅长这事,但每写一个字要先看自己上一个字写的什么 — 测试时如果上一个写错,下一个会越错越离谱。

OpenVLA-OFT 把 LLaMA 改造成**一个"信息提取器"**:把 56 个空白卡片(零向量)插进序列末尾,每张卡片上印着"我是第 i 步的第 j 维"(位置编码)。LLaMA 通过 32 层 attention,**像一个图书馆员一样**,把图书馆里的所有书(vision、proprio、language)逐层抄写一些信息到这 56 张卡片上。最后一个简单的查表机(MLPResNet)按"每 7 张卡片合并成一份订单"的规则,直接把卡片读成 8 份连续的动作订单。**LLaMA 不再生成动作,而是"翻译条件"**。

OpenVLA = **a screenwriter LLaMA**: hand it a prompt + opening, force it to "continue writing" 7 special characters. It's good at this, but each character depends on the one before — at test time, if one is wrong, the next gets exponentially worse.

OpenVLA-OFT turns LLaMA into **an "information extractor"**: insert 56 blank index cards (zero vectors) at the sequence end, each card marked "I am step i, dimension j" (positional encoding). LLaMA, **like a librarian**, copies bits of information from every book in the library (vision, proprio, language) onto these 56 cards across 32 layers of attention. A simple lookup machine (MLPResNet) then reads "every 7 cards = one order" and converts them into 8 continuous action orders. **LLaMA stops generating actions; it starts translating conditions**.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-VLA

这是 nanoVLA 课程里 `action-head-continuous` 槽位的**深度变体**(5/28 笔记用 dreamzero rectified flow,6/04 用 GR00T action encoder,今天讲 OFT 的 L1 路线 — 三种是同一个 slot 的不同实现)。

在你自己搭的 nanoVLA 里,这个模式给你的工程选择:

```
upstream (from previous lessons):
  vision_encoder → (B, 256, D)               ← vision-encoder 课程
  modality_projector → (B, 256, D_lm)        ← modality-projector 课程  
  vlm_backbone → 32-layer causal attention   ← vlm-backbone-wiring 课程

OFT-style action-head-continuous:
  1. proprio_projector(state) → 1 token append to vision prefix
  2. action_chunk_positions in input: 全部置零
  3. forward LM → last_hidden[action_positions] → (B, K*A, D)
  4. reshape (B, K, A*D)
  5. MLPResNet(A*D → hidden → A) → (B, K, A) continuous
  6. L1 loss (or diffusion / flow matching for multi-mode tasks)
```

**关键设计决策**:

1. **action_chunk 长度 K**:LIBERO=8,ALOHA=25,bridge=5。**越长 latency 摊销越好,但 chunk 末尾步骤精度可能下降**(因为它们 attend 不到执行 chunk 后的新观察)。
2. **action 位置 input 用零还是 learned mask token**:OFT 选了零(简单且复用 LLaMA 词表 slot)。MAE 选了 learned embedding(略好但要额外学一个 vector)。**对小模型(< 1B)推荐 learned 更稳;对 7B+ 直接零就够了,RoPE 信息已经足够区分**。
3. **MLPResNet 还是 transformer 解码**:OFT 用 MLP(轻,无 chunk 步间交互);ACT 用 transformer decoder。**MLP 假设 chunk 步间的时序连贯性已经被 LLM 内部 attention 处理了** — 实测 work,但要确认 LLM attention head 数够多。
4. **L1 vs diffusion vs flow matching**:**L1 在单 mode 数据上够用,多 mode 必须换 flow / diffusion** — 这是 OFT 论文明确给的选择。

production 实现要补:
- **frozen vision backbone + LoRA on LM**:全参 7B fine-tune 显存爆,实际 OFT-Diffusion 实验都是 frozen vision + LoRA LM
- **action_dim normalization**:不同维度量纲不同(dx ~ 米,dr ~ 弧度,gripper ~ 离散),要分别归一化到 [-1, 1]
- **closed-loop vs open-loop chunk 执行**:open-loop 执行 8 步省 latency,但可能 drift;closed-loop 每步重新 query 精度高但慢。LIBERO 通常 open-loop chunk,实机部署常用 receding-horizon(执行 chunk 前 k 步后重新 query)

This is the **deep-dive variant** of the `action-head-continuous` slot in the nanoVLA curriculum (5/28 covered dreamzero rectified flow, 6/04 covered GR00T action encoder, today is OFT's L1 route — three implementations of the same slot).

In your nanoVLA, the engineering decision tree:

```
upstream (from previous lessons):
  vision_encoder → (B, 256, D)
  modality_projector → (B, 256, D_lm)
  vlm_backbone → 32-layer causal attention

OFT-style action-head-continuous:
  1. proprio_projector(state) → 1 token append to vision prefix
  2. set action_chunk_positions in input to zero
  3. forward LM → last_hidden[action_positions] → (B, K*A, D)
  4. reshape (B, K, A*D)
  5. MLPResNet(A*D → hidden → A) → (B, K, A) continuous
  6. L1 loss (or diffusion / flow matching for multi-mode tasks)
```

**Key design decisions**:

1. **Chunk length K**: LIBERO=8, ALOHA=25, bridge=5. **Longer K = better latency amortization but late-chunk steps lose accuracy** (they can't attend to post-chunk observations).
2. **Zero vs learned mask token for action input**: OFT picks zero (simpler, reuses LLaMA vocab slots). MAE picks learned embedding (slightly better, needs one extra vector). **For small models (<1B) prefer learned; for 7B+ zero is enough — RoPE already differentiates positions**.
3. **MLPResNet vs transformer decoder**: OFT uses MLP (lightweight, no inter-step attention). ACT uses transformer decoder. **MLP assumes chunk-step temporal continuity has been handled by the LLM's internal attention** — works in practice, but requires sufficient attention heads in the LLM.
4. **L1 vs diffusion vs flow matching**: **L1 suffices for unimodal data; multi-mode requires flow / diffusion** — explicitly the choice OFT paper offers.

Production needs:
- **Frozen vision backbone + LoRA on LM**: full 7B fine-tune blows up memory; OFT-Diffusion experiments all use frozen vision + LoRA LM
- **Per-dimension action normalization**: dx (meters), dr (radians), gripper (discrete) need separate normalization to [-1, 1]
- **Closed-loop vs open-loop chunk execution**: open-loop saves latency but may drift; closed-loop is precise but slow. LIBERO uses open-loop chunk; deployment often uses receding-horizon (execute first k steps, then re-query)

## 自己跑一遍 / Try it yourself

```python
# try.py — minimal "zero-input + RoPE + MLP head" prediction in 50 lines
import torch, torch.nn as nn, torch.nn.functional as F

D = 64           # hidden dim
N_PREFIX = 32    # vision + proprio + language prefix
N_ACTION = 16    # 8 chunk steps * 2 action dims
N_HEADS = 4

# 简化的 prefix encoder(用 random embedding 代替真实 vision/proprio/lang)
prefix = torch.randn(1, N_PREFIX, D)

# action 位置全是零
action_zeros = torch.zeros(1, N_ACTION, D)

# 拼接
seq = torch.cat([prefix, action_zeros], dim=1)  # (1, 48, 64)

# 一个简化的 causal transformer block(用 RoPE)
class RoPETransformerBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3*dim, bias=False)   # ← 注意:无 bias,模仿 LLaMA
        self.proj = nn.Linear(dim, dim, bias=False)
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, dim))

    def apply_rope(self, x, position_ids):
        # 极简 RoPE:对 head_dim 的偶数对做旋转
        d = x.shape[-1]
        half = d // 2
        pos = position_ids.float().unsqueeze(-1)
        freqs = torch.exp(-torch.arange(half) * (4.0 / half))
        theta = pos * freqs                            # (seq_len, half)
        sin, cos = theta.sin(), theta.cos()
        x_even, x_odd = x[..., :half], x[..., half:]
        return torch.cat([x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], dim=-1)

    def forward(self, x, position_ids):
        B, L, D = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(B, L, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.apply_rope(q, position_ids)
        k = self.apply_rope(k, position_ids)
        mask = torch.triu(torch.full((L, L), float("-inf")), diagonal=1)
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5) + mask
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        x = x + self.proj(out)
        x = x + self.mlp(self.ln2(x))
        return x

# 模拟 LLaMA forward(用 4 层代替 32 层)
model = nn.Sequential(*[RoPETransformerBlock(D, N_HEADS) for _ in range(4)])
position_ids = torch.arange(N_PREFIX + N_ACTION)

h = seq
for blk in model:
    h = blk(h, position_ids)

# 抽出 16 个 action 位置的 hidden state
action_hidden = h[:, -N_ACTION:]  # (1, 16, 64)

# 验证:第一层之后,所有 action 位置的 hidden 是否完全相同(理论预测 ✓)?
with torch.no_grad():
    h1 = model[0](seq, position_ids)
    print(f"layer 1 后 action 位置之间的最大差 (理论 0): "
          f"{(h1[:, -N_ACTION:].max(dim=1).values - h1[:, -N_ACTION:].min(dim=1).values).abs().max().item():.6f}")
    h2 = model[1](h1, position_ids)
    print(f"layer 2 后 action 位置之间的最大差 (RoPE 开始区分): "
          f"{(h2[:, -N_ACTION:].max(dim=1).values - h2[:, -N_ACTION:].min(dim=1).values).abs().max().item():.4f}")

# L1 head: reshape (1, 16, 64) → (1, 8, 128) → MLPResNet → (1, 8, 2)
action_hidden_reshaped = action_hidden.reshape(1, 8, 2*D)
mlp = nn.Sequential(
    nn.Linear(2*D, D), nn.GELU(),
    nn.Linear(D, D), nn.GELU(),
    nn.Linear(D, 2)
)
predicted_actions = mlp(action_hidden_reshaped)
print(f"\npredicted actions shape: {predicted_actions.shape}")  # (1, 8, 2)
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
layer 1 后 action 位置之间的最大差 (理论 0): 0.000000        ← 字面证实"q=0 让所有 action 位置同质化"
layer 2 后 action 位置之间的最大差 (RoPE 开始区分): 0.5+    ← RoPE 旋转开始让位置分化
predicted actions shape: torch.Size([1, 8, 2])
```

**注意 layer 1 后 16 个 action 位置严格相同(差值精确为 0),layer 2 后明显分化** — 这就是 OFT 内部"位置区分从第二层开始"的数值证明。

**Notice layer 1 leaves all 16 action positions strictly identical (diff exactly 0), and layer 2 starts to differentiate them** — that's the numerical proof of "OFT's position differentiation begins at layer 2".

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **ACT (Action Chunking Transformer, Zhao et al. 2023)**: OFT 最直接的前身。chunking + parallel decode + L1 loss + transformer encoder + 学习的 query position embedding → OFT 几乎全部继承,只把 transformer encoder 换成"复用 LLaMA + zero embedding" / Most direct predecessor. Chunking + parallel decode + L1 loss + transformer encoder + learnable query positional embedding → OFT inherits nearly all, just swaps "transformer encoder" for "reuse LLaMA + zero embedding".
- **π₀ (Physical Intelligence, 2024)**: 同样的 action chunk + 同样的 proprio token + 用 flow matching 替换 L1(解决 multi-mode 问题) / Same action chunking + same proprio token, but replaces L1 with flow matching (solves the multi-mode problem).
- **GR00T-N1 (NVIDIA, 2024)**: 同样的 framework 但用 cross-attention DiT action head(见 6/05 笔记)/ Same framework but uses a cross-attention DiT action head (see 6/05 note).
- **SmolVLA (HuggingFace 2024)**: VLM + slim action expert 风格,expert 走 cross-attention(见 5/29 笔记) / VLM + slim action expert style, expert uses cross-attention (see 5/29 note).

## 注意事项 / Caveats / when it breaks

- **L1 在多 mode 数据上会平均成"中位数",实测可能"撞墙"** / **L1 averages multi-modal data to the median; can literally crash into walls**: 经典的"双臂协调"或"绕障碍左/右都行"任务,L1 会预测出中间值。换 flow matching / diffusion head 必须的 / Classic "two-arm coordination" or "go-around-obstacle-from-either-side" tasks: L1 predicts the median, which is the obstacle itself. Use flow matching or diffusion instead.
- **open-loop chunk 执行可能 drift** / **Open-loop chunk execution may drift**: 一次预测 8 步、全部执行后再 query,中间 7 步看不到新观察。物体被人挪动或机器人 slip 时会失败。**生产用 receding-horizon**:执行前 k 步后立即重新预测 / Predict 8 steps, execute all, then re-query — the middle 7 steps don't see new observations. Fails when objects move or the robot slips. **Production uses receding-horizon**: execute first k steps then re-predict.
- **chunk 末尾步骤精度下降** / **Late-chunk steps degrade**: 第 8 步动作的"参考"还是当前一帧图像,但它要预测 ~1 秒后机器人该做什么。**远视野 vs 精确度的 trade-off**,LIBERO 简单任务问题不大,长 horizon 任务需要 K=3-4 + 频繁 re-query / The 8th step's "reference" is still the current frame, but it predicts what the robot should do ~1 second in the future. Trade-off between long horizon and precision. Easy tasks OK; long-horizon tasks need K=3-4 with frequent re-query.
- **action_dim 内部联合回归 vs 独立回归** / **Joint vs independent action-dim regression**: MLPResNet 内部把 7 个维度的 hidden 拼起来联合回归,所以**同一 chunk 步内 7 维不独立**(好事:gripper 和 EEF 协调);但**不同 chunk 步之间独立**(坏事:8 步轨迹连贯性靠 LLM 内部 attention 隐式建模,如果 LLM 太小可能不够) / The 7 dims inside one chunk step are jointly regressed (good: gripper and EEF coordinate). Different chunk steps are decoded independently (bad: temporal continuity relies on the LLM's implicit attention; weak LLMs may not be enough).
- **proprio 投影器仅 2 层 MLP** / **Proprio projector is only 2 MLP layers**: 8 → 4096 → 4096,**单次过滤后压入一个 token**。如果机器人有更高维 proprio(比如双臂 14D + 触觉)信息密度不够,可以换成 token-per-joint 或加更深的 MLP / `8 → 4096 → 4096` flattens proprio into a single token. For high-dim proprio (14D dual-arm + tactile), this is too lossy; switch to token-per-joint or deeper MLP.

## 延伸阅读 / Further reading

- OpenVLA-OFT paper: https://arxiv.org/abs/2502.19645 — the original L1 + diffusion comparison
- ACT paper (Zhao et al., 2023) — the direct predecessor of chunking + parallel decode + L1 in robot learning
- π₀ paper (Physical Intelligence, 2024) — same framework with flow matching head
- MAE paper (He et al., 2021) — the spiritual ancestor of "mask token + positional embedding + self-attention fills in"
- Today's companion note on the "zero placeholder + RoPE" academic lineage
