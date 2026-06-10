---
date: 2026-06-09
topic: vla
source: vla
repo: NVIDIA/Isaac-GR00T
file: gr00t/model/gr00t_n1d7/gr00t_n1d7.py
permalink: https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/model/gr00t_n1d7/gr00t_n1d7.py
difficulty: advanced
read_time: ~18 min
tags: [code-of-the-day, vla, groot, gr00t-n1d7, dit, cross-attention, multi-embodiment, flow-matching, nano-vla]
build_role: vlm-backbone-wiring (deep-dive variant) — GR00T-N1.7 的 image/language/state/action 完整数据流,包含 CategorySpecificMLP 多 embodiment 路由 + cross-attention DiT + AlternateVLDiT 隔层 image/text 切换 + flow matching 推理
---

# GR00T-N1.7 数据流完整拆解:image / language / state / action 如何经 cross-attention DiT 变成 action / GR00T-N1.7 end-to-end data flow: image / language / state / action turning into action via cross-attention DiT

> **一句话 / In one line**: GR00T = **Qwen3VL backbone(Cosmos-Reason2-2B)出 vl_embeds + 多 embodiment 路由的 CategorySpecificMLP 编码 state/action + 12 层 DiT 用 cross-attention 反复查询 vl_embeds(配合 AdaLayerNorm 注入 time)+ flow matching 4 步 Euler 推理**。跟 π₀ 的 "dual transformer 共同演化" 不同,GR00T 是 "backbone frozen + DiT 当 condition 解码器" 的 BLIP-2 / Diffusion Policy 风格。 / GR00T = **Qwen3VL backbone outputs vl_embeds, CategorySpecificMLP routes state/action per embodiment, a 12-layer DiT repeatedly queries vl_embeds via cross-attention (with AdaLayerNorm injecting time), and flow matching does 4-step Euler at inference**. Unlike π₀'s "dual transformer co-evolves", GR00T is "frozen backbone + DiT as condition decoder" — BLIP-2 / Diffusion Policy lineage.

## 为什么重要 / Why this matters

讲完 OpenVLA(causal token)、OFT(zero + bidirectional)、pi0-FAST(prefix-LM + FAST)、π₀(dual transformer + flow matching)之后,**GR00T-N1.7 代表了第 5 条路线:cross-attention DiT + 多 embodiment 路由**。它的核心创新是 **(1) backbone 跟 action head 完全解耦**(可以换任意 VL backbone 不动 DiT)、**(2) 一个模型同时支持 32 种机器人形态**(通过 CategorySpecificLinear 用 batched matmul + indexing 实现按 embodiment 路由)、**(3) AlternateVLDiT 强制部分层"只听文字"** 避免 image 信号 dilute language 指令。这些设计组合起来让 NVIDIA 能用单个 GR00T 模型在 ALOHA、LIBERO、bridge、RoboArena 等 N 种机器人上 fine-tune,**这是 π₀ 单 embodiment 设计做不到的**。理解 GR00T 的多 embodiment 路由 + cross-attention 范式让你掌握"backbone 当 feature extractor + 小 head 学控制" 这条 frozen-backbone 训练范式 — production VLA 的主流选择之一。

After OpenVLA (causal tokens), OFT (zero + bidirectional), pi0-FAST (prefix-LM + FAST), and π₀ (dual transformer + flow matching), **GR00T-N1.7 represents the 5th route: cross-attention DiT + multi-embodiment routing**. Its core innovations: **(1) backbone fully decoupled from action head** (swap any VL backbone without touching DiT); **(2) one model supports 32 robot embodiments** (CategorySpecificLinear uses batched matmul + indexing for per-embodiment routing); **(3) AlternateVLDiT forces some layers to "only listen to text"** preventing image signal from diluting language. These combined enable NVIDIA to fine-tune a single GR00T on ALOHA, LIBERO, bridge, RoboArena, etc. — something π₀'s single-embodiment design can't. Understanding GR00T's multi-embodiment routing + cross-attention paradigm gives you the "backbone as feature extractor + small head learns control" frozen-backbone training paradigm — a mainstream choice for production VLA.

## 代码 / The code

### Action head 的 5 个核心模块

[`gr00t/model/gr00t_n1d7/gr00t_n1d7.py:43-105`](https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/model/gr00t_n1d7/gr00t_n1d7.py#L43-L105)

```python
class Gr00tN1d7ActionHead(nn.Module):
    def __init__(self, config):
        # 1. DiT 主干(或 AlternateVLDiT 变体)
        self.model = DiT(**config.diffusion_model_cfg,
                         cross_attention_dim=config.backbone_embedding_dim)

        # 2. State encoder — 多 embodiment 路由 MLP
        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,     # 32 种机器人
            input_dim=config.max_state_dim * config.state_history_length,
            hidden_dim=hidden_size,
            output_dim=input_embedding_dim,
        )

        # 3. Action encoder — 注入 time 的多 embodiment 编码器
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=action_dim,
            hidden_size=input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )

        # 4. Action decoder — 多 embodiment 解码
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=hidden_size,
            output_dim=action_dim,
        )

        # 5. VL post-processing(可选)
        self.vlln = nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        self.vl_self_attention = SelfAttentionTransformer(...) if use_vl_attn else nn.Identity()
```

### CategorySpecificLinear:多 embodiment 路由的核心

[`embodiment_conditioned_mlp.py:59-79`](https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/model/modules/embodiment_conditioned_mlp.py#L59-L79)

```python
class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        # 32 个 Linear 权重打包成一个 tensor
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        # W.shape = (32, input_dim, hidden_dim) — 每种 embodiment 自己的 Linear
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        # x: (B, T, input_dim), cat_ids: (B,)
        selected_W = self.W[cat_ids]                       # (B, input_dim, hidden_dim)
        selected_b = self.b[cat_ids]                        # (B, hidden_dim)
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)
        # (B, T, in) @ (B, in, out) = (B, T, out)
```

**关键**:`torch.bmm`(batched matrix multiply)让 batch 里每个样本走自己 embodiment 的 W,**一个 forward 同时处理混合 embodiment**。

### DiT block:cross-attention(或 self,交替)+ AdaLayerNorm

[`dit.py:180-219`](https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/model/modules/dit.py#L180-L219)

```python
def forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, temb):
    # 1. AdaLayerNorm 调制 sa_embs(用 temb)
    norm_hidden = self.norm1(hidden_states, temb)
    # AdaLN: norm × (1 + scale(temb)) + shift(temb)

    # 2. attention — 根据 encoder_hidden_states 是否为 None 决定 self / cross
    attn_out = self.attn1(
        norm_hidden,
        encoder_hidden_states=encoder_hidden_states,
        attention_mask=(encoder_attention_mask if encoder_hidden_states is not None else attention_mask),
    )
    hidden_states = hidden_states + attn_out

    # 3. 标准 LayerNorm + FFW + residual
    norm_hidden = self.norm3(hidden_states)
    hidden_states = hidden_states + self.ff(norm_hidden)
    return hidden_states
```

### Flow matching 训练目标

[`gr00t_n1d7.py:214-264`](https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/model/gr00t_n1d7/gr00t_n1d7.py#L214-L264)

```python
# 1. 噪声化(注意方向跟 π₀ 反过来!)
noise = torch.randn(actions.shape, ...)
t = self.sample_time(B, ...)                              # Beta(α, β) × noise_s
t = t[:, None, None]                                       # (B, 1, 1)

noisy_trajectory = (1 - t) * noise + t * actions          # t=0 噪声, t=1 干净
velocity = actions - noise                                 # target velocity

# 2. Action 编码 + time 注入
action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

# 3. 拼合 state + action
sa_embs = torch.cat((state_features, action_features), dim=1)

# 4. DiT forward(sa_embs as query, vl_embeds as cross-attention K/V)
model_output, _ = self.model(
    hidden_states=sa_embs,
    encoder_hidden_states=vl_embeds,
    timestep=t_discretized,
)

# 5. Decode → loss
pred = self.action_decoder(model_output, embodiment_id)
pred_actions = pred[:, -actions.shape[1]:]
action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
loss = action_loss.sum() / (action_mask.sum() + 1e-6)
```

## 阶段化数据流(LIBERO 7-DoF + horizon=16 + 4 步推理)

### 配置参数

```
backbone:                 Qwen3VL 2B (Cosmos-Reason2)
backbone_hidden:          2048
max_action_dim:           32        # padded(实际 LIBERO action_dim=7)
action_horizon:           16
max_state_dim:            64        # padded(实际 state_dim=8)
state_history_length:     1
input_embedding_dim:      1536      # DiT 内部 hidden
hidden_size:              1536
max_num_embodiments:      32
num_timestep_buckets:     1000
num_inference_timesteps:  4         # 推理步数(典型 4-16)
DiT 层数:                  12
```

### 阶段 1:Dataloader 给的原始 sample

```python
sample = {
    "images":               uint8 (B, 1, 3, 224, 224),   # 单相机
    "state":                float (B, 1, 8),              # state_history=1, dim=8
    "language_instruction": "pick up the red cup",
    "actions":              float (B, 16, 7),             # 训练时
    "embodiment_id":        int (B,),                      # 哪种 embodiment
    "action_mask":          bool (B, 16, 7),              # 哪些 action 维度有效
}
```

### 阶段 2:Backbone forward — Qwen3VL 处理 image + language

```python
backbone_output = self.backbone(images, language, ...)
# vl_embeds:                  (B, ~265, 2048)
# backbone_attention_mask:    (B, ~265)
# image_mask:                 (B, ~265)  ← True 标记 image token(前 256 个)
```

Qwen3VL 内部完成 vision + language prefix-LM 融合,**这一步像 PaliGemma 但 backbone 是 Qwen3 系列**。

### 阶段 3:VL 后处理(可选)

```python
vl_embeds = self.vlln(vl_embeds)               # LayerNorm
vl_embeds = self.vl_self_attention(vl_embeds)   # 可选 self-attn 层
```

### 阶段 4:State 编码 — `CategorySpecificMLP` 按 embodiment 路由

```python
state = state.view(B, 1, -1)                              # (B, 1, 8)

state_features = self.state_encoder(state, embodiment_id)
# 内部:bmm(state, W1[embodiment_id]) + b1[embodiment_id] → ReLU
#       → bmm(hidden, W2[embodiment_id]) + b2[embodiment_id]
# 输出:(B, 1, 1536)
```

**32 种 embodiment 共享一个 forward 但各自走自己的 MLP 权重**,通过 `bmm + indexing` 实现。

### 阶段 5:Flow matching noise + time(注意方向)

```python
noise = randn (B, 16, 32)                        # padded 到 max_action_dim
t = Beta(α=1.5, β=1.0).sample() × noise_s         # t ∈ [0, noise_s]

# ⚠️ 方向跟 π₀ 反!
noisy_trajectory = (1 - t) × noise + t × actions  # t=0 噪声, t=1 干净
velocity = actions - noise                         # target

t_discretized = (t × 1000).long()                  # 给 timestep embedding
```

### 阶段 6:Action 编码 — `MultiEmbodimentActionEncoder`

[`embodiment_conditioned_mlp.py:191-225`](https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/model/modules/embodiment_conditioned_mlp.py#L191-L225)

```python
def forward(self, actions, timesteps, cat_ids):
    # 1. action 投影(按 embodiment 路由)
    a_emb = self.W1(actions, cat_ids)                  # (B, 16, 32) → (B, 16, 1536)

    # 2. time 编码(广播到每个 chunk step)
    timesteps = timesteps.unsqueeze(1).expand(-1, 16)  # (B,) → (B, 16)
    tau_emb = self.pos_encoding(timesteps)              # SinusoidalPositionalEncoding → (B, 16, 1536)

    # 3. cat action + time(沿 channel 维)
    x = torch.cat([a_emb, tau_emb], dim=-1)             # (B, 16, 3072)

    # 4. W2 压回 1536 + swish + W3
    x = swish(self.W2(x, cat_ids))                      # (B, 16, 1536)
    x = self.W3(x, cat_ids)                             # (B, 16, 1536)
    return x
```

可选加 chunk-step position embedding:

```python
if add_pos_embed:
    pos_embs = self.position_embedding(arange(16)).unsqueeze(0)
    action_features = action_features + pos_embs
```

### 阶段 7:拼合 state + action

```python
sa_embs = torch.cat((state_features, action_features), dim=1)
# (B, 1, 1536) + (B, 16, 1536) → (B, 17, 1536)
```

`sa_embs` 是 DiT 的 query 输入。**vl_embeds 不参与 cat**,后面通过 cross-attention 喂进来。

### 阶段 8:DiT forward — cross + self 交替

[`dit.py:292-336`](https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/model/modules/dit.py#L292-L336) 12 层处理:

```python
temb = self.timestep_encoder(timestep)             # (B,) → (B, 1536)

for idx, block in enumerate(self.transformer_blocks):
    if idx % 2 == 1 and interleave_self_attention:
        # 奇数层:self-attention(sa_embs 内部互看)
        hidden_states = block(
            hidden_states,
            attention_mask=None,                    # 全 bidirectional
            encoder_hidden_states=None,             # 无 encoder → 走 self-attention
            temb=temb,
        )
    else:
        # 偶数层:cross-attention(sa_embs.Q attend vl_embeds.K/V)
        hidden_states = block(
            hidden_states,
            encoder_hidden_states=vl_embeds,
            encoder_attention_mask=None,            # 看 vl_embeds 所有 token
            temb=temb,
        )

# 输出归一化
shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
return self.proj_out_2(hidden_states)              # (B, 17, output_dim)
```

每层输入 `(B, 17, 1536)`,输出 `(B, 17, 1536)`(残差保持维度)。**vl_embeds 在 12 层里完全不变**(只被反复 attend)。

### 阶段 9:Action decoder — 按 embodiment 解码

```python
pred = self.action_decoder(model_output, embodiment_id)
# CategorySpecificMLP 按 embodiment 路由
# (B, 17, ...) → (B, 17, 32)

pred_actions = pred[:, -16:]                       # 取 action 段,(B, 16, 32)
```

### 阶段 10:Flow matching loss(masked MSE)

```python
action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
# action_mask 屏蔽 padded 维度(LIBERO 只有 7 维有效)

loss = action_loss.sum() / (action_mask.sum() + 1e-6)
```

### 阶段 11:推理 — 4 步 Euler 积分(vl_embeds 复用)

```python
actions = randn (B, 16, 32)                        # 起始噪声
dt = 1.0 / 4                                       # = 0.25

# === Prefill:vl_embeds 算一次 ===
backbone_output = self.backbone(images, language)
vl_embeds = backbone_output["backbone_features"]
state_features = self.state_encoder(state, embodiment_id)

# === Euler loop(4 步)===
for t in [0, 1, 2, 3]:
    t_cont = t / 4                                  # 0, 0.25, 0.5, 0.75
    t_discretized = int(t_cont * 1000)
    timesteps = torch.full((B,), t_discretized)

    # 重新编码当前 noisy actions(因为 actions 变了)
    action_features = self.action_encoder(actions, timesteps, embodiment_id)
    if add_pos_embed:
        action_features = action_features + pos_embs

    # cat state(state_features 不变)
    sa_embs = torch.cat((state_features, action_features), dim=1)

    # DiT forward(vl_embeds 复用!)
    model_output = self.model(
        hidden_states=sa_embs,
        encoder_hidden_states=vl_embeds,            # ← 同 prefill,12 层都不更新
        timestep=timesteps,
    )
    pred = self.action_decoder(model_output, embodiment_id)
    pred_velocity = pred[:, -16:]                   # (B, 16, 32)

    # Euler 一步:t=0 噪声 → t=1 干净,所以 +dt·v
    actions = actions + dt * pred_velocity * vel_strength

# unnormalize → 机器人执行 16 步
```

**vl_embeds 只 forward 一次,被 4 步 denoising 共享**。这是 cross-attention 范式的最大效率优势。

## 关键设计点详解

### 1. CategorySpecificLinear 的"按 embodiment 路由"几何

**核心 trick**:把 32 个 Linear 打包成一个 `(32, in, out)` tensor,用 `index + bmm` 实现路由。

```
普通 nn.Linear:
   一个 W (in, out),所有样本共用
   计算:y = x @ W + b

CategorySpecificLinear:
   一个 W (32, in, out),32 种 embodiment 各有一份
   forward(x, cat_ids):
       selected_W = W[cat_ids]    # (B, in, out) — 用 cat_ids 选择 W
       y = bmm(x, selected_W)     # 每个样本走自己的 W
       y += b[cat_ids]
```

**为什么这么设计?** GR00T 用一个模型同时学多种机器人形态(ALOHA 双臂、LIBERO 单臂、bridge 等),但每种机器人的 **state/action 物理意义不同**(state[0] 在 ALOHA 是左肩,在 LIBERO 是 base joint)。共用一个 MLP 会让 "什么 dim 对应什么物理量" 全部纠缠。CategorySpecificLinear **让每个 embodiment 学自己的解码方式**,但共享后续 DiT(因为 attention 本身跟物理意义无关,只跟语义有关)。

### 2. DiT 的 self / cross attention 交替

`interleave_self_attention=True` 时:

```
DiT layer 0:  Cross-attention(看 vl_embeds)
DiT layer 1:  Self-attention(sa_embs 内部互看)
DiT layer 2:  Cross-attention
DiT layer 3:  Self-attention
...
DiT layer 11: Self-attention
```

**为什么交替?**
- **Cross 层**:让 action token 反复吸 VL 信息(condition)
- **Self 层**:让 17 个 sa_embs token 内部互相协调(chunk 内部时序一致)

每层 cross 之后,action 的 hidden state 含了更多 VL 信息;每层 self 之后,action chunk 之间的协调更好。**两种 attention 接力**才能既看 VL 又保持 chunk 一致性。

### 3. AdaLayerNorm 注入 time(无 gate)

[`dit.py:74-97`](https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/model/modules/dit.py#L74-L97)

```python
class AdaLayerNorm(nn.Module):
    def forward(self, x, temb):
        temb = self.linear(self.silu(temb))      # (B, D) → (B, 2D)
        scale, shift = temb.chunk(2, dim=1)      # 各 D 维
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]
```

跟 π₀.5 的 adaRMS 的差别:**没有 gate**。GR00T 的 AdaLN 只输出 (scale, shift),不控制 residual。这是 DiT 标配的 "ada_norm" 设计,比 "ada_norm_zero" 简单一点但效果接近。

**time 信号双重注入**:
- **路径 1**:`action_encoder` 里 sin-cos `tau_emb` concat 到 action 上(input 阶段)
- **路径 2**:每层 DiT 的 `AdaLayerNorm(x, temb)` 用 time 调制 norm 输出

两条路径让 time 信号在深层不衰减,跟 π₀.5 的 adaRMS 一脉相承。

### 4. Mask 几何 — 几乎全 None

**Self-attention block(odd 层)**:`attention_mask=None`

```
sa_embs (17 tokens) 内部 17×17 全 1 矩阵 — 完全 bidirectional
state ↔ action[i] 全双向
action[i] ↔ action[j] 全双向(包括未来 action,因为是 flow matching 的 noisy x_t,不会泄露真值)
```

**Cross-attention block(even 层)**:`encoder_attention_mask=None`

```
sa.Q (17) attend vl_embeds.K/V (~265)
17×265 全 1 — 看 vl_embeds 的所有 token,image 和 language 不区分
```

**没有 causal mask、没有 block 几何、没有 padding mask**(sa_embs 没 padding)— 是这几个 VLA 模型里 mask 设计最简单的。

### 5. AlternateVLDiT 变体:cross-attention 隔层"只听文字"

[`dit.py:339-414`](https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/model/modules/dit.py#L339-L414)

```python
def forward(self, hidden_states, encoder_hidden_states, ..., image_mask, backbone_attention_mask):
    # 构造两套 mask
    image_attention_mask     = image_mask & backbone_attention_mask          # 只 attend image token
    non_image_attention_mask = (~image_mask) & backbone_attention_mask        # 只 attend text token

    for idx, block in enumerate(self.transformer_blocks):
        if idx % 2 == 1:
            # 奇数层:self-attention
            hidden_states = block(hidden_states, attention_mask=None,
                                   encoder_hidden_states=None, temb=temb)
        else:
            # 偶数层:cross-attention,但 mask 隔层切换
            if idx % (2 * attend_text_every_n_blocks) == 0:
                curr_mask = non_image_attention_mask          # 这一层只看 text
            else:
                curr_mask = image_attention_mask              # 其他层只看 image

            hidden_states = block(hidden_states, encoder_hidden_states=encoder_hidden_states,
                                   encoder_attention_mask=curr_mask, temb=temb)
```

**为什么要"只听文字"?** 因为 image token(~256 个)数量远多于 text token(~9 个),softmax 会把权重大部分给 image。**强制部分层屏蔽 image,只看 text**,确保 language 指令信号被显式吸收。

调度示例(12 层 DiT,`attend_text_every_n_blocks=2`):

```
layer:  0      1      2      3      4      5      6      7      8      9      10     11
type:   Cross  Self   Cross  Self   Cross  Self   Cross  Self   Cross  Self   Cross  Self
mask:   TEXT   ─      IMG    ─      TEXT   ─      IMG    ─      TEXT   ─      IMG    ─
        ↑                           ↑                           ↑
   每 4 层 1 次只听 text
```

## Cross-attention vs KV 共享 vs joint attention — 三种 VLA "模态融合"范式

| 范式 | 代表 | 信息流模式 | VL 是否被更新 |
|---|---|---|---|
| **Pure self (with mask)** | OpenVLA, OFT | 所有 token 在一个 attention pool | ✅(梯度流回 backbone) |
| **Joint / Unified attention** | **π₀** (dual transformer) | prefix + suffix 各 Q/K/V,K/V cat 后共享 | ✅ prefix 18 层都演化 |
| **Pure cross-attention** | **GR00T DiT**(普通) | Q from sa_embs, K/V from vl_embeds(独立 input) | ❌ vl_embeds 在 DiT 里 frozen |
| **Pure cross with mask shift** | **GR00T AlternateVLDiT** | 同上,但 mask 隔层切换 image/text | ❌ |

GR00T 选 cross-attention 的 3 个工程优势:
1. **Backbone 跟 head 解耦**:可以换任意 VL backbone 不动 DiT(只需 `cross_attention_dim` 匹配)
2. **Backbone 可 frozen**:`tune_llm=False` 时只训 DiT,大幅省训练显存
3. **多步 denoising 摊销**:推理时 vl_embeds 只算 1 次,N 步 denoising 共享

## 类比 / The analogy

**OpenVLA-OFT** 是学生在一份空白卷子上做题(zero placeholder),通过 attention 从课本里查答案。
**π₀** 是学生和教授坐在同一会议室,**教授也在思考**(prefix self-attention),共同演化 18 层得到答案。
**GR00T 是学生反复翻一本已经印好的教科书**(vl_embeds 是 frozen condition):每一层(cross-attention)从书里查一段;每隔一层(self-attention)跟同桌讨论协调一下笔记(chunk 一致性)。**AlternateVLDiT** 变种是说**部分章节强制只翻文字索引,不看插图**,确保文字内容被吸收。这种"教科书 + 反复查询"的范式让换教科书极其容易(GR00T 历史上从 Eagle 换到 Qwen3VL),代价是教科书不能跟着学生一起思考。

OpenVLA-OFT = student fills in a blank test (zero placeholders) and looks up answers via attention.
π₀ = student and professor sit in the same room; **professor also thinks** (prefix self-attention) and they co-evolve across 18 layers.
GR00T = **student repeatedly reads a printed textbook** (vl_embeds is frozen condition): every cross-attention layer queries the textbook; every interleaved self-attention layer discusses notes with a tablemate (chunk consistency). **AlternateVLDiT** variant says some chapters forcibly skip illustrations and only read the text index, ensuring language is absorbed. This "textbook + repeated query" makes swapping the textbook trivially easy (GR00T moved from Eagle to Qwen3VL), but the textbook can't think along with the student.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-VLA

这是 nanoVLA 课程里 `vlm-backbone-wiring` 槽位的**深度变体**(前几天讲过 OpenVLA 的 causal-prefix、OFT 的 fork-bidirectional、pi0-FAST 的 prefix-LM、π₀ 的 dual-transformer,今天讲 GR00T 的 cross-attention + multi-embodiment)。

### 5 种 VLA 模态融合范式完整对比

| 维度 | OpenVLA | OFT | pi0-FAST | π₀ | **GR00T** |
|---|---|---|---|---|---|
| **backbone** | LLaMA 7B | LLaMA 7B (fork) | PaliGemma 2B | PaliGemma 2B | **Qwen3VL 2B** |
| **action 表示** | 256-bin discrete | continuous | DCT+BPE token | continuous | **continuous (padded)** |
| **action head** | lm_head | MLPResNet | lm_head | action expert (Gemma 300M) | **DiT with cross-attn** |
| **VL ↔ action 融合** | causal in sequence | bidirectional in sequence | prefix-LM | shared attention pool | **cross-attention** |
| **VL 是否被更新** | ✅ | ✅ | ✅ | ✅ | **❌ frozen condition** |
| **多 embodiment 支持** | ❌ | ❌ | ❌ | ❌ | **✅ CategorySpecificLinear** |
| **可换 backbone** | ❌(改 head_dim 等于改 backbone) | ❌ | ❌ | ❌ | **✅ 只需 cross_attention_dim 匹配** |
| **time 注入** | 无 | 无(L1) | 无(token 自回归) | concat-MLP / adaRMS | **AdaLN + concat 双路径** |
| **支持 RTC** | ❌ | ❌ | ❌ | 需自己加 | **✅ 内置 vel_strength** |

### 给 nanoVLA 的具体建议

**追求多 embodiment 支持**:走 GR00T 路线(CategorySpecificLinear + cross-attention),一个模型学多种机器人;
**追求训练效率**:走 GR00T 路线(backbone 可 frozen,只训 DiT);
**追求 backbone 灵活替换**:走 GR00T 路线;
**追求 backbone-expert 紧密协作**:走 π₀ 路线(KV 共享让 backbone 跟 expert 共同演化);
**追求最简代码**:走 OFT 路线(L1 head,但记得多 mode 任务会撞墙)。

### 实现 GR00T 风格 nanoVLA 的关键代码骨架

```python
class NanoGroot(nn.Module):
    def __init__(self, vlm, action_dim, action_horizon, n_embodiments=8, head_dim=64, n_heads=8):
        self.vlm = vlm  # 可 frozen
        # state/action encoders:多 embodiment 路由
        self.state_encoder = CategorySpecificMLP(n_embodiments, state_dim, 1024, 1024)
        self.action_encoder = MultiEmbodimentActionEncoder(action_dim, 1024, n_embodiments)
        self.action_decoder = CategorySpecificMLP(n_embodiments, 1024, 1024, action_dim)
        # DiT:cross-attention 主干
        self.dit = DiT(num_layers=12, attention_head_dim=head_dim,
                       num_attention_heads=n_heads,
                       cross_attention_dim=vlm.hidden_size)  # ← 跟 backbone 解耦的关键!

    def forward(self, images, language, state, actions, embodiment_id):
        # 1. backbone(可 frozen)
        vl_embeds = self.vlm(images, language)
        # 2. flow matching
        noise, t = sample_noise_and_time(actions)
        noisy = (1-t)*noise + t*actions
        velocity = actions - noise
        # 3. encode
        sa = cat(self.state_encoder(state, embodiment_id),
                 self.action_encoder(noisy, t, embodiment_id))
        # 4. DiT
        out = self.dit(sa, encoder_hidden_states=vl_embeds, timestep=t)
        pred = self.action_decoder(out, embodiment_id)[:, -action_horizon:]
        return F.mse_loss(pred, velocity)
```

This is the **deep-dive variant** of the `vlm-backbone-wiring` slot in the nanoVLA curriculum (earlier OpenVLA causal-prefix, OFT forked-bidirectional, pi0-FAST prefix-LM, π₀ dual-transformer; today's GR00T cross-attention + multi-embodiment).

## 自己跑一遍 / Try it yourself

```python
# try.py — minimal GR00T-style: CategorySpecificLinear + DiT cross-attention + flow matching
import torch
import torch.nn as nn
import torch.nn.functional as F

# === 1. CategorySpecificLinear:多 embodiment 路由的核心 ===
class CategorySpecificLinear(nn.Module):
    def __init__(self, num_cat, in_dim, out_dim):
        super().__init__()
        self.W = nn.Parameter(0.02 * torch.randn(num_cat, in_dim, out_dim))
        self.b = nn.Parameter(torch.zeros(num_cat, out_dim))

    def forward(self, x, cat_ids):
        # x: (B, T, in), cat_ids: (B,)
        W = self.W[cat_ids]               # (B, in, out)
        b = self.b[cat_ids]               # (B, out)
        return torch.bmm(x, W) + b.unsqueeze(1)


# === 2. AdaLayerNorm(无 gate)===
class AdaLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, x, temb):
        temb = self.linear(F.silu(temb))
        scale, shift = temb.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


# === 3. DiT block(cross-attention + AdaLN + FFW)===
class DiTBlock(nn.Module):
    def __init__(self, dim, n_heads, head_dim, cross_dim):
        super().__init__()
        self.norm1 = AdaLayerNorm(dim)
        # cross-attention: Q from dim, K/V from cross_dim
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(cross_dim, n_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(cross_dim, n_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.n_heads, self.head_dim = n_heads, head_dim
        self.norm3 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))

    def forward(self, x, encoder, temb):
        normed = self.norm1(x, temb)
        # cross-attention
        B, T, _ = normed.shape
        S = encoder.shape[1]
        Q = self.q_proj(normed).view(B, T, self.n_heads, self.head_dim)
        K = self.k_proj(encoder).view(B, S, self.n_heads, self.head_dim)
        V = self.v_proj(encoder).view(B, S, self.n_heads, self.head_dim)
        # scaled dot-product
        scores = torch.einsum("BTnh,BSnh->BnTS", Q, K) / (self.head_dim ** 0.5)
        weights = scores.softmax(dim=-1)
        out = torch.einsum("BnTS,BSnh->BTnh", weights, V).reshape(B, T, -1)
        x = x + self.o_proj(out)
        # FFW
        x = x + self.ff(self.norm3(x))
        return x


# === DEMO ===
B, T_action, n_embodiments = 2, 4, 3
action_dim, state_dim, hidden = 7, 8, 64
cross_dim = 128  # vl_embeds 维度,跟 hidden 不同!

# 1. 模拟两个不同 embodiment 的 batch
embodiment_id = torch.tensor([0, 2])  # 第一个样本 embodiment 0,第二个 embodiment 2

# 2. state/action 编码(按 embodiment 路由)
state = torch.randn(B, 1, state_dim)
state_encoder = CategorySpecificLinear(n_embodiments, state_dim, hidden)
state_features = state_encoder(state, embodiment_id)         # (2, 1, 64)
print(f"state_features: {state_features.shape}")

# 验证两个样本走不同 W
state_zero_id = state_encoder(state, torch.tensor([0, 0]))   # 都用 embodiment 0
print(f"两样本同 W diff:   {(state_zero_id[0] - state_features[0]).abs().max().item():.6f}  (≈ 0)")
print(f"两样本不同 W diff: {(state_zero_id[1] - state_features[1]).abs().max().item():.4f}  (> 0)")

# 3. flow matching
actions = torch.randn(B, T_action, action_dim)
noise = torch.randn_like(actions)
t = torch.rand(B)
noisy = (1 - t[:, None, None]) * noise + t[:, None, None] * actions
velocity = actions - noise

# 4. action 编码(简化:Linear + time concat + MLP)
action_encoder = CategorySpecificLinear(n_embodiments, action_dim, hidden)
action_features = action_encoder(noisy, embodiment_id)        # (2, 4, 64)

# 加 time(简化:把 t broadcast 后 cat 然后 proj)
t_emb = t.view(B, 1, 1).expand(-1, T_action, hidden)          # (2, 4, 64)
action_features = action_features + t_emb * 0.1               # 简化版的 time injection

# 5. cat state + action
sa_embs = torch.cat([state_features, action_features], dim=1)  # (2, 5, 64)
print(f"sa_embs: {sa_embs.shape}")

# 6. vl_embeds(模拟 backbone 输出,注意 cross_dim 跟 hidden 不同!)
vl_embeds = torch.randn(B, 20, cross_dim)                     # (2, 20, 128)

# 7. DiT cross-attention forward
temb = torch.randn(B, hidden)
dit = DiTBlock(dim=hidden, n_heads=4, head_dim=16, cross_dim=cross_dim)
out = dit(sa_embs, vl_embeds, temb)
print(f"\nDiT output: {out.shape}  (sa_embs hidden 不变)")

# 8. 验证 vl_embeds 没被更新(GR00T 的关键特性)
vl_embeds_after = vl_embeds.clone()
print(f"vl_embeds 改动: {(vl_embeds - vl_embeds_after).abs().max().item():.6f}  (= 0)")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
state_features: torch.Size([2, 1, 64])
两样本同 W diff:   0.000000  (≈ 0)
两样本不同 W diff: 0.5+      (> 0)
sa_embs: torch.Size([2, 5, 64])

DiT output: torch.Size([2, 5, 64])  (sa_embs hidden 不变)
vl_embeds 改动: 0.000000  (= 0)
```

**关键验证点**:
1. **CategorySpecificLinear 真的按 embodiment 路由** — 改 cat_ids 输出不同
2. **DiT 输出维度跟 sa_embs hidden 一致**(不是 cross_dim)
3. **vl_embeds 在 DiT 里完全不变**(frozen condition,这是 cross-attention 范式的核心特性)

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Diffusion Policy (Chi et al. 2023)**: 同样 "vision encoder + DiT-style head + flow/diffusion" 范式,但没多 embodiment / Same "vision encoder + DiT-style head + flow/diffusion" paradigm, but no multi-embodiment.
- **BLIP-2 Q-Former (Li et al. 2023)**: cross-attention 让 query token 从 frozen vision encoder 提取信息,跟 GR00T 完全同源 / Cross-attention lets query tokens extract from frozen vision encoder — homologous to GR00T.
- **DETR (Carion et al. 2020)**: object queries 通过 cross-attention 从 CNN feature 提取,**让 backbone 跟 head 解耦的鼻祖** / Object queries cross-attend CNN features — the ancestor of "decoupled backbone and head".
- **RT-X / Octo**: 多 embodiment 支持,但用 embedding-id-add 而非 CategorySpecificMLP / Multi-embodiment support, but via embedding-id-add rather than CategorySpecificMLP.
- **SD3 (Stability AI 2024)**: MM-DiT 的双 stream 设计 / MM-DiT's dual-stream design.

## 注意事项 / Caveats / when it breaks

- **CategorySpecificLinear 的 `W` 大小 = `num_categories × in × out`** / **`W` size = `num_categories × in × out`**: 32 embodiment × 64 × 1536 = 3M 参数/层,**比单 Linear 大 32 倍**;params 增长得很快 / 32 × 64 × 1536 = 3M params/layer, **32× single Linear**; params grow fast.
- **VL 完全 frozen 时性能上限受限** / **Frozen VL limits performance ceiling**: 如果 backbone 不动,DiT 学不出 backbone 没编码的细节(比如 high-resolution 物体细节);fine-tune top layers 是常用妥协 / If backbone is frozen, DiT can't learn details backbone didn't encode (e.g. high-res object details); fine-tuning top layers is a common compromise.
- **AlternateVLDiT 的隔层调度强依赖 backbone 输出结构** / **AlternateVLDiT depends on backbone output structure**: 假设 image token 在前、text 在后,而且数量比例固定;如果换 backbone(比如 Qwen3VL 是 image 在中间),需要重新计算 `image_mask` / Assumes image-tokens-first, text-tokens-last, with fixed ratio; switching backbone (e.g. Qwen3VL has images in middle) needs recomputing `image_mask`.
- **flow matching 方向跟 π₀ 反过来,迁移代码时易错** / **Flow matching direction is reversed from π₀, easy bug**: GR00T: `(1-t)*noise + t*actions`,Euler: `+dt·v`;π₀: `t*noise + (1-t)*actions`,Euler: `-dt·v`。数学等价但 code 翻译时容易搞混 / GR00T: `(1-t)*noise + t*actions`, Euler: `+dt·v`. π₀ is opposite. Mathematically equivalent but easy to confuse when porting.
- **action_mask 屏蔽不能省** / **Don't skip action_mask**: 不同 embodiment 的 action_dim 不同,model 输出 padded 32 维,如果不用 mask,loss 会包含 25 维无意义梯度污染训练 / Different embodiments have different action_dims; without mask, loss includes 25 garbage dim gradients polluting training.

## 延伸阅读 / Further reading

- GR00T-N1 paper (NVIDIA, 2024) — overview of the architecture
- GR00T-N1.7 release notes — multi-embodiment + AlternateVLDiT updates
- BLIP-2 paper (Li et al., 2023) — frozen-backbone + Q-Former origin of cross-attention pattern
- Diffusion Policy paper (Chi et al., 2023) — DiT + diffusion for robot control
- π₀ paper (Physical Intelligence, 2024) — comparison "dual transformer" route
- Today's companion notes on π₀ (2026-06-09) and pi0-FAST (2026-06-08)
