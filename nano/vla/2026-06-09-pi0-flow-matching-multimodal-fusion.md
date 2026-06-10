---
date: 2026-06-09
topic: vla
source: vla
repo: Physical-Intelligence/openpi
file: src/openpi/models/pi0.py
permalink: https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py
difficulty: advanced
read_time: ~18 min
tags: [code-of-the-day, vla, pi0, flow-matching, action-expert, dual-transformer, block-prefix-lm, rectified-flow, nano-vla]
build_role: action-head-continuous (deep-dive variant) — pi0 的"PaliGemma + 并行 action expert + flow matching"组合,以及 image/language/state/action 四模态如何一步步流到 action 的完整数据流
---

# pi0 完整数据流:image / language / state / action 四模态如何流到最终 action / pi0 end-to-end data flow: how image / language / state / action turn into final action

> **一句话 / In one line**: pi0 = **PaliGemma 2B(prefix:vision+language)+ 并行 300M action expert(suffix:state+noisy action+time)+ 4-block prefix-LM mask + rectified flow MSE 训练 + 10 步 Euler 推理积分**。两个 sub-transformer 在 `head_dim=256` 维度共享 attention,在 hidden(2048 vs 1024)各自独立 — 既能用 PaliGemma 大容量学 VLM 知识,又能用小 expert 高效学 robot 控制。 / pi0 = **PaliGemma 2B (prefix: vision+language) + parallel 300M action expert (suffix: state + noisy action + time) + 4-block prefix-LM mask + rectified flow MSE training + 10-step Euler inference integration**. The two sub-transformers share attention at `head_dim=256` while keeping their hidden dims independent (2048 vs 1024) — combining PaliGemma's large-capacity VLM knowledge with the small expert's efficient robot-control learning.

> 📌 **本笔记综合了 6/8-6/9 跟我的多轮对话**,系统讲清楚 pi0 各模态信息如何被一步步操作变成 action 的全流程,**包括 dual transformer、d_head 共享、flow matching 训练-推理一致性、time embedding 的 sin-cos 几何**等多个易混点的精细解释。

## 为什么重要 / Why this matters

讲完 OpenVLA(纯 causal + 256-bin)、OFT(fork bidirectional + zero embedding + L1)、pi0-FAST(prefix-LM + FAST autoregressive)之后,**pi0 原版是 Physical Intelligence 路线的旗舰**:它放弃了 "action as text",改用**独立的 action expert + flow matching** 直接输出连续动作。理解 pi0 的全栈数据流让你掌握 3 个生产级 VLA 设计模式:**(1) dual transformer 让大 VLM 和小机器人 expert 协作**;**(2) block prefix-LM mask 用 ar_mask + cumsum 一行实现 4 个 block 的精细几何**;**(3) flow matching 用 noisy input 天然消除 exposure bias**(训练和推理看到的都是 noisy x_t,不一致问题不存在)。这套组合后来被 π₀-FAST、SmolVLA、GR00T 不同程度借鉴,是当前 VLA 领域最干净的"VLM + 机器人 expert"分工范式。

After OpenVLA (pure causal + 256-bin), OFT (forked bidirectional + zero embedding + L1), and pi0-FAST (prefix-LM + FAST autoregressive), **pi0 original is Physical Intelligence's flagship design**: it abandons "action as text" and uses **a separate action expert + flow matching** to output continuous actions directly. Understanding pi0's full data flow teaches you 3 production-grade VLA design patterns: **(1) dual transformer lets a big VLM and a small robot expert collaborate**; **(2) block prefix-LM mask realizes 4 fine-grained block geometries with one line of `ar_mask + cumsum`**; **(3) flow matching naturally eliminates exposure bias via noisy input** (both training and inference see noisy x_t, so the "mismatch" problem doesn't exist). This combination has since been borrowed in varying degrees by π₀-FAST, SmolVLA, and GR00T — the cleanest "VLM + robot expert" division of labor in current VLA.

## 代码 / The code

### 配置:Gemma sub-network 参数(d_head 共享是关键)

[`src/openpi/models/gemma.py:62-106`](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/gemma.py#L62-L106) 和 [`gemma.py:166-168`](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/gemma.py#L166-L168):

```python
# PaliGemma 2B 配置
gemma_2b:    width=2048, num_heads=8, num_kv_heads=1, head_dim=256

# Action expert 300M 配置
gemma_300m:  width=1024, num_heads=8, num_kv_heads=1, head_dim=256

# 硬约束:两个 sub-network 必须 head_dim, num_heads, num_kv_heads 一致
assert all(config.head_dim    == self.configs[0].head_dim    for config in self.configs)
assert all(config.num_heads   == self.configs[0].num_heads   for config in self.configs)
assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)
```

### 主 forward(compute_loss)

[`src/openpi/models/pi0.py:188-214`](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py#L188-L214):

```python
def compute_loss(self, rng, observation, actions, *, train=False):
    # 1. flow matching 噪声化
    noise = jax.random.normal(noise_rng, actions.shape)
    time  = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
    time_expanded = time[..., None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions   # noisy action
    u_t = noise - actions                                          # target velocity

    # 2. embed prefix(vision + language)和 suffix(state + x_t + time)
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)

    # 3. 4-block prefix-LM mask 构造
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask    = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask  = make_attn_mask(input_mask, ar_mask)

    # 4. dual transformer forward(PaliGemma + action expert,共享 attention)
    (prefix_out, suffix_out), _ = self.PaliGemma.llm(
        [prefix_tokens, suffix_tokens], mask=attn_mask, ...,
        adarms_cond=[None, adarms_cond],
    )

    # 5. 取 suffix 末尾 action_horizon 个 hidden,投到 action_dim,算 velocity MSE
    v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
    return jnp.mean(jnp.square(v_t - u_t), axis=-1)
```

### attention mask 构造(cumsum trick)

[`pi0.py:19-44`](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py#L19-L44):

```python
def make_attn_mask(input_mask, mask_ar):
    """
    mask_ar[i] = True  → 位置 i 开启新 attention block
    mask_ar[i] = False → 位置 i 跟前面同 block
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)                    # 每个位置的 block 编号
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]    # cumsum[j] <= cumsum[i] 可见
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)
```

### 推理:10 步 Euler 积分

[`pi0.py:217-279`](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py#L217-L279):

```python
def sample_actions(self, rng, observation, *, num_steps=10, noise=None):
    dt = -1.0 / num_steps                                    # = -0.1
    noise = noise or jax.random.normal(rng, ...)             # 从纯噪声起

    # prefill prefix:vision + language 只算一次,K/V 缓存
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], ...)

    # Euler 循环 10 步
    def step(carry):
        x_t, time = carry
        suffix_tokens, *, adarms_cond = self.embed_suffix(observation, x_t, time)
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens], ..., kv_cache=kv_cache, adarms_cond=[None, adarms_cond],
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
        return x_t + dt * v_t, time + dt                    # Euler 更新

    x_0, _ = jax.lax.while_loop(cond=lambda c: c[1] >= -dt/2, body=step, init=(noise, 1.0))
    return x_0                                               # 干净 action chunk
```

## 阶段化数据流:4 模态如何变成 action(ALOHA 双臂配置)

用 ALOHA 配置(`action_dim=14, action_horizon=50, 3 个相机`)走完整流程。

### 阶段 1:原始 sample

```python
sample = {
    "images": {"base": (3, 224, 224), "wrist_left": (3, 224, 224), "wrist_right": (3, 224, 224)},
    "state":            (14,) ∈ [-1, 1],     # 关节角,归一化(连续值,不离散化!)
    "tokenized_prompt": (48,) int32,          # 只有 task 文字,无 state 字符串
    "actions":          (50, 14) ∈ [-1, 1],   # 训练时有,推理时无
}
```

### 阶段 2:训练时 sample noise + time(rectified flow 噪声化)

```python
noise = N(0, I) (B, 50, 14)
time  = Beta(1.5, 1) × 0.999 + 0.001 ∈ (0.001, 1.0)  # 偏向后段
x_t = time × noise + (1 - time) × actions             # noisy action(线性插值)
u_t = noise - actions                                  # target velocity(直线斜率)
```

**关键**:模型从这一阶段开始**只看到 `x_t` 和 `time`**,真值 `actions` 仅在最终 loss 出现一次。

### 阶段 3:embed_prefix(image + language)

**3.1 Image → SigLIP**:
```
images (3 个相机) × (3, 224, 224)
  → SigLIP-So400m/14:patchify (k=14, s=14) + 27 层 ViT → (256, 1152)
  → 内置 projector Linear(1152→2048)
  → (256, 2048) × 3 cat → (B, 768, 2048)

ar_mask += [False] × 768  # 整个 vision 内部 bidirectional
```

**3.2 Language → PaliGemma BPE embed**:
```
tokenized_prompt (48,) int32
  → Gemma embedding lookup (vocab=257152, width=2048)
  → (B, 48, 2048)

ar_mask += [False] × 48   # language 也 bidirectional
```

**3.3 拼合 prefix**: `(B, 816, 2048)`, `ar_mask = [False] × 816`(整个 prefix 一个 block)

### 阶段 4:embed_suffix(state + noisy action + time)

**4.1 state → 1 个 token**:
```
state (14,) → Linear(14→1024) → (1, 1024)  ← 直接连续值,无离散化!
ar_mask += [True]                          # state 单独一个新 block
```

**4.2 x_t → 50 个 token**:
```
x_t (50, 14) → Linear(14→1024) → (50, 1024)
```

**4.3 time → sin-cos embedding → MLP**(详见后面"time embedding"章节):
```
time scalar → posemb_sincos(time, 1024, 4e-3, 4.0) → (1024,)
```

**4.4 time 与 action concat + MLP**:
```
time_tokens = repeat(time_emb, "emb -> 50 emb") → (50, 1024)
action_time = cat([action_tokens, time_tokens], axis=-1) → (50, 2048)
action_time = MLP_in(2048→1024) → swish → MLP_out(1024→1024) → (50, 1024)

ar_mask += [True] + [False] × 49    # action 整体一个新 block,内部 bidirectional
```

**4.5 拼合 suffix**: `(B, 51, 1024)`, `ar_mask` 末尾 51 位 = `[True, True, False × 49]`

### 阶段 5:4-block prefix-LM mask 构造

```
位置:     0..815          816    817        818..866
content:  vision+language state  action[0]  action[1..49]
ar_mask:  0..0            1      1          0, 0, ..., 0
cumsum:   0..0            1      2          2, 2, ..., 2
          ↑               ↑      ↑          ↑
        block 0         block 1  block 2(action 整体一个 block)
```

`attn_mask[i, j] = (cumsum[j] <= cumsum[i])` 决定可见性:

| | prefix(cumsum=0) | state(cumsum=1) | action(cumsum=2) |
|---|---|---|---|
| **prefix** | ✓ bidir | ✗ | ✗ |
| **state** | ✓ | ✓(自己) | ✗ |
| **action** | ✓ | ✓ | ✓ **50 个互相 bidir** |

**关键**:action 之间是 **bidirectional** — 50 个 action token 内部全连接。但 π₀ 通过 `ar_mask` 实现,**不需要 fork transformers**(OFT 必须做的事情)。

### 阶段 6:dual transformer forward(两个 sub-network 共享 attention)

#### 6.1 各自算 Q/K/V,投到共享的 `head_dim=256` 空间

```
prefix tokens (B, 816, 2048)              suffix tokens (B, 51, 1024)
       ↓ RMSNorm_p                                 ↓ RMSNorm_s
       ↓ paligemma_q_proj                          ↓ expert_q_proj
       ↓ 权重 (8, 2048, 256)                       ↓ 权重 (8, 1024, 256)
       ↓                                          ↓
Q_p (B, 816, 8, 256)                       Q_s (B, 51, 8, 256)
                                                  ↑
                          ←──── head_dim=256 强制一致 ────→
```

**关键**:
- PaliGemma:`num_heads × head_dim = 8 × 256 = 2048` = hidden(标准 MHA)
- Expert:`num_heads × head_dim = 8 × 256 = 2048` ≠ hidden=1024(**non-standard 扩张**!attention 内部空间比 hidden 大 2×)

**为什么 d_head 一致就能共享?** 因为 `Q·K^T / sqrt(d_head)` 内积发生在 `head_dim=256` 这一维,跟 hidden 多大无关。Q/K/V proj 的本质是 `Linear(hidden → head_dim × num_heads)`,**只要 output 维度统一,input 维度可以任意**。

#### 6.2 K/V cat 共享,attention 一起算

```
K_all = cat([K_p, K_s], dim=1) → (B, 867, 1, 256)  # num_kv_heads=1, GQA
V_all = cat([V_p, V_s], dim=1) → (B, 867, 1, 256)

# 两段 Q 各自 attend 同一个 K_all/V_all:
attn_out_p = softmax(Q_p @ K_all.T + mask_p) @ V_all  → (B, 816, 8, 256)
attn_out_s = softmax(Q_s @ K_all.T + mask_s) @ V_all  → (B, 51, 8, 256)
```

数学上等价于 "Q 也 cat 起来" 然后一起算 — **"K/V 拼 Q 不拼" 是 misleading 的说法**,真正的不对称在于 prefix 和 suffix 用不同的 `*_proj` 权重 + 不同的 `o_proj` 投回独立的 hidden。

#### 6.3 O proj 投回各自 hidden

```
attn_out_p (8, 256, 2048)             attn_out_s (8, 256, 1024)
       ↓ paligemma_o_proj                       ↓ expert_o_proj
       ↓                                        ↓
(B, 816, 2048)                          (B, 51, 1024)
       ↓ residual + MLP_p              ↓ residual + MLP_s
       ↓                                        ↓
prefix (B, 816, 2048)                  suffix (B, 51, 1024)
```

**18 层都走这套流程**。Hidden dim 只在进 attention 之前和出 attention 之后有差异;attention 内部全部用 `head_dim=256` 共享空间。

### 阶段 7:action_out_proj → velocity

```
suffix_out[:, -50:] (B, 50, 1024)
       ↓ action_out_proj = Linear(1024 → 14)
v_t (B, 50, 14)  ← 模型预测的 velocity field
```

### 阶段 8:训练 loss(MSE on velocity)

```
loss = mean((v_t - u_t)²)  =  mean((v_t - (noise - actions))²)
```

### 阶段 9:推理 — 10 步 Euler 积分

```
prefill:  prefix forward 1 次 → KV cache(vision+language 永久缓存)

Euler loop:
  x_t = noise  (time = 1.0)
  for step in 10:
    suffix = embed_suffix(state, x_t, time)
    v_t = action_out_proj(forward_suffix_using_kv_cache)
    x_t ← x_t + (-0.1) * v_t     # Euler 一步
    time += -0.1
  
  x_0 ≈ clean action chunk → unnormalize → robot 执行
```

## 关键设计点详解

### 1. time embedding 的 sin-cos 几何(易混点)

[`pi0.py:48-63`](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py#L48-L63):

```python
def posemb_sincos(pos, embedding_dim, min_period=4e-3, max_period=4.0):
    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)   # 512 个
    period = min_period * (max_period / min_period) ** fraction  # 几何级数 [4e-3, 4.0]
    sinusoid_input = jnp.einsum("i,j->ij", pos, 1.0 / period * 2 * jnp.pi)
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)
```

**正确的几何理解**(易错!):

- **前 512 维全是 sin**,**后 512 维全是 cos**
- **每个 512 内部**索引从 0 到 511,**频率从高到低**(`ω₀ = 2π/4e-3 ≈ 1571 rad`,`ω₅₁₁ = 2π/4.0 ≈ 1.57 rad`)
- index `i` 和 index `i+512` 共享同一频率 `ω_i`,只是相位不同(sin vs cos)
- **高低频按"前段索引 vs 后段索引"分,跟 sin/cos 划分无关**

```
              ω 高 ────────────────► ω 低
              index→ 0   1   2  ... 510 511

前 512 维     sin: ────────────────────────
              (相位 0)

后 512 维     cos: ────────────────────────
              (相位 π/2)
              index→ 512 513 ...      1023
              ω 高 ────────────────► ω 低
```

低 index 区域(高频)对 `t` 微小变化敏感(微调精度),高 index 区域(低频)感知 `t` 的大尺度量级(整体 noise level)。sin/cos 配对是为了给每个频率提供完整复数信息 `cos(ωt) + i·sin(ωt)`,**不是用来分高低频的**。

### 2. dual transformer 的 d_head 共享(易混点)

#### 为什么 hidden 不同但 d_head 一致就能共享?

**hidden 是 token 的"外部表示维度"**,`d_head` 是 attention 内部的"内积维度"。Q/K/V proj 是 `Linear(hidden, num_heads × head_dim)` 的矩阵,可以把任意 hidden 投到固定的 `num_heads × head_dim` 空间。

```
PaliGemma:
  hidden=2048
  → Q proj (8, 2048, 256)
  → 输出: (B, L, 8, 256)
  → num_heads × head_dim = 8 × 256 = 2048 (= hidden,标准 MHA)

Expert:
  hidden=1024
  → Q proj (8, 1024, 256)
  → 输出: (B, L, 8, 256)
  → num_heads × head_dim = 8 × 256 = 2048 (> hidden=1024,扩张 MHA)
```

**Expert 是 non-standard 设计**:`num_heads × head_dim > hidden`,attention 内部空间比 hidden 大 2×。这给 Expert 的 attention 部分"更多容量"而 FFN 部分维持小 hidden(省 FLOPs)。

#### 为什么不用一套权重 + mask 就完了?

数学上**可以**!OpenVLA / OFT / pi0-FAST 都是"一套权重 + mask"。π₀ 选 dual transformer 的真正原因是:

| 原因 | 一套权重(全 2B) | dual(2B + 300M) |
|---|---|---|
| **FLOPs** | suffix 51 token 也用 2B 算 | suffix 用 300M,**省 4×** |
| **预训练保护** | suffix 梯度污染 PaliGemma | suffix 梯度只更新 expert |
| **LR / 冻结策略** | 只能一刀切 | PaliGemma 低 LR,expert 高 LR |
| **容量分配** | 均匀 | **不均**:VLM 大容量,robot 小容量 |

**Mask 解决信息流方向,dual transformer 解决参数容量分配**。

### 3. flow matching 训练-推理一致性(为什么 bidirectional 不会泄露)

**有人可能会问**:action 内部 bidirectional,推理时没有真值 action,训练测试不就不一致了吗?

**答案**:π₀ 训练时模型也**从来不看 ground truth `actions`**,看到的只有:
- `x_t = t·noise + (1-t)·actions`(被噪声污染的中间状态)
- `time`(当前 noise level)

ground truth `actions` 只在 loss 那一行出现一次(`u_t = noise - actions`)。

**推理时每一步的 `(x_t, t)`** 都落在训练分布里:

```
推理 step 0: x_t = noise         (time = 1.0)  ← 训练 t=1.0 时见过
推理 step 1: x_t = noise + dt·v_t (time = 0.9)  ← 训练 t=0.9 时见过类似 x_t
...
推理 step 10: x_t ≈ clean action  (time = 0.0)  ← 训练 t≈0 时见过
```

训练时 `t ~ Beta(1.5, 1)` 覆盖 (0.001, 1.0) 整个连续区间,**推理 10 个离散 t 点都落在训练分布里 → 没有 train-test mismatch**。

这是 flow matching / diffusion 模型设计的普遍优势 — 它们用"加噪-去噪"循环天然消除 exposure bias。

### 4. 4 个模态在 attention mask 里的方向性总结

```
                key (被看)
              prefix  state  action
query  prefix  bidir   ✗      ✗     ← VLM 内部双向融合
       state    ✓     bidir   ✗     ← state 看 VLM,自己一个 block
       action   ✓      ✓    bidir   ← action 看全部,内部 50 个互相
```

- **vision 看 language**:✓(prefix 内部双向)— 不像 OpenVLA(causal 钉死)需要 FiLM 补救
- **action 看 state**:✓ — 机器人控制必须知道当前关节状态
- **state 看不到 action**:✗ — 因果上 state 是"现在"不该看到"未来"
- **prefix 看不到 state/action**:✗ — VLM 不被 robot 特定信号"污染"

## 类比 / The analogy

想象一个**大型百科教授(2B PaliGemma)+ 小型机器人控制学生(300M expert)**的会诊室。

教授桌上摆着百科全书(images, 256 patches)和病历提问(language tokens, 48 个),教授内部互相对照(prefix bidirectional),整理出对场景的理解。

学生坐在教授旁边,听教授介绍场景(student.Q attend prefix),自己手里有当前的关节状态(state token)和一份"模糊的未来 action 草稿"(noisy x_t)。学生不能直接看到真实未来 action(那是预知未来),但可以看草稿的所有 50 步(action 内部 bidirectional)— 这就像看到一份**水印模糊的答案纸**,看得到草稿的整体形状,但具体数字被噪声盖住。

学生的任务:**给定当前的模糊草稿,告诉自己"现在应该往哪个方向擦掉一点模糊"**(预测 velocity field)。重复 10 次擦淡(10 步 Euler),最终草稿变清晰,这就是要执行的真实 action chunk。

整个会诊过程**没有任何环节学生看到真实未来 action**;但学生通过反复练习"任意模糊程度下都能往清晰方向走一步",学会了一种 universal 的"去模糊"能力。这就是 flow matching 训练-推理一致性的精髓。

Imagine a **big encyclopedia professor (2B PaliGemma) + small robot-control student (300M expert)** in a consultation room. On the professor's desk are the encyclopedia (image patches) and the patient's question (language tokens); the professor cross-references internally (prefix bidirectional) to form an understanding of the scene. The student sits next to the professor, hears the scene explanation (student.Q attends prefix), holds the current joint state (state token) and a "blurry draft of future actions" (noisy x_t). The student can't see the true future action, but can look at all 50 steps of the draft (action bidirectional) — like seeing a **watermarked, blurry answer sheet** with overall shape visible but specific values obscured. The student's job: given the current blurry draft, **say which direction to erase a bit of blur** (predict velocity field). Repeat 10 times, and the draft becomes clear — that's the action chunk to execute. **At no point does the student see the true future action**; through practice the student learns a universal "deblurring" skill at any noise level — that's the heart of flow matching's train-test consistency.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-VLA

这是 nanoVLA 课程里 `action-head-continuous` 槽位的**综合 deep-dive 变体**(之前 6/08 笔记讲过 GR00T 的 flow matching action head 和 OFT 的 L1 head,今天系统讲 π₀ 的"dual transformer + flow matching"组合)。

### 设计选择空间

```
        ┌─────────────────────────────────────────┐
        │ nanoVLA action 生成的 4D 设计空间          │
        ├─────────────────────────────────────────┤
        │ 1. action 表示 │ discrete (256-bin/FAST) │
        │               │ vs continuous (linear)   │
        │ 2. 训练目标    │ CE                       │
        │               │ vs L1                    │
        │               │ vs flow matching MSE     │
        │               │ vs diffusion ε-pred      │
        │ 3. 解码方式    │ autoregressive           │
        │               │ vs parallel (1-step)     │
        │               │ vs iterative (N-step)    │
        │ 4. backbone   │ pure VLM (shared params) │
        │               │ vs VLM + expert (dual)   │
        └─────────────────────────────────────────┘
```

**几个历史名作的组合**:
- OpenVLA:**discrete + CE + AR + pure VLM**
- OpenVLA-OFT (L1):**continuous + L1 + parallel + pure VLM (forked)**
- pi0-FAST:**discrete (FAST) + CE + AR + pure VLM**
- **pi0 原版:continuous + flow matching MSE + iterative (10 step) + dual VLM + expert**
- π₀.5:同 pi0 但用 adaRMS 替代 concat-MLP 注入 time
- SmolVLA:**continuous + flow matching + iterative + dual VLM + expert** (借鉴 pi0)
- GR00T:**continuous + flow matching + DiT head + dual VLM + DiT expert**

### 给 nanoVLA 的具体建议

**对小机器人任务(< 7-DoF, horizon < 10)**:用 OFT-L1 路线即可,简单快;
**对多 mode 任务(双臂,有多种合法解)**:必须 flow matching(L1 会平均出错解);
**对长 horizon + 高精度(ALOHA, dexterous)**:走 π₀ 路线 dual transformer + flow matching;
**资源紧张但要 fast inference**:pi0-FAST 路线;
**追求架构简单 + 复用预训练**:OpenVLA 路线。

### 写一个最小 π₀ 在自己 nanoVLA 里的关键代码

```python
# Inside your nanoVLA module:
class NanoPi0(nn.Module):
    def __init__(self, vlm, action_dim, action_horizon, vlm_hidden=2048, expert_hidden=1024, head_dim=256, n_heads=8):
        # 必须强制 d_head 和 n_heads 一致(vlm 和 expert 都用)
        self.vlm = vlm
        self.expert = ParallelTransformer(hidden=expert_hidden, n_heads=n_heads, head_dim=head_dim)
        self.state_proj = nn.Linear(action_dim, expert_hidden)
        self.action_in_proj = nn.Linear(action_dim, expert_hidden)
        self.action_out_proj = nn.Linear(expert_hidden, action_dim)
        self.action_time_mlp_in = nn.Linear(2 * expert_hidden, expert_hidden)
        self.action_time_mlp_out = nn.Linear(expert_hidden, expert_hidden)
    
    def embed_suffix(self, state, x_t, time):
        state_token = self.state_proj(state)[:, None]              # (B, 1, 1024)
        action_tokens = self.action_in_proj(x_t)                    # (B, H, 1024)
        time_emb = posemb_sincos(time, 1024, 4e-3, 4.0)            # (B, 1024)
        time_tokens = einops.repeat(time_emb, "b e -> b h e", h=action_tokens.shape[1])
        action_time = self.action_time_mlp_out(swish(self.action_time_mlp_in(
            torch.cat([action_tokens, time_tokens], dim=-1)
        )))
        return torch.cat([state_token, action_time], dim=1)         # (B, 1+H, 1024)
```

This is the **comprehensive deep-dive variant** of the `action-head-continuous` slot in the nanoVLA curriculum (6/08 covered GR00T's flow matching head and OFT's L1 head; today systematically covers π₀'s "dual transformer + flow matching" combo).

## 自己跑一遍 / Try it yourself

```python
# try.py — minimal dual transformer with shared d_head + flow matching forward
import torch
import torch.nn as nn
import torch.nn.functional as F

D_HEAD = 8        # 共享 head_dim
N_HEADS = 4
ACTION_DIM = 3
ACTION_HORIZON = 5

class TinyDualAttention(nn.Module):
    def __init__(self, prefix_hidden=32, suffix_hidden=16):
        super().__init__()
        # 两套权重,但 num_heads × head_dim 都投到一致空间 (4 × 8 = 32)
        self.prefix_qkv = nn.Linear(prefix_hidden, 3 * N_HEADS * D_HEAD, bias=False)
        self.suffix_qkv = nn.Linear(suffix_hidden, 3 * N_HEADS * D_HEAD, bias=False)
        self.prefix_o = nn.Linear(N_HEADS * D_HEAD, prefix_hidden, bias=False)
        self.suffix_o = nn.Linear(N_HEADS * D_HEAD, suffix_hidden, bias=False)

    def forward(self, prefix_tokens, suffix_tokens, attn_mask):
        B, L_p, _ = prefix_tokens.shape
        _, L_s, _ = suffix_tokens.shape
        L = L_p + L_s
        
        # Q/K/V 投到共享 head_dim 空间
        qkv_p = self.prefix_qkv(prefix_tokens).reshape(B, L_p, 3, N_HEADS, D_HEAD)
        qkv_s = self.suffix_qkv(suffix_tokens).reshape(B, L_s, 3, N_HEADS, D_HEAD)
        Q_p, K_p, V_p = qkv_p.unbind(dim=2)  # (B, L_p, n_heads, d_head)
        Q_s, K_s, V_s = qkv_s.unbind(dim=2)
        
        # K/V cat 共享(Q 在数学上也可以 cat,这里分开算)
        K_all = torch.cat([K_p, K_s], dim=1)  # (B, L, n_heads, d_head)
        V_all = torch.cat([V_p, V_s], dim=1)
        Q_all = torch.cat([Q_p, Q_s], dim=1)
        
        # attention 在 head_dim 维度内积
        scores = torch.einsum("blnh,bLnh->bnlL", Q_all, K_all) / (D_HEAD ** 0.5)
        scores = scores.masked_fill(~attn_mask[:, None, :, :], float("-inf"))
        weights = scores.softmax(dim=-1)
        out = torch.einsum("bnlL,bLnh->blnh", weights, V_all)  # (B, L, n_heads, d_head)
        out = out.reshape(B, L, N_HEADS * D_HEAD)
        
        # O proj 还原回各自 hidden
        prefix_out = self.prefix_o(out[:, :L_p])  # 回到 prefix_hidden=32
        suffix_out = self.suffix_o(out[:, L_p:])  # 回到 suffix_hidden=16
        return prefix_out, suffix_out


def make_4block_attn_mask(L_vision, L_lang, L_state, L_action):
    """4-block prefix-LM mask via ar_mask + cumsum"""
    ar_mask = torch.tensor(
        [False] * (L_vision + L_lang)  # prefix block 0
        + [True]                        # state block 1
        + [True] + [False] * (L_action - 1)  # action block 2 (1 个 True 启动 + 内部互相 attend)
    )
    cumsum = ar_mask.cumsum(0)
    # attn_mask[i, j] = cumsum[j] <= cumsum[i]
    return cumsum[None, None, :] <= cumsum[None, :, None]


def posemb_sincos(t, dim=16, min_p=4e-3, max_p=4.0):
    half = dim // 2
    fraction = torch.linspace(0, 1, half)
    period = min_p * (max_p / min_p) ** fraction
    angles = t.unsqueeze(-1) * (1.0 / period * 2 * 3.14159)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


# === DEMO ===
B = 1
L_vision, L_lang = 4, 3
prefix_tokens = torch.randn(B, L_vision + L_lang, 32)

# flow matching: sample noise, time, x_t
actions = torch.randn(B, ACTION_HORIZON, ACTION_DIM)
noise   = torch.randn(B, ACTION_HORIZON, ACTION_DIM)
time    = torch.distributions.Beta(1.5, 1.0).sample((B,)) * 0.999 + 0.001
x_t     = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
u_t     = noise - actions

# embed suffix
state = torch.randn(B, ACTION_DIM)
state_proj = nn.Linear(ACTION_DIM, 16)
action_in_proj = nn.Linear(ACTION_DIM, 16)
state_token = state_proj(state)[:, None]
action_tokens = action_in_proj(x_t)
time_emb = posemb_sincos(time)
time_tokens = time_emb[:, None].expand(-1, ACTION_HORIZON, -1)
mlp_in = nn.Linear(32, 16); mlp_out = nn.Linear(16, 16)
action_time = mlp_out(F.silu(mlp_in(torch.cat([action_tokens, time_tokens], dim=-1))))
suffix_tokens = torch.cat([state_token, action_time], dim=1)

# build 4-block mask
attn_mask = make_4block_attn_mask(L_vision, L_lang, 1, ACTION_HORIZON)
print(f"attn_mask shape: {attn_mask.shape}")
print(f"prefix (block 0) sees: {attn_mask[0, 0, :].int().tolist()}  ← only prefix")
print(f"state (block 1) sees:  {attn_mask[0, L_vision+L_lang, :].int().tolist()}  ← prefix + state")
print(f"action[0] (block 2) sees: {attn_mask[0, L_vision+L_lang+1, :].int().tolist()}  ← all!")

# forward
dual_attn = TinyDualAttention(prefix_hidden=32, suffix_hidden=16)
prefix_out, suffix_out = dual_attn(prefix_tokens, suffix_tokens, attn_mask.expand(B, -1, -1))
print(f"\nprefix_out: {prefix_out.shape}   (prefix_hidden=32)")
print(f"suffix_out: {suffix_out.shape}   (suffix_hidden=16)")

# velocity prediction + flow matching loss
action_out_proj = nn.Linear(16, ACTION_DIM)
v_t = action_out_proj(suffix_out[:, -ACTION_HORIZON:])
loss = (v_t - u_t).pow(2).mean()
print(f"\nflow matching loss: {loss.item():.4f}")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
attn_mask shape: torch.Size([1, 13, 13])
prefix (block 0) sees: [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  ← only prefix
state (block 1) sees:  [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  ← prefix + state
action[0] (block 2) sees: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  ← all!

prefix_out: torch.Size([1, 7, 32])   (prefix_hidden=32)
suffix_out: torch.Size([1, 6, 16])   (suffix_hidden=16)

flow matching loss: 1.xxxx
```

**注意 prefix_hidden=32 ≠ suffix_hidden=16**(模仿 π₀ 的 2048 vs 1024),但 attention 内部都用 `head_dim=8 × n_heads=4 = 32` 共享空间。

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **π₀.5 (Physical Intelligence)**:同 π₀ 但用 adaRMS 替代 concat-MLP 注入 time / Same as π₀ but uses adaRMS instead of concat-MLP for time injection.
- **SmolVLA (HuggingFace)**:借鉴 π₀ 的 dual transformer + flow matching,但用 SmolVLM 作 backbone / Borrows π₀'s dual transformer + flow matching, but uses SmolVLM as backbone.
- **GR00T-N1 (NVIDIA)**:同样 dual VLM + expert,但 expert 用 DiT block(而非 transformer)/ Same dual VLM + expert, but expert uses DiT block (not transformer).
- **RoboFlamingo / OpenFlamingo VLA**:用 cross-attention 而非 KV 共享让 expert 跟 VLM 通信 / Use cross-attention instead of KV sharing for expert-VLM communication.
- **Diffusion Policy (Chi et al., 2023)**:用 DDPM iterative sampling 但没有 VLM,只用 vision encoder + diffusion head / Uses DDPM iterative sampling but no VLM, just vision encoder + diffusion head.

## 注意事项 / Caveats / when it breaks

- **head_dim 必须严格一致** / **head_dim must strictly match**: 两个 sub-network 的 `head_dim, num_heads, num_kv_heads` 必须完全一致,否则 K/V cat 就 shape 不匹配 / The two sub-networks' `head_dim, num_heads, num_kv_heads` must be exactly equal, otherwise K/V cat won't shape-match.
- **rectified flow 要求训练数据足够 dense** / **Rectified flow needs dense training data**: 噪声水平 `t ~ Beta(1.5, 1)` 要在 (0, 1) 全区间被覆盖,否则推理时遇到训练未见过的 t 会失败 / The noise level `t ~ Beta(1.5, 1)` must cover (0, 1) — out-of-distribution `t` at inference will fail.
- **10 步 Euler 不够时 action 不收敛** / **10-step Euler insufficient leads to non-convergent action**: 简单任务足够,但对复杂双臂协调可能需要 20-50 步 / Sufficient for simple tasks; complex bimanual coordination may need 20-50 steps.
- **KV cache 内存压力** / **KV cache memory**: prefix 816 token × 18 层 × `head_dim=256 × num_kv_heads=1` = ~3.7M float per sample;batch=8 + bf16 = 60 MB,对小 GPU 仍是压力 / Prefix 816 tokens × 18 layers × 256 × 1 = ~3.7M float per sample; batch=8 + bf16 = 60 MB, still pressure on small GPUs.
- **bidirectional action 会让 flow matching 更难训** / **Bidirectional actions make flow matching harder to train**: 因为每个 token 看到的"模糊邻居"互相影响,训练 loss 比单向高 ~30%,但最终精度更好 / Each token sees noisy neighbors that interact, training loss is ~30% higher than unidirectional, but final precision is better.
- **time 用 sin-cos 比 learned positional embedding 更 robust** / **Sin-cos time is more robust than learned positional embedding**: 因为可以推广到训练未见过的 t,而 learned embedding 不能 / Generalizes to unseen `t`, learned embedding doesn't.

## 延伸阅读 / Further reading

- π₀ paper: https://www.pi.website/research/pi0 — original "vision-language-action model"
- Rectified Flow paper (Liu et al., 2022): https://arxiv.org/abs/2209.03003 — the linear noise interpolation that π₀ uses
- PaliGemma paper (Beyer et al., 2024) — the prefix-LM mask design source
- Gemma paper — Gemma 2B and the GQA design
- Flow Matching for Generative Modeling (Lipman et al., 2023) — theory background
- Today's pi0-FAST companion notes (2026-06-08) — comparison between flow-matching and FAST tokenization
