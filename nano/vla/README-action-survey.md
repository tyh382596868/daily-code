# VLA 仓库"如何添加 action"对比汇总 / How different VLA repos add action — a survey

> 本文汇总各 VLA / WAM 仓库把**机器人动作**接入模型的不同方式,作为 nanoVLA / nanoWAM
> 从头实现时的选型参考。每种方式都链接到对应的逐行讲解笔记。
>
> This doc surveys how different VLA / WAM repos wire **robot actions** into the model,
> as a design-selection reference for building nanoVLA / nanoWAM from scratch. Each
> approach links to its line-by-line note.

最后更新 / Last updated: 2026-05-29

---

## 0. 三个正交的设计维度 / Three orthogonal design axes

把"添加 action"拆成三个独立问题,任何 VLA 都是这三个维度的一种组合:

Adding action decomposes into three independent questions; every VLA is one combination of these axes:

| 维度 / Axis | 选项 / Options |
|---|---|
| **A. 动作表示 / Action representation** | 离散 token (discrete bins) ↔ 连续回归 (continuous regression) |
| **B. 动作生成方式 / Generation mechanism** | 自回归 token (autoregressive) · 一步回归 (single-shot) · flow matching / diffusion · CVAE |
| **C. 动作与骨干的关系 / Action-vs-backbone wiring** | 共享骨干 (shared) · 独立 expert (separate expert) · 平行 DiT (parallel DiT) · register token |

下面按"维度 A + B"分两大流派(离散 / 连续),再讲"维度 C"的接线方式。

Below: two schools by axes A+B (discrete / continuous), then the axis-C wiring patterns.

---

## 1. 离散派 / The discrete school

### OpenVLA — action tokenizer (256 分箱)

**做法 / Approach**: 把每个动作维度均匀分成 256 个 bin,映射到 LLM 词表里**最少用的 256 个 token**,动作预测就变成普通的 next-token 自回归 + 交叉熵 —— 跟语言建模一模一样。

Bin each action dimension into 256 uniform buckets, map them onto the **256 least-used tokens** in the LLM vocabulary, and action prediction becomes ordinary next-token autoregression + cross-entropy — identical to language modeling.

```python
# openvla/prismatic/vla/action_tokenizer.py
self.bins = np.linspace(min_action, max_action, self.n_bins)   # 256 bins per dim
# action value → bin index → least-used LLM token id
```

- **维度 / Axes**: A=离散, B=自回归 token, C=共享骨干(就是 LLM 本身)
- **优点 / Pros**: 零新增架构 —— 复用 LLM 的全部机制(词表、解码、KV cache);天然支持变长。 / Zero new architecture — reuses the entire LLM stack.
- **缺点 / Cons**: 离散化损失精度(256 档对精细操作不够);自回归逐 token 解码慢。 / Discretization loses precision; autoregressive decoding is slow.
- **笔记 / Note**: [2026-05-10 OpenVLA action tokenizer](../../2026/05/2026-05-10-openvla-action-tokenizer-example.md) · 训练入口 [2026-05-28 OpenVLA training step](./2026-05-28-openvla-training-step.md)

---

## 2. 连续派 / The continuous school

### π₀ (openpi) — action expert + flow matching

**做法 / Approach**: 在 PaliGemma 旁边挂一个独立的 "action expert"(另一个 Gemma 配置),用 `action_in_proj` 把连续动作投成 token,flow matching 训练,`action_out_proj` 投回动作维度,一次预测 `action_horizon` 步。

Attach a separate "action expert" (another Gemma config) beside PaliGemma; `action_in_proj` projects continuous actions to tokens, flow matching trains it, `action_out_proj` projects back, predicting `action_horizon` steps at once.

```python
# openpi/src/openpi/models/pi0.py
self.action_in_proj  = nnx.Linear(action_dim, action_expert_config.width)
self.action_out_proj = nnx.Linear(action_expert_config.width, action_dim)
v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])   # predicted velocity
```

- **维度 / Axes**: A=连续, B=flow matching, C=独立 expert (cross-attn 进 VLM)
- **优点 / Pros**: 精度高;flow matching 少步推理;expert 比全 VLM 小。 / High precision; few-step inference; expert smaller than full VLM.
- **缺点 / Cons**: 比离散派多一套 expert 参数和 flow matching 机制。 / Extra expert params + flow-matching machinery.

### GR00T (lerobot/groot + Isaac-GR00T) — flow matching head + 多本体

**做法 / Approach**: 跟 π₀ 同属 flow-matching 连续派,但 action encoder 用 `MultiEmbodimentActionEncoder`(= `CategorySpecificLinear`),**每种机器人本体一份权重**,一个模型服务多种机器人。

Same flow-matching continuous school as π₀, but the action encoder is a `MultiEmbodimentActionEncoder` (= `CategorySpecificLinear`) — **one weight slice per robot embodiment**, one model serving many robots.

```python
# lerobot groot flow_matching_action_head.py
noisy_trajectory = (1 - t) * noise + t * actions
velocity = actions - noise                    # flow-matching target
loss = F.mse_loss(pred_actions, velocity) * action_mask
```

- **维度 / Axes**: A=连续, B=flow matching, C=独立 head + 多本体路由
- **笔记 / Note**: [2026-05-29 GR00T flow-matching action head](./2026-05-29-groot-flow-matching-action-head.md) · 多本体路由 [2026-05-29 CategorySpecificLinear](../../2026/05/2026-05-29-isaac-groot-category-specific-linear.md)

### ACT (lerobot) — CVAE + L1 + temporal ensemble

**做法 / Approach**: 最早的 chunking 派 —— transformer encoder-decoder 一次预测一整段动作,CVAE 建模多模态,L1 loss(不是 MSE),推理用 temporal ensemble 平滑。

The original chunking approach — a transformer encoder-decoder predicts a whole action chunk, a CVAE models multimodality, L1 loss (not MSE), and temporal ensemble smooths inference.

```python
# lerobot act modeling_act.py
l1_loss = (abs_err * valid_mask).sum() / num_valid       # L1, not MSE
loss = l1_loss + mean_kld * self.config.kl_weight        # + CVAE KL term
```

- **维度 / Axes**: A=连续, B=CVAE 单步回归, C=独立 encoder-decoder
- **笔记 / Note**: [2026-05-29 ACT action chunking](./2026-05-29-act-action-chunking.md)

---

## 3. "动作放哪"接线方式 / Where action tokens live (axis C)

同样是连续动作,"action token 跟视觉/语言 token 怎么共处"有四种工程方案。WAM 那边把这四种讲得最清楚(因为 WAM 还要额外塞视频 latent):

Even for continuous actions, "how action tokens coexist with vision/language tokens" has four engineering patterns. The WAM notes cover these most clearly (WAM also juggles video latents):

| 方案 / Pattern | 代表 / Example | 一句话 / In one line | 笔记 / Note |
|---|---|---|---|
| **共享骨干 single-stream** | lingbot-va | action 当 token 塞进同一个 DiT,两个 Linear 进出,靠 mask 隔离 | [lingbot action embedder](../wam/2026-05-29-lingbot-action-embedder.md) |
| **平行 DiT decoupled** | FastWAM | 单独搭一个 ActionDiT,复用 DiTBlock 类但参数独立,可单独部署 | [FastWAM ActionDiT](../wam/2026-05-29-fastwam-action-dit.md) |
| **register token integrated** | dreamzero | action/state 当 register 挂在 video 序列尾,各用独立 1-D RoPE | [dreamzero action registers](../wam/2026-05-29-dreamzero-action-registers.md) |
| **cross-attn expert** | SmolVLA / π₀ | 小 expert 做 query,cross-attend 大 VLM 的 KV | [SmolVLA VLM+expert](./2026-05-29-smolvla-vlm-with-expert.md) |

---

## 4. 选型决策树 / Selection decision tree

```
要从头搭 action 输出?
build action output from scratch?
│
├─ 想最快跑通、复用 LLM 全套机制?
│  want fastest path, reuse the whole LLM stack?
│  └─→ 离散派 discrete: OpenVLA action tokenizer (256 bins + cross-entropy)
│
├─ 需要精细操作精度(插孔/倒水)?
│  need fine-manipulation precision (insertion/pouring)?
│  └─→ 连续派 continuous + flow matching: π₀ / GR00T head
│       └─ 多种机器人? multi-robot? → 加 CategorySpecificLinear 多本体编码
│
└─ action 和 video latent 要联合建模(WAM)?
   joint action + video modeling (WAM)?
   ├─ 想最简单、改动最小? → single-stream (lingbot-va)
   ├─ action 要独立部署到机器人? → parallel DiT (FastWAM)
   └─ 要精细的 video↔action 对齐? → register tokens (dreamzero)
```

**几乎所有现代 VLA 都额外叠加这两件 / Almost every modern VLA also layers on:**

- **Action chunking**(一次预测一段、队列消费)— 解决控制频率 vs 推理延迟矛盾。
  Predict a horizon, consume via queue — resolves control-frequency vs inference-latency.
  → [ACT action chunking](./2026-05-29-act-action-chunking.md)
- **异步推理 client-server** — 推理延迟被"边走边算"掩盖。
  Async client-server — inference latency hidden behind "compute while moving".
  → [lerobot async inference](./2026-05-29-lerobot-async-inference.md)
- **LoRA 微调** — 把预训练大模型适配到自己的机器人。
  LoRA fine-tuning — adapt a pretrained big model to your own robot.
  → [OpenVLA LoRA finetune](./2026-05-29-openvla-lora-finetune.md)

---

## 5. 一句话总结每家 / One-liner per repo

| 仓库 / Repo | 动作表示 / Repr | 生成 / Gen | 接线 / Wiring | 特色 / Signature |
|---|---|---|---|---|
| **OpenVLA** | 离散 256-bin | 自回归 token | 共享 LLM | 把动作当语言,零新增架构 |
| **π₀ (openpi)** | 连续 | flow matching | 独立 action expert | PaliGemma + Gemma expert |
| **GR00T** | 连续 | flow matching | 独立 head + 多本体 | CategorySpecificLinear 一模型多机器人 |
| **ACT** | 连续 | CVAE 单步 | encoder-decoder | temporal ensemble 平滑 |
| **SmolVLA** | 连续 | flow matching | cross-attn expert | 冻结 VLM + 瘦 expert,消费级 GPU |
| **lingbot-va** (WAM) | 连续 | flow matching | single-stream | action 与 video latent 共流 |
| **FastWAM** (WAM) | 连续 | flow matching | 平行 ActionDiT | action DiT 可独立部署 |
| **dreamzero** (WAM) | 连续 | flow matching | register token | action/state 各用独立 1-D RoPE |

---

## 6. 相关笔记索引 / Related notes index

**nanoVLA 组件 / nanoVLA components:**
- [Vision encoder — patch embedding](./2026-05-29-nanovlm-patch-embed.md)
- [VLM backbone + action expert wiring](./2026-05-29-smolvla-vlm-with-expert.md)
- [Continuous action head (flow matching)](./2026-05-29-groot-flow-matching-action-head.md)
- [Action chunking](./2026-05-29-act-action-chunking.md)
- [LoRA fine-tune](./2026-05-29-openvla-lora-finetune.md)
- [Async inference loop](./2026-05-29-lerobot-async-inference.md)
- [Training step (discrete route)](./2026-05-28-openvla-training-step.md)
- [Action tokenizer (discrete representation)](../../2026/05/2026-05-10-openvla-action-tokenizer-example.md)

**nanoWAM 的 action 接线 / nanoWAM action wiring:**
- [single-stream (lingbot-va)](../wam/2026-05-29-lingbot-action-embedder.md)
- [parallel DiT (FastWAM)](../wam/2026-05-29-fastwam-action-dit.md)
- [register tokens (dreamzero)](../wam/2026-05-29-dreamzero-action-registers.md)
- [action-frame fusion mask (lingbot-va)](../wam/2026-05-29-lingbot-flex-mask-compose.md)
