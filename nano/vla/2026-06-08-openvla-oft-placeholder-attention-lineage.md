---
date: 2026-06-08
topic: vla
source: vla
repo: openvla/openvla-oft
file: prismatic/extern/hf/modeling_prismatic.py
permalink: https://github.com/moojink/openvla-oft/blob/main/prismatic/extern/hf/modeling_prismatic.py#L618-L621
difficulty: advanced
read_time: ~15 min
tags: [code-of-the-day, vla, design-pattern, mask-token, parallel-decode, openvla-oft, mae, act, perceiver, nano-vla]
build_role: vlm-backbone-wiring (deep-dive variant) — the "placeholder + positional encoding + attention fills in" design pattern, traced from BERT (2018) to OpenVLA-OFT (2025)
---

# 把要预测的位置塞 placeholder + 让 attention bidirectional 填空 — 这条设计线从 BERT 走到了 OFT / Place placeholders at to-be-predicted positions + let bidirectional attention fill in — a design lineage from BERT (2018) to OpenVLA-OFT (2025)

> **📌 订正 (2026-06-08) / Correction**: 本笔记初版把 OFT 归类为 "causal self-attention + zero placeholder",**这是错的**。OFT 实际通过 fork 过的 `transformers` 库([commit `bc339d9`](https://github.com/moojink/transformers-openvla-oft/commit/bc339d9))把 LLaMA 的 causal mask 替换成 **全序列 bidirectional**,然后再叠加 zero placeholder。所以 OFT 在 4D 设计空间里的位置是 "**zero + RoPE + bidirectional self + MLP**",不是 "causal self"。下面所有"4D 表"和"attention 类型"已修正。
>
> **📌 Correction (2026-06-08)**: An earlier version of this note classified OFT as "causal self-attention + zero placeholder" — **this was wrong**. OFT actually forks the `transformers` library ([commit `bc339d9`](https://github.com/moojink/transformers-openvla-oft/commit/bc339d9)) to replace LLaMA's causal mask with **full-sequence bidirectional attention**, then stacks zero placeholders on top. So OFT's position in the 4D design space is "**zero + RoPE + bidirectional self + MLP**", not "causal self". All "4D tables" and "attention type" sections below are corrected accordingly.

> **一句话 / In one line**: OpenVLA-OFT 的"action 位置塞零 + 全序列 bidirectional 填空"**不是新发明**,它是 NAT (2017) → BERT (2018) → DETR (2020) → Perceiver (2021) → MAE (2021) → Q-Former (2023) → ACT (2023) 这条 8 年学术谱系的最新一环 — 同一个 idea 的不同变体,OFT 选了"零 placeholder + bidirectional mask"这个组合。 / OFT's "zero action positions + full-sequence bidirectional fill-in" **isn't a new invention** — it's the latest variant of an 8-year academic lineage: NAT (2017) → BERT (2018) → DETR (2020) → Perceiver (2021) → MAE (2021) → Q-Former (2023) → ACT (2023). Same idea, different incarnations; OFT picked the "zero placeholder + bidirectional mask" combination.

## 为什么重要 / Why this matters

"把 transformer 输入里要预测的位置塞 placeholder,让 self/cross-attention 从 condition 单向把信息流过来填补" — 这是过去 8 年深度学习里反复出现的设计模式。每次出现都被研究者当作创新发表(BERT 的 mask LM、DETR 的 object queries、MAE 的 mask token、ACT 的 action queries、OFT 的 zero action embedding),其实**底层是同一个 idea 的不同实例化**。掌握这条谱系有两个价值:(1) 你看任何新的多模态/生成模型,都能秒看出它用的是哪种 placeholder 策略;(2) 你设计 nanoVLA 时知道自己有几个选择(zero / learned / cross-attended / iterative mask)及它们的工程取舍。

"Putting placeholders at positions to be predicted, letting self/cross-attention pull information from a condition prefix" — this design pattern has repeatedly resurfaced in deep learning for the past 8 years. Each time it gets published as an innovation (BERT's masked LM, DETR's object queries, MAE's mask token, ACT's action queries, OFT's zero action embedding), but underneath they're **all instantiations of the same idea**. Mastering this lineage gives you two things: (1) you can immediately read any new multimodal/generative model and identify its placeholder strategy; (2) when designing your own nanoVLA, you know your options (zero / learned / cross-attended / iterative mask) and their engineering trade-offs.

## 代码 / The code

### OFT 的"灵魂 1 行"再次出现 / OFT's soul-line, again

[`prismatic/extern/hf/modeling_prismatic.py:618-621`](https://github.com/moojink/openvla-oft/blob/main/prismatic/extern/hf/modeling_prismatic.py#L618-L621)

```python
all_actions_mask = all_actions_mask.unsqueeze(-1)        # (B, seq_len, 1)
input_embeddings = input_embeddings * ~all_actions_mask  # ← action 位置 input → 0
```

让我们把这条"placeholder-attention"谱系的每个里程碑也用代码对照一下。

### 谱系 1: BERT (Devlin et al. 2018) — 起源

```python
# BERT 训练: 15% token 替换成 [MASK]
inputs = ["The", "cat", "[MASK]", "on", "the", "mat"]
embeddings = embed(inputs)
# [MASK] 是一个**learnable non-zero embedding**,attention 让它吸上下文
outputs = bert(embeddings)
logits_at_mask = lm_head(outputs[:, 2, :])  # predict "sits"
```

### 谱系 2: NAT (Gu et al. 2017) — 并行 decode 鼻祖

```python
# Non-Autoregressive Translation:整个 target sequence 全 placeholder
encoder_out = encoder(source_sentence)
target_placeholders = nn.Parameter(torch.randn(target_length, D))  # learnable
# decoder 一次 forward 出整句话(用 source 作 cross-attention condition)
target_sentence = decoder(target_placeholders, encoder_out)
```

### 谱系 3: DETR (Carion et al. 2020) — Object Queries

```python
# 100 个 learnable object queries,从 CNN feature 提取 100 个物体
object_queries = nn.Parameter(torch.randn(100, D))   # learnable, non-zero
cnn_features = backbone(image)
detections = transformer_decoder(object_queries, cnn_features)
# 每个 query 输出一个 bbox + class
```

### 谱系 4: Perceiver IO (DeepMind 2021) — 统一接口

```python
# N 个 learned latent 阵列,从任意模态 input 提取 N 个输出
input_array = encoder(any_modality)                  # 任意尺寸
latents = nn.Parameter(torch.randn(N, D))            # learnable
# cross-attention:latents 从 input_array 吸信息
extracted = cross_attn(latents, input_array)
# 再 self-attention on latents
output = self_attn_blocks(extracted)
```

### 谱系 5: MAE (He et al. 2021) — **最像 OFT**

```python
# Encoder 只看 25% visible patches
encoded_visible = encoder(visible_patches)

# Decoder 在 mask 位置塞 [MASK] token + positional embedding
mask_token = nn.Parameter(torch.randn(1, 1, D))      # learnable, non-zero, shared across all mask positions
positional_embedding = nn.Parameter(torch.randn(N_PATCHES, D))
decoder_input = scatter(encoded_visible, mask_token, mask_positions)
decoder_input = decoder_input + positional_embedding  # ← position 信息让不同 mask 位置区分

reconstructed = decoder(decoder_input)
# loss: 重建原始 pixel
```

### 谱系 6: Q-Former (BLIP-2, Li et al. 2023) — 多模态版本

```python
# 32 个 learnable queries 从 frozen vision encoder 提取
queries = nn.Parameter(torch.randn(32, D))           # learnable, non-zero
vision_features = frozen_vit(image)
text_features = bert_text(text)

# 两步 attention:queries 既 cross-attend vision 又 self-attend text
queries = cross_attn(queries, vision_features)
queries = self_attn(torch.cat([queries, text_features]))
# queries 浓缩了 vision-language 联合信息,送给下游 LLM
```

### 谱系 7: ACT (Zhao et al. 2023) — **OFT 的直接前身**

```python
# Action Chunking Transformer
class ACT(nn.Module):
    def __init__(self, chunk_len, action_dim):
        self.encoder = TransformerEncoder()
        self.decoder = TransformerDecoder()
        self.action_queries = nn.Parameter(torch.randn(chunk_len, D))  # learnable

    def forward(self, image, joint_state, language):
        condition = self.encoder(image, joint_state, language)
        # action_queries 通过 cross-attention 吸 condition,output K 个 action
        action_hidden = self.decoder(self.action_queries, condition)
        actions = self.action_head(action_hidden)
        return actions  # (B, chunk_len, action_dim)

loss = F.l1_loss(actions, ground_truth)  # ← L1 loss,OFT 直接复用
```

### 谱系 8: OpenVLA-OFT (2025) — 最新一环

```python
# 在 LLaMA-2 7B 内部用 zero placeholder
input_embeddings = input_embeddings * ~all_actions_mask  # 56 个 action 位置 → 0
# RoPE 位置编码自动区分 56 个槽位(MAE 的 positional embedding 角色)
output = llama_2_7b(input_embeddings)
action_hidden = output.hidden_states[-1][:, action_positions]
# MLPResNet 解码(ACT 的 action_head 角色)
actions = MLPResNet(action_hidden.reshape(B, K, -1))
loss = F.l1_loss(actions, ground_truth)  # ← 沿用 ACT 的 L1
```

## 逐行讲解 / What's happening

### 1. **同一 idea 的 4 个轴**
中文: 8 个工作都在同一 4D 空间里:
   - **placeholder 形式**: 0 (OFT) / learned shared (MAE [MASK]) / learned per-slot (ACT, DETR queries) / random init (Perceiver)
   - **位置编码**: RoPE (OFT) / additive positional embedding (MAE, ACT) / implicit by query identity (DETR, Perceiver)
   - **attention 类型**: bidirectional self-attn (**OFT**, MAE, NAT) / cross-attn (DETR, Perceiver, ACT) / causal self-attn (BERT MLM 时段除外、原版 OpenVLA、π0-FAST 的 action 段)
   - **解码方式**: lm_head (BERT) / detection head (DETR) / action head (ACT, OFT) / linear (MAE)

English: All 8 works live in the same 4D design space:
   - **Placeholder form**: 0 (OFT) / learned shared (MAE [MASK]) / learned per-slot (ACT, DETR queries) / random init (Perceiver)
   - **Positional encoding**: RoPE (OFT) / additive positional embedding (MAE, ACT) / implicit query identity (DETR, Perceiver)
   - **Attention type**: bidirectional self-attn (**OFT**, MAE, NAT) / cross-attn (DETR, Perceiver, ACT) / causal self-attn (OpenVLA original, π0-FAST action segment)
   - **Decode head**: lm_head (BERT) / detection head (DETR) / action head (ACT, OFT) / linear (MAE)

### 2. **为什么 OFT 选"字面零"** — 与 MAE 选"learned mask token"的对比
中文: MAE 选 learned mask token,因为 ViT 是从头训的,加一个 vector 参数无所谓。OFT 是 fine-tune 现成的 LLaMA-2,**不想动 embedding 矩阵**(LLaMA vocab 末尾本来就有 action token id 占着 slot,如果换 learned mask 还得新加 parameter,且要决定加在哪)。**零的方案完美利用了已有 vocab slot,只乘个掩码,零额外参数**。

English: MAE picks a learned mask token because ViT trains from scratch — adding a vector parameter is trivial. OFT fine-tunes pretrained LLaMA-2 and **doesn't want to touch the embedding matrix** (the last 256 vocab slots are already occupied by action token ids; a learned mask would require adding new parameters and deciding where). **The zero approach perfectly reuses existing vocab slots — just multiply by a mask, zero extra parameters**.

### 3. **为什么 RoPE 比 additive positional embedding 更适合 OFT**
中文: MAE 用 additive positional embedding `decoder_input = input + pos_emb`,即使 input 是 mask token,加上 pos_emb 也立刻非零,第一层就开始位置区分。OFT 用 RoPE,**RoPE 是乘法旋转,作用在 Q 和 K 上**,如果 input 是零,Q 也是零,RoPE 旋转零还是零 → 第一层位置无法区分。但 LLaMA-2 是 RoPE 训的,**改成 additive pos_emb 等于背叛预训练权重**。所以 OFT 只能选 RoPE,代价是第一层"位置区分"延迟 1 层(从第二层开始)。

English: MAE uses additive positional embedding `decoder_input = input + pos_emb` — even if input is the mask token, adding pos_emb makes it non-zero, and layer 1 already differentiates positions. OFT uses RoPE, which is **multiplicative rotation applied to Q and K**. If input is zero, Q is zero, RoPE(0) = 0, so layer 1 cannot differentiate. But LLaMA-2 was pretrained with RoPE — switching to additive pos_emb would betray the pretrained weights. So OFT picked RoPE and accepts a 1-layer delay in position differentiation (starts at layer 2).

### 4. **bidirectional self vs cross-attention 的选择**
中文: DETR / Perceiver / ACT 都用 cross-attention,query 独立于 condition prefix。**这种设计的好处**:query 数量和 condition 长度解耦,可以处理可变长度 input。**坏处**:增加了 cross-attention 模块,训练参数多。OFT 选了"复用 LLaMA self-attention 模块,但通过 fork transformers 把 causal mask 改成 bidirectional",query 直接坐在序列里 — **完全不加模块,完全复用 LLaMA 32 层的 attention 权重** — 但代价是 (1) 要 fork transformers 改 mask 几何(参数全 0、不破坏 LLaMA 预训练 attention 模式),(2) action positions 必须坐在 condition 之后才能利用 RoPE 的相对位置信息。

English: DETR / Perceiver / ACT all use cross-attention; queries are independent of the condition prefix. **Pros**: query count and condition length are decoupled, can handle variable-length inputs. **Cons**: adds cross-attention modules, more training params. OFT picked "reuse LLaMA self-attention modules, but fork transformers to swap causal mask for bidirectional"; queries sit directly in the sequence — **zero added modules, full reuse of LLaMA's 32 layers of attention weights** — but the costs are (1) needing a transformers fork to modify mask geometry (parameters all zero, so it doesn't break LLaMA's pretrained attention patterns), and (2) action positions must sit after the condition to leverage RoPE relative positions.

### 5. **从 BERT 到 OFT — 真正"被发明"的部分只有 chunking + L1 + zero**
中文: 8 年 8 个工作,大部分都在重复一个 idea。**真正属于 OFT 的"新东西"**有 3 项:(1) action chunking — 借自 ACT;(2) L1 regression head — 借自 ACT;(3) **literally zero embedding** — 这是 OFT 的工程美学贡献,因为它是为"复用 7B LLaMA"才有意义。如果换成 100M 小模型,选 learned mask 反而更稳。

English: 8 years, 8 works, mostly repeating one idea. **What's actually new in OFT** is 3 things: (1) action chunking — borrowed from ACT; (2) L1 regression head — borrowed from ACT; (3) **literally zero embedding** — OFT's engineering aesthetic contribution, only meaningful because they reuse 7B LLaMA. For a 100M model, learned mask would be safer.

## 类比 / The analogy

想象**一座图书馆 + 一台填表机**。

- **BERT**: 让你抄一本书,中间 15% 的字被涂掉,你按上下文猜。
- **NAT**: 让你给一句法语句子写英文翻译,但所有 placeholder "____" 都摆在桌上,你一次性填完(不允许逐字填)。
- **DETR**: 给你 100 张写着"图里第 i 个物体是什么"的查询卡,从图书馆翻找答案填上。
- **MAE**: 把一本旧书 75% 的页撕掉,在缺页位置塞标着页码的空白页,你按现有页 + 页码顺序补全。
- **ACT**: 给你 8 张写着"未来第 i 步动作是什么"的查询卡,从图书馆(图像+状态)找答案。
- **OFT**: 跟 ACT 一样,但**查询卡是完全空白的,只有页码** — 你必须从图书馆里反复抄信息到这些卡上,32 轮抄完后,卡片自然变得有内容。

Picture **a library + a form-filling machine**.

- **BERT**: Transcribe a book, 15% of the words blanked out; you guess from context.
- **NAT**: Translate French to English, but **all** placeholder "____"s sit on the table; you fill all at once (no word-by-word).
- **DETR**: 100 query cards saying "what's object #i in this image?" — fetch answers from the library to fill them.
- **MAE**: 75% of an old book's pages are torn out; you get blank pages with page numbers at the missing spots; fill from remaining pages + page-number order.
- **ACT**: 8 query cards saying "what's action #i in the future?" — fetch answers from the library (images + state).
- **OFT**: Same as ACT, but **the query cards are completely blank, only page numbers** — you must copy info from the library onto them repeatedly; after 32 rounds, the cards spontaneously have content.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-VLA

这是 nanoVLA 课程里 `vlm-backbone-wiring` 槽位的**深度变体**(5/29 SmolVLA 讲 expert-cross-attn,6/08 早些时候讲 OpenVLA causal-prefix,今天讲 OFT 的"零 placeholder 退化 LM 成 condition 编码器")。

理解这条谱系给你的工程价值:**当你为自己的 nanoVLA 选择 action 解码策略时,你实际上是在 4D 设计空间里做选择**。

```
nanoVLA action 解码空间:
┌─────────────────────────────────────────────────────────┐
│ Placeholder       │ 0 / learned-shared / learned-per-slot │
│ Position 编码     │ RoPE / additive pos / query identity  │
│ Attention 类型    │ causal self / cross / bidirectional   │
│ Decode head       │ lm_head / MLP / DiT / linear          │
└─────────────────────────────────────────────────────────┘

某些组合 = 历史名作:
   零        + RoPE     + bidirectional self + MLP   → OpenVLA-OFT (L1)        ← fork transformers 改 mask
   learned   + additive + bidirectional        + MLP → MAE (image, not action)
   learned   + identity + cross                + DiT → DETR / ACT / GR00T
   noisy GT  + RoPE     + bidirectional self   + DiT → OpenVLA-OFT (Diffusion) ← 同样的 fork
   learned   + cross    + self                 + cross → BLIP-2 Q-Former
   token id  + RoPE     + causal self          + lm_head → 原版 OpenVLA / π0-FAST action 段
```

**对 nanoVLA 的具体建议**:

1. **如果复用预训练 LLaMA-2 / Qwen / Mistral 且追求极致 parallel decode**:走 OFT 路线(零 + RoPE + bidirectional + MLP),代价是要 fork transformers 改 mask(参数全 0,不破坏 LLaMA 预训练 attention 模式);
2. **如果从头训小模型(<1B)**:走 ACT 路线(learned query + cross-attn),收敛更稳;
3. **如果要 multi-mode action 分布**:走 OFT-Diffusion 路线(noisy continuous + DiT),flow matching 头也行;
4. **如果只想极简快速 prototype**:用 BERT-style learned mask + bidirectional attention + linear head,代码 100 行内写完。

**production-grade 考虑**:实际工程上,**频繁打错的点是 placeholder 形式 vs 模型规模的 mismatch**。给 100M 模型用 OFT 的"字面零"会非常难收敛(信息量不够 bootstrap 出位置感),给 7B 模型用 BERT-style learned mask 又浪费(本来 RoPE 就够了)。读懂上面的 4D 表格能帮你避坑。

This is the **deep-dive variant** of the `vlm-backbone-wiring` slot in the nanoVLA curriculum (5/29 SmolVLA covered expert cross-attention, 6/08 earlier covered OpenVLA causal prefix, today covers OFT's "zero placeholder degrades LM into condition encoder" pattern).

The engineering value of understanding this lineage: **when you pick an action-decoding strategy for your nanoVLA, you're actually choosing in a 4D design space**.

```
nanoVLA action-decoding design space:
┌─────────────────────────────────────────────────────────────┐
│ Placeholder       │ 0 / learned-shared / learned-per-slot     │
│ Position encoding │ RoPE / additive pos / query identity      │
│ Attention type    │ causal self / cross / bidirectional       │
│ Decode head       │ lm_head / MLP / DiT / linear              │
└─────────────────────────────────────────────────────────────┘

Specific combinations = famous works:
   zero      + RoPE     + bidirectional self + MLP   → OpenVLA-OFT (L1)        ← forks transformers to flip mask
   learned   + additive + bidirectional       + MLP  → MAE (image, not action)
   learned   + identity + cross               + DiT  → DETR / ACT / GR00T
   noisy GT  + RoPE     + bidirectional self  + DiT  → OpenVLA-OFT (Diffusion) ← same fork
   learned   + cross    + self                + cross → BLIP-2 Q-Former
   token id  + RoPE     + causal self         + lm_head → OpenVLA original / π0-FAST action segment
```

**Concrete recommendations for nanoVLA**:

1. **Reusing pretrained LLaMA-2 / Qwen / Mistral and want maximum parallel decode**: take the OFT route (zero + RoPE + bidirectional + MLP); the cost is forking `transformers` to modify the mask geometry (params all zero, so it doesn't damage LLaMA's pretrained attention patterns);
2. **Training a small model from scratch (<1B)**: take the ACT route (learned query + cross-attn), more stable convergence;
3. **Need multi-mode action distribution**: take the OFT-Diffusion route (noisy continuous + DiT), flow matching head also works;
4. **Just want a fast minimal prototype**: BERT-style learned mask + bidirectional + linear head, under 100 lines.

**Production-grade pitfall**: the most common mistake is a **mismatch between placeholder form and model scale**. Giving a 100M model the "literal zero" treatment makes it hard to bootstrap position awareness; giving a 7B model BERT-style learned mask is wasteful (RoPE already suffices). The 4D table above is your map to avoid these.

## 自己跑一遍 / Try it yourself

```python
# try.py — compare 3 placeholder strategies (zero / learned-shared / learned-per-slot)
# on the same toy "predict from condition" task
import torch, torch.nn as nn, torch.nn.functional as F

D, N_COND, N_PRED, EPOCHS = 32, 16, 8, 500

# Toy task: prediction position i 的答案 = condition 的第 i mod N_COND 个 token 内容(线性变换后)
condition_data = torch.randn(64, N_COND, D)
ground_truth_w = torch.randn(D, D)
ground_truth = torch.stack([condition_data[:, i % N_COND, :] @ ground_truth_w for i in range(N_PRED)], dim=1)

class Encoder(nn.Module):
    def __init__(self, mode):  # 'zero' / 'shared' / 'per_slot'
        super().__init__()
        self.mode = mode
        self.pos_emb = nn.Parameter(torch.randn(N_COND + N_PRED, D) * 0.02)
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(D, 4, dim_feedforward=4*D, batch_first=True)
            for _ in range(2)
        )
        self.head = nn.Linear(D, D)
        if mode == 'shared':
            self.mask_token = nn.Parameter(torch.randn(1, 1, D) * 0.02)
        elif mode == 'per_slot':
            self.per_slot_queries = nn.Parameter(torch.randn(1, N_PRED, D) * 0.02)

    def forward(self, condition):
        B = condition.shape[0]
        if self.mode == 'zero':
            placeholders = torch.zeros(B, N_PRED, D)
        elif self.mode == 'shared':
            placeholders = self.mask_token.expand(B, N_PRED, -1)
        else:  # per_slot
            placeholders = self.per_slot_queries.expand(B, -1, -1)
        seq = torch.cat([condition, placeholders], dim=1) + self.pos_emb[None, :, :]
        for blk in self.blocks:
            seq = blk(seq)
        return self.head(seq[:, -N_PRED:])

for mode in ['zero', 'shared', 'per_slot']:
    model = Encoder(mode)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(EPOCHS):
        pred = model(condition_data)
        loss = F.mse_loss(pred, ground_truth)
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"{mode:10s} | final loss = {loss.item():.4f} | params = {sum(p.numel() for p in model.parameters())}")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
zero       | final loss = 0.05~0.20 | params = ~25000   ← OFT 路线,需要更多 epoch 学位置
shared     | final loss = 0.02~0.10 | params = ~25032   ← MAE 路线,+1 个 D-dim vector
per_slot   | final loss = 0.01~0.05 | params = ~25256   ← ACT/DETR 路线,+N_PRED 个 D-dim vector,最稳
```

**收敛速度**:`per_slot > shared > zero`。**参数量**:`per_slot > shared > zero`。这就是 4D 设计空间的真实 trade-off — **更多 placeholder 容量 → 收敛更快、效果更好,但参数更多**。生产场景按"模型规模 × 数据量"决定选哪个。

**Convergence speed**: `per_slot > shared > zero`. **Param count**: `per_slot > shared > zero`. That's the real trade-off in the 4D design space — **more placeholder capacity = faster convergence + better quality, but more params**. Production picks based on "model scale × data volume".

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **MaskGIT (Chang et al. 2022)**: image generation 用 mask token,iterative decode / Image generation with mask tokens, iterative decode.
- **Anticipation Music Transformer (Thickstun et al. 2023)**: 音乐生成里用零位置占位 / Music generation uses zero placeholders.
- **AudioLM / Soundstream (Borsos et al. 2022)**: speech 生成同样的 placeholder pattern / Speech generation uses the same placeholder pattern.
- **GPT 系的 cloze prompting**: 例如 `The capital of France is ____`,虽然没改架构,但**思路一致** — placeholder 让 LM 从前文 attend 出答案 / GPT-style cloze prompting like `The capital of France is ____` — no architectural change, but **same idea**: placeholder lets the LM attend to context to fill in the answer.
- **π₀, GR00T, SmolVLA, RoboFlamingo**: 都是 OFT 同思想的不同实例 / All instantiate the same idea with different choices in the 4D space.

## 注意事项 / Caveats / when it breaks

- **placeholder 形式必须配模型规模** / **Placeholder choice must match model scale**: 7B 用 zero ✓,100M 用 learned per-slot;反过来都浪费/难收敛 / 7B uses zero ✓, 100M uses learned per-slot; the reverse is wasteful / hard to converge.
- **causal vs cross 选错 = 信息流方向反了** / **Causal vs cross wrong = info flow direction reversed**: causal 要 query 在 suffix;cross 不限位置但要分开 condition/query module / Causal requires queries to sit at the suffix; cross has no position constraint but needs separate condition/query modules.
- **RoPE 要求 placeholder 不是零才能第一层就区分** / **RoPE requires non-zero placeholder for layer-1 differentiation**: 如果你坚持用 RoPE + zero placeholder,前 2 层位置是同质的(OFT 接受了这个代价) / RoPE + zero placeholder = first 2 layers are positionally homogeneous (OFT accepts this cost).
- **混用多种 placeholder 策略很危险** / **Mixing placeholder strategies is dangerous**: 比如一半 query 是 learned 一半是 zero,attention 会被 learned query 主导,zero 那部分实际不学 / E.g. half learned + half zero — attention gets dominated by the learned queries, the zero half effectively doesn't train.

## 延伸阅读 / Further reading

- BERT paper (Devlin et al., 2018) — the origin of mask-and-predict
- NAT paper (Gu et al., 2017) — first parallel decode with placeholders
- DETR paper (Carion et al., 2020) — object queries pattern
- Perceiver IO paper (Jaegle et al., 2021) — learned latents as universal interface
- MAE paper (He et al., 2021) — the most direct spiritual ancestor of OFT
- BLIP-2 paper (Li et al., 2023) — Q-Former for multimodal
- ACT paper (Zhao et al., 2023) — OFT's direct robotics predecessor
- OpenVLA-OFT paper (2025) — the latest variant
- Today's companion note on the L1 head + zero embedding implementation
