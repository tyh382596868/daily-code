---
date: 2026-06-08
topic: vla
source: vla
repo: openvla/openvla
file: prismatic/models/vlms/prismatic.py
permalink: https://github.com/openvla/openvla/blob/main/prismatic/models/vlms/prismatic.py#L367-L420
difficulty: intermediate
read_time: ~12 min
tags: [code-of-the-day, vla, openvla, multimodal-fusion, causal-mask, prefix-lm, nano-vla]
build_role: vlm-backbone-wiring (deep-dive variant) — how vision/language/action actually fuse inside the LLM, with no separate fusion module
---

# OpenVLA 没有"融合模块":vision 钉前缀,action 钉后缀,32 层 causal attention 自己融 / OpenVLA has no "fusion module": vision pinned at prefix, action pinned at suffix, 32 layers of causal attention do the rest

> **一句话 / In one line**: OpenVLA 把视觉 patch、语言 token、动作 token **拼成一条序列丢给 LLaMA**,所有跨模态交互只靠 causal mask 的几何形状 + 32 层的层叠完成 — 没有 cross-attention,没有 fusion block,没有 modality gating。 / OpenVLA cats vision patches, language tokens, and action tokens into one sequence and hands the whole thing to LLaMA. All cross-modal interaction is implemented by the geometry of the causal mask + the 32-layer stack — no cross-attention, no fusion block, no modality gating.

## 为什么重要 / Why this matters

刚接触 VLA 时大多数人都会问:"视觉和语言怎么融合的?是不是有专门的 cross-attention 模块?" — 答案出人意料地朴素:**没有**。OpenVLA 的"融合"是一个**几何现象**,不是一个模块。把 vision 钉在序列的最前面,把 action 钉在最后,然后 causal mask 的下三角形状自动决定了谁能 attend 谁:vision 只能 attend vision,language 看 vision + 之前的 language,action 看 vision + 全部 language + 之前的 action。这个看似简单的安排,堆叠 32 层之后就形成了"vision 信息被 language 和 action 反复萃取"的深度融合。理解了这一点,你就理解了所有"prefix-LM 风格"VLA 的底层机制 — RT-2、PaLI-X、Qwen-VL、SmolVLA 全都是这个套路。

The first thing anyone new to VLA asks is "how do vision and language actually fuse? Is there a cross-attention module?" — the answer is surprisingly plain: there isn't one. OpenVLA's "fusion" is a **geometric phenomenon**, not a module. Pin vision at the start of the sequence, pin action at the end, and the lower-triangular causal mask automatically settles who attends to whom: vision only attends vision, language sees vision + prior language, action sees vision + all of language + prior action. Stack that 32 times and you get "vision information being repeatedly extracted into the language and action representations". Once you see this, you've seen the mechanism behind every prefix-LM-style VLA — RT-2, PaLI-X, Qwen-VL, SmolVLA all use the exact same trick.

## 代码 / The code

`openvla/openvla` — [`prismatic/models/vlms/prismatic.py`](https://github.com/openvla/openvla/blob/main/prismatic/models/vlms/prismatic.py#L367-L420)

```python
# === Run Visual Feature Extraction ===
with torch.set_grad_enabled(self.vision_backbone_requires_grad):
    patch_features = self.vision_backbone(pixel_values[multimodal_indices])
# patch_features: (B, 514, 2304)   ← 双塔 SigLIP + DINOv2 fused along channel dim

# Projection Logic :: [bsz, num_patches, llm_embed_dim]
projected_patch_embeddings = self.projector(patch_features)
# projected_patch_embeddings: (B, 514, 4096)   ← projected to LLaMA hidden dim

# Get Input Embeddings from LLM Backbone :: [bsz, input_seq_len, llm_embed_dim]
input_embeddings = self.llm_backbone.embed_input_ids(input_ids)
# input_embeddings: (B, L, 4096)   ← text + action tokens via LLaMA embedding table

# === Build Multimodal Embeddings (this single torch.cat IS the "fusion") ===
multimodal_embeddings = torch.cat(
    [
        input_embeddings[multimodal_indices, :1, :],   # position 0:      BOS
        projected_patch_embeddings,                     # positions 1..514: vision patches  ← pinned to prefix
        input_embeddings[multimodal_indices, 1:, :],   # positions 515..: text + action  ← action naturally at suffix
    ],
    dim=1,
)
# multimodal_embeddings: (B, 1 + 514 + (L-1), 4096) ≈ (B, 540, 4096)
```

然后 `multimodal_embeddings` 被喂进 LLaMA-2 7B,在 HF transformers 的 `LlamaModel.forward` 里走标准的 32 层循环([`transformers/models/llama/modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L375-L420)):

```python
causal_mask = create_causal_mask(
    config=self.config,
    inputs_embeds=inputs_embeds,
    attention_mask=attention_mask,
    past_key_values=past_key_values,
    position_ids=position_ids,
)

hidden_states = inputs_embeds
position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

for decoder_layer in self.layers[: self.config.num_hidden_layers]:   # ← 32 次
    hidden_states = decoder_layer(
        hidden_states,
        attention_mask=causal_mask,        # 复用同一个下三角 mask
        position_embeddings=position_embeddings,
        ...,
    )
```

每一层 attention 内部就是 Q @ K^T + mask + softmax + @ V,**mask 这一加,vision 就只能看自己,language 和 action 单向"吸"vision**。

## 逐行讲解 / What's happening

1. **`patch_features = self.vision_backbone(pixel_values)`**:
   - 中文: 双塔 SigLIP + DINOv2 并行跑同一张图,输出按 channel 维 concat — 一塔 SigLIP 1152,一塔 DINOv2 1152,合成 2304 维,每塔 257 个 patch(256 + 1 CLS)→ `(B, 514, 2304)`。
   - English: SigLIP and DINOv2 run in parallel on the same image, their outputs concatenated along the channel dim — 1152 + 1152 = 2304 features, 257 patches each (256 + CLS) → `(B, 514, 2304)`.

2. **`self.projector(patch_features)`**:
   - 中文: 一个 MLP 把 2304 维投影到 LLaMA 的 4096 hidden dim。**这是 vision 和 text 唯一一处"维度对齐"**,projector 训完之后,vision token 在数值上就和 text token 同型了,可以直接进同一个 LM。
   - English: an MLP projects 2304 down to LLaMA's 4096 hidden dim. **This is the only place where vision and text get dimension-aligned**; once the projector is trained, vision tokens are numerically interchangeable with text tokens and can enter the same LM.

3. **`input_embeddings = self.llm_backbone.embed_input_ids(input_ids)`**:
   - 中文: 用 LLaMA 自带的 embedding 表把 `[BOS, "USER:", ..., T1, T2, ..., T7, EOS]` 这串 id 查表成 4096 维向量。**action token 也走这张表** — 因为它们就是借了 LLaMA vocab 末尾 256 个 slot,没有专门的 action embedding 矩阵。
   - English: LLaMA's own embedding table looks up `[BOS, "USER:", ..., T1, T2, ..., T7, EOS]` into 4096-dim vectors. **Action tokens go through the same table** — they reuse the last 256 slots of LLaMA's vocab, no separate action embedding matrix.

4. **`torch.cat([BOS, vision, text+action], dim=1)`**:
   - 中文: 这一行是整个 OpenVLA 的"融合点"。`[:, :1, :]` 切出 BOS,中间一整块塞进 514 个 vision token,最后 `[:, 1:, :]` 是 text + action + EOS。**vision 的位置 1..514 是物理上钉死的,谁先入 prefix 决定了它被谁看到**。
   - English: this single line is OpenVLA's "fusion point". `[:, :1, :]` peels off BOS, 514 vision tokens go in the middle, `[:, 1:, :]` is text + action + EOS. **Vision physically sits at positions 1..514 — whoever sits earlier in the sequence determines who sees whom**.

5. **`for decoder_layer in self.layers`**:
   - 中文: 32 层循环。每一层都做一次 `attn(causal_mask) + FFN + residual`。**同一个 causal_mask 被复用 32 次,但每一层 vision/language/action 的 hidden state 都被精炼了一轮**,所以第 32 层 action token 的表示是经过 32 轮反复融合后的产物。
   - English: 32 iterations. Each layer does `attn(causal_mask) + FFN + residual`. **The same causal mask is reused 32 times, but every layer further refines the vision/language/action hidden states**, so the action token representation at layer 32 has been through 32 rounds of fusion.

6. **causal mask 的几何含义 / The geometry of the causal mask**:
   - 中文: mask 不知道什么叫 vision/language/action,它只用位置:位置 i 能看到位置 j ≤ i。**因为 vision 物理上坐在 1..514,它就成了"prefix"; action 坐在 last 7,它就成了"suffix"**。模态划分藏在坐标里,不藏在 mask 形状里。
   - English: the mask knows nothing about modalities; it just uses positions — position i can see position j ≤ i. **Because vision physically occupies positions 1..514, it becomes the "prefix"; action occupies the last 7 and becomes the "suffix"**. Modality boundaries live in coordinates, not in mask shape.

## 类比 / The analogy

想象一场会议。Vision 是会议桌上摊开的资料(514 页),language 是主持人提的问题,action 是与会者要回答的 7 个选项。会议规则:**只能看会议进程中"已经出现过"的内容**(因果约束)。

- 资料(vision)最先放上桌 → 后面所有人都能翻它,但资料自己不会因为有人翻就改变。
- 主持人(language)提问时,可以引用资料 + 之前已经问过的问题。
- 与会者(action)给答案时,可以引用资料 + 所有问题 + 之前已经给出的部分答案。

32 层 = 这场会议被重新开了 32 遍,**每一轮主持人和与会者都更熟悉资料**,但资料本身从头到尾是同一份。这就是"vision 被单向吸 32 次"的会议室版本。

Picture a meeting. Vision is the briefing materials laid out on the table (514 pages), language is the chair's questions, and action is the participants' 7-option answer. Meeting rule: **you can only reference content that has already appeared earlier in the meeting** (the causal constraint).

- Materials (vision) go on the table first → everyone later can flip through them, but the materials themselves never change because someone read them.
- The chair (language) asks questions, citing materials + earlier questions.
- Participants (action) give answers, citing materials + every question + earlier parts of the answer.

32 layers = the meeting is replayed 32 times, with the chair and participants **getting more familiar with the materials each round**, but the materials themselves stay the same throughout. That's the meeting-room version of "vision single-direction-absorbed 32 times".

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-VLA

这是 nanoVLA 课程里 `vlm-backbone-wiring` 槽位的**深度变体**(之前 5/29 用 SmolVLA 的 expert-cross-attn 接法讲过)。今天这个变体讲的是更基础的问题:**接线之前,vision/language/action 怎么进同一个 LM**。所有 prefix-LM 风格的 VLA 都要解决这个问题。

在你自己搭的 nanoVLA 里,这个模块的接口非常薄:

```
vision encoder → (B, 256, D_v)          # ← 上游(已在 vision-encoder 课程学过)
modality projector → (B, 256, D_lm)    # ← 上游(已在 modality-projector 课程学过)
language tokens (B, L_text, D_lm)
action tokens (B, K, D_lm)              # ← K = action token 数(原版 OpenVLA 是 7,OFT 是 56)

multimodal_embeddings = torch.cat([
    BOS,
    projected_vision_tokens,            # PREFIX
    language_tokens,
    action_tokens,                      # SUFFIX
], dim=1)

→ LLaMA / Qwen / SmolLM (causal)
```

下游消费者是 action head(`action-head-continuous` 或 lm_head 上的 next-token-prediction)。**关键设计选择**:
1. **modality 顺序**:vision 必须在 prefix(让所有人都能看到它),action 必须在 suffix(让它能看到所有 condition)。**调换顺序会让 vision 反向"吸"language,但 language 看不到 vision — 通常没人这么干**。
2. **vision 编码方式**:用 1 个 vision tower(SigLIP)还是 2 个(SigLIP + DINOv2)还是 N 个(多相机融合 GR00T 风格)?都不会改变 fusion 机制,只会改变 prefix 的长度。
3. **language tokenize 方式**:复用 LLaMA 的 BPE 即可,无需扩词表 — **action token 借用 vocab 末尾槽位的 trick(见 5/10 笔记)避免了扩 embedding**。
4. **要不要加 proprio token**:原版 OpenVLA 不加;OFT 加 1 个 token 在 vision prefix 末尾(也是 prefix 一部分)。

production 实现需要补:flash attention 2 / SDPA dispatch、KV cache(prefix 的 K/V 算一次后缓存,decode 时复用)、gradient checkpointing(32 层 attention 全 backprop 显存爆,要 chunk)、FSDP(7B 参数单卡放不下)。

This is the **deep-dive variant** of the `vlm-backbone-wiring` slot in the nanoVLA curriculum (the SmolVLA expert + cross-attention interpretation was covered 2026-05-29). Today's variant addresses a more fundamental question: **before any "wiring", how do vision/language/action enter the same LM at all?** Every prefix-LM-style VLA solves this exact problem.

In your nanoVLA, the contract is thin:

```
vision encoder → (B, 256, D_v)          ← upstream (vision-encoder lesson)
modality projector → (B, 256, D_lm)    ← upstream (modality-projector lesson)
language tokens (B, L_text, D_lm)
action tokens (B, K, D_lm)              ← K = number of action tokens (7 in OpenVLA, 56 in OFT)

multimodal_embeddings = torch.cat([
    BOS,
    projected_vision_tokens,            # PREFIX
    language_tokens,
    action_tokens,                      # SUFFIX
], dim=1)

→ LLaMA / Qwen / SmolLM (causal)
```

The downstream consumer is either an action head (`action-head-continuous` slot) or `lm_head` doing next-token-prediction. **Critical design choices**:

1. **Modality ordering**: vision must be prefix (visible to everyone), action must be suffix (sees everything). Swap them and vision absorbs language but language can't see vision — nobody does that.
2. **Number of vision towers**: 1 (SigLIP), 2 (SigLIP + DINOv2), or N (GR00T-style multi-camera)? Doesn't change the fusion mechanism, just lengthens the prefix.
3. **Language tokenization**: reuse LLaMA BPE, no vocab expansion. The action-token-borrows-last-256-slots trick (see 5/10 note) avoids any embedding-matrix surgery.
4. **Proprio token?** OpenVLA original: no. OpenVLA-OFT: 1 token appended to vision prefix.

Production needs: flash attention 2 / SDPA dispatch, KV cache (cache the prefix's K/V once, reuse during decode), gradient checkpointing (32 layers blow up memory; chunk them), FSDP (7B params don't fit on one GPU).

## 自己跑一遍 / Try it yourself

```python
# try.py — minimal "prefix-LM VLA" forward in 45 lines, no openvla install needed
import torch, torch.nn as nn, torch.nn.functional as F

D = 64           # tiny hidden dim for demo
N_V = 16         # vision tokens
N_L = 8          # language tokens
N_A = 4          # action tokens
L = N_V + N_L + N_A

class TinyVLA(nn.Module):
    def __init__(self, n_layers=4, n_heads=4):
        super().__init__()
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(D, n_heads, dim_feedforward=4*D,
                                       batch_first=True, dropout=0.0)
            for _ in range(n_layers)
        )

    def forward(self, vision_tokens, lang_tokens, action_tokens):
        # 1. 拼接:vision prefix → language → action suffix
        seq = torch.cat([vision_tokens, lang_tokens, action_tokens], dim=1)  # (B, L, D)
        # 2. 建因果 mask(下三角)
        causal_mask = torch.triu(torch.full((L, L), float("-inf")), diagonal=1)
        # 3. 32-style 层叠(这里只跑 4 层)
        x = seq
        for blk in self.blocks:
            x = blk(x, src_mask=causal_mask)
        return x

vision = torch.randn(1, N_V, D)
lang   = torch.randn(1, N_L, D)
action = torch.randn(1, N_A, D)

model = TinyVLA().eval()
with torch.no_grad():
    out = model(vision, lang, action)

# 验证 "vision 看不到 language" — 改 language 的内容,vision 块的 hidden state 应该不变
lang_changed = torch.randn(1, N_L, D)
with torch.no_grad():
    out_changed = model(vision, lang_changed, action)

print(f"vision-block hidden state change (should be 0): {(out[:, :N_V] - out_changed[:, :N_V]).abs().max().item():.6f}")
print(f"action-block hidden state change (should be > 0): {(out[:, -N_A:] - out_changed[:, -N_A:]).abs().max().item():.6f}")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
vision-block hidden state change (should be 0): 0.000000
action-block hidden state change (should be > 0): 1.23  # 具体数字不一样,关键是 > 0
```

**注意 vision 改了 language 后 hidden state 严格不变,而 action 大幅变化**。这就是 causal mask 把"vision 钉前缀、action 钉后缀"在数值上证实给你看 — vision 在 prefix 上根本看不到后面任何东西,只能被吸。

**Notice vision's hidden state is strictly unchanged when language changes, while action's hidden state moves a lot**. That's the numerical proof that the causal mask pins vision at the prefix — vision physically cannot see anything later in the sequence; it can only be absorbed by downstream tokens.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **RT-2 (Google)**: 同样的 prefix-LM 风格,vision-language tokens 在前,action token 在后,用 PaLI-X 当 backbone / Same prefix-LM style, vision-language prefix + action suffix, with PaLI-X as the backbone.
- **PaLI-X / PaLI-3 / Qwen-VL**: 全是 prefix-LM 多模态 LLM,只是任务从 captioning 换成 VQA、segmentation,fusion 机制完全一样 / Pure prefix-LM multimodal LLMs; only the task swaps (captioning → VQA → segmentation), the fusion mechanism is identical.
- **SmolVLA (lerobot)**: 加了一个 slim action expert 走 cross-attention 与 VLM 通信 (5/29 笔记),是 prefix-LM 的"对偶":expert 不在主序列里,但仍然单向吸 VLM 信息 / Adds a slim action expert that cross-attends back into the VLM (5/29 note); the dual of prefix-LM — the expert isn't in the main sequence but still absorbs VLM info unidirectionally.
- **GPT-4V, Gemini, Claude (vision)**: 闭源但论文/猜测都指向同一种 prefix-LM 路径 — vision tokens 放前 / Closed-source but papers and inference suggest the same prefix-LM route — vision tokens go first.

## 注意事项 / Caveats / when it breaks

- **prefix 太长会让 prefill 阶段昂贵** / **Long prefix makes prefill expensive**: 514 个 vision token 走一次 32-layer attention 是平方复杂度,所以 OpenVLA 第一次 forward 慢。后续 decode 用 KV cache 摊销,但第一步 prefill 占总时间 80%+ / 514 vision tokens through 32 layers of quadratic attention is what makes OpenVLA's first forward slow. KV cache amortizes later decode steps, but prefill is still 80%+ of total inference time.
- **vision 永远不知道 language 是什么** / **Vision never sees language**: 这是 prefix-LM 的固有局限,如果想让 vision 也根据指令调整(比如"看左上角"),要用 FiLM 或 cross-attention 反向注入 — OpenVLA-OFT 的 `use_film=True` 就是为此 / This is the intrinsic limit of prefix-LM. If you want vision to adapt to the instruction (e.g. "look at the top-left"), you need FiLM or reverse cross-attention — that's what OpenVLA-OFT's `use_film=True` flag is for.
- **多模态混排不能乱顺序** / **Modality ordering is non-trivial**: 把 action 放前面 = LM 在"看到动作"之前就要预测它,根本不可能;把 vision 放后面 = vision 不被 language 看到,task 完全脱节 / Put action first = LM has to predict action before seeing anything, impossible. Put vision last = language never sees vision, task fully decouples.
- **causal 是单向的,bidirectional 要重训** / **Causal is one-way; bidirectional needs retraining**: LLaMA-2 是 causal pretrained 的,中途换 bidirectional mask 模型会乱掉。OFT 通过把 action embedding 置零让 LM "无视" action 之间的依赖,绕过了重训 / LLaMA-2 is causal-pretrained. Switching to bidirectional mid-finetune breaks the model. OFT bypasses this by zeroing action embeddings so the LM "ignores" action-to-action dependencies, no retraining needed.

## 延伸阅读 / Further reading

- OpenVLA paper: https://openvla.github.io — § 3 explains the design decisions
- RT-2 paper (Brohan et al., 2023) — the predecessor that introduced "action as text" tokens
- Prismatic VLM paper (Karamcheti et al., 2024) — the VLM backbone OpenVLA reuses (SigLIP+DINOv2 fused tower + LLaMA-2 7B)
- Today's companion note on the next-token-prediction training target — explains how the 7 action positions are supervised
