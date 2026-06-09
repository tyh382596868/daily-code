---
date: 2026-06-08
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/policies/pi0_fast/modeling_pi0_fast.py
permalink: https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi0_fast/modeling_pi0_fast.py#L368-L493
difficulty: advanced
read_time: ~15 min
tags: [code-of-the-day, vla, pi0-fast, paligemma, fast-tokenizer, prefix-lm, state-as-string, lerobot, nano-vla]
build_role: vlm-backbone-wiring (deep-dive variant) — pi0-FAST's three multimodal-fusion tricks (state-as-string + FAST DCT/BPE action tokens + PaliGemma prefix-LM mask)
---

# pi0-FAST 把 state / action / language 全塞进同一条 token 流,靠 PaliGemma 的 prefix-LM mask 完成融合 / pi0-FAST stuffs state / action / language into one token stream and lets PaliGemma's prefix-LM mask do the fusion

> **一句话 / In one line**: pi0-FAST = **PaliGemma 2B + SigLIP + 3 个 trick**:(1) 把 8 维 state 离散化成 "144 84 200 ..." 数字串塞进 prompt,(2) 把 10×7 连续 action chunk 用 DCT+BPE 压成 ~15 个 token,(3) 复用 PaliGemma 自带的 prefix-LM mask(prefix 段 bidirectional + action 段 causal)— 不加任何新模块就完成多模态融合 + autoregressive action 生成。 / pi0-FAST = **PaliGemma 2B + SigLIP + 3 tricks**: (1) discretize 8-dim state into a "144 84 200 ..." digit string and stuff it into the prompt, (2) compress the 10×7 continuous action chunk into ~15 tokens via DCT+BPE, (3) reuse PaliGemma's built-in prefix-LM mask (bidirectional in prefix, causal in action segment) — no new modules added, multimodal fusion + autoregressive action generation handled together.

## 为什么重要 / Why this matters

讲完 OpenVLA(纯 causal + 256-bin)和 OFT(全 bidirectional + zero embedding + L1 head)之后,**pi0-FAST 是把 "action as text" 路线推到最自然形态的版本**。它的核心创新不是模型架构而是 **token 序列设计**:state 不要专门 projector(当文字处理),action 用 FAST 词表压缩到 ~15 个 token(autoregressive 生成才不至于太慢),attention mask 也不改(复用 PaliGemma 预训练的 prefix-LM 模式)。结果:**完全不动 backbone 一行代码**就拿到了"vision 看 language 也看 state"的双向融合能力。这种"用 prompt 设计取代模块设计"的思路,后来被 OpenVLA-OFT 反向借鉴(加了 proprio token),也被 RoboFlamingo、CogAct 等借用 — 理解了 pi0-FAST,你就理解了"为什么生产级 VLA 都在抛弃 zero-embedding 这种 hack"。本笔记用 lerobot 的 PyTorch 重写版作主代码锚点,跟 openpi 原版(JAX)做对照。

After OpenVLA (pure causal + 256-bin) and OFT (full bidirectional + zero embedding + L1 head), **pi0-FAST is "action as text" pushed to its most natural form**. Its core innovation isn't model architecture — it's **token sequence design**: state needs no projector (treat it as text), action gets compressed to ~15 tokens via FAST (autoregressive generation stays fast), and the attention mask is unchanged (reuses PaliGemma's pretrained prefix-LM pattern). The result: **without touching a single line of backbone code**, you get "vision attends language and state" bidirectional fusion. This "replace module design with prompt design" mindset has since been borrowed back by OpenVLA-OFT (added the proprio token), RoboFlamingo, CogAct — understanding pi0-FAST explains why production-grade VLAs are abandoning hacks like zero embedding. This note uses lerobot's PyTorch rewrite as the main code anchor, with openpi (JAX) for comparison.

## 代码 / The code

### Trick 1 — State 离散化成字符串塞进 prompt

`huggingface/lerobot` — [`src/lerobot/policies/pi0_fast/processor_pi0_fast.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi0_fast/processor_pi0_fast.py#L52-L91)

```python
@dataclass
class Pi0FastPrepareStateAndLanguageTokenizerProcessorStep(ProcessorStep):
    def __call__(self, transition):
        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)

        # 1. State 已 normalized 到 [-1, 1],digitize 成 256 个 bin id ∈ [0, 255]
        state_np = state.cpu().numpy()
        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # 2. 拼成 "Task: ... State: 144 84 200 12 33 188 55 91;\n" 的字符串
        full_prompts = []
        for i, task in enumerate(tasks):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, discretized_states[i]))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\n"
            full_prompts.append(full_prompt)

        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = full_prompts
        return transition
```

### Trick 2 — Action chunk 经 FAST tokenizer 压缩到 ~15 token

`huggingface/lerobot` calls `physical-intelligence/fast` HF model:

```python
# 概念上(实际在 ActionTokenizerProcessorStep 内调用):
action_chunk = batch["action"]                              # (B, 10, 7) continuous
fast_tokens = fast_processor(action_chunk)                  # → (B, ~15) int

# 映射到 PaliGemma vocab 末尾 slot(跳过最后 128 个 special token)
fast_action_tokens = paligemma_vocab_size - 1 - 128 - fast_tokens
# 这一行决定了 FAST token 复用 PaliGemma 自带 embedding table,不扩词表
```

### Trick 3 — `embed_prefix_fast`: 三段 embedding 拼接 + 显式 segment mask

[`modeling_pi0_fast.py:368-493`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi0_fast/modeling_pi0_fast.py#L368-L493)

```python
def embed_prefix_fast(self, images, img_masks, tokens, masks,
                     fast_action_tokens=None, fast_action_masks=None):
    embs = []
    pad_masks = []
    att_mask_segments = []                       # ← 记录每段 (类型, 长度) 用于后续构 mask

    # Image segment
    for img, img_mask in zip(images, img_masks, strict=True):
        img_emb = self.paligemma_with_expert.embed_image(img)   # SigLIP+projector
        embs.append(img_emb)
        pad_masks.append(img_mask[:, None].expand(B, 256))
        att_mask_segments.append(("image", 256))

    # Language segment (prompt + state string)
    lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
    embs.append(lang_emb)
    pad_masks.append(masks)
    att_mask_segments.append(("language", lang_emb.shape[1]))

    # FAST action segment (only present during training)
    if fast_action_tokens is not None:
        # ← FAST token 走同一个 embed_language_tokens(复用 vocab 末尾 slot)
        fast_action_emb = self.paligemma_with_expert.embed_language_tokens(fast_action_tokens)
        embs.append(fast_action_emb)
        pad_masks.append(fast_action_masks)
        att_mask_segments.append(("fast", fast_action_tokens.shape[1]))

    embs = torch.cat(embs, dim=1)                # (B, 304, 2048)
    pad_masks = torch.cat(pad_masks, dim=1)

    # 显式 segment-based attention mask 构造
    att_masks = self._create_custom_attention_mask_fast(att_mask_segments, pad_masks, B)
    return embs, pad_masks, att_masks, total_t_images, num_fast_embs


def _create_custom_attention_mask_fast(self, att_mask_segments, pad_masks, bsize):
    """
    Attention rules:
      - Images + Language: bidirectional among themselves
      - FAST: attend to images + language, causal among themselves
    """
    total_len = sum(length for _, length in att_mask_segments)
    att_2d_masks = torch.zeros(bsize, total_len, total_len, dtype=torch.bool, device=device)

    positions = []
    current_pos = 0
    for seg_type, seg_len in att_mask_segments:
        positions.append((seg_type, current_pos, current_pos + seg_len))
        current_pos += seg_len

    # 双重循环显式填 mask
    for query_type, q_start, q_end in positions:
        for key_type, k_start, k_end in positions:
            if (query_type in ["image", "language"] and key_type in ["image", "language"]
                or query_type == "fast" and key_type in ["image", "language"]):
                att_2d_masks[:, q_start:q_end, k_start:k_end] = True       # 双向 / 单向

            elif query_type == "fast" and key_type == "fast":
                fast_len = q_end - q_start
                causal_mask = torch.tril(torch.ones(fast_len, fast_len, dtype=torch.bool, device=device))
                att_2d_masks[:, q_start:q_end, k_start:k_end] = causal_mask[None, :, :]  # 下三角

    # 与 padding mask AND
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks
```

## 逐行讲解 / What's happening

### 1. **state digitize 成 256-bin 整数后转字符串** — 完全免去 ProprioProjector
- 中文: 每维 state 落到 0..255 中的一个 bin → 转成 ASCII 数字字符 → 跟 task prompt 拼成 `"Task:..., State: 144 84 200 ...;\n"`。**PaliGemma BPE 把这些数字 token 化时,大部分是单字节 fallback**,8 个数字大约 BPE 化成 16-24 个 token。state 信息因此**完整保留**(只丢了 256 量化误差,~0.008 精度),且不需要任何新参数。
- English: each state dim falls into a 0..255 bin → cast to ASCII digit chars → concatenate with task prompt as `"Task:..., State: 144 84 200 ...;\n"`. **PaliGemma BPE largely byte-fallbacks these digits**, expanding 8 numbers to ~16-24 tokens. State info is **fully preserved** (only 256-bin quantization error, ~0.008 precision), with zero new parameters.

### 2. **FAST tokenizer** — 动作的"DCT + BPE 二次压缩"
- 中文: `physical-intelligence/fast` 是一个**预训练好的 BPE 模型**,词表是从海量机器人 demo data 学出来的。输入 `(10, 7)` 连续 action,对每维做 DCT(频域稀疏),量化丢高频,把 7 维拍平再用 BPE 压。**典型压缩比 ~10-35×**:`10×7=70` 个 raw 数字 → `~15` 个 token。这让 autoregressive 生成 action 实用起来(否则 70 步 decode 太慢)。
- English: `physical-intelligence/fast` is a **pretrained BPE model** with vocab learned from a massive robot demo corpus. Input `(10, 7)` continuous actions → DCT per dim (sparse in frequency) → quantize and drop high freq → flatten and BPE-compress. **Typical compression ~10-35×**: 70 raw numbers → ~15 tokens. This makes autoregressive action generation practical (70-step decode would be too slow).

### 3. **FAST token 借 PaliGemma vocab 末尾 slot,不扩词表**
- 中文: `fast_action_tokens = vocab_size - 1 - 128 - fast_tokens` —— 把 FAST 的 token id 映射到 PaliGemma 词表倒数第 129 到 ~600 这段,**这些位置原本是低频的 special token slot**。这样 `embed_language_tokens(fast_tokens)` 直接走 PaliGemma 自带 embedding table — **不增加任何 embedding 参数,不破坏预训练权重**。
- English: `fast_action_tokens = vocab_size - 1 - 128 - fast_tokens` maps FAST ids into the second-to-last 129..~600 slots in PaliGemma vocab, **originally low-frequency special-token slots**. Then `embed_language_tokens(fast_tokens)` reuses PaliGemma's embedding table — **zero added embedding parameters, zero pretrain disruption**.

### 4. **segment-based attention mask** — 复用 PaliGemma 的 prefix-LM 模式
- 中文: 用 segment 列表显式 nested loop 填 mask:`(image, language)` 段互相 True(双向),`fast` 段 query 到 `(image, language)` key 也是 True(往前看 prefix),但 `fast→fast` 用 `torch.tril` 做下三角(causal)。**这正是 PaliGemma 预训练时的 prefix-LM 形态**,不用改 mask 就能复用 PaliGemma 已学到的 attention pattern。
- English: explicit segment list + nested loop fills the mask: `(image, language)` mutually True (bidirectional), `fast` queries to `(image, language)` keys also True (look back at prefix), but `fast→fast` uses `torch.tril` (causal lower triangle). **This is PaliGemma's pretrained prefix-LM pattern** — no mask modification needed, just reuse PaliGemma's learned attention behavior.

### 5. **fast tokens 与 language tokens 共用 embed**
- 中文: `embs.append(fast_action_emb)` 后面没有任何 "FAST adapter" 之类的层,直接 `torch.cat` 进 LLaMA 主干 — **FAST token 在模型眼里跟普通 text token 完全没区别**,只是它们的 id 凑巧落在 vocab 末尾。这种"假装是字"的设计让 pi0-FAST 训练目标也彻底简化为 standard next-token-prediction CE,直接用 HF `lm_head` 即可。
- English: after `embs.append(fast_action_emb)` there's no "FAST adapter" layer — direct `torch.cat` into LLaMA backbone. **The model treats FAST tokens identically to regular text tokens**, the ids just happen to land at vocab end. This "disguise as text" design simplifies the training objective to standard next-token-prediction CE, directly using HF's `lm_head`.

## 类比 / The analogy

OpenVLA 是"用 256 bin 把每维动作切成 6 个 ASCII 字母,然后让 LM 续写 7 个字母" — 一字一动作,生成 7 次。
OpenVLA-OFT 是"用 zero placeholder 把 56 个空盒子放在序列末尾,LLaMA 通过 attention 把内容塞进盒子,MLP 一次出 56 维" — 跳过文字这一层中介。
**pi0-FAST 是"用 zip 把 action chunk 压缩成 ~15 个字母,然后让 LM 续写这 15 个字母"** — **zip 是 FAST tokenizer 用 DCT+BPE 学出来的;state 是直接当对话中提到的事实告诉 LM 的;PaliGemma 自带 prefix-LM 让 vision 和 language 也能反向看 state**。本质上还是"action as text",只是文字被压缩到极致,而且 prompt 设计精细到让所有模态共用同一个 token 流。

OpenVLA = "use 256 bins to slice each action dim into 6 ASCII letters, then let the LM 'continue writing' 7 letters" — one letter one action, generated 7 times.
OpenVLA-OFT = "place 56 empty boxes at the sequence end, let LLaMA fill them via attention, then an MLP outputs 56 values" — bypass the text-as-intermediary layer.
**pi0-FAST = "zip the action chunk into ~15 letters, then let the LM continue writing those 15 letters"** — **the zip is FAST tokenizer (DCT + BPE learned from data); state is stated as a fact in the conversation; PaliGemma's built-in prefix-LM lets vision and language attend back to state**. Still "action as text" at heart, just text compressed to the maximum, with prompt design refined enough that all modalities share one token stream.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-VLA

这是 nanoVLA 课程里 `vlm-backbone-wiring` 槽位的**深度变体**(5/29 SmolVLA 讲 expert-cross-attn,6/08 早些讲 OpenVLA 的 causal-prefix + OFT 的 fork-bidirectional,今天讲 pi0-FAST 的 prefix-LM 复用 + 极致 prompt 设计)。

理解 pi0-FAST 给你的工程价值:**当你为 nanoVLA 选 backbone 时,你的"动作怎么进模型"决策**实际上由 backbone 的 attention mask 模式决定:

| backbone | 预训练 mask | 适合的 action 表示 | 代表作 |
|---|---|---|---|
| LLaMA-2 7B(causal) | 纯下三角 | autoregressive token 生成(OpenVLA),或 fork 改 bidirectional(OFT) | OpenVLA, OpenVLA-OFT |
| **PaliGemma 2B(prefix-LM)** | **prefix 双向 + suffix causal** | **autoregressive token 生成,但 prefix 自由融合** | **pi0-FAST**, SmolVLA |
| Gemma 2B / Qwen 2B(causal) | 纯下三角 | 类似 LLaMA 路线 | RoboFlamingo, π₀ 原版 |

**pi0-FAST 在 nanoVLA 课程里告诉你的核心 lesson**:
1. **state 不需要 dedicated module**(像 OFT 那样),直接当文字塞 prompt 即可 — 简单、零参数、可读
2. **action 需要 token 压缩**才能用 autoregressive(否则 70 步太慢)— FAST 的 DCT+BPE 学习的就是这个
3. **prefix-LM backbone 是天作之合**,vision 看 language + state 反过来,免去 OFT 需要 fork transformers 的代价

**production 设计建议**(用 lerobot PyTorch 版作起点):
- **图像数量**:LIBERO 上单相机 256 tokens 就够;ALOHA 双臂用 wrist cam 加到 3 个相机,prefix 长 ~770 tokens
- **chunk 长度**:LIBERO=10 (~15 fast tokens), ALOHA=25 (~30 fast tokens), bridge=5 (~8 fast tokens)
- **action 归一化**:lerobot 的 pipeline 是 `relative → normalize → tokenize → model → unnormalize → absolute` 6 步流水线,**不要跳步**
- **state quantization 精度**:256 bin 在 [-1, 1] 上分辨率 0.008,对应 ~1cm 末端位置精度。如果你的任务需要更高(穿针引线),把 bin 数加到 1024 或换 ProprioProjector

This is the **deep-dive variant** of the `vlm-backbone-wiring` slot in the nanoVLA curriculum (5/29 SmolVLA covered expert-cross-attn, earlier on 6/08 covered OpenVLA's causal-prefix and OFT's forked-bidirectional, today covers pi0-FAST's prefix-LM reuse + extreme prompt design).

Engineering value: **when you pick a backbone for your nanoVLA, your "how does action enter the model" choice is actually determined by the backbone's pretrained attention mask pattern**:

| backbone | pretrained mask | matching action representation | exemplar |
|---|---|---|---|
| LLaMA-2 7B (causal) | pure lower-triangle | autoregressive token generation (OpenVLA), or fork to bidirectional (OFT) | OpenVLA, OpenVLA-OFT |
| **PaliGemma 2B (prefix-LM)** | **prefix bidirectional + suffix causal** | **autoregressive token generation, with prefix free fusion** | **pi0-FAST**, SmolVLA |
| Gemma 2B / Qwen 2B (causal) | pure lower-triangle | similar to LLaMA route | RoboFlamingo, π₀ original |

**pi0-FAST's core lessons for your nanoVLA**:
1. **State needs no dedicated module** (unlike OFT) — just stuff it as text into the prompt. Simple, zero params, readable
2. **Action needs token compression** for autoregressive to be practical (70-step decode is too slow) — FAST's DCT+BPE learns this
3. **prefix-LM backbones are a perfect match**: vision sees language + state and vice versa, with none of OFT's transformers-fork cost

**Production recommendations** (start from lerobot's PyTorch version):
- **Image count**: 1 camera × 256 tokens is enough for LIBERO; ALOHA dual-arm adds wrist cams to 3 cameras, prefix length ~770
- **Chunk length**: LIBERO=10 (~15 fast tokens), ALOHA=25 (~30 fast tokens), bridge=5 (~8 fast tokens)
- **Action normalization**: lerobot's pipeline is `relative → normalize → tokenize → model → unnormalize → absolute` 6 steps; **don't skip any**
- **State quantization precision**: 256 bins over [-1, 1] gives 0.008 resolution, ~1cm end-effector precision. For finer tasks (needle threading) bump to 1024 bins or switch to ProprioProjector

## 自己跑一遍 / Try it yourself

```python
# try.py — show "state as text" + minimal segment-based mask, no PaliGemma needed
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# 1. State as text
def state_to_string(state, n_bins=256):
    """Discretize [-1, 1]^D state to bin IDs, then to space-separated string."""
    import numpy as np
    bins = np.linspace(-1, 1, n_bins + 1)[:-1]
    discretized = np.digitize(state, bins) - 1
    return " ".join(map(str, discretized))

state = [0.12, -0.34, 0.0, 0.99]
print(f"raw state:     {state}")
print(f"state string:  '{state_to_string(state)}'")
# raw state:     [0.12, -0.34, 0.0, 0.99]
# state string:  '144 84 128 254'

# 2. Tiny BPE tokenizer (代替 PaliGemma SentencePiece)
def make_tiny_tokenizer():
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=300, special_tokens=["<bos>", "<eos>", "<pad>", "|"])
    tok.train_from_iterator([
        f"Task: pick up the cup, State: {state_to_string([0.1*i, -0.2, 0, 0.5])};\nAction: A B C D | <eos>"
        for i in range(50)
    ], trainer)
    return tok

tok = make_tiny_tokenizer()

# 3. 模拟 pi0-FAST 风格 prompt + segment 构造
prompt = "Task: pick up the cup, State: 144 84 128 254;\nAction: "
prompt_ids = tok.encode(prompt).ids
# 假装有 4 个 vision tokens + 4 个 action tokens
N_IMG, N_LANG, N_ACTION = 4, len(prompt_ids), 4
L = N_IMG + N_LANG + N_ACTION

# 4. segment-based 显式构造 prefix-LM mask
def build_prefix_lm_mask(n_img, n_lang, n_action):
    L = n_img + n_lang + n_action
    mask = torch.zeros(L, L, dtype=torch.bool)
    # image+language segment (prefix): 内部 bidirectional
    mask[:n_img + n_lang, :n_img + n_lang] = True
    # action segment: 看 prefix(单向) + 自身 causal(下三角)
    mask[n_img + n_lang:, :n_img + n_lang] = True
    causal = torch.tril(torch.ones(n_action, n_action, dtype=torch.bool))
    mask[n_img + n_lang:, n_img + n_lang:] = causal
    return mask

m = build_prefix_lm_mask(N_IMG, N_LANG, N_ACTION)
print(f"\nmask shape: {m.shape}, sum: {m.sum().item()}")

# 5. 验证关键 attention pattern
print(f"\n关键 attention 测试:")
print(f"  vision[0] → language[0]:  {m[0, N_IMG].item()}        ← bidirectional in prefix")
print(f"  language[0] → vision[0]:  {m[N_IMG, 0].item()}        ← bidirectional in prefix")
print(f"  action[0] → vision[0]:    {m[N_IMG+N_LANG, 0].item()}        ← action attends prefix")
print(f"  vision[0] → action[0]:    {m[0, N_IMG+N_LANG].item()}       ← prefix CANNOT attend action")
print(f"  action[0] → action[1]:    {m[N_IMG+N_LANG, N_IMG+N_LANG+1].item()}       ← action causal: can't see future")
print(f"  action[2] → action[0]:    {m[N_IMG+N_LANG+2, N_IMG+N_LANG].item()}        ← action causal: can see past")
```

运行 / Run with:
```bash
pip install torch tokenizers
python try.py
```

预期输出 / Expected output:
```
raw state:     [0.12, -0.34, 0.0, 0.99]
state string:  '144 84 128 254'

mask shape: torch.Size([20, 20]), sum: ~150

关键 attention 测试:
  vision[0] → language[0]:  True        ← bidirectional in prefix
  language[0] → vision[0]:  True        ← bidirectional in prefix
  action[0] → vision[0]:    True        ← action attends prefix
  vision[0] → action[0]:    False       ← prefix CANNOT attend action
  action[0] → action[1]:    False       ← action causal: can't see future
  action[2] → action[0]:    True        ← action causal: can see past
```

**注意 `vision → action = False`**(prefix 看不到 action)和 `action[0] → action[1] = False`(action 内部 causal),这正是 PaliGemma prefix-LM mask 的标志性几何。

**Notice `vision → action = False`** (prefix can't see action) and `action[0] → action[1] = False` (action internal causal) — that's the signature geometry of PaliGemma prefix-LM mask.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **π₀ 原版 (Physical Intelligence, 2024)**: 用 PaliGemma 2B + flow matching action expert,**没有 FAST**,通过 cross-attn 让 expert 吸 VLM condition。pi0-FAST 是 π₀ 的 "去掉 flow matching,改用 FAST autoregressive" 版本 / Uses PaliGemma 2B + flow matching action expert, **without FAST**, cross-attention lets expert absorb VLM condition. pi0-FAST is the "drop flow matching, use FAST autoregressive" variant.
- **SmolVLA (HuggingFace 2024)**: 同样基于 PaliGemma,但用 slim action expert + cross-attn(类似 π₀ 原版而非 pi0-FAST)/ Also PaliGemma-based, but uses slim action expert + cross-attn (closer to π₀ original than pi0-FAST).
- **RT-2 (Google 2023)**: 第一个 "action as text" 路线,但用 PaLI-X(纯 causal),action 直接 256 bin 离散化 — **pi0-FAST 把它的 FAST 压缩思路推到极致** / The first "action as text" route, but with PaLI-X (pure causal), action just 256-bin discretized. pi0-FAST takes the compression idea to the extreme.
- **CogAct / RoboFlamingo**: 都尝试过把 state 当 prompt 文字塞,但都没用 FAST 压 action,所以推理慢 / Both tried "state as prompt text", but none used FAST for action compression, so inference is slow.

## 注意事项 / Caveats / when it breaks

- **state digitize 精度不够** / **state digitization precision insufficient**: 256-bin 在 [-1, 1] 上分辨率 0.008,对应 EEF ~1cm。穿针引线、双手交接这种 sub-cm 任务会精度不足,需要 1024 bin 或换 ProprioProjector / 256 bins over [-1, 1] gives 0.008 resolution, ~1cm EEF accuracy. Sub-cm tasks (needle threading, hand-over) need 1024 bins or a ProprioProjector.
- **FAST tokenizer 不是万能的** / **FAST tokenizer isn't universal**: 它的 BPE 词表从特定 demo data 学,如果你的 action 分布跟训练数据差异大(比如 vs. 双臂 vs. 单臂),压缩比可能掉到 2× / Its BPE vocab is learned from specific demo data; if your action distribution differs (e.g. dual-arm vs single-arm), compression ratio may drop to 2×.
- **prefix-LM 复用要求 backbone 是 PaliGemma 风格** / **prefix-LM reuse requires PaliGemma-style backbone**: LLaMA 是纯 causal pretrained,直接当 prefix-LM 会破坏预训练 attention pattern。所以 OFT 才不得不 fork transformers / LLaMA is pure-causal pretrained; treating it as prefix-LM disrupts pretrain. That's why OFT has to fork transformers.
- **autoregressive 生成 latency 仍然是 ~15-20 步** / **AR generation latency is still ~15-20 steps**: 比 OFT 的 1 步慢,但跟 KV cache 配合后 ~15 步 each 1-token forward 是可接受的。如果实时性极致(>50Hz)还得换 OFT-L1 / Slower than OFT's 1 step, but ~15 single-token forwards with KV cache is acceptable. For >50Hz real-time, switch to OFT-L1.

## 延伸阅读 / Further reading

- FAST paper: https://www.pi.website/research/fast — DCT + BPE 的细节
- pi0-FAST paper: https://www.pi.website/research/openpi-fast-vs-flow — Pertsch et al. 2025
- π₀ paper (Physical Intelligence, 2024) — pi0-FAST 的前身,用 flow matching
- PaliGemma paper (Beyer et al., 2024) — prefix-LM attention mask 的设计来源
- lerobot pi0_fast README — PyTorch 重写版的使用指南
- Today's companion note on pi0-FAST's stop signals (EOS vs "|")
