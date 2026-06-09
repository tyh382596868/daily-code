---
date: 2026-06-08
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/policies/pi0_fast/modeling_pi0_fast.py
permalink: https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi0_fast/modeling_pi0_fast.py#L690-L810
difficulty: advanced
read_time: ~12 min
tags: [code-of-the-day, vla, pi0-fast, autoregressive, decode-loop, stop-token, torch-compile, jax-while-loop, nano-vla]
build_role: inference-loop (deep-dive variant) — pi0-FAST's two stop signals (EOS vs "|") and why JAX/PyTorch implementations chose opposite decode strategies
---

# pi0-FAST 怎么知道 action 该停了:训练埋两个 stop signal,JAX 和 PyTorch 各用一个 / How pi0-FAST knows when actions should stop: training plants two stop signals, JAX and PyTorch each pick a different one

> **一句话 / In one line**: pi0-FAST 训练时在 action 序列末尾埋两个停止信号(`"|"` 软标记 + `EOS` 硬终止);**openpi (JAX) 用 EOS 在 decode loop 里 early-stop**(`lax.while_loop` 检测全 batch EOS),**lerobot (PyTorch) 死跑满 `max_decoding_steps` 然后在 detokenize 阶段用 `"|"` 字符串截断** — 两套方案数学等价,选哪个全看 framework 的动态控制流友好度。 / At training time pi0-FAST plants two stop signals at the action sequence end (`"|"` soft marker + `EOS` hard terminator); **openpi (JAX) early-stops in the decode loop using EOS** (`lax.while_loop` checks all-batch EOS), **lerobot (PyTorch) runs to `max_decoding_steps` and truncates at the `"|"` character during detokenize** — mathematically equivalent, the choice depends on framework friendliness to dynamic control flow.

## 为什么重要 / Why this matters

讲完 pi0-FAST 的多模态融合后,有一个非常实际的工程问题被掩盖了:**"|" 字符 + EOS 这套设计是怎么在训练 + 推理两端配合的?**autoregressive 解码必须有可靠的停止机制,否则要么生成无意义的 token 直到 max_steps(浪费),要么在 action 序列中途停止(损失精度)。pi0-FAST 用了一个**双重保险**的设计:`"|"` 作为软停止标记(语义上 "action 结束了"),`EOS` 作为硬停止(整条 prompt 结束)。**两个实现版本(openpi JAX 和 lerobot PyTorch)分别选了不同信号,反映了底层 framework 对动态控制流的支持差异** — 这是 deep learning 工程里一个非常生动的"代码风格被 framework 塑造"的案例。理解这个 case,你就理解了为什么 PyTorch 项目通常推荐"死跑 + 后处理截断"而不是"loop 内 early-stop"。

After pi0-FAST's multimodal fusion, a very practical engineering question gets buried: **how do `"|"` and `EOS` cooperate across training + inference?** Autoregressive decoding needs a reliable stop mechanism — otherwise you either generate gibberish until max_steps (wasteful) or stop mid-sequence (lose precision). pi0-FAST uses a **dual-safety** design: `"|"` as a soft stop marker (semantic "action ended"), `EOS` as hard terminator (whole prompt ended). **The two implementations (openpi JAX and lerobot PyTorch) pick different signals, reflecting framework differences in dynamic-control-flow support** — a vivid example of "code style shaped by framework". Understanding this case explains why PyTorch projects often prefer "run-to-end + post-process truncate" over "loop-internal early-stop".

## 代码 / The code

### 训练时埋下双停止信号

`Physical-Intelligence/openpi` — [`src/openpi/models/tokenizer.py:82-87`](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/tokenizer.py#L82-L87)

```python
postfix_tokens = (
    self._paligemma_tokenizer.encode("Action: ")
    + action_tokens_in_pg.tolist()
    + self._paligemma_tokenizer.encode("|", add_eos=True)   # ← 这一行藏着两个 stop signal
)
# encode("|", add_eos=True) 返回 [id("|"), EOS_id],所以训练时 sequence 末尾长这样:
# ... T15  "|"  EOS  [padding]
#         ↑    ↑
#         软停止 硬停止 (PALIGEMMA_EOS_TOKEN = 1)
```

### openpi (JAX) — 用 EOS 在 decode loop 里 early-stop

[`src/openpi/models/pi0_fast.py:286-311`](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0_fast.py#L286-L311)

```python
PALIGEMMA_EOS_TOKEN = 1   # 文件顶部定义

def step(carry):
    ...
    token = jnp.argmax(last_logit, axis=-1)
    output_tokens = put_along_last_axis(output_tokens, ..., token)

    # === early-stop 检查:每步 token 是否 == EOS ===
    has_eos = jnp.any(token == PALIGEMMA_EOS_TOKEN, axis=-1)
    all_eos = jnp.all(has_eos)
    ...
    return rng, last_logit, output_tokens, kv_cache, all_eos, step + 1

def cond(carry):
    _, _, _, _, all_eos, step = carry
    # 两个退出条件:全 batch EOS,或达到 max_decoding_steps
    return (~all_eos) & (step < max_decoding_steps)

# JAX 的关键魔法:lax.while_loop 在 JIT 下支持数据依赖的动态退出
_, _, output_tokens, _, _, _ = jax.lax.while_loop(
    cond, step, (rng, last_logit, output_tokens, kv_cache, False, 0)
)
return output_tokens
```

### lerobot (PyTorch) — 死跑满,然后在 detokenize 阶段用 "|" 截断

[`src/lerobot/policies/pi0_fast/modeling_pi0_fast.py:768-810`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi0_fast/modeling_pi0_fast.py#L768-L810)

```python
def sample_actions_fast_kv_cache(self, images, img_masks, tokens, masks,
                                  max_decoding_steps=None, temperature=0.0):
    ...
    generated_action_tokens = torch.zeros((B, max_decoding_steps), dtype=torch.long, ...)
    generated_action_tokens[:, 0] = first_token

    # ⚠️ 注意:整个 loop 没有 early-stop break!
    for t in range(1, max_decoding_steps):
        next_token_emb = self.paligemma_with_expert.embed_language_tokens(next_token)
        ...
        # forward 1 token using KV cache
        (step_out, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=step_att_mask,
            position_ids=current_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[next_token_emb, None],
            use_cache=True,
            ...
        )

        # sample next token, write to output buffer
        last_logits = lm_head(step_out[:, -1:, :])
        next_token = (torch.multinomial(softmax(last_logits[:,-1]/T), 1)
                      if temperature > 0
                      else torch.argmax(last_logits[:, -1], dim=-1, keepdim=True))
        generated_action_tokens[:, t] = next_token.squeeze(-1)

    return generated_action_tokens  # ← 永远返回 (B, max_decoding_steps) 满长度!
```

[`src/lerobot/policies/pi0_fast/modeling_pi0_fast.py:1209-1233`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi0_fast/modeling_pi0_fast.py#L1209-L1233) — **真正的停止逻辑在 detokenize**:

```python
def detokenize_actions(self, tokens, action_horizon, action_dim):
    # token id → token string
    decoded_tokens = [self._paligemma_tokenizer.convert_ids_to_tokens(seq.tolist())
                      for seq in tokens]

    cleaned_tokens = []
    for token_seq in decoded_tokens:
        # === 这一行就是 lerobot 的"停止信号"识别 ===
        if "|" in token_seq:
            token_seq = token_seq[: token_seq.index("|")]   # 找第一个 "|",之后全丢

        # 顺便清掉中间可能出现的 "Action: " 残留
        ...
        cleaned_tokens.append(token_seq)

    raw_action_tokens = [
        torch.tensor(
            self._paligemma_tokenizer.convert_tokens_to_ids(token_seq), ...
        )
        for token_seq in cleaned_tokens
    ]
    action_tokens = [
        self._paligemma_tokens_to_act_tokens(raw) for raw in raw_action_tokens
    ]

    # 用 FAST tokenizer 反编码成连续 action
    return self.decode_actions_with_fast(action_tokens, action_horizon, action_dim)
```

## 逐行讲解 / What's happening

### 1. **训练时 `encode("|", add_eos=True)` 一行写两个信号**
- 中文: `encode("|", add_eos=True)` 内部会返回 `[id("|"), EOS_id]` 两个 token,所以训练时序列末尾是 `... T15 "|" EOS`。**两个 token 都参与 next-token-prediction loss**:模型学会"T15 之后是 |","|" 之后是 EOS。两个信号都被训出来,推理时哪个都能用。
- English: `encode("|", add_eos=True)` returns `[id("|"), EOS_id]` internally, so training sequences end `... T15 "|" EOS`. **Both tokens contribute to next-token-prediction loss**: the model learns "T15 → |" and "| → EOS". Both signals get trained; either can be used at inference.

### 2. **openpi 用 EOS 不用 "|" 的真实原因 — token id 的唯一性**
- 中文: EOS 在 PaliGemma 的 SentencePiece 词表里**只对应一个固定 id = 1**(全局唯一,语言无关)。"|" 经过 BPE 编码后**可能拆成多个 sub-token**(取决于上下文,比如 "abc|def" 可能跟 "abc | def" 的 "|" 拆法不同)。**检测 EOS 一行 `==1` 搞定;检测 "|" 要先 detokenize 成字符串**,在 decode loop 里做字符串操作很麻烦。
- English: EOS has a **fixed unique id = 1** in PaliGemma's SentencePiece vocab (language-agnostic, globally unique). "|" can be **split into multiple sub-tokens** after BPE (context-dependent: "abc|def" might tokenize "|" differently than "abc | def"). **Detecting EOS is a one-line `==1`; detecting "|" requires detokenizing to string first**, painful inside a decode loop.

### 3. **JAX 的 `lax.while_loop`** — 让数据依赖退出 JIT-friendly
- 中文: `jax.lax.while_loop(cond_fn, step_fn, init)` 是 JAX 标准控制流原语,**JIT 编译时把它编成一个真正的循环 + 数据依赖的退出条件**。这就是 openpi 能在 `lax.while_loop` 里检查 `all_eos` 而不破坏 JIT 的根本原因。换 PyTorch 没有等价物 — 直接写 `while` 或 `for + break` 在 `torch.compile` 下大概率破图。
- English: `jax.lax.while_loop(cond_fn, step_fn, init)` is a JAX standard control-flow primitive that **JIT-compiles into a real loop with data-dependent exit**. That's why openpi can check `all_eos` inside `lax.while_loop` without breaking JIT. PyTorch has no equivalent — writing `while` or `for + break` likely breaks `torch.compile`'s graph.

### 4. **lerobot 不 early-stop 的真实原因(混合 4 个工程考量)**
- 中文:
  - **(a) `torch.compile` 兼容性**(主因): `if next_token == EOS: break` 是数据依赖的破图操作,会导致 compile fall back 到 eager 或重新 trace
  - **(b) 代码简洁**: plain for loop 调试方便,KV cache 状态更容易追踪
  - **(c) 后处理截断本来就要做**:detokenize 时必须把 PaliGemma id 转回 FAST id 再做 IDCT,**字符串清洗反正绕不开**,decode 里 early-stop 是冗余
  - **(d) batch B>1 时 per-sample early-stop 实现复杂**:虽然单样本部署 B=1 是常态,但 batch eval (B=8~64) 时不同样本停止时机不同,统一死跑+截断省事
- English:
  - **(a) `torch.compile` compatibility** (primary): `if next_token == EOS: break` is data-dependent and breaks the graph, causing compile to fall back to eager or retrace
  - **(b) Code simplicity**: plain for loop is easier to debug, KV cache state easier to track
  - **(c) Post-process truncation is mandatory anyway**: detokenize must convert PaliGemma ids back to FAST ids before IDCT, **string cleaning can't be avoided**, so loop early-stop is redundant
  - **(d) Per-sample early-stop is complex with batch B>1**: while single-robot deployment has B=1, batch eval (B=8~64) has samples stopping at different times — "run-to-end + truncate" simplifies

### 5. **JAX 的 "全 batch sync" early-stop 限制**
- 中文: openpi 的 `all_eos = jnp.all(has_eos)` 要求 **batch 里每个样本都 EOS 才退出**。如果 batch 里一个样本生成长 action chunk(比如 30 token),其他 5 个早 EOS(15 token)的样本也得**陪跑 15 步**,FLOPs 浪费。但这比死跑满 256 步好,而且 JAX 的 vmap 哲学就是"batch 一起跑"。
- English: openpi's `all_eos = jnp.all(has_eos)` requires **every sample in batch to EOS before exit**. If one sample generates a long chunk (e.g. 30 tokens), the 5 samples that EOS-ed at 15 tokens **wait 15 extra steps** — wasted FLOPs. But it's still better than running all 256 steps, and JAX's vmap philosophy is "batch runs together".

## 类比 / The analogy

**openpi 是 "教室里所有人都考完才能下课"**:JAX 的 `lax.while_loop` 检测全 batch EOS,只要还有一个样本没生成 EOS,所有样本都接着 forward。早 EOS 的样本"等"一下,但比死跑满好。
**lerobot 是 "下课铃响了所有人都走,试卷上没写完的部分回家自己整理"**:PyTorch 的 for loop 跑满 256 步,**真正的"哪里是答案末尾"在 detokenize 字符串处理里用 `if "|" in token_seq` 一行截断**。代码简单、`torch.compile` 友好。
**两套方案数学等价**(都是用 "|" 或 EOS 作为停止信号),只是工程实现风格不同 — 反映了 JAX(函数式 + 显式控制流)和 PyTorch(动态图 + Pythonic) 的设计哲学差异。

**openpi is "the whole classroom finishes before anyone can leave"**: JAX's `lax.while_loop` checks all-batch EOS; as long as any sample hasn't EOS-ed, everyone keeps forwarding. Early-EOS samples "wait", but still better than running to max_steps.
**lerobot is "the bell rings everyone leaves, sort out your incomplete test at home"**: PyTorch's for loop runs all 256 steps, **the real "where the answer ends" lives in detokenize's `if "|" in token_seq`** one-liner. Code is simple, `torch.compile`-friendly.
**Mathematically equivalent** (both use "|" or EOS as the stop signal), engineering style differs — reflecting JAX (functional + explicit control flow) vs PyTorch (dynamic graph + Pythonic) design philosophy.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-VLA

这是 nanoVLA 课程里 `inference-loop` 槽位的**深度变体**(之前讲过 lerobot 的 async policy server,今天专门讲 token decode 阶段的停止机制设计)。

### 给 nanoVLA 的具体 lesson

**Lesson 1**: **训练时埋多个停止信号,推理时只用一个**。`encode("|", add_eos=True)` 让模型对 "|" 和 EOS 都熟悉,推理时根据 framework 选最方便的那一个 — **代码量基本相同,鲁棒性翻倍**。

**Lesson 2**: **token 唯一性决定可检测性**。EOS 在 SentencePiece 全局唯一 id = 1,**用 byte 比较**即可。BPE 词表里的多字符 token(如 "|")可能根据上下文拆词,**只能在字符串层面比较**,所以 decode loop 里慎用。

**Lesson 3**: **framework 决定 decode loop 设计**:
- 用 JAX → 用 `lax.while_loop` + EOS early-stop(标准做法)
- 用 PyTorch → 死跑 + detokenize 字符串截断(`torch.compile` 友好)
- 用 PyTorch + 不需要 `compile` → 也可以 plain Python `if break`,但失去 1-2× 推理加速
- 用 PyTorch + 想 batch eval → 必须死跑 + 截断,因为 per-sample early-stop 实现复杂

**Lesson 4**: **stop signal 设计与 action 编码绑定**。如果你换 action tokenizer(比如不用 FAST 而用 256-bin),停止信号设计也要改 —— 原版 OpenVLA 因为固定 7 token,完全不需要 stop signal,跑 7 步就停。FAST 的可变长度才让这个问题浮出水面。

### 在你自己的 nanoVLA 里怎么实现

**最简方案**(prototype 阶段):
- 训练数据 postfix 用 `+ tokenizer.encode("|", add_eos=True)` 埋两个信号
- 推理 decode loop 写 `if next_token_id == EOS_id: break`(JAX 或 PyTorch eager 都行)
- 不需要后处理字符串截断

**生产方案**(配合 `torch.compile`):
- 训练同上
- 推理死跑 `max_decoding_steps`(典型 256)
- detokenize 阶段做 `if "|" in token_seq: token_seq = token_seq[: token_seq.index("|")]`
- 浪费一些 FLOPs 但 `torch.compile` 提供 5-10× 整体加速

**batch eval 方案**(评估时 B=32~64):
- 必须走"死跑 + 截断"路径
- 用 `done_mask = torch.zeros(B)` 维护 per-sample 状态(可选,省 FLOPs 但代码复杂)

This is the **deep-dive variant** of the `inference-loop` slot in the nanoVLA curriculum (lerobot's async policy server was covered earlier; today specifically covers the stop-mechanism design at the token decode stage).

### Concrete lessons for nanoVLA

**Lesson 1**: **Plant multiple stop signals at training, use one at inference**. `encode("|", add_eos=True)` makes the model familiar with both "|" and EOS; pick the more convenient one at inference per your framework — **same code size, doubled robustness**.

**Lesson 2**: **Token uniqueness determines detectability**. EOS has globally unique id = 1 in SentencePiece, **byte-comparable**. Multi-character BPE tokens (like "|") may split context-dependently, **only comparable at string level**, so beware inside decode loops.

**Lesson 3**: **Framework determines decode loop design**:
- Using JAX → use `lax.while_loop` + EOS early-stop (standard)
- Using PyTorch → run-to-end + detokenize string truncation (`torch.compile`-friendly)
- Using PyTorch without needing `compile` → plain Python `if break` also fine, but lose 1-2× inference speedup
- Using PyTorch + batch eval → must run-to-end + truncate, per-sample early-stop is complex

**Lesson 4**: **Stop signal design is coupled with action encoding**. If you swap action tokenizer (e.g. 256-bin instead of FAST), stop signal design must change — OpenVLA had fixed 7 tokens, needed no stop signal, just ran 7 steps. FAST's variable length is what surfaced this problem.

### How to implement in your nanoVLA

**Minimal approach** (prototype phase):
- Training data postfix: `+ tokenizer.encode("|", add_eos=True)` plants both signals
- Inference decode loop: `if next_token_id == EOS_id: break` (JAX or PyTorch eager both fine)
- No post-process string truncation needed

**Production approach** (with `torch.compile`):
- Training: same
- Inference: run-to-end `max_decoding_steps` (typically 256)
- Detokenize: `if "|" in token_seq: token_seq = token_seq[: token_seq.index("|")]`
- Wastes some FLOPs but gets 5-10× speedup from `torch.compile`

**Batch eval approach** (eval B=32~64):
- Must use "run-to-end + truncate"
- Optionally maintain `done_mask = torch.zeros(B)` per-sample (saves FLOPs but adds complexity)

## 自己跑一遍 / Try it yourself

```python
# try.py — compare two stop strategies on a toy autoregressive model
import torch
import torch.nn as nn

VOCAB = 100
EOS_ID = 1
PIPE_ID = 5      # 假装 "|" 占 token id 5

class TinyAR(nn.Module):
    """Tiny AR model that emits EOS after some random length."""
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, 8)
        self.lm_head = nn.Linear(8, VOCAB)

    def forward(self, x):
        return self.lm_head(self.emb(x))

torch.manual_seed(0)
model = TinyAR().eval()
B, MAX_STEPS = 4, 30

# 训练一个简单的"看到 PIPE_ID 就输出 EOS"的模型(简化)
# 真实场景训练 100 epoch,我们用伪 stub:每步根据 prefix 长度 sample 一个 token
@torch.no_grad()
def mock_step(prefix):
    """返回看到 PIPE 后就 EOS 的 token,否则随机 [10..50]"""
    last = prefix[:, -1]
    out = torch.where(last == PIPE_ID,
                      torch.full_like(last, EOS_ID),
                      torch.randint(10, 50, last.shape))
    return out

# Strategy 1: Run-to-end + post-process truncate (lerobot style)
prefix = torch.randint(10, 50, (B, 1))
generated = torch.full((B, MAX_STEPS), -1, dtype=torch.long)
for t in range(MAX_STEPS):
    if t == 7:   # 假装第 7 步生成 PIPE,触发 EOS in next step
        prefix = torch.cat([prefix, torch.full((B, 1), PIPE_ID, dtype=torch.long)], dim=1)
        generated[:, t] = PIPE_ID
        continue
    next_tok = mock_step(prefix)
    prefix = torch.cat([prefix, next_tok.unsqueeze(1)], dim=1)
    generated[:, t] = next_tok

print("=== Strategy 1: run-to-end + truncate at '|' (lerobot style) ===")
for b in range(B):
    seq = generated[b].tolist()
    if PIPE_ID in seq:
        seq_trimmed = seq[:seq.index(PIPE_ID)]
    else:
        seq_trimmed = seq
    print(f"  batch {b}: full = {seq[:15]}..., truncated = {seq_trimmed}, length = {len(seq_trimmed)}")

# Strategy 2: Early-stop on EOS (openpi style)
torch.manual_seed(0)
prefix = torch.randint(10, 50, (B, 1))
generated2 = []
for t in range(MAX_STEPS):
    if t == 7:
        prefix = torch.cat([prefix, torch.full((B, 1), PIPE_ID, dtype=torch.long)], dim=1)
        generated2.append(torch.full((B,), PIPE_ID, dtype=torch.long))
        continue
    next_tok = mock_step(prefix)
    prefix = torch.cat([prefix, next_tok.unsqueeze(1)], dim=1)
    generated2.append(next_tok)
    # 检查是否全 batch EOS
    if (next_tok == EOS_ID).all():
        print(f"\n=== Strategy 2: early-stop on EOS at step {t+1} (openpi style) ===")
        break

generated2 = torch.stack(generated2, dim=1)
for b in range(B):
    seq = generated2[b].tolist()
    print(f"  batch {b}: full = {seq}, length = {len(seq)}")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
=== Strategy 1: run-to-end + truncate at '|' (lerobot style) ===
  batch 0: full = [..., 5, 1, 1, 1, ..., 1]..., truncated = [random 7 ints], length = 7
  batch 1: ...
  batch 2: ...
  batch 3: ...

=== Strategy 2: early-stop on EOS at step 9 (openpi style) ===
  batch 0: full = [random 7 ints, 5, 1], length = 9
  batch 1: ...
```

**Strategy 1 跑完了 30 步**,Strategy 2 在第 9 步就停了。两者得到的 action token 序列内容**完全一致**(因为模型行为一样),只是 Strategy 2 早结束省 21 步 forward。Strategy 1 的代码更简单(无 break),Strategy 2 的 wall clock 更快但代码多一行 if。

**Strategy 1 runs all 30 steps**, Strategy 2 stops at step 9. The resulting action token sequences are **identical** (same model behavior), Strategy 2 just saves 21 forward steps. Strategy 1 has simpler code (no break), Strategy 2 has faster wall clock but one extra if line.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **HF transformers `generate()`**: 有 `eos_token_id` 参数支持 early-stop,内部用 `stopping_criteria`,但默认下 batch 内 per-sample 停止后用 padding 填,而不是 sync 等所有样本 / Has `eos_token_id` parameter for early-stop, uses `stopping_criteria` internally; default behavior pads per-sample after their stop instead of syncing across batch.
- **vLLM batched decoding**: 多 request 各自 EOS 后从 batch 移除,**continuous batching** 中替补新 request — 是 lerobot "死跑+截断"方案的极致优化版 / Per-request EOS removes from batch, **continuous batching** swaps in new requests — extreme version of lerobot's "run-to-end + truncate".
- **OpenVLA 不需要 stop signal**: 固定生成 7 个 action token 然后停 — 因为 action 表示固定长度,**根本没有"什么时候停"的问题** / OpenVLA needs no stop signal — fixed 7 action tokens then stop, fixed-length representation means no "when to stop" question.
- **OpenVLA-OFT 也不需要**: parallel decode 一次出 56 个 hidden state,**没有 decode loop** / OFT also doesn't need it: parallel decode emits 56 hidden states at once, no decode loop.
- **GPT 系 generate**: 同样的 `eos_token_id` 设计,但 GPT 训练时只埋 EOS,没有 "|" 这种语义标记 — pi0-FAST 比 GPT 多一层"语义停止"鲁棒性 / Same `eos_token_id` design, but GPT trains only with EOS, no semantic marker like "|" — pi0-FAST adds one more layer of "semantic stop" robustness.

## 注意事项 / Caveats / when it breaks

- **lerobot 死跑 256 步对 short chunk 是浪费** / **lerobot's 256-step run is wasteful for short chunks**: LIBERO 单步 action chunk 只有 ~15 个 token,死跑剩下 241 步是 ~94% 浪费 FLOPs。但 `torch.compile` 加速 + 单 token forward,绝对开销 ~10ms,可接受 / LIBERO single-step chunks have only ~15 tokens; running 241 extra steps is ~94% FLOP waste. But `torch.compile` + single-token forward, absolute overhead ~10ms, acceptable.
- **openpi 早 EOS 的 batch 浪费 vs lerobot 死跑** / **openpi's early-EOS batch waste vs lerobot's run-to-end**: openpi 浪费 = max(seq lengths) - min(seq lengths) 步;lerobot 浪费 = max_decoding_steps - avg(seq lengths) 步。**openpi 在 batch 同质化时高效,lerobot 在 batch 异质化时也不变差** / openpi waste = max - min seq lengths; lerobot waste = max_decoding_steps - avg seq lengths. **openpi efficient on homogeneous batches, lerobot equally bad on heterogeneous batches**.
- **"|" 在 prompt 文本里出现的风险** / **"|" appearing inside prompt text risk**: 如果 task prompt 文本里偶然出现 "|"(比如 `"pick up A | the cup"`),detokenize 会误截断。lerobot 用 `assert token_seq[0] == "Action"` 保证前缀正确,但 prompt 中间的 "|" 不防 / If task prompt accidentally contains "|" (e.g. `"pick up A | the cup"`), detokenize will mis-truncate. lerobot asserts `token_seq[0] == "Action"` to guarantee prefix correctness, but middle-of-prompt "|" isn't guarded.
- **JAX `lax.while_loop` 不能 carry 任意 dtype** / **JAX `lax.while_loop` can't carry arbitrary dtypes**: openpi 把 `output_tokens` 设计为 `(B, max_steps)` 预分配 zeros,**避免动态 shape**。新手容易写成 list.append 然后挂掉 / openpi pre-allocates `output_tokens` as `(B, max_steps)` zeros to **avoid dynamic shape**. Newcomers easily write list.append and break.

## 延伸阅读 / Further reading

- JAX `lax.while_loop` docs: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.while_loop.html
- HF transformers GenerationMixin: https://huggingface.co/docs/transformers/main_classes/text_generation
- PyTorch `torch.compile` graph break docs: https://docs.pytorch.org/docs/main/torch.compiler_troubleshooting.html
- vLLM continuous batching paper / docs — extreme optimization of "run-to-end + per-sample drop"
- Today's companion note on pi0-FAST multimodal fusion — covers training-time postfix construction
