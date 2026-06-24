---
date: 2026-06-24
topic: diffusion
source: trending
repo: OpenHelix-Team/Unified-Diffusion-VLA
file: models/inference/inference_action.py
permalink: https://github.com/OpenHelix-Team/Unified-Diffusion-VLA/blob/main/models/inference/inference_action.py
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, vla, discrete-diffusion, logits-processor, constrained-decoding, iclr2026]
---

# ActionIDConstraintLogitsProcessor：一个 mask 把 LLM 的输出空间限制到动作词表 / ActionIDConstraintLogitsProcessor: One Mask That Constrains an LLM's Output Space to the Action Vocabulary

> **一句话 / In one line**: 把所有非动作 token 的 logit 置为 `-inf`，让标准 LLM 采样自然只输出动作 token——这就是 UD-VLA（ICLR 2026）把离散扩散 VLA 推理压成 10 行的核心技巧。/ Set every non-action token's logit to `-inf` so that standard LLM sampling only ever emits action tokens — this 10-line trick is how UD-VLA (ICLR 2026) implements discrete-diffusion VLA inference.

## 为什么重要 / Why this matters

传统 VLA（如 OpenVLA）在 LLM 之外加一个独立的动作解码器，pipeline 复杂。UD-VLA 的做法更激进：用 Emu3 的词表末尾 1024 个 token 编码 7 维机器人动作（每维 ≈ 256 个分箱），推理时只要一个 `LogitsProcessor` 把其余 token 全部屏蔽，LLM 就成了天然的离散扩散 VLA。不需要改模型结构，不需要新的解码器，5 行 `__call__` 搞定一切。

Traditional VLAs (like OpenVLA) add a separate action decoder on top of the LLM — a complex pipeline. UD-VLA takes a more radical approach: it encodes 7-dimensional robot actions in the last 1024 tokens of Emu3's vocabulary (roughly 256 bins per dimension), then uses a single `LogitsProcessor` to mask everything else to `-inf`. The LLM becomes a natural discrete-diffusion VLA with no structural changes — no new decoder, 5 lines of `__call__` handle everything.

## 代码 / The code

`OpenHelix-Team/Unified-Diffusion-VLA` — [`models/inference/inference_action.py`](https://github.com/OpenHelix-Team/Unified-Diffusion-VLA/blob/main/models/inference/inference_action.py)

```python
from transformers import LogitsProcessor
import torch


class ActionIDConstraintLogitsProcessor(LogitsProcessor):
    """
    At each autoregressive decoding step, zero out every token that is NOT
    in `allowed_token_ids` by setting its logit to -inf.
    This forces the LLM to sample only from the action-token sub-vocabulary.

    UD-VLA encodes 7-DoF robot actions as tokens in the tail of Emu3's vocabulary:
        action_token_id = last_token_id - quantized_action_value
    Decoding inverts this:
        quantized_action_value = last_token_id - sampled_token_id
    """

    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = allowed_token_ids   # list or 1-D tensor of valid action token IDs

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            input_ids: shape (batch, seq_len) — tokens generated so far
            scores:    shape (batch, vocab_size) — raw logits for the next token
        Returns:
            scores with all non-action logits set to -inf
        """
        mask = torch.zeros_like(scores, dtype=torch.bool)
        if mask.ndim == 1:
            # Single sample (no batch dimension)
            mask[self.allowed_token_ids] = True
        else:
            # Batched decoding
            mask[:, self.allowed_token_ids] = True
        scores[~mask] = -float("inf")
        return scores


# ---- How UD-VLA uses this processor ----

# Build the allowed token set at model-load time:
VOCAB_SIZE   = tokenizer.vocab_size          # e.g. 184,622 for Emu3
LAST_TOKEN   = VOCAB_SIZE - 1
NUM_ACTION_BINS = 1024                       # 7 dims × 256 bins each → 1024 unique IDs

allowed_ids  = list(range(LAST_TOKEN - NUM_ACTION_BINS, LAST_TOKEN + 1))
processor    = ActionIDConstraintLogitsProcessor(allowed_ids)

# At inference: pass processor into generate()
with torch.no_grad():
    output_ids = model.generate(
        input_ids=vision_language_tokens,    # image + instruction prefix
        logits_processor=[processor],        # only allow action tokens
        max_new_tokens=7,                    # one token per DoF
        do_sample=False,                     # greedy decoding
    )

# Decode: invert the token-encoding arithmetic
generated = output_ids[:, input_ids.shape[1]:]   # strip the prefix
action_bins = LAST_TOKEN - generated              # shape (batch, 7)
actions = (action_bins.float() / 255.0) * 2 - 1  # rescale to [-1, 1]
```

## 逐行讲解 / What's happening

1. **`mask = torch.zeros_like(scores, dtype=torch.bool)`**:
   - 中文: 建一个全为 False 的布尔掩码，形状和 logit 矩阵完全一致（`batch × vocab_size`）。
   - English: Create an all-False boolean mask shaped exactly like the logits tensor (`batch × vocab_size`).

2. **`mask[:, self.allowed_token_ids] = True`**:
   - 中文: 批量索引把允许的动作 token 位置全部设为 True。`allowed_token_ids` 是一个列表，PyTorch 的 advanced indexing 一次性把这些列全部选中。
   - English: Batch-index assignment sets all allowed action token positions to True in one shot. PyTorch's advanced indexing selects all those columns simultaneously.

3. **`scores[~mask] = -float("inf")`**:
   - 中文: 把所有不在动作词表里的 token 的 logit 直接打成负无穷。softmax 之后这些 token 的概率精确为 0，LLM 绝对不会采样到它们。
   - English: Set every non-action token's logit to negative infinity. After softmax, those tokens have exactly zero probability — the LLM can never sample them.

4. **动作的编码方式 `action_token_id = last_token_id - action_bin`**:
   - 中文: Emu3 的词表末尾 1024 个位置被征用为动作 token。把量化值编码为"从末尾倒数多少"，解码时反向计算即可。这样完全不需要改词表，只需要一个 offset 运算。
   - English: The last 1024 slots of Emu3's vocabulary are repurposed as action tokens. Encoding is "distance from the end"; decoding inverts the arithmetic. No vocabulary changes needed — just one offset operation.

5. **`generated = output_ids[:, input_ids.shape[1]:]`**:
   - 中文: `generate()` 返回完整序列（prefix + 新 token），用切片只保留新生成的 7 个动作 token。
   - English: `generate()` returns the full sequence (prefix + new tokens); slice off the prefix to get just the 7 newly-generated action tokens.

6. **`action_bins = LAST_TOKEN - generated`**:
   - 中文: 解码算术。token ID 越大 → 动作 bin 越小（倒序编码），减法还原 bin 值。再做归一化就得到实际的连续动作值。
   - English: Decoding arithmetic. Higher token ID → smaller action bin (inverted encoding), subtraction recovers the bin value. Rescaling then gives the actual continuous action value.

## 类比 / The analogy

想象一台只能印刷特定词语的打字机：你拔掉除了"机器人动作指令"之外的所有按键（logit → -inf），让打字员（LLM）用剩下的 1024 个键打字，输出天然就只有动作指令，不需要再做分词器解析或格式校验。

Imagine a typewriter where you physically remove every key except the 1024 "robot action command" keys (logit → -inf). Whatever the typist (LLM) types, the output is guaranteed to be an action command — no need for post-hoc parsing or format validation.

## 自己跑一遍 / Try it yourself

```python
import torch
from transformers import LogitsProcessor

class ActionIDConstraintLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_ids):
        self.allowed_ids = allowed_ids
    def __call__(self, input_ids, scores):
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask[:, self.allowed_ids] = True
        scores[~mask] = float("-inf")
        return scores

VOCAB_SIZE = 1000; LAST = VOCAB_SIZE - 1; N_ACTION = 7
allowed = list(range(LAST - N_ACTION + 1, LAST + 1))  # top-7 tokens
proc = ActionIDConstraintLogitsProcessor(allowed)

scores = torch.randn(2, VOCAB_SIZE)          # batch=2
masked = proc(None, scores.clone())
print("Non-inf tokens per sample:", (masked != float("-inf")).sum(1))  # should be 7
print("Sampled token IDs:", masked.argmax(1))                          # must be in allowed
action_bins = LAST - masked.argmax(1)
print("Decoded action bins:", action_bins)  # 0-6
```

运行 / Run with:
```bash
pip install torch transformers
python try.py
```

预期输出 / Expected output:
```
Non-inf tokens per sample: tensor([7, 7])
Sampled token IDs: tensor([...])   # both in range [993, 999]
Decoded action bins: tensor([...]) # both in range [0, 6]
```

中文：无论原始 logit 分布如何，mask 之后每个样本只有 7 个 token 有非负无穷的 logit，argmax 一定落在动作词表里。

Regardless of the original logit distribution, after masking each sample has exactly 7 non-(-inf) logits — argmax is guaranteed to land in the action vocabulary.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **`transformers` `ForceTokensLogitsProcessor`** / **`transformers` `ForceTokensLogitsProcessor`**: 强制在特定步输出特定 token，和 `ActionIDConstraintLogitsProcessor` 是同一类 logits processor，用于 beam search 控制 / forces specific tokens at specific steps — same `LogitsProcessor` interface, used for beam-search control.
- **OpenVLA 的 un_norm 解码** / **OpenVLA's un_norm decoding**: 不用 logits mask，而是只预测 action token 序列然后反归一化，思路类似但结构上更重（需要单独 forward）/ no logits mask — predicts only the action token sequence then denorms; similar idea but heavier architecture (separate forward pass).
- **ICLR 2026 UD-VLA 论文所称的"统一离散扩散 VLA"** / **"Unified Discrete Diffusion VLA" (ICLR 2026 paper)**: 这个 processor 是其推理侧的全部秘密——把 LLM generation 变成 masked diffusion 推理，无需修改模型 / this processor is the entire inference-side secret — turns LLM generation into masked diffusion inference, zero model changes.

## 注意事项 / Caveats / when it breaks

- **`allowed_token_ids` 必须在 GPU 上（或兼容设备）** / **`allowed_token_ids` must be on GPU (or compatible device)**: 如果在 CPU list 上做 `mask[:, cpu_list]` 而 `scores` 在 GPU，会有隐式拷贝或报错 / indexing a GPU tensor with a CPU list triggers implicit copy or errors — convert to a GPU tensor first.
- **greedy decoding 和 sampling 都支持，但 beam search 可能有问题** / **Greedy and sampling both work; beam search may have issues**: beam search 在 `logits_processor` 调用时的 `scores` shape 可能是 `(batch × num_beams, vocab_size)`，确认 ndim 分支正确处理。
- **只适用于 7 DoF 固定 token 数** / **Only works for fixed token count per action**: 当前实现假设每个 DoF 独立输出一个 token，`max_new_tokens=7`。可变长度动作或连续 token 流需要额外的停止逻辑。

## 延伸阅读 / Further reading

- [Unified Diffusion-based Robotics Representation (UD-VLA, ICLR 2026)](https://openreview.net/forum?id=UD-VLA)
- [Emu3: Next-Token Prediction is All You Need (Wang et al., 2024)](https://arxiv.org/abs/2409.18869)
- [HuggingFace Transformers LogitsProcessor docs](https://huggingface.co/docs/transformers/internal/generation_utils#transformers.LogitsProcessor)
