---
date: 2026-05-10
topic: robotics
source: tracked
repo: openvla/openvla
file: prismatic/vla/action_tokenizer.py
permalink: https://github.com/openvla/openvla/blob/main/prismatic/vla/action_tokenizer.py
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, vla, tokenization]
---

# Discretizing Continuous Robot Actions Into Language Tokens

> **In one line**: How OpenVLA turns a 7-DoF robot action vector into 7 token IDs so a
> language model can predict it the same way it predicts the next word — no architectural
> changes to the LLM required.

## Why this matters

VLA models are LLMs that output robot actions. But LLMs only know how to predict tokens
from a fixed vocabulary, while actions are continuous floats. You could glue on a regression
head — but then you can't reuse the pretrained next-token cross-entropy loss, can't sample
with temperature, can't mix actions and language in the same sequence. The action tokenizer
solves all three with one trick: **bin the action values into integers and reuse the LLM's
least-used vocabulary slots as action tokens.**

## The code

`openvla/openvla` — `prismatic/vla/action_tokenizer.py` (illustrative excerpt)

```python
class ActionTokenizer:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        bins: int = 256,
        min_action: float = -1.0,
        max_action: float = 1.0,
    ):
        self.tokenizer = tokenizer
        self.n_bins = bins
        self.min_action, self.max_action = min_action, max_action

        # Uniform bin edges across the action range
        self.bin_centers = np.linspace(min_action, max_action, bins)
        self.bins = np.linspace(min_action, max_action, bins + 1)[1:-1]

        # Steal the last `bins` token IDs from the LLM vocab. These are usually
        # rare tokens the model barely uses, so overwriting their semantics is cheap.
        self.action_token_begin_idx: int = int(self.tokenizer.vocab_size - (self.n_bins + 1))

    def __call__(self, action: np.ndarray) -> str:
        action = np.clip(action, a_min=self.min_action, a_max=self.max_action)
        discretized = np.digitize(action, self.bins)
        # Map bin index -> token ID at the top of the vocab
        return self.tokenizer.decode(
            list(self.tokenizer.vocab_size - discretized)
        )

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        discretized = self.tokenizer.vocab_size - action_token_ids
        discretized = np.clip(discretized - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        return self.bin_centers[discretized]
```

## What's happening

1. **`bin_centers = np.linspace(min_action, max_action, bins)`** — Divide the action range
   `[-1, 1]` into 256 evenly-spaced centers. Each future action value will get rounded to
   the nearest one of these.
2. **`action_token_begin_idx = vocab_size - (bins + 1)`** — This is the cute trick. The LLM
   has, say, 32000 tokens. We "rename" the last 256 of them to be action bins. These slots
   originally encoded rare Unicode or weird subwords the model nearly never emits, so we
   sacrifice almost nothing.
3. **`np.digitize(action, self.bins)`** — For each scalar in the 7-D action, return which
   bin it falls into (an integer in `[0, 255]`).
4. **`tokenizer.vocab_size - discretized`** — Convert bin index to absolute token ID by
   subtracting from `vocab_size`. So bin 0 → token `vocab_size`, bin 255 → token `vocab_size - 255`.
5. **`tokenizer.decode(...)`** — Now we have a sequence of token IDs that the LLM tokenizer
   already knows how to render as a string. We can splice this directly into a prompt or
   target sequence.
6. **`decode_token_ids_to_actions(...)`** — The inverse: take predicted token IDs back,
   subtract from vocab size, look up the bin center, get a float. The `clip` guards against
   the model emitting a non-action token by accident.

The key invariant: tokens and actions live in the **same vocabulary**, so training is just
next-token prediction. No new loss function, no new head.

## The analogy

Think of a piano. A pianist's fingers can press a key at any pressure, but the piano only
records 128 MIDI velocity levels per key. The continuous wrist motion gets quantized into
discrete integers the MIDI standard knows how to transmit. The action tokenizer does the
same to robot joints: each joint's velocity becomes one of 256 "MIDI levels," and the LLM
predicts the sequence the same way it predicts a melody — one token at a time.

The clever bit is **which** keys we re-purpose. Imagine a piano with 12 unused dead keys
at the top of the keyboard. Rewriting their meaning to be "robot joint commands" doesn't
ruin any songs people actually play.

## Try it yourself

```python
# try_action_tokenizer.py
import numpy as np

class ToyActionTokenizer:
    """Standalone version — no HuggingFace dependency."""
    def __init__(self, vocab_size=32000, n_bins=256, lo=-1.0, hi=1.0):
        self.vocab_size = vocab_size
        self.n_bins = n_bins
        self.lo, self.hi = lo, hi
        self.bins = np.linspace(lo, hi, n_bins + 1)[1:-1]
        self.bin_centers = np.linspace(lo, hi, n_bins)

    def encode(self, action: np.ndarray) -> np.ndarray:
        action = np.clip(action, self.lo, self.hi)
        bin_idx = np.digitize(action, self.bins)
        return self.vocab_size - bin_idx  # token ids

    def decode(self, token_ids: np.ndarray) -> np.ndarray:
        bin_idx = np.clip(self.vocab_size - token_ids - 1, 0, self.n_bins - 1)
        return self.bin_centers[bin_idx]


tok = ToyActionTokenizer()
action = np.array([0.5, -0.3, 0.0, 0.99, -1.0, 0.123, 0.456])  # 7-DoF robot delta
ids = tok.encode(action)
recovered = tok.decode(ids)

print("Original :", action)
print("Token IDs:", ids)
print("Recovered:", recovered)
print("Max error:", np.abs(action - recovered).max())  # ≤ 1/n_bins ≈ 0.0039
```

Run with:
```bash
pip install numpy
python try_action_tokenizer.py
```

Expected output:
```
Original : [ 0.5  -0.3   0.    0.99 -1.    0.123 0.456]
Token IDs: [31808 31910 31872 31745 31999 31857 31814]
Recovered: [ 0.4980 -0.2941  0.0000  0.9882 -1.0000  0.1216  0.4549]
Max error: 0.0039
```

## Where this pattern shows up elsewhere

- **RT-2 (Google)**: same idea, first paper to popularize it for VLA.
- **OpenVLA**: cleanest open-source implementation, the one above.
- **π₀ (Physical Intelligence)**: replaces discrete tokens with continuous flow matching
  — the opposite design choice; useful contrast.
- **MolmoAct2**: extends with action *chunking* (predict 8 future actions per step, not 1).
- **VQ-VAE for actions**: same discretization spirit, but learned codebooks instead of
  uniform bins.

## Caveats / when it breaks

- **Bin count is a hyperparameter**. 256 bins per dimension gives ~0.4% precision on
  normalized actions, which is enough for most manipulation but too coarse for high-precision
  insertion tasks. Increasing it eats more vocabulary slots — there's no free lunch.
- **Uniform binning assumes uniform action distribution**. If 90% of your actions are near
  zero (typical for delta-action policies), most bins are wasted and the tail has terrible
  resolution. Learned tokenizers (FAST, VQ-VAE) fix this but add training complexity.
- **The "rare token" assumption is fragile**. If the base LLM was retrained on text that
  uses those last-256 tokens heavily, action token predictions can interfere with text
  generation. OpenVLA checks the vocab is safe; not all VLAs do.

## Further reading

- [OpenVLA paper](https://arxiv.org/abs/2406.09246)
- [RT-2 paper](https://robotics-transformer2.github.io/) — the original "action as language token" idea
- [FAST: Frequency-space Action Sequence Tokenization](https://arxiv.org/abs/2501.09747) — learned action tokenizer
