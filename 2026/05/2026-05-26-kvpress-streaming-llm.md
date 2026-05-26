---
date: 2026-05-26
topic: infrastructure
source: trending
repo: NVIDIA/kvpress
file: kvpress/presses/streaming_llm_press.py
permalink: https://github.com/NVIDIA/kvpress/blob/243f71baa229ca1308e2540de164d064c053465c/kvpress/presses/streaming_llm_press.py#L1-L54
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, infrastructure, kv-cache, compression, attention-sink, streaming-llm]
---

# StreamingLLM in 30 Lines: Why You Must Keep the First Tokens

> **In one line**: A paper-level KV cache compression algorithm — "keep the sink tokens and the recent window, throw the middle away" — collapses to five lines of tensor ops once the right abstraction is in place.

## Why this matters

If you served today's earlier note on vLLM's block eviction, you saw KV cache management at the **block** level: 16-token chunks get LRU'd in and out of GPU memory. But there's another lever — *token-level* compression: keep only the most important individual tokens within whatever context window survives. NVIDIA's [kvpress](https://github.com/NVIDIA/kvpress) is a library that does exactly this, with about thirty different scoring strategies plugged into a single `ScorerPress` base class.

The smallest of them — `StreamingLLMPress` — is also the most surprising. It implements the famous result from the [StreamingLLM paper (arXiv 2309.17453)](https://arxiv.org/abs/2309.17453): if you naively use a sliding window over the KV cache (keep only the last N tokens, drop the older ones), **the model collapses into gibberish almost immediately**. The fix is comically simple — keep the very first 4 tokens, always — and the model recovers nearly full quality. Those first tokens are called **attention sinks**: empirically, the softmax of attention dumps a huge fraction of its probability mass onto them, regardless of their semantic content, because softmax has to sum to 1 and the model uses the early tokens as a "default place to look" when nothing else is relevant.

This snippet is a textbook example of how a research result that took a paper to motivate becomes a one-screen implementation once the right abstraction (`score()` → top-k → gather) is factored out.

## The code

`NVIDIA/kvpress` — [`kvpress/presses/streaming_llm_press.py`](https://github.com/NVIDIA/kvpress/blob/243f71baa229ca1308e2540de164d064c053465c/kvpress/presses/streaming_llm_press.py#L1-L54)

```python
from dataclasses import dataclass
import torch
from torch import nn
from kvpress.presses.scorer_press import ScorerPress


@dataclass
class StreamingLLMPress(ScorerPress):
    """
    StreamingLLM: Window-based KV cache compression with sink tokens.

    Implements sliding window approach preserving first few tokens (sink tokens)
    and most recent tokens, while pruning middle tokens.

    Based on StreamingLLM (https://arxiv.org/abs/2309.17453).
    """

    compression_ratio: float = 0.0
    n_sink: int = 4

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:

        k_len = keys.shape[2]
        assert k_len > self.n_sink, f"Input should contain more tokens than n_sink={self.n_sink}"
        n_pruned = k_len - int(k_len * (1 - self.compression_ratio))
        scores = torch.ones_like(keys[..., 0])
        scores[:, :, self.n_sink : self.n_sink + n_pruned] = 0

        return scores
```

And the parent class that turns scores into actual eviction — `kvpress/presses/scorer_press.py` lines 76–102:

```python
def compress(self, module, hidden_states, keys, values, attentions, kwargs):
    if self.compression_ratio == 0:
        return keys, values

    # Compute scores (subclass implements this)
    scores = self.score(module, hidden_states, keys, values, attentions, kwargs)

    # Get indices of KV pairs with the highest scores
    k_len = keys.shape[2]
    n_kept = int(k_len * (1 - self.compression_ratio))
    indices = scores.topk(n_kept, dim=-1).indices
    indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

    # Prune keys and values
    keys = keys.gather(2, indices).contiguous()
    values = values.gather(2, indices).contiguous()
    return keys, values
```

## What's happening

1. **`keys` has shape `(batch, num_kv_heads, seq_len, head_dim)`.** This is the standard transformer KV cache layout. `k_len = keys.shape[2]` reads off the current sequence length.

2. **Compute how many tokens to drop.** `n_kept = int(k_len * (1 - compression_ratio))` — if `k_len = 1000` and `compression_ratio = 0.5`, you keep 500 and prune 500. `n_pruned = k_len - n_kept` is the gap.

3. **Build a `scores` tensor full of ones.** `torch.ones_like(keys[..., 0])` produces a `(batch, num_kv_heads, seq_len)` tensor of ones — one score per token position. (The slice `keys[..., 0]` is just a tensor shape trick to drop the `head_dim` axis cheaply.)

4. **Zero out exactly the middle.** `scores[:, :, n_sink : n_sink + n_pruned] = 0`. Everything before `n_sink` (the sink tokens) stays at 1. Everything from `n_sink + n_pruned` onward (the recent window) also stays at 1. The block between them is zeroed.

5. **The base class does the rest.** `compress()` calls `scores.topk(n_kept)` to grab the indices of the `n_kept` highest-scoring positions, then `keys.gather(2, indices)` plucks those positions out of the KV tensor. Tokens with score 0 lose the top-k race and disappear.

6. **Why `score` returns 1s and 0s — couldn't it just return indices?** Because every subclass overrides `score` with a different formula (`SnapKVPress`, `ExpectedAttentionPress`, `Knorm`, etc.) and they all share `compress`. By making `score` return a *per-position importance value*, the framework gets pluggability for free. StreamingLLM happens to use a binary mask, but `ExpectedAttentionPress` returns real-valued estimated attention weights, and they both go through the same top-k machinery.

7. **`n_sink = 4` is the magic number.** The StreamingLLM paper finds that 4 is plenty — the attention sink phenomenon is concentrated almost entirely on the *very first* tokens. With `n_sink = 0` (pure sliding window), perplexity blows up to thousands within a few hundred tokens. With `n_sink = 4`, you can decode millions of tokens with perplexity barely above baseline.

## The analogy

Imagine you're a courtroom stenographer running out of paper. You can't keep transcribing everything forever. The obvious move: tear off the oldest pages and only keep the last 50. **The model running with a naive sliding window does exactly this — and produces gibberish.**

Why? Because the judge keeps glancing at the *opening statement* throughout the entire trial. Even when the testimony is about cross-examination from page 200, the judge's eye reflexively returns to "Your Honor, my client...". Tear off the opening statement and the judge becomes disoriented — they don't have an anchor.

`n_sink = 4` is just **keeping the first four pages of the opening statement permanently**, no matter how many other pages you shred. Cheap, simple, and it turns out to be enough: the attention pattern has a stable "default fallback" to land on. StreamingLLM is the result of someone empirically discovering the judge's tic. The five lines of `score()` are the code version of "always keep page 1."

## Try it yourself

A toy version that runs without transformers — just NumPy:

```python
"""Minimal StreamingLLM-style compression on a fake KV cache."""
import numpy as np

def streaming_llm_compress(keys, n_sink=4, compression_ratio=0.5):
    """
    keys: (batch, num_kv_heads, seq_len, head_dim) — the cache to compress.
    Returns the compressed keys.
    """
    batch, heads, k_len, head_dim = keys.shape
    n_kept = int(k_len * (1 - compression_ratio))
    n_pruned = k_len - n_kept

    # Score every position: 1 = keep, 0 = drop.
    scores = np.ones((batch, heads, k_len))
    scores[:, :, n_sink : n_sink + n_pruned] = 0

    # Pick the top-n_kept positions.
    keep_idx = np.argsort(-scores, axis=-1)[..., :n_kept]
    keep_idx = np.sort(keep_idx, axis=-1)  # preserve original order

    # Gather along the seq_len axis.
    return np.take_along_axis(keys, keep_idx[..., None], axis=2)


# Fake KV cache: 1 batch, 1 head, 16 tokens, 4 dims/head. Tokens labeled 0..15.
keys = np.arange(16).reshape(1, 1, 16, 1) * np.ones((1, 1, 16, 4))
out = streaming_llm_compress(keys, n_sink=4, compression_ratio=0.5)

# Print which token IDs survived
print("kept token positions:", out[0, 0, :, 0].astype(int).tolist())
```

Run with:
```bash
pip install numpy
python try_streaming.py
```

Expected output:
```
kept token positions: [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15]
```

Notice the shape: the first 4 tokens (sinks) are preserved, the middle 4 (positions 4–7) are dropped, and the last 8 (recent window) are kept. Out of 16 tokens we kept 8 — a 0.5 compression ratio.

## Where this pattern shows up elsewhere

- **HuggingFace `transformers` sliding-window cache** with `n_sink > 0`: the same idea, integrated into Llama/Mistral KV caches.
- **vLLM `SlidingWindowSpec`** ([source](https://github.com/vllm-project/vllm/blob/main/vllm/v1/kv_cache_interface.py)): vLLM's block-level cousin — handles sliding windows by limiting which blocks remain attended.
- **Anthropic's prompt caching** uses a related "stick the system prompt + early tokens at the front of the cache" heuristic for a different reason (cache-hit alignment).
- **Many MoE long-context tricks**: pyramidKV, AdaKV, SnapKV (also in kvpress) — they all subclass `ScorerPress` and override exactly one function: `score()`.

## Caveats / when it breaks

- **The 0/1 binary score still has to win a top-k.** If `compression_ratio = 0`, then `n_pruned = 0` and the middle slice `[n_sink : n_sink]` is empty — every position scores 1 and `topk` becomes order-dependent (PyTorch's `topk` is stable, so OK in practice).
- **No re-ranking inside the kept set.** Within the recent window, every token has score 1 — you can't prefer position 999 over 998. If a downstream method wants "keep recent tokens *and* prefer the high-attention ones among them," it needs a non-binary score (that's what `ExpectedAttentionPress` does).
- **Position embeddings.** When you drop tokens 5–7 from the cache, the surviving tokens still carry their *original* RoPE position IDs. Some models (e.g., long-RoPE variants) misbehave when there are gaps. The paper's "Key Rerotation" trick — and kvpress's `KeyRerotationPress` wrapper — re-rotate the surviving keys to closes those gaps. Plain `StreamingLLMPress` skips that step.
- **The "first 4" number is empirical.** It works because LLMs are trained with a `<bos>` token plus a system prompt, and those positions accumulate sink weight. A model trained without that structure might need a different `n_sink`.

## Further reading

- StreamingLLM paper — the experimental result that motivates the algorithm: https://arxiv.org/abs/2309.17453
- Attention sink phenomenon (Xiao et al., HuggingFace blog explainer): https://huggingface.co/blog/tomaarsen/attention-sinks
- kvpress repo & leaderboard (KV-cache compression Olympics): https://github.com/NVIDIA/kvpress
- `ScorerPress` base class: [`scorer_press.py`](https://github.com/NVIDIA/kvpress/blob/243f71baa229ca1308e2540de164d064c053465c/kvpress/presses/scorer_press.py)
- Companion algorithms — try replacing `score()` with `keys.norm(dim=-1).neg()` to reproduce `KnormPress`, or the L2 of attention weights for `SnapKVPress`.
