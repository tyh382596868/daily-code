---
date: 2026-10-28
topic: infrastructure
source: tracked
repo: vllm-project/vllm
file: vllm/v1/core/kv_cache_manager.py
permalink: https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, infrastructure, kv-cache]
---

# vLLM 前缀命中：最后一个 token 仍要重算 / vLLM Prefix Cache: Recompute the Last Token

> **一句话 / In one line**: 即使 prompt 全部命中 KV cache，vLLM 也刻意留下最后一个 token 给模型执行，以取得它的 logits。 / Even a full prefix-cache hit leaves the final token for execution so the model produces its logits.

## 为什么重要 / Why this matters

KV cache 保存的是“处理完某个 token 后”的注意力状态，不等于已经得到该 token 之后的下一个 token 分布。

A KV cache stores attention state after a token, not the next-token distribution that serving still needs.

## 代码 / The code

`vllm-project/vllm` — [`kv_cache_manager.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py)

```python
# When all tokens hit, recompute the last token to obtain logits.
max_cache_hit_length = request.num_tokens - 1
computed_blocks, num_new_computed_tokens, num_uncached = (
    self.coordinator.find_longest_cache_hit(
        request.block_hashes, max_cache_hit_length
    )
)
```

## 逐行讲解 / What's happening

命中长度上限是 `num_tokens - 1`，因此 cache 只跳过可安全复用的前缀；末 token 重新进 forward，输出才包含下一步采样所需的 logits。

The cap makes only the reusable prefix skippable. Sending the final token through forward yields the logits required for sampling.

## 类比 / The analogy

你可以沿用前面所有的草稿，但仍要读到最后一个词，才能决定下一句怎么接。

Reuse the draft up to the last word, but read that word to decide what comes next.

## 自己跑一遍 / Try it yourself

```python
tokens = [11, 12, 13, 14]
cached_prefix = tokens[:len(tokens) - 1]
to_execute = tokens[len(cached_prefix):]
print(cached_prefix, to_execute)  # [11, 12, 13] [14]
```

运行 / Run with `python try.py`。输出 / Expected output: 最后一个 token 单独保留执行。 / The final token remains to execute.

## 注意事项 / Caveats / when it breaks

- **块边界 / Block alignment**：实际 KV 分配按 block 进行，重算可能覆盖整块，不只一个 token。 / Allocation is block-based, so recomputation may cover a full block.
