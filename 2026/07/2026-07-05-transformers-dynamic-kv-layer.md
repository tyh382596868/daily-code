---
date: 2026-07-05
topic: huggingface
source: huggingface
repo: huggingface/transformers
file: src/transformers/cache_utils.py
permalink: https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, huggingface, transformers, kv-cache]
---

# Transformers DynamicLayer：KV cache 就是沿时间维拼接 / Transformers DynamicLayer: A KV Cache Is Concatenation Along Time

> **一句话 / In one line**: decode 时每步只产出新 KV，`DynamicLayer` 把它们沿 sequence 维追加起来。 / During decoding each step emits only new KV, and `DynamicLayer` appends them along the sequence dimension.

## 为什么重要 / Why this matters

KV cache 的直觉很简单：历史 token 的 key/value 已经算过，就不要再算。Transformers 的动态 cache layer 把这个想法压成一个小状态机：第一次保存张量，之后 `torch.cat` 到末尾，并用 `get_seq_length()` 告诉 attention 当前能看多长。

The KV-cache idea is simple: key/value tensors for historical tokens have already been computed, so do not compute them again. Transformers' dynamic cache layer compresses that into a small state machine: save tensors on the first step, then `torch.cat` new tensors at the end, and report the visible length through `get_seq_length()`.

## 代码 / The code

`huggingface/transformers` — [`src/transformers/cache_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py)

```python
class DynamicLayer(CacheLayerMixin):
    def __init__(self):
        self.keys: Optional[torch.Tensor] = None
        self.values: Optional[torch.Tensor] = None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.keys is None:
            self.keys = key_states
            self.values = value_states
        else:
            self.keys = torch.cat([self.keys, key_states], dim=-2)
            self.values = torch.cat([self.values, value_states], dim=-2)

        return self.keys, self.values

    def get_seq_length(self) -> int:
        if self.keys is None:
            return 0
        return self.keys.shape[-2]
```

## 逐行讲解 / What's happening

1. **两块状态 / Two pieces of state**: 中文: layer 只记 `keys` 和 `values`，没有额外索引结构，适合最通用的 eager decode。 / English: The layer stores only `keys` and `values`, with no extra indexing structure, which fits the most general eager decoding path.
2. **第一次直接接管 / First update takes ownership**: 中文: 第一步没有历史，直接把当前 KV 放进去。 / English: On the first step there is no history, so the layer simply stores the current KV tensors.
3. **后续沿 `dim=-2` 追加 / Later updates append on `dim=-2`**: 中文: 在 attention 约定里倒数第二维是 token 序列长度，拼接这里就等于扩展上下文。 / English: In the attention convention, the second-to-last dimension is sequence length; concatenating there extends the context.
4. **长度来自 shape / Length comes from shape**: 中文: `get_seq_length()` 不维护单独 counter，直接读 `keys.shape[-2]`，减少状态不同步的可能。 / English: `get_seq_length()` reads `keys.shape[-2]` instead of maintaining a separate counter, reducing synchronization bugs.

## 类比 / The analogy

像会议纪要：每次只写最新一句话，但把它粘到同一个文档末尾。下次发言时，参会者不用重新听整场会议，只看这份累积文档。

It is like meeting notes. Each turn writes only the newest sentence, but it is pasted at the end of the same document. The next speaker does not replay the whole meeting; they read the accumulated notes.

## 自己跑一遍 / Try it yourself

```python
import torch

keys = None
for step in range(3):
    new = torch.full((1, 2, 1, 4), float(step))
    keys = new if keys is None else torch.cat([keys, new], dim=-2)
    print(keys.shape, keys[0, 0, :, 0].tolist())
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```text
torch.Size([1, 2, 1, 4]) [0.0]
torch.Size([1, 2, 2, 4]) [0.0, 1.0]
torch.Size([1, 2, 3, 4]) [0.0, 1.0, 2.0]
```

中文: 每一步 batch/head/head_dim 都不变，只有 sequence 维增长。

English: Batch, head, and head-dim stay fixed at each step; only the sequence dimension grows.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **nanoGPT inference logits** / **nanoGPT inference logits**: 中文: 主干看完整历史，但每步只消费最后一个位置。 / English: The backbone sees the full history, but each decode step consumes only the newest position.
- **vLLM paged attention** / **vLLM paged attention**: 中文: 同样复用历史 KV，只是把 `cat` 换成页表和 block allocator。 / English: It reuses historical KV too, but replaces `cat` with page tables and block allocators.

## 注意事项 / Caveats / when it breaks

- **`torch.cat` 会重新分配 / `torch.cat` reallocates**: 中文: 这是简单通用版本；高吞吐 serving 通常要预分配或分页。 / English: This is the simple general version; high-throughput serving usually preallocates or pages memory.
- **维度约定必须一致 / Dimension conventions must match**: 中文: 如果模型把 sequence 维放到别处，`dim=-2` 就会拼错。 / English: If a model places the sequence dimension elsewhere, `dim=-2` appends the wrong axis.

## 延伸阅读 / Further reading

- Transformers cache utilities source linked above.
- Hugging Face generation cache documentation.
