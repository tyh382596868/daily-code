---
date: 2026-07-21
topic: huggingface
source: huggingface
repo: huggingface/transformers
file: src/transformers/cache_utils.py
permalink: https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L2395-L2446
difficulty: intermediate
read_time: ~9 min
tags: [code-of-the-day, huggingface, transformers, kv-cache, generation]
---

# Transformers DynamicLayer：KV cache 默认就是沿时间维追加 / Transformers DynamicLayer: The Default KV Cache Appends Along Time

> **一句话 / In one line**: `DynamicLayer` 第一次看到 K/V 时延迟初始化空 cache，之后每步把新 K/V 沿 `seq_len` 维拼到旧 cache 后面。 / `DynamicLayer` lazily initializes an empty cache on the first K/V tensors, then appends new K/V along the `seq_len` dimension at every step.

## 为什么重要 / Why this matters

自回归生成的性能关键是不要重复算历史 token 的 K/V。Transformers 把默认 cache layer 做得非常直接：cache tensor 的形状就是 `[batch, heads, seq_len, head_dim]`，新 token 只需要沿倒数第二维追加。

The key performance trick in autoregressive generation is avoiding repeated K/V computation for old tokens. Transformers keeps the default cache layer direct: the cache tensor shape is `[batch, heads, seq_len, head_dim]`, and new tokens append along the second-to-last dimension.

## 代码 / The code

`huggingface/transformers` — [`src/transformers/cache_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L2395-L2446)

```python
# Simplified teaching slice, not a verbatim copy.
class DynamicLayer:
    def lazy_initialization(self, key_states, value_states):
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = empty_cache(dtype=self.dtype, device=self.device)
        self.values = empty_cache(dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(self, key_states, value_states):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        self.keys = concat([self.keys, key_states], dim=-2)
        self.values = concat([self.values, value_states], dim=-2)
        return self.keys, self.values
```

## 逐行讲解 / What's happening

1. **第一次才建 cache / Allocate on first use**: 中文: dtype 和 device 从真实 K/V 继承，避免构造时猜环境。 English: dtype and device come from real K/V tensors, avoiding guesses at construction.
2. **keys 和 values 对称 / Keys and values stay symmetric**: 中文: 两个 cache 生命周期相同，只是内容不同。 English: both caches share the same lifecycle, only their contents differ.
3. **追加维度是 `-2` / Append dimension is `-2`**: 中文: K/V 的倒数第二维是时间长度。 English: the second-to-last dimension is sequence length.
4. **返回完整 cache / Return the full cache**: 中文: attention 下一步直接用旧 token 加新 token 的完整 K/V。 English: attention receives complete K/V for old plus new tokens.
5. **动态简单但会增长 / Dynamic is simple but grows**: 中文: 长对话会持续占显存，滑窗或静态 cache 是其他折中。 English: long conversations keep consuming memory; sliding-window or static caches are other tradeoffs.

## 类比 / The analogy

像课堂笔记本：第一节课才拿出本子，每来一页新笔记就贴到最后。复习时读整本，不用重新听前面的课。

It is like a class notebook: you open it at the first lecture and append each new page at the end. When reviewing, you read the notebook instead of replaying earlier lectures.

## 自己跑一遍 / Try it yourself

```python
class TinyCache:
    def __init__(self):
        self.keys = []
        self.values = []
    def update(self, keys, values):
        self.keys += keys
        self.values += values
        return self.keys, self.values

cache = TinyCache()
print(cache.update(["k0", "k1"], ["v0", "v1"]))
print(cache.update(["k2"], ["v2"]))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
(['k0', 'k1'], ['v0', 'v1'])
(['k0', 'k1', 'k2'], ['v0', 'v1', 'v2'])
```

## 注意事项 / Caveats / when it breaks

- **拼接会重新分配 / Concatenation reallocates**: 动态 cache 简单，但频繁 cat 可能不如预分配 cache 友好。 / dynamic cache is simple, but repeated concatenation may be less friendly than preallocation.
- **beam search 要重排 batch / Beam search must reorder batches**: cache 需要随 beam index 一起重排。 / cache tensors must follow beam indices.
- **compile 可能偏好静态 cache / Compile may prefer static cache**: 固定地址和固定 shape 更适合 CUDA graph。 / fixed addresses and shapes are better for CUDA graphs.

## 延伸阅读 / Further reading

- [Transformers cache_utils.py DynamicLayer](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py#L2395-L2446)
- [Transformers generation cache docs](https://huggingface.co/docs/transformers/main/en/kv_cache)
