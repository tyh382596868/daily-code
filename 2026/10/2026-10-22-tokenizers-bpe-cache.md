---
date: 2026-10-22
topic: huggingface
source: huggingface
repo: huggingface/tokenizers
file: tokenizers/src/models/bpe/model.rs
permalink: https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/models/bpe/model.rs
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, bpe]
---

# Tokenizers BPE 缓存：把重复切词变成查表 / Tokenizers BPE Cache: Turn Repeated Segmentation into a Lookup

> **一句话 / In one line**: BPE 合并昂贵但重复率高，缓存完整词的编码结果比微调单次 merge 更有效。 / BPE merges are costly but repeated; caching whole-word encodings often beats optimizing one merge.

## 为什么重要 / Why this matters

自然语言里常见词会反复出现。Tokenizers 的 BPE 模型把已经编码过的词存入缓存；下一次输入相同词时直接复用 token 序列，同时保持词表和 merge 规则不变。

Common words recur in natural language. The BPE model caches a completed word encoding, so later occurrences reuse token IDs without changing vocabulary or merge rules.

## 代码 / The code

`huggingface/tokenizers` — [`model.rs`](https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/models/bpe/model.rs)

```rust
if let Some(ref cache) = self.cache {
    if let Some(word) = cache.get(sequence) {
        return Ok(word);
    }
}
let word = self.merge_word(sequence)?;
if let Some(ref cache) = self.cache { cache.set(sequence.to_owned(), word.clone()); }
Ok(word)
```

## 逐行讲解 / What's happening

1. **先读缓存 / Read first**：命中时不触碰 merge 优先队列。 / A hit avoids the merge priority queue entirely.
2. **只缓存完成品 / Cache completed work**：缓存的是稳定 token 序列，不是中间 pair。 / The cache stores stable token sequences, not intermediate pairs.

## 类比 / The analogy

像咖啡店记住常客的固定订单：规则没变，但第二次不必重新逐项确认。 / Like a café remembering a regular’s order: policy is unchanged, repeated work disappears.

## 自己跑一遍 / Try it yourself

```python
cache = {}
def encode(word):
    if word not in cache: cache[word] = list(word)
    return cache[word]
print(encode("hello"), encode("hello"))
```

运行 / Run with: `python try.py`  
预期输出 / Expected output: 两次得到同一编码，第二次走缓存。 / Both calls produce the same encoding; the second is cached.

## 注意事项 / Caveats / when it breaks

- **容量 / Capacity**：无界缓存会被长尾词耗尽内存。 / An unbounded cache lets long-tail vocabulary consume memory.
