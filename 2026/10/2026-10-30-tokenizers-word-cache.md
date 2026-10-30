---
date: 2026-10-30
topic: huggingface
source: huggingface
repo: huggingface/tokenizers
file: tokenizers/tk-encode/src/utils/word_cache.rs
permalink: https://github.com/huggingface/tokenizers/blob/514869d55cf0784d1253d105f8c97ced3810a520/tokenizers/tk-encode/src/utils/word_cache.rs#L70-L137
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, huggingface, tokenizer-cache]
---

# Tokenizers 词级 tag 缓存 / Tokenizers Word-Level Tag Cache

> **一句话 / In one line**: 先用 16 个 hash tag 快速筛候选，再读取完整槽位确认词是否命中。 / Scan 16 hash tags cheaply before touching full cache slots.

## 为什么重要 / Why this matters

反复 tokenization 常常重复同样的词。这个缓存把热路径设计成一条 cache line 的 tag 扫描，避免先读取较大的 payload。

Tokenization repeatedly sees the same words. This cache makes its hot path a one-cache-line tag scan before loading larger payloads.

## 代码 / The code

`huggingface/tokenizers` — [`word_cache.rs`](https://github.com/huggingface/tokenizers/blob/514869d55cf0784d1253d105f8c97ced3810a520/tokenizers/tk-encode/src/utils/word_cache.rs#L70-L137)

```rust
pub fn new(num_slots: usize) -> Self {
    let next_pow2 = num_slots.next_power_of_two();
    let n = next_pow2 + Self::WINDOW_SIZE;
    Self {
        cached_words: vec![WordCacheSlot::default(); n].into_boxed_slice(),
        quick_lookup: vec![0; n].into_boxed_slice(),
        placement_mask: (next_pow2 as u64) - 1,
        spilled_buffer: Vec::new(),
        ..Default::default()
    }
}
```

## 逐行讲解 / What's happening

1. 中文：容量上取整为 2 的幂，hash 位置可用 bit mask 计算。
   English: Capacity rounds to a power of two, so a bit mask computes placement.
2. 中文：槽位与一字节 tag 分开存，tag 扫描更紧凑。
   English: Slots and one-byte tags are stored separately, making tag scans compact.
3. 中文：长 token id 序列再放入共享溢出区。
   English: Long token-id sequences go to shared spill storage.

## 类比 / The analogy

像图书馆先扫书架标签，再只抽出标签相同的几本书核对书名。

Like scanning shelf labels in a library before pulling only the matching books to check their titles.

## 自己跑一遍 / Try it yourself

```python
cache = {}
def encode(word):
    if word not in cache:
        cache[word] = [ord(c) for c in word]
    return cache[word]
print(encode("cache"), encode("cache"), len(cache))
```

运行 / Run with: `python try.py`

预期输出 / Expected output: 同一词只建立一次条目。 / The same word creates one entry.

中文：示例省略了碰撞处理；真实实现不能只相信 tag。 English: The example omits collision handling; the real implementation cannot trust a tag alone.

## 注意事项 / Caveats / when it breaks

- 中文：缓存命中必须保留完整 key 校验。 / English: Cache hits must retain a full-key verification path.

## 延伸阅读 / Further reading

- [Tokenizers word cache](https://github.com/huggingface/tokenizers/blob/514869d55cf0784d1253d105f8c97ced3810a520/tokenizers/tk-encode/src/utils/word_cache.rs#L70-L137)
