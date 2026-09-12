---
date: 2026-08-30
topic: huggingface
source: huggingface
repo: huggingface/tokenizers
file: tokenizers/src/pre_tokenizers/byte_level.rs
permalink: https://github.com/huggingface/tokenizers/blob/d5827816baedcbf1cb5b452dea8048150b6872df/tokenizers/src/pre_tokenizers/byte_level.rs#L119-L172
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, tokenizers, byte-level-bpe]
---

# tokenizers ByteLevel：每个 byte 都有路可走 / tokenizers ByteLevel: Every Byte Has a Path

> **一句话 / In one line**: `ByteLevel` 先可选地补前导空格，再把文本拆分并映射到可逆 Unicode 字符，decode 时再还原成原始 byte。 / `ByteLevel` optionally adds a leading space, splits text, maps bytes into reversible Unicode characters, and decodes them back into original bytes.

## 为什么重要 / Why this matters

GPT 系列 tokenizer 的一个关键优势是“不认识的字符也能表示”。Byte-level 方案不把输入限制在词表能覆盖的 Unicode 字符上，而是退到底层 byte，再用一套安全字符表承载它们。

A key advantage of GPT-style tokenizers is that even unfamiliar characters remain representable. The byte-level approach does not restrict inputs to Unicode characters covered by the vocabulary; it falls back to bytes and carries them through a safe character table.

## 代码 / The code

`huggingface/tokenizers` — [`tokenizers/src/pre_tokenizers/byte_level.rs`](https://github.com/huggingface/tokenizers/blob/d5827816baedcbf1cb5b452dea8048150b6872df/tokenizers/src/pre_tokenizers/byte_level.rs#L119-L172)

```rust
impl PreTokenizer for ByteLevel {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        let re_ref: &SysRegex = &RE;
        pretokenized.split(|_, mut normalized| {
            if self.add_prefix_space && !normalized.get().starts_with(' ') {
                normalized.prepend(" ");
            }
            if self.use_regex {
                normalized.split(re_ref, SplitDelimiterBehavior::Isolated)
            } else {
                Ok(vec![normalized])
            }
        })?;
        pretokenized.normalize(|normalized| {
            let s = normalized.get();
            let mut transformations: Vec<(char, isize)> = Vec::with_capacity(s.len());
            for (i, cur_char) in s.char_indices() {
                let size = cur_char.len_utf8();
                transformations.extend(
                    s.as_bytes()[i..i + size]
                        .iter()
                        .enumerate()
                        .map(|(i, b)| (BYTES_CHAR[b], isize::from(i > 0))),
                );
            }
            normalized.transform(transformations, 0);
            Ok(())
        })
    }
}

/// As a `Decoder`, `ByteLevel` is in charge of converting any byte-level characters to their
/// unicode counterpart, before merging everything back into a single String.
/// This decoder will consume the tokens and merge them in one step to alleviate
/// the fact that single token decoded might be a byte not representable as
/// as String.
impl Decoder for ByteLevel {
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        let toks = tokens
            .into_iter()
            .flat_map(|t| {
                t.chars()
                    .try_fold(vec![], |mut acc, c| {
                        CHAR_BYTES.get(&c).map(|b| {
                            acc.push(*b);
                            acc
                        })
                    })
                    .unwrap_or_else(|| t.as_bytes().to_vec())
            })
            .collect::<Vec<u8>>();
        Ok(vec![String::from_utf8_lossy(&toks).to_string()])
    }
}
```

## 逐行讲解 / What's happening

1. **第 121-131 行 / Lines 121-131**:
   - 中文: 正则先把文本切成片段；`add_prefix_space` 会在开头补一个空格，让词首和词中行为更一致。
   - English: The regex splits text into pieces; `add_prefix_space` prepends a space so beginning-of-text and middle-of-text behavior are more consistent.
2. **第 132-145 行 / Lines 132-145**:
   - 中文: 每个 Unicode 字符被拆回它的 UTF-8 byte，再用 `BYTES_CHAR` 映射到 tokenizer 可处理的字符。
   - English: Each Unicode character is decomposed into its UTF-8 bytes, then each byte is mapped through `BYTES_CHAR` into tokenizer-friendly characters.
3. **第 150-170 行 / Lines 150-170**:
   - 中文: decode 反向查 `CHAR_BYTES`，把 byte-level 字符重新收集成 byte 数组。
   - English: Decoding looks up `CHAR_BYTES` in reverse and collects byte-level characters back into a byte array.
4. **第 170 行 / Line 170**:
   - 中文: 最后用 `String::from_utf8_lossy` 合并，保证就算局部 byte 不合法也能产出字符串。
   - English: `String::from_utf8_lossy` produces a string even when a local byte sequence is not valid UTF-8.

## 类比 / The analogy

这像国际快递给每种货物都贴统一条码：不管里面是书、杯子还是奇怪零件，运输系统只认条码；到终点再按条码还原成原货物。

It is like an international shipping system that gives every item a standardized barcode. Whether the item is a book, a cup, or an odd spare part, the logistics system only sees barcodes; the destination maps them back to the original goods.

## 自己跑一遍 / Try it yourself

```python
def encode_bytes(text):
    return [b + 1000 for b in text.encode("utf-8")]

def decode_bytes(ids):
    return bytes(i - 1000 for i in ids).decode("utf-8", errors="replace")

ids = encode_bytes("hi 你")
print(ids)
print(decode_bytes(ids))
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
[1104, 1105, 1032, 1228, 1189, 1160]
hi 你
```

真实实现不会直接用 `b + 1000`，但核心思想一样：先把任何文本变成 byte，再映射到可训练的符号空间。

The real implementation does not use `b + 1000`, but the core idea is the same: turn any text into bytes, then map bytes into a trainable symbol space.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **GPT-2 BPE** / **GPT-2 BPE**: byte-level BPE 让模型不需要 `<unk>` 就能覆盖输入字符。 / Byte-level BPE lets the model cover input characters without relying on `<unk>`.
- **Metaspace pre-tokenizer** / **Metaspace pre-tokenizer**: 也是先把不可见结构变成可见符号，再交给子词模型。 / It also turns invisible structure into visible symbols before subword modeling.
- **SentencePiece byte fallback** / **SentencePiece byte fallback**: 未登录字符可以拆成 byte token。 / Unknown characters can fall back to byte tokens.

## 注意事项 / Caveats / when it breaks

- **offset 会变复杂** / **Offsets become harder**: 一个 Unicode 字符可能对应多个 byte-level 字符，必须维护原文对齐。 / One Unicode character can map to multiple byte-level characters, so source alignment must be maintained.
- **lossy decode 是兜底** / **Lossy decode is a fallback**: 如果 token 边界切坏了 byte 序列，decode 会替换非法片段。 / If token boundaries split an invalid byte sequence, decoding replaces the broken pieces.
- **前导空格影响词表** / **Leading spaces affect the vocabulary**: 是否补空格会改变训练出的 token 频率和词首 token。 / Adding a prefix space changes learned token frequencies and beginning-of-word tokens.

## 延伸阅读 / Further reading

- tokenizers `ByteLevel`: https://github.com/huggingface/tokenizers/blob/d5827816baedcbf1cb5b452dea8048150b6872df/tokenizers/src/pre_tokenizers/byte_level.rs
- Hugging Face tokenizers docs: https://huggingface.co/docs/tokenizers/
