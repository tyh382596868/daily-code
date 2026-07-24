---
date: 2026-07-24
topic: huggingface
source: huggingface
repo: huggingface/tokenizers
file: tokenizers/src/tokenizer/added_vocabulary.rs
permalink: https://github.com/huggingface/tokenizers/blob/b62132e4e0ec7518caba201408a680819dfdcd22/tokenizers/src/tokenizer/added_vocabulary.rs#L361-L542
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, huggingface, tokenizers, added-tokens, aho-corasick]
---

# tokenizers AddedToken：原文匹配和规范化匹配分两棵树 / tokenizers AddedToken: Split Raw and Normalized Matching into Two Tries

> **一句话 / In one line**: `AddedVocabulary` 为 `normalized=true/false` 的 token 分别构建 matcher，先在原文里切 special token，再在规范化文本里切普通 added token。 / `AddedVocabulary` builds separate matchers for `normalized=true/false` tokens, splitting raw special tokens first and normalized added tokens second.

## 为什么重要 / Why this matters

Tokenizer 既要保住 `<s>`、`[INST]` 这类 special token 的原样边界，又要允许用户输入 `Yesterday` 时匹配到经过 lowercase normalizer 的 `yesterday`。把两类 token 塞进同一个 matcher 会让 offset 和规范化顺序混乱；Hugging Face tokenizers 用两棵 Aho-Corasick 树把规则拆开。

A tokenizer must preserve exact boundaries for special tokens like `<s>` or `[INST]`, while also letting `Yesterday` match a lowercase-normalized `yesterday`. Putting both rules in one matcher makes offsets and normalization order messy. Hugging Face tokenizers separates them into two Aho-Corasick tries.

## 代码 / The code

`huggingface/tokenizers` — [`tokenizers/src/tokenizer/added_vocabulary.rs`](https://github.com/huggingface/tokenizers/blob/b62132e4e0ec7518caba201408a680819dfdcd22/tokenizers/src/tokenizer/added_vocabulary.rs#L361-L542)

```rust
    fn refresh_added_tokens(&mut self) -> Result<()> {
        let mut normalized_pairs: Vec<(&str, u32)> = Vec::new();
        let mut non_normalized_pairs: Vec<(&str, u32)> = Vec::new();
        for (id, token) in &self.added_tokens_map_r {
            if token.normalized {
                let pattern = self
                    .normalized_cache
                    .get(id)
                    .map(String::as_str)
                    .unwrap_or(token.content.as_str());
                normalized_pairs.push((pattern, *id));
            } else {
                non_normalized_pairs.push((token.content.as_str(), *id));
            }
        }

        self.split_trie = if non_normalized_pairs.is_empty() {
            None
        } else {
            Some(
                DoubleArrayAhoCorasickBuilder::new()
                    .match_kind(MatchKind::LeftmostLongest)
                    .build_with_values(non_normalized_pairs)
                    .map_err(|e| e.to_string())?,
            )
        };

        self.split_normalized_trie = if normalized_pairs.is_empty() {
            None
        } else {
            Some(
                DoubleArrayAhoCorasickBuilder::new()
                    .match_kind(MatchKind::LeftmostLongest)
                    .build_with_values(normalized_pairs)
                    .map_err(|e| e.to_string())?,
            )
        };

        Ok(())
    }

    pub fn extract_and_normalize<N: Normalizer>(
        &self,
        normalizer: Option<&N>,
        sequence: &str,
    ) -> PreTokenizedString {
        let mut pretokenized: PreTokenizedString = sequence.into();

        pretokenized
            .split(|_, sequence| Ok(self.split_with_indices(sequence, &self.split_trie)))
            .expect("AddedVocabulary bad split");

        pretokenized
            .split(|_, mut sequence| {
                normalizer.map(|n| n.normalize(&mut sequence));
                Ok(self.split_with_indices(sequence, &self.split_normalized_trie))
            })
            .expect("AddedVocabulary bad split");
```

## 逐行讲解 / What's happening

1. **第 364-376 行 / Lines 364-376 (`normalized_pairs`)**: 中文: `normalized=true` 的 token 用缓存里的规范化字符串匹配；`false` 的 token 保持原文。 English: `normalized=true` tokens match their cached normalized form; `false` tokens keep their raw content.
2. **第 379-391 行 / Lines 379-391 (`split_trie`)**: 中文: 原文 matcher 用 `LeftmostLongest`，重叠时取最左最长，避免短 token 抢走长 token。 English: the raw matcher uses `LeftmostLongest`, so a short token does not steal a longer overlap.
3. **第 393-405 行 / Lines 393-405 (`split_normalized_trie`)**: 中文: 规范化 matcher 单独构建，后面只作用在已经被 normalizer 处理过的片段上。 English: the normalized matcher is separate and later runs only on normalized pieces.
4. **第 522-524 行 / Lines 522-524 (first split)**: 中文: 先切 `normalized=false` token，典型例子是 special token。 English: `normalized=false` tokens are split first, usually special tokens.
5. **第 537-541 行 / Lines 537-541 (second split)**: 中文: 剩余片段先 normalize，再跑第二棵树。 English: remaining pieces are normalized before the second matcher runs.

## 类比 / The analogy

像机场安检先走一条“护照原件”通道，再走一条“姓名大小写忽略”通道。护照号不能改写，但旅客名字可以按系统规则统一大小写后再查。

It is like airport screening with one lane for exact passport IDs and another lane for case-normalized names. Passport numbers must stay exact; names can be normalized before lookup.

## 自己跑一遍 / Try it yourself

```python
raw_tokens = {"<s>": 1}
normalized_tokens = {"yesterday": 2}

text = "I saw <s> Yesterday"
pieces = []
for part in text.split("<s>"):
    if pieces:
        pieces.append(("<s>", raw_tokens["<s>"]))
    pieces.append((part, None))

out = []
for value, tok_id in pieces:
    if tok_id:
        out.append((value, tok_id))
    else:
        for word in value.split():
            out.append((word, normalized_tokens.get(word.lower())))
print(out)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[('I', None), ('saw', None), ('<s>', 1), ('Yesterday', 2)]
```

中文: `<s>` 按原文保留，`Yesterday` 则通过 lowercase 规则匹配到 `yesterday`。 English: `<s>` is preserved exactly, while `Yesterday` matches `yesterday` through lowercase normalization.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Transformers generation warpers** / **Transformers generation warpers**: 先保持 logits 的原始结构，再在受控阶段应用过滤规则。 / keep raw logits structure first, then apply filtering in a controlled stage.
- **Datasets streaming transforms** / **Datasets streaming transforms**: 先保留样本 offset/边界，再做批量 map。 / preserve sample boundaries before batched mapping.

## 注意事项 / Caveats / when it breaks

- **offset 是字节级 / Offsets are byte-based**: Rust 里切片要保证 byte offsets 对齐到合法 UTF-8 边界。 / Rust slicing must keep byte offsets on valid UTF-8 boundaries.
- **重建 matcher 有成本 / Rebuilding matchers costs time**: 注释里也提示大量 added tokens 时重建会慢。 / the source notes trie rebuilds can be slow with many added tokens.
- **special token 策略要一致 / Special-token policy must be consistent**: `encode_special_tokens` 会影响 special token 是否被直接匹配。 / `encode_special_tokens` changes whether special tokens are directly matched.

## 延伸阅读 / Further reading

- [tokenizers added_vocabulary.rs](https://github.com/huggingface/tokenizers/blob/b62132e4e0ec7518caba201408a680819dfdcd22/tokenizers/src/tokenizer/added_vocabulary.rs#L361-L542)

