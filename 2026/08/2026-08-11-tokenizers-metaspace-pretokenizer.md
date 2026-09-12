---
date: 2026-08-11
topic: huggingface
source: huggingface
repo: huggingface/tokenizers
file: tokenizers/src/pre_tokenizers/metaspace.rs
permalink: https://github.com/huggingface/tokenizers/blob/447890f84deab25794ccfdee2a877901b6893569/tokenizers/src/pre_tokenizers/metaspace.rs#L122-L172
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, tokenizers]
---

# tokenizers Metaspace：把空格变成可见字符 / tokenizers Metaspace: Make Spaces Visible

> **一句话 / In one line**: `Metaspace` 先把空格替换成 `▁`，再按这个字符切分，并在 decode 时把它还原为空格。 / `Metaspace` replaces spaces with `▁`, splits around that marker, and turns it back into spaces during decoding.

## 为什么重要 / Why this matters

子词 tokenizer 不能把“词前空格”当成无关细节。`hello` 和 ` hello` 在语言模型里通常不是同一个上下文。Metaspace 的做法很直接：把不可见空格变成普通字符，让后续 BPE/Unigram 能把“带词边界的 token”学进去。

Subword tokenizers cannot treat a leading word boundary as irrelevant. `hello` and ` hello` often mean different contexts to a language model. Metaspace makes that boundary visible by turning spaces into a normal character that BPE or Unigram can learn.

## 代码 / The code

`huggingface/tokenizers` — [`tokenizers/src/pre_tokenizers/metaspace.rs`](https://github.com/huggingface/tokenizers/blob/447890f84deab25794ccfdee2a877901b6893569/tokenizers/src/pre_tokenizers/metaspace.rs#L122-L172)

```rust
impl PreTokenizer for Metaspace {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, mut normalized| {
            normalized.replace(' ', &self.str_rep)?;
            match self.prepend_scheme {
                PrependScheme::Always => {
                    if !normalized.get().starts_with(self.replacement) {
                        normalized.prepend(&self.str_rep);
                    }
                }
                PrependScheme::First => {
                    if !normalized.get().starts_with(self.replacement)
                        && normalized.offsets_original().0 == 0
                    {
                        normalized.prepend(&self.str_rep);
                    }
                }
                PrependScheme::Never => {}
            };
            if self.split {
                normalized.split(self.replacement, SplitDelimiterBehavior::MergedWithNext)
            } else {
                Ok(vec![normalized])
            }
        })
    }
}

impl Decoder for Metaspace {
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        Ok(tokens
            .iter()
            .enumerate()
            .map(|(i, token)| {
                token
                    .chars()
                    .flat_map(|c| {
                        if c == self.replacement {
                            if i == 0 && self.prepend_scheme != PrependScheme::Never {
                                None
                            } else {
                                Some(' ')
                            }
                        } else {
                            Some(c)
                        }
                    })
                    .collect::<String>()
            })
            .collect())
    }
}
```

## 逐行讲解 / What's happening

1. **第 124-125 行 / Lines 124-125 (space replacement)**:
   - 中文: 每个 normalized segment 先把普通空格替换成 `self.str_rep`，默认就是 `▁`。
   - English: Each normalized segment first replaces ordinary spaces with `self.str_rep`, usually `▁`.
2. **第 126-140 行 / Lines 126-140 (prefix policy)**:
   - 中文: `Always` 总是补词首标记，`First` 只在原始 offset 为 0 的第一个 split 补，`Never` 不补。
   - English: `Always` adds a prefix marker, `First` only adds it at original offset 0, and `Never` skips it.
3. **第 141-145 行 / Lines 141-145 (split behavior)**:
   - 中文: 切分时用 `MergedWithNext`，让 `▁word` 这种词边界和后面的词绑在一起。
   - English: Splitting uses `MergedWithNext`, so a marker like `▁word` stays attached to the following word.
4. **第 150-172 行 / Lines 150-172 (decode)**:
   - 中文: decode 时第一个前缀标记通常被删除，后面的标记还原成空格。
   - English: During decoding, the first prefix marker is usually dropped, while later markers become spaces.

## 类比 / The analogy

像在稿纸上把每个词前的空格改成小旗子。排版时小旗子很有用，最后印刷时再把旗子换回空格。

It is like replacing word-leading spaces in a manuscript with little flags. The flags help layout and segmentation; final printing turns them back into spaces.

## 自己跑一遍 / Try it yourself

```python
def metaspace(text, prepend=True):
    s = text.replace(" ", "_")
    if prepend and not s.startswith("_"):
        s = "_" + s
    return [p for p in s.split("_") if p], s

pieces, marked = metaspace("hello world")
decoded = marked[1:].replace("_", " ")
print(marked)
print(pieces)
print(decoded)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
_hello_world
['hello', 'world']
hello world
```

真实实现保留 offset 对齐和 split 对象；这个 toy 只展示“空格可见化”的核心想法。

The real implementation preserves offsets and split objects; this toy only shows the core idea of making spaces visible.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **SentencePiece** / **SentencePiece**: `▁` 也是常见的词边界标记。 / `▁` is also the common word-boundary marker in SentencePiece.
- **Byte-level BPE** / **Byte-level BPE**: GPT-style tokenizer 也会把空格编码成可学习的 token 形态。 / GPT-style tokenizers also encode spaces as learnable token-visible forms.

## 注意事项 / Caveats / when it breaks

- **offset 很重要** / **Offsets matter**: 只做字符串替换会丢掉原文到 token 的对齐，真实库必须维护 `NormalizedString` offset。 / Plain string replacement loses source-token alignment; the real library must preserve `NormalizedString` offsets.
- **prefix 规则影响 vocab** / **Prefix policy changes the vocab**: `Always`、`First`、`Never` 训练出的 token 分布不同。 / `Always`, `First`, and `Never` produce different token distributions.

## 延伸阅读 / Further reading

- [tokenizers `Metaspace`](https://github.com/huggingface/tokenizers/blob/447890f84deab25794ccfdee2a877901b6893569/tokenizers/src/pre_tokenizers/metaspace.rs#L122-L172)

