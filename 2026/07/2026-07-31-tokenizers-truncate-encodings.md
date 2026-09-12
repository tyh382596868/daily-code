---
date: 2026-07-31
topic: huggingface
source: huggingface
repo: huggingface/tokenizers
file: tokenizers/src/utils/truncation.rs
permalink: https://github.com/huggingface/tokenizers/blob/b62132e4e0ec7518caba201408a680819dfdcd22/tokenizers/src/utils/truncation.rs#L70-L162
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, tokenizers, truncation]
---

# tokenizers truncation：max_length 最后会变成两段预算 / tokenizers Truncation: max_length Becomes Two Budgets

> **一句话 / In one line**: `truncate_encodings` 把一个全局 `max_length` 分配给单句或句对，并按策略截断。 / `truncate_encodings` turns one global `max_length` into budgets for one or two encodings, then truncates by strategy.

## 为什么重要 / Why this matters

Tokenizer 截断看似只是“砍到最大长度”，但句对任务里要决定砍第一句、第二句，还是两边都砍。这里的 Rust 代码把 `LongestFirst`、`OnlyFirst`、`OnlySecond` 三种语义落成明确预算。

Tokenizer truncation looks like “cut to max length,” but pair tasks must decide whether to cut the first sequence, the second sequence, or both. This Rust code turns `LongestFirst`, `OnlyFirst`, and `OnlySecond` into exact budgets.

## 代码 / The code

`huggingface/tokenizers` — [`tokenizers/src/utils/truncation.rs`](https://github.com/huggingface/tokenizers/blob/b62132e4e0ec7518caba201408a680819dfdcd22/tokenizers/src/utils/truncation.rs#L70-L162)

```rust
pub fn truncate_encodings(
    mut encoding: Encoding,
    mut pair_encoding: Option<Encoding>,
    params: &TruncationParams,
) -> Result<(Encoding, Option<Encoding>)> {
    if params.max_length == 0 {
        encoding.truncate(0, params.stride, params.direction);
        if let Some(other_encoding) = pair_encoding.as_mut() {
            other_encoding.truncate(0, params.stride, params.direction);
        }
        return Ok((encoding, pair_encoding));
    }

    let total_length = encoding.get_ids().len()
        + pair_encoding
            .as_ref()
            .map(|e| e.get_ids().len())
            .unwrap_or(0);
    let to_remove = if total_length > params.max_length {
        total_length - params.max_length
    } else {
        return Ok((encoding, pair_encoding));
    };

    match params.strategy {
        TruncationStrategy::LongestFirst => {
            if let Some(other_encoding) = pair_encoding.as_mut() {
                let mut n1 = encoding.get_ids().len();
                let mut n2 = other_encoding.get_ids().len();
                let mut swap = false;
                if n1 > n2 {
                    swap = true;
                    mem::swap(&mut n1, &mut n2);
                }
                if n1 > params.max_length {
                    n2 = n1;
                } else {
                    n2 = cmp::max(n1, params.max_length - n1);
                }
                if n1 + n2 > params.max_length {
                    n1 = params.max_length / 2;
                    n2 = n1 + params.max_length % 2;
                }
                if swap {
                    mem::swap(&mut n1, &mut n2);
                }
                encoding.truncate(n1, params.stride, params.direction);
                other_encoding.truncate(n2, params.stride, params.direction);
            } else {
                encoding.truncate(total_length - to_remove, params.stride, params.direction);
            }
        }
        TruncationStrategy::OnlyFirst | TruncationStrategy::OnlySecond => {
            let target = if params.strategy == TruncationStrategy::OnlyFirst {
                Ok(&mut encoding)
            } else if let Some(encoding) = pair_encoding.as_mut() {
                Ok(encoding)
            } else {
                Err(Box::new(TruncationError::SecondSequenceNotProvided))
            }?;
            let target_len = target.get_ids().len();
            if target_len > to_remove {
                target.truncate(target_len - to_remove, params.stride, params.direction);
            } else {
                return Err(Box::new(TruncationError::SequenceTooShort));
            }
        }
    }
    Ok((encoding, pair_encoding))
}
```

## 逐行讲解 / What's happening

1. **第 75-81 行 / Lines 75-81 (`max_length == 0`)**:
   - 中文: 零长度是特殊情况，两个 encoding 都直接截成空。
   - English: Zero length is a special case; both encodings are truncated to empty.
2. **第 83-92 行 / Lines 83-92 (`to_remove`)**:
   - 中文: 先算总长度；如果没超长，函数直接返回，不做额外工作。
   - English: It first computes total length; if nothing exceeds the limit, the function returns immediately.
3. **第 94-139 行 / Lines 94-139 (`LongestFirst`)**:
   - 中文: 先把较短段记为 `n1`，再决定较长段还能留多少；必要时两段平分预算。
   - English: It treats `n1` as the shorter side, then decides how much the longer side may keep; if needed, both sides split the budget.
4. **第 144-158 行 / Lines 144-158 (`OnlyFirst/OnlySecond`)**:
   - 中文: 指定只砍一边时，目标序列必须足够长，否则返回错误而不是偷偷砍另一边。
   - English: When only one side may be truncated, that side must be long enough; otherwise the function errors instead of silently cutting the other side.

## 类比 / The analogy

这像两个乘客共享一个行李重量上限：`LongestFirst` 会先让大箱子减重；`OnlyFirst` 则规定只能从第一个人的箱子里拿东西。

It is like two passengers sharing one baggage weight limit: `LongestFirst` asks the larger suitcase to lose weight first; `OnlyFirst` says only the first passenger’s bag may be reduced.

## 自己跑一遍 / Try it yourself

```python
def longest_first(a, b, max_len):
    n1, n2, swapped = len(a), len(b), False
    if n1 > n2:
        n1, n2, swapped = n2, n1, True
    n2 = n1 if n1 > max_len else max(n1, max_len - n1)
    if n1 + n2 > max_len:
        n1 = max_len // 2
        n2 = n1 + max_len % 2
    if swapped:
        n1, n2 = n2, n1
    return a[:n1], b[:n2]

print(longest_first([1, 2, 3], list("abcdef"), 7))
print(longest_first(list("abcdef"), [1, 2, 3], 7))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
([1, 2, 3], ['a', 'b', 'c', 'd'])
(['a', 'b', 'c', 'd'], [1, 2, 3])
```

两次输入顺序不同，但较短序列都被完整保留，较长序列承担截断。

The input order differs, but the shorter sequence is preserved and the longer one pays the truncation cost.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Transformers pair classification** / **Transformers pair classification**: NLI、QA 常把 question 和 context 作为句对，需要明确哪边能被截断。 / NLI and QA often use question-context pairs and must decide which side may be truncated.
- **Streaming datasets** / **Streaming datasets**: 流式预处理也会把全局 token budget 分配给多个字段。 / Streaming preprocessing also allocates one token budget across multiple fields.

## 注意事项 / Caveats / when it breaks

- **stride 不是免费午餐** / **Stride is not free**: stride 会保留重叠窗口，增加 overflow 片段数量。 / Stride keeps overlapping windows and increases the number of overflow chunks.
- **错误比静默截断好** / **Errors beat silent truncation**: `OnlySecond` 没有第二段时返回错误，避免训练数据悄悄变形。 / `OnlySecond` errors when no second sequence exists, preventing silent data shape changes.

## 延伸阅读 / Further reading

- [tokenizers truncation source permalink](https://github.com/huggingface/tokenizers/blob/b62132e4e0ec7518caba201408a680819dfdcd22/tokenizers/src/utils/truncation.rs#L70-L162)
