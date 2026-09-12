---
date: 2026-08-10
topic: huggingface
source: huggingface
repo: huggingface/trl
file: trl/trainer/utils.py
permalink: https://github.com/huggingface/trl/blob/2396dfe5d2be7b18c0b615d80957d64ecdeb7cc0/trl/trainer/utils.py#L480-L524
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, trl, logprobs]
---

# TRL selective log-softmax：只算被选 token 的 logprob / TRL Selective Log-Softmax: Compute Logprobs Only for Selected Tokens

> **一句话 / In one line**: `selective_log_softmax` 避免完整保存 `log_softmax` 大矩阵，只把目标 token 的 logprob 取出来。 / `selective_log_softmax` avoids materializing a full `log_softmax` matrix and returns only the selected token logprobs.

## 为什么重要 / Why this matters

偏好优化和 RLHF 训练经常只关心“模型给已采样 token 的概率”。如果先对整个 vocabulary 做 `log_softmax` 再 `gather`，中间张量会很大。TRL 这里把公式拆开：选中的 logits 减去每行的 `logsumexp`，峰值显存更低。

Preference optimization and RLHF often only need the logprob of already sampled tokens. Computing full-vocabulary `log_softmax` and then gathering wastes memory. TRL decomposes the formula: selected logits minus row-wise `logsumexp`.

## 代码 / The code

`huggingface/trl` — [`trl/trainer/utils.py`](https://github.com/huggingface/trl/blob/2396dfe5d2be7b18c0b615d80957d64ecdeb7cc0/trl/trainer/utils.py#L480-L524)

```python
def selective_log_softmax(logits, index) -> torch.Tensor:
    squeeze = index.ndim == logits.ndim - 1
    if squeeze:
        index = index.unsqueeze(-1)

    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values.unsqueeze(-1)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficient approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index, strict=True):
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)

    if squeeze:
        per_token_logps = per_token_logps.squeeze(-1)

    return per_token_logps
```

## 逐行讲解 / What's happening

1. **第 503-505 行 / Lines 503-505 (shape normalization)**:
   - 中文: 如果 `index` 少一个 gather 维度，就先 `unsqueeze(-1)`，最后再挤回去。
   - English: If `index` lacks the gather dimension, it is temporarily unsqueezed and later squeezed back.
2. **第 507-511 行 / Lines 507-511 (stable formula)**:
   - 中文: 对 float32/float64，`log_softmax(x_i) = x_i - logsumexp(x)`，所以只需要 gather 目标 logits。
   - English: For float32/float64, `log_softmax(x_i) = x_i - logsumexp(x)`, so only target logits need gathering.
3. **第 512-519 行 / Lines 512-519 (dtype fallback)**:
   - 中文: bf16 下直接用拆公式可能不稳定，因此退回逐行 `F.log_softmax`。
   - English: For lower precision, the decomposed formula may be less stable, so it falls back to row-wise `F.log_softmax`.
4. **第 521-524 行 / Lines 521-524 (restore shape)**:
   - 中文: 返回形状和原始 `index` 对齐，调用方不用关心内部是否扩维。
   - English: The return shape matches the original `index`, hiding the internal shape adjustment.

## 类比 / The analogy

像查一本巨大电话簿：你只需要三个号码，不必把整本电话簿复印一遍再用荧光笔标出来。

It is like looking up three numbers in a huge phone book: you do not need to photocopy the whole book before highlighting the entries.

## 自己跑一遍 / Try it yourself

```python
import math

logits = [[1.0, 2.0, 4.0], [3.0, 1.0, 0.0]]
indices = [2, 0]
out = []
for row, idx in zip(logits, indices):
    lse = math.log(sum(math.exp(x) for x in row))
    out.append(round(row[idx] - lse, 4))
print(out)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[-0.1698, -0.1698]
```

这个最小版没有构造完整的 `log_softmax` 表，只计算每行被选中的 token。

This minimal version never builds the full `log_softmax` table; it computes only the selected token per row.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **DPO/GRPO token logprobs** / **DPO/GRPO token logprobs**: 训练目标只需要答案 token 的概率。 / The objective only needs probabilities of answer tokens.
- **Sampled softmax** / **Sampled softmax**: 大 vocabulary 场景常常只计算候选子集。 / Large-vocabulary training often computes only a candidate subset.

## 注意事项 / Caveats / when it breaks

- **低精度要保守** / **Be conservative with low precision**: bf16/fp16 下数值稳定性比省显存更重要。 / In bf16/fp16, numerical stability can matter more than memory savings.
- **循环牺牲吞吐** / **Loops trade throughput for memory**: 逐行 loop 降低峰值显存，但可能慢于一次大 kernel。 / Row-wise loops reduce peak memory but can be slower than one large kernel.

## 延伸阅读 / Further reading

- [TRL `selective_log_softmax`](https://github.com/huggingface/trl/blob/2396dfe5d2be7b18c0b615d80957d64ecdeb7cc0/trl/trainer/utils.py#L480-L524)
