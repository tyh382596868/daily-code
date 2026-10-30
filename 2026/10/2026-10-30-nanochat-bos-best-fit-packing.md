---
date: 2026-10-30
topic: infrastructure
source: trending
repo: karpathy/nanochat
file: nanochat/dataloader.py
permalink: https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/dataloader.py#L73-L147
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, infrastructure, data-packing]
---

# nanochat BOS 对齐的最佳适配打包 / nanochat BOS-Aligned Best-Fit Packing

> **一句话 / In one line**: 每一行从 BOS 开始，优先塞入刚好合适的文档，最后才裁剪。 / Start every row at BOS, pack the best-fitting documents first, and crop only as a last resort.

## 为什么重要 / Why this matters

训练 token 的利用率与上下文边界通常互相拉扯。nanochat 用一个候选缓冲区寻找最大可放入文档，让每行保持完整开头，同时不留下 padding。

Token utilization and clean context boundaries usually pull in opposite directions. nanochat searches a candidate buffer for the largest fitting document, keeping clean row starts without padding.

## 代码 / The code

`karpathy/nanochat` — [`dataloader.py`](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/dataloader.py#L119-L147)

```python
best_idx, best_len = -1, 0
for i, doc in enumerate(doc_buffer):
    doc_len = len(doc)
    if doc_len <= remaining and doc_len > best_len:
        best_idx, best_len = i, doc_len

if best_idx >= 0:
    doc = doc_buffer.pop(best_idx)
    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc)
    pos += len(doc)
else:
    doc = doc_buffer.pop(min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i])))
    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining])
    pos += remaining
```

## 逐行讲解 / What's happening

1. 中文：线性扫描 buffer，挑不超过剩余空间的最长文档。
   English: Linearly scan the buffer for the longest document that fits.
2. 中文：有匹配就完整放入，保住文档边界。
   English: When one fits, insert it whole and preserve the document boundary.
3. 中文：没有匹配才裁最短文档填满行。
   English: Only when nothing fits, crop the shortest document to fill the row.

## 类比 / The analogy

像往行李箱塞衣服：先选最合适的一件，实在放不下再折一件小衣服补角落。

Like packing a suitcase: choose the best-fitting garment first, and only fold a small one when you must fill the final gap.

## 自己跑一遍 / Try it yourself

```python
docs, remaining = [[1, 2, 3], [4, 5], [6]], 3
fit = max((d for d in docs if len(d) <= remaining), key=len)
print(fit)
```

运行 / Run with: `python try.py`

预期输出 / Expected output: `[1, 2, 3]`

中文：这是最佳适配的核心；真实代码再加上 BOS、DDP 和断点续训。 English: This is the best-fit core; the real code adds BOS, DDP, and resumability.

## 注意事项 / Caveats / when it breaks

- 中文：buffer 太小会退化为贪心；太大会增加 CPU 开销。 / English: A tiny buffer becomes greedy; a huge one adds CPU overhead.

## 延伸阅读 / Further reading

- [nanochat dataloader](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/dataloader.py#L73-L147)
