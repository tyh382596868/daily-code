---
date: 2026-08-18
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/modules/t5.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/t5.py#L506-L513
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, wam, nano-wam, text-conditioning]
component: text-conditioning
variant: advanced
---

# Wan2.1 T5 context：先 padding 编码，再按 mask 剪回有效长度 / Wan2.1 T5 Context: Encode Padded Text, Then Trim by Mask Length

> **一句话 / In one line**: Wan2.1 的 T5 wrapper 批量 tokenize 文本、计算每条 prompt 的有效长度、跑文本模型，然后把 padding 后的 context 剪回变长列表。 / Wan2.1's T5 wrapper tokenizes text in a batch, computes each prompt's valid length, runs the text model, and trims padded context back into a variable-length list.

## 为什么重要 / Why this matters

视频 DiT 的 cross-attention 需要文本 context，但不同 prompt 长度不同。训练和推理为了吞吐会先 padding 成 batch；进入视频模型前再剪掉 padding，可以减少无效 attention token，也让 `context_lens` 语义更干净。

Video DiT cross-attention needs text context, but prompts have different lengths. Training and inference pad them into a batch for throughput; trimming before the video model removes useless attention tokens and keeps `context_lens` semantics clean.

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/modules/t5.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/t5.py#L506-L513)

```python
def __call__(self, texts, device):
    ids, mask = self.tokenizer(
        texts, return_mask=True, add_special_tokens=True)
    ids = ids.to(device)
    mask = mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    context = self.model(ids, mask)
    return [u[:v] for u, v in zip(context, seq_lens)]
```

## 逐行讲解 / What's happening

1. **第 506-508 行 / Lines 506-508**: 中文: tokenizer 同时返回 token ids 和 mask；mask 记录哪些位置是真 token。 / English: The tokenizer returns both token ids and a mask; the mask marks real tokens.
2. **第 509-510 行 / Lines 509-510**: 中文: ids 和 mask 一起搬到目标 device，避免模型和 mask 不在同一设备。 / English: Ids and mask move to the target device together so the model and mask agree.
3. **第 511 行 / Line 511**: 中文: `mask.gt(0).sum(dim=1)` 得到每条 prompt 的真实长度。 / English: `mask.gt(0).sum(dim=1)` computes the true length of each prompt.
4. **第 512-513 行 / Lines 512-513**: 中文: 文本模型仍然吃 padded batch，返回时再按各自长度切成 list。 / English: The text model still consumes a padded batch, but outputs are sliced back into a list by length.

## 在 nanoWAM 中的位置 / Where this fits in nanoWAM

这是 `text-conditioning` 的入口层：文本 encoder 可以批量跑，但视频 DiT 最好接收干净的变长 context。nanoWAM 可以复用这个模式，把 tokenizer/padding 和 DiT cross-attention 的长度合同隔离开。

This is the `text-conditioning` entry layer: the text encoder can run batched, while the video DiT receives clean variable-length context. nanoWAM can reuse this pattern to isolate tokenizer padding from the DiT cross-attention contract.

## 类比 / The analogy

像把不同长度的信件放进同样大小的信封统一运输；到达目的地后，拆掉空白填充，只把真正的信纸交给读信的人。

It is like shipping letters of different lengths in same-sized envelopes. At the destination, the empty padding is removed and only the real pages are handed to the reader.

## 自己跑一遍 / Try it yourself

```python
def trim_context(context, mask):
    seq_lens = [sum(1 for x in row if x > 0) for row in mask]
    return [row[:length] for row, length in zip(context, seq_lens)]

context = [["a", "b", "PAD"], ["x", "PAD", "PAD"], ["u", "v", "w"]]
mask = [[1, 1, 0], [1, 0, 0], [1, 1, 1]]
print(trim_context(context, mask))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[['a', 'b'], ['x'], ['u', 'v', 'w']]
```

模型内部可以批量处理，交给 DiT 前恢复每条 prompt 的真实长度。

The model can run batched internally, then restore each prompt's true length before the DiT sees it.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **packed sequence** / **packed sequences**: RNN 和序列模型常用长度把 padded batch 拆回变长表示。 / RNN and sequence models often use lengths to recover variable-sized items from padded batches.
- **varlen attention** / **variable-length attention**: 高性能 attention kernel 通常需要真实长度或 cumulative lengths。 / High-performance attention kernels often need true lengths or cumulative lengths.

## 注意事项 / Caveats / when it breaks

- **mask 必须和 tokenizer 一致** / **The mask must match tokenization**: 如果 mask 错了，context 会被提前截断或保留 padding。 / If the mask is wrong, context is truncated early or padding survives.
- **list 会丢失 dense batch 形状** / **The list loses dense batch shape**: 后续代码必须支持变长 list 或重新 pad。 / Downstream code must support variable-length lists or pad again.

## 延伸阅读 / Further reading

- [Wan2.1 source](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/t5.py#L506-L513)
