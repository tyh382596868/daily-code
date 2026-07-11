---
date: 2026-07-12
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/modules/model.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py#L171-L194
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, wam, text-conditioning]
build_role: text-conditioning cross-attention for a from-scratch nanoWAM
---

# Wan2.1 T2V cross-attention：视频 token 查询文本 token / Wan2.1 T2V Cross-Attention: Video Tokens Query Text Tokens

> **一句话 / In one line**: Wan2.1 的 T2V cross-attention 用视频 latent 做 query，用文本 context 做 key/value。 / Wan2.1 T2V cross-attention uses video latents as queries and text context as keys and values.

## 为什么重要 / Why this matters

世界动作模型或视频扩散模型需要把“我要生成什么”注入到每个视频 token。自注意力让视频 token 彼此沟通，cross-attention 则让视频 token 去读取文本条件。这段代码把两者拆得很清楚：`x` 只产生 query，`context` 产生 key/value。

A world-action or video diffusion model must inject "what to generate" into every video token. Self-attention lets video tokens talk to each other; cross-attention lets video tokens read conditioning text. This code makes the split explicit: `x` produces queries, while `context` produces keys and values.

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/modules/model.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py#L171-L194)

```python
class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        b, n, d = x.size(0), self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        x = flash_attention(q, k, v, k_lens=context_lens)

        x = x.flatten(2)
        x = self.o(x)
        return x
```

## 逐行讲解 / What's happening

1. **视频 token 只出 query / Video tokens only produce queries**:
   - 中文: `q = self.q(x)` 表示每个视频 latent 在问：“我现在应该看哪些文本 token？”
   - English: `q = self.q(x)` means each video latent asks, "which text tokens should I read now?"
2. **文本 context 出 key/value / Text context produces keys and values**:
   - 中文: key 决定匹配位置，value 承载被读出的条件信息。
   - English: Keys decide matching positions, while values carry the conditioning information being read.
3. **`context_lens` 控制有效文本长度 / `context_lens` controls valid text length**:
   - 中文: batch 内 prompt 长度不同，attention kernel 需要知道每条样本真正有多长。
   - English: Prompts differ in length within a batch, so the attention kernel needs the valid length per sample.
4. **输出投影回模型宽度 / Project back to model width**:
   - 中文: 多头输出 flatten 后再过 `self.o`，回到 DiT block 的主通道。
   - English: Multi-head outputs are flattened and passed through `self.o`, returning to the DiT block's main channel.

## 类比 / The analogy

像导演在片场拿着分镜问编剧：当前镜头是视频 token，剧本是文本 context。镜头自己提出问题，剧本文字提供答案，最后导演把答案变成具体拍法。

It is like a director asking a scriptwriter for guidance on a shot. The current shot is the video token, the script is text context, and the answer becomes concrete visual direction.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

这是 `text-conditioning` 组件，位于 DiT block 内部，通常接在 self-attention 之后或与它并列。输入是 noisy video/action latent 和文本 encoder 输出，输出仍是同形状 latent。如果省掉它，模型只能学无条件动力学；生产级实现还要处理 CFG、prompt padding、长文本截断和跨模态归一化。

This is the `text-conditioning` component inside the DiT block, usually after or beside self-attention. Inputs are noisy video/action latents plus text-encoder states, and the output has the same latent shape. Without it, the model learns only unconditional dynamics. A production version also handles CFG, prompt padding, long-text truncation, and cross-modal normalization.

## 自己跑一遍 / Try it yourself

```python
import math

q = [1.0, 0.0]
keys = [[1.0, 0.0], [0.0, 1.0]]
values = ["red cube", "blue wall"]
scores = [sum(a*b for a, b in zip(q, k)) / math.sqrt(2) for k in keys]
best = max(range(len(scores)), key=scores.__getitem__)
print(values[best])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
red cube
```

这个例子把 cross-attention 压成最近邻读取：query 更像哪个 key，就读出哪个 value。

This example compresses cross-attention into nearest-neighbor lookup: whichever key best matches the query determines which value is read.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Stable Diffusion UNet** / **Stable Diffusion UNet**: latent feature 查询 CLIP text embedding。 / Latent features query CLIP text embeddings.
- **GR00T / VLA action expert** / **GR00T / VLA action experts**: action latent 查询语言和视觉 context，决定动作预测。 / Action latents query language and visual context to shape action prediction.

## 注意事项 / Caveats / when it breaks

- **文本长度 mask 错误会污染 attention** / **bad text-length masks pollute attention**: padding token 可能被当成条件读取。 / Padding tokens may be read as real conditioning.
- **query/key 归一化很关键** / **query/key normalization matters**: Wan2.1 在 Q/K 上做 RMSNorm，稳定大模型 attention。 / Wan2.1 normalizes Q/K with RMSNorm to stabilize large-model attention.

## 延伸阅读 / Further reading

- Wan2.1 model source — https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py
