---
date: 2026-07-14
topic: wam
source: wam
repo: facebookresearch/DiT
file: models.py
permalink: https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L24-L58
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, wam, noise-scheduler, timestep-embedding]
build_role: noise-scheduler timestep conditioning for a from-scratch nanoWAM
---

# DiT TimestepEmbedder：把噪声步数变成条件向量 / DiT TimestepEmbedder: Turn a Noise Step into a Conditioning Vector

> **一句话 / In one line**: DiT 先用 sin/cos 频率编码标量 timestep，再用两层 MLP 投到 transformer hidden size。 / DiT first encodes a scalar timestep with sinusoidal frequencies, then projects it to transformer hidden size with a two-layer MLP.

## 为什么重要 / Why this matters

扩散模型每一步都要知道“现在噪声有多重”。这个信息不能只当普通数字丢进去，因为 transformer 需要一个和 hidden state 同维度的条件向量。`TimestepEmbedder` 是最小但完整的答案：固定频率基底负责覆盖时间尺度，MLP 负责学成模型内部可用的调制信号。

A diffusion model must know “how noisy is this step?” That information cannot be passed as a raw number because the transformer needs a conditioning vector in hidden-state space. `TimestepEmbedder` is the minimal complete answer: fixed frequencies cover time scales, and an MLP learns a modulation signal usable inside the model.

## 代码 / The code

`facebookresearch/DiT` — [`models.py`](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L24-L58)

```python
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
```

## 逐行讲解 / What's happening

1. **固定频率 / Fixed frequencies**:
   - 中文: `max_period` 和指数频率让同一个 timestep 同时拥有快慢不同的周期信号。
   - English: `max_period` and exponential frequencies give one timestep both fast and slow periodic signals.
2. **sin/cos 拼接 / Concatenate sin and cos**:
   - 中文: cos/sin 成对出现，模型能区分相位而不是只看一个标量大小。
   - English: Cosine and sine appear as pairs, so the model sees phase rather than only scalar magnitude.
3. **奇数维补零 / Pad odd dimensions**:
   - 中文: 如果 embedding 维度是奇数，补一列零保持输出维度准确。
   - English: If the embedding dimension is odd, one zero column preserves the requested width.
4. **MLP 投到 hidden size / MLP projection to hidden size**:
   - 中文: 频率特征只是原料，MLP 把它变成 DiT block 可用于 adaLN 的条件向量。
   - English: Frequency features are raw material; the MLP converts them into a conditioning vector for DiT blocks.

## 类比 / The analogy

像钟表不只告诉你“第几秒”，还同时给秒针、分针、时针的位置。模型能从不同尺度判断当前处在采样过程的哪个阶段。

It is like a clock that shows not only “which second,” but also second-hand, minute-hand, and hour-hand positions. The model can read the sampling phase at several scales.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

这是 `noise-scheduler` 和 `dit-block` 之间的接口。scheduler 给出标量 timestep；embedder 把它变成 hidden vector；DiT block 再用它调制归一化、attention 或 MLP。没有这层，模型只能看到 noisy latent，却不知道该去噪到哪一步。

This is the interface between `noise-scheduler` and `dit-block`. The scheduler emits scalar timesteps; the embedder turns them into hidden vectors; DiT blocks use them to modulate normalization, attention, or MLP paths. Without it, the model sees noisy latents but not which denoising stage it is in.

## 自己跑一遍 / Try it yourself

```python
import math

def embed(t, dim=4, max_period=10000):
    half = dim // 2
    freqs = [math.exp(-math.log(max_period) * i / half) for i in range(half)]
    args = [t * f for f in freqs]
    return [round(math.cos(a), 4) for a in args] + [round(math.sin(a), 4) for a in args]

print(embed(0))
print(embed(10))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[1.0, 1.0, 0.0, 0.0]
[-0.8391, 0.995, -0.544, 0.0998]
```

同一个 timestep 被展开成多个频率下的位置，后续 MLP 可以学习哪些尺度更有用。

The same timestep is expanded into positions at multiple frequencies, and the later MLP can learn which scales matter.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Wan2.1 timestep embedding** / **Wan2.1 timestep embedding**: 同样先用频率特征表示噪声时间，再投影到模型维度。 / It also represents noise time with frequency features before projection.
- **Transformer positional encoding** / **Transformer positional encoding**: 文本位置编码也用 sin/cos 给标量位置加多尺度坐标。 / Text positional encodings also use sin/cos to give scalar positions multi-scale coordinates.

## 注意事项 / Caveats / when it breaks

- **timestep 标度要一致 / Timestep scale must match**: 训练用 0..1000，推理却传 0..1，会改变频率输入分布。 / Training on 0..1000 but inferring on 0..1 changes the frequency distribution.
- **MLP 初始化会影响调制强度 / MLP initialization affects modulation strength**: 条件向量太大可能让早期训练不稳定。 / Too-large conditioning vectors can destabilize early training.

## 延伸阅读 / Further reading

- [DiT `TimestepEmbedder`](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L24-L58)
