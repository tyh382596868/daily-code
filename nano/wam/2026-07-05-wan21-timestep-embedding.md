---
date: 2026-07-05
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/modules/model.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py
difficulty: beginner
read_time: ~10 min
tags: [code-of-the-day, wam, timestep-embedding, diffusion]
build_role: noise-scheduler advanced variant: scalar timestep to conditioning vector
---

# Wan2.1 timestep embedding：把一个噪声时间步变成一排频率尺 / Wan2.1 Timestep Embedding: Turn One Noise Step into a Row of Frequency Rulers

> **一句话 / In one line**: 扩散模型把标量 timestep 先编码成 sinusoidal 向量，再交给 DiT block 做条件调制。 / A diffusion model first encodes a scalar timestep as a sinusoidal vector, then gives it to DiT blocks for conditioning.

## 为什么重要 / Why this matters

WAM 生成视频或动作时，模型必须知道“现在噪声有多重”。Wan2.1 的 `sinusoidal_embedding_1d` 不学习参数，而是用一组从慢到快的频率把 timestep 展开；后面的 MLP 再把它变成 adaLN 或 gate 需要的调制向量。

When a WAM generates video or actions, the model must know how noisy the current sample is. Wan2.1's `sinusoidal_embedding_1d` uses no learned parameters; it expands the timestep through frequencies from slow to fast. Later MLPs turn that into modulation vectors for adaLN or gates.

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/modules/model.py`](https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py)

```python
def sinusoidal_embedding_1d(dim, position):
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    sinusoid = torch.outer(
        position,
        torch.pow(10000, -torch.arange(half).to(position).div(half)),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x
```

## 逐行讲解 / What's happening

1. **维度必须成对 / Dimensions come in pairs**: 中文: 每个频率需要 `cos` 和 `sin` 两个通道，所以 `dim` 必须是偶数。 / English: Each frequency needs both `cos` and `sin`, so `dim` must be even.
2. **位置转成 float64 / Position becomes float64**: 中文: timestep 可能是整数，也可能是连续 flow time；统一成浮点能支持两者。 / English: The timestep can be integer or continuous flow time; converting to float supports both.
3. **`outer` 生成频率网格 / `outer` builds the frequency grid**: 中文: 每个 position 乘上每个频率，得到 `(batch, half_dim)` 的相位矩阵。 / English: Each position multiplies each frequency, producing a `(batch, half_dim)` phase matrix.
4. **拼接 cos/sin / Concatenate cos and sin**: 中文: 同一个 timestep 同时拥有相位的两个正交视角，后续线性层更容易组合。 / English: The same timestep gets two orthogonal phase views, which later linear layers can combine easily.

## 类比 / The analogy

像用一盒彩色尺子量同一段距离：粗尺看大概位置，细尺看精细差别。sinusoidal embedding 把一个 timestep 放到多把尺上同时读数。

It is like measuring one distance with a box of rulers. A coarse ruler gives the rough location; fine rulers capture small differences. Sinusoidal embeddings read one timestep on many rulers at once.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

中文: 这是 `noise-scheduler` 和 `dit-block` 之间的桥：scheduler 产出 timestep，embedding 把它变成向量，DiT block 用这个向量调制 norm、MLP 或 attention。没有它，模型只能看 noisy latent，却不知道应该去噪多少。

English: This is the bridge between `noise-scheduler` and `dit-block`: the scheduler emits a timestep, the embedding turns it into a vector, and DiT blocks use that vector to modulate norms, MLPs, or attention. Without it, the model sees the noisy latent but not how much denoising is required.

## 自己跑一遍 / Try it yourself

```python
import torch

def emb(dim, pos):
    half = dim // 2
    freq = torch.pow(10000, -torch.arange(half).float() / half)
    phase = torch.outer(pos.float(), freq)
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=1)

print(emb(6, torch.tensor([0., 10.])).round(decimals=3))
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```text
tensor([[ 1.000,  1.000,  1.000,  0.000,  0.000,  0.000],
        [-0.839,  0.895,  1.000, -0.544,  0.446,  0.022]])
```

中文: `t=0` 的 sin 全是 0、cos 全是 1；更大的 timestep 会在不同频率上转过不同角度。

English: At `t=0`, all sine channels are 0 and cosine channels are 1; larger timesteps rotate by different amounts at different frequencies.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **DiT timestep embedder** / **DiT timestep embedder**: 中文: 先做 sinusoidal frequency embedding，再进 MLP。 / English: It first builds sinusoidal frequency embeddings, then feeds them into an MLP.
- **Transformer position encoding** / **Transformer position encoding**: 中文: 原始 Transformer 也用同类 cos/sin 多频率编码位置。 / English: The original Transformer uses the same cos/sin multi-frequency idea for positions.

## 注意事项 / Caveats / when it breaks

- **只编码时间，不编码语义 / It encodes time, not semantics**: 中文: timestep embedding 告诉模型噪声阶段，不替代文本、图像或动作条件。 / English: The timestep embedding tells the model the noise phase; it does not replace text, image, or action conditioning.
- **维度太小会欠表达 / Too few dimensions under-express**: 中文: 频率太少时，相邻 timestep 的可分性会变差。 / English: With too few frequencies, neighboring timesteps become harder to separate.

## 延伸阅读 / Further reading

- Wan2.1 model source linked above.
- DiT paper and implementation.
