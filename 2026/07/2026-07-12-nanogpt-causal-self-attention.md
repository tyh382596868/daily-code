---
date: 2026-07-12
topic: infrastructure
source: tracked
repo: karpathy/nanoGPT
file: model.py
permalink: https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/model.py#L29-L76
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, infrastructure, causal-attention]
---

# nanoGPT attention：一层里同时投影 QKV / nanoGPT Attention: Project QKV in One Layer

> **一句话 / In one line**: nanoGPT 用一个 `Linear` 一次性产出 Q/K/V，再把 head 维度搬到前面交给 causal attention。 / nanoGPT uses one `Linear` to emit Q/K/V together, then moves the head dimension forward before causal attention.

## 为什么重要 / Why this matters

这段代码是“小模型也要有生产级形状意识”的好例子。它没有把 Q、K、V 写成三层，而是用一个 `3 * n_embd` 的投影合并内存访问；如果 PyTorch 版本支持 SDPA，就直接走 fused causal attention，否则退回手写 mask。

This is a compact example of shape-aware implementation. Q, K, and V are packed into one projection to reduce overhead; when PyTorch exposes SDPA, the code delegates causal attention to the fused path, otherwise it falls back to an explicit triangular mask.

## 代码 / The code

`karpathy/nanoGPT` — [`model.py`](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/model.py#L29-L76)

```python
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y
```

## 逐行讲解 / What's happening

1. **初始化投影 / Projection setup**:
   - 中文: `c_attn` 的输出维度是 `3 * n_embd`，所以一次矩阵乘法就拿到 Q、K、V。
   - English: `c_attn` outputs `3 * n_embd`, so a single matrix multiply produces Q, K, and V.
2. **拆分和变形 / Split and reshape**:
   - 中文: `split` 后还是 `[B, T, C]`，再 reshape 成 `[B, heads, T, head_dim]`，这是 attention kernel 想要的布局。
   - English: After `split`, each tensor is still `[B, T, C]`; reshaping to `[B, heads, T, head_dim]` matches the attention kernel layout.
3. **两条 attention 路径 / Two attention paths**:
   - 中文: 新 PyTorch 走 `scaled_dot_product_attention(..., is_causal=True)`，旧环境用显式三角 mask。
   - English: Newer PyTorch uses `scaled_dot_product_attention(..., is_causal=True)`; older environments use an explicit lower-triangular mask.
4. **拼回 embedding / Reassemble embeddings**:
   - 中文: attention 输出先从 head 维度转回 token 维度，再通过 `c_proj` 回到残差流。
   - English: The attention result moves head outputs back beside the token dimension, then `c_proj` returns it to the residual stream.

## 类比 / The analogy

像餐厅后厨一次收齐三张单子：主菜、配菜、饮料一起打印，然后分给三个工位。比每个工位单独找服务员要省事，最后再把三路结果装回同一个餐盘。

Think of a kitchen printing one combined ticket for entree, side, and drink, then splitting the work across stations. It is cheaper than asking the waiter three separate times, and the station outputs are assembled back onto one tray.

## 自己跑一遍 / Try it yourself

```python
import math
import random

B, T, C, H = 1, 4, 8, 2
head_dim = C // H
x = [[random.random() for _ in range(C)] for _ in range(T)]

# Pretend one packed linear produced [q | k | v].
packed = [row * 3 for row in x]
q = [row[:C] for row in packed]
k = [row[C:2*C] for row in packed]
v = [row[2*C:] for row in packed]

scores = []
for i in range(T):
    row = []
    for j in range(T):
        row.append("-inf" if j > i else round(sum(q[i][d] * k[j][d] for d in range(C)) / math.sqrt(C), 3))
    scores.append(row)
print(scores)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
[[...], [...], [...], [...]]
```

注意每一行右侧未来 token 都是 `-inf`，这就是 causal mask 的核心效果。

The key observation is that future-token positions become `-inf` on each row; that is the essence of causal masking.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Transformers GPT-like blocks** / **GPT-style blocks in Transformers**: 也常把 QKV 合成一个投影，再按 head 拆开。 / They often use a packed QKV projection before splitting by heads.
- **vLLM attention kernels** / **vLLM attention kernels**: 同样把布局整理成 kernel 喜欢的形状，再把 cache 和 attention 算子接起来。 / They similarly normalize tensor layout before handing work to cache-aware attention kernels.

## 注意事项 / Caveats / when it breaks

- **`n_embd` 必须整除 `n_head`** / **`n_embd` must divide by `n_head`**: 否则每个 head 的宽度不是整数。 / Otherwise each head cannot get an equal-width slice.
- **fallback mask 占用 `block_size^2` 存储** / **the fallback mask costs `block_size^2` storage**: 长上下文时这会变贵。 / For long contexts, the explicit mask becomes expensive.

## 延伸阅读 / Further reading

- nanoGPT `model.py` — https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/model.py
- PyTorch SDPA docs — https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
