---
date: 2026-08-20
topic: robotics
source: tracked
repo: starVLA/starVLA
file: starVLA/model/modules/projector/QFormer.py
permalink: https://github.com/starVLA/starVLA/blob/0ed0aad2c83f587714f6167ef60cf7218b786590/starVLA/model/modules/projector/QFormer.py#L5-L105
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, q-former, layerwise-features]
---

# StarVLA Layerwise Q-Former：每层视觉特征都问一遍 / StarVLA Layerwise Q-Former: Query Every Vision Layer

> **一句话 / In one line**: 这段代码用一组可学习 query 逐层 cross-attend 视觉 encoder 的隐藏层，把多层视觉信息压成固定数量的 VLA token。 / This code uses learnable queries to cross-attend over each vision encoder layer, compressing multi-layer visual features into a fixed number of VLA tokens.

## 为什么重要 / Why this matters

机器人 VLA 不一定只需要最后一层视觉特征。浅层更像边缘、纹理和局部形状，深层更像物体和语义。StarVLA 的 `LayerwiseQFormer` 保留一组固定 query，让它们按层读取隐藏状态，等于给动作模型一个可控大小的视觉摘要。

A robot VLA may need more than the final vision feature map. Early layers carry texture and local geometry; later layers carry object and semantic structure. StarVLA's `LayerwiseQFormer` keeps a fixed query budget and lets those queries read layer outputs one by one, producing a bounded visual summary for the action model.

## 代码 / The code

`starVLA/starVLA` — [`starVLA/model/modules/projector/QFormer.py`](https://github.com/starVLA/starVLA/blob/0ed0aad2c83f587714f6167ef60cf7218b786590/starVLA/model/modules/projector/QFormer.py#L5-L105)

```python
class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, encoder_hidden_state, encoder_attention_mask=None):
        q = self.norm1(query)
        kv = encoder_hidden_state

        if encoder_attention_mask is not None:
            attn_mask = encoder_attention_mask.unsqueeze(1).to(dtype=torch.bool)  # [B, 1, L]
        else:
            attn_mask = None

        attn_output, _ = self.cross_attn(q, kv, kv, key_padding_mask=attn_mask)
        query = query + attn_output
        query = query + self.dropout(self.mlp(self.norm2(query)))
        return query


class LayerwiseQFormer(nn.Module):
    def __init__(
        self, input_hidden_dim=2048, output_hidden_dim=768, num_query_tokens=64, num_layers=37, num_heads=8, config=None
    ):
        super().__init__()
        self.input_hidden_dim = input_hidden_dim
        self.output_hidden_dim = output_hidden_dim
        self.num_query_tokens = num_query_tokens
        self.num_layers = num_layers
        self.config = config
        # Project input to output dimension
        self.proj = nn.Linear(input_hidden_dim, output_hidden_dim)
        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.randn(num_query_tokens, output_hidden_dim))

        # Independent cross-attention blocks (one per encoder layer)
        self.layers = nn.ModuleList([CrossAttentionBlock(output_hidden_dim, num_heads) for _ in range(num_layers)])

    def forward(self, hidden_states_list, encoder_attention_mask=None):
        assert (
            len(hidden_states_list) == self.num_layers
        ), f"Expected {self.num_layers} layers, got {len(hidden_states_list)}"

        B = hidden_states_list[0].size(0)
        # Project input hidden states to output dimension
        #    Result shape [B, N, L, Din]
        hs = torch.stack(hidden_states_list, dim=1)
        #    proj_hs shape [B, N, L, Dout]
        proj_hs = self.proj(hs)
        # 3) Unbind back to list, each element restored to [B, L, Dout]
        hidden_states_list = list(proj_hs.unbind(dim=1))

        # Expand query tokens for each batch
        query = self.query_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, Q, D]

        # Iterate through each layer and apply cross-attention
        for i, layer in enumerate(self.layers):
            query = layer(query, hidden_states_list[i], encoder_attention_mask)

        return query
```

## 逐行讲解 / What's happening

1. **第 5-19 行 / Lines 5-19 (`CrossAttentionBlock`)**:
   - 中文: 一个 block 先归一化 query，再用 query 读视觉层的 K/V，最后走一层 MLP 残差。
   - English: Each block normalizes the queries, uses them to read one visual layer as K/V, then applies an MLP residual.
2. **第 49-65 行 / Lines 49-65 (`LayerwiseQFormer.__init__`)**:
   - 中文: `proj` 把 encoder 输出维度投到 action/VLM 需要的维度，`query_tokens` 决定压缩后的 token 数量。
   - English: `proj` maps encoder features into the target width, while `query_tokens` fixes the compressed token budget.
3. **第 85-96 行 / Lines 85-96 (stack and project)**:
   - 中文: 多层 hidden states 先堆成 `[B, N, L, Din]`，一次线性投影后再拆回每层。
   - English: The layer states are stacked as `[B, N, L, Din]`, projected in one shot, then unbound back into per-layer tensors.
4. **第 99-105 行 / Lines 99-105 (layerwise querying)**:
   - 中文: 同一组 query 依次经过每个 cross-attention block，逐层吸收视觉信息。
   - English: The same query set passes through every cross-attention block and accumulates information layer by layer.

## 类比 / The analogy

像一个质检小组参观工厂。每个 query 是一位固定岗位的质检员，他们不把整条流水线搬走，而是在每一站询问关键细节，最后带回一份固定页数的检查报告。

It is like a fixed inspection team walking through a factory. Each query is one inspector. They do not carry the whole assembly line away; they ask targeted questions at every station and return a fixed-size report.

## 自己跑一遍 / Try it yourself

```python
layers = [
    [[1, 2], [3, 4]],
    [[10, 20], [30, 40]],
    [[100, 200], [300, 400]],
]
queries = [0, 0]

for layer in layers:
    column_means = [sum(col) / len(col) for col in zip(*layer)]
    queries = [q + m for q, m in zip(queries, column_means)]

print(queries)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[222.0, 333.0]
```

中文: toy 版本没有 attention，只演示固定 query 逐层累积信息这个结构。

English: The toy version has no attention; it only shows the structure of fixed queries accumulating layerwise information.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **BLIP-2 Q-Former** / **BLIP-2 Q-Former**: 用 learnable queries 从视觉 encoder 里抽取少量视觉 token。 / It uses learnable queries to extract a small set of visual tokens from a vision encoder.
- **Perceiver resampler** / **Perceiver resampler**: 用固定 latent array cross-attend 大输入，控制输出 token 数。 / It cross-attends a fixed latent array over a large input to bound the output token count.

## 注意事项 / Caveats / when it breaks

- **mask 语义要核对** / **Mask semantics need checking**: PyTorch `key_padding_mask=True` 表示屏蔽位置，和很多 attention mask 的 keep 语义相反。 / In PyTorch, `key_padding_mask=True` means masked, which is opposite to many keep-style masks.
- **层数是硬合同** / **Layer count is a hard contract**: `len(hidden_states_list)` 必须等于 `num_layers`，换视觉骨干时要同步配置。 / `len(hidden_states_list)` must match `num_layers`; changing the vision backbone requires config updates.

## 延伸阅读 / Further reading

- [StarVLA `LayerwiseQFormer`](https://github.com/starVLA/starVLA/blob/0ed0aad2c83f587714f6167ef60cf7218b786590/starVLA/model/modules/projector/QFormer.py#L5-L105)
- [StarVLA repository](https://github.com/starVLA/starVLA)
