---
date: 2026-06-23
topic: huggingface
source: huggingface
repo: huggingface/diffusers
file: src/diffusers/models/transformers/transformer_krea2.py
permalink: https://github.com/huggingface/diffusers/blob/6a71b6e332abae01a05d36133003e5370ca1d0a8/src/diffusers/models/transformers/transformer_krea2.py#L167-L247
difficulty: advanced
read_time: ~12 min
tags: [code-of-the-day, huggingface, diffusers, text-conditioning, cross-attention, dit, krea2]
---

# Krea2TextFusion：融合文本编码器所有隐藏层输出的跨层注意力 / Krea2TextFusion: Fusing All Text-Encoder Hidden-Layer Outputs via Cross-Attention

> **一句话 / In one line**: Krea2 不只用文本编码器最后一层的输出，而是把每个 token 在所有层的表示叠在一起，用两个 layerwise cross-attention block 跨层压缩，再用一个 `nn.Linear` 折叠层维度，最后用两个 refiner block 精炼 token 序列。 / Instead of using only the final layer of the text encoder, Krea2 stacks all layer representations per token, runs two layerwise cross-attention blocks across the layer axis, collapses that axis with a single `nn.Linear`, then refines the token sequence with two more blocks.

## 为什么重要 / Why this matters

标准的 DiT 文本条件化（CLIP、T5）只使用文本编码器最后一层的输出——要么是 CLS token，要么是完整的 token 序列。但文本编码器的浅层保留着句法结构，中间层保留着语义聚合，深层保留着抽象语义。丢弃这些信息等于白白扔掉了一个 20+ 层的特征塔。Krea2 的 `TextFusion` 模块在 2026-06-22 刚合并进 diffusers，它通过"跨层注意力 → 线性投影折叠 → token 精炼"的四步流程把所有层的信息汇聚成一条 token 序列，让扩散模型的文本对齐更精细。

Standard DiT text conditioning (CLIP, T5) uses only the text encoder's final-layer output — either the CLS token or the full token sequence. But the encoder's shallower layers retain syntactic structure, mid-layers capture semantic composition, and deep layers hold abstract semantics. Throwing that away wastes a 20+-layer feature tower. Krea2's `TextFusion` module — merged into diffusers on 2026-06-22 — fuses all layers via layerwise cross-attention, a learned linear projection, and token-sequence refinement, giving the diffusion model richer text alignment.

## 代码 / The code

`huggingface/diffusers` — [`src/diffusers/models/transformers/transformer_krea2.py`](https://github.com/huggingface/diffusers/blob/6a71b6e332abae01a05d36133003e5370ca1d0a8/src/diffusers/models/transformers/transformer_krea2.py#L167-L247)

```python
class Krea2TextFusion(nn.Module):
    """Fuses tapped hidden states from all text-encoder layers into one sequence.

    Input:  (B, seq_len, num_text_layers, dim)  — all layers stacked
    Output: (B, seq_len, dim)                   — one fused vector per token
    """
    def __init__(
        self,
        num_text_layers: int,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        num_layerwise_blocks: int,
        num_refiner_blocks: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        block_kwargs = dict(
            dim=dim, num_heads=num_heads, num_kv_heads=num_kv_heads,
            intermediate_size=intermediate_size, eps=eps,
        )
        self.layerwise_blocks = nn.ModuleList(
            [Krea2TextFusionBlock(**block_kwargs) for _ in range(num_layerwise_blocks)]
        )
        self.projector = nn.Linear(num_text_layers, 1, bias=False)
        self.refiner_blocks = nn.ModuleList(
            [Krea2TextFusionBlock(**block_kwargs) for _ in range(num_refiner_blocks)]
        )

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,   # (B, seq_len, num_text_layers, dim)
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, num_text_layers, dim = encoder_hidden_states.shape

        # Step 1: treat each (token, layer) pair as a "sequence over layers"
        hidden_states = encoder_hidden_states.reshape(
            batch_size * seq_len, num_text_layers, dim
        )  # (B*seq_len, num_text_layers, dim)

        # Step 2: two blocks of self-attention across the layer axis (per token)
        for block in self.layerwise_blocks:
            hidden_states = block(hidden_states.contiguous())

        # Step 3: collapse num_text_layers → 1 with a learned linear
        hidden_states = hidden_states.reshape(
            batch_size, seq_len, num_text_layers, dim
        ).permute(0, 1, 3, 2)              # (B, seq_len, dim, num_text_layers)
        hidden_states = self.projector(hidden_states).squeeze(-1)
        # → (B, seq_len, dim)

        # Step 4: two refiner blocks of self-attention across the token sequence
        for block in self.refiner_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask)

        return hidden_states
```

## 逐行讲解 / What's happening

1. **输入形状 `(B, seq_len, num_text_layers, dim)` / Input shape**:
   - 中文: 文本编码器在前向传播时被 "tapped"——每层的隐藏状态都被保留下来，最终拼成一个 4D 张量。`num_text_layers` 通常是 12（CLIP-L）或 32（T5-XXL）。
   - English: The text encoder is "tapped" during its forward pass — every layer's hidden state is saved and stacked into a 4D tensor. `num_text_layers` is typically 12 (CLIP-L) or 32 (T5-XXL).

2. **`reshape(B * seq_len, num_text_layers, dim)` — 展平 batch×token / flatten batch×token**:
   - 中文: 把 batch 维和 token 维合并，这样 `layerwise_blocks` 里的自注意力就在**层轴**上操作——每个 token 独立地在它自己的 `num_text_layers` 个层表示之间做注意力。这是一个"per-token layer attention"操作。
   - English: Merges the batch and token dimensions so that attention inside `layerwise_blocks` operates over the **layer axis** — each token independently attends across its own `num_text_layers` representations. This is per-token layer attention.

3. **`layerwise_blocks` — 两个跨层注意力 block / two cross-layer attention blocks**:
   - 中文: `Krea2TextFusionBlock` 是一个带 GQA、QK-RMSNorm 和 sigmoid 输出门的注意力块，在此处处理长度为 `num_text_layers` 的序列（而非 token 序列）。通过两层叠加，每个 token 的层间表示相互混合，而不同 token 之间暂不交互。
   - English: `Krea2TextFusionBlock` is an attention block with GQA, QK-RMSNorm, and a sigmoid output gate. Here it processes a sequence of length `num_text_layers` (not token length). After two blocks, each token's representations across layers are mixed; token-to-token interactions come later.

4. **`projector = nn.Linear(num_text_layers, 1, bias=False)` — 折叠层维 / collapse layer dim**:
   - 中文: 这是整个模块里最简洁的一步：`permute` 把层维放到最后，然后一个无偏置线性层用一组学到的加权平均把 `num_text_layers` 折叠成 1，`squeeze(-1)` 去掉多余的 1 维。比 `mean` 更灵活——模型可以学会"第 8 层最重要"。
   - English: The cleanest step: `permute` puts the layer dimension last, then a bias-free linear collapses `num_text_layers` → 1 via a learned weighted average. `squeeze(-1)` removes the trailing singleton. More expressive than `mean` — the model can learn "layer 8 matters most".

5. **`refiner_blocks` — token 序列精炼 / token-sequence refinement**:
   - 中文: 经过折叠后，形状回到 `(B, seq_len, dim)`，现在做**跨 token** 的注意力，即标准的文本序列 self-attention。`attention_mask` 在此处起作用，遮掉 padding token 的注意力。
   - English: After collapsing, shape is `(B, seq_len, dim)` and attention now runs **across tokens** — standard text-sequence self-attention. The `attention_mask` masks out padding tokens at this stage.

## 类比 / The analogy

想象一个翻译团队，每个翻译员负责一个单词（token），但团队有 32 个资历不同的顾问（层）——初级顾问看句法，中级顾问看语义，高级顾问看抽象含义。`layerwise_blocks` 让每个翻译员先在自己的 32 个顾问之间开内部会议（跨层注意力），`projector` 把这 32 条意见压缩成一条摘要，`refiner_blocks` 最后让所有翻译员之间开大会（跨 token 注意力），确保语义一致。

Imagine a translation team where each translator handles one word (token), but the team has 32 consultants of different seniority (layers) — junior consultants handle syntax, mid-level consultants handle semantics, senior ones handle abstraction. The `layerwise_blocks` let each translator first hold an internal meeting among their 32 consultants (cross-layer attention), the `projector` compresses those 32 opinions into one summary per translator, and `refiner_blocks` then hold a team-wide meeting (cross-token attention) to ensure semantic consistency.

## 自己跑一遍 / Try it yourself

```python
import torch, torch.nn as nn

class TinyTextFusion(nn.Module):
    def __init__(self, num_layers=4, dim=64, num_heads=4):
        super().__init__()
        self.layerwise = nn.TransformerEncoderLayer(dim, num_heads, dim_feedforward=dim*4, batch_first=True)
        self.projector  = nn.Linear(num_layers, 1, bias=False)
        self.refiner    = nn.TransformerEncoderLayer(dim, num_heads, dim_feedforward=dim*4, batch_first=True)

    def forward(self, x):              # x: (B, seq_len, num_layers, dim)
        B, S, L, D = x.shape
        h = x.reshape(B * S, L, D)    # per-token layer sequence
        h = self.layerwise(h)          # cross-layer attention
        h = h.reshape(B, S, L, D).permute(0, 1, 3, 2)  # (B, S, D, L)
        h = self.projector(h).squeeze(-1)               # (B, S, D)
        h = self.refiner(h)            # cross-token attention
        return h

B, S, L, D = 2, 10, 4, 64
x = torch.randn(B, S, L, D)
model = TinyTextFusion(num_layers=L, dim=D)
out = model(x)
print("output shape:", out.shape)  # expect (2, 10, 64)
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
output shape: torch.Size([2, 10, 64])
```

中文：注意 `reshape(B*S, L, D)` 这一步——它是"跨层注意力"能独立作用于每个 token 的关键。如果直接在 `(B, S, L, D)` 上做注意力，token 间就会提前交互，丢失了先"per-token 跨层融合"再"跨 token 精炼"的设计意图。

English: The `reshape(B*S, L, D)` step is critical — it isolates the cross-layer attention to each token independently. If you ran attention directly on `(B, S, L, D)`, tokens would interact prematurely, breaking the intended two-phase design of "per-token layer fusion first, then cross-token refinement."

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Stable Diffusion 3 / SD3 文本编码器融合** / **SD3 text encoder fusion**: SD3 把 CLIP-L、CLIP-G 和 T5-XXL 三个编码器的最后层拼接——比 Krea2 更粗糙（只用最后层），但思路相同：用多个文本编码器的输出融合文本条件。 / SD3 concatenates the final-layer outputs of CLIP-L, CLIP-G, and T5-XXL — coarser than Krea2 (final layer only), but the same philosophy of fusing multi-encoder text conditions.
- **Krea2 的 adaLN timestep conditioning** / **adaLN in Krea2**: 每个 `Krea2TransformerBlock` 用 adaLN 向量来调制，其 shift/scale 来自一个共享的 timestep embedding 加上每 block 一个独立的加法表，这是比标准 adaLN-Zero 更精细的条件化。 / Each `Krea2TransformerBlock` uses adaLN with shift/scale from a shared timestep embedding plus a per-block learned additive table — finer-grained conditioning than standard adaLN-Zero.
- **Perceiver IO / cross-attention 压缩** / **Perceiver IO cross-attention compression**: Perceiver IO 用固定大小的 latent array 作为 query，原始输入作为 key/value 做 cross-attention 压缩——和 Krea2 的"层 → 单向量"压缩在结构上类似。 / Perceiver IO uses a fixed-size latent array as queries and raw input as keys/values for cross-attention compression — structurally similar to Krea2's layer-to-single-vector compression.

## 注意事项 / Caveats / when it breaks

- **计算成本随层数线性增长** / **cost scales with num_text_layers**: The `layerwise_blocks` process sequences of length `num_text_layers` for every token — with T5-XXL (32 layers) and a 77-token prompt, that's 77 × 32 = 2464 attention computations just for layer fusion. With a 512-token prompt it becomes expensive.
- **必须 tap 所有层** / **must tap all layers**: Using `TextFusion` requires modifying the text encoder's forward pass to return all hidden states, not just the last one. This complicates inference pipelines that load frozen text encoders via `CLIPTextModel` without modification.
- **RMSNorm weight 初始化为 0** / **RMSNorm weight init to 0**: Krea2 initializes `Krea2RMSNorm.weight` to 0, making the effective multiplier `1 + weight = 1` at init — this is a residual-friendly initialization, but it means the normalization starts as identity, not learned scale.

## 延伸阅读 / Further reading

- [diffusers PR — Krea2 transformer](https://github.com/huggingface/diffusers/blob/6a71b6e332abae01a05d36133003e5370ca1d0a8/src/diffusers/models/transformers/transformer_krea2.py)
- [Perceiver IO paper](https://arxiv.org/abs/2107.14795) — cross-attention compression as a general design principle
- [Stable Diffusion 3 technical report](https://arxiv.org/abs/2403.03206) — multi-encoder text conditioning with CLIP+T5
