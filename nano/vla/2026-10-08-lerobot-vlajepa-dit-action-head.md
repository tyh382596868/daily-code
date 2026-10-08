---
date: 2026-10-08
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/policies/vla_jepa/action_head.py
permalink: https://github.com/huggingface/lerobot/blob/ca69a2068462a37f7cdcb74180927a2f863d2bf7/src/lerobot/policies/vla_jepa/action_head.py#L47-L187
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, vla, action-head-continuous]
build_role: action-head-continuous advanced variant, DiT action head with cross-attention and timestep modulation
---

# LeRobot VLA-JEPA action head：动作 token 自己去噪，语义 token 做条件 / LeRobot VLA-JEPA Action Head: Denoise Action Tokens While Conditioning on Semantic Tokens

> **一句话 / In one line**: 这个动作头把 noisy action 编成 DiT token，交替做 cross-attention/self-attention，再用 timestep 调制输出动作特征。 / This action head encodes noisy actions into DiT tokens, alternates cross-attention and self-attention, then uses timestep modulation to emit action features.

## 为什么重要 / Why this matters

VLA 的连续动作头不是“VLM 最后一层接个 MLP”这么简单。这里把动作序列、flow timestep、外部语义条件拆成三种角色：动作 token 是要被去噪的对象，encoder hidden states 是条件，timestep embedding 控制每一层的归一化和最终输出。

A continuous VLA action head is more than "put an MLP after the VLM." This code separates action sequence, flow timestep, and semantic conditioning into three roles: action tokens are the object being denoised, encoder hidden states provide context, and timestep embeddings modulate normalization and output.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/vla_jepa/action_head.py`](https://github.com/huggingface/lerobot/blob/ca69a2068462a37f7cdcb74180927a2f863d2bf7/src/lerobot/policies/vla_jepa/action_head.py#L47-L187)

```python
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()
        batch_size, seq_len = timesteps.shape
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=timesteps.device)
        exponent = exponent * (torch.log(torch.tensor(10000.0, device=timesteps.device)) / max(half_dim, 1))
        freqs = timesteps.unsqueeze(-1) * exponent.exp()
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1).view(batch_size, seq_len, -1)


class ActionEncoder(nn.Module):
    def __init__(self, action_dim: int, hidden_size: int):
        super().__init__()
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size * 2, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = actions.shape
        if timesteps.ndim != 1 or timesteps.shape[0] != batch_size:
            raise ValueError("timesteps must have shape [batch_size].")
        timesteps = timesteps.unsqueeze(1).expand(-1, seq_len)
        action_emb = self.layer1(actions)
        time_emb = self.pos_encoding(timesteps).to(dtype=action_emb.dtype)
        return self.layer3(F.silu(self.layer2(torch.cat([action_emb, time_emb], dim=-1))))


class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        require_package("diffusers", extra="vla_jepa")
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        projected = self.time_proj(timesteps).to(dtype=next(self.parameters()).dtype)
        return self.timestep_embedder(projected)


class AdaLayerNorm(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, eps=1e-5, elementwise_affine=False)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(self.silu(temb)).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float,
        cross_attention_dim: int,
        is_cross_attention: bool = True,
    ) -> None:
        super().__init__()
        self.is_cross_attention = is_cross_attention
        self.norm1 = AdaLayerNorm(dim)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=True,
            cross_attention_dim=cross_attention_dim,
            out_bias=True,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-5, elementwise_affine=False)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn="gelu-approximate", final_dropout=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        temb: torch.Tensor,
    ) -> torch.Tensor:
        attn_input = self.norm1(hidden_states, temb)
        attention_context = encoder_hidden_states if self.is_cross_attention else None
        hidden_states = hidden_states + self.attn1(attn_input, encoder_hidden_states=attention_context)
        hidden_states = hidden_states + self.ff(self.norm2(hidden_states))
        return hidden_states


class DiT(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float,
        cross_attention_dim: int,
    ) -> None:
        super().__init__()
        self.inner_dim = num_attention_heads * attention_head_dim
        self.timestep_encoder = TimestepEncoder(self.inner_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim if layer_idx % 2 == 0 else self.inner_dim,
                    is_cross_attention=layer_idx % 2 == 0,
                )
                for layer_idx in range(num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(self.inner_dim, eps=1e-6, elementwise_affine=False)
        self.proj_out_1 = nn.Linear(self.inner_dim, self.inner_dim * 2)
        self.proj_out_2 = nn.Linear(self.inner_dim, output_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        temb = self.timestep_encoder(timestep)
        x = hidden_states
        for block in self.transformer_blocks:
            x = block(x, encoder_hidden_states=encoder_hidden_states, temb=temb)
        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=-1)
        x = self.norm_out(x) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out_2(x)
```

## 逐行讲解 / What's happening

1. **第 62-77 行 / Lines 62-77**:
   - 中文: `ActionEncoder` 先投影动作，再把同一个 timestep 扩展到 horizon 每个位置，和动作 embedding 拼起来。
   - English: `ActionEncoder` projects actions, expands the same timestep across the horizon, and concatenates time embeddings with action embeddings.
1. **第 92-101 行 / Lines 92-101**:
   - 中文: `AdaLayerNorm` 用 timestep 生成 scale/shift，让不同噪声时刻看到不同的归一化状态。
   - English: `AdaLayerNorm` turns the timestep into scale and shift, so each noise time sees a different normalized state.
1. **第 129-139 行 / Lines 129-139**:
   - 中文: block 可以是 cross-attention，也可以退化成 self-attention；偶数层看外部条件，奇数层只整理动作 token 内部关系。
   - English: A block can use cross-attention or self-attention; even layers read external context, odd layers organize action-token relationships.
1. **第 181-187 行 / Lines 181-187**:
   - 中文: DiT 先跑所有 blocks，最后再用 timestep 对输出做一次 AdaLN-like 调制，然后投到动作特征维度。
   - English: The DiT runs all blocks, then applies one final timestep-conditioned modulation before projecting to action features.

## 类比 / The analogy

像舞蹈排练：动作 token 是舞者，语言/视觉 token 是导演提示，timestep 是“现在排练到第几遍”。有些轮次听导演，有些轮次舞者彼此对齐。

It is like a dance rehearsal. Action tokens are dancers, language/vision tokens are the director's cues, and timestep says which rehearsal pass this is. Some rounds listen to the director; others align the dancers with each other.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

这是 `action-head-continuous` 的高级版本，依赖前面的 `vlm-backbone-wiring` 和动作条件构造。nanoVLA 最小版可以用 MLP 或小 Transformer 从 context 直接回归动作；生产版则需要这种 DiT/flow-style 头，把 noisy action、timestep、condition tokens 分开建模，才能处理多步 action chunk 和采样过程。

This is an advanced `action-head-continuous` slot and depends on `vlm-backbone-wiring` plus action conditioning. A minimal nanoVLA can regress actions from context with an MLP or tiny Transformer; a production version benefits from this DiT/flow-style head, which models noisy actions, timestep, and condition tokens separately for multi-step action chunks and sampling.

## 自己跑一遍 / Try it yourself

```python
import math

def encode_action(action, t):
    time = [math.sin(t), math.cos(t)]
    return [round(action[0] + time[0], 3), round(action[1] + time[1], 3)]

def block(x, context=None):
    ctx = sum(context) / len(context) if context else sum(x) / len(x)
    return [round(v + 0.1 * ctx, 3) for v in x]

x = [encode_action([0.2, -0.1], 0.3), encode_action([0.5, 0.4], 0.3)]
flat = [v for pair in x for v in pair]
print(block(flat, context=[1.0, 2.0]))
print(block(block(flat, context=[1.0, 2.0])))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[0.646, 1.005, 0.946, 1.505]
[0.749, 1.108, 1.049, 1.608]
```

第一轮用外部 context 推动作 token，第二轮让动作 token 自己互相整理，正对应 cross/self 交替的骨架。

The first round nudges action tokens with external context; the second lets action tokens reorganize themselves, mirroring the cross/self alternation.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LeRobot Evo1 flow matching head** / **LeRobot Evo1 flow matching head**: 同样把动作 horizon 编成 token，再用条件 token 辅助去噪。 / It also tokenizes the action horizon and uses condition tokens to guide denoising.
- **GR00T cross-attention DiT** / **GR00T cross-attention DiT**: 也用 timestep embedding 和 Transformer block 预测连续动作。 / It also uses timestep embeddings and Transformer blocks to predict continuous actions.

## 注意事项 / Caveats / when it breaks

- **cross-attention 维度要对齐** / **Cross-attention dimensions must match**: `cross_attention_dim` 来自上游 VLM/context，和 action head 的 inner dim 不一定一样。 / `cross_attention_dim` comes from upstream VLM/context and may differ from the action head inner dimension.
- **采样时间分布影响行为** / **The time distribution affects behavior**: 训练时怎么采 timestep，会改变模型在高噪声和低噪声区间的能力。 / How training samples timesteps changes the model's ability at high- and low-noise regions.

## 延伸阅读 / Further reading

- [LeRobot repository](https://github.com/huggingface/lerobot)
- [Diffusers DiT-style blocks](https://huggingface.co/docs/diffusers/)
