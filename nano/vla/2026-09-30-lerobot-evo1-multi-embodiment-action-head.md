---
date: 2026-09-30
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/policies/evo1/flow_matching.py
permalink: https://github.com/huggingface/lerobot/blob/e0d50211ef236143ae867228662b7dfaba554f02/src/lerobot/policies/evo1/flow_matching.py#L110-L245
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, vla, action-head-continuous]
build_role: action-head-continuous advanced variant, multi-embodiment flow-matching action head
---

# LeRobot Evo1 action head：动作先变 token，再按 embodiment 走不同线性层 / LeRobot Evo1 Action Head: Tokenize Actions, Then Route by Embodiment

> **一句话 / In one line**: 多机器人动作头不是只预测一串数字，它先把 horizon 内的动作变成带位置和 embodiment 的 token。 / A multi-robot action head does not just predict numbers; it first turns horizon actions into position- and embodiment-aware tokens.

## 为什么重要 / Why this matters

一个 production VLA 往往要服务不同机械臂、不同夹爪、不同动作维度。Evo1 这里的关键做法是：共享 action-head 框架，但用 category-specific linear 给不同 embodiment 留出自己的投影，同时把 horizon 位置编码进 token 序列。

A production VLA may serve different arms, grippers, and action dimensions. Evo1 keeps the action-head scaffold shared, but uses category-specific linear layers to give each embodiment its own projection while encoding horizon position into the action token sequence.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/evo1/flow_matching.py`](https://github.com/huggingface/lerobot/blob/e0d50211ef236143ae867228662b7dfaba554f02/src/lerobot/policies/evo1/flow_matching.py#L110-L245)

```python
class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(
        self, action_dim: int, embed_dim: int, hidden_dim: int, horizon: int, num_categories: int = 1
    ):
        super().__init__()
        self.horizon = horizon
        self.embed_dim = embed_dim
        self.num_categories = num_categories

        self.W1 = CategorySpecificLinear(action_dim, hidden_dim, num_categories)
        self.W2 = CategorySpecificLinear(hidden_dim, hidden_dim, num_categories)
        self.W3 = CategorySpecificLinear(hidden_dim, embed_dim, num_categories)

        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim, max_len=horizon)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, action_seq: torch.Tensor, category_id: torch.LongTensor):
        batch_size, horizon, action_dim = action_seq.shape
        if self.horizon != horizon:
            raise ValueError(
                f"Action sequence length must match horizon: got {horizon}, expected {self.horizon}."
            )

        x = action_seq.reshape(batch_size * horizon, action_dim)
        if category_id.dim() == 0:
            cat_ids = category_id.expand(horizon * batch_size)
        else:
            cat_ids = category_id.unsqueeze(1).expand(batch_size, horizon).reshape(batch_size * horizon)

        out = self.activation(self.W1(x, cat_ids))
        pos_enc = self.pos_encoding(horizon).to(device=out.device, dtype=out.dtype)
        out = out.view(batch_size, horizon, -1) + pos_enc
        out = out.view(batch_size * horizon, -1)
        out = self.activation(self.W2(out, cat_ids))
        out = self.W3(out, cat_ids)
        return out.view(batch_size, horizon, self.embed_dim)


class BasicTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, embed_dim))

    def forward(
        self,
        action_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        time_emb: torch.Tensor,
        context_key_padding_mask: torch.Tensor | None = None,
    ):
        x = self.norm1(action_tokens)
        attn_out, _ = self.attn(x, context_tokens, context_tokens, key_padding_mask=context_key_padding_mask)
        x = action_tokens + attn_out
        x2 = self.norm2(x)
        if time_emb is not None:
            x2 = x2 + time_emb.unsqueeze(1)
        ff_out = self.ff(x2)
        return x + ff_out


class FlowmatchingActionHead(nn.Module):
    def __init__(
        self,
        embed_dim: int = 896,
        hidden_dim: int = 1024,
        action_dim: int = 16 * 7,
        horizon: int = 16,
        per_action_dim: int = 7,
        num_heads: int = 8,
        num_layers: int = 8,
        dropout: float = 0.0,
        num_inference_timesteps: int = 20,
        num_categories: int = 1,
        state_dim: int | None = None,
        state_hidden_dim: int | None = None,
    ):
        super().__init__()

        logger.info("FlowmatchingActionHead num_inference_timesteps=%s", num_inference_timesteps)
        self.embed_dim = embed_dim
        self.horizon = horizon
        self.per_action_dim = per_action_dim
        self.action_dim = action_dim
        self.num_inference_timesteps = num_inference_timesteps
        self.num_categories = num_categories

        self.time_pos_enc = SinusoidalPositionalEncoding(embed_dim, max_len=1000)
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    hidden_dim=embed_dim * 4,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(embed_dim)
        self.seq_pool_proj = nn.Linear(self.horizon * self.embed_dim, self.embed_dim)
        self.mlp_head = CategorySpecificMLP(
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
            num_categories=num_categories,
        )

        self.state_encoder = None
        if state_dim is not None:
            state_hidden = state_hidden_dim if state_hidden_dim is not None else embed_dim
            self.state_encoder = CategorySpecificMLP(
                input_dim=state_dim,
                hidden_dim=state_hidden,
                output_dim=embed_dim,
                num_categories=num_categories,
            )

        if horizon > 1:
            self.action_encoder: MultiEmbodimentActionEncoder | None = MultiEmbodimentActionEncoder(
                action_dim=self.per_action_dim,
                embed_dim=embed_dim,
                hidden_dim=embed_dim,
                horizon=horizon,
                num_categories=num_categories,
            )
            self.single_action_proj: nn.Linear | None = None
        else:
            self.action_encoder = None
            self.single_action_proj = nn.Linear(self.per_action_dim, self.embed_dim)

    def _project_actions(self, action_seq: torch.Tensor, embodiment_id: torch.LongTensor) -> torch.Tensor:
        if self.horizon > 1 and self.action_encoder is not None:
            return self.action_encoder(action_seq, embodiment_id)
```

## 逐行讲解 / What's happening

1. **第 110-124 行 / Lines 110-124**:
   - 中文: `MultiEmbodimentActionEncoder` 建三层 category-specific projection，并准备 horizon 长度的位置编码。
   - English: `MultiEmbodimentActionEncoder` builds three category-specific projections and a positional encoding for the action horizon.
1. **第 126-145 行 / Lines 126-145**:
   - 中文: 动作从 `[batch, horizon, action_dim]` 拉平成 token，再把 `category_id` 扩展到每个时间步；这一步让每个机器人走自己的线性权重。
   - English: Actions are flattened from `[batch, horizon, action_dim]`, and `category_id` is expanded to every timestep so each robot uses its own linear weights.
1. **第 173-218 行 / Lines 173-218**:
   - 中文: `FlowmatchingActionHead` 组装 time embedding、Transformer blocks、sequence pooling 和最终 MLP，输出整个 action horizon。
   - English: `FlowmatchingActionHead` assembles time embeddings, Transformer blocks, sequence pooling, and the final MLP that emits the whole action horizon.
1. **第 220-245 行 / Lines 220-245**:
   - 中文: 可选 state encoder 和 action encoder 把状态、动作、机器人类别统一到 `embed_dim`；单步 horizon 则退化成一个普通 Linear。
   - English: Optional state and action encoders bring state, action, and embodiment id into `embed_dim`; a one-step horizon falls back to a plain Linear.

## 类比 / The analogy

像同一个乐谱给不同乐器演奏：节拍和旋律结构共享，但小提琴、长笛、鼓组各自有自己的指法和音域映射。

It is like one score played by different instruments. The rhythm and structure are shared, but violin, flute, and drums each need their own fingering and range mapping.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

在 nanoVLA 里，这对应 `action-head-continuous` 的高级版本：输入是 VLM/context tokens、当前 noisy action、flow timestep 和 embodiment id；输出是 action velocity 或整段 action chunk。省掉 category-specific 部分，单机器人 demo 还能跑；但多机器人 production 会把不同动作空间硬塞进同一组权重，容易互相污染。

In a nanoVLA, this is an advanced `action-head-continuous` component. It consumes VLM/context tokens, noisy actions, a flow timestep, and an embodiment id, then emits action velocity or an action chunk. A single-robot demo can omit category-specific layers, but a multi-robot production system needs them to avoid forcing different action spaces through one projection.

## 自己跑一遍 / Try it yourself

```python
import math

def encode(actions, category, weights):
    out = []
    for t, value in enumerate(actions):
        pos = math.sin(t)
        scale, bias = weights[category]
        out.append(round(value * scale + bias + pos, 3))
    return out

weights = {'arm_a': (1.0, 0.0), 'arm_b': (0.5, 2.0)}
print(encode([0.2, 0.4, 0.6], 'arm_a', weights))
print(encode([0.2, 0.4, 0.6], 'arm_b', weights))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[0.2, 1.241, 1.509]
[2.1, 3.041, 3.209]
```

同一段动作因为 embodiment 权重不同，进入模型前已经变成不同 token；这就是 category-specific projection 的直觉版本。

The same action sequence becomes different tokens before entering the model because the embodiment weights differ. That is the intuition behind category-specific projection.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **GR00T action head** / **GR00T action head**: 也用多 embodiment action encoder，把机器人差异前置到动作表示里。 / It also moves robot-specific differences into the action representation.
- **openpi pi0 suffix** / **openpi pi0 suffix**: 把 state、noisy action 和 timestep 做成 action suffix tokens，再交给 expert。 / It packs state, noisy action, and timestep as suffix tokens for the expert.

## 注意事项 / Caveats / when it breaks

- **category id 必须可信** / **Category ids must be trustworthy**: 标错 embodiment 会让动作走错权重，比普通噪声更难调试。 / A wrong embodiment id routes actions through the wrong weights and is harder to debug than ordinary noise.
- **horizon 要对齐** / **The horizon must match**: 代码明确检查输入 horizon；训练和推理 horizon 不一致会直接破坏位置编码和输出形状。 / The code explicitly checks horizon; training and inference mismatches break positional encoding and output shape.

## 延伸阅读 / Further reading

- [LeRobot repository](https://github.com/huggingface/lerobot)
- [Flow matching for generative modeling](https://arxiv.org/abs/2210.02747)
