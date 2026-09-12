---
date: 2026-09-12
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/policies/groot/action_head/cross_attention_dit.py
permalink: https://github.com/huggingface/lerobot/blob/b6ec0060779550c0a157ae34feb89e0cf86012a8/src/lerobot/policies/groot/action_head/cross_attention_dit.py#L50-L87
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, action-head, flow-matching, ada-norm, timestep-conditioning]
build_role: action-head-continuous advanced variant
---

# GR00T 时间条件：把 flow step 变成 AdaNorm 调制 / GR00T Time Conditioning: Turn the Flow Step into AdaNorm Modulation

> **一句话 / In one line**: GR00T 先把 timestep 编成向量，再用它生成 scale/shift 去调制动作 token 的归一化结果。 / GR00T encodes the timestep, then turns it into scale/shift values that modulate normalized action tokens.

## 为什么重要 / Why this matters

中文：连续动作 head 的输入不只是一段 noisy action，还包括“现在正在第几步去噪”。如果 timestep 只是一个旁路 scalar，Transformer 很难在每一层稳定地使用它。GR00T 把时间步编码成和 hidden state 同宽的 `temb`，再用 AdaNorm 将它注入 token 流，使同一套动作 head 可以在不同噪声强度下工作。

English: A continuous action head consumes more than a noisy action sequence; it also needs to know which denoising step it is on. Passing a scalar timestep through a side channel is awkward for every transformer block. GR00T turns the timestep into a hidden-width `temb`, then injects it through AdaNorm so one action head can operate at different noise levels.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/groot/action_head/cross_attention_dit.py`](https://github.com/huggingface/lerobot/blob/b6ec0060779550c0a157ae34feb89e0cf86012a8/src/lerobot/policies/groot/action_head/cross_attention_dit.py#L50-L87)

```python
class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim, compute_dtype=torch.float32):
        require_package("diffusers", extra="groot")
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps):
        dtype = next(self.parameters()).dtype
        timesteps_proj = self.time_proj(timesteps).to(dtype)
        timesteps_emb = self.timestep_embedder(timesteps_proj)  # (N, D)
        return timesteps_emb


class AdaLayerNorm(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()
        self.chunk_dim = chunk_dim
        output_dim = embedding_dim * 2
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        temb = self.linear(self.silu(temb))
        scale, shift = temb.chunk(2, dim=1)
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x
```

## 逐行讲解 / What's happening

1. **第 50-55 行 / Lines 50-55 (`TimestepEncoder.__init__`)**:
   - 中文：时间编码分两层：先把 scalar timestep 展成 sinusoidal features，再用 learned embedding 投影到动作 head 的 hidden width。
   - English: Time encoding has two stages: expand the scalar timestep into sinusoidal features, then learn a projection into the action-head hidden width.
2. **第 57-61 行 / Lines 57-61 (`forward`)**:
   - 中文：`next(self.parameters()).dtype` 让 timestep embedding 跟当前模型权重对齐，避免混合精度下出现 dtype mismatch。
   - English: Reading the parameter dtype keeps the timestep path aligned with the model weights, avoiding mixed-precision mismatches.
3. **第 73-77 行 / Lines 73-77 (twice the width)**:
   - 中文：AdaNorm 的线性层输出 `2 * embedding_dim`，因为后面要拆成一半 scale、一半 shift。
   - English: The AdaNorm projection emits `2 * embedding_dim` because the result is split into a scale half and a shift half.
4. **第 79-86 行 / Lines 79-86 (`temb` to modulation)**:
   - 中文：先过 SiLU 和 Linear，再按 batch 维拆成 `scale` 与 `shift`。`scale[:, None]` 在 token 维广播，于是同一个 timestep 可以调制整段 action sequence。
   - English: The timestep goes through SiLU and Linear, then splits into `scale` and `shift`. `scale[:, None]` broadcasts over tokens, so one timestep modulates the whole action sequence.
5. **第 86 行 / Line 86 (the residual-friendly formula)**:
   - 中文：`1 + scale` 保留 identity 起点；当 scale 接近零时，调制不会把归一化特征整体抹掉。
   - English: `1 + scale` keeps an identity-like starting point. When scale is near zero, modulation does not erase the normalized features.

## 类比 / The analogy

中文：把 timestep 想成“去雾旋钮”。动作 token 是一排零件，LayerNorm 先把零件尺寸标准化；timestep 再告诉工人当前雾气有多浓，从而统一调高或调低这一整排零件的处理力度。

English: Think of the timestep as a “defogging knob.” The action tokens are a row of parts; LayerNorm standardizes their scale, and the timestep tells the worker how aggressively to adjust the whole row for the current noise level.

## 在 nanoVLA 中的位置 / Where this lives in your nanoVLA

中文：这是 `action-head-continuous` 的 advanced variant，位于 noisy action token 化之后、Transformer block 内部。上游给它 `[B, H, D]` 的动作 hidden states 和 `[B]` 的 flow timestep；它输出带时间条件的 `[B, H, D]` token，继续交给 self-attention、cross-attention 和最终 action projection。省掉这条路径，模型就只能把不同去噪阶段混在一起。生产版还要补上 timestep dtype/device 管理、action horizon mask、不同机器人 action width、以及训练和采样使用同一套时间参数化。

English: This is an advanced `action-head-continuous` component inside the transformer, after noisy actions have become hidden tokens. The upstream path provides `[B, H, D]` action states and a `[B]` flow timestep; the module returns timestep-conditioned `[B, H, D]` states for attention and the final action projection. Without it, denoising stages become ambiguous. Production code also needs dtype/device handling, horizon masks, robot-specific action widths, and one consistent time parameterization for training and sampling.

## 自己跑一遍 / Try it yourself

```python
def ada_norm(tokens, temb):
    mean = sum(tokens) / len(tokens)
    centered = [x - mean for x in tokens]
    scale, shift = temb, temb * 0.5
    return [(1 + scale) * x + shift for x in centered]

print([round(x, 2) for x in ada_norm([1.0, 2.0, 4.0], 0.2)])
print([round(x, 2) for x in ada_norm([1.0, 2.0, 4.0], 1.0)])
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
[-1.5, -0.3, 2.1]
[-2.17, -0.17, 3.83]
```

中文：同一组 token 在不同 timestep 下会得到不同的整体调制；真实模型只是把 scalar `temb` 换成 learned vector，并在 token 维广播。

English: The same token sequence receives a different global modulation at each timestep. The real model replaces the scalar `temb` with a learned vector and broadcasts it across tokens.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **DiT adaLN-Zero** / **DiT adaLN-Zero**: 用时间或类别条件生成 shift、scale 和 gate。 / Generate shift, scale, and gate values from time or class conditions.
- **openpi flow suffix** / **openpi flow suffix**: 把 noisy action 和 timestep 接进同一个 suffix 表示。 / Put noisy actions and timestep into one suffix representation.
- **Wan2.1 block modulation** / **Wan2.1 block modulation**: 一个条件向量分出多组 block-level modulation。 / Split one condition vector into several block-level modulation groups.

## 注意事项 / Caveats / when it breaks

- **`temb` 不能是错 batch size** / **`temb` must have the right batch size**: `[B, D]` 和 `[B, H, D]` 的广播关系是设计的一部分。
- **LayerNorm 不是 timestep encoder** / **LayerNorm is not the timestep encoder**: 前者稳定 token，后者告诉模型当前噪声阶段。
- **scale/shift 需要和采样器匹配** / **Scale and shift must match the sampler**: 训练用 flow time、采样用 diffusion sigma 时要定义清楚映射。

## 延伸阅读 / Further reading

- [LeRobot GR00T cross-attention DiT](https://github.com/huggingface/lerobot/blob/b6ec0060779550c0a157ae34feb89e0cf86012a8/src/lerobot/policies/groot/action_head/cross_attention_dit.py)
- [Diffusers AdaLayerNorm patterns](https://github.com/huggingface/diffusers)
