---
date: 2026-09-13
topic: wam
source: wam
repo: huggingface/lerobot
file: src/lerobot/policies/fastwam/wan/modular.py
permalink: https://github.com/huggingface/lerobot/blob/b6ec0060779550c0a157ae34feb89e0cf86012a8/src/lerobot/policies/fastwam/wan/modular.py#L132-L203
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, action-conditioning, dit, shape-contract, rope]
build_role: action-conditioning advanced variant
---

# FastWAM ActionDiT：先验证契约，再把动作、文字和时间装进 token / FastWAM ActionDiT: Validate the Contract Before Packing Action, Text, and Time

> **一句话 / In one line**: FastWAM 的 `pre_dit` 先把 batch、长度、mask 和 RoPE 边界检查干净，再把动作、文本和 timestep 变成 DiT 可以消费的统一状态。 / FastWAM's `pre_dit` validates batch, length, mask, and RoPE boundaries before packing action, text, and timestep into a DiT-ready state.

## 为什么重要 / Why this matters

中文：WAM 的动作 expert 同时接收连续 action token、视频或语言 context、以及 flow timestep。若这些输入在进入 DiT 前没有统一 shape、dtype 和 mask 语义，错误往往只会在 attention kernel 里以难读的 CUDA 报错出现。`pre_dit` 把边界契约集中在一个小函数里，并返回一个显式的 `pre_state` 给后续 block。

English: A WAM action expert consumes continuous action tokens, video or language context, and a flow timestep at once. Without normalized shapes, dtypes, and mask semantics before the DiT, failures often surface as opaque CUDA errors inside attention kernels. `pre_dit` centralizes the boundary contract and returns an explicit `pre_state` for later blocks.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/fastwam/wan/modular.py`](https://github.com/huggingface/lerobot/blob/b6ec0060779550c0a157ae34feb89e0cf86012a8/src/lerobot/policies/fastwam/wan/modular.py#L132-L203)

```python
def pre_dit(
    self,
    action_tokens: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    if action_tokens.ndim != 3:
        raise ValueError(
            f"`action_tokens` must be 3D [B, T, action_dim], got shape {tuple(action_tokens.shape)}"
        )
    if action_tokens.shape[2] != self.action_dim:
        raise ValueError(
            f"`action_tokens` last dim must be {self.action_dim}, got {action_tokens.shape[2]}"
        )
    if timestep.ndim != 1:
        raise ValueError(f"`timestep` must be 1D [B] or [1], got shape {tuple(timestep.shape)}")
    if context.ndim != 3:
        raise ValueError(f"`context` must be 3D [B, L, D], got shape {tuple(context.shape)}")

    batch_size = action_tokens.shape[0]
    if context.shape[0] != batch_size:
        raise ValueError(
            f"Batch mismatch between action tokens and text context: {batch_size} vs {context.shape[0]}"
        )
    if timestep.shape[0] not in (1, batch_size):
        raise ValueError(
            f"`timestep` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}"
        )
    if timestep.shape[0] == 1 and batch_size > 1:
        if self.training:
            raise ValueError("During training, action timestep length must match batch_size.")
        timestep = timestep.expand(batch_size)

    if context_mask is None:
        context_mask = torch.ones((batch_size, context.shape[1]), dtype=torch.bool, device=context.device)
    else:
        if context_mask.ndim != 2:
            raise ValueError(f"`context_mask` must be 2D [B, L], got shape {tuple(context_mask.shape)}")
        if context_mask.shape[0] != batch_size or context_mask.shape[1] != context.shape[1]:
            raise ValueError(
                f"`context_mask` shape must match `context` shape [B, L], got {tuple(context_mask.shape)} vs {tuple(context.shape)}"
            )

    seq_len = action_tokens.shape[1]
    if seq_len > self.freqs.shape[0]:
        raise ValueError(f"Action token length {seq_len} exceeds RoPE cache {self.freqs.shape[0]}.")

    model_dtype = self.action_encoder.weight.dtype
    action_tokens = action_tokens.to(dtype=model_dtype)
    context = context.to(dtype=model_dtype)
    t_emb = sinusoidal_embedding_1d(self.freq_dim, timestep).to(dtype=model_dtype)
    t = self.time_embedding(t_emb)
    t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))

    tokens = self.action_encoder(action_tokens)
    context_emb = self.text_embedding(context)
    context_attn_mask = context_mask.unsqueeze(1).expand(-1, seq_len, -1)
    freqs = self.freqs[:seq_len].view(seq_len, 1, -1).to(tokens.device)

    return {
        "tokens": tokens,
        "freqs": freqs,
        "t": t,
        "t_mod": t_mod,
        "context": context_emb,
        "context_mask": context_attn_mask,
        "meta": {
            "batch_size": batch_size,
            "seq_len": seq_len,
        },
    }
```

## 逐行讲解 / What's happening

1. **第 139-150 行 / Lines 139-150 (rank and width)**:
   - 中文：动作是 `[B, T, action_dim]`，context 是 `[B, L, D]`，timestep 是 `[B]` 或 `[1]`。这些 rank 检查把错误挡在 kernel 之前。
   - English: Actions are `[B, T, action_dim]`, context is `[B, L, D]`, and timestep is `[B]` or `[1]`. Rank checks stop bad inputs before they reach a kernel.
2. **第 152-165 行 / Lines 152-165 (batch and timestep policy)**:
   - 中文：推理时一个 scalar timestep 可以广播到整个 batch；训练时要求每个样本显式提供 timestep，避免无意中让 batch 内样本共享噪声阶段。
   - English: In evaluation, one scalar timestep can broadcast across the batch. During training, every sample must provide its own timestep to avoid accidental shared noise stages.
3. **第 166-174 行 / Lines 166-174 (mask default)**:
   - 中文：没有 context mask 就默认所有 context 有效；有 mask 时同时检查 batch 和长度，保证后面扩展到 action query 维不会错位。
   - English: With no context mask, all context is valid. When a mask is supplied, both batch and length are checked before expanding it over action queries.
4. **第 176-190 行 / Lines 176-190 (time, token, and RoPE preparation)**:
   - 中文：先检查 action 序列不超过 RoPE cache，再统一 dtype；timestep 产生六组 modulation，action/context 分别进入自己的 embedding，mask 扩展到 `[B, T, L]`。
   - English: The action sequence is checked against the RoPE cache, then dtypes are aligned. The timestep produces six modulation vectors; action and context get separate embeddings, and the mask expands to `[B, T, L]`.
5. **第 192-203 行 / Lines 192-203 (explicit pre-state)**:
   - 中文：返回值不是一个模糊 tuple，而是带 `tokens`、`freqs`、`t_mod`、context 和 metadata 的字典。后续 DiT block 不需要重新推断输入契约。
   - English: The return value is an explicit dictionary containing tokens, RoPE frequencies, modulation, context, and metadata. Later DiT blocks do not need to rediscover the input contract.

## 类比 / The analogy

中文：像机场安检和登机牌系统。先检查每个人的证件、队伍长度和登机口，再把旅客分成座位、行李和航班信息。过了这一关，后面的飞机装载流程可以假设输入已经整齐。

English: Think of airport security and boarding passes. First validate identity, queue length, and gate; then package passengers into seats, luggage, and flight metadata. After that boundary, aircraft loading can assume the inputs are consistent.

## 在 nanoWAM 中的位置 / Where this lives in your nanoWAM

中文：这是 `action-conditioning` 的 advanced variant，位于 noisy action 进入 DiT block 之前。输入是 `[B, T, action_dim]` 的动作、`[B, L, D]` 的视频/语言 context 和 timestep；输出是带 RoPE、六路时间调制、context mask 的 pre-state，供 action expert 和 video expert 的联合 attention 消费。实现 nanoWAM 时可把它拆成 `validate_inputs`、`embed_action`、`embed_context`、`make_time_modulation` 四个纯函数，再由 `pre_dit` 组合。生产版还要测试 train/eval timestep 广播、短 context、超长 action horizon、padding mask 和不同 dtype。

English: This is an advanced `action-conditioning` component immediately before noisy actions enter the DiT blocks. It consumes action tokens, video/language context, and a timestep, then returns a RoPE-aware pre-state with six-way time modulation and context masks for joint action/video attention. In nanoWAM, split it into pure `validate_inputs`, `embed_action`, `embed_context`, and `make_time_modulation` helpers, then compose them in `pre_dit`. Production tests should cover train/eval timestep broadcasting, short context, long horizons, padding masks, and mixed dtypes.

## 自己跑一遍 / Try it yourself

```python
def pre_state(actions, timestep, context, training=False):
    batch, length, width = actions
    if len(timestep) not in (1, batch):
        raise ValueError("bad timestep batch")
    if len(timestep) == 1 and batch > 1 and training:
        raise ValueError("training needs per-sample timestep")
    if len(context) != batch:
        raise ValueError("context batch mismatch")
    if len(timestep) == 1:
        timestep = timestep * batch
    mask = [[True] * len(context[0]) for _ in range(batch)]
    return {"shape": (batch, length, width), "timestep": timestep, "mask": mask}


print(pre_state((2, 4, 7), [0.5], [[1, 2], [3, 4]]))
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
{'shape': (2, 4, 7), 'timestep': [0.5, 0.5], 'mask': [[True, True], [True, True]]}
```

中文：推理允许一个 timestep 广播到两个样本，mask 也按 context 长度扩展。真实实现再把这些 Python 列表换成 tensor、embedding 和 RoPE 频率。

English: Evaluation allows one timestep to broadcast to two samples, and the mask expands to the context length. The real implementation replaces these lists with tensors, embeddings, and RoPE frequencies.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **FastWAM MoT** / **FastWAM MoT**: 每个 expert 先完成自己的 QKV 准备，再交给共享 attention。 / Each expert prepares its own QKV before shared attention.
- **Wan2.1 block modulation** / **Wan2.1 block modulation**: timestep 产生多组 adaLN 调制向量。 / The timestep produces multiple adaLN modulation vectors.
- **Open-Sora attention metadata** / **Open-Sora attention metadata**: kernel 前先把 mask、长度和布局变成显式 metadata。 / Convert masks, lengths, and layout into explicit metadata before the kernel.

## 注意事项 / Caveats / when it breaks

- **训练和推理的广播规则不同** / **Train and eval broadcasting differ**: 这是刻意的契约，不应在重构时悄悄统一。
- **mask 的维度不是装饰** / **Mask dimensions are semantic**: `[B, L]` 扩成 `[B, T, L]` 后，每个 action query 才能看到同一组有效 context。
- **RoPE cache 是硬上限** / **The RoPE cache is a hard limit**: action horizon 改大时必须同步扩大频率缓存。
- **六路 modulation 要和 block 对齐** / **Six-way modulation must match the block**: 改变 DiT block 的归一化或 gate 数量时，`unflatten(1, (6, hidden_dim))` 也要一起改。

## 延伸阅读 / Further reading

- [LeRobot FastWAM modular ActionDiT](https://github.com/huggingface/lerobot/blob/b6ec0060779550c0a157ae34feb89e0cf86012a8/src/lerobot/policies/fastwam/wan/modular.py)
- [Wan2.1 DiT conditioning](https://github.com/Wan-Video/Wan2.1)
