---
date: 2026-08-13
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/policies/evo1/flow_matching.py
permalink: https://github.com/huggingface/lerobot/blob/a16f34c085c9597fcbdb9fde395a3334d78df716/src/lerobot/policies/evo1/flow_matching.py#L250-L390
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, vla, flow-matching]
build_role: action-head-continuous advanced variant, context preparation and action masks
---

# LeRobot Evo1 action mask：动作 head 也要知道哪些维度有效 / LeRobot Evo1 Action Mask: The Action Head Must Know Which Dimensions Are Valid

> **一句话 / In one line**: Evo1 的 flow-matching head 先统一 VL context、state、embodiment id 和 action mask，再只在有效动作维度上构造噪声插值目标。 / Evo1 normalizes VL context, state, embodiment id, and action mask before constructing noisy interpolation targets only on valid action dimensions.

## 为什么重要 / Why this matters

真实 VLA 数据集很少拥有完全相同的动作维度。这里的关键不是 flow-matching 公式本身，而是训练前把边界条件整理成 action head 能稳定消费的 contract。

Real VLA datasets rarely share identical action vectors. The key lesson is not just the flow-matching formula; it is the boundary contract that makes the action head robust before the loss is formed.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/evo1/flow_matching.py`](https://github.com/huggingface/lerobot/blob/a16f34c085c9597fcbdb9fde395a3334d78df716/src/lerobot/policies/evo1/flow_matching.py#L250-L390)

```python
def _expand_action_mask(
    self,
    action_mask: torch.Tensor,
    batch_size: int,
    per_action_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if action_mask is None:
        raise ValueError("action_mask must be provided for flow matching inference.")

    if action_mask.dim() == 2:
        expected_last_dim = self.horizon * per_action_dim
        if action_mask.shape == (batch_size, expected_last_dim):
            expanded_mask = action_mask.reshape(batch_size, self.horizon, per_action_dim)
        elif action_mask.shape == (batch_size, per_action_dim):
            expanded_mask = action_mask.unsqueeze(1).expand(batch_size, self.horizon, per_action_dim)
        else:
            raise ValueError(
                f"Expected action_mask shape {(batch_size, expected_last_dim)} or "
                f"{(batch_size, per_action_dim)}, got {tuple(action_mask.shape)}"
            )
    elif action_mask.dim() == 3:
        expected_shape = (batch_size, self.horizon, per_action_dim)
        if tuple(action_mask.shape) != expected_shape:
            raise ValueError(
                f"Expected action_mask shape {expected_shape}, got {tuple(action_mask.shape)}"
            )
        expanded_mask = action_mask
    else:
        raise ValueError(f"Unsupported action_mask rank: {action_mask.dim()}")

    return expanded_mask.to(device=device, dtype=dtype)

def _prepare_context(
    self,
    fused_tokens: torch.Tensor,
    state: torch.Tensor | None,
    embodiment_id: torch.LongTensor | None,
    context_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.LongTensor]:
    """Normalize the VL context and embodiment ids shared by training and inference.

    Returns the context tokens ``(B, S, E)``, a key_padding_mask for
    ``nn.MultiheadAttention`` (True = ignore) or None, and the resolved embodiment ids.
    """
    batch_size = fused_tokens.size(0)
    device = fused_tokens.device
    if embodiment_id is None:
        embodiment_id = torch.zeros(batch_size, dtype=torch.long, device=device)
    elif self.num_categories > 1 and (
        int(embodiment_id.min()) < 0 or int(embodiment_id.max()) >= self.num_categories
    ):
        raise ValueError(
            f"embodiment ids must be in [0, num_categories={self.num_categories}), "
            f"got range [{int(embodiment_id.min())}, {int(embodiment_id.max())}]"
        )

    context_tokens = fused_tokens
    if context_tokens.dim() == 2:
        # A single pooled VL token (return_cls_only): give it a sequence dim of 1.
        context_tokens = context_tokens.unsqueeze(1)
        context_mask = None
    if state is not None and self.state_encoder is not None:
        state_emb = self.state_encoder(state, embodiment_id).unsqueeze(1)
        context_tokens = torch.cat([context_tokens, state_emb], dim=1)
        if context_mask is not None:
            state_valid = torch.ones(batch_size, 1, dtype=torch.bool, device=context_mask.device)
            context_mask = torch.cat([context_mask.to(torch.bool), state_valid], dim=1)

    key_padding_mask = None if context_mask is None else ~context_mask.to(torch.bool)
    return context_tokens, key_padding_mask, embodiment_id

# ... forward() samples flow time, masks action/noise, then projects action tokens ...
t = (
    torch.distributions.Beta(2, 2)
    .sample((batch_size,))
    .clamp(0.02, 0.98)
    .to(device)
    .to(dtype=self.dtype)
)
time_index = (t * 999).long().clamp_(0, 999)
time_emb = self.time_pos_enc(1000)[:, time_index, :].squeeze(0).to(dtype=context_tokens.dtype)

actions_gt_seq = actions_gt
noise = torch.rand_like(actions_gt) * 2 - 1
if action_mask is not None:
    action_mask = action_mask.to(dtype=noise.dtype, device=noise.device)
    if action_mask.shape != noise.shape:
        raise ValueError(f"action_mask shape {action_mask.shape} != noise shape {noise.shape}")
    actions_gt_seq = actions_gt_seq * action_mask
    noise = noise * action_mask

if self.horizon > 1:
    noise_seq = noise.view(batch_size, self.horizon, self.per_action_dim)
else:
    noise_seq = noise if noise.dim() == 3 else noise.unsqueeze(1)
t_broadcast = t.view(batch_size, 1, 1)
action_intermediate_seq = (1 - t_broadcast) * noise_seq + t_broadcast * actions_gt_seq

action_tokens = self._project_actions(action_intermediate_seq, embodiment_id)
```

## 逐行讲解 / What's happening

1. **第 250-282 行 / Lines 250-282: mask 可以是扁平动作向量，也可以是 `(B,H,D)`，最终统一成 horizon-aware 张量。 / The mask can be flat or `(B,H,D)`, and is normalized into a horizon-aware tensor.**
2. **第 296-321 行 / Lines 296-321: 单 token context 补 sequence 维；state 会变成额外 context token。 / A single context token gets a sequence dimension; state becomes an extra context token.**
3. **第 347-355 行 / Lines 347-355: `Beta(2,2)` 避开极端时间步，再离散成 time embedding index。 / `Beta(2,2)` avoids extreme times, then discretizes into a time-embedding index.**
4. **第 357-373 行 / Lines 357-373: ground truth 和 noise 乘同一个 action mask，缺失维度不参与目标。 / Ground truth and noise share the same action mask, so missing dimensions do not enter the target.**

## 类比 / The analogy

像给不同机器人填同一张训练表：缺胳膊的列先划掉，额外状态另加一列，老师只批改真正填写过的格子。

It is like using one worksheet for different robots: missing-joint columns are crossed out, extra state gets its own column, and grading only checks filled cells.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

这是 `action-head-continuous` 的 advanced variant，依赖 `vlm-backbone-wiring` 和 `training-step`。在 nanoVLA 里，它位于 VLM 输出之后、loss 之前：输入是 `fused_tokens/state/action_gt/action_mask/embodiment_id`。省掉 mask，跨机器人训练会把不存在的动作维度当成 0 目标。

This is an advanced `action-head-continuous` variant, depending on `vlm-backbone-wiring` and `training-step`. In a nanoVLA, it sits after VLM features and before the loss, consuming `fused_tokens/state/action_gt/action_mask/embodiment_id`. Without the mask, cross-robot training treats nonexistent action dimensions as zero targets.

## 自己跑一遍 / Try it yourself

```python
def expand_mask(mask, horizon, dim):
    if len(mask[0]) == horizon * dim:
        return [[row[i*dim:(i+1)*dim] for i in range(horizon)] for row in mask]
    if len(mask[0]) == dim:
        return [[row[:] for _ in range(horizon)] for row in mask]
    raise ValueError("bad mask shape")

def interpolate(noise, target, mask, t):
    return [[(1-t)*n*m + t*a*m for n, a, m in zip(nr, ar, mr)]
            for nr, ar, mr in zip(noise, target, mask)]

mask = expand_mask([[1, 0]], horizon=2, dim=2)[0]
noise = [[-1, 9], [1, 9]]
target = [[3, 7], [5, 7]]
print(interpolate(noise, target, mask, t=0.25))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[[0.0, 0.0], [2.0, 0.0]]
```

第二个动作维度被 mask 掉后，无论 noise 和 target 是多少，训练目标都保持 0。

Once the second action dimension is masked out, the training target stays zero no matter what noise and target values were.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **GR00T state/action masks** / **GR00T state/action masks**: heterogeneous batch 也要明确哪些维度有效。 / Heterogeneous batches also need explicit validity masks.
- **OpenVLA action bins** / **OpenVLA action bins**: 离散 action head 也需要知道哪些输出维度参与训练。 / Discrete action heads also need to know which output dimensions are trained.

## 注意事项 / Caveats / when it breaks

- **mask 形状要和 horizon 对齐** / **Mask shape must match horizon**: 扁平 mask 和逐步 mask 含义不同。 / Flat masks and per-step masks mean different things.
- **`embodiment_id` 是索引** / **`embodiment_id` is an index**: 它会进入 category-specific layer，越界会报错。 / It feeds category-specific layers and must stay in range.

## 延伸阅读 / Further reading

- [huggingface/lerobot source](https://github.com/huggingface/lerobot/blob/a16f34c085c9597fcbdb9fde395a3334d78df716/src/lerobot/policies/evo1/flow_matching.py#L250-L390)
