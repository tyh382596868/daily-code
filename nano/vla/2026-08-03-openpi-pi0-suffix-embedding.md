---
date: 2026-08-03
topic: vla
source: vla
repo: Physical-Intelligence/openpi
file: src/openpi/models_pytorch/pi0_pytorch.py
permalink: https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/models_pytorch/pi0_pytorch.py#L238-L315
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, vla, action-head-continuous, flow-matching]
build_role: action-head-continuous advanced variant, suffix embedding for state/action/time tokens
---

# openpi pi0 suffix embedding：动作、状态和时间步要先变成 token / openpi pi0 Suffix Embedding: Turn Action, State, and Time into Tokens First

> **一句话 / In one line**: `embed_suffix` 把 state、noisy actions 和 timestep 投影到 transformer 宽度，并构造 action suffix 的 mask。 / `embed_suffix` projects state, noisy actions, and timestep to transformer width and builds masks for the action suffix.

## 为什么重要 / Why this matters

连续动作 VLA 不是直接把浮点动作塞进语言模型。它需要一层“翻译器”：把机器人状态、带噪动作和扩散时间步转成和 VLM token 同宽的向量，交给 action expert 去更新。

A continuous-action VLA cannot feed raw floats directly into a language model. It needs a translator that turns robot state, noisy actions, and diffusion time into vectors with the same width as transformer tokens.

## 代码 / The code

`Physical-Intelligence/openpi` — [`src/openpi/models_pytorch/pi0_pytorch.py`](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/models_pytorch/pi0_pytorch.py#L238-L315)

```python
def embed_suffix(self, state, noisy_actions, timestep):
    """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
    embs = []
    pad_masks = []
    att_masks = []

    if not self.pi05:
        if self.state_proj.weight.dtype == torch.float32:
            state = state.to(torch.float32)

        def state_proj_func(state):
            return self.state_proj(state)

        state_emb = self._apply_checkpoint(state_proj_func, state)

        embs.append(state_emb[:, None, :])
        bsize = state_emb.shape[0]
        device = state_emb.device

        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        att_masks += [1]

    time_emb = create_sinusoidal_pos_embedding(
        timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
    )
    time_emb = time_emb.type(dtype=timestep.dtype)

    def action_proj_func(noisy_actions):
        return self.action_in_proj(noisy_actions)

    action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

    if not self.pi05:
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        def mlp_func(action_time_emb):
            x = self.action_time_mlp_in(action_time_emb)
            x = F.silu(x)
            return self.action_time_mlp_out(x)

        action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
        adarms_cond = None
    else:
        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
        action_time_emb = action_emb
        adarms_cond = time_emb

    embs.append(action_time_emb)

    bsize, action_time_dim = action_time_emb.shape[:2]
    action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
    pad_masks.append(action_time_mask)

    att_masks += [1] + ([0] * (self.config.action_horizon - 1))

    embs = torch.cat(embs, dim=1)
    pad_masks = torch.cat(pad_masks, dim=1)
    att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
    att_masks = att_masks[None, :].expand(bsize, len(att_masks))

    return embs, pad_masks, att_masks, adarms_cond
```

## 逐行讲解 / What's happening

1. **第 244-263 行 / Lines 244-263 (state token)**:
   - 中文: state 先过 `state_proj`，再作为一个 token 放进 suffix。
   - English: The state vector goes through `state_proj` and becomes one token in the suffix.
2. **第 264-286 行 / Lines 264-286 (time + action)**:
   - 中文: timestep 走 sin/cos embedding，动作走 linear，然后二者拼接进 MLP。
   - English: The timestep becomes a sin/cos embedding, actions go through a linear layer, and both are fused by an MLP.
3. **第 300-315 行 / Lines 300-315 (masks)**:
   - 中文: action token 都是有效 token；`att_masks` 定义 suffix 内的依赖边界。
   - English: Action tokens are all valid; `att_masks` defines dependency boundaries inside the suffix.

## 类比 / The analogy

这像机器人进会议前要先领翻译耳机：关节角、噪声动作和时间步都不是自然语言，必须先翻译成会议系统听得懂的频道。

It is like giving a robot a translation headset before a meeting: joint state, noisy actions, and time are not natural language, so they must become channels the meeting system understands.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

这是 `action-head-continuous` 的 advanced variant。上游是视觉/语言 prefix、机器人 state、当前 noisy action 和扩散时间步；下游是 transformer/action expert。省掉它，连续动作就没有稳定入口，模型只能看到离散文本 token。生产级实现还要处理多机器人 action dim、mask、dtype 和 checkpointing。

This is an advanced variant of `action-head-continuous`. Upstream are the visual/language prefix, robot state, current noisy action, and diffusion timestep; downstream is the transformer action expert. Without this module, continuous actions have no stable entry point. A production version also needs multi-robot action dimensions, masks, dtype handling, and checkpointing.

## 自己跑一遍 / Try it yourself

```python
import math

def embed_suffix(state, actions, t):
    time = [math.sin(t), math.cos(t)]
    state_tok = [state[0], state[1], 0.0, 0.0]
    action_toks = []
    for a in actions:
        action_toks.append([a[0], a[1], time[0], time[1]])
    att = [1] + [1] + [0] * (len(actions) - 1)
    return [state_tok] + action_toks, att

tokens, mask = embed_suffix([0.2, -0.1], [[0.0, 0.5], [0.1, 0.4]], 0.25)
print(tokens)
print(mask)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[[0.2, -0.1, 0.0, 0.0], [0.0, 0.5, 0.24740395925452294, 0.9689124217106447], [0.1, 0.4, 0.24740395925452294, 0.9689124217106447]]
[1, 1, 0]
```

中文: 重点是 action token 里同时含有动作值和时间条件。  
English: The key is that each action token carries both action values and time conditioning.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LeRobot pi0 denoise loop** / **LeRobot pi0 denoise loop**: 同样把 action suffix 迭代更新。 / It also iteratively updates an action suffix.
- **Diffusion Policy action head** / **Diffusion Policy action head**: 也把动作轨迹当成带噪序列来去噪。 / It also denoises an action trajectory as a sequence.

## 注意事项 / Caveats / when it breaks

- **mask 语义要统一** / **Mask semantics must match**: `att_masks` 后面会变成 2D attention mask，含义错了就会泄漏未来动作。 / `att_masks` later becomes a 2D attention mask; wrong semantics leak future actions.
- **时间尺度很敏感** / **Time scale is sensitive**: `min_period` 和 `max_period` 决定模型能分辨的时间频率。 / `min_period` and `max_period` determine the time frequencies the model can resolve.

## 延伸阅读 / Further reading

- openpi PyTorch pi0 source: https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/models_pytorch/pi0_pytorch.py#L238-L315
