---
date: 2026-08-01
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/policies/pi0/modeling_pi0.py
permalink: https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi0/modeling_pi0.py#L587-L678
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, vla, inference-loop, flow-matching]
build_role: inference-loop advanced variant, iterative action denoising
---

# LeRobot pi0 denoise loop：从纯噪声动作一路积分出来 / LeRobot pi0 Denoise Loop: Integrate Actions Out of Noise

> **一句话 / In one line**: `sample_actions` 先生成高斯噪声动作，再按离散时间步反复预测速度场并更新动作。 / `sample_actions` starts from Gaussian action noise, repeatedly predicts a velocity field over discrete timesteps, and updates the action tensor.

## 为什么重要 / Why this matters

连续动作 VLA 的推理不是一次 `linear(hidden)` 就结束。pi0 这类 flow-matching 策略把动作当作需要逐步去噪的轨迹；语言和图像前缀提供条件，动作后缀在每个时间步被改写。

Continuous-action VLA inference is not just one `linear(hidden)` call. Flow-matching policies such as pi0 treat actions as trajectories to denoise; language and image prefixes provide context while the action suffix is rewritten at each timestep.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/pi0/modeling_pi0.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi0/modeling_pi0.py#L587-L678)

```python
def sample_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
    """Do a full inference forward and return denoised actions."""
    device = next(self.parameters()).device
    if noise is None:
        batch_size = batch[OBS_STATE].shape[0]
        noise = torch.randn(
            batch_size,
            self.config.n_action_steps,
            self.config.max_action_dim,
            device=device,
        )

    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(batch)
    suffix_pad_masks, suffix_att_masks = self.prepare_action_masks(batch)
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    action = noise
    dt = -1.0 / self.config.num_steps
    dt = torch.tensor(dt, dtype=torch.float32, device=device)

    for step in range(self.config.num_steps):
        timestep = torch.full((action.shape[0],), step / self.config.num_steps, device=device)
        suffix_embs = self.embed_suffix(batch, action, timestep)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat(
            [
                torch.cat([prefix_att_2d_masks, torch.zeros_like(prefix_att_2d_masks[:, :, : suffix_embs.shape[1]])], dim=2),
                torch.cat([torch.ones_like(suffix_att_2d_masks[:, :, : prefix_embs.shape[1]]), suffix_att_2d_masks], dim=2),
            ],
            dim=1,
        )
        embs = torch.cat([prefix_embs, suffix_embs], dim=1)
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        position_ids = position_ids.clamp(min=0)
        _, action_hidden_states = self.model.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=embs,
            use_cache=False,
        )
        action_vel = self.action_out_proj(action_hidden_states[:, -self.config.n_action_steps :])
        action = action + dt * action_vel

    return action
```

## 逐行讲解 / What's happening

1. **第 590-599 行 / Lines 590-599 (noise start)**:
   - 中文: 没传入噪声时，函数按 batch、action horizon 和动作维度采样一个标准高斯动作张量。
   - English: When no noise is supplied, the function samples a Gaussian action tensor shaped by batch, action horizon, and action dimension.
2. **第 601-606 行 / Lines 601-606 (prefix context)**:
   - 中文: 图像、语言、状态等条件先被嵌入成 prefix，并建立 prefix attention mask。
   - English: Image, language, and state context are embedded as a prefix with its own attention mask.
3. **第 610-635 行 / Lines 610-635 (denoise loop)**:
   - 中文: 每个 step 把当前 action 和 timestep 编成 suffix，再和 prefix 拼成一次完整 forward。
   - English: Each step embeds the current action and timestep as a suffix, then concatenates it with the prefix for one full forward pass.
4. **第 636-678 行 / Lines 636-678 (Euler update)**:
   - 中文: action head 输出速度场 `action_vel`，再用固定 `dt` 积分到下一版 action。
   - English: The action head emits a velocity field `action_vel`, then a fixed `dt` integrates to the next action estimate.

## 类比 / The analogy

这像雕塑家从一块粗糙石头开始，每一轮只削一点。语言和图像告诉他要雕什么，当前动作就是还没完成的石头。

It is like a sculptor starting from rough stone and shaving it a little each round. Language and images describe the target, while the current action is the unfinished stone.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

这是 `inference-loop` 的 advanced variant，依赖 `vlm-backbone-wiring` 和 `action-head-continuous`。上游是 observation batch 和任务文本，下游是机器人控制器要执行的动作 chunk；如果省掉循环，只做一次预测，flow-matching head 就失去“逐步去噪”的语义。

This is an advanced variant of the `inference-loop` component, depending on `vlm-backbone-wiring` and `action-head-continuous`. Upstream is the observation batch plus task text; downstream is the action chunk for the robot controller. If you skip the loop and predict once, the flow-matching head loses its iterative denoising meaning.

## 自己跑一遍 / Try it yourself

```python
action = [1.0, -1.0]
steps = 4
dt = -1.0 / steps
for step in range(steps):
    t = step / steps
    velocity = [a - target for a, target in zip(action, [0.2, 0.4])]
    action = [a + dt * v for a, v in zip(action, velocity)]
    print(step, [round(x, 3) for x in action])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
0 [0.8, -0.65]
1 [0.65, -0.388]
2 [0.537, -0.191]
3 [0.453, -0.043]
```

这个玩具例子把 velocity 写成“离目标还有多远”，负 `dt` 让 action 一步步靠近目标。

The toy velocity is “how far we are from the target,” and the negative `dt` moves the action toward that target step by step.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **openpi pi0 suffix** / **openpi pi0 suffix**: 同样把 noisy action 和 timestep 放进动作后缀。 / It also places noisy actions and timesteps into the action suffix.
- **LeRobot DiffusionPolicy** / **LeRobot DiffusionPolicy**: 用 diffusion scheduler 迭代动作 chunk，接口目标相同。 / It iteratively denoises action chunks with a diffusion scheduler for the same interface goal.

## 注意事项 / Caveats / when it breaks

- **步数影响延迟** / **Step count affects latency**: `num_steps` 越大动作越平滑，但推理越慢。 / More `num_steps` can improve smoothness but increases inference latency.
- **mask 形状必须对齐** / **Mask shapes must align**: prefix/suffix attention mask 拼错会让动作 token 看见不该看的位置。 / Incorrect prefix/suffix masks let action tokens see invalid positions.

## 延伸阅读 / Further reading

- [LeRobot pi0 modeling source](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi0/modeling_pi0.py#L587-L678)

