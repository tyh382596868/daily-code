---
date: 2026-07-17
topic: huggingface
source: huggingface
repo: huggingface/diffusers
file: src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
permalink: https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, huggingface, diffusers, scheduler, flow-matching]
---

# Diffusers FlowMatch：每个 token 可以有自己的 timestep / Diffusers FlowMatch: Each Token Can Carry Its Own Timestep

> **一句话 / In one line**: `FlowMatchEulerDiscreteScheduler.step()` 支持 `per_token_timesteps`，让同一个 sample 里的不同 token 沿不同噪声进度前进。 / `FlowMatchEulerDiscreteScheduler.step()` supports `per_token_timesteps`, letting different tokens in one sample advance from different noise levels.

## 为什么重要 / Why this matters

普通扩散采样默认整张图共享一个 timestep。视频、局部编辑、分块生成和长序列生成里，token 可能不在同一个进度上：有的区域刚开始去噪，有的区域接近完成。per-token timestep 把 scheduler 从“全局时钟”变成“每个 token 一个时钟”。

Standard diffusion sampling assumes one timestep for the whole image. In video, local editing, tiled generation, or long sequences, tokens may sit at different progress levels. Per-token timesteps turn the scheduler from a global clock into a per-token clock.

## 代码 / The code

`huggingface/diffusers` — [`scheduling_flow_match_euler_discrete.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py)

```python
if per_token_timesteps is not None:
    per_token_sigmas = per_token_timesteps / self.config.num_train_timesteps
    sigmas = self.sigmas[:, None, None]

    lower_mask = sigmas < per_token_sigmas[None] - 1e-6
    lower_sigmas = lower_mask * sigmas
    lower_sigmas, _ = lower_sigmas.max(dim=0)

    current_sigma = per_token_sigmas[..., None]
    next_sigma = lower_sigmas[..., None]
    dt = current_sigma - next_sigma
else:
    sigma = self.sigmas[self.step_index]
    sigma_next = self.sigmas[self.step_index + 1]
    dt = sigma_next - sigma

prev_sample = sample + dt * model_output
```

## 逐行讲解 / What's happening

1. **timestep 先归一成 sigma / Timesteps become sigmas**: 中文: flow matching 里 scheduler 真正推进的是 sigma。 English: flow matching advances sigma, not the raw timestep label.
2. **为每个 token 找下一个更低 sigma / Find the next lower sigma per token**: 中文: `lower_mask` 在全局 sigma 表里筛选低于当前 token 的候选。 English: `lower_mask` filters the global sigma table below each token's current value.
3. **取最大的低 sigma / Pick the nearest lower sigma**: 中文: `max(dim=0)` 得到下一步，不会一跳到底。 English: `max(dim=0)` chooses the nearest lower step, not the end.
4. **`dt` 变成张量 / `dt` becomes a tensor**: 中文: 更新步长可按 token 广播。 English: the step size broadcasts per token.
5. **Euler 公式不变 / Euler stays the same**: 中文: 只有 `dt` 从标量变成 per-token 张量。 English: only `dt` changes from scalar to per-token tensor.

## 类比 / The analogy

像一排工位同时做不同进度的零件。主管不是喊“全体下一步”，而是给每个工位看自己的工单进度，再推进到各自的下一格。

It is like an assembly line where every station works on a part at a different stage. The controller does not shout one global step; it advances each station to its own next slot.

## 自己跑一遍 / Try it yourself

```python
sigmas = [1.0, 0.7, 0.4, 0.0]
per_token = [0.9, 0.5]
model_output = [2.0, 2.0]
sample = [10.0, 10.0]

next_sigmas = []
for s in per_token:
    lower = [x for x in sigmas if x < s - 1e-6]
    next_sigmas.append(max(lower))

prev = [x + (n - s) * v for x, s, n, v in zip(sample, per_token, next_sigmas, model_output)]
print(next_sigmas)
print(prev)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
[0.7, 0.4]
[9.6, 9.8]
```

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **局部重绘 / Local inpainting**: mask 区域和保留区域可以处于不同噪声进度。 / Edited and preserved regions can have different noise levels.
- **视频长上下文 / Long video context**: 新帧、历史帧、参考帧可分配不同时间。 / New, history, and reference frames can use different clocks.

## 注意事项 / Caveats / when it breaks

- **形状必须能广播 / Shapes must broadcast**: `per_token_timesteps` 要和 sample token 维匹配。 / `per_token_timesteps` must align with the sample token dimension.
- **精度边界用 `1e-6` / Boundary uses `1e-6`**: 防止等值 sigma 被误选为下一步。 / Prevents selecting the current sigma as the next one.
- **dtype 回转只在全局路径做 / dtype cast-back only happens on global path**: per-token 路径保留计算 dtype，调用方要注意。 / The per-token path keeps compute dtype, so callers must watch dtype.

## 延伸阅读 / Further reading

- [Diffusers FlowMatchEulerDiscreteScheduler](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py)
- [Diffusers repository](https://github.com/huggingface/diffusers)
