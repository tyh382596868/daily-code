---
date: 2026-07-13
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/policies/act/modeling_act.py
permalink: https://github.com/huggingface/lerobot/blob/e40b58a8dfa9e7b86918c374791599d070518d11/src/lerobot/policies/act/modeling_act.py#L170-L270
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, action-chunking]
build_role: action-chunking temporal ensemble for a from-scratch nanoVLA
---

# LeRobot ACT temporal ensemble：重叠动作 chunk 在线加权平均 / LeRobot ACT Temporal Ensemble: Online Averaging for Overlapping Action Chunks

> **一句话 / In one line**: ACT 不只预测一段 action，还把多次重叠预测按指数权重在线融合，再每步吐出一个动作。 / ACT predicts action chunks and fuses overlapping predictions online with exponential weights before emitting one action per step.

## 为什么重要 / Why this matters

动作 chunk 能减少推理频率，但相邻 chunk 对同一个未来时刻会给出不同预测。如果直接拼接，控制会抖；如果缓存所有历史再平均，显存和代码都膨胀。LeRobot 的 `ACTTemporalEnsembler` 用在线平均解决这件事。

Action chunks reduce inference frequency, but neighboring chunks can disagree about the same future timestep. Direct concatenation jitters; storing all history is heavy. LeRobot's `ACTTemporalEnsembler` solves it with an online weighted average.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/act/modeling_act.py`](https://github.com/huggingface/lerobot/blob/e40b58a8dfa9e7b86918c374791599d070518d11/src/lerobot/policies/act/modeling_act.py#L170-L270)

```python
class ACTTemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        self.ensembled_actions = None
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            self.ensembled_actions = actions.clone()
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action
```

## 逐行讲解 / What's happening

1. **指数权重 / Exponential weights**:
   - 中文: `exp(-m * i)` 给 chunk 内不同位置不同权重，`m` 控制新旧预测的偏好。
   - English: `exp(-m * i)` assigns different weights across the chunk; `m` controls preference for older or newer predictions.
2. **第一次直接缓存 / Cache the first chunk directly**:
   - 中文: 没有历史时，第一段动作就是当前 ensemble。
   - English: With no history, the first action chunk becomes the current ensemble.
3. **在线平均 / Online averaging**:
   - 中文: 先把旧平均乘回累计权重和，再加新预测，最后除以新的累计权重和。
   - English: The old average is scaled back by its accumulated weight sum, the new prediction is added, then the result is normalized by the new sum.
4. **消费第一个动作 / Consume the first action**:
   - 中文: 每次返回最前面的动作，并把缓存整体左移一格。
   - English: Each update returns the first action and shifts the remaining ensemble left by one step.

## 类比 / The analogy

像天气预报每天都预测未来 7 天。明天的天气可能被今天、昨天、前天的预报都覆盖到；最终播报时，不是只信最新一版，而是把几版预报按可信度加权。

It is like a weather forecast that predicts seven days ahead every morning. Tomorrow may appear in several forecasts; the final report blends those forecasts instead of trusting only the newest one.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

这是 `action-chunking` 的高级变体，依赖视觉/语言到动作头的完整路径。nanoVLA 可以先实现“预测 chunk，只执行第一个动作”的简单队列；生产级控制再加入 temporal ensemble，降低 chunk 边界处的抖动。

This is an advanced `action-chunking` component after the vision-language-to-action path exists. A nanoVLA can start with “predict a chunk, execute the first action”; a production controller adds temporal ensembling to reduce jitter at chunk boundaries.

## 自己跑一遍 / Try it yourself

```python
weights = [1.0, 0.8, 0.64]
avg = [10, 20, 30]
count = [1, 1, 1]
new = [12, 21, 33]

avg = [(a * weights[c-1] + n * weights[c]) / (weights[c-1] + weights[c])
       for a, n, c in zip(avg[:-1], new[:-1], count[:-1])]
avg.append(new[-1])
print(avg)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
[10.88888888888889, 20.444444444444446, 33]
```

同一个未来动作位置被新旧 chunk 共同解释，平均后比硬切换更平滑。

The same future action slot is explained by old and new chunks together, so the blended output is smoother than a hard switch.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Diffusion Policy action queue** / **Diffusion Policy action queue**: 也用 chunk 降低每步采样成本，但通常先做队列版本。 / It also uses chunks to reduce per-step sampling cost, often starting with a simple queue.
- **Model predictive control** / **Model predictive control**: 重叠 horizon 的计划常需要平滑或 warm-start。 / Plans over overlapping horizons often need smoothing or warm starts.

## 注意事项 / Caveats / when it breaks

- **权重方向要验证** / **validate weight direction**: 正负 `temporal_ensemble_coeff` 会改变偏好旧预测还是新预测。 / Positive and negative coefficients change whether older or newer predictions dominate.
- **reset 很关键** / **reset matters**: 环境重置时不清空缓存会把上一条 episode 的动作带进来。 / Failing to clear state on environment reset leaks actions from the previous episode.

## 延伸阅读 / Further reading

- LeRobot ACT source — https://github.com/huggingface/lerobot/blob/e40b58a8dfa9e7b86918c374791599d070518d11/src/lerobot/policies/act/modeling_act.py
