---
date: 2026-08-18
topic: diffusion
source: trending
repo: brysonjones/multitask_dit_policy
file: src/multitask_dit_policy/model/objectives.py
permalink: https://github.com/brysonjones/multitask_dit_policy/blob/2d4ad814af602b83218ed019eb193ef66fdc4a5d/src/multitask_dit_policy/model/objectives.py#L161-L216
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, robotics]
---

# Multitask DiT flow matching：动作从噪声直线走向数据 / Multitask DiT Flow Matching: Move Actions from Noise to Data Along a Line

> **一句话 / In one line**: `FlowMatchingObjective` 在 action 序列和噪声之间采样线性中间点，让模型预测把噪声推向真实动作的速度场。 / `FlowMatchingObjective` samples linear intermediate points between action sequences and noise, then trains the model to predict the velocity field from noise toward data.

## 为什么重要 / Why this matters

机器人 action 也可以当生成目标。相比逐步 DDPM 噪声预测，flow matching 把训练目标写成一条清晰的路径：任意时间 `t` 上的点 `x_t` 应该沿固定速度走向真实 action。

Robot actions can be a generation target too. Compared with stepwise DDPM noise prediction, flow matching describes training as a clear path: at any time `t`, point `x_t` should move toward the real action with a fixed target velocity.

## 代码 / The code

`brysonjones/multitask_dit_policy` — [`src/multitask_dit_policy/model/objectives.py`](https://github.com/brysonjones/multitask_dit_policy/blob/2d4ad814af602b83218ed019eb193ef66fdc4a5d/src/multitask_dit_policy/model/objectives.py#L172-L216)

```python
def _sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
    if self.config.timestep_sampling.strategy_name == "uniform":
        return torch.rand(batch_size, device=device)
    elif self.config.timestep_sampling.strategy_name == "beta":
        beta_dist = torch.distributions.Beta(
            self.config.timestep_sampling.alpha, self.config.timestep_sampling.beta
        )
        u = beta_dist.sample((batch_size,)).to(device)
        return self.config.timestep_sampling.s * (1.0 - u)

def compute_loss(self, model: nn.Module, batch: dict[str, Tensor], conditioning_vec: Tensor) -> Tensor:
    data = batch["action"]
    noise = torch.randn_like(data)
    t = self._sample_timesteps(batch_size, device)
    t_expanded = t.view(-1, 1, 1)
    x_t = t_expanded * data + (1 - (1 - self.config.sigma_min) * t_expanded) * noise
    target_velocity = data - (1 - self.config.sigma_min) * noise
```

## 逐行讲解 / What's happening

1. **第 172-189 行 / Lines 172-189**: 中文: timestep 可以均匀采样，也可以从 Beta 分布变换而来，让训练更关注某些噪声区间。 / English: Timesteps can be uniform or transformed from a Beta distribution, letting training emphasize selected noise regions.
2. **第 196-203 行 / Lines 196-203**: 中文: `x_t` 是 action data 和 noise 的线性混合；`sigma_min` 保留一点最小噪声尺度。 / English: `x_t` is a linear mix of action data and noise; `sigma_min` keeps a small minimum noise scale.
3. **第 205-208 行 / Lines 205-208**: 中文: target velocity 不依赖 `t`，表示从当前 path 朝 data 方向走的速度。 / English: The target velocity is independent of `t`, describing the direction from the path toward data.
4. **第 210-215 行 / Lines 210-215**: 中文: 如果 batch 有 padding 标记，loss 会把无效 action step 屏蔽掉。 / English: If the batch carries padding flags, the loss masks out invalid action steps.

## 类比 / The analogy

像在一条从雾气到目标轨迹的直线上随机挑点训练导航员：无论从哪个点出发，它都要指出通向真实轨迹的方向。

It is like training a navigator by sampling random points on a line from fog to the target trajectory: from any point, it must point toward the real trajectory.

## 自己跑一遍 / Try it yourself

```python
def flow_point(data, noise, t, sigma_min):
    return t * data + (1 - (1 - sigma_min) * t) * noise

def target_velocity(data, noise, sigma_min):
    return data - (1 - sigma_min) * noise

data = 10.0
noise = -2.0
sigma_min = 0.1
for t in [0.0, 0.5, 1.0]:
    print(round(flow_point(data, noise, t, sigma_min), 2))
print(round(target_velocity(data, noise, sigma_min), 2))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
-2.0
3.9
9.8
11.8
```

`x_t` 从噪声端逐渐靠近数据端；velocity 是模型要学的方向。

`x_t` moves from the noise end toward the data end; the velocity is what the model learns.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **flow-matching VLA heads** / **flow-matching VLA heads**: pi0/SmolVLA 类模型也把连续动作作为速度场目标。 / pi0/SmolVLA-style models also train continuous actions as velocity-field targets.
- **rectified flow video models** / **rectified-flow video models**: 视频 latent 生成也常用同样的线性 path 和 velocity target。 / Video-latent generation often uses the same linear path and velocity target.

## 注意事项 / Caveats / when it breaks

- **时间采样会改变训练重心** / **Timestep sampling changes training focus**: Beta 策略不只是优化技巧，它会改变模型在哪些噪声区间更熟。 / A Beta strategy is not just an optimization detail; it changes which noise regions the model learns best.
- **padding mask 不能漏** / **Padding masks must not be omitted**: 多任务 action horizon 不一致时，无效 step 参与 loss 会污染目标。 / With varied action horizons, invalid steps in the loss pollute the target.

## 延伸阅读 / Further reading

- [multitask_dit_policy source](https://github.com/brysonjones/multitask_dit_policy/blob/2d4ad814af602b83218ed019eb193ef66fdc4a5d/src/multitask_dit_policy/model/objectives.py#L161-L216)
