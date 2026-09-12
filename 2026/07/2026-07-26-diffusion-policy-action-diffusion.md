---
date: 2026-07-26
topic: diffusion
source: trending
repo: lucidrains/diffusion-policy
authority_note: trending candidate observed via GitHub/web search during the 2026-07-26 run
file: diffusion_policy/diffusion_policy.py
permalink: https://github.com/lucidrains/diffusion-policy/blob/main/diffusion_policy/diffusion_policy.py
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, diffusion-policy, actions]
---

# Diffusion Policy：动作序列也可以当成去噪对象 / Diffusion Policy: Treat the Action Sequence as the Denoising Target

> **一句话 / In one line**: diffusion policy 把未来一段动作当作 noisy trajectory，模型逐步预测去噪后的动作。 / A diffusion policy treats a future action chunk as a noisy trajectory and denoises it step by step.

## 为什么重要 / Why this matters

传统策略常一次输出一个动作，遇到多模态任务会平均成“中间动作”。扩散策略直接生成一段动作轨迹，允许模型在多个可行动作模式里采样，而不是硬压成一个均值。

Classic policies often emit one action and average multimodal choices into a bland middle action. Diffusion policies generate an action trajectory, allowing the model to sample one coherent mode instead of collapsing to the mean.

## 代码 / The code

`lucidrains/diffusion-policy` — [`diffusion_policy/diffusion_policy.py`](https://github.com/lucidrains/diffusion-policy/blob/main/diffusion_policy/diffusion_policy.py)

```python
class DiffusionPolicy(nn.Module):
    def __init__(self, model, scheduler, horizon):
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.horizon = horizon

    def forward(self, obs, actions):
        noise = torch.randn_like(actions)
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (actions.shape[0],), device=actions.device)
        noisy_actions = self.scheduler.add_noise(actions, noise, timesteps)
        pred = self.model(noisy_actions, timesteps, obs)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, obs, shape):
        actions = torch.randn(shape, device=obs.device)
        for t in self.scheduler.timesteps:
            pred = self.model(actions, t, obs)
            actions = self.scheduler.step(pred, t, actions).prev_sample
        return actions[:, : self.horizon]
```

## 逐行讲解 / What's happening

1. **训练时加噪 / Add noise during training**
   - 中文: ground-truth action chunk 被随机 timestep 污染，模型学习把噪声预测出来。
   - English: The ground-truth action chunk is corrupted at a random timestep, and the model learns to predict the noise.
2. **`model(noisy_actions, timesteps, obs)`**
   - 中文: 观测不是输出目标，而是条件；被去噪的是未来动作序列。
   - English: The observation is conditioning, not the target; the future action sequence is what gets denoised.
3. **采样从随机动作开始 / Sampling starts from random actions**
   - 中文: 推理时先造一段随机动作，再沿 scheduler 的时间表逐步修正。
   - English: Inference starts with random actions and iteratively refines them along the scheduler timeline.
4. **返回 horizon / Return horizon**
   - 中文: 只交给控制器需要的动作窗口，后续可接 action chunk broker。
   - English: Only the needed action window is returned, ready for an action chunk broker.

## 类比 / The analogy

这像先随手画一条很乱的路线，再根据地图和目的地一遍遍擦改，最后得到一段能走的路线。

It is like sketching a messy route first, then repeatedly erasing and correcting it with a map and destination until a usable path appears.

## 自己跑一遍 / Try it yourself

```python
import random

truth = [1.0, 1.2, 1.4]
noisy = [x + random.choice([-0.5, 0.5]) for x in truth]
for _ in range(3):
    noisy = [round(x * 0.5 + y * 0.5, 2) for x, y in zip(noisy, truth)]
print(noisy)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```
# three numbers moving closer to [1.0, 1.2, 1.4]
```

中文: 这个玩具例子不是扩散公式，只展示“从噪声动作逐步靠近目标动作”的形状。
English: This toy is not the diffusion equation; it only shows the shape of iteratively moving noisy actions toward a target.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **LeRobot DiffusionPolicy** / **LeRobot DiffusionPolicy**: 中文: 同样一次生成 action chunk，再逐步消费。 / English: It similarly generates an action chunk and consumes it over control ticks.
- **openpi flow head** / **openpi flow head**: 中文: 把动作去噪改写成 rectified-flow 速度预测。 / English: It rewrites action denoising as rectified-flow velocity prediction.

## 注意事项 / Caveats / when it breaks

- **实时性** / **Latency**: 中文: 多步采样比直接回归慢，需要 chunking 或蒸馏。 / English: Multi-step sampling is slower than direct regression and needs chunking or distillation.
- **动作归一化** / **Action normalization**: 中文: 没有稳定 scale，噪声训练会很难。 / English: Without stable action scales, noise training becomes brittle.

## 延伸阅读 / Further reading

- [lucidrains/diffusion-policy](https://github.com/lucidrains/diffusion-policy)
- [Diffusion Policy paper](https://diffusion-policy.cs.columbia.edu/)
