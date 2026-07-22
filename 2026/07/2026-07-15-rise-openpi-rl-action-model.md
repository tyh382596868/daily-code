---
date: 2026-07-15
topic: infrastructure
source: trending
repo: OpenDriveLab/RISE
file: policy_and_value/policy_online/rlinf/models/embodiment/openpi_action_model.py
permalink: https://github.com/OpenDriveLab/RISE/blob/5fac1e6ab9d50d4cc1ba4daeadf14023c8415955/policy_and_value/policy_online/rlinf/models/embodiment/openpi_action_model.py#L38-L193
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, infrastructure, rl, vla, flow-matching]
---

# RISE OpenPI RL：给 pi0 接上 value、dynamics、reward 和探索噪声 / RISE OpenPI RL: Add Value, Dynamics, Reward, and Exploration Noise to pi0

> **一句话 / In one line**: RISE 把 openpi 的 pi0 action predictor 包成 RL 模型，在同一类里接入 value head、dynamics model、reward model 和 flow-noise 探索头。 / RISE wraps openpi's pi0 action predictor as an RL model, attaching a value head, dynamics model, reward model, and flow-noise exploration head in one class.

## 为什么重要 / Why this matters

VLA 如果只做 imitation learning，模型学的是数据里出现过的动作；online RL 需要估值、奖励、环境动态或探索噪声来改进策略。这个文件的核心不是某个 fancy layer，而是“如何把一个预训练 action predictor 改造成 RL actor”：保留 pi0 主干，再按配置插上 critic/value、world model、reward model 和噪声头。

A VLA trained only with imitation learning copies actions seen in data; online RL needs value estimates, rewards, dynamics, or exploration noise to improve the policy. The core idea here is not a fancy layer but a wiring pattern: keep the pi0 backbone and attach critic/value, world model, reward model, and noise modules by configuration.

## 代码 / The code

`OpenDriveLab/RISE` — [`policy_and_value/policy_online/rlinf/models/embodiment/openpi_action_model.py`](https://github.com/OpenDriveLab/RISE/blob/5fac1e6ab9d50d4cc1ba4daeadf14023c8415955/policy_and_value/policy_online/rlinf/models/embodiment/openpi_action_model.py#L38-L193)

```python
def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))

def sample_time(bsize, inx_len, device):
    """
        Modified sample_time function
    """
    # Generate Beta distribution samples of shape (bsize*inx_len)
    time_beta = sample_beta(1.5, 1.0, bsize * inx_len, device)
    
    # Adjust the shape to (bsize, inx_len)
    time_beta = time_beta.view(bsize, inx_len)
    
    time = time_beta * 0.999 + 0.001
    
    # Sort the inx_len values ​​of each bsize dimension in descending order
    time, _ = torch.sort(time, dim=1, descending=True)
    
    return time.to(dtype=torch.float32, device=device)

class OpenPi0ForRLActionPrediction(PI0Pytorch):
    """
    Pi0 model for reinforcement learning action prediction.
    """

    config: OpenPi0Config

    def __init__(
        self,
        config: OpenPi0Config,
    ):
        # Override `sample_actions` to prevent parent class polymorphic call
        sample_actions_func = self.sample_actions
        super().__init__(config)
        self.sample_actions = sample_actions_func
        self.global_step = 0
        # assert
        assert not (self.config.double_layer and self.config.joint_logprob), (
            "double_layer and joint_logprob can not be set at the same time"
        )

        # rl model init
        if self.config.value_after_vlm:
            proj_width = 2048
        else:
            proj_width = 1024
        # value head
        if self.config.add_value_head:
            self.value_head = ValueHead(
                input_dim=proj_width,
                hidden_sizes=(512, 256, 128),
                output_dim=1,
                activation="relu",
                bias_last=True,
            )
        self.use_vlm_value = getattr(self.config, "value_after_vlm", False) and getattr(
            self.config, "add_value_head", False
        )
        
        use_torch_compile = getattr(self.config, "use_torch_compile", True)

        self.reward_mode = getattr(self.config, "reward_mode", "v1")
        
        if self.config.add_dynamics_model:
            self.dynamics_model = DynamicsModel(self.config.dynamics_model_config)
            if use_torch_compile:
                self.dynamics_model.pipe.transformer = torch.compile(self.dynamics_model.pipe.transformer)

        if self.config.add_reward_model:
            self.reward_model = RewardModel(self.config.reward_model_config, self.config.reward_model_ckpt)
            if use_torch_compile:
                self.reward_model.model.sample_values = torch.compile(self.reward_model.model.sample_values, mode="reduce-overhead")

        # noise head for flow-noise
        if self.config.noise_method == "flow_noise":
            self.noise_head = ExploreNoiseNet(
                in_dim=1024,
                out_dim=self.config.action_dim,
                hidden_dims=[128, 64],
                activation_type="tanh",
                noise_logvar_range=self.config.noise_logvar_range,
                noise_scheduler_type="learn",
            )
```

## 逐行讲解 / What's happening

1. **`sample_time`**:
   - 中文: 用 Beta(1.5, 1.0) 偏向较大时间，再排序成降序，像一次性准备多个去噪时间点。
   - English: It samples from Beta(1.5, 1.0), biased toward larger times, then sorts descending to prepare multiple denoising times.
2. **`sample_actions_func` 保存再恢复**:
   - 中文: 父类初始化可能覆盖或调用多态方法，先抓住当前实现，`super()` 后再恢复。
   - English: The parent initializer may override or call polymorphic methods, so the class saves the current implementation and restores it after `super()`.
3. **value head width**:
   - 中文: 如果 value 接在 VLM 后面，特征宽度是 2048；否则用 action expert 路径的 1024。
   - English: If value is read after the VLM, feature width is 2048; otherwise it uses the 1024-wide action-expert path.
4. **可选 dynamics/reward**:
   - 中文: world model 和 reward model 都按配置接入，并可单独 `torch.compile` 热路径。
   - English: Dynamics and reward models are attached by config and can compile their hot paths independently.
5. **flow-noise head**:
   - 中文: 当探索噪声不是固定方差，而是可学习模块时，策略可以按状态输出动作噪声尺度。
   - English: With a learned flow-noise module, exploration variance can depend on the policy state instead of being fixed.

## 类比 / The analogy

这像把一辆手动挡车改成赛车：发动机还在，但你加了转速表、赛道模拟器、计分器和可调避震。车能不能跑本来就会；这些附加模块让它能在赛道上持续改进。

It is like turning a manual car into a race car. The engine remains, but you add a tachometer, track simulator, scoring system, and adjustable suspension. The car could already drive; the new modules let it improve on the track.

## 自己跑一遍 / Try it yourself

```python
import random

class Policy:
    def __init__(self, add_value=False, add_reward=False, noise="fixed"):
        self.parts = ["pi0_backbone"]
        if add_value:
            self.parts.append("value_head")
        if add_reward:
            self.parts.append("reward_model")
        if noise == "flow_noise":
            self.parts.append("learned_noise_head")

def sample_time(batch, n):
    rows = []
    for _ in range(batch):
        xs = [0.001 + 0.999 * random.betavariate(1.5, 1.0) for _ in range(n)]
        rows.append(sorted(xs, reverse=True))
    return rows

p = Policy(add_value=True, add_reward=True, noise="flow_noise")
print(p.parts)
print([[round(x, 3) for x in row] for row in sample_time(2, 3)])
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
['pi0_backbone', 'value_head', 'reward_model', 'learned_noise_head']
[[... three descending values ...], [... three descending values ...]]
```

中文: 重点是“按配置接模块”和“时间采样排序”两个结构，不依赖真实 openpi 环境也能看懂。

English: The important structures are configurable module attachment and sorted time sampling; they are visible without a full openpi setup.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **RLHF policy + value head** / **RLHF policy + value head**: 语言模型也常在 backbone 上接一个 value head 做 PPO。
- **Dreamer / TD-MPC** / **Dreamer / TD-MPC**: actor 外围接 dynamics model 和 reward/value 估计，用想象 rollout 更新策略。
- **Diffusion policy exploration** / **Diffusion policy exploration**: 学习噪声或时间分布可以让探索集中在更有效的 action 区域。

## 注意事项 / Caveats / when it breaks

- **模块多不等于训练稳 / More modules do not guarantee stability**: value、reward、dynamics 任一漂移都会影响 policy update。
- **`torch.compile` 要分模块评估 / Compile module by module**: reward 或 dynamics 热路径适合 compile，但 debug 时要能单独关闭。
- **时间分布是算法选择 / Time distribution is an algorithm choice**: Beta 参数改变会改变去噪训练/采样关注的噪声区间。

## 延伸阅读 / Further reading

- [RISE `openpi_action_model.py`](https://github.com/OpenDriveLab/RISE/blob/5fac1e6ab9d50d4cc1ba4daeadf14023c8415955/policy_and_value/policy_online/rlinf/models/embodiment/openpi_action_model.py)
- [OpenDriveLab/RISE](https://github.com/OpenDriveLab/RISE)
