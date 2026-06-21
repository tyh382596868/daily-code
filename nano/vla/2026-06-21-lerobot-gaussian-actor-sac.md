---
date: 2026-06-21
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/policies/gaussian_actor/modeling_gaussian_actor.py
permalink: https://github.com/huggingface/lerobot/blob/2d7a42011a4f8e05a8c85d5fb908da258d4cc7b1/src/lerobot/policies/gaussian_actor/modeling_gaussian_actor.py#L401-L476
difficulty: intermediate
read_time: ~12 min
tags: [code-of-the-day, vla, action-head, gaussian-actor, sac, reparameterization, tanh-squash]
build_role: action-head-continuous (SAC Gaussian actor variant — reparameterize-sample + Tanh squash for RL fine-tuning)
---

# SAC 高斯动作头：预测 (μ, σ)、重参数化采样、Tanh 压缩到 [-1,1] / SAC Gaussian Action Head: Predict (μ, σ), Reparameterize-Sample, Tanh-Squash to [-1,1]

> **一句话 / In one line**: 这是 VLA 动作头的第三种范式——连续高斯分布 + 重参数化技巧 + Tanh 压缩，一次 forward 返回 (action, log_prob, mean)，这正是 SAC 离线策略 RL 所需要的全部三个值。 / This is the third action-head paradigm in the VLA curriculum — continuous Gaussian with reparameterization trick + Tanh squash, returning (action, log_prob, mean) in one forward pass, exactly the three values SAC off-policy RL requires.

## 为什么重要 / Why this matters

在 VLA 课程中，我们已经见过两种动作头范式：
- **OpenVLA**：离散化动作，把连续关节角度量化为 token，用交叉熵分类。
- **GR00T / pi0**：流匹配动作头，从噪声迭代去噪到确定性动作。

今天的第三种是 **SAC 高斯动作头**：策略网络输出一个多维高斯分布的均值 μ 和标准差 σ，通过重参数化技巧（reparameterization trick）从中采样，再通过 Tanh 把动作压缩到 [-1, 1]。一次 forward 需要返回三个值：动作 a、对数概率 log π(a|s)、以及均值 μ——SAC 的 critic 更新和策略熵最大化都需要这三个值。

这个设计让策略天然支持**离线策略 RL 微调**：因为我们能精确计算任意动作的对数概率（不像流匹配需要 SDE 积分），SAC 的 soft policy improvement 步骤可以直接在这个分布上做。

In the VLA curriculum, we've seen two action-head paradigms:
- **OpenVLA**: discretize actions — quantize continuous joint angles into tokens, train with cross-entropy.
- **GR00T / pi0**: flow-matching action head — iteratively denoise from noise to deterministic action.

Today's third paradigm is the **SAC Gaussian action head**: the policy network outputs the mean μ and standard deviation σ of a multivariate Gaussian, samples from it using the reparameterization trick, then squashes the action through Tanh to [-1, 1]. A single forward must return three values: action a, log probability log π(a|s), and mean μ — all three are needed by SAC's critic update and entropy maximization.

This design makes the policy naturally amenable to **off-policy RL fine-tuning**: because we can compute the exact log probability of any action (unlike flow matching which requires SDE integration), SAC's soft policy improvement step can be applied directly to this distribution.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/gaussian_actor/modeling_gaussian_actor.py`](https://github.com/huggingface/lerobot/blob/2d7a42011a4f8e05a8c85d5fb908da258d4cc7b1/src/lerobot/policies/gaussian_actor/modeling_gaussian_actor.py#L401-L476)

```python
class Policy(nn.Module):
    def __init__(
        self,
        encoder: GaussianActorObservationEncoder,
        network: nn.Module,
        action_dim: int,
        std_min: float = -5,
        std_max: float = 2,
        fixed_std: torch.Tensor | None = None,
        init_final: float | None = None,
        use_tanh_squash: bool = False,
        encoder_is_shared: bool = False,
    ):
        super().__init__()
        self.encoder: GaussianActorObservationEncoder = encoder
        self.network = network
        self.action_dim = action_dim
        self.std_min = std_min
        self.std_max = std_max
        self.fixed_std = fixed_std
        self.use_tanh_squash = use_tanh_squash
        self.encoder_is_shared = encoder_is_shared

        # Find the last Linear layer's output dimension
        for layer in reversed(network.net):
            if isinstance(layer, nn.Linear):
                out_features = layer.out_features
                break
        # Mean layer
        self.mean_layer = nn.Linear(out_features, action_dim)
        if init_final is not None:
            nn.init.uniform_(self.mean_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.mean_layer.bias, -init_final, init_final)
        else:
            orthogonal_init()(self.mean_layer.weight)

        # Standard deviation layer or parameter
        if fixed_std is None:
            self.std_layer = nn.Linear(out_features, action_dim)
            if init_final is not None:
                nn.init.uniform_(self.std_layer.weight, -init_final, init_final)
                nn.init.uniform_(self.std_layer.bias, -init_final, init_final)
            else:
                orthogonal_init()(self.std_layer.weight)

    def forward(
        self,
        observations: torch.Tensor,
        observation_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # We detach the encoder if it is shared to avoid backprop through it
        obs_enc = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)

        # Get network outputs
        outputs = self.network(obs_enc)
        means = self.mean_layer(outputs)

        # Compute standard deviations
        if self.fixed_std is None:
            log_std = self.std_layer(outputs)
            std = torch.exp(log_std)          # Match JAX "exp"
            std = torch.clamp(std, self.std_min, self.std_max)
        else:
            std = self.fixed_std.expand_as(means)

        # Build transformed distribution
        dist = TanhMultivariateNormalDiag(loc=means, scale_diag=std)

        # Sample actions (reparameterized)
        actions = dist.rsample()

        # Compute log_probs
        log_probs = dist.log_prob(actions)

        return actions, log_probs, means
```

## 逐行讲解 / What's happening

1. **`mean_layer` 和 `std_layer` — 双头输出**:
   - 中文: 网络最后一层的特征被分别送入两个独立的线性层，一个预测均值 μ（`action_dim` 维），一个预测对数标准差 log σ。两个头共享同一条编码器-网络特征流，只在最后分叉。
   - English: The final network features are fed into two separate linear heads — one predicts mean μ (`action_dim`-dim), one predicts log standard deviation log σ. Both heads share the same encoder-network feature stream, diverging only at the final step.

2. **`orthogonal_init()` — 正交初始化**:
   - 中文: SAC 策略对权重初始化敏感。正交初始化（Gram-Schmidt 正交化随机高斯矩阵）能保证初始特征空间中不同方向近似正交，有助于训练早期的梯度传播稳定性。
   - English: SAC policies are sensitive to weight initialization. Orthogonal initialization (Gram-Schmidt orthogonalization of a random Gaussian matrix) ensures approximately orthogonal directions in the initial feature space, helping stable gradient flow early in training.

3. **`std = torch.exp(log_std)` + `torch.clamp(std, self.std_min, self.std_max)`**:
   - 中文: 网络输出 log σ（无界实数），取指数确保 σ > 0。再 clamp 到 [exp(-5), exp(2)] ≈ [0.007, 7.4]，防止标准差过大（随机爆炸）或过小（分布退化成 delta 函数，梯度消失）。
   - English: The network outputs log σ (unbounded real), exp ensures σ > 0. Clamping to [exp(-5), exp(2)] ≈ [0.007, 7.4] prevents std from getting too large (random explosions) or too small (distribution degenerating to a delta, killing gradients).

4. **`TanhMultivariateNormalDiag` — Tanh 压缩的多元正态**:
   - 中文: 这不是一个普通的 Normal 分布——它在正态分布外面套了一层 Tanh 变换，把动作范围从 (-∞, +∞) 压缩到 (-1, 1)。log_prob 的计算需要 Jacobian 修正（减去 Tanh 变换的对数行列式），否则密度估计错误。
   - English: This is not a plain Normal — it wraps a Normal distribution with a Tanh transform, squashing actions from (-∞, +∞) to (-1, 1). The log_prob calculation requires a Jacobian correction (subtract the log-determinant of the Tanh transform), otherwise density estimates are wrong.

5. **`actions = dist.rsample()` — 重参数化采样**:
   - 中文: `.rsample()` 实现重参数化技巧：先从标准正态采样 ε，再令 a = tanh(μ + σ·ε)。梯度可以通过 ε 反传到 μ 和 σ，使策略梯度 ∇_θ E[Q(s,a)] 可以直接用自动微分计算，而不需要 REINFORCE 那样的高方差估计器。
   - English: `.rsample()` implements the reparameterization trick: sample ε from standard Normal, then a = tanh(μ + σ·ε). Gradients flow through ε back to μ and σ, allowing the policy gradient ∇_θ E[Q(s,a)] to be computed via autodiff rather than high-variance REINFORCE estimators.

6. **返回 `(actions, log_probs, means)` 三元组**:
   - 中文: SAC 的 actor 更新需要 `log_probs`（用于熵项 -α·log π(a|s)）；critic 更新需要 `actions`；评估时取确定性动作用 `means`（不带随机性的 tanh(μ)）。一次 forward 同时产出三个值避免了重复计算。
   - English: SAC's actor update needs `log_probs` (for the entropy term -α·log π(a|s)); the critic update needs `actions`; deterministic evaluation uses `means` (tanh(μ) without stochasticity). Returning all three from one forward avoids redundant computation.

## 类比 / The analogy

想象一个射击运动员（策略网络）在每次出手之前预测自己的瞄准中心（μ）和手部颤抖幅度（σ）。实际射击时，在预估范围内加入一点随机偏移（ε·σ），然后通过 Tanh 把射击角度限制在靶纸范围内（[-1, 1]）。教练（SAC 算法）事后根据子弹落点（action）和落在那个位置的概率（log_prob）来调整运动员的训练——他需要同时知道"你射到哪里"和"你射到那里的可能性有多大"，这正是返回 (action, log_prob, mean) 三个值的原因。

Imagine a marksman (the policy network) who, before each shot, estimates their aiming center (μ) and hand-tremor magnitude (σ). When actually shooting, they add a small random offset within that estimated range (ε·σ), then Tanh-squash the angle to stay within the target board ([-1, 1]). The coach (SAC algorithm) afterward adjusts training based on where the bullet landed (action) and how likely that position was (log_prob) — the coach needs both "where did you shoot" and "how probable was that shot," which is exactly why the forward returns (action, log_prob, mean).

## 在 nanoVLA 中的位置 / Where this lives in your nanoVLA

这个 SAC 高斯动作头是 nanoVLA 中 `action-head-continuous` 课程槽位的一个变体实现，与之前的流匹配头并列。

中文：在你自己的 nanoVLA 里，这个组件处于系统最末端——上游是视觉编码器 + 语言模型 backbone，下游是执行器。输入是观测编码向量（来自编码器的特征），输出是三元组 `(a, log π(a|s), μ)`。如果你想用 RL 对 VLA 做在线微调（例如在真实机器人上收集数据并最大化任务成功率），就用这个头替换流匹配头：SAC critic 的 TD 更新需要 `log_probs`；策略网络的 soft policy improvement 步骤需要 `actions` 梯度可传（靠 `rsample()`）；部署时用 `means` 给出确定性策略。如果省掉 Tanh 压缩，动作可能超出机器人关节范围，导致硬件保护触发。生产级实现还需要：automatic entropy tuning（自动调温度参数 α）、重放缓冲区、目标 critic 网络的软更新。

English: In your nanoVLA, this component sits at the very end of the pipeline — upstream is the vision encoder + language model backbone, downstream is the robot actuator. Inputs are observation encoding vectors (features from the encoder); outputs are the triple `(a, log π(a|s), μ)`. If you want to RL fine-tune your VLA online (e.g. collecting data on a real robot to maximize task success rate), swap the flow-matching head for this Gaussian head: SAC's TD update needs `log_probs`; the soft policy improvement step needs gradient-connected `actions` (via `rsample()`); deployment uses `means` for a deterministic policy. Omitting the Tanh squash means actions could exceed joint limits, triggering hardware safety stops. A production implementation additionally needs: automatic entropy tuning (learnable temperature α), a replay buffer, and soft target critic updates.

## 自己跑一遍 / Try it yourself

```python
import torch
import torch.nn as nn
import torch.distributions as dist

class TanhNormal:
    def __init__(self, mu, std):
        self.base = dist.Normal(mu, std)
    def rsample(self):
        x = self.base.rsample()
        return torch.tanh(x), x  # return (action, pre-tanh)
    def log_prob(self, action, pre_tanh):
        base_lp = self.base.log_prob(pre_tanh).sum(-1)
        correction = (2 * (torch.log(torch.tensor(2.0)) -
                           pre_tanh - nn.functional.softplus(-2 * pre_tanh))).sum(-1)
        return base_lp - correction

action_dim = 6
net = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
mean_head = nn.Linear(256, action_dim)
std_head  = nn.Linear(256, action_dim)

obs = torch.randn(4, 64)
feats = net(obs)
mu  = mean_head(feats)
log_std = std_head(feats).clamp(-5, 2)
std = log_std.exp()

tn = TanhNormal(mu, std)
action, pre_tanh = tn.rsample()
log_prob = tn.log_prob(action, pre_tanh)
print("action shape:  ", action.shape)   # (4, 6) in [-1, 1]
print("log_prob shape:", log_prob.shape) # (4,)
print("action range:  ", action.min().item(), "to", action.max().item())
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
action shape:   torch.Size([4, 6])
log_prob shape: torch.Size([4])
action range:   -0.99... to 0.99...
```

中文：注意动作值严格在 (-1, 1) 之间（Tanh 保证），而 log_prob 是标量（每个样本一个概率），这正是 SAC 的 actor loss 所需要的形式：`actor_loss = (α * log_prob - Q(s, a)).mean()`。

English: Note actions are strictly in (-1, 1) (Tanh guarantees this), and log_prob is a per-sample scalar — exactly the form SAC's actor loss needs: `actor_loss = (α * log_prob - Q(s, a)).mean()`.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **OpenVLA (`prismatic/vla/action_tokenizer.py`)** / **OpenVLA (离散 action head)**: 把连续动作分 bin 量化成 token，用交叉熵监督学习；无法直接计算 log_prob，不适合 RL 微调。 / Quantizes actions into bins as tokens, trains with cross-entropy supervised learning; can't compute closed-form log_prob, not suitable for RL fine-tuning.
- **GR00T / pi0 flow-matching head** / **GR00T/pi0 流匹配动作头**: 从高斯噪声迭代去噪到动作，表达能力更强（多峰分布），但 log_prob 需要 ODE 求解器积分，计算成本高；SAC 风格 RL 不适用。 / Iteratively denoises from Gaussian noise to action, more expressive (multimodal), but log_prob requires ODE solver integration, computationally expensive; SAC-style RL doesn't apply.
- **SpinningUp SAC** / **SpinningUp SAC**: OpenAI 的 SAC 参考实现，相同的 TanhNormal + reparameterize 结构，是本代码的原型之一。 / OpenAI's reference SAC implementation uses the same TanhNormal + reparameterize structure, one of the prototypes for this code.

## 注意事项 / Caveats / when it breaks

- **Tanh log-prob 修正不能省** / **Tanh log-prob correction is mandatory**: 如果直接用 Normal 的 log_prob 而不加 Tanh 的 Jacobian 修正（`-Σ log(1 - tanh²(x))`），密度估计偏低，导致 actor 高估熵，训练发散。 / Skipping the Tanh Jacobian correction (`-Σ log(1 - tanh²(x))`) causes density underestimation, making the actor overestimate entropy, diverging training.
- **std_min/std_max 的选择** / **std_min/std_max tuning**: `std_min=-5`（对应 exp(-5)≈0.007）过小会让策略几乎确定性，导致探索不足；`std_max=2`（对应 exp(2)≈7.4）过大会使动作接近均匀分布，批评家无法学习。 / `std_min=-5` too small makes the policy near-deterministic (no exploration); `std_max=2` too large makes actions near-uniform (critic can't learn).
- **encoder_is_shared 影响梯度流** / **encoder_is_shared affects gradient flow**: 在 SAC 里 actor 和 critic 共享编码器时，要 `detach` 编码器梯度（否则 actor 和 critic 的梯度互相干扰），这就是 `encoder_is_shared` 参数的作用。 / When actor and critic share an encoder in SAC, encoder gradients must be detached to prevent actor and critic from interfering with each other's updates — that's what `encoder_is_shared` does.

## 延伸阅读 / Further reading

- [Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor (Haarnoja et al., 2018)](https://arxiv.org/abs/1801.01290)
- [lerobot gaussian_actor policy source](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/gaussian_actor/modeling_gaussian_actor.py)
- [Reparameterization trick explained (Kingma & Welling)](https://arxiv.org/abs/1312.6114)
