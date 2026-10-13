---
date: 2026-10-12
topic: robotics
source: trending
repo: keivalya/mini-vla
file: models/diffusion_head.py
permalink: https://github.com/keivalya/mini-vla/blob/acd096218737794f50eb1288ad926c9b05ba3fc4/models/diffusion_head.py#L91-L155
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, diffusion-policy, action-head]
---

# mini-vla diffusion head：动作先加噪，再学会还原 / mini-vla Diffusion Head: Add Noise to Actions, Then Learn to Undo It

> **一句话 / In one line**: 这个小型 VLA action head 用 DDPM 风格的噪声预测，把 fused condition token 转成连续动作。 / This small VLA action head uses DDPM-style noise prediction to turn a fused condition token into continuous actions.

## 为什么重要 / Why this matters

`mini-vla` 的价值在于小。它不是生产级 VLA，但代码短到足够看清“扩散动作 head”的核心：训练时随机选 timestep，把真实动作加噪，让网络预测噪声；推理时从随机动作开始，一步步去噪回动作。这个结构是很多大型 VLA 连续动作 head 的教学缩影。

The value of `mini-vla` is its size. It is not a production VLA, but it is short enough to expose the core of a diffusion action head: during training, pick a random timestep, corrupt the real action, and predict the noise; during inference, start from random actions and denoise step by step. This is a compact teaching version of many larger continuous-action VLA heads.

## 代码 / The code

`keivalya/mini-vla` - [`models/diffusion_head.py`](https://github.com/keivalya/mini-vla/blob/acd096218737794f50eb1288ad926c9b05ba3fc4/models/diffusion_head.py#L91-L155)

```python
class DiffusionPolicyHead(nn.Module):
    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        self.cfg = cfg
        self.denoise_model = ActionDenoiseModel(cfg)
        betas, alphas, alpha_bar = make_beta_schedule(cfg)
        # register as buffers so they move with the module’s device
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

    def q_sample(self, x0, t, noise):
        """
        Forward diffusion: x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*noise
        x0: (B, action_dim)
        t:  (B,) integer
        """
        # gather alpha_bar_t for each t in batch
        alpha_bar_t = self.alpha_bar[t].unsqueeze(-1)  # (B, 1)
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise

    def loss(self, actions, cond):
        """
        actions: (B, action_dim)  ground-truth actions
        cond:    (B, cond_dim)    fused VLA token
        """
        B = actions.size(0)
        device = actions.device
        t = torch.randint(0, self.cfg.T, (B,), device=device) # uniform sampling t
        noise = torch.randn_like(actions)
        x_t = self.q_sample(actions, t, noise)  # noisy actions
        eps_pred = self.denoise_model(x_t, t, cond)
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def sample(self, cond, n_samples=None):
        """
        cond: (B, cond_dim) or (1, cond_dim)
        returns: (B, action_dim) sampled actions x_0
        """
        self.eval()
        if n_samples is None:
            B = cond.size(0)
        else:
            B = n_samples
            cond = cond.expand(B, -1)

        x_t = torch.randn(B, self.cfg.action_dim, device=cond.device)
        for t_step in reversed(range(self.cfg.T)):
            t = torch.full((B,), t_step, device=cond.device, dtype=torch.long)
            eps_pred = self.denoise_model(x_t, t, cond)
            beta_t = self.betas[t_step]
            alpha_t = self.alphas[t_step]
            alpha_bar_t = self.alpha_bar[t_step]

            # now reverse diffusion  (aka DDPM)
            # mean = (1 / torch.sqrt(alpha_t)) * (x_t - beta_t / torch.sqrt(1 - alpha_bar_t) * eps_pred) # original
            x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t) # simplified
            if t_step > 0:
                noise = torch.randn_like(x_t)
                # x_t = mean + torch.sqrt(beta_t) * noise # original
                x_t = torch.sqrt(alpha_t) * x0_pred + torch.sqrt(beta_t) * noise
            else:
                x_t = x0_pred # x0_pred or mean
        return x_t
```

## 逐行讲解 / What's happening

1. **第 91-100 行 / Lines 91-100 (buffers)**:
   - 中文: beta、alpha、alpha_bar 是噪声时间表。注册成 buffer 后，它们会跟模型一起搬到 CPU/GPU，但不会被优化器训练。
   - English: `betas`, `alphas`, and `alpha_bar` form the noise schedule. Registering them as buffers moves them with the module without making them trainable parameters.

2. **第 102-110 行 / Lines 102-110 (`q_sample`)**:
   - 中文: 前向扩散把真实动作 `x0` 和随机噪声按 `alpha_bar_t` 混合，得到第 `t` 步的 noisy action。
   - English: Forward diffusion mixes the real action `x0` with random noise according to `alpha_bar_t`, producing the noisy action at timestep `t`.

3. **第 112-123 行 / Lines 112-123 (`loss`)**:
   - 中文: 训练时对 batch 里每个样本随机抽时间步，让 denoise model 根据 noisy action、timestep 和条件 token 预测噪声。
   - English: During training, each sample gets a random timestep. The denoiser predicts the noise from the noisy action, timestep, and condition token.

4. **第 125-155 行 / Lines 125-155 (`sample`)**:
   - 中文: 推理从随机动作开始，按时间步倒序迭代。每步预测噪声，估计 `x0_pred`，再走到更干净的动作。
   - English: Inference starts from random actions and iterates backward through timesteps. Each step predicts noise, estimates `x0_pred`, and moves toward a cleaner action.

## 类比 / The analogy

这像学会修复一张逐渐被涂黑的路线图。训练时你随机把路线图涂到不同程度，让模型猜“刚才涂上的那层黑色”。推理时从一张乱涂的图开始，一层一层擦回可执行路线。

It is like learning to restore a route map that has been gradually scribbled over. During training, you corrupt the map by different amounts and ask the model to predict the latest scribble. During inference, you start from a messy map and erase it layer by layer into an executable route.

## 自己跑一遍 / Try it yourself

```python
import math, random

T = 4
betas = [0.01, 0.02, 0.03, 0.04]
alphas = [1 - b for b in betas]
alpha_bar = []
prod = 1
for a in alphas:
    prod *= a
    alpha_bar.append(prod)

x0 = 2.0
t = random.randrange(T)
noise = 0.5
x_t = math.sqrt(alpha_bar[t]) * x0 + math.sqrt(1 - alpha_bar[t]) * noise
eps_pred = noise
x0_pred = (x_t - math.sqrt(1 - alpha_bar[t]) * eps_pred) / math.sqrt(alpha_bar[t])
print(round(x_t, 3), round(x0_pred, 3))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
<a noisy value> 2.0
```

如果噪声预测完全正确，公式能把 noisy action 还原回原始动作。真实模型的训练目标就是让 `eps_pred` 越来越接近那份噪声。

If the noise prediction is exact, the formula recovers the original action from the noisy one. The real model's training objective is to make `eps_pred` approach that noise.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Diffusion Policy** / **Diffusion Policy**: 机器人 diffusion policy 通常也用噪声预测或 velocity prediction 训练连续动作序列。 / Robot diffusion policies often train continuous action sequences with noise prediction or velocity prediction.
- **openpi pi0** / **openpi pi0**: `openpi` 的 flow-matching action head 走的是连续时间速度场，但同样是“条件 token + noisy action + time”的结构。 / `openpi` uses a continuous-time flow-matching velocity field, but the structure is still "condition token + noisy action + time."

## 注意事项 / Caveats / when it breaks

- **这是教学版 DDPM** / **This is a teaching DDPM**: 采样公式很简化，没有高阶 solver、动作约束、normalizer 或 chunk 级 smoothing。 / The sampler is simplified and omits higher-order solvers, action constraints, normalizers, and chunk-level smoothing.
- **condition 质量决定动作质量** / **Condition quality controls action quality**: 如果视觉/语言/state fusion 很弱，再好的 denoiser 也只是在错误条件下去噪。 / If vision/language/state fusion is weak, the denoiser merely denoises under the wrong condition.

## 延伸阅读 / Further reading

- [Source permalink](https://github.com/keivalya/mini-vla/blob/acd096218737794f50eb1288ad926c9b05ba3fc4/models/diffusion_head.py#L91-L155)
- [mini-vla repository](https://github.com/keivalya/mini-vla)

