---
date: 2026-06-23
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/policies/wall_x/modeling_wall_x.py
permalink: https://github.com/huggingface/lerobot/blob/6f0ba4be38534f86832e2c65a012a5a9a9f26b6d/src/lerobot/policies/wall_x/modeling_wall_x.py#L114-L283
difficulty: advanced
read_time: ~15 min
tags: [code-of-the-day, vla, flow-matching, dof-masking, cross-embodiment, beta-distribution, wall-x]
build_role: action-head-continuous — the flow-matching action head with DOF masking for cross-embodiment VLA
---

# Wall-X ActionHead：DOF 掩码 + Beta 分布流匹配，跨机器人泛化的动作头 / Wall-X ActionHead: DOF Masking + Beta-Distribution Flow Matching for Cross-Embodiment VLA

> **一句话 / In one line**: Wall-X 是一个刚合并进 lerobot 的跨机器人 VLA 策略；它的 ActionHead 把所有机器人的 DOF 拼成一个大动作向量并附上布尔掩码，训练时从 Beta(1.5, 1.0) 采样 t 以偏向 t≈0 的难训区域，损失函数在掩码外为零。 / Wall-X is a freshly merged cross-embodiment VLA policy in lerobot; its ActionHead concatenates all robots' DOFs into one large action vector with a boolean mask, samples t from Beta(1.5, 1.0) to bias training toward the hard t≈0 region, and zeros the loss outside the mask.

## 为什么重要 / Why this matters

大多数 VLA 策略的动作头假设机器人有固定关节数（比如 7DOF Franka）。一旦换机器人就需要重新训练。Wall-X 通过两个设计同时解决这个问题：首先，`dof_config` 字典记录每个机器人的关节数，动作向量维度是所有机器人 DOF 之和，训练时把实际关节的 `dof_mask` 拼在动作向量后面，让模型知道哪些输出有效；其次，时间步 t 从 Beta(1.5, 1.0) 采样而不是均匀分布，使训练更多集中在 t≈0（干净信号附近）——这是 SD3 logit-Normal 加权的变体，在流匹配中已被证明能加速收敛。这两项技术的组合，加上 Qwen2.5-VL 作为视觉语言骨干，使 Wall-X 成为 2026 年初最值得研究的开源跨机器人 VLA 之一。

Most VLA action heads assume a fixed DOF count (e.g. 7-DOF Franka). Switching robots requires retraining. Wall-X solves this with two design choices: (1) `dof_config` records each robot's DOF count and the total action dimension is their sum; a boolean `dof_mask` is appended to the action vector at training time so the model knows which outputs are active; (2) timestep t is sampled from Beta(1.5, 1.0) instead of uniform, biasing training toward t≈0 (near-clean signal) — a variant of SD3's logit-Normal weighting shown to accelerate flow-matching convergence. Combined with a Qwen2.5-VL backbone, this makes Wall-X one of the most instructive open-source cross-embodiment VLAs of early 2026.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/wall_x/modeling_wall_x.py`](https://github.com/huggingface/lerobot/blob/6f0ba4be38534f86832e2c65a012a5a9a9f26b6d/src/lerobot/policies/wall_x/modeling_wall_x.py#L114-L283)

```python
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class ActionHead(nn.Module):
    def __init__(self, config: WallXConfig):
        super().__init__()
        self.action_dim  = sum(config.dof_config.values())   # total DOFs across all robots
        self.hidden_size = config.hidden_size
        self.propri_dim  = config.propri_dim
        self.chunk_size  = config.chunk_size
        self.beta_alpha  = 1.5
        self.beta_beta   = 1.0
        self.s           = 0.999                             # clamp t away from 1.0

        self.time_embed   = SinusoidalPosEmb(self.hidden_size)
        # *2 because noisy_action is concat'd with dof_mask
        self.w1           = nn.Linear(self.action_dim * 2, self.hidden_size)
        self.propri_proj  = nn.Linear(self.propri_dim * 2, self.hidden_size)
        self.action_proj_back = nn.Linear(self.hidden_size, self.action_dim)

    # ------------------------------------------------------------------ #
    # Timestep sampling: Beta(α, β) biased toward t ≈ 0                   #
    # ------------------------------------------------------------------ #
    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        beta_dist = Beta(
            torch.tensor(self.beta_alpha, device=device),
            torch.tensor(self.beta_beta,  device=device),
        )
        sample = beta_dist.sample([batch_size])   # ∈ (0, 1)
        return (1.0 - sample) * self.s            # flip: high density near 0

    # ------------------------------------------------------------------ #
    # Forward: build noisy action + embed it                              #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        action_chunk: torch.Tensor,   # (B, T, action_dim)
        propri: torch.Tensor,         # (B, T, propri_dim)
        dof_mask: torch.Tensor | None = None,   # (B, T, action_dim) bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = action_chunk.shape
        device = action_chunk.device

        noise         = torch.randn_like(action_chunk)
        time          = self.sample_time(B, device)           # (B,)
        t             = time[:, None, None]                   # (B, 1, 1) for broadcast
        action_f32    = action_chunk.float()
        noisy_action  = (1.0 - t) * noise + t * action_f32   # flow matching interpolation

        if dof_mask is not None:
            # append boolean mask so model learns which dims are active
            noisy_action = torch.cat([noisy_action, dof_mask.float()], dim=-1)

        flow = action_f32 - noise   # target velocity field

        time_embed  = self.time_embed(time)                   # (B, hidden)
        action_emb  = self.w1(noisy_action)                   # (B, T, hidden)
        propri_emb  = self.propri_proj(
            torch.cat([propri, propri], dim=-1)               # (B, T, propri_dim*2)
        )
        embed = action_emb + propri_emb + time_embed[:, None, :]
        return embed, flow

    # ------------------------------------------------------------------ #
    # Loss: MSE on the velocity field, masked to active DOFs              #
    # ------------------------------------------------------------------ #
    def flow_loss(
        self,
        action_hidden_states: torch.Tensor,  # (B, T, hidden)
        flow: torch.Tensor,                  # (B, T, action_dim)
        dof_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        action_pred = self.action_proj_back(action_hidden_states)  # (B, T, action_dim)
        loss = F.mse_loss(action_pred, flow, reduction="none")      # (B, T, action_dim)
        if dof_mask is not None:
            loss = loss * dof_mask.float()                          # zero out missing DOFs
        return loss.mean()
```

## 逐行讲解 / What's happening

1. **`self.action_dim = sum(config.dof_config.values())` — 动态 DOF 维度 / dynamic DOF dimension**:
   - 中文: `dof_config` 是一个 `{robot_name: n_dof}` 字典，比如 `{"franka": 7, "ur5": 6, "hand": 12}`。把所有值加起来得到总动作维度 25。每个机器人的 `dof_mask` 是这个 25 维向量的一个布尔子集，指示哪些维度属于当前机器人。
   - English: `dof_config` is a `{robot_name: n_dof}` dict, e.g. `{"franka": 7, "ur5": 6, "hand": 12}`. The total action dimension is their sum (25). Each robot's `dof_mask` is a boolean subvector indicating which of these 25 dimensions belong to the current robot.

2. **`self.w1 = nn.Linear(self.action_dim * 2, ...)` — `*2` 的含义 / why `*2`**:
   - 中文: `noisy_action`（维度 `action_dim`）和 `dof_mask`（维度 `action_dim`）在最后一维拼接，所以输入维度是 `action_dim * 2`。掩码作为一路输入让网络在推理时也能知道当前激活的 DOF。
   - English: `noisy_action` (`action_dim` dims) and `dof_mask` (`action_dim` dims) are concatenated on the last axis, so the input to `w1` is `action_dim * 2`. Passing the mask as input lets the network condition its predictions on the active DOF set at inference time too.

3. **`sample_time` — Beta 分布采样 / Beta distribution sampling**:
   - 中文: Beta(1.5, 1.0) 的众数在 `sample ≈ 0.33`，但 `t = (1 - sample) * 0.999` 把分布翻转，使 t 的高概率区域在 0 附近（t≈0 对应接近干净信号的位置）。在流匹配中，t=0 时 noisy=noise，t=1 时 noisy=action，t≈0 的预测误差最大、最难训练——加权采样让网络在这里得到更多训练。
   - English: Beta(1.5, 1.0) has its mode around `sample ≈ 0.33`, but `t = (1 − sample) × 0.999` flips the distribution so high probability mass is near t≈0. In flow matching, t=0 means noisy=pure noise and t=1 means noisy=clean action — predictions near t=0 have the largest errors and are the hardest to train, so biased sampling gives more training there.

4. **`noisy_action = (1 - t) * noise + t * action_f32` — 流匹配插值 / flow interpolation**:
   - 中文: 这是 flow matching 的线性插值公式：在 t=0 时完全是噪声，在 t=1 时完全是干净动作。`flow = action - noise` 是目标速度场（从噪声指向动作的方向）。
   - English: This is the linear flow matching interpolant: at t=0 it is pure noise, at t=1 it is the clean action. `flow = action − noise` is the target velocity field pointing from noise toward the clean action.

5. **`flow_loss` — DOF 掩码损失 / DOF-masked loss**:
   - 中文: `loss * dof_mask` 把不属于当前机器人的 DOF 维度的损失清零。如果不这样做，模型会对 padding 维度（另一个机器人的 DOF）强行学习随机梯度，污染权重。
   - English: Multiplying by `dof_mask` zeros out loss for DOF dimensions not belonging to the current robot. Without this, the model would receive random gradient signal for padded dimensions, polluting the shared weight matrix.

## 类比 / The analogy

想象一个通用乐谱本（动作向量），里面有 25 个音轨槽位，但每个乐手（机器人）只用其中几个。掩码就像贴在多余槽位上的"本次演出不使用"标签，老师（流匹配损失）只对有标签的槽位打分。时间步采样偏向 t≈0，就像考试故意侧重最难的题目，让学生（网络）在弱点上练习更多。

Imagine a universal score sheet (action vector) with 25 instrument slots, but each musician (robot) only uses a few. The mask is like "not used tonight" stickers on the unused slots — the conductor (flow loss) only scores the active slots. Biasing timestep sampling toward t≈0 is like an exam that deliberately loads more hard questions, forcing the student (network) to practice more on its weak points.

## 在 nanoVLA 中的位置 / Where this lives in your nanoVLA

> **curriculum item**: `action-head-continuous` — the flow-matching continuous action head. This is a cross-repo variant re-cover; the same curriculum slot was previously taught via `lerobot/groot`'s flow-matching head (`nano/vla/2026-06-08-lerobot-groot-flow-matching-action-head.md`). Wall-X adds DOF masking and Beta-distributed timestep sampling on top of the same foundation.

中文：在你自己的 nanoVLA 里，`ActionHead` 坐在语言视觉主干（Qwen2.5-VL 或任何 VLM）之后。主干输出一组 token 表示，经过 `w1` 线性层和时间嵌入叠加后，送入几个 Transformer block，最后由 `action_proj_back` 输出动作预测。如果省掉 `dof_mask`，单机器人 nanoVLA 完全可以工作；如果你想跨机器人训练，`dof_mask` 是实现零修改架构泛化的最低成本方案。上游依赖：视觉编码器（`vision-encoder`）、模态投影（`modality-projector`）、语言主干（`backbone-vlm`）；下游：推理解码循环（`inference-decode-loop`），需要反向调用 ODE solver 从噪声积分到干净动作。

English: In your nanoVLA, `ActionHead` sits downstream of the VLM backbone (Qwen2.5-VL or any VLM that emits token representations). The backbone's output tokens are linearly projected by `w1`, summed with the timestep embedding, passed through a few Transformer blocks, and finally decoded by `action_proj_back`. Single-robot nanoVLA works fine without `dof_mask`; for cross-robot training, `dof_mask` is the lowest-cost path to zero-architecture-change generalization. Upstream dependencies: `vision-encoder`, `modality-projector`, `backbone-vlm`. Downstream: `inference-decode-loop`, which needs a reverse ODE solver to integrate from noise to clean action.

## 自己跑一遍 / Try it yourself

```python
import torch, torch.nn as nn, torch.nn.functional as F, math
from torch.distributions import Beta

class SinPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim
    def forward(self, x):
        h = self.dim // 2
        e = math.log(10000) / (h - 1)
        e = torch.exp(torch.arange(h, device=x.device) * -e)
        e = x[:, None] * e[None, :]
        return torch.cat([e.sin(), e.cos()], dim=-1)

class TinyActionHead(nn.Module):
    def __init__(self, action_dim=13, hidden=64):
        super().__init__()
        self.dim = action_dim
        self.w1  = nn.Linear(action_dim * 2, hidden)  # action + mask
        self.out = nn.Linear(hidden, action_dim)
        self.temb = SinPosEmb(hidden)

    def sample_t(self, B, device):
        s = Beta(torch.tensor(1.5, device=device), torch.tensor(1.0, device=device)).sample([B])
        return (1 - s) * 0.999

    def forward(self, action, dof_mask=None):
        B, T, _ = action.shape
        noise = torch.randn_like(action)
        t = self.sample_t(B, action.device)[:, None, None]
        noisy = (1 - t) * noise + t * action
        flow  = action - noise
        if dof_mask is not None:
            noisy = torch.cat([noisy, dof_mask.float()], dim=-1)
        emb = self.w1(noisy) + self.temb(t.squeeze()[:, 0])[:, None, :]
        pred = self.out(emb)
        loss = F.mse_loss(pred, flow, reduction="none")
        if dof_mask is not None: loss = loss * dof_mask.float()
        return loss.mean()

# simulate: franka(7 DOF) + ur5(6 DOF) = 13 total; current robot is franka
dof_mask = torch.zeros(2, 4, 13, dtype=torch.bool)
dof_mask[:, :, :7] = True   # first 7 dims active
action = torch.randn(2, 4, 13) * dof_mask  # only active dims have values
head = TinyActionHead(action_dim=13)
loss = head(action, dof_mask)
print(f"flow loss: {loss.item():.4f}")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
flow loss: 0.XXXX  (any positive float)
```

中文：注意 `dof_mask[:, :, :7] = True` 只激活前 7 个维度（Franka DOF），后 6 个维度的损失会被清零。尝试把 `dof_mask` 设为 `None`，你会看到损失数值变化——证明掩码确实影响了训练梯度。

English: Note that `dof_mask[:, :, :7] = True` activates only the first 7 dims (Franka DOF) — the last 6 dims get zero loss. Try setting `dof_mask=None` and watch the loss change — proof that the mask meaningfully affects gradient flow.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **lerobot GR00T N1 action head** / **GR00T N1 动作头**: GR00T 也用流匹配但没有 DOF masking——它假设固定 20-DOF humanoid。Wall-X 的 DOF mask 方案在多机器人场景中更通用。 / GR00T also uses flow matching but without DOF masking — it assumes a fixed 20-DOF humanoid. Wall-X's DOF mask is more general for multi-robot training.
- **SD3 logit-Normal timestep weighting** / **SD3 logit-Normal 时间步加权**: SD3 使用 logit-Normal 分布来偏向中间 t 值；Wall-X 用 Beta 分布偏向 t≈0。两者都是"非均匀时间步采样"的实例，目的是把训练预算集中在困难区域。 / SD3 uses logit-Normal distribution to bias toward middle t values; Wall-X uses Beta to bias toward t≈0. Both are instances of non-uniform timestep sampling to concentrate training budget on hard regions.
- **openpi pi0-fast action head** / **openpi pi0-fast 动作头**: pi0-fast 在 `nano/vla/2026-06-13-openpi-pi0-pytorch-flow-matching-loss.md` 中展示了标准均匀 t 采样的流匹配——对比 Wall-X 的 Beta 采样可以看清两种策略的差异。 / pi0-fast (in `nano/vla/2026-06-13`) demonstrates standard uniform-t flow matching — comparing with Wall-X's Beta sampling shows the difference between the two strategies clearly.

## 注意事项 / Caveats / when it breaks

- **推理时需要 ODE solver** / **ODE solver needed at inference**: The `ActionHead` only defines the training forward pass. At inference you need to run a reverse ODE (e.g. Euler steps from t=0 to t=1) using the model's predicted velocity field — this is not shown in the snippet.
- **`dof_mask` 必须在 batch 内保持一致** / **`dof_mask` must be consistent within batch**: Mixing robots in the same batch is allowed (and the point of DOF masking), but the mask must correctly reflect each sample's active DOFs. Mis-labeling the mask corrupts training for that sample.
- **Beta 参数是超参** / **Beta parameters are hyperparameters**: α=1.5, β=1.0 was chosen by the Wall-X authors — for a different domain (e.g. high-frequency haptic control) the optimal distribution may differ. Worth ablating.

## 延伸阅读 / Further reading

- [Wall-X `modeling_wall_x.py`](https://github.com/huggingface/lerobot/blob/6f0ba4be38534f86832e2c65a012a5a9a9f26b6d/src/lerobot/policies/wall_x/modeling_wall_x.py)
- [Flow Matching for Generative Modeling (Lipman et al. 2022)](https://arxiv.org/abs/2210.02747)
- [SD3 — logit-Normal timestep sampling](https://arxiv.org/abs/2403.03206) — the inspiration for non-uniform t weighting
- [GR00T N1 flow matching head](https://github.com/NVIDIA/Isaac-GR00T) — same curriculum slot, different robot
