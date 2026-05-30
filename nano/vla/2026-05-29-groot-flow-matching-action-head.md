---
date: 2026-05-29
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/policies/groot/action_head/flow_matching_action_head.py
permalink: https://github.com/huggingface/lerobot/blob/24017e960c39a24fe1b6ea6248522460fa5aa4b3/src/lerobot/policies/groot/action_head/flow_matching_action_head.py#L300-L343
difficulty: advanced
read_time: ~12 min
tags: [code-of-the-day, vla, action-head, flow-matching, continuous-actions]
build_role: Continuous action head — produces continuous action trajectories via flow matching instead of discrete tokens
---

# 连续动作 head:flow matching 让机器人输出实数轨迹 / Continuous action head: flow matching emits real-valued trajectories

> **一句话 / In one line**: GR00T 的 action head 不把动作离散成 token,而是用 flow matching —— 在干净动作和高斯噪声之间线性插值得到 noisy 轨迹,让 DiT 预测"指向干净动作的速度场" `velocity = actions - noise`,训练就是一个 MSE。 / GR00T's action head skips token discretisation and uses flow matching — interpolate between clean actions and Gaussian noise, let a DiT predict the velocity field `velocity = actions - noise`, and training is one MSE.

## 为什么重要 / Why this matters

VLA 输出动作有两条路线:**离散**(像 OpenVLA 把动作分箱成 token,用交叉熵,详见 action tokenizer 笔记)和**连续**(像 π₀、GR00T 用 flow matching / diffusion 直接回归实数)。连续路线的优势:(1) 动作天然是连续量(关节角、末端位姿),离散化会损失精度;(2) flow matching 一步训练目标极简(就是 MSE on velocity);(3) 推理可以少步数(4-10 步)。这段代码是 GR00T 风格连续 head 的核心 forward —— 看懂这 40 行,你就掌握了 2025-2026 主流 VLA 的动作生成方式。它跟昨天 WAM 那边的 flow matching 是**同一个数学**,只是把"去噪视频"换成"去噪动作轨迹"。

VLAs emit actions two ways: **discrete** (OpenVLA bins actions into tokens, cross-entropy — see the action tokenizer note) and **continuous** (π₀, GR00T regress real values via flow matching / diffusion). Continuous wins because (1) actions are inherently continuous (joint angles, end-effector poses) and binning loses precision, (2) flow matching's training target is dead simple (MSE on velocity), (3) inference can be few-step (4-10). This is the core forward of a GR00T-style continuous head — 40 lines that capture how most 2025-2026 VLAs generate actions. It's the *same maths* as yesterday's WAM flow matching, just "denoise an action trajectory" instead of "denoise video".

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/groot/action_head/flow_matching_action_head.py`](https://github.com/huggingface/lerobot/blob/24017e960c39a24fe1b6ea6248522460fa5aa4b3/src/lerobot/policies/groot/action_head/flow_matching_action_head.py#L300-L343)

```python
# Embed noised action trajectory.
actions = action_input.action
noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
t = t[:, None, None]  # shape (B,1,1) for broadcast

noisy_trajectory = (1 - t) * noise + t * actions
velocity = actions - noise

# Convert (continuous) t -> discrete if needed
t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

# Maybe add position embedding.
if self.config.add_pos_embed:
    pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
    pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
    action_features = action_features + pos_embs

# Join vision, language, state and action embedding along sequence dimension.
future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

model_output = self.model(
    hidden_states=sa_embs,
    encoder_hidden_states=vl_embs,                       # vision-language cross-attention context
    encoder_attention_mask=backbone_output.backbone_attention_mask,
    timestep=t_discretized,
)
pred = self.action_decoder(model_output, embodiment_id)
pred_actions = pred[:, -actions.shape[1] :]

# Slice out only the action portion of pred and target.
action_mask = action_input.action_mask
loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
loss = loss.sum() / action_mask.sum()
```

## 逐行讲解 / What's happening

1. **`sample_time` 用 Beta 分布而非均匀 / Beta-distributed timesteps**:
   - 中文:上方 `sample_time` 用 `Beta(1.5, 1.0)` 采样 t,不是均匀分布。这是 π₀ 引入的技巧 —— 让训练更多落在 t 接近 1(接近干净动作)的区域,因为那一端的 velocity 预测对最终精度影响最大。`(noise_s - sample) / noise_s` 做了个缩放。
   - English: `sample_time` draws t from `Beta(1.5, 1.0)`, not uniform. A π₀ trick — concentrate training near t≈1 (close to clean actions), where velocity prediction most affects final precision. The `(noise_s - sample) / noise_s` rescales it.

2. **`noisy_trajectory = (1 - t) * noise + t * actions` / Linear interpolation**:
   - 中文:flow matching 的"加噪"就是干净动作和噪声的线性插值。t=0 全是噪声,t=1 全是干净动作。注意这跟 DDPM 的非线性 schedule 不同 —— rectified flow 走直线,所以推理可以少步。
   - English: flow matching's "noising" is a straight-line interpolation between clean actions and noise. t=0 is pure noise, t=1 is clean. Unlike DDPM's curved schedule — rectified flow goes straight, enabling few-step inference.

3. **`velocity = actions - noise` 是训练目标 / The training target**:
   - 中文:这是整段最关键的一行。模型不预测噪声(DDPM),也不预测干净动作,而是预测**速度场** —— 从当前 noisy 点指向干净动作的方向。因为插值是直线,velocity 是常量 `actions - noise`,与 t 无关,数学极干净。
   - English: the single most important line. The model predicts neither the noise (DDPM) nor the clean action, but the **velocity field** — the direction from the current noisy point toward the clean action. Because the interpolation is a straight line, the velocity is the constant `actions - noise`, independent of t. Beautifully simple.

4. **`t_discretized` 喂给 DiT 的 timestep embedding / Discretised timestep**:
   - 中文:连续 t 乘以 `num_timestep_buckets` 取整,变成离散桶索引,送进 DiT 的 timestep embedder(查表)。这是工程取舍 —— 连续 t 也能用 sinusoidal embedding,但分桶查表更稳。
   - English: continuous t is multiplied by `num_timestep_buckets` and floored into a discrete bucket index for the DiT's timestep embedder (a lookup table). An engineering choice — continuous t could use sinusoidal embedding, but bucketed lookup is more stable.

5. **`action_encoder(noisy_trajectory, t_discretized, embodiment_id)` 带 embodiment / Multi-embodiment action encoder**:
   - 中文:注意这里的 `action_encoder` 是 `MultiEmbodimentActionEncoder`,带 `embodiment_id` —— 用的就是前几天讲的 `CategorySpecificLinear`!每种机器人本体一份权重。这把"多本体"和"flow matching head"两件事缝在了一起。
   - English: the `action_encoder` here is a `MultiEmbodimentActionEncoder` taking `embodiment_id` — it uses the `CategorySpecificLinear` from an earlier note! One weight slice per robot body. This stitches "multi-embodiment" and "flow-matching head" together.

6. **`sa_embs = cat(state, future_tokens, action_features)` / Sequence assembly**:
   - 中文:把 state token、可学习的 `future_tokens`(给模型留的"思考槽")、和 noisy action token 沿序列维拼起来,作为 DiT 的 `hidden_states`;而 vision-language 特征 `vl_embs` 作为 `encoder_hidden_states` 走 cross-attention。这是典型的"action 做 query、VL 做 context"结构 —— 跟昨天 SmolVLA expert 的设计同源。
   - English: concatenate state tokens, learnable `future_tokens` (thinking slots), and noisy action tokens along the sequence as the DiT's `hidden_states`; the vision-language features `vl_embs` enter as `encoder_hidden_states` via cross-attention. The classic "action as query, VL as context" structure — same lineage as yesterday's SmolVLA expert.

7. **`loss = MSE(pred_actions, velocity) * action_mask` / Masked MSE**:
   - 中文:损失就是预测速度和真实速度的 MSE,乘 `action_mask` 排除 padding 步。`pred[:, -actions.shape[1]:]` 切出序列尾部的 action 部分(前面是 state 和 future token)。整个训练目标就这一个 MSE,没有 KL、没有对抗、没有交叉熵。
   - English: the loss is MSE between predicted and true velocity, masked to exclude padding steps. `pred[:, -actions.shape[1]:]` slices the action portion off the sequence tail (state and future tokens are in front). The entire training objective is this one MSE — no KL, no adversarial, no cross-entropy.

## 类比 / The analogy

像教人开车从随机位置回到车道中央。离散派(OpenVLA)是把方向盘角度切成 256 档,让学员"选一档";连续派(flow matching)是直接教"现在该往哪个方向打多少、打多快"——给学员一个起点(noisy 动作)和一个箭头(velocity),学员只要照箭头走就能到车道中央。因为路线是直的,几步就能到位,不用反复微调。

Picture teaching someone to steer back to the lane center from a random position. The discrete school (OpenVLA) chops the steering angle into 256 bins and asks "pick a bin". The continuous school (flow matching) directly teaches "turn this direction, this much, this fast" — give the learner a start point (noisy action) and an arrow (velocity), and they just follow the arrow to center. Because the path is straight, a few steps suffice with no fiddly re-adjustment.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-VLA

中文:在 nanoVLA 里这是 `nano/vla/heads/flow_matching_head.py` —— 接在 VLM backbone 之后的动作生成模块。上游:VLM backbone 输出的 `vl_embs`(视觉语言特征,做 cross-attention context)+ 机器人 state;下游:训练时输出 loss,推理时跑 4-10 步 ODE 积分(从纯噪声出发,每步 `x += dt * velocity_pred`)解出动作轨迹。它依赖:`dit-block`(action 用的小 DiT)、`MultiEmbodimentActionEncoder`(= CategorySpecificLinear)。如果你的 nanoVLA 选离散路线(action tokenizer + 交叉熵),就**不需要**这个 head;但连续路线在精细操作任务(插孔、倒水)上明显更准。生产实现要补:(1) **推理 ODE 求解器**(Euler 足够,Heun 更准);(2) **action chunking**(一次预测未来 H 步,见今天的 chunking 笔记);(3) **CFG**(对 language instruction 做 classifier-free guidance,增强指令服从)。

English: in nanoVLA this is `nano/vla/heads/flow_matching_head.py` — the action generator after the VLM backbone. Upstream: the backbone's `vl_embs` (vision-language features as cross-attention context) + robot state. Downstream: in training it emits the loss; at inference it runs a 4-10 step ODE integration (start from pure noise, each step `x += dt * velocity_pred`) to solve for the action trajectory. It depends on `dit-block` (a small action DiT) and `MultiEmbodimentActionEncoder` (= CategorySpecificLinear). If your nanoVLA picks the discrete route (action tokenizer + cross-entropy), you **don't need** this head; but the continuous route is clearly more precise on fine manipulation (peg insertion, pouring). Production additions: (1) **inference ODE solver** (Euler suffices, Heun is more accurate), (2) **action chunking** (predict H future steps at once — see today's chunking note), (3) **CFG** on the language instruction to boost instruction-following.

## 自己跑一遍 / Try it yourself

```python
# pip install torch
# Minimal flow-matching action head: train to "denoise" toward a target action, then sample.
import torch, torch.nn as nn

torch.manual_seed(0)
B, horizon, action_dim, hidden = 64, 4, 7, 128

class FlowHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc  = nn.Linear(action_dim, hidden)
        self.t_emb = nn.Linear(1, hidden)
        self.net  = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, action_dim))
    def forward(self, noisy, t):
        h = self.enc(noisy) + self.t_emb(t)[:, None, :]
        return self.net(h)                      # predicted velocity

head = FlowHead()
opt  = torch.optim.Adam(head.parameters(), lr=1e-3)
target = torch.randn(B, horizon, action_dim)    # the "expert" actions

for step in range(400):
    noise = torch.randn_like(target)
    t = torch.rand(B, 1)
    noisy = (1 - t)[:, :, None] * noise + t[:, :, None] * target
    velocity = target - noise                    # flow-matching target
    pred = head(noisy, t)
    loss = ((pred - velocity) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0:
        print(f"step {step:3d}  loss={loss.item():.4f}")

# Inference: integrate the ODE from pure noise
x = torch.randn(1, horizon, action_dim)
steps = 8
for i in range(steps):
    t = torch.full((1, 1), i / steps)
    x = x + (1 / steps) * head(x, t)
print("sampled action[0,0]:", x[0, 0].tolist())
print("target action[0,0] :", target[0, 0].tolist())
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
step   0  loss=1.9...
step 100  loss=0.3...
step 300  loss=0.1...
sampled action[0,0]: [...]   # 8-step ODE integration from noise
target action[0,0] : [...]
```

中文:8 步 ODE 积分就能从噪声还原出接近目标的动作。注意训练目标 `velocity = target - noise` 是常量,这是 rectified flow 直线插值带来的简洁。

English: 8 ODE steps recover an action close to the target from noise. Note the training target `velocity = target - noise` is constant — the simplicity that rectified flow's straight-line interpolation buys you.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **π₀ flow-matching head** / **π₀ flow-matching head**: 中文 — 几乎一模一样的 Beta 采样 + velocity 目标,GR00T 直接借鉴 π₀。 / English — nearly identical Beta sampling + velocity target; GR00T borrows directly from π₀.
- **Diffusion Policy** / **Diffusion Policy**: 中文 — 早期连续动作派,用 DDPM 而非 flow matching,推理步数多但思想一致。 / English — the early continuous-action approach, using DDPM instead of flow matching; more inference steps, same idea.
- **昨天的 WAM flow matching (lingbot/dreamzero)** / **Yesterday's WAM flow matching**: 中文 — 完全相同的 `(1-t)*noise + t*x` 插值和 velocity 目标,只是去噪对象从动作换成视频 latent。 / English — identical `(1-t)*noise + t*x` interpolation and velocity target, just denoising video latents instead of actions.
- **OpenVLA action tokenizer(对照组)** / **OpenVLA action tokenizer (the contrast)**: 中文 — 离散路线,把动作分箱成 token 用交叉熵;两条路线的根本分歧。 / English — the discrete route, binning actions into tokens with cross-entropy; the fundamental fork.

## 注意事项 / Caveats / when it breaks

- **velocity 目标只对直线插值成立** / **The velocity target assumes straight-line interpolation**: 中文 — `velocity = actions - noise` 是 rectified flow 特有的。如果换成 DDPM 的非线性 schedule,目标要改成 epsilon 或 v-prediction。 / English — `velocity = actions - noise` is specific to rectified flow. Switch to a DDPM curved schedule and the target becomes epsilon or v-prediction.
- **Beta 采样的超参影响大** / **Beta sampling hyperparameters matter**: 中文 — `Beta(1.5, 1.0)` 偏向 t≈1;改成均匀分布会让精细动作精度下降。这是 π₀ 调出来的经验值。 / English — `Beta(1.5, 1.0)` biases toward t≈1; uniform sampling degrades fine-action precision. These are π₀'s tuned values.
- **action_mask 不能漏** / **Don't forget `action_mask`**: 中文 — episode 尾部 padding 步必须 mask,否则模型学着去预测 padding 的"假动作"。 / English — padded trailing steps must be masked or the model learns to predict fake padding actions.
- **推理步数 vs 精度权衡** / **Inference steps vs precision**: 中文 — flow matching 4 步就能用,但精细任务可能要 10 步。步数是延迟和精度的直接权衡。 / English — flow matching works at 4 steps, but fine tasks may need 10. Step count directly trades latency against precision.

## 延伸阅读 / Further reading

- [π₀: A Vision-Language-Action Flow Model (Physical Intelligence, 2024)](https://arxiv.org/abs/2410.24164)
- [Flow Matching for Generative Modeling (Lipman et al., 2022)](https://arxiv.org/abs/2210.02747)
- [GR00T N1 tech report](https://github.com/NVIDIA/Isaac-GR00T)
- [Today's VLA action survey doc](./README-action-survey.md)
