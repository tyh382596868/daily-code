---
date: 2026-05-26
topic: robotics
source: tracked
repo: Physical-Intelligence/openpi
file: src/openpi/models/pi0.py
permalink: https://github.com/Physical-Intelligence/openpi/blob/c23745b5ad24e98f66967ea795a07b2588ed6c79/src/openpi/models/pi0.py#L189-L214
difficulty: intermediate
read_time: ~12 min
tags: [code-of-the-day, robotics, vla, flow-matching, pi0, jax]
---

# π₀ 的 Flow Matching 训练损失:25 行代码教会机器人动起来 / π₀'s Flow Matching Loss: How a Robot Learns to Move in 25 Lines

> **一句话 / In one line**: π₀ 不用扩散模型,而是用 rectified flow + beta 分布时间采样,把"学会动作"压成 25 行 JAX 代码 / Instead of denoising diffusion, π₀ trains its action head with rectified flow and a beta-distributed time prior — and the whole thing fits in 25 lines of JAX.

## 为什么重要 / Why this matters

Physical Intelligence 的 π₀ 是 2024–2025 年最火的开源 VLA(Vision-Language-Action)模型之一。它的"看图说话+生成动作"主干是 PaliGemma,但真正决定机器人执行质量的,是接在 LLM 后面的 **action expert**:它接收一段从噪声出发的"假动作",通过若干步去噪迭代,逐步生成真正可以下发到机械臂的关节轨迹。市面上大多数实现选了 DDPM 风格的扩散损失;π₀ 团队偏偏选了 **rectified flow matching**——一种更直接、迭代步数更少的连续归一化流变体。这段 `compute_loss` 就是整个训练目标的源头,看懂它就看懂了为什么 π₀ 推理时只需要 10 步就能输出高质量动作,而经典扩散往往要 50 步。

Physical Intelligence's π₀ is one of the most-cited open-weight VLA (Vision-Language-Action) models of 2024–2025. The vision-language backbone is PaliGemma, but the part that actually decides whether the robot's hand lands on the right cup is the **action expert**: an MLP-on-transformer head that takes a noisy "fake action" and refines it through a handful of denoising steps into joint commands you can send to the arm. Most public implementations use a DDPM-style diffusion loss. π₀ instead uses **rectified flow matching** — a continuous-normalizing-flow variant that needs far fewer inference steps. The `compute_loss` function below is the entire training objective. Once you can read it, you also understand why π₀ generates high-quality actions in 10 sampling steps instead of the 50 you typically need for diffusion.

## 代码 / The code

`Physical-Intelligence/openpi` — [`src/openpi/models/pi0.py`](https://github.com/Physical-Intelligence/openpi/blob/c23745b5ad24e98f66967ea795a07b2588ed6c79/src/openpi/models/pi0.py#L189-L214)

```python
@override
def compute_loss(
    self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
) -> at.Float[at.Array, "*b ah"]:
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

    batch_shape = actions.shape[:-2]
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
    time_expanded = time[..., None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions

    # one big forward pass of prefix + suffix at once
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask = make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1
    (prefix_out, suffix_out), _ = self.PaliGemma.llm(
        [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
    )
    v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

    return jnp.mean(jnp.square(v_t - u_t), axis=-1)
```

## 逐行讲解 / What's happening

1. **`jax.random.split(rng, 3)` (第 192 行 / Line 192)**:
   - 中文: JAX 是函数式的,随机数生成器必须显式地分裂、传递。这里一次性切出三个独立的随机数子流,分别给到"图像增广预处理"、"采样噪声"、"采样时间"。任何一个环节都不会污染其他环节的随机性,这是 JAX 训练循环的标准开局。
   - English: JAX is functional, so PRNG state must be split and passed explicitly. The call carves three independent sub-streams for image augmentation, noise sampling, and time sampling. None of them contaminate one another's randomness — the canonical opening move of any JAX training step.

2. **`time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001` (第 197 行 / Line 197)**:
   - 中文: 这一行是整段代码里最有"研究感"的一处。如果你直接采样 `time ~ Uniform(0,1)`,模型就把所有时间步当作同等重要的训练样本——但实践中,接近 `t=0`(更接近真实动作)的去噪比 `t=1`(纯噪声)更难。`Beta(1.5, 1)` 的概率密度在 `[0,1]` 区间上左低右高,采到的 `t` 偏向大值——再用 `1 - time_expanded` 翻转一下,等效于让训练时更多看到"几乎是真动作、只差一点点"的样本,这是模型最该练好的区域。`* 0.999 + 0.001` 是为了把 `t` 限制在 `(0.001, 1)` 之间,避开数值边界。
   - English: The most "research-flavored" line in the snippet. With `Uniform(0,1)` time sampling, the model treats every noise level as equally important — but empirically, the hard region is near `t=0` where it has to commit to a specific action mode (not somewhere on the noise-to-actions interpolation). `Beta(1.5, 1)` skews the density toward 1, and the later flip via `(1 - time_expanded)` effectively concentrates training on the "nearly-real action, last few percent of refinement" region. The `* 0.999 + 0.001` cutoff just avoids the numerical boundaries.

3. **`x_t = time_expanded * noise + (1 - time_expanded) * actions` (第 199 行 / Line 199)**:
   - 中文: Rectified flow 的核心定义。`x_t` 是噪声和真实动作之间的**线性插值**,`t=0` 对应真动作,`t=1` 对应纯高斯噪声。整条从噪声到动作的"轨迹"是一条直线——这正是 rectified ("拉直") 的含义。
   - English: The defining equation of rectified flow. `x_t` is the **linear interpolation** between noise and the real action — `t=0` gives you the action, `t=1` gives you pure Gaussian noise. The entire denoising trajectory is a straight line in action space; that's literally what "rectified" means.

4. **`u_t = noise - actions` (第 200 行 / Line 200)**:
   - 中文: 这是对应的**速度场**(velocity field)。沿着 `x_t` 这条直线,从 `actions` 走到 `noise` 的方向向量,在每一步都是常数 `noise - actions`。模型要学习的就是这个常数——给定任意中间状态 `x_t`,预测它应该往哪个方向"走"才能还原出动作。
   - English: The corresponding **velocity field**. Along the straight line from `actions` to `noise`, the direction vector is a constant `noise - actions`. The model's job is to predict exactly this constant: given any intermediate `x_t`, output the direction it should move to undo the noise.

5. **`embed_prefix(observation)` 与 `embed_suffix(observation, x_t, time)` (第 203–204 行 / Lines 203-204)**:
   - 中文: π₀ 把整个序列拆成了两段。**Prefix** 是图像 + 语言指令的 token(走视觉编码器 + LLM 嵌入);**Suffix** 是 `x_t`(噪声动作)+ 时间步的 token(走 action expert 的输入投影)。两段会拼起来送进一个 PaliGemma + action-expert 双塔。
   - English: π₀ splits the sequence into two halves. The **prefix** is image + language tokens (going through SigLIP + the LLM embedding). The **suffix** is the noisy actions `x_t` + the time embedding (going through the action expert's input projection). The two halves are concatenated into a single sequence and fed through PaliGemma + the action expert in one pass.

6. **`attn_mask = make_attn_mask(input_mask, ar_mask)` (第 207 行 / Line 207)**:
   - 中文: `ar_mask` 标记"哪些 token 之间需要 autoregressive 因果遮罩"。在 π₀ 里:图像和语言 token 之间是 full attention(它们就是"上下文",彼此可见);动作 token 之间也是 full attention(整段动作 chunk 一次性预测,而不是逐 token 自回归);但**动作 token 必须看到 prefix,prefix 不能看到动作**。这就是 prefix-LM 风格的混合 attention 模式。
   - English: `ar_mask` controls *which* tokens get causal masking. In π₀: image and language tokens use full attention (they're context, mutually visible). Action tokens also use full attention among themselves (the whole action chunk is predicted at once, not autoregressively). But **action tokens must attend to the prefix, and the prefix must not attend to actions**. This is the prefix-LM hybrid attention pattern.

7. **`v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])` (第 212 行 / Line 212)**:
   - 中文: 从 LLM 输出中只取最后 `action_horizon` 个 token(就是动作位的输出),投影回动作维度。这就是模型对速度场 `u_t` 的预测 `v_t`。
   - English: Slice off the last `action_horizon` tokens (the action positions) from the LLM's output and project them back to action dimensionality. That's the model's prediction `v_t` of the velocity field `u_t`.

8. **`jnp.mean(jnp.square(v_t - u_t), axis=-1)` (第 214 行 / Line 214)**:
   - 中文: 整个 flow matching 的损失就这么朴素——预测速度场和真实速度场之间的 MSE。**没有 KL 散度,没有 ELBO,没有 VLB**。这就是 flow matching 比扩散更简洁的地方:训练目标退化成最直白的回归。
   - English: The entire flow matching loss collapses to one MSE between predicted and target velocity. **No KL divergence, no ELBO, no variational lower bound.** This is the conceptual gift of flow matching over diffusion: training reduces to plain regression.

## 类比 / The analogy

想象你在教一个新手骑自行车。你不会让 ta 一上来就在杂乱的车流里学(那对应均匀采样时间——所有难度等同对待);你也不会让 ta 整天只在原地撑车把(那对应只采样 `t=0`)。你会让 ta **大部分时间练接近终点的稳定骑行**(对应 beta(1.5,1) 偏向接近 `t=0` 的高密度区),少部分时间体验"差点摔倒"的边缘情况(对应 `t` 接近 1)。这就是 `Beta(1.5, 1)` 时间先验在做的事——把训练资源更多分配在"接近成功"的、难度最高的关键区域。

而 `x_t = t * noise + (1-t) * actions` 这条**线性插值**轨迹,就像是教练在你的训练日志上画一条笔直的虚线:"从纯随机蹬车开始,顺着这条直线匀速走 10 步,你就能稳稳骑到目标动作"。Rectified flow 把"如何还原动作"的问题从弯弯曲曲的概率轨迹拉成一条直线,模型要学的不再是"未来怎么走",而是单纯的"方向往哪指"——速度场。

Imagine teaching someone to ride a bike. You wouldn't drop them straight into rush-hour traffic (uniform time sampling — treating all difficulty equal), nor make them just sit in place holding the handlebars (sampling only `t=0`). You'd spend **most of the lesson on the polishing phase — when they're already almost stable** (the beta(1.5,1) prior, concentrated near `t=0` after the flip), with occasional excursions into the "almost falling" failure mode (`t` near 1). That's the time-prior trick.

And the `x_t = t * noise + (1-t) * actions` straight-line trajectory is like the coach drawing a perfectly straight dashed line on your training journal: "Start from random pedaling, walk down this line at constant speed for 10 steps, and you'll arrive at the target motion." Rectified flow straightens the curved probabilistic paths of diffusion into a single ruler-line, and the model only ever needs to predict one thing: the direction. The velocity field. That's it.

## 自己跑一遍 / Try it yourself

下面这个最小 PyTorch 示例把 π₀ 的训练目标剥成 2D 玩具版,让你在自己的笔记本上跑通:

A minimal PyTorch version that strips the π₀ training target down to a 2-D toy you can run on your laptop:

```python
"""Toy rectified flow matching in 2D — same loss as π₀.compute_loss."""
import torch
import torch.nn as nn

class TinyVelocityNet(nn.Module):
    def __init__(self, dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )
    def forward(self, x_t, t):
        return self.net(torch.cat([x_t, t.unsqueeze(-1)], dim=-1))

torch.manual_seed(0)
net = TinyVelocityNet()
opt = torch.optim.Adam(net.parameters(), lr=3e-3)

# "Real actions" = a ring in 2D
def real_actions(b):
    theta = torch.rand(b) * 2 * torch.pi
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

for step in range(2000):
    actions = real_actions(256)
    noise = torch.randn_like(actions)
    # The π₀ trick: beta(1.5, 1) time prior, NOT uniform
    t = torch.distributions.Beta(1.5, 1.0).sample((256,)) * 0.999 + 0.001
    x_t = t[:, None] * noise + (1 - t[:, None]) * actions
    u_t = noise - actions                       # target velocity
    v_t = net(x_t, t)                           # predicted velocity
    loss = ((v_t - u_t) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0:
        print(f"step {step:4d} loss {loss.item():.4f}")

# Sample 5 points by integrating the flow for 10 steps
with torch.no_grad():
    x = torch.randn(5, 2)
    for i in range(10):
        t = torch.full((5,), 1.0 - i / 10)
        x = x - (1 / 10) * net(x, t)
print("sampled points (should sit near the unit circle):", x.tolist())
```

运行 / Run with:
```bash
pip install torch
python try_flow.py
```

预期输出 / Expected output:
```
step    0 loss 1.4xx
step  500 loss 0.0xx
step 1000 loss 0.0xx
step 1500 loss 0.0xx
sampled points (should sit near the unit circle): [[~1, ~0], [~0.7, ~0.7], ...]
```

注意每个采样出来的 `(x, y)` 都接近 `x² + y² ≈ 1`——这就是 flow matching 学会了从噪声生成"动作"(这里动作是 2D 圆上的点)。把 `Beta(1.5, 1)` 改成 `torch.rand(256)`(均匀采样),你会看到训练初期收敛更慢——这就是 π₀ 时间先验的实际收益。

Each sampled `(x, y)` should sit near `x² + y² ≈ 1` — the model has learned to map noise into the ring-shaped "action distribution." If you swap `Beta(1.5, 1)` for plain `torch.rand(256)`, you'll see slower early-training convergence. That gap is exactly the empirical reason π₀ uses the beta prior.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Stable Diffusion 3 / Flux**: 同样用 rectified flow 替代 DDPM,推理步数从 50 降到 4–8。π₀ 把同一套机制从图像生成搬到了机器人动作生成。
  / Stable Diffusion 3 and Flux both replaced DDPM with rectified flow, slashing inference steps from 50 down to 4–8. π₀ simply ported the same machinery from image generation to robot action generation.
- **Pi0-FAST / Pi0.5**: openpi 同仓库里的姊妹模型,把动作头从 flow matching 换成自回归 token 预测(FAST tokenizer),走的是完全不同的路径——对比两份 `compute_loss` 很涨知识。
  / Pi0-FAST and Pi0.5 (sister models in the same openpi repo) swap the flow-matching head for an autoregressive token-prediction head (using the FAST tokenizer). Reading both `compute_loss` files side by side teaches you a lot about the design trade-off.
- **ReinFlow**(今日的 trending 笔记): 把同一套 rectified flow 重新做成模块化 PyTorch 代码,并加上 RL 微调。
  / ReinFlow (today's trending note): re-implements the same rectified flow modularly in PyTorch and adds online RL fine-tuning on top.
- **Min-SNR / Logit-Normal time schedules**: 学界已经探索过多种"非均匀时间采样"——π₀ 选的 beta(1.5,1) 是其中一种,SD3 用的是 logit-normal。它们都属于同一思想:把训练资源往最难的时间区间倾斜。
  / Min-SNR and logit-normal time schedules are alternative non-uniform time priors explored in the literature. π₀ went with beta(1.5,1); SD3 chose logit-normal. Same idea — concentrate gradient on the hardest time region.

## 注意事项 / Caveats / when it breaks

- **`Beta(1.5, 1)` 不是普适最优**: 这个超参是 π₀ 团队针对机械臂动作数据调出来的。换一个数据分布(例如灵巧手、汽车驾驶),最优时间先验可能完全不同。先用 `Uniform(0,1)` 跑通,再调时间先验。
  / `Beta(1.5, 1)` is not universally optimal. The π₀ team tuned this for arm manipulation. For a different action distribution (dexterous hands, driving), the optimal prior can differ. Always train with `Uniform(0,1)` first; tune the prior afterwards.
- **Rectified flow 不等于"直线就一定能走通"**: 当真实动作分布是多模态(同一个图像里抓杯子可以从左也可以从右),线性插值出的 `x_t` 会跨越不同模式之间的"低密度走廊"。模型预测的 `v_t` 在这些区域可能矛盾(指向两个不同模式的平均方向)。解决办法包括多次 reflow 蒸馏(Liu et al., 2023)或者引入条件信息让模型挑一个模式。
  / Rectified flow does *not* guarantee the straight-line path is feasible. When the action distribution is multimodal (you can grasp the cup from the left or the right given the same image), the linear `x_t` interpolation crosses low-density "corridors" between modes. The predicted `v_t` in those regions can be contradictory (the mean of two opposing directions). The fix is reflow distillation (Liu et al., 2023) or richer conditioning that lets the model pick a mode.
- **`x_t` 边界的数值稳定性**: 如果不加 `* 0.999 + 0.001`,在极端的 `t=0` 或 `t=1` 处,`u_t = noise - actions` 仍然是良定义的,但下游的 `posemb_sincos(time, ...)` 时间编码可能在边界处梯度异常。
  / Without the `* 0.999 + 0.001` cutoff, `u_t = noise - actions` is still well-defined at the boundaries, but the downstream `posemb_sincos(time, ...)` time encoding can produce abnormal gradients exactly at `t=0` or `t=1`.
- **整段动作 chunk 一次预测**: π₀ 不是逐 token 自回归地生成动作。这让吞吐快,但也意味着你必须知道 `action_horizon`(比如 50 步)且全部一次预测。对实时反馈控制场景(每收到一个新观测就重规划),这是一个权衡点。
  / π₀ predicts the entire action chunk in one shot, not autoregressively. This gives throughput but means you must fix `action_horizon` (say, 50 steps) and emit it all at once. For high-frequency closed-loop control where you re-plan on every new observation, that's a trade-off to consider.

## 延伸阅读 / Further reading

- π₀ paper: https://www.physicalintelligence.company/blog/pi0
- Flow Matching for Generative Modeling (Lipman et al., 2023): https://arxiv.org/abs/2210.02747
- Rectified Flow (Liu et al., 2023): https://arxiv.org/abs/2209.03003
- openpi 同文件中的 `sample_actions` (推理时的反向 ODE 积分): [pi0.py#L217-L279](https://github.com/Physical-Intelligence/openpi/blob/c23745b5ad24e98f66967ea795a07b2588ed6c79/src/openpi/models/pi0.py#L217-L279)
- `pi0_fast.py` (同一仓库的 autoregressive 变体,与今天的 flow-matching 形成对照): [pi0_fast.py](https://github.com/Physical-Intelligence/openpi/blob/c23745b5ad24e98f66967ea795a07b2588ed6c79/src/openpi/models/pi0_fast.py)
- Stable Diffusion 3 paper, §3.1 (logit-normal time prior, the SD3 cousin of π₀'s beta prior): https://arxiv.org/abs/2403.03206
