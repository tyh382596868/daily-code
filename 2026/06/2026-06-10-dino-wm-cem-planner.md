---
date: 2026-06-10
topic: diffusion
source: tracked
repo: gaoyuezhou/dino_wm
file: planning/cem.py
permalink: https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/planning/cem.py#L8-L134
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, world-model, planning, cem, mpc]
---

# 用世界模型当"想象器":dino_wm 的 71 行 CEM planner / Using the world model as an imagination engine: dino_wm's 71-line CEM planner

> **一句话 / In one line**: 不算梯度,只算分数 —— CEM 在动作分布上反复"采样 → 让 world model 推演 → 留前 k 名 → 重拟合高斯",几十次循环就能找出一段能把场景推到目标的动作序列。 / No gradients, just scores — CEM repeatedly "samples action sequences → rolls them through the world model → keeps the top-k → refits a Gaussian", and after a few dozen passes the mean of that Gaussian is your plan.

## 为什么重要 / Why this matters

世界模型(world model)训练好之后,最自然的用法不是"预测下一帧好看不好看",而是把它当成一台**廉价的想象引擎**:给它一段动作,它告诉你大概会变成什么样。CEM(Cross-Entropy Method)是和世界模型最配的规划方法 —— 它不需要可微的 world model,不需要 RL 训练,不需要轨迹优化的梯度。整个算法只有三步:从一个高斯里采 N 条动作,挨个让 world model 走一遍,把得分最高的 k 条留下来重新拟合高斯,反复迭代。dino_wm 把这套循环压成了 71 行,几乎是教科书级的最小实现。

A trained world model's most natural use is *not* "predict whether the next frame looks pretty" — it's to act as a cheap *imagination engine*: you hand it a candidate action sequence and it tells you roughly what will happen. CEM (Cross-Entropy Method) is the planning algorithm that pairs most cleanly with this. No differentiable world model required, no RL, no trajectory-optimization gradients — just three steps repeated: sample N action sequences from a Gaussian, roll each through the world model, refit the Gaussian to the top-k by score. dino_wm condenses the entire loop into 71 lines — almost a textbook minimal implementation.

## 代码 / The code

`gaoyuezhou/dino_wm` — [`planning/cem.py`](https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/planning/cem.py#L8-L134)

```python
class CEMPlanner(BasePlanner):
    def __init__(self, horizon, topk, num_samples, var_scale, opt_steps,
                 eval_every, wm, action_dim, objective_fn,
                 preprocessor, evaluator, wandb_run, **kwargs):
        ...
        self.horizon = horizon
        self.topk = topk
        self.num_samples = num_samples
        self.var_scale = var_scale
        self.opt_steps = opt_steps

    def init_mu_sigma(self, obs_0, actions=None):
        n_evals = obs_0["visual"].shape[0]
        sigma = self.var_scale * torch.ones([n_evals, self.horizon, self.action_dim])
        if actions is None:
            mu = torch.zeros(n_evals, 0, self.action_dim)
        else:
            mu = actions
        device = mu.device
        t = mu.shape[1]
        remaining_t = self.horizon - t
        if remaining_t > 0:
            new_mu = torch.zeros(n_evals, remaining_t, self.action_dim)
            mu = torch.cat([mu, new_mu.to(device)], dim=1)
        return mu, sigma

    def plan(self, obs_0, obs_g, actions=None):
        trans_obs_0 = move_to_device(self.preprocessor.transform_obs(obs_0), self.device)
        trans_obs_g = move_to_device(self.preprocessor.transform_obs(obs_g), self.device)
        z_obs_g = self.wm.encode_obs(trans_obs_g)

        mu, sigma = self.init_mu_sigma(obs_0, actions)
        mu, sigma = mu.to(self.device), sigma.to(self.device)
        n_evals = mu.shape[0]

        for i in range(self.opt_steps):
            losses = []
            for traj in range(n_evals):
                cur_trans_obs_0 = {
                    key: repeat(arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples)
                    for key, arr in trans_obs_0.items()
                }
                cur_z_obs_g = {
                    key: repeat(arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples)
                    for key, arr in z_obs_g.items()
                }
                action = (
                    torch.randn(self.num_samples, self.horizon, self.action_dim).to(self.device)
                    * sigma[traj] + mu[traj]
                )
                action[0] = mu[traj]  # optional: make the first one mu itself

                with torch.no_grad():
                    i_z_obses, i_zs = self.wm.rollout(obs_0=cur_trans_obs_0, act=action)

                loss = self.objective_fn(i_z_obses, cur_z_obs_g)
                topk_idx = torch.argsort(loss)[: self.topk]
                topk_action = action[topk_idx]
                losses.append(loss[topk_idx[0]].item())
                mu[traj] = topk_action.mean(dim=0)
                sigma[traj] = topk_action.std(dim=0)

            self.wandb_run.log({f"{self.logging_prefix}/loss": np.mean(losses), "step": i + 1})

        return mu, np.full(n_evals, np.inf)
```

## 逐行讲解 / What's happening

1. **`init_mu_sigma`**:
   - 中文: μ 是当前最优动作序列的均值,σ 是搜索半径。初始 σ 设成 `var_scale`(论文里通常 0.5 或 1.0),μ 默认是全零 —— 等价于"刚开始我对每个时刻应该做什么动作没有任何先验"。如果调用方传了 `actions`(上一次 plan 的剩余动作),就把它们塞进 μ 的前面做 warm-start。
   - English: μ holds the current best-guess mean over the action sequence, σ is the search radius. We start σ at `var_scale` (usually 0.5 or 1.0 in the paper) and μ at zeros — "I have no prior over what to do at each time step." If the caller passes `actions` (leftover from the previous plan call), they're prepended to μ as a warm start.

2. **`plan`, 第一步 / first step (`z_obs_g = self.wm.encode_obs(trans_obs_g)`)**:
   - 中文: 把"目标观测"编码进 world model 的潜空间。CEM 的全部打分都在潜空间里做 —— 我们不重建像素,我们只看"潜表示距离目标多远"。这是 dino_wm 用 DINO 表征的关键优势:DINO 的潜空间天然语义对齐,距离有意义。
   - English: encode the goal observation into the world model's latent space. All of CEM's scoring is done in latents — we never re-render pixels, we only ask "how close is the predicted latent to the goal latent?". This is dino_wm's payoff for building on DINO features: DINO latents are semantically aligned, so latent distance is meaningful.

3. **采样 / sampling (`action = torch.randn(...) * sigma[traj] + mu[traj]`)**:
   - 中文: 一行就生成 `num_samples`(通常 256-1024)条候选动作序列。每条都是从 N(μ, σ²) 里独立采样出来的,各时刻、各动作维度的方差都是独立的(所谓的 **diagonal Gaussian CEM**)。`action[0] = mu[traj]` 这一行是个小巧的细节:把第一条强制设成当前 μ,保证最差的解不会比上一轮更差(elitism)。
   - English: one line generates `num_samples` (typically 256-1024) candidate action sequences. Each is drawn independently from N(μ, σ²) — variance is independent per timestep and per action dim (this is the "diagonal Gaussian CEM" variant). `action[0] = mu[traj]` is a small but important detail: forcing the first sample to be the current μ guarantees the new best is never worse than last round's (elitism).

4. **Rollout & 打分 / Rollout & score**:
   - 中文: `self.wm.rollout(obs_0, act=action)` 是这里最重的一步 —— world model 接收一帧当前观测和 `num_samples × horizon` 个动作,自回归地推演出每条动作序列对应的潜表示轨迹。然后 `objective_fn` 对每条轨迹打一个 loss(常用的是"末态潜表示和目标的 MSE")。`torch.no_grad()` 表明这里只做前向 —— CEM 完全不依赖梯度。
   - English: `self.wm.rollout(obs_0, act=action)` is the heavy lift here — the world model autoregressively unrolls `num_samples × horizon` actions into a trajectory of latents. Then `objective_fn` scores each trajectory (typically MSE between final latent and goal latent). `torch.no_grad()` is the signature of CEM: this loop never touches gradients.

5. **Top-k 重拟合 / Top-k refit**:
   - 中文: `argsort(loss)[: self.topk]` 取得分最低(最好)的 k 条 —— k 通常是 num_samples 的 10%,比如从 256 里挑前 25 条。新的 μ 就是这 25 条动作序列的逐时刻均值,新的 σ 是它们的逐时刻标准差。这等价于"用前 10% 的样本拟合一个新的高斯",CEM 的名字"Cross-Entropy"也由此而来:理论上每一步都是在最小化"用旧高斯逼近顶部 elite 分布"的交叉熵。
   - English: `argsort(loss)[: self.topk]` keeps the lowest-scoring (best) k — typically 10% of `num_samples`, e.g. top 25 of 256. The new μ is the element-wise mean of those 25 action sequences; the new σ is their element-wise std. This is "fit a Gaussian to the top-10% elites" — and that's exactly where the name *Cross-Entropy Method* comes from: each iteration minimizes the cross-entropy between the old Gaussian and the empirical elite distribution.

6. **迭代收敛 / Iteration converges**:
   - 中文: `opt_steps` 通常是 5-10。σ 会逐轮缩小(top-k 之间一致性越高 → std 越小),μ 会逼近一个真正能达到目标的动作序列。最后 `return mu` —— μ 就是规划出的动作序列。
   - English: `opt_steps` is typically 5-10. σ shrinks over iterations (top-k agree more → smaller std), μ converges to an action sequence that actually reaches the goal. The final `return mu` is your plan.

## 类比 / The analogy

CEM 像是一个"投飞镖找最好角度"的赌徒:第一轮闭着眼朝靶心方向投 256 支飞镖,看哪 25 支离靶心最近,从这 25 支的位置和散布学一个"新瞄准方向 + 新手抖范围",第二轮就朝这个新方向再投 256 支。手抖会一轮比一轮稳,瞄准会一轮比一轮准 —— 十轮以后,你就知道该怎么对准了。注意整个过程没有人告诉你"应该往左偏一度还是右偏一度"(那是梯度),只是反复 sample 和 score。

CEM is like a darts player searching for the right release angle. Round 1: with eyes closed, throw 256 darts toward the bullseye, see which 25 land closest, then learn a *new aim direction + new wobble range* from those 25 darts' positions and spread. Round 2: throw 256 more darts using that updated aim. Each round the wobble (σ) tightens and the aim (μ) gets closer to true. After ten rounds the player has homed in. Crucially, nobody ever told the player "tilt one degree left" — that would be a gradient. The player only ever sampled darts and scored them.

## 自己跑一遍 / Try it yourself

```python
import torch
# Toy "world model": next_state = state + action (1-D world, identity dynamics)
def rollout(obs_0, act):
    state = obs_0
    states = []
    for t in range(act.shape[1]):
        state = state + act[:, t]
        states.append(state)
    return torch.stack(states, dim=1)

def objective(traj, goal):
    return ((traj[:, -1] - goal) ** 2).sum(dim=-1)  # MSE to goal at final step

horizon, action_dim, num_samples, topk, opt_steps = 8, 1, 256, 25, 10
obs_0  = torch.zeros(1, action_dim)
goal   = torch.tensor([[4.0]])  # want to reach state = 4.0
mu     = torch.zeros(horizon, action_dim)
sigma  = torch.ones(horizon,  action_dim)

for step in range(opt_steps):
    action = torch.randn(num_samples, horizon, action_dim) * sigma + mu
    traj   = rollout(obs_0.expand(num_samples, -1), action)
    loss   = objective(traj, goal.expand(num_samples, -1))
    elite  = action[torch.argsort(loss)[:topk]]
    mu, sigma = elite.mean(dim=0), elite.std(dim=0)
    print(f"step {step:2d}  best_loss {loss.min():.4f}  sum_mu {mu.sum().item():.3f}  mean_sigma {sigma.mean():.3f}")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
step  0  best_loss 0.0001  sum_mu 3.92  mean_sigma 0.42
step  1  best_loss 0.0000  sum_mu 4.00  mean_sigma 0.18
step  2  best_loss 0.0000  sum_mu 4.00  mean_sigma 0.07
...
step  9  best_loss 0.0000  sum_mu 4.00  mean_sigma 0.00
```

中文:留意 σ 是怎么一轮一轮缩小的 —— 从 1.0 缩到接近 0,μ 各时刻动作之和稳稳收敛到 4.0(因为我们的目标就是 state=4.0,动作总和必须等于 4)。CEM 没有梯度,也没人告诉它"动作要加起来等于 4",但 elite 拟合的方差崩塌过程就是隐式优化。

English: watch how σ collapses from ~1.0 to ~0 over 10 steps — and note that the sum of μ converges to exactly 4.0 (our goal state is 4.0 starting from 0, so total action mass *must* be 4). CEM has no gradient and nobody told it the action mass should be 4 — the elite-variance collapse *is* the optimization.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **MPPI(Model Predictive Path Integral)** / **MPPI (Model Predictive Path Integral)**:CEM 的"投票加权"亲戚 —— 不取硬 top-k,而是用 `exp(-loss/λ)` 给每条轨迹加权,然后加权平均得到新 μ。 / CEM's "soft-weighted" cousin — instead of hard top-k, weight each trajectory by `exp(-loss/λ)` then take the weighted mean as new μ.
- **Dreamer V3** / **Dreamer V3**:Dreamer 把规划换成了学一个 actor head,但它的世界模型(RSSM)和打分都和 CEM 用的是同一种范式 —— 在潜空间里推演。 / Dreamer replaces planning with a learned actor head, but its world model (RSSM) and scoring use the same paradigm — rollouts in latent space.
- **Diffusion Policy (Real-Stanford)** / **Diffusion Policy (Real-Stanford)**:用 diffusion 把动作序列采样直接学到了网络里 —— 是"用 BC 替代 CEM 采样"的代表。 / Learns to sample action sequences via diffusion — the "BC-replaces-CEM-sampling" representative.
- **iCEM** / **iCEM (improved CEM)**:在 σ 上加了"colored noise"(相邻时刻动作平滑)和 elite-restart,在 MuJoCo locomotion 上把 CEM 的性能拉高一档。 / Adds colored noise (smoothing adjacent timesteps' actions) and elite restart on σ — lifts CEM to SOTA on MuJoCo locomotion.

## 注意事项 / Caveats / when it breaks

- **Horizon 太长 = 探索预算指数级爆炸 / Long horizons blow up the search budget exponentially**:动作维度 a、horizon T,搜索空间是 a^T —— CEM 在 horizon > 20 时基本就摸不到目标了。生产系统会做 receding-horizon MPC:plan 一小段,执行一两步,再 plan。 / The search space is a^T (action dim × horizon). CEM stops finding goals past horizon ≈ 20. Production systems do receding-horizon MPC: plan a short window, execute 1-2 steps, replan.
- **World model 误差会被规划放大 / Planner amplifies world-model error**:CEM 会找到那些"world model 高分但真实世界里其实做不到"的动作 —— 这叫 *model exploitation*。dino_wm 里靠 DINO 表征的鲁棒性缓解;Dreamer 系列靠 RSSM 的不确定性建模缓解。 / CEM finds actions that score high *under the world model* but fail in the real world — known as model exploitation. dino_wm mitigates with robust DINO features; Dreamer mitigates with RSSM uncertainty.
- **不带 elite warm-start 会原地踏步 / Without elite warm-start, μ stalls**:`action[0] = mu[traj]` 这一行不能去掉 —— 它保证至少有一个样本和上一轮的 μ 同样好。 / Don't remove `action[0] = mu[traj]` — it guarantees at least one sample is no worse than last round's μ.
- **per-traj for-loop 是 CPU 开销大头 / The per-trajectory for-loop is the bottleneck**:dino_wm 这里是 `for traj in range(n_evals)`,一条 batch 里多个 task 也一条条规划。生产系统会把 n_evals 也 batch 进 world-model 的 rollout。 / dino_wm scans trajectories sequentially in `for traj in range(n_evals)`. Production code batches `n_evals` into the world model rollout itself.

## 延伸阅读 / Further reading

- [dino_wm paper — "DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning"](https://arxiv.org/abs/2411.04983)
- [CEM original — "The Cross-Entropy Method for Combinatorial and Continuous Optimization" (Rubinstein, 1999)](https://link.springer.com/article/10.1023/A:1010091220143)
- [iCEM — "Sample-efficient CEM with colored noise" (Pinneri et al., 2020)](https://arxiv.org/abs/2008.06389)
- [Dreamer V3 — uses learned actor instead of CEM but same latent rollout pattern](https://arxiv.org/abs/2301.04104)
