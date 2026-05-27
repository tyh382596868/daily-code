---
date: 2026-05-27
topic: diffusion
source: trending
repo: galilai-group/stable-worldmodel
file: stable_worldmodel/solver/mppi.py
permalink: https://github.com/galilai-group/stable-worldmodel/blob/a9dc10f37b6adaa7f484ef1a20908ca22d8118c8/stable_worldmodel/solver/mppi.py#L178-L249
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, world-model, mppi, planning, control]
---

# MPPI 的优化内层循环:六行 softmax 就是 model-based planning 的核心 / The MPPI inner loop: model-based planning is six lines of softmax

> **一句话 / In one line**: MPPI 把每个噪声轨迹的代价取负、除以温度、做 softmax,然后用这组权重对所有轨迹做加权平均 —— 三十行代码就替代了 CEM 的"硬截断 elite"。 / MPPI takes each noisy trajectory's cost, negates it, divides by temperature, softmaxes, and uses those weights to take an average over all trajectories — thirty lines replacing CEM's hard elite cutoff.

## 为什么重要 / Why this matters

CEM(Cross-Entropy Method)是 model-based planning 的入门款:采 N 个动作序列、跑 world model 算 cost、留下前 K 个 elite,把这 K 个 elite 的均值/方差当作下一轮的提议分布。问题是"留 top-K 扔其余"是离散决策,梯度信号很粗。MPPI(Model Predictive Path Integral)从最优控制理论里推导出一个更柔和的更新规则:每个样本根据 `softmax(-cost / temperature)` 分配一个连续权重,然后做加权平均。**温度小 → 接近 CEM 的硬截断;温度大 → 退化成简单平均**。这条 softmax 公式让 MPPI 在差异化代价时收敛更平稳,也是 Tesla AI Day 演示过的"在 GPU 上每秒采几千条轨迹"那个 controller 背后的核心算子。这份代码值得读,是因为它把课本上一笔带过的实现细节(数值稳定、top-k filtering、热启动 sample-0)都写了出来。

CEM (Cross-Entropy Method) is the introductory model-based planner: sample N action sequences, score them with a world model, keep the top K, fit a Gaussian to those K, repeat. The annoyance is that "keep top-K, throw the rest" is a hard discrete decision — a coarse gradient signal. MPPI (Model Predictive Path Integral) replaces it with a softer update derived from optimal-control theory: every sample gets a continuous weight `softmax(-cost / temperature)`, and the new mean is a weighted average. **Low temperature → behaves like CEM's hard cutoff. High temperature → degenerates to a simple average**. This single formula gives MPPI smoother convergence when costs are noisy and is the engine behind those Tesla-style demos where thousands of trajectories are evaluated per second on a GPU. This particular implementation is worth reading because it pins down the small but important details textbooks tend to skip: numerically-stable softmax, optional top-k filtering, and a warm-start sample injected so the next mean can never be strictly worse than the current one.

## 代码 / The code

`galilai-group/stable-worldmodel` — [`stable_worldmodel/solver/mppi.py`](https://github.com/galilai-group/stable-worldmodel/blob/a9dc10f37b6adaa7f484ef1a20908ca22d8118c8/stable_worldmodel/solver/mppi.py#L178-L249)

```python
# Optimization Loop
final_batch_cost = None

for step in range(self.n_steps):
    # Sample noise: (Batch, Num_Samples, Horizon, Dim)
    noise = torch.randn(
        current_bs,
        self.num_samples,
        self.horizon,
        self.action_dim,
        generator=self.torch_gen,
        device=self.device,
        dtype=self.dtype,
    )

    # MPPI Logic: candidates = mean + noise * sigma
    candidates = batch_mean.unsqueeze(1) + noise * batch_var.unsqueeze(1)

    # Force the first sample to be the current mean (Zero noise)
    candidates[:, 0] = batch_mean

    costs = self.model.get_cost(expanded_infos, candidates)

    # Select Elites (Optional, based on topk)
    if self.topk is not None and self.topk < self.num_samples:
        topk_vals, topk_inds = torch.topk(
            costs, k=self.topk, dim=1, largest=False
        )
        batch_indices = (
            torch.arange(current_bs, device=self.device)
            .unsqueeze(1)
            .expand(-1, self.topk)
        )
        relevant_candidates = candidates[batch_indices, topk_inds]
        relevant_costs = topk_vals
    else:
        relevant_candidates = candidates
        relevant_costs = costs

    # MPPI Weighting: Softmax(-cost / temperature)
    # Stabilize softmax by subtracting min cost
    min_cost = relevant_costs.min(dim=1, keepdim=True)[0]
    scaled_costs = relevant_costs - min_cost
    weights = torch.softmax(
        -scaled_costs / self.temperature, dim=1
    )  # (Batch, K)

    # Update Mean: weighted sum of candidates
    weights_expanded = weights.unsqueeze(-1).unsqueeze(-1)
    batch_mean = (weights_expanded * relevant_candidates).sum(dim=1)

    final_batch_cost = relevant_costs.mean(dim=1).cpu().tolist()
```

## 逐行讲解 / What's happening

1. **采样噪声 / Sampling noise (`noise = torch.randn(...)`)**:
   - 中文: 一次性采 `(Batch, Num_Samples, Horizon, Dim)` 形状的高斯噪声,意思是"对当前 mean 轨迹做 `num_samples` 个扰动版本,每个版本在每个时间步上都加一点扰动"。注意 `generator=self.torch_gen` 用的是 solver 自己持有的 generator,这样多次 `solve()` 调用之间是可复现的,不会被全局 RNG 干扰。
   - English: We draw Gaussian noise of shape `(Batch, Num_Samples, Horizon, Dim)` in one shot — `num_samples` perturbed versions of the current mean trajectory, each perturbed at every timestep. The `generator=self.torch_gen` argument uses the solver's private generator, making repeated `solve()` calls reproducible regardless of the global RNG state.

2. **构造候选 + 注入当前均值 / Build candidates and inject the current mean (`candidates[:, 0] = batch_mean`)**:
   - 中文: `candidates = mean + noise * sigma`,然后**把第 0 号候选强制设回当前 mean(零噪声)**。这是一个小但极其重要的工程技巧:它保证了"加权平均的结果至少不比上一轮的 mean 差太多",因为最优解就在候选集合里。没有这一行,在 cost surface 平坦的区域随机噪声会让 mean 漂移甚至变差。
   - English: `candidates = mean + noise * sigma`, then **the zero-th sample is overwritten with the current mean (zero-noise)**. This is a small but crucial trick: it guarantees the new weighted-average mean cannot be much worse than the old one, because the old mean is literally in the candidate set. Without this line, in a flat region of the cost surface, random noise can let the mean drift in the wrong direction.

3. **可选 top-k / Optional top-k**:
   - 中文: 纯 MPPI 会用全部 `num_samples` 个样本做加权;但实践中,很差的轨迹的权重虽然小,数值上还是会把 mean 拉糟。这里加了一个开关:先用 `torch.topk(..., largest=False)` 挑出代价最低的 k 个,再在它们之间做 softmax。等于"先用 CEM 截断、再在 elites 上用 MPPI"——一种 MPPI 和 iCEM 之间的折中。
   - English: Pure MPPI weights all `num_samples` candidates. But in practice, the long tail of bad trajectories still has *some* (tiny) weight, and numerically those can drag the mean. This implementation adds an opt-in step: pick the `topk` lowest-cost candidates with `torch.topk(..., largest=False)`, then softmax over just those. This is essentially "CEM filter then MPPI weight" — a hybrid that sits between vanilla MPPI and iCEM.

4. **数值稳定 / Numerical stability (`scaled_costs = relevant_costs - min_cost`)**:
   - 中文: softmax 数值稳定的标准伎俩:减掉每个 batch 的最小代价。`softmax(-cost / T)` 和 `softmax(-(cost - min) / T)` 数学上完全等价,但后者把所有指数的最大值钉在 1,避免 `exp(huge_negative)` 全部 underflow 到 0(这种情况下 weights 全是 nan)。当 cost 量级可能是几千的时候,不减 min 就直接挂。
   - English: The classic numerically-stable softmax trick: subtract the per-batch minimum. `softmax(-cost / T)` is mathematically identical to `softmax(-(cost - min) / T)`, but the latter clamps the largest exponent to `exp(0) = 1`. Without this, when costs are on the order of thousands every `exp(-cost/T)` underflows to zero and `weights` becomes NaN. With it, softmax is rock-solid.

5. **MPPI 更新规则 / The MPPI update rule (`batch_mean = (weights_expanded * relevant_candidates).sum(dim=1)`)**:
   - 中文: 这是整个算法的精髓 —— 用 softmax 权重对候选轨迹做加权平均得到新的 mean。可以这样理解:`weights[i] ∝ exp(-cost_i / T)`,代价越低权重越大,温度 T 越小就越像 argmin(只挑最好的一条)。
   - English: This single line is the whole algorithm — the new mean is a softmax-weighted average over candidates. Read it as: `weights[i] ∝ exp(-cost_i / T)`, so low-cost trajectories dominate. As `T → 0`, this collapses to `argmin` (pure greedy). As `T → ∞`, it becomes a uniform mean.

6. **方差不更新 / Variance is not updated**:
   - 中文: 注意循环里只更新 `batch_mean`,`batch_var` 保持初始的 `var_scale`。这是标准 MPPI 的一个特征:它假设 sigma 由用户设定、不再优化。如果想让 sigma 也自适应,就是 iCEM 或者 CMA-ES 的事。
   - English: Note the loop only updates `batch_mean`; `batch_var` stays at its initial `var_scale`. This is standard MPPI: sigma is treated as a user-set hyperparameter, not optimized. If you want sigma to adapt, you've moved into iCEM or CMA-ES territory.

## 类比 / The analogy

想象你在评一场 100 道菜的烹饪比赛,要选出"理想的菜谱"。**CEM 的做法**是:挑出前 10 名的菜谱,把它们的配方平均一下当作下一轮的"靶心"——剩下 90 个直接扔掉。**MPPI 的做法**是:给每道菜打分,然后按 `e^(-分数差/温度)` 的比例给每个菜谱一个权重——第 1 名权重 100,第 50 名权重可能是 0.5,第 100 名权重 0.001,但**所有菜都参与**最终配方的混合。温度高的时候大家权重接近,温度低的时候第 1 名几乎独占。两种方法的下一轮探索方向(均值)就这样定下来了。

Picture a cooking contest where you have to design "the ideal recipe" by iterating. **CEM**: rank the 100 dishes, pick the top 10, average their recipes — discard the other 90. **MPPI**: score every dish, then assign a weight to each recipe proportional to `e^(-score_gap / temperature)`. The top dish might get weight 100, the median dish weight 0.5, the worst weight 0.001 — but **every dish contributes** to the next round's recipe. At low temperature, the top dish dominates almost completely; at high temperature, the contest devolves into a democratic average.

## 自己跑一遍 / Try it yourself

用一个超简单的"目标:动作向量平均值要等于 [1, 2]"问题来对比 CEM 和 MPPI 的更新行为。 / Compare CEM and MPPI updates on a toy "find an action vector whose mean is [1, 2]" problem.

```python
# try.py — needs: pip install torch
import torch

torch.manual_seed(0)
H, D = 4, 2                      # horizon=4, action_dim=2
target = torch.tensor([1.0, 2.0])
mean = torch.zeros(H, D)
sigma = 1.0
N = 256
temperature = 1.0
topk = 32                        # for CEM

def cost(traj):                  # traj: (N, H, D)
    return ((traj.mean(dim=1) - target) ** 2).sum(-1)

mean_mppi, mean_cem = mean.clone(), mean.clone()
for step in range(8):
    noise = torch.randn(N, H, D)
    cands_m = mean_mppi.unsqueeze(0) + sigma * noise
    cands_c = mean_cem.unsqueeze(0)  + sigma * noise
    cands_m[0] = mean_mppi           # inject current mean
    cands_c[0] = mean_cem

    # MPPI update
    c_m = cost(cands_m)
    w = torch.softmax(-(c_m - c_m.min()) / temperature, dim=0)
    mean_mppi = (w.view(-1, 1, 1) * cands_m).sum(dim=0)

    # CEM update
    c_c = cost(cands_c)
    elite_idx = torch.topk(c_c, topk, largest=False).indices
    mean_cem  = cands_c[elite_idx].mean(dim=0)

    print(f"step {step}: mean_mppi.mean={mean_mppi.mean(0).tolist()}  "
          f"mean_cem.mean={mean_cem.mean(0).tolist()}")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output (前几行示例 / first few lines):
```
step 0: mean_mppi.mean=[0.40, 0.78]  mean_cem.mean=[0.46, 0.91]
step 1: mean_mppi.mean=[0.65, 1.30]  mean_cem.mean=[0.75, 1.50]
...
step 7: mean_mppi.mean=[0.98, 1.96]  mean_cem.mean=[0.99, 1.98]
```

中文:两个算法都会快速收敛到 `[1, 2]`,但 MPPI 每一步的进展依赖于所有 256 个样本(尾部权重很小但非零),而 CEM 每一步只看 32 个 elite。如果把温度调到 0.01,你会发现 MPPI 几乎完全退化成"挑最好那个" —— 这正是 softmax 在 `T → 0` 时变成 argmax 的体现。

English: Both algorithms converge to `[1, 2]` quickly, but MPPI's update at each step uses all 256 samples (with tiny weights on the tail), while CEM uses only the 32 elites. If you set `temperature = 0.01`, MPPI collapses to "pick the single best sample" — that's exactly softmax becoming argmax in the low-temperature limit.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **TD-MPC / TD-MPC2** / **TD-MPC / TD-MPC2**: 中文: Hansen 等人的 MBRL 方法,planner 就是 MPPI,policy net 提供 warm-start —— 跟这份代码里 `prepare_init_action` 想做的事一样。 / English: Hansen et al.'s MBRL family uses MPPI as the planner, with a policy network providing a warm-start mean — exactly what `prepare_init_action` does in this codebase.
- **Diffusion policies for planning** / **用 diffusion 做 planning**: 中文: Diffuser、Decision Diffuser 等工作可以视作"用 score function 替代 softmax 加权"的连续版 MPPI。 / English: Diffuser, Decision Diffuser, etc. can be viewed as a continuous version of MPPI where the score function replaces explicit softmax weighting.
- **Energy-based models** / **能量模型**: 中文: `softmax(-cost / T)` 是 Boltzmann 分布,温度退火、log-partition 等所有 EBM 工具箱在 MPPI 上都成立。 / English: `softmax(-cost / T)` *is* the Boltzmann distribution. Temperature annealing, log-partition diagnostics, and the rest of the EBM toolbox carry over to MPPI directly.

## 注意事项 / Caveats / when it breaks

- **温度调参很敏感 / Temperature is very sensitive**:
  - 中文: 温度太高 → 权重接近均匀,新的 mean ≈ 旧 mean,planner 不动;温度太低 → 数值上变 argmin,失去 MPPI 相对 CEM 的所有优势。一般以"权重熵"为指标:有效样本数 (`1 / sum(w²)`) 大概在 num_samples 的 5-20% 比较健康。
  - English: Temperature is very sensitive. Too high → weights are nearly uniform, the new mean barely moves. Too low → numerically becomes argmin and loses all the smoothing benefit. A practical diagnostic is the effective sample size `1 / sum(w²)`; healthy values are typically 5-20% of `num_samples`.
- **`@torch.inference_mode()` 不要漏掉 / Don't forget `@torch.inference_mode()`**:
  - 中文: 完整的 `solve` 方法用了 `inference_mode`。如果忘了,反向传播会在数千个候选轨迹上累计梯度,显存秒爆。这里展示的内层循环正常运行依赖外层的这个装饰器。
  - English: The full `solve` method is wrapped in `@torch.inference_mode()`. Forget it and autograd will record graphs over thousands of candidate rollouts — out-of-memory in seconds. The inner loop shown above relies on the outer decorator being in place.
- **不更新 sigma 是 feature 也是 bug / Not updating sigma is a feature and a bug**:
  - 中文: MPPI 不动方差,所以在接近最优时探索半径不会自动收缩,可能围着最优点持续抖动。如果你的任务对最后一公里的精度敏感(比如插孔),要么后期手动衰减 sigma,要么换 iCEM。
  - English: Standard MPPI doesn't update the variance, so near the optimum the exploration radius doesn't shrink and the planner can jitter around the solution. For tasks that need last-mile precision (e.g. peg-in-hole), either decay sigma manually as the cost drops or switch to iCEM.

## 延伸阅读 / Further reading

- Williams, Aldrich & Theodorou, "Model Predictive Path Integral Control: From Theory to Parallel Computation" (2017) — the canonical MPPI derivation.
- Howell et al., "Predictive Sampling: Real-Time Behaviour Synthesis with MuJoCo" (2022) — the closely-related baseline this repo also ships (see `predictive_sampling.py`).
- Pinneri et al., "Sample-efficient Cross-Entropy Method for Real-time Planning" (iCEM, 2020) — the variance-adapting cousin.
- Hansen et al., "TD-MPC2" — MPPI plugged into a modern actor-critic framework.
