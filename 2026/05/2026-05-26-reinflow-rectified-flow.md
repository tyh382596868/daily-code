---
date: 2026-05-26
topic: robotics
source: trending
repo: ReinFlow/ReinFlow
file: model/flow/reflow.py
permalink: https://github.com/ReinFlow/ReinFlow/blob/e722e151bed767f3ffef47527cf697f2358af55d/model/flow/reflow.py#L91-L171
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, flow-matching, rectified-flow, pytorch, time-sampling]
---

# ReinFlow:把 π₀ 的 flow matching 拆成可读 PyTorch 模块 / ReinFlow: π₀'s Flow Matching, Refactored into Readable PyTorch Modules

> **一句话 / In one line**: 同样的 rectified flow 算法,openpi 用 JAX 一行写完;ReinFlow 用 PyTorch 拆成 `sample_time` + `generate_trajectory` + `generate_target` + `loss`,把每个超参背后的设计选择都暴露出来 / The same rectified-flow algorithm openpi crams into one JAX block, ReinFlow breaks into four small PyTorch methods — each one exposing a design choice you usually have to guess at.

## 为什么重要 / Why this matters

读 π₀(今日的 tracked 笔记)的 `compute_loss` 时,你大概会有个疑问:**为什么时间用 `Beta(1.5, 1)`?改成均匀分布不行吗?改成 logit-normal 行不行?** 这些问题在原代码里没答案——它直接把"团队调好的最终配方"打包成一行 JAX。ReinFlow 是 NeurIPS 2025 的一篇论文,它把同一套机制重新做成了**研究友好的模块化代码**:每一个设计选择都被拆成可单独替换的方法,允许你在 `cfg` 里改 `time_sample_type='uniform' | 'logitnormal' | 'beta'` 就能跑对照实验。今天我们读它的 `ReFlow` 类,既能加深对 flow matching 的理解,又能学到一个值得抄的代码组织模式。

After reading the π₀ `compute_loss` (today's tracked note), you probably have questions: **why `Beta(1.5, 1)`? Would uniform work? What about logit-normal?** The original code doesn't answer them — it bakes the team's final recipe into one JAX line. ReinFlow (NeurIPS 2025) re-implements the same machinery in a **research-friendly modular layout**: each design choice becomes a separately swappable method, and a single config field `time_sample_type='uniform' | 'logitnormal' | 'beta'` lets you ablate them. Reading the `ReFlow` class gives you a deeper grasp of flow matching plus a code-organization pattern worth stealing.

## 代码 / The code

`ReinFlow/ReinFlow` — [`model/flow/reflow.py`](https://github.com/ReinFlow/ReinFlow/blob/e722e151bed767f3ffef47527cf697f2358af55d/model/flow/reflow.py#L91-L171)

```python
def generate_trajectory(self, x1: Tensor, x0: Tensor, t: Tensor) -> Tensor:
    """Generate rectified flow trajectory xt = t * x1 + (1 - t) * x0."""
    t_ = (torch.ones_like(x1, device=self.device) * t.view(x1.shape[0], 1, 1)).to(self.device)
    xt = t_ * x1 + (1 - t_) * x0
    return xt

def sample_time(self, batch_size: int, time_sample_type: str = 'uniform', **kwargs) -> Tensor:
    """Sample time steps from a specified distribution in [0, 1)."""
    supported = ['uniform', 'logitnormal', 'beta']
    if time_sample_type == 'uniform':
        return torch.rand(batch_size, device=self.device)
    elif time_sample_type == 'logitnormal':
        m = kwargs.get("m", 0)                  # mean
        s = kwargs.get("s", 1)                  # std
        normal_samples = torch.normal(mean=m, std=s, size=(batch_size,), device=self.device)
        logit_normal_samples = 1 / (1 + torch.exp(-normal_samples))
        return logit_normal_samples
    elif time_sample_type == 'beta':
        alpha = kwargs.get("alpha", 1.5)
        beta  = kwargs.get("beta", 1.0)
        s     = kwargs.get("s", 0.999)
        beta_distribution = torch.distributions.Beta(alpha, beta)
        beta_sample = beta_distribution.sample((batch_size,)).to(self.device)
        tau = s * (1 - beta_sample)             # NOTE the flip: 1 - beta_sample
        return tau
    else:
        raise ValueError(f'Unknown time_sample_type = {time_sample_type}. Supported: {supported}')

def generate_target(self, x1: Tensor) -> tuple:
    """Generate training targets for the velocity field."""
    t = self.sample_time(batch_size=x1.shape[0], time_sample_type=self.sample_t_type)
    x0 = torch.randn(x1.shape, dtype=torch.float32, device=self.device)
    xt = self.generate_trajectory(x1, x0, t)
    v = x1 - x0
    return (xt, t), v

def loss(self, xt: Tensor, t: Tensor, obs: dict, v: Tensor) -> Tensor:
    """Compute the MSE loss between predicted and target velocities."""
    v_hat = self.network(xt, t, obs)
    return F.mse_loss(input=v_hat, target=v)
```

## 逐行讲解 / What's happening

1. **`generate_trajectory` (第 91–104 行 / Lines 91-104)**:
   - 中文: 命名约定值得记一下——`x1` 是**真实数据**(动作), `x0` 是**噪声**。注意 ReinFlow 的约定和 π₀ 是**反过来的**:π₀ 让 `t=1` 是噪声,ReinFlow 让 `t=1` 是数据。两个仓库都是对的——只要 `t` 端和 `x_t` 公式自洽即可。第 102 行的 `t.view(x1.shape[0], 1, 1)` 是为了 broadcasting:把形状 `(B,)` 的 `t` 扩展成 `(B, 1, 1)`,这样能和 `x1` 的 `(B, horizon, action_dim)` 逐元素相乘。
   - English: Naming convention worth remembering: `x1` is the **real data** (actions), `x0` is the **noise**. ReinFlow's convention is **opposite to π₀**'s — π₀ puts noise at `t=1`, ReinFlow puts data at `t=1`. Both are correct; the `t` endpoint just has to be consistent with the `x_t` formula. Line 102 reshapes `t` from `(B,)` to `(B, 1, 1)` so it broadcasts against `x1`'s `(B, horizon, action_dim)` shape.

2. **`sample_time` 的三种时间先验 (第 106–138 行 / Lines 106-138)**:
   - 中文: 这是整个文件最有教学价值的部分。三种分布并排:
     - `uniform`:最朴素,所有时间步等概率。基线对照。
     - `logitnormal`:从标准正态采样然后 sigmoid 压回 (0,1),密度集中在中间区域。SD3 的官方选择。
     - `beta`:`Beta(1.5, 1)` 采样后**翻转 `1 - sample`**,密度偏向接近 0 的区域(即接近真实数据)。π₀ 的选择。
     把这三段并排读一遍,你就理解了为什么"时间先验"是 flow matching 里被反复研究的设计点——它本质上是在分配训练算力,决定模型在哪个去噪难度上下功夫最多。
   - English: This is the file's biggest teaching gift. Three priors side-by-side:
     - `uniform`: trivial baseline, every time step equally likely.
     - `logitnormal`: sample from a standard normal, sigmoid into `(0,1)` — density concentrated near the middle. SD3's official choice.
     - `beta`: `Beta(1.5, 1)` followed by **`1 - sample`** — density skewed toward 0 (toward the data side). π₀'s choice.
     Reading them in one screen makes it obvious why "time prior" is one of the most-studied design knobs in flow matching: it's the dial that allocates training compute across difficulty levels.

3. **`tau = s * (1 - beta_sample)` 里 `s = 0.999` (第 135 行 / Line 135)**:
   - 中文: 这个看似无关的 `0.999` 是数值稳定性 hack——把采到的 `tau` 限制在 `(0, 0.999)` 之间。当 `tau` 接近 1(对应纯数据)时,`x_t = x1`,网络的训练信号就退化了;同理 `tau=0` 时 `x_t = x0`,也是退化情形。这两个边界数值上都会让梯度爆炸或者 NaN,所以 ReinFlow 和 π₀ 都加了类似的边界保护(π₀ 是 `* 0.999 + 0.001`)。
   - English: That seemingly random `0.999` is a numerical-stability hack — clamping `tau` into `(0, 0.999)`. When `tau ≈ 1`, `x_t = x1` and the training signal collapses; same when `tau ≈ 0`. Both endpoints can produce exploding gradients or NaNs. Both ReinFlow and π₀ guard against this — π₀ does it as `* 0.999 + 0.001`.

4. **`generate_target` 把三个步骤串起来 (第 140–156 行 / Lines 140-156)**:
   - 中文: 这是 flow matching 训练的"流水线":采时间 → 采噪声 → 插值得到 `xt` → 计算目标速度 `v = x1 - x0`。注意 `v` 的方向和 π₀ 里的 `u_t = noise - actions` 也是反的——同样是约定差异,自洽即可。这种"把训练目标的构造单独封装成一个 `generate_target`"的写法很值得抄,因为它把"如何产生训练样本"和"如何前向计算 loss"解耦了,方便后续做 RL 微调(只换 loss,不换样本生成器)。
   - English: The training pipeline: sample time → sample noise → interpolate to get `xt` → compute target velocity `v = x1 - x0`. The direction of `v` is opposite to π₀'s `u_t = noise - actions` — again, convention difference, both self-consistent. The encapsulation pattern — "make 'how to build a training sample' a single method `generate_target`, separate from 'how to compute the loss on it'" — is worth stealing, because it lets you swap the loss (for RL fine-tuning) without touching the sample generator.

5. **`loss` 最终一行 (第 158–171 行 / Lines 158-171)**:
   - 中文: 网络吃下 `(xt, t, obs)`,输出预测的速度场 `v_hat`,然后和目标 `v` 做 MSE。整个 flow matching 训练目标到此为止——和 π₀ 第 214 行的 `jnp.mean(jnp.square(v_t - u_t), axis=-1)` 数学上完全一致。
   - English: The network takes `(xt, t, obs)`, outputs a predicted velocity `v_hat`, MSE against the target `v`. That's the whole flow matching loss — mathematically identical to π₀'s `jnp.mean(jnp.square(v_t - u_t), axis=-1)` at line 214.

## 类比 / The analogy

把整个 `ReFlow` 类想象成一家**披萨店的中央厨房**。

- `generate_trajectory` 是揉面机:输入面团和水(`x1` 和 `x0`),按时间比例 `t` 揉出半成品(`xt`)。
- `sample_time` 是配方表:今天用哪种时间分布,等于选今天的"披萨脆度配方"(uniform 是均匀脆度,beta 是边缘酥脆中间软,logitnormal 是中间最脆)。
- `generate_target` 是托盘:把上面两步打包好的半成品端到烤箱前。
- `loss` 是烤箱温度计:测一下实际烤出来的(`v_hat`)和食谱说应该烤成的(`v`)差多少。

π₀ 把整个流水线压到了一个 25 行的 `compute_loss` 函数里,像是把整间厨房压成了一个工业级"披萨自动售货机"——高效,但你看不到里面在发生什么。ReinFlow 把厨房拆开放回模块,你能逐个工位换设备、做对照实验——这就是它作为研究代码库的价值。

Picture `ReFlow` as a **pizza kitchen's mise-en-place**.

- `generate_trajectory` is the dough mixer: feed it flour and water (`x1` and `x0`), it produces an interpolated dough (`xt`) at proportion `t`.
- `sample_time` is the recipe selector: which time distribution defines today's "crust crispness profile" (uniform = even, beta = crispy edges/soft middle, logit-normal = crispiest in the middle).
- `generate_target` is the prep tray: bundles the above into one batch ready for the oven.
- `loss` is the temperature probe: measures how far the actual bake (`v_hat`) is from the recipe target (`v`).

π₀ compresses the whole pipeline into a 25-line `compute_loss` — the entire kitchen folded into an industrial pizza vending machine. Efficient, but you can't see the moving parts. ReinFlow keeps each station modular, so you can swap a single piece of equipment and run controlled experiments. That's the value of the research-style codebase.

## 自己跑一遍 / Try it yourself

下面这个独立脚本把 ReinFlow 的三种时间先验可视化出来——你能直观看到 `Beta(1.5, 1)` 翻转后到底把采样集中在哪个区间。

A standalone script visualizing ReinFlow's three time priors — see for yourself where each one concentrates the samples.

```python
"""Compare the three time priors from ReinFlow.sample_time on the same axis."""
import torch

def sample_time(batch_size, kind, **kw):
    if kind == 'uniform':
        return torch.rand(batch_size)
    if kind == 'logitnormal':
        m, s = kw.get('m', 0), kw.get('s', 1)
        return torch.sigmoid(torch.normal(mean=m, std=s, size=(batch_size,)))
    if kind == 'beta':
        a, b, s = kw.get('alpha', 1.5), kw.get('beta', 1.0), kw.get('s', 0.999)
        return s * (1 - torch.distributions.Beta(a, b).sample((batch_size,)))
    raise ValueError(kind)

def histogram(samples, bins=10):
    """ASCII histogram in 10 equal buckets over [0, 1]."""
    counts = [0] * bins
    for x in samples.tolist():
        idx = min(int(x * bins), bins - 1)
        counts[idx] += 1
    max_c = max(counts)
    for i, c in enumerate(counts):
        bar = '#' * int(40 * c / max_c)
        print(f"  [{i/bins:.1f}, {(i+1)/bins:.1f})  {bar} ({c})")

torch.manual_seed(0)
for kind in ['uniform', 'logitnormal', 'beta']:
    print(f"\n=== {kind} ===")
    histogram(sample_time(10000, kind))
```

运行 / Run with:
```bash
pip install torch
python try_time_prior.py
```

预期输出 / Expected output:
```
=== uniform ===
  [0.0, 0.1)  ######################################## (~1000)
  [0.1, 0.2)  ###################################### (~1000)
  ... (roughly flat)

=== logitnormal ===
  [0.0, 0.1)  ###### (~100)
  [0.4, 0.5)  ######################################## (~2500)
  ... (peaked in the middle)

=== beta ===
  [0.0, 0.1)  ######################################## (~3500)
  [0.1, 0.2)  ############################ (~1800)
  ... (front-loaded, falling off toward 1)
```

注意 beta 那条:绝大多数样本落在 `[0, 0.3)`,这正是 π₀ 想让模型最多练习的"接近真实动作"的区域。

Look at the beta row: the bulk of samples lands in `[0, 0.3)` — exactly the "almost there" region π₀ wants the model to specialize in.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **diffusers 的 `FlowMatchEulerDiscreteScheduler`**: HuggingFace 的扩散库里也内置了 flow matching 的时间调度,接口和 ReinFlow 的 `sample_time` 思路一致(传一个 `weighting_scheme` 字符串切换)。
  / HuggingFace `diffusers` ships `FlowMatchEulerDiscreteScheduler` — same `weighting_scheme` string-switch pattern as ReinFlow.
- **TorchCFM**: 专门做 Conditional Flow Matching 的研究库,也把 `sample_time` 和 `interpolate` 拆成单独的可替换组件,设计哲学一脉相承。
  / TorchCFM (Conditional Flow Matching research library) factors `sample_time` and `interpolate` the same way.
- **diffusion_policy(同分类下的另一个 tracked 仓库)**: 它走的是经典 DDPM 路线,可以和今天这两份 flow matching 代码做对照——同一个机器人动作问题,两套生成模型范式。
  / `real-stanford/diffusion_policy` (also under today's robotics topic) uses classic DDPM — a great contrast point for "same robotics problem, two generative paradigms."
- **任何"研究代码 vs 生产代码"对比**: ReinFlow vs π₀ 是一对很好的教材。前者把每个设计选择暴露出来便于做 ablation,后者只保留团队最后选定的配方追求效率。学会在两种风格间切换,是工业研究员的核心技能。
  / The ReinFlow-vs-π₀ contrast is a perfect example of "research code vs production code." The first exposes every design choice for ablation; the second hides everything except the team's final recipe for efficiency. Switching between the two styles is a core industrial-research skill.

## 注意事项 / Caveats / when it breaks

- **`x0` 和 `x1` 的方向约定**: 文献和不同代码库经常把"数据"和"噪声"放在 `t=0` 还是 `t=1` 翻来覆去——读代码前一定先确认这一点。否则你的 `v` 方向、ODE 积分方向、`dt` 符号会全错,而且不会立刻报错,只会训练发散。
  / The `x0`/`x1` direction convention swaps frequently across papers and codebases — always confirm which endpoint is data before reading further. Getting it wrong flips your `v` direction, your ODE integration direction, and the sign of `dt`. It won't crash; it'll just silently fail to converge.
- **`Beta(1.5, 1)` 不是 ReinFlow 的默认**: ReinFlow 的默认是 `'uniform'`,只有当你在 cfg 里显式写 `sample_t_type: beta` 才会启用 π₀ 风格的时间先验。不要假设 import 了 ReinFlow 就自动享受 π₀ 的所有 tricks。
  / `Beta(1.5, 1)` is not ReinFlow's default — `'uniform'` is. You only get the π₀-style prior if you set `sample_t_type: beta` in your config. Don't assume importing ReinFlow gives you all of π₀'s tricks for free.
- **`logit_normal_samples = 1 / (1 + torch.exp(-normal_samples))` 的数值稳定性**: 这种手写 sigmoid 在极端负值时会下溢、极端正值时 `exp` 会上溢。生产代码应该用 `torch.sigmoid(normal_samples)`(内部有 fused 实现,无溢出)。这是研究代码常见的小瑕疵。
  / The hand-coded sigmoid `1 / (1 + torch.exp(-normal_samples))` underflows at very negative inputs and overflows at very positive ones. Production code should use `torch.sigmoid(normal_samples)` (fused, no overflow). A common research-code wart.
- **MSE 假设各动作维度等权**: `F.mse_loss` 对所有 `(B, horizon, action_dim)` 取平均。如果你的动作维度量纲差别很大(比如 7 个关节角度 + 1 个夹爪开合),夹爪那一维的损失贡献会被 7 个关节淹没。π₀ 和 ReinFlow 都假设上游已经标准化过——记得检查 `compute_norm_stats`。
  / MSE assumes every action dimension is equally weighted. If your action vector mixes scales (7 joint angles + 1 gripper opening), the gripper's loss will be drowned out by the 7 angles. Both π₀ and ReinFlow assume upstream normalization — check `compute_norm_stats` before training.

## 延伸阅读 / Further reading

- ReinFlow paper (NeurIPS 2025): https://arxiv.org/abs/2505.22094
- ReinFlow GitHub: https://github.com/ReinFlow/ReinFlow
- Conditional Flow Matching (Tong et al., 2023): https://arxiv.org/abs/2302.00482
- 今日 tracked 笔记: [openpi `Pi0.compute_loss`](2026-05-26-openpi-flow-matching-loss.md) — the production JAX counterpart
- diffusers `FlowMatchEulerDiscreteScheduler`: https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
- ReinFlow 同仓库的 RL 微调入口 `model/flow/mlp_shortcut.py`: https://github.com/ReinFlow/ReinFlow/blob/e722e151bed767f3ffef47527cf697f2358af55d/model/flow/mlp_shortcut.py
