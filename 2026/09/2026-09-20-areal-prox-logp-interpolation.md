---
date: 2026-09-20
topic: infrastructure
source: trending
repo: areal-project/AReaL
file: areal/trainer/ppo/actor.py
permalink: https://github.com/areal-project/AReaL/blob/3aad281e011d0254d598cff9c3d16be7c7a44312/areal/trainer/ppo/actor.py#L1351-L1431
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, trending, areal, asynchronous-rl, ppo]
stars: 5778
---

# AReaL proximal log-prob 插值：异步 RL 不必每次重跑旧 policy / AReaL Proximal Log-Prob Interpolation: Async RL Without Re-running the Old Policy

> **一句话 / In one line**: AReaL 用 rollout token 的 policy version 在 behavior policy 与 current policy 之间做版本感知插值，近似 PPO 所需的 proximal log-prob。 / AReaL uses per-token rollout policy versions to interpolate between behavior and current policies, approximating PPO's proximal log-prob.

## 为什么重要 / Why this matters

异步 RL 系统里，rollout worker 生成样本时的 policy 版本可能落后于 learner 当前版本。标准 PPO 想要 behavior、current 和 proximal 三份 log-prob；如果为 proximal policy 再做一次完整 forward，通信和计算成本会抵消异步带来的吞吐收益。

In asynchronous RL, the policy version that generated a rollout may lag behind the learner's current version. Standard PPO wants behavior, current, and proximal log-probabilities; a full extra forward for the proximal policy can erase the throughput benefit of asynchronous execution.

AReaL 的函数把“上一次 broadcast 的 policy”视为 `current_version - 1`，对生成 token 按版本 gap 计算 `alpha`。它默认提供 log-space geometric interpolation、probability-space arithmetic interpolation 和 rollout baseline 三种方法，同时明确排除 prompt token，因为 prompt 没有 generation version。

AReaL treats the last broadcast policy as `current_version - 1` and computes `alpha` from the version gap for generated tokens. It exposes log-space geometric interpolation, probability-space arithmetic interpolation, and the rollout baseline, while explicitly excluding prompt tokens because they have no generation version.

## 代码 / The code

`areal-project/AReaL` — [`areal/trainer/ppo/actor.py`](https://github.com/areal-project/AReaL/blob/3aad281e011d0254d598cff9c3d16be7c7a44312/areal/trainer/ppo/actor.py#L1351-L1431)

```python
def compute_prox_logp_approximations(
    old_logp: torch.Tensor,
    logprobs: torch.Tensor,
    versions: torch.Tensor,
    current_version: int,
    method: str | None = None,
) -> dict[str, torch.Tensor]:
    """Compute approximation(s) for proximal policy log-probabilities."""
    v_proximal = current_version - 1
    v_behave = versions.float()
    v_theta = float(current_version)

    # Only approximate generated tokens (version >= 0).
    generated_tokens_mask = versions >= 0
    version_diff = v_theta - v_behave
    version_gap = v_proximal - v_behave
    alpha = torch.where(
        (version_diff > 0) & generated_tokens_mask,
        version_gap / version_diff,
        torch.zeros_like(v_behave),
    )
    alpha = torch.clamp(alpha, 0.0, 1.0)

    approximations = {}
    methods_to_compute = [method] if method else PROX_APPROX_METHODS_ALL

    for m in methods_to_compute:
        if m == PROX_APPROX_METHOD_LOGLINEAR:
            approximations[PROX_APPROX_METHOD_LOGLINEAR] = old_logp + alpha * (
                logprobs - old_logp
            )
        elif m == PROX_APPROX_METHOD_LINEAR:
            p_behave = torch.exp(old_logp)
            p_theta = torch.exp(logprobs)
            p_arithmetic = (1 - alpha) * p_behave + alpha * p_theta
            approximations[PROX_APPROX_METHOD_LINEAR] = torch.log(p_arithmetic + 1e-10)
        elif m == PROX_APPROX_METHOD_ROLLOUT:
            approximations[PROX_APPROX_METHOD_ROLLOUT] = old_logp.clone()

    return approximations
```

## 逐行讲解 / What's happening

1. **proximal version / Proximal version**:
   - 中文: 假设最近一次 broadcast 是 `current_version - 1`，不需要保存一份完整旧模型。
   - English: The last broadcast is assumed to be `current_version - 1`, so a full old model need not be retained.
2. **prompt 排除 / Exclude prompts**:
   - 中文: `versions < 0` 的 prompt token 没有 rollout generation version，`alpha` 被强制为 0。
   - English: Prompt tokens with `versions < 0` have no rollout generation version, so `alpha` is forced to zero.
3. **版本插值 / Version interpolation**:
   - 中文: behavior 版本等于 proximal 时 `alpha=0`；behavior 已经等于 current 时 `alpha=1`；中间版本线性落在两者之间。
   - English: `alpha=0` when behavior equals proximal, `alpha=1` when behavior equals current, and intermediate versions fall linearly between them.
4. **log-linear 方法 / Log-linear method**:
   - 中文: 在 log probability 空间插值，等价于 probability 空间的 geometric mean，数值上自然适配 logp。
   - English: Interpolating in log-probability space is equivalent to a geometric mean in probability space and fits logp numerics directly.
5. **linear 方法 / Linear method**:
   - 中文: 先恢复 probability，再做 arithmetic mean，最后取 log；这是另一种可比较的近似。
   - English: It converts back to probability, takes an arithmetic mean, and logs again as a second comparable approximation.
6. **rollout baseline / Rollout baseline**:
   - 中文: 原样使用 `old_logp`，为指标比较提供不插值的 baseline。
   - English: It returns `old_logp` unchanged as a no-interpolation baseline for metric comparison.

## 类比 / The analogy

这像三个版本的天气预报：行为 policy 是已经出发的船收到的预报，current policy 是今天的预报，proximal policy 是上一版通告。每个 token 都带自己的发布日期，所以可以按时间差估计它更接近哪一版，而不用把上一版预报员重新请回来。

It is like three editions of a weather forecast: the behavior policy guided the departing ship, the current policy is today's edition, and the proximal policy is the previous bulletin. Each token carries its publication date, so distance can estimate which edition it resembles without rehiring the previous forecaster.

## 自己跑一遍 / Try it yourself

```python
import torch

old_logp = torch.log(torch.tensor([0.25, 0.50, 0.80]))
current_logp = torch.log(torch.tensor([0.50, 0.25, 0.60]))
versions = torch.tensor([2, 3, -1])
current_version = 4
proximal = current_version - 1
alpha = torch.where(
    (current_version - versions > 0) & (versions >= 0),
    (proximal - versions) / (current_version - versions),
    torch.zeros_like(versions, dtype=torch.float32),
).clamp(0, 1)
print(alpha)
print(old_logp + alpha * (current_logp - old_logp))
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
tensor([0.6667, 0.0000, 0.0000])
tensor([-1.0397, -0.6931, -0.2231])
```

中文: 版本 2 的 token 处于 behavior 与 proximal/current 之间；版本 3 已经是 proximal；prompt 版本 -1 不插值。
English: Version 2 lies between behavior and proximal/current; version 3 is already proximal; prompt version -1 is never interpolated.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **stale-gradient correction** / **stale-gradient correction**: 中文: 异步优化会携带版本或时间戳，再根据陈旧程度调整贡献。 / English: Asynchronous optimization carries versions or timestamps and adjusts contributions based on staleness.
- **replay-buffer importance sampling** / **replay-buffer importance sampling**: 中文: 行为策略与目标策略的比值用于修正数据分布；这里的插值是更便宜的 proximal 近似。 / English: Behavior/target policy ratios correct data distribution; this interpolation is a cheaper proximal approximation.
- **distributed checkpoint versions** / **distributed checkpoint versions**: 中文: 版本号让系统能在不传整个对象的情况下判断状态新旧。 / English: Versions let a system reason about freshness without transferring the whole object.

## 注意事项 / Caveats / when it breaks

- **假设 proximal 是上一版 / Assumed proximal version**: 中文: 如果广播策略不总是连续递增一版，`current_version - 1` 必须改成真实的 last-broadcast version。 / English: If broadcasts do not advance by exactly one, replace `current_version - 1` with the actual last-broadcast version.
- **logp 必须对齐 token / Logp must align with tokens**: 中文: `old_logp`、`logprobs` 和 `versions` 的 shape 与 mask 必须完全对应。 / English: `old_logp`, `logprobs`, and `versions` must have identical token alignment and masking.
- **插值不是精确旧模型 / Interpolation is not an exact old model**: 中文: 这是吞吐与精确性的折中，仍要通过 KL、clip fraction 和 reward 监控验证。 / English: This trades exactness for throughput and should be validated with KL, clip fraction, and reward metrics.

## 延伸阅读 / Further reading

- [AReaL proximal approximation](https://github.com/areal-project/AReaL/blob/3aad281e011d0254d598cff9c3d16be7c7a44312/areal/trainer/ppo/actor.py#L1351-L1431)
- [AReaL project](https://github.com/areal-project/AReaL)
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
