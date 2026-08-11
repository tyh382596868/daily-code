---
date: 2026-08-11
topic: diffusion
source: trending
repo: next-state/open-dreamer
file: dreamer/training.py
permalink: https://github.com/next-state/open-dreamer/blob/797e41f052b5996740938fd2fe8161f1866de3a2/dreamer/training.py#L244-L287
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, diffusion, world-model]
---

# Open-Dreamer shortcut forcing：两次半步蒸成一步 / Open-Dreamer Shortcut Forcing: Distill Two Half-Steps into One

> **一句话 / In one line**: `compute_bootstrap_loss()` 用两个 stop-gradient 半步构造目标，让 world model 学会用更大步长直接到达同一个 endpoint。 / `compute_bootstrap_loss()` builds a target from two stop-gradient half-steps so the world model learns to reach the same endpoint with one larger step.

## 为什么重要 / Why this matters

World model rollout 慢，通常是因为每个未来 latent 要走很多去噪小步。shortcut forcing 的目标是训练模型“抄近路”：两次半步老师给出一个平均速度，学生用一次大步预测同一个干净 latent。这样推理时可以少走步数。

World-model rollout is slow when each future latent needs many denoising steps. Shortcut forcing trains the model to take a shortcut: two teacher half-steps define an average velocity, and the student predicts the same clean endpoint with one larger step.

## 代码 / The code

`next-state/open-dreamer` — [`dreamer/training.py`](https://github.com/next-state/open-dreamer/blob/797e41f052b5996740938fd2fe8161f1866de3a2/dreamer/training.py#L244-L287)

```python
def compute_bootstrap_loss(
    z_pred: jnp.ndarray,
    z_tilde: jnp.ndarray,
    b_prime: jnp.ndarray,
    b_doubleprime: jnp.ndarray,
    sigma: jnp.ndarray,
    weighting: str,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Bootstrap self-consistency loss for shortcut forcing.

    Trains the model to predict the same endpoint whether taking one large step
    or two smaller steps. Computed directly in x-space by algebraically cancelling
    the (1-σ)² and 1/(1-σ) terms:
        (1-σ)² · ||(z_pred - z_tilde)/(1-σ) - v_target||²
      = ||z_pred - (z_tilde + (1-σ)·v_target)||²
    """
    # Target velocity is average of two half-steps (stop gradient, clipped)
    v_target = jax.lax.stop_gradient((b_prime + b_doubleprime) / 2.0)
    v_target = jnp.clip(v_target, -4.0, 4.0)

    # X-space target avoids the numerically unstable 1/(1-σ) division in v_hat
    one_minus_sigma = (1.0 - sigma)[..., None, None]
    x_target = z_tilde + one_minus_sigma * v_target

    boot_per_token = (z_pred - x_target) ** 2
    boot_per_step = jnp.mean(boot_per_token, axis=(2, 3))  # (B, T)

    # Apply sigma-dependent weighting and reduce
    weights = loss_weight(sigma, weighting)
    return jnp.mean(boot_per_step * weights), jnp.mean(boot_per_step)
```

## 逐行讲解 / What's happening

1. **第 244-251 行 / Lines 244-251 (inputs)**:
   - 中文: `z_pred` 是学生一次大步的预测，`b_prime/b_doubleprime` 是两次半步老师算出的速度。
   - English: `z_pred` is the student's one-large-step prediction; `b_prime/b_doubleprime` are velocities from two teacher half-steps.
2. **第 274-276 行 / Lines 274-276 (stop-gradient target)**:
   - 中文: 两个半步速度平均后 `stop_gradient`，再 clip，避免 bootstrap target 反向拖动老师路径。
   - English: The two half-step velocities are averaged, stop-gradiented, and clipped so the bootstrap target does not backprop through the teacher path.
3. **第 278-280 行 / Lines 278-280 (x-space endpoint)**:
   - 中文: 直接构造 `x_target`，避开 `(z_pred - z_tilde)/(1-sigma)` 在 `sigma` 接近 1 时的数值放大。
   - English: It constructs `x_target` directly, avoiding the unstable `(z_pred - z_tilde)/(1-sigma)` when `sigma` is near 1.
4. **第 282-287 行 / Lines 282-287 (weighted MSE)**:
   - 中文: loss 先对 token 维度平均成每个时间步的误差，再按 sigma-dependent weight 聚合。
   - English: The loss averages over token dimensions into per-step errors, then applies sigma-dependent weights.

## 类比 / The analogy

像学开车导航：老师先示范“两段路怎么走”，学生要学会“一次转向直接到同一个路口”。目标路口固定，老师路线不参与反向传播。

It is like learning a driving shortcut: the teacher demonstrates two road segments, and the student learns one turn that reaches the same intersection. The destination is fixed; the teacher route is not updated.

## 自己跑一遍 / Try it yourself

```python
z_tilde = 0.2
sigma = 0.5
b1, b2 = 1.0, 1.4
v_target = max(-4.0, min(4.0, (b1 + b2) / 2))
x_target = z_tilde + (1 - sigma) * v_target
z_pred = 0.75
loss = (z_pred - x_target) ** 2
print(round(x_target, 3))
print(round(loss, 4))
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
0.8
0.0025
```

学生预测 `0.75` 已经接近两个半步定义的 endpoint `0.8`，所以 bootstrap loss 很小。

The student prediction `0.75` is close to the endpoint `0.8` defined by two half-steps, so the bootstrap loss is small.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Consistency distillation** / **Consistency distillation**: 也训练模型跨过多个小步，直接匹配更远的去噪状态。 / It also trains a model to skip small steps and match a farther denoised state.
- **DMD-style world models** / **DMD-style world models**: student/fake-score 路径常用 stop-gradient target 保持训练稳定。 / Student/fake-score paths often use stop-gradient targets for stability.

## 注意事项 / Caveats / when it breaks

- **teacher target 质量决定上限** / **Teacher target quality caps performance**: 两个半步如果本身很差，学生会稳定地学到差 endpoint。 / If the two half-steps are poor, the student reliably learns a poor endpoint.
- **`sigma` 接近 1 要小心** / **Be careful near `sigma=1`**: 代码转到 x-space 正是为了避免速度空间除以很小的 `1-sigma`。 / The x-space target exists to avoid dividing by a tiny `1-sigma`.

## 延伸阅读 / Further reading

- [Open-Dreamer `compute_bootstrap_loss`](https://github.com/next-state/open-dreamer/blob/797e41f052b5996740938fd2fe8161f1866de3a2/dreamer/training.py#L244-L287)

