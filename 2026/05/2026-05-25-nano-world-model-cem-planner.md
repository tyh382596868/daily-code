---
date: 2026-05-25
topic: diffusion
source: trending
repo: simchowitzlabpublic/nano-world-model
file: src/planning/cem_planner.py
permalink: https://github.com/simchowitzlabpublic/nano-world-model/blob/4a76f10defe32587b0ec691ba940d94fb6ea1050/src/planning/cem_planner.py#L126-L214
difficulty: intermediate
read_time: ~12 min
tags: [code-of-the-day, diffusion, world-model, planning, cem, mpc]
---

# CEM planning inside a learned world model

> **In one line**: Sample N random action sequences, dream them forward through a world model, keep the top-k that get closest to your goal, refit a Gaussian to those — and repeat.

## Why this matters

`nano-world-model` (449⭐ and rising fast in May 2026) is a minimalist diffusion-forcing video world model — it learns to predict future frames conditioned on actions. Once you have such a model, the next question is: *how do you use it to decide what to do?* Gradient-based action optimization is finicky through long generative rollouts, but a simple gradient-free recipe — the **Cross-Entropy Method (CEM)** — works remarkably well. It's the trick behind PETS, MBRL, TD-MPC, and most "world model + planning" papers since 2018.

This 90-line implementation is the cleanest reference I've seen, including a one-line fix for CEM's classic failure mode.

## The code

`simchowitzlabpublic/nano-world-model` — [`src/planning/cem_planner.py`](https://github.com/simchowitzlabpublic/nano-world-model/blob/4a76f10defe32587b0ec691ba940d94fb6ea1050/src/planning/cem_planner.py#L126-L214)

```python
def plan(
    self,
    obs_0: Dict[str, torch.Tensor],
    obs_g: Dict[str, torch.Tensor],
    actions: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict]:
    batch_size = obs_0["visual"].shape[0]

    obs_0 = {k: v.to(self.device) for k, v in obs_0.items()}
    obs_g = {k: (v.to(self.device) if v is not None else None) for k, v in obs_g.items()}

    # Encode goal observation
    with torch.no_grad():
        z_obs_g = self.world_model.encode_obs(obs_g)

    # Initialize action distribution
    mu, sigma = self.init_mu_sigma(batch_size, actions)
    mu, sigma = mu.to(self.device), sigma.to(self.device)

    losses_history = []

    for i in range(self.opt_steps):
        batch_losses = []

        for b in range(batch_size):
            # Sample action sequences
            action_samples = (
                torch.randn(self.num_samples, self.horizon, self.action_dim).to(self.device)
                * sigma[b]
                + mu[b]
            )
            action_samples[0] = mu[b]  # First sample is current mean
            if self.action_low is not None or self.action_high is not None:
                action_samples = action_samples.clamp(
                    min=self.action_low if self.action_low is not None else -float("inf"),
                    max=self.action_high if self.action_high is not None else float("inf"),
                )

            obs_0_single = {k: v[b:b+1] for k, v in obs_0.items()}
            z_obs_g_single = {
                k: v[b:b+1] if v is not None else None
                for k, v in z_obs_g.items()
            }
            loss = self._compute_losses_for_samples(
                obs_0_single=obs_0_single,
                z_obs_g_single=z_obs_g_single,
                action_samples=action_samples,
            )

            # Select top-k
            topk_idx = torch.argsort(loss)[:self.topk]
            topk_actions = action_samples[topk_idx]
            batch_losses.append(loss[topk_idx[0]].item())

            # Update distribution, flooring sigma so it doesn't collapse.
            mu[b] = topk_actions.mean(dim=0)
            sigma[b] = topk_actions.std(dim=0).clamp(min=self.sigma_min)

        avg_loss = np.mean(batch_losses)
        losses_history.append(avg_loss)

    return mu, info
```

And the inner helper that does the world-model rollout:

```python
def _compute_losses_for_samples(self, obs_0_single, z_obs_g_single, action_samples):
    # ... chunked rollout to avoid OOM ...
    with torch.no_grad():
        z_obses, _ = self.world_model.rollout(obs_0=obs_0_expanded, act=cur_actions)
        losses.append(self.objective_fn(z_obses, z_obs_g_expanded))
    return torch.cat(losses, dim=0)
```

## What's happening

1. **Encode the goal frame** (`z_obs_g = encode_obs(obs_g)`) — comparison happens in the world model's latent space, not pixel space. That's far cheaper and more meaningful than per-pixel MSE.
2. **`init_mu_sigma`** sets up a Gaussian over action sequences of shape `[horizon, action_dim]`. Mean starts at zero (or at a provided warm-start), std starts at `var_scale`.
3. **For `opt_steps` iterations**:
   - **Sample**: draw `num_samples` action sequences from `N(mu, sigma²)`. The first sample is forced to be `mu` exactly — an elitism trick so the best-so-far never gets lost to noise.
   - **Clamp** to the legal action range.
   - **Roll out**: call `world_model.rollout(obs_0, act)` to dream `num_samples` futures in latent space. No gradient — pure inference.
   - **Score**: `objective_fn` measures how close the predicted latent trajectory ends up to the goal latent.
   - **Top-k**: `torch.argsort(loss)[:self.topk]` keeps the `topk` best sequences.
   - **Refit**: new `mu = top-k.mean`, new `sigma = top-k.std`, **clamped to `sigma_min`** so it never collapses to zero.
4. **Return** the final `mu` — the best action sequence the planner converged to.

The whole loop is gradient-free. The world model's parameters don't change. The only thing being optimized is the Gaussian over actions.

### The one-line bug fix worth memorizing

```python
sigma[b] = topk_actions.std(dim=0).clamp(min=self.sigma_min)
```

Without that `.clamp(min=sigma_min)`, CEM has a famous failure mode: after a few iterations, the top-k samples become very similar to each other, so their std collapses toward zero, so the next iteration samples are nearly identical to `mu`, so std collapses further — **premature convergence**. You end up locked onto whatever the noise happened to favor early. Flooring `sigma` at a small positive value keeps the planner exploring. PETS and TD-MPC do exactly the same thing.

## The analogy

Imagine you're trying to throw a paper airplane into a trash can across the room, but you can't actually throw it — you have a **simulator** that tells you where any given throw would land.

- Round 1: Throw 100 airplanes with **random** angles and speeds, all centered roughly at "forward". Most miss wildly.
- Pick the **10 throws** that landed closest to the can. They cluster around some angle and speed.
- Round 2: Throw 100 more airplanes from a Gaussian centered on the average of those 10, with std equal to their spread. Now you're sampling tighter, around the promising region.
- Repeat 5 times. By round 5, your 100 throws are all clustered around the optimum.

CEM is **survival of the closest-to-goal**, in the space of trajectories.

The `sigma_min` floor is the same instinct as occasionally throwing one wild airplane "just in case" — without it, you'd commit too early to a slightly-suboptimal throwing style and never discover the better one.

## Try it yourself

```python
# CEM optimizing a 2D function — no world model needed.
import torch

def objective(x):  # 2D Rosenbrock; min at (1, 1)
    return (1 - x[:, 0])**2 + 100 * (x[:, 1] - x[:, 0]**2)**2

mu = torch.zeros(2)
sigma = torch.ones(2)
N, K, ITERS, SIGMA_MIN = 200, 20, 12, 0.01

for i in range(ITERS):
    samples = torch.randn(N, 2) * sigma + mu
    losses  = objective(samples)
    topk    = samples[torch.argsort(losses)[:K]]
    mu      = topk.mean(0)
    sigma   = topk.std(0).clamp(min=SIGMA_MIN)
    print(f"iter {i:2d}  mu={mu.tolist()}  loss={losses.min().item():.4f}")
```

Run with:
```bash
pip install torch
python try.py
```

Expected output (last lines):
```
iter 10  mu=[0.998..., 0.996...]  loss=0.0001
iter 11  mu=[0.999..., 0.998...]  loss=0.0000
```

The optimum of Rosenbrock is at `(1, 1)` and CEM finds it in ~12 iterations with 200 samples. Now replace `objective` with `world_model.rollout(...) → distance_to_goal` and you have the planner above.

## Where this pattern shows up elsewhere

- **PETS** (Chua et al., 2018) — CEM over an ensemble of dynamics models, the canonical model-based RL benchmark on MuJoCo tasks.
- **TD-MPC / TD-MPC2** — CEM-MPPI hybrid on a learned latent world model, currently SOTA for continuous-control RL.
- **DreamerV3** — uses actor-critic instead of CEM, but the latent-rollout part is the same idea.
- **Diffusion Policy planning variants** — CEM over short action chunks, scored by a trained value function.
- **Hardware engineering** — antenna design, circuit optimization, and protein engineering all use CEM for the same reason: noisy black-box objective, no usable gradient.

## Caveats / when it breaks

- **Per-batch Python loop** (`for b in range(batch_size)`) — fine for batch=1 MPC at robot control rates, but quadratic if you try to batch many environments. Real high-throughput implementations vectorize across the batch.
- **Action-sequence independence** — CEM treats every timestep's action as independent under the Gaussian. The "top-k smoothing" implicitly couples them, but if your task needs strongly correlated action sequences (e.g. high-frequency oscillations), MPPI or a colored-noise prior beats vanilla CEM.
- **Cost of `num_samples × horizon` rollouts** — every iteration runs `num_samples` independent imagined trajectories of length `horizon` through the world model. With diffusion-based world models (slow!), this becomes the runtime bottleneck. `rollout_batch_size` exists in this file precisely to chunk it and avoid OOM.
- **World-model bias** — the planner exploits whatever errors the model has. If the world model thinks "fly through this wall" is a great idea, CEM will gladly find that adversarial action sequence. This is the well-known *exploitation-of-model-error* problem.
- **The first sample being `mu` exactly** — an "elitist" trick that preserves the best-so-far. Without it, sampling noise alone can degrade your incumbent across iterations.

## Further reading

- ["Deep Reinforcement Learning in a Handful of Trials"](https://arxiv.org/abs/1805.12114) (PETS) — the paper that made CEM-on-learned-models popular.
- ["TD-MPC2: Scalable, Robust World Models for Continuous Control"](https://arxiv.org/abs/2310.16828) — the modern variant.
- ["The Cross-Entropy Method"](https://link.springer.com/book/10.1007/978-1-4757-4321-0) by Rubinstein & Kroese — the original textbook.
- The companion `objective.py` in this repo shows a typical latent-distance objective; pair it with this planner and `diffusion_world_model.py` for a full minimal MPC loop.
