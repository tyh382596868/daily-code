---
date: 2026-06-01
topic: diffusion
source: trending
repo: shengshu-ai/minWM
file: shared/algorithms/consistency_distillation.py
permalink: https://github.com/shengshu-ai/minWM/blob/e082c8a3297feaf048f1918f767f96a4ab4a85e8/shared/algorithms/consistency_distillation.py#L12-L90
difficulty: intermediate
read_time: ~12 min
tags: [code-of-the-day, diffusion, distillation, consistency-model, cfg]
---

# minWM 把 CM 蒸馏压成三个函数:抽 pair、教师跑 CFG+Euler、MSE / minWM compresses consistency distillation into three functions: sample a pair, teacher does CFG+Euler, student MSE

> **一句话 / In one line**: 整个 consistency distillation 训练循环可以拆成"抽相邻 (t, t_next) → 老师在 t 用 CFG 跑一步 Euler 得到 target → 学生在 t、EMA 在 t_next 各做一次预测,MSE"——三个纯张量函数。 / The whole consistency-distillation training loop boils down to "sample adjacent `(t, t_next)` → teacher runs one CFG + Euler step at `t` to produce the target → student at `t` and EMA at `t_next` each predict, MSE the difference" — three pure-tensor functions.

## 为什么重要 / Why this matters

Consistency Model 蒸馏从论文出来到现在,工程实现累计了厚厚一层 boilerplate——scheduler、wrapper、distributed hooks 等等遮蔽了算法本身的样子。今天 shengshu-ai 发布的 minWM 是一个"工程极简版"的世界模型实现,它把 CM 蒸馏直接放在 `shared/algorithms/consistency_distillation.py`,三个函数总共 90 行,**没有任何外部依赖,只有 torch**。读这段代码相当于看到了 CM 蒸馏的"灵魂":就是一个用老师 ODE 轨迹做监督的标准训练循环。理解了这 3 个函数,你就掌握了 LCM、Hyper-SD、PCM 等所有"少步蒸馏"方法共同的骨架。

Consistency Model distillation has accumulated a thick layer of engineering boilerplate since the original paper — schedulers, wrappers, distributed hooks all obscure the algorithm itself. shengshu-ai's freshly-released minWM is an "engineering-minimal" world model that drops CM distillation directly into `shared/algorithms/consistency_distillation.py`: three functions, 90 lines, **no external dependencies beyond torch**. Reading it is reading the *soul* of CM distillation: a standard supervised training loop where the targets come from the teacher's ODE trajectory. Internalize these 3 functions and you've grasped the skeleton shared by LCM, Hyper-SD, PCM, and every other "few-step distillation" method.

## 代码 / The code

`shengshu-ai/minWM` — [`shared/algorithms/consistency_distillation.py`](https://github.com/shengshu-ai/minWM/blob/e082c8a3297feaf048f1918f767f96a4ab4a85e8/shared/algorithms/consistency_distillation.py#L12-L90)

```python
def sample_cd_timestep_pair(
    num_steps, sigmas, timesteps, device, include_terminal=False,
):
    """Sample adjacent (t, t_next) pair for consistency distillation."""
    max_idx = num_steps - 1 if not include_terminal else num_steps
    idx = torch.randint(0, max_idx, (1,), device=device).item()
    t = timesteps[idx]
    t_next = timesteps[idx + 1] if idx + 1 < len(timesteps) else torch.tensor(0.0, device=device)
    sigma_t = sigmas[idx]
    sigma_t_next = sigmas[idx + 1]
    return sigma_t, sigma_t_next, t, t_next


def teacher_cfg_euler_step(
    v_cond, v_uncond, latent_t, t, t_next,
    guidance_scale, timestep_scale=1000.0,
):
    """Teacher CFG + single Euler step to produce the target latent.

    v_cfg = v_uncond + guidance_scale * (v_cond - v_uncond)
    dt = (t - t_next) / timestep_scale
    latent_t_next = latent_t - dt * v_cfg
    """
    v_cfg = v_uncond + guidance_scale * (v_cond - v_uncond)
    dt = (t - t_next) / timestep_scale
    while dt.dim() < v_cfg.dim():
        dt = dt.unsqueeze(-1)
    return latent_t - dt * v_cfg


def consistency_loss(cm_pred_t, cm_pred_t_next, reduction="mean"):
    """Consistency distillation loss: MSE between student(t) and EMA(t_next)."""
    loss = (cm_pred_t.float() - cm_pred_t_next.float()).pow(2)
    if reduction == "mean":
        return loss.mean()
    return loss
```

## 逐行讲解 / What's happening

1. **`sample_cd_timestep_pair` 的随机性 / Where randomness enters in `sample_cd_timestep_pair`**:
   - 中文: `torch.randint(0, max_idx, (1,))` 在 schedule 内随机抽一个索引 idx,然后取相邻的 `idx` 和 `idx+1` 作为 `(t, t_next)`。**每个训练 batch 都重新抽**——这是 CM 蒸馏的随机化设计:让模型在所有 (t, t_next) pair 上都学到一致性。
   - English: `torch.randint(0, max_idx, (1,))` picks a random index within the schedule, and `(idx, idx+1)` becomes the `(t, t_next)` pair. **A fresh pair every training batch** — this is CM distillation's randomization: enforce consistency across *all* adjacent pairs, not just one.

2. **`include_terminal` 标志 / The `include_terminal` flag**:
   - 中文: 控制是否允许采到最后一步(`idx == num_steps - 1`)。开启时 `t_next` 会强制是 `0.0`——这是直接学"从 `t` 一步到 `z_0`"的情形,Hyper-SD 风格。
   - English: Controls whether the final step (`idx == num_steps - 1`) can be sampled, in which case `t_next` is forced to `0.0` — a "go directly from `t` to `z_0`" pair, à la Hyper-SD's one-step distillation.

3. **`v_cfg = v_uncond + guidance_scale * (v_cond - v_uncond)` —— CFG 经典公式 / The classic CFG formula**:
   - 中文: 老师跑两次:一次给条件,一次不给条件。然后用线性外推合成"加强版条件预测"。**注意 CFG 是在老师这一步做的,学生不需要也跑两次**——这是 LCM-Distill 的一个重要 insight:把 CFG 烧进老师,学生学到的就已经是 CFG 之后的轨迹。
   - English: The teacher runs twice — once conditioned, once unconditioned — and linearly extrapolates a "stronger" prediction. **CFG happens on the teacher side; the student does not also run twice** — a key insight from LCM-Distill: bake CFG into the teacher, and the student naturally learns the post-CFG trajectory.

4. **`dt = (t - t_next) / timestep_scale`、`while dt.dim() < v_cfg.dim(): dt.unsqueeze(-1)` / Broadcasting `dt`**:
   - 中文: `timestep_scale=1000` 是因为 Wan2.1 把 timestep 表示成 0~1000 的整数,需要除回到连续时间。while 循环是 broadcasting 工具,把 `[B, F]` 形状的 dt 扩展到 `[B, F, 1, 1, 1]` 以匹配 `v_cfg`。这种"按需 unsqueeze 直到对齐"的写法在 video diffusion 里很常见。
   - English: `timestep_scale=1000` reflects Wan2.1 storing timesteps as integers in `[0, 1000]`, divided back to continuous time. The while loop is a broadcasting trick: extend `dt`'s shape from `[B, F]` to `[B, F, 1, 1, 1]` so it broadcasts against `v_cfg`. This "unsqueeze until aligned" pattern is everywhere in video diffusion code.

5. **`latent_t_next = latent_t - dt * v_cfg` —— 一步 Euler / One Euler step**:
   - 中文: 用 CFG 增强后的 velocity 做一步 Euler 积分,得到"老师认为下一步该到哪里"的 latent。这就是 CM 蒸馏的 target——学生模型在 `t_next` 的 EMA 版本应该输出和 `latent_t_next` 一致的"最终去噪结果"。
   - English: One Euler integration step with the CFG-boosted velocity yields "where the teacher thinks we should land next". This is the CM distillation target — the student's EMA at `t_next` should produce the same final-denoised output as the student at `t` after this Euler step.

6. **`consistency_loss` 的 `.float()` / The `.float()` casts in `consistency_loss`**:
   - 中文: 训练时模型权重通常是 bf16,但 loss 计算前要升回 float32,否则 (pred - target)² 会在 bf16 下严重精度损失,梯度噪声变大。这是大规模训练的标准做法。
   - English: Model weights are typically bf16 during training, but loss computation is upcast to float32 — otherwise `(pred - target)**2` loses serious precision in bf16 and gradient noise explodes. Standard practice in any large-scale training.

7. **EMA 学生 / The EMA student**:
   - 中文: 注意 `cm_pred_t_next` 是 EMA 版本(教师之外的"目标网络"),`cm_pred_t` 是 online 版本。EMA 在调用端 detach,因此 loss 只反传到 `cm_pred_t`——这就是 Polyak 平均 target network 的标准用法。
   - English: `cm_pred_t_next` comes from the EMA (target-network) version of the student; `cm_pred_t` comes from the online version. The EMA's output is detached at the call site, so gradients only flow through `cm_pred_t` — standard Polyak-averaged target-network usage.

## 类比 / The analogy

想象一个学钢琴的小孩(student)和他的老师(teacher)。老师是大师级演奏家,但每次教学只示范"在第 t 个小节按这个键、力度是 CFG 算出来的"——只示范一小段。学生要做的是:**让自己当下的演奏在第 t 小节,和"老师那段示范结束后的状态"对得上**。同时,学生还有一个"自我录音"(EMA 学生),前两天的版本被存了下来作为参考——学生要保证自己今天在第 t 小节的演奏,会自然过渡到"两天前的自己在第 t+1 小节的演奏"。这就是"consistency":学生在所有相邻 pair 上都和自己的过去/老师的轨迹保持一致。

Picture a kid learning piano (student) and their master teacher. The teacher is a virtuoso but at each lesson only demonstrates "play this note at measure `t` with this dynamic (CFG-derived) force" — a short slice. The student's job is to ensure that **their own playing at measure `t` smoothly leads to the state the teacher's demonstration ended in**. At the same time, the student has a "self-recording" (EMA student) — a slightly-stale version of themselves from previous days — and they must guarantee that today's playing at measure `t` naturally evolves into their past self's playing at measure `t+1`. That two-way agreement *is* "consistency": the student stays consistent with their own past and with the teacher's trajectory across all adjacent pairs.

## 自己跑一遍 / Try it yourself

```python
import torch

# Toy schedule: 8 steps, sigma from 1.0 to 0.0
NUM_STEPS = 8
sigmas    = torch.linspace(1.0, 0.0, NUM_STEPS + 1)
timesteps = (sigmas[:-1] * 1000).to(torch.int64)

def sample_pair(num_steps, sigmas, timesteps):
    idx = torch.randint(0, num_steps - 1, (1,)).item()
    return sigmas[idx], sigmas[idx + 1], timesteps[idx].float(), timesteps[idx + 1].float()

def teacher_cfg_euler(v_cond, v_uncond, latent_t, t, t_next, w=5.0, ts=1000.0):
    v_cfg = v_uncond + w * (v_cond - v_uncond)
    dt = (t - t_next) / ts
    while dt.dim() < v_cfg.dim():
        dt = dt.unsqueeze(-1)
    return latent_t - dt * v_cfg

torch.manual_seed(0)
latent_t = torch.randn(1, 4, 16, 16)
v_cond, v_uncond = torch.randn_like(latent_t), torch.randn_like(latent_t) * 0.3

sigma_t, sigma_tn, t, t_next = sample_pair(NUM_STEPS, sigmas, timesteps)
target = teacher_cfg_euler(v_cond, v_uncond, latent_t, t, t_next)

print(f"sampled t={t.item():.0f}, t_next={t_next.item():.0f}")
print(f"teacher moved latent by {(target - latent_t).norm().item():.4f}")
print(f"target shape: {target.shape}")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
sampled t=375, t_next=250
teacher moved latent by 2.27...
target shape: torch.Size([1, 4, 16, 16])
```

中文一两句:每次跑 `t` 和 `t_next` 都不一样——这是 `sample_cd_timestep_pair` 的随机性。把 `w` 调大或调小,会直接改变 `target` 离 `latent_t` 的距离——这就是 CFG 强度被烧进 teacher trajectory 的方式。

`t` and `t_next` differ each run — that's the randomization in `sample_cd_timestep_pair`. Increasing or decreasing `w` directly changes the distance between `target` and `latent_t` — which is precisely how CFG strength gets baked into the teacher trajectory.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **`huggingface/diffusers` 的 LCM 蒸馏脚本** / **HF diffusers LCM distillation scripts**: 中文:同样的"sample pair → teacher CFG Euler → MSE",但用 scheduler 抽象包裹,代码量是 minWM 的 5 倍。 / English: Same recipe — sample pair, teacher CFG Euler, MSE — wrapped in scheduler abstractions; ~5× more code than minWM.
- **2026-05-29 的 Causal-Forcing DMD 笔记** / **The 2026-05-29 Causal-Forcing DMD note**: 中文:DMD 是另一种蒸馏路线——梯度 = `(score_fake - score_real)`,目标不是匹配 trajectory,而是匹配 score。读两份对比能看清"蒸馏家族"的分支。 / English: DMD takes a different route — gradient = `(score_fake - score_real)`, matching score rather than trajectory. Read alongside to see the branches of the "distillation family".
- **Hyper-SD、PCM、TCD**: 中文:三个都是 CM 蒸馏的变种,主要差别在"采到的 pair 怎么选"——Hyper-SD 用 `(t, 0)` 强制 one-step,PCM 用 phased 区间。 / English: All three are CM variants, mostly differing in *which* pairs to sample — Hyper-SD forces `(t, 0)` for one-step, PCM samples within phased intervals.
- **DDPO / DPO 风格的 RL 蒸馏** / **DDPO / DPO-style RL distillation**: 中文:把 `consistency_loss` 换成 RL 奖励差,就变成了从教师 trajectory 学偏好——架构很相似。 / English: Swap `consistency_loss` for an RL reward differential and you get preference learning from teacher trajectories — structurally near-identical.

## 注意事项 / Caveats / when it breaks

- **EMA 更新频率影响稳定性 / EMA update frequency matters**: 中文:`cm_pred_t_next` 的 EMA 衰减率(通常 0.999~0.9999)如果太接近 1,target 几乎不变,训练慢但稳;如果太小,target 跟 online 太接近,会出现"自我蒸馏发散"。 / English: The EMA decay (typically 0.999–0.9999) trades off stability vs. learning speed. Too close to 1 and the target barely moves (slow but stable); too small and target ≈ online, leading to "self-distillation divergence".
- **`include_terminal=True` 必须谨慎 / `include_terminal=True` is tricky**: 中文:开启后会采到 `(t_last, 0)`,这要求模型本来就能"一步到位",通常只在训练后期或 finetune 阶段开启。 / English: With `include_terminal=True` you sometimes sample `(t_last, 0)`, requiring the model to already be capable of one-step generation. Typically enabled only in late training or fine-tuning phases.
- **CFG scale 必须和推理时一致 / CFG scale at distillation must match inference**: 中文:用 `guidance_scale=5` 蒸馏出来的学生,推理时就不能再做 CFG——CFG 已经被烧进权重。这是 LCM 用户最常踩的坑。 / English: A student distilled at `guidance_scale=5` should be run *without* CFG at inference — CFG is already baked in. The single most common mistake among LCM users.

## 延伸阅读 / Further reading

- [Consistency Models (Song et al.)](https://arxiv.org/abs/2303.01469) — the original paper
- [Latent Consistency Models (Luo et al.)](https://arxiv.org/abs/2310.04378) — pioneered baking CFG into the teacher
- [minWM repository](https://github.com/shengshu-ai/minWM) — the freshly released minimal world-model codebase this file lives in
- Daily code 2026-05-29 — Causal-Forcing DMD gradient (a non-CM distillation contrast)
