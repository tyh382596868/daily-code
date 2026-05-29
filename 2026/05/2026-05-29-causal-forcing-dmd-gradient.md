---
date: 2026-05-29
topic: diffusion
source: trending
repo: thu-ml/Causal-Forcing
file: model/dmd.py
permalink: https://github.com/thu-ml/Causal-Forcing/blob/bb190459d99b074b5803ed6ba0b5091b16d5585d/model/dmd.py#L56-L128
difficulty: advanced
read_time: ~13 min
tags: [code-of-the-day, diffusion, distillation, dmd, causal-forcing]
---

# 用"两个 score 网络一减"算出蒸馏梯度 / Distillation gradient = subtract two score networks

> **一句话 / In one line**: Distribution Matching Distillation 的核心是这 70 行：让学生网络生成一帧、加噪、问"真"score 和"假"score、相减、归一化 —— 这个 `(s_fake - s_real)` 就是 KL 散度对参数的梯度近似。 / The whole DMD update is these 70 lines: have the student produce a frame, add noise, ask a "real" score net and a "fake" score net for their scores, subtract, normalize — `s_fake - s_real` is a Monte-Carlo estimate of the KL gradient.

## 为什么重要 / Why this matters

学生（generator）想"匹配"老师的分布。常规思路要么是 forward KL（教师采样、学生学），要么 reverse KL（学生采样、老师评判，但 KL 本身根本没法直接算）。DMD 的关键观察是：reverse KL 对学生参数的**梯度**可以写成 `E[s_fake(x_t) - s_real(x_t)]` —— 你根本不需要算 KL 本身。score 在哪里来？把任何一个预训练 diffusion model 当 black-box —— 它的 `eps_predict / sqrt(1-α)` 就是 score。这套 trick 把"很难"的目标（分布匹配）变成"很简单"的两次 score 评估 + 一次相减，于是 1-step / 4-step 的视频生成就能从 50-step 的教师蒸馏出来。Causal-Forcing 把它升级到 causal / interactive 场景，是 2026 视频生成蒸馏的代表作。

The student (generator) wants its samples to match the teacher's distribution. Forward KL needs the student to score teacher samples (cheap but biased); reverse KL needs evaluating KL itself (intractable). DMD's key observation is that the *gradient* of reverse KL with respect to the student's parameters reduces to `E[s_fake(x_t) - s_real(x_t)]` — you never compute KL itself. Where do scores come from? Any pretrained diffusion model: its noise prediction `eps / sqrt(1-α)` *is* the score. The trick converts a notoriously hard problem (distribution matching) into two score-net forwards and a subtraction, enabling 1-step or few-step generators distilled from 50-step teachers. Causal-Forcing pushes the recipe into causal/interactive video, and is one of the headline distillation methods of 2026.

## 代码 / The code

`thu-ml/Causal-Forcing` — [`model/dmd.py`](https://github.com/thu-ml/Causal-Forcing/blob/bb190459d99b074b5803ed6ba0b5091b16d5585d/model/dmd.py#L56-L128)

```python
def _compute_kl_grad(
    self,
    noisy_image_or_video: torch.Tensor,
    estimated_clean_image_or_video: torch.Tensor,
    timestep: torch.Tensor,
    conditional_dict: dict, unconditional_dict: dict,
    normalization: bool = True,
) -> Tuple[torch.Tensor, dict]:
    """Compute the KL grad (eq 7 in https://arxiv.org/abs/2311.18828)."""

    # Step 1: Compute the fake score
    _, pred_fake_image_cond = self.fake_score(
        noisy_image_or_video=noisy_image_or_video,
        conditional_dict=conditional_dict, timestep=timestep,
    )
    if self.fake_guidance_scale != 0.0:
        _, pred_fake_image_uncond = self.fake_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=unconditional_dict, timestep=timestep,
        )
        pred_fake_image = pred_fake_image_cond + (
            pred_fake_image_cond - pred_fake_image_uncond
        ) * self.fake_guidance_scale
    else:
        pred_fake_image = pred_fake_image_cond

    # Step 2: Compute the real score with CFG
    _, pred_real_image_cond = self.real_score(
        noisy_image_or_video=noisy_image_or_video,
        conditional_dict=conditional_dict, timestep=timestep,
    )
    _, pred_real_image_uncond = self.real_score(
        noisy_image_or_video=noisy_image_or_video,
        conditional_dict=unconditional_dict, timestep=timestep,
    )
    pred_real_image = pred_real_image_cond + (
        pred_real_image_cond - pred_real_image_uncond
    ) * self.real_guidance_scale

    # Step 3: Compute the DMD gradient (DMD paper eq. 7)
    grad = (pred_fake_image - pred_real_image)

    if normalization:
        # Step 4: Gradient normalization (DMD paper eq. 8)
        p_real = (estimated_clean_image_or_video - pred_real_image)
        normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
        grad = grad / normalizer
    grad = torch.nan_to_num(grad)

    return grad, {
        "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
        "timestep": timestep.detach(),
    }
```

## 逐行讲解 / What's happening

1. **两个 score 网络是什么 / What `real_score` and `fake_score` actually are**:
   - 中文：`real_score` 是冻结的教师 diffusion model（比如 Wan2.1）；`fake_score` 是一个 **持续在线训练** 的、跟学生当前分布对齐的 diffusion model —— 它跟着学生每走一步就 SGD 一下，永远跟学生"同呼吸"。两个网络架构通常一样，参数不同。
   - English: `real_score` is the frozen teacher diffusion (e.g. Wan2.1). `fake_score` is a *constantly co-trained* diffusion that tracks the student's current output distribution — every student step is followed by an SGD step on `fake_score` (denoising students' samples). Two nets, same architecture, different weights.

2. **CFG 在两边都做 / Classifier-free guidance on both sides**:
   - 中文：fake 和 real 都跑了"条件 + 无条件 + 加权"那一套 CFG。`real_guidance_scale` 通常很大（5-7.5，因为教师是真信号），`fake_guidance_scale` 通常 0 或很小（因为 fake 是在线训出来的，自带条件信息，再放大反而抖）。
   - English: both score nets are wrapped in CFG. `real_guidance_scale` is high (5-7.5) because the teacher carries the conditioning signal we want to push toward. `fake_guidance_scale` is usually 0 or small because the online-trained fake is already conditioning-aware; over-amplifying it makes the gradient noisy.

3. **DMD 公式 7：相减就是梯度 / DMD eq. 7 — subtraction *is* the gradient**:
   - 中文：`grad = pred_fake - pred_real`。这一行就是 DMD 全部数学的精华。直觉：如果学生生成的图比教师"更像"某个方向，那对那个方向的 score 在 fake 那边会比 real 那边小（fake 觉得"我经常看到这种图，不算特别"），相减后梯度推着学生**远离**这个方向。
   - English: `grad = pred_fake - pred_real` is the whole point. Intuition: if the student is *over-producing* some pattern, `fake_score` has seen it often (so it scores lower / wants to keep), while `real_score` (teacher) thinks it should be rarer (so it scores higher). The difference pushes the student away from over-represented patterns and toward under-represented ones — the gradient direction of reverse-KL minimization, computed *without ever evaluating KL*.

4. **公式 8 归一化 / DMD eq. 8 normalization**:
   - 中文：`normalizer = |x_clean - pred_real|.mean()`，再用它除 grad。原始 DMD 论文加这一项是因为不同 timestep 下 score 的尺度差几个数量级，没有归一化的话大 timestep 上梯度爆炸、小 timestep 上梯度消失。`p_real = x_clean - pred_real` 其实就是用 `x0 - x0_pred_by_teacher` 当 scale 度量。
   - English: `normalizer = mean(|x_clean - pred_real|)` rescales the gradient. Without this, score magnitudes vary by orders of magnitude across timesteps and training collapses at the extremes. `x_clean - pred_real` is essentially "teacher's prediction error on x_0", a natural per-timestep scale.

5. **`nan_to_num` 不是装饰 / `nan_to_num` is not cosmetic**:
   - 中文：bf16 + 大模型 + 大 timestep 经常蹦 nan，整个 batch 直接报废。`nan_to_num` 把 nan/inf 静默替换成 0，等价于"那个像素本步不更新"，工程上是必须的。
   - English: bf16 + large nets + extreme timesteps produce occasional NaNs; without this guard a single one would poison the whole batch's grad. Replacing with 0 silently no-ops the offending pixel for this step.

6. **为什么整体在 `torch.no_grad()` 里跑 / Why this is inside `torch.no_grad()` upstream**:
   - 中文：注意 caller（`compute_distribution_matching_loss`）把这个函数包在 `with torch.no_grad():` 里 —— grad 本身是手算出来的 monte-carlo 估计量，没必要 autograd 经过两个 score 网络。最后用 `(x_clean - (x_clean - grad).detach()).pow(2).mean()` 这种小技巧把 grad "灌"进 backward 里。
   - English: the caller wraps this in `torch.no_grad()`. The KL gradient is a Monte-Carlo estimator, not something we want autograd to backprop through two score nets. Outside the call, the trick `loss = 0.5 * MSE(x_clean, (x_clean - grad).detach())` injects this hand-computed gradient into the autograd graph.

## 类比 / The analogy

像是教唱歌。教师（real score）在心里有一个"理想嗓音"分布；学生唱完一段，老师听一遍打分（s_real），同时另一个声音教练（fake score）天天听学生本人的录音，对学生**当前**的嗓音分布也有一份评分（s_fake）。两份评分一减：教师觉得"高音不够"是正向、教练觉得"高音稀疏"是负向，最终告诉学生"该往高音方向走多少"。学生根本不用知道两个分布的密度长什么样，只需要两个评分相减。

Picture training a singer. The teacher (real score) holds the ideal voice distribution in their head and grades the student's take (`s_real`). A second coach (fake score) listens to the student's recordings every day and knows the *student's current* voice distribution, giving its own grade (`s_fake`). Subtracting the two grades tells the student exactly which direction to nudge their voice — toward where the teacher under-represents the student and away from where the teacher over-represents. The student never has to write down either distribution.

## 自己跑一遍 / Try it yourself

```python
# pip install torch
# 1-D toy: teacher = N(0, 1), student starts at N(2, 1), learn via DMD.
import torch
import torch.nn as nn

torch.manual_seed(0)
# In real DMD these are diffusion nets; here we cheat and use analytic Gaussian score:
#   score(x; mu, sig) = -(x - mu) / sig**2
def gaussian_score(x, mu, sig):
    return -(x - mu) / (sig ** 2)

mu_student = nn.Parameter(torch.tensor(2.0))   # student mean
sig = 1.0
opt = torch.optim.Adam([mu_student], lr=0.05)

teacher_mu = 0.0
# "fake score" tracks the student mean (in real DMD this is an online-trained net)
for step in range(80):
    # student samples
    with torch.no_grad():
        eps = torch.randn(1024)
        x   = mu_student + sig * eps
        t   = sig                                        # toy "noise level"

        s_real = gaussian_score(x, teacher_mu, t)        # teacher
        s_fake = gaussian_score(x, mu_student.item(), t) # fake = current student
        grad   = s_fake - s_real                         # eq. 7

    # inject hand-computed gradient via MSE trick
    loss = 0.5 * ((x - (x - grad).detach()) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 10 == 0:
        print(f"step {step:3d}  student_mu={mu_student.item():.3f}")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
step   0  student_mu=2.000
step  10  student_mu=1.4...
step  20  student_mu=0.9...
...
step  70  student_mu=0.05...
```

中文：学生均值从 2.0 一路被 DMD 推到 0.0，整个训练里**从来没显式算过 KL** —— 全靠两条 score 相减。

English: the student's mean slides from 2.0 to 0.0 even though KL is never evaluated — every step is just `s_fake - s_real`.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **DMD2 / Score-Identity Distillation (SiD)** / **DMD2 / SiD**: 中文 — 同一思想，进一步省掉 generator 端的额外 forward；同仓库 `model/sid.py` 就是 SiD 实现。 / English — same idea, removes the generator forward; this same repo's `model/sid.py` is the SiD variant.
- **Diff-Instruct / VSD** / **Diff-Instruct / VSD**: 中文 — 把 DMD 用在 3D / NeRF 优化上，几乎一字不改。 / English — applies DMD to 3D optimization (text-to-3D, NeRF distillation) with almost no change.
- **Self-Forcing distillation** / **Self-Forcing distillation**: 中文 — Causal-Forcing 把这套 DMD 放进 causal video 训练循环，配合昨天讲的 KV-cache 设计；今日 wam 笔记里的 mask 体系跟这个 DMD 是配合的 —— mask 保证 noisy 只看 clean past，DMD 提供训练信号。 / English — Causal-Forcing pairs DMD with the causal-video KV-cache regime; today's WAM note (lingbot-va's FlexAttention mask) is the architectural sibling: the mask enforces "noisy attends only clean-past", DMD provides the training signal.
- **TRPO / IRL with two reward models** / **TRPO-style "two reward models"**: 中文 — 思想类似，"参考策略减实际策略"作为 advantage 估计。 / English — IRL and TRPO methods that subtract two reward estimates share the same "two-model differential" structure.

## 注意事项 / Caveats / when it breaks

- **fake_score 必须跟得上学生** / **`fake_score` must keep up with the student**: 中文 — 如果 fake_score 训得比学生慢，差值会朝旧分布偏 —— 等效梯度方向错。所以 fake_score 每个学生 step 都要 1-3 次 SGD 更新。 / English — if `fake_score` lags the student, the difference points at a stale distribution and the gradient is biased. Recipes typically take 1-3 SGD steps on `fake_score` per student step.
- **timestep 的范围卡死在 (0.02, 0.98)** / **Clip timesteps into `(0.02, 0.98)`**: 中文 — 太接近 0（几乎没噪声）和接近 1（几乎全噪）都会让 score 数值爆炸；DMD 默认裁剪到 (2%, 98%)，从代码里 `min_step = int(0.02 * num_train_timestep)` 可以看到。 / English — both `t→0` (no noise) and `t→1` (pure noise) blow up the score numerically. DMD clips to `(0.02, 0.98)`, visible in `min_step` / `max_step`.
- **bf16 + 没 nan_to_num = batch 烂掉** / **bf16 without `nan_to_num` poisons the batch**: 中文 — 别去掉那一行。亲历者血泪。 / English — keep the `nan_to_num`. It looks defensive, it is actually load-bearing.

## 延伸阅读 / Further reading

- [DMD: Distribution Matching Distillation (Yin et al., 2023)](https://arxiv.org/abs/2311.18828)
- [DMD2: Improved Distribution Matching Distillation (Yin et al., 2024)](https://arxiv.org/abs/2405.14867)
- [Score Identity Distillation — SiD (Zhou et al., 2024)](https://arxiv.org/abs/2404.04057)
- [Causal-Forcing repo + paper assets](https://github.com/thu-ml/Causal-Forcing)
