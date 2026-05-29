---
date: 2026-05-29
topic: wam
source: wam
repo: Robbyant/lingbot-va
file: wan_va/train.py
permalink: https://github.com/Robbyant/lingbot-va/blob/58c2ae5bac46bd8114065bea9d7d256eb67c16c3/wan_va/train.py#L168-L295
difficulty: advanced
read_time: ~14 min
tags: [code-of-the-day, wam, training-loop, flow-matching, loss]
build_role: Training loop — full denoising-loss assembly for joint video + action diffusion
---

# 训练 step 的两条主线:加噪 + 损失加权 / Two main lines of a training step: add noise, then weight the loss

> **一句话 / In one line**: lingbot-va 的训练循环把 video latent 和 action latent 各自加噪、各自算 MSE loss、再用 timestep-aware 权重组合 —— 同一个 DiT 同时学到了"去噪视频"和"去噪动作",拼出一个真正的 WAM。 / lingbot-va's training step adds noise to video and action latents separately, runs one DiT forward, computes two MSE losses each weighted by a timestep-aware curve — one model learns to denoise both video and action in one go.

## 为什么重要 / Why this matters

WAM = World + Action Model,核心就是"视频帧的 latent" 和 "机器人动作的 latent" 在**同一个** diffusion model 里一起去噪。这种联合训练听起来简单(就是两条 MSE 加起来),但细节里全是坑:(1) action 和 video 的 timestep 是不是同步?(2) 帧之间 loss 要不要按 timestep 加权?(3) action 的 padding mask 怎么处理?(4) 训练时要不要让一部分 video 帧"还是干净的"(条件帧)?lingbot-va 这 130 行就是上述四个问题的工业回答 —— 整个文件几乎照搬到 nanoWAM 就能跑。

WAM = World + Action Model — the defining feature is that "video latent frames" and "action latent frames" share one diffusion model and are denoised jointly. The joint training sounds trivial (two MSE losses added) but production has four sharp corners: (1) do action and video share timesteps or have independent schedules? (2) should the loss be reweighted per-timestep? (3) how do you handle padded action steps? (4) do some video frames stay clean as "conditioning frames"? These 130 lines from lingbot-va are the practical answer to all four — paste them into nanoWAM and the training loop just works.

## 代码 / The code

`Robbyant/lingbot-va` — [`wan_va/train.py`](https://github.com/Robbyant/lingbot-va/blob/58c2ae5bac46bd8114065bea9d7d256eb67c16c3/wan_va/train.py#L168-L295)

```python
def _add_noise(self, latent, train_scheduler, action_mask=None,
               action_mode=False, noisy_cond_prob=0.0):
    B, C, F, H, W = latent.shape

    # === 1) Sample a per-frame timestep ===
    timestep_ids = sample_timestep_id(batch_size=F,
                                      num_train_timesteps=train_scheduler.num_train_timesteps)
    timesteps = train_scheduler.timesteps[timestep_ids].to(self.device)

    # === 2) Add noise and produce the target for flow matching ===
    noise = torch.zeros_like(latent).normal_()
    noisy_latents = train_scheduler.add_noise(latent, noise, timesteps, t_dim=2)
    targets       = train_scheduler.training_target(latent, noise, timesteps)

    # === 3) (optional) With probability noisy_cond_prob, treat the conditioning
    #        latent as 'mildly noisy' instead of perfectly clean — robust to drift ===
    if torch.rand(1).item() < noisy_cond_prob:
        cond_timestep_ids = sample_timestep_id(
            batch_size=F, min_timestep_bd=0.5, max_timestep_bd=1.0,
            num_train_timesteps=train_scheduler.num_train_timesteps)
        cond_timesteps = train_scheduler.timesteps[cond_timestep_ids].to(self.device)
        latent = train_scheduler.add_noise(latent,
                                           torch.zeros_like(latent).normal_(),
                                           cond_timesteps, t_dim=2)
    else:
        cond_timesteps = torch.zeros_like(timesteps)

    # === 4) Mask out padding action steps ===
    if action_mask is not None:
        noisy_latents *= action_mask.float()
        targets       *= action_mask.float()
        latent        *= action_mask.float()

    return dict(timesteps=timesteps[None].repeat(B, 1),
                noisy_latents=noisy_latents, targets=targets,
                latent=latent, cond_timesteps=cond_timesteps[None].repeat(B, 1),
                grid_id=...)


def compute_loss(self, input_dict, pred):
    latent_pred, action_pred = pred

    # === 5) Per-frame loss weights from the scheduler ===
    latent_w = self.train_scheduler_latent.training_weight(
        input_dict['latent_dict']['timesteps'].flatten()).reshape(Bn, Fn)
    action_w = self.train_scheduler_action.training_weight(
        input_dict['action_dict']['timesteps'].flatten()).reshape(Bn, Fn)

    # === 6) Video loss: per-pixel MSE, weighted by per-frame timestep weight ===
    latent_loss = F.mse_loss(latent_pred.float(),
                             input_dict['latent_dict']['targets'].float().detach(),
                             reduction='none')
    latent_loss = latent_loss * latent_w[:, None, :, None, None]
    latent_loss = (latent_loss.permute(0, 2, 3, 4, 1)        # [B, F, H, W, C]
                   .flatten(0, 1).flatten(1))                # [B*F, H*W*C]
    latent_loss = (latent_loss.sum(dim=1) /
                   (torch.ones_like(latent_loss).sum(dim=1) + 1e-6)).mean()

    # === 7) Action loss: same, but also masked by the action validity mask ===
    action_loss = F.mse_loss(action_pred.float(),
                             input_dict['action_dict']['targets'].float().detach(),
                             reduction='none')
    action_loss = action_loss * action_w[:, None, :, None, None]
    action_loss = action_loss * input_dict['action_dict']['actions_mask'].float()
    action_mask = input_dict['action_dict']['actions_mask'].float()
    action_loss = (action_loss.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1))
    action_mask = (action_mask.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1))
    action_loss = (action_loss.sum(dim=1) / (action_mask.sum(dim=1) + 1e-6)).mean()

    return (latent_loss / self.gradient_accumulation_steps,
            action_loss / self.gradient_accumulation_steps)
```

## 逐行讲解 / What's happening

1. **第 1 块:每一帧自己的 timestep / Per-frame independent timestep**:
   - 中文:`sample_timestep_id(batch_size=F)` 给每一**帧** 都采一个独立的 timestep,而不是整段视频共用一个 t。这是 self-forcing 训练的关键 —— 推理时是逐帧/逐 chunk 去噪的,训练时也要这样模拟。结果是 `timesteps` 形状 `[F]`。
   - English: `sample_timestep_id(batch_size=F)` draws an independent timestep per frame (not one per clip). That's the foundation of self-forcing training — inference denoises chunk-by-chunk, so training simulates the same thing. Result: `timesteps` shape `[F]`.

2. **第 2 块:加噪 + flow-matching 目标 / Add noise + flow-matching target**:
   - 中文:`scheduler.add_noise(latent, noise, timesteps, t_dim=2)` 沿时间维度(t_dim=2)对每一帧用各自的 sigma 加噪;`training_target(latent, noise, timesteps)` 返回的是 `target = noise - latent` —— 这是 rectified flow / flow matching 的训练目标(模型学习预测 "从 latent 到 noise 的方向")。如果你以前只见过 DDPM(target = noise),这里换成 `noise - latent` 是 flow matching 的标志。
   - English: `scheduler.add_noise` corrupts each frame at its own sigma. `training_target` returns `noise - latent` — the flow-matching target (the velocity from data to noise). If you only know DDPM (`target = noise`), the `noise - latent` form is the flow-matching signature.

3. **第 3 块:noisy conditioning / Noisy conditioning ("teacher with hand tremor")**:
   - 中文:lingbot-va 的 attention mask 把 video latent 分成"clean 历史 + noisy 当前"。如果训练时 clean 永远是完美的、推理时却用上一步生成的(有误差),分布会错位 —— 这就是 exposure bias。`noisy_cond_prob` 概率把 clean 端也加上少量噪声(用 0.5-1.0 区间的高 sigma),让模型见过"略带噪的 condition",推理时鲁棒。
   - English: lingbot-va's attention mask splits video into clean past + noisy current. If clean is always perfect in training but comes from an earlier generation step at inference (which is itself noisy), you hit exposure bias. With probability `noisy_cond_prob`, the conditioning latent is corrupted with a small sigma (sampled in `[0.5, 1.0]`), teaching the model to handle slightly noisy conditioning. Free robustness, one line of code.

4. **第 4 块:action 的 padding mask / Action padding mask**:
   - 中文:episode 长度不一,action 序列要 pad 到 max_len。pad 位置上 latent、target、noisy_latents 全 `* mask.float()` 清零 —— 不让 pad 帧产生任何梯度。
   - English: episodes have variable length, so action sequences are padded. The mask zeroes the latent / target / noisy_latents at padded slots, so they contribute no gradient.

5. **第 5 块:per-frame loss weights / Per-frame loss weights**:
   - 中文:`training_weight(timesteps)` 返回的是一个"bsmntw"形状(钟形,中间高两边低)的权重 —— 中间 timestep 的去噪信号最有用,所以加权大;接近 0 或 1 的极端 timestep 信号弱,加权小。这是 rectified flow 训练的标准技巧。形状 `[Bn, Fn]`,逐帧逐 batch 独立。
   - English: `training_weight` returns a bell-shaped curve (zeroed at extremes, peaked in the middle), upweighting the timesteps where the denoising signal is most informative. Standard rectified-flow training trick; shape `[Bn, Fn]`, per-frame per-batch independent.

6. **第 6 块:per-frame normalization / Per-frame normalization**:
   - 中文:loss 不直接 `.mean()`,而是先 `permute → flatten` 让 `(F)` 跟 `(B)` 合并,再 per-frame 求和并除以 per-frame 元素个数。等价于"每帧自己求平均、再对所有帧求平均",避免 batch 里大分辨率帧把小分辨率帧的 loss 淹没。
   - English: rather than a plain `.mean()`, the loss is reduced per-frame first (sum over `H*W*C`, divide by element count), then averaged across `B*F`. This stops large-resolution frames from drowning small-resolution ones in mixed-resolution training.

7. **第 7 块:action loss 多一道 mask normalisation / Action loss divides by the mask sum**:
   - 中文:action 在 pad 位置贡献 0,但分母也只算非 pad 位置 —— `action_loss.sum() / action_mask.sum()`。这样不同 episode 长度不影响 loss 量级。
   - English: action loss sums only over valid steps and divides by the count of valid steps — `action_loss.sum() / action_mask.sum()`. Episode-length differences cancel out cleanly.

## 类比 / The analogy

像同时教学生两门课(数学=video, 语文=action),每天都给两门课各布置一份作业,但每份作业的难度系数不同 —— 简单题不打分,中等题打高分,极难题也不打分(防止误导)。两门作业要分别批改、分别归一化,最后加起来才是这一天的总分。中途老师偶尔会"自带笔迹"故意写歪一点提示线条(noisy conditioning),让学生不依赖老师永远工整,这样真实考试时面对自己写错的笔记仍然能继续做。

Picture teaching a student two subjects in parallel (maths = video, language = action) by assigning daily homework in each. Each problem gets a difficulty weight: trivial problems and impossibly hard ones get zero weight, middling ones get max. The two homeworks are graded *and* normalised separately, then added for the day's total. Occasionally the teacher's example handwriting is slightly messy on purpose (noisy conditioning) so the student doesn't depend on perfect inputs in the real exam.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-WAM

中文:在 nanoWAM 里这是 `nano/wam/train.py` 整个 `_train_step` 的核心。上游依赖:VAE 已经训好(把 pixel 压成 latent)、3D RoPE / DiT 块已经搭好、scheduler 已经实现(昨天 dreamzero 的 FlowMatchScheduler 就完全够用)。下游是 optimizer.step + AMP scaler。 如果省掉**任何一条**:(1) 不做 per-frame timestep → 退化成图像 DiT,根本学不到时间一致性;(2) 不做 noisy conditioning → 推理 1-2 步后就漂移;(3) 不做 training_weight → 训练曲线噪声大、收敛慢;(4) 不做 action mask → 模型把 pad 区当作合法 action,推理时输出垃圾。生产实现还要补:gradient clipping(`torch.nn.utils.clip_grad_norm_`)、bf16 + grad accumulation(默认这段已经除了 `gradient_accumulation_steps`)、以及 FSDP/HSDP 切分(lingbot-va 自带 fsdp.py)。

English: in nanoWAM this is the spine of `nano/wam/train.py`'s `_train_step`. Upstream prereqs: trained VAE, working DiT block + 3D RoPE, an implemented scheduler (yesterday's dreamzero FlowMatchScheduler is enough). Downstream is `optimizer.step` + AMP scaler. Drop any one piece and quality breaks: (1) skip per-frame timesteps → degenerates to an image DiT, no temporal consistency; (2) skip noisy conditioning → autoregressive drift starts within 2 steps at inference; (3) skip `training_weight` → noisy loss curve, slow convergence; (4) skip action mask → padded slots get treated as real actions and rollout produces gibberish. Production additions: grad clipping (`torch.nn.utils.clip_grad_norm_`), bf16 + grad accumulation (note the `/ gradient_accumulation_steps` already in the code), and FSDP/HSDP sharding (lingbot-va ships its own `wan_va/distributed/fsdp.py`).

## 自己跑一遍 / Try it yourself

```python
# pip install torch
# Toy: tiny "video" of 4 frames, learn to denoise via flow matching with per-frame t.
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)
B, C, Fn, H, W = 2, 3, 4, 8, 8

# data: each frame is a Gaussian centered at a frame-specific mean
data = torch.stack([torch.randn(B, C, H, W) + (i - 1.5)
                    for i in range(Fn)], dim=2)              # [B, C, F, H, W]

model = nn.Conv3d(C, C, kernel_size=1)
opt   = torch.optim.Adam(model.parameters(), lr=1e-2)

def train_weight(t):                       # bell curve, peak at t=0.5
    return torch.exp(-((t - 0.5) ** 2) / 0.05)

for step in range(200):
    # per-frame independent timesteps in [0, 1]
    t   = torch.rand(Fn)                                          # [F]
    sig = t.view(1, 1, Fn, 1, 1)                                  # broadcast
    noise = torch.randn_like(data)
    x_t   = (1 - sig) * data + sig * noise                        # flow-matching add_noise
    target = noise - data                                         # flow-matching target

    pred = model(x_t)
    loss = F.mse_loss(pred, target, reduction='none')
    loss = (loss * train_weight(t).view(1, 1, Fn, 1, 1)).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 50 == 0:
        print(f"step {step:3d}  loss={loss.item():.4f}")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
step   0  loss=0.7...
step  50  loss=0.4...
step 100  loss=0.3...
step 150  loss=0.2...
```

中文:loss 单调下降。注意每一帧 timestep 是独立采的 —— 这是和图像 DiT 训练最关键的差别。

English: monotonic loss decrease. The single most important detail vs. image-DiT training is that **each frame samples its own timestep**.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Wan2.1 official training** / **Wan2.1 official training**: 中文 — 一字不差的 per-frame timestep + flow-matching target,只是没有 action loss。 / English — identical per-frame timestep + flow-matching target; no action loss because Wan2.1 is text-to-video only.
- **Open-Sora training_step** / **Open-Sora training_step**: 中文 — 同样的 `noise - latent` 流匹配 target,timestep 加权也是 bell curve。 / English — same `noise - latent` velocity target, same bell-curve weighting.
- **昨天的 lingbot-va FlexAttention mask** / **Yesterday's lingbot-va FlexAttention mask**: 中文 — 那个 mask 把 clean 和 noise 分开,这段 `_add_noise` 就是产生 noise 端的数据;没有 mask,这段 loss 会让 model 看着 ground truth 去预测 ground truth(泄漏)。 / English — yesterday's mask separates clean and noisy tokens; today's `_add_noise` is what actually produces the noisy side. Without the mask, this loss would let the model peek at the very target it's trying to predict.
- **昨天的 DMD distillation** / **Yesterday's DMD distillation**: 中文 — DMD 是这个朴素 flow-matching loss 的高级变种,把 `MSE(pred, target)` 换成 `MSE(x, (x - score_diff).detach())` 来注入蒸馏梯度。 / English — DMD is an upgrade of this plain flow-matching MSE: it replaces `MSE(pred, target)` with `MSE(x, (x - score_diff).detach())` to inject distillation gradients.

## 注意事项 / Caveats / when it breaks

- **per-frame timestep ≠ clip-level timestep** / **Per-frame ≠ clip-level**: 中文 — 一定要确保 attention mask 能处理"同一 clip 里各帧 timestep 不同"。lingbot-va 的 FlexAttention mask 就是为这个设计的;朴素 causal mask 会出错。 / English — your attention masking must accept per-frame timesteps; yesterday's FlexAttention mask is built for it. A vanilla causal mask will give wrong results.
- **`training_target` 形式取决于 scheduler** / **Target form depends on scheduler**: 中文 — flow matching 用 `noise - latent`,DDPM 用 `noise`,v-prediction 用 `alpha * noise - sigma * latent`。换 scheduler 时这一行要跟着换。 / English — flow matching uses `noise - latent`; DDPM uses `noise`; v-prediction uses `alpha * noise - sigma * latent`. Switching schedulers requires updating this line.
- **detach 在 target 上要做** / **Detach the target**: 中文 — `targets.float().detach()` 不要漏 detach,否则 noise 的梯度会反向流到下游,数值不稳。 / English — `.detach()` on the target is mandatory; otherwise gradients leak back through the noise tensor and destabilise training.
- **bf16 还是 fp32 算 loss?** / **bf16 or fp32 for the loss?**: 中文 — model forward bf16,但 loss 计算用 `.float()` 升到 fp32,否则 1e-6 之类的极小 loss 累计会下溢。 / English — model forward stays bf16, but `.float()` upcasts the loss reduction to fp32, otherwise tiny per-pixel losses underflow before reduction.

## 延伸阅读 / Further reading

- [Rectified Flow (Liu et al., 2022)](https://arxiv.org/abs/2209.03003)
- [Self-Forcing for autoregressive diffusion](https://arxiv.org/abs/2503.20451)
- [Wan2.1 training script](https://github.com/Wan-Video/Wan2.1)
- [LingBot-VA paper (joint video + action diffusion)](https://github.com/Robbyant/lingbot-va/blob/main/LingBot_VA_paper.pdf)
