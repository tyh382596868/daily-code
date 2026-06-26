---
date: 2026-06-26
topic: wam
source: wam
repo: facebookresearch/DiT
file: train.py
permalink: https://github.com/facebookresearch/DiT/blob/58fe9c286baa9fcd2d08278a57b178b0650c3eff/train.py#L140-L210
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, wam, training-loop, ema, diffusion, dit, latent-diffusion, curriculum]
build_role: training-loop — DiT 极简实现 (cross-repo variant vs FastWAM/Lingbot-va)
---

# DiT 训练循环五步法：VAE 编码 → 随机时间步 → 扩散损失 → backward → EMA 更新 / DiT's 5-Step Training Loop: VAE Encode → Random Timestep → Diffusion Loss → Backward → EMA Update

> **一句话 / In one line**: DiT 的训练循环只需五步：冻结 VAE 编码图像 → 均匀采样随机时间步 → 调用 `diffusion.training_losses` 计算 MSE 预测损失 → backward + AdamW → EMA 更新，EMA 模型始终在 eval 模式用于推理。 / DiT's training loop is just five steps: frozen VAE encodes images → sample a random timestep uniformly → call `diffusion.training_losses` for MSE denoising loss → backward + AdamW → EMA update; the EMA model always stays in eval mode for inference.

## 为什么重要 / Why this matters

理解一个扩散模型的训练循环，是搭建 nanoWAM（World Action Model）的最核心一课。DiT 的 `train.py` 是目前最清晰、最小依赖的参考实现：整个内层循环只有 10 行实质代码，却包含了所有关键组件的正确摆放——冻结的 VAE 编码器、随机时间步采样、`training_losses` 抽象（把加噪 + 前向传播 + MSE 封装进一行）、EMA 的 eval 模式。

这个循环结构对 WAM 来说几乎可以直接复用：把"图像 latent"换成"视频帧 latent + 机器人动作 embedding"，把"类别标签 y"换成"动作条件 c_a"，其余保持不变。所以理解 DiT 等于理解 WAM 训练循环的 80%。

Understanding a diffusion model's training loop is the most critical single lesson for building a nanoWAM (World Action Model). DiT's `train.py` is the clearest, most dependency-light reference: the inner loop is only 10 lines of real code, yet every key component is placed correctly — frozen VAE encoder, uniform timestep sampling, the `training_losses` abstraction (noise + forward + MSE in one call), EMA in eval mode.

This loop structure transfers almost directly to WAM: replace "image latents" with "video frame latents + robot action embeddings", replace class label `y` with action condition `c_a`, and keep everything else identical. Understanding DiT means understanding 80% of a WAM training loop.

## 代码 / The code

`facebookresearch/DiT` — [`train.py`](https://github.com/facebookresearch/DiT/blob/58fe9c286baa9fcd2d08278a57b178b0650c3eff/train.py#L140-L210)

```python
# Prepare models for training:
update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
model.train()   # important! This enables embedding dropout for classifier-free guidance
ema.eval()      # EMA model should always be in eval mode

# Variables for monitoring/logging purposes:
train_steps = 0
log_steps = 0
running_loss = 0
start_time = time()

logger.info(f"Training for {args.epochs} epochs...")
for epoch in range(args.epochs):
    sampler.set_epoch(epoch)
    logger.info(f"Beginning epoch {epoch}...")
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            x = vae.encode(x).latent_dist.sample().mul_(0.18215)
        t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
        model_kwargs = dict(y=y)
        loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
        loss = loss_dict["loss"].mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        update_ema(ema, model.module)

        # Log loss values:
        running_loss += loss.item()
        log_steps += 1
        train_steps += 1
        if train_steps % args.log_every == 0:
            torch.cuda.synchronize()
            end_time = time()
            steps_per_sec = log_steps / (end_time - start_time)
            avg_loss = torch.tensor(running_loss / log_steps, device=device)
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss.item() / dist.get_world_size()
            logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Steps/Sec: {steps_per_sec:.2f}")
            running_loss = 0
            log_steps = 0
            start_time = time()

        # Save DiT checkpoint:
        if train_steps % args.ckpt_every == 0 and train_steps > 0:
            if rank == 0:
                checkpoint = {
                    "model": model.module.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args,
                }
                torch.save(checkpoint, f"{checkpoint_dir}/{train_steps:07d}.pt")
            dist.barrier()
```

## 逐行讲解 / What's happening

1. **`model.train()` vs `ema.eval()`**
   - 中文: 这两行的分工至关重要。`model.train()` 开启 classifier-free guidance（CFG）所需的 dropout：DiT 训练时会随机用空条件替换真实标签（`y = None`），以让模型同时学会有条件和无条件生成。`ema.eval()` 固定 EMA 模型不做 dropout，保证推理时有稳定的参数——EMA 模型才是最终发布的模型。
   - English: Critical split. `model.train()` enables the dropout needed for classifier-free guidance (CFG): DiT randomly replaces true labels with a null embedding during training so the model learns both conditional and unconditional generation. `ema.eval()` keeps the EMA model dropout-free for stable inference — the EMA model is the one you actually ship.

2. **`x = vae.encode(x).latent_dist.sample().mul_(0.18215)`**
   - 中文: 三件事合在一行：(1) 冻结 VAE 把像素图像编码成潜变量分布；(2) 采样一个具体的潜变量；(3) 乘以 `0.18215`（SD VAE 的经验标准差倒数，使 latent 近似标准正态分布）。`torch.no_grad()` 确保梯度不流入冻结的 VAE。
   - English: Three operations in one line: (1) frozen VAE encodes pixel images to a latent distribution; (2) sample one latent; (3) multiply by `0.18215` — the empirical reciprocal standard deviation of SD's VAE latent space, making latents approximately standard-normal. `torch.no_grad()` prevents gradients from flowing into the frozen VAE.

3. **`t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)`**
   - 中文: 均匀采样随机时间步，每个 batch 里每个样本独立采一个 `t ∈ [0, T-1]`。这是 DDPM 的训练方式：每次 backward 只"看到"一个随机时间步的去噪任务，多次迭代后模型学会了所有时间步的去噪。
   - English: Uniformly sample one random timestep per example in the batch, independently. This is the DDPM training recipe: each backward only "sees" a single random timestep's denoising task; over many iterations the model learns all timesteps. Note: some diffusion variants (e.g. EDM, flow matching with skewed schedules) sample non-uniformly.

4. **`loss_dict = diffusion.training_losses(model, x, t, model_kwargs)`**
   - 中文: 这一行封装了扩散损失的核心：给 `x`（干净 latent）加 `t` 对应的高斯噪声，把含噪 latent 和条件 `y` 送入 DiT 模型，预测噪声（ε-prediction）或原始图像（x-prediction），计算 MSE 损失。`model_kwargs` 携带的 `y` 就是类别条件。
   - English: This line encapsulates the diffusion loss core: add Gaussian noise at level `t` to clean latent `x`, pass the noisy latent and condition `y` to the DiT, predict either the noise (ε-prediction) or the clean image (x-prediction), compute MSE. `model_kwargs` carries `y` as the class condition.

5. **`update_ema(ema, model.module)` — EMA 永远在 `eval()` 模式**
   - 中文: EMA（Exponential Moving Average）是一个影子权重：`ema_param = decay * ema_param + (1-decay) * model_param`，默认 `decay=0.9999`。EMA 权重平滑掉了训练噪声，推理质量显著优于普通 checkpoint。DiT 的 `ema` 在整个训练过程中始终保持 `eval()` 模式——`model.train()` 切换不影响它。
   - English: EMA maintains a shadow copy of weights: `ema_param = decay * ema_param + (1-decay) * model_param`, default `decay=0.9999`. EMA weights smooth out training noise; inference quality is significantly better than a raw checkpoint. DiT's `ema` stays in `eval()` mode throughout training — `model.train()` doesn't affect it.

6. **Checkpoint 保存格式：`{"model": ..., "ema": ..., "opt": ..., "args": ...}`**
   - 中文: 同时保存训练模型和 EMA 模型，以及优化器状态（断点续训必须）和 args（复现必须）。这是一个完整的 checkpoint，可以从任意 step 恢复训练。
   - English: Saves both the training model and EMA model, plus optimizer state (required for resuming) and args (required for exact reproducibility). This is a complete checkpoint that allows training to resume from any step.

## 类比 / The analogy

想象一个学徒厨师每天练习切菜。每次练习（step），助理随机叫出一个"难度等级"（随机时间步 t），学徒在那个难度下练切（前向传播 + 损失计算），然后根据错误改进刀法（backward + optimizer）。同时，有一本特殊的"经验总结笔记"（EMA 权重），每次练习后都悄悄平均更新，从不直接参与练习。考核时用的是这本笔记，而不是当天最新的手感——因为它更稳定，代表了长期积累的最佳状态，而不是一次偶然表现。

Imagine an apprentice chef practicing knife skills daily. Each session (step), an assistant calls out a random "difficulty level" (random timestep t); the apprentice practices at that difficulty (forward + loss), then corrects their technique based on errors (backward + optimizer). Meanwhile, a special "accumulated wisdom notebook" (EMA weights) is quietly averaged after each session and never participates in practice directly. The evaluation uses this notebook — not today's most recent edge — because it's more stable, representing a long-term best rather than a single lucky performance.

## 在 nanoWAM 中的位置 / Where this lives in your nanoWAM

这是 nanoWAM 课程的 **training-loop** 组件，是整个课程的最后一层，依赖：**vae-encoder-decoder**（冻结 VAE）、**dit-backbone**（DiT 模型）、**noise-scheduler**（`diffusion` 对象）、**action-conditioning**（`model_kwargs` 里的 `y` 对应 WAM 里的动作条件）。

**从 DiT 到 nanoWAM 的三个替换 / Three substitutions from DiT to nanoWAM**:

1. `x = vae.encode(image).latent_dist.sample() * 0.18215`
   → `x = video_vae.encode(video_frames).latent * video_scale`
   （视频帧序列的 VAE，scale 因子要重新测量）

2. `model_kwargs = dict(y=y)` （类别条件）
   → `model_kwargs = dict(y=y, action=action_embed)` （视频生成 + 机器人动作条件）

3. EMA + checkpoint 结构完全复用，加上 action encoder 的权重。

**上游 / Upstream**: DataLoader 提供 `(video_frames, robot_actions)` 对；冻结 VAE 把视频帧编码成 latent；action encoder 把动作序列嵌入为条件向量。
**本组件 / This component**: 整个训练循环——随机时间步采样 + `training_losses` + backward + EMA 更新。
**下游 / Downstream**: EMA 模型用于推理（DDIM/DPM-Solver 采样），给定初始帧和动作序列，生成未来 T 帧的视频预测。

This is the **training-loop** component of the nanoWAM curriculum — the final layer, depending on **vae-encoder-decoder** (frozen VAE), **dit-backbone** (DiT model), **noise-scheduler** (`diffusion` object), and **action-conditioning** (`y` in `model_kwargs` corresponds to action conditions in WAM).

**Upstream**: DataLoader provides `(video_frames, robot_actions)` pairs; frozen VAE encodes video to latents; action encoder embeds actions as conditioning vectors.
**This component**: The full training loop — random timestep sampling + `training_losses` + backward + EMA update.
**Downstream**: The EMA model is used for inference (DDIM/DPM-Solver sampling): given an initial frame and an action sequence, generate a video prediction of T future frames.

## 自己跑一遍 / Try it yourself

```python
import torch
import torch.nn as nn
import copy

# Minimal DiT-style training loop (no VAE, directly on latents)
class TinyDiT(nn.Module):
    def __init__(self, latent_dim=4, num_classes=10, hidden=64):
        super().__init__()
        self.embed = nn.Linear(latent_dim + num_classes + 1, hidden)
        self.net = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, latent_dim))
    def forward(self, x, t, y):
        cond = torch.cat([x, y, t.float().unsqueeze(1) / 1000], dim=-1)
        return self.net(self.embed(cond).relu())

def update_ema(ema, model, decay=0.9999):
    with torch.no_grad():
        for ep, mp in zip(ema.parameters(), model.parameters()):
            ep.mul_(decay).add_(mp, alpha=1 - decay)

model = TinyDiT()
ema = copy.deepcopy(model)
ema.eval()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
T = 1000  # num timesteps

for step in range(200):
    x_clean = torch.randn(8, 4)  # batch of latents
    y = torch.zeros(8, 10).scatter_(1, torch.randint(0, 10, (8,)).unsqueeze(1), 1.0)
    t = torch.randint(0, T, (8,))
    noise = torch.randn_like(x_clean)
    alpha = 1 - t.float() / T  # simple linear schedule
    x_noisy = alpha[:, None] * x_clean + (1 - alpha[:, None]) * noise
    noise_pred = model(x_noisy, t, y)
    loss = ((noise_pred - noise) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    update_ema(ema, model)
    if step % 50 == 0:
        print(f"step={step:3d}  loss={loss.item():.4f}")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
step=  0  loss=X.XXXX
step= 50  loss=X.XXXX
step=100  loss=X.XXXX
step=150  loss=X.XXXX
```

中文：注意 EMA 模型（`ema`）和训练模型（`model`）的参数从一开始就分叉，训练模型在 `opt.step()` 后立即跳变，而 EMA 在每次 `update_ema` 后缓慢收敛。这两个模型的参数差异在训练初期最大，后期随着 loss 稳定而缩小。

English: The EMA model (`ema`) and training model (`model`) diverge immediately from the start — the training model jumps on each `opt.step()` while the EMA converges slowly. The gap between them is largest early in training and shrinks as the loss stabilizes.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **FastWAM 训练循环** / **FastWAM training loop**: 与 DiT 结构完全相同，但 `y` 换成了动作序列 + 视频帧，`training_losses` 换成了 WAM 专用的 action-conditioned diffusion loss。（参见 2026-06-13 的 FastWAM Accelerate 训练循环笔记。） / Structurally identical to DiT, but `y` is replaced by action sequences + video frames, and `training_losses` is a WAM-specific action-conditioned diffusion loss.
- **Stable Diffusion / SDXL** / **Stable Diffusion / SDXL**: 同样的五步循环，但 VAE 尺度因子换成 SD 的 `0.18215`（SD1.5）或 `0.13025`（SDXL），EMA 在 SD 里被称为 `accelerator.unwrap_model(unet)` 的平滑版本。 / Same 5-step loop, but VAE scale factor differs (`0.18215` for SD1.5, `0.13025` for SDXL), and EMA is managed by Accelerate under a different alias.
- **EDM（Elucidated Diffusion）** / **EDM (Elucidated Diffusion)**: 时间步采样换成了 log-normal 分布而不是均匀分布，`training_losses` 内部用了 `sigma` 参数化而不是 `t` 参数化，其余结构完全相同。 / Timestep sampling switches from uniform to log-normal; `training_losses` uses `sigma` parameterization instead of `t`; everything else is identical.

## 注意事项 / Caveats / when it breaks

- **`update_ema(ema, model.module, decay=0)`（首次初始化）** / **`update_ema(ema, model.module, decay=0)` at init**: `decay=0` 把 EMA 直接复制成 `model` 的权重，消除了 DDP 初始化时可能的权重不同步。省掉这一步会导致多卡训练里 EMA 从不同起点出发，破坏 EMA 的语义。 / Setting `decay=0` copies model weights directly into EMA, fixing potential DDP weight desync at init. Omitting this causes EMA to start from a different initialization than the model on each rank, breaking its semantics.
- **EMA 的 `decay` 值** / **EMA `decay` value**: `decay=0.9999` 在 200K 步里才能看到明显效果；步数太少（如 10K 步）时 EMA 和 checkpoint 差距不大。选错 decay 会导致 EMA 要么过于滞后（decay 太高）要么等同于普通 checkpoint（decay 太低）。 / `decay=0.9999` shows meaningful smoothing only after ~200K steps; with too few steps (e.g. 10K) EMA barely differs from a regular checkpoint. Wrong decay causes EMA to lag too far behind (too high) or behave like a raw checkpoint (too low).
- **`0.18215` 常数是 SD VAE 专用的** / **`0.18215` is SD-VAE-specific**: 如果换用其他 VAE（如 CogVideoX 的 3D VAE，scale=`0.7`），必须重新测量 latent 的标准差并替换这个常数，否则扩散过程的信噪比会严重偏离设计目标。 / This constant is specific to the SD VAE. If using a different VAE (e.g. CogVideoX's 3D VAE with scale `0.7`), you must remeasure the latent std and replace this constant, or the signal-to-noise ratio of the diffusion process will be far off target.

## 延伸阅读 / Further reading

- [DiT paper: Scalable Diffusion Models with Transformers (arXiv:2212.09748)](https://arxiv.org/abs/2212.09748)
- [DDPM paper: Denoising Diffusion Probabilistic Models (arXiv:2006.11239)](https://arxiv.org/abs/2006.11239)
- [EDM paper: Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)
- [FastWAM Accelerate training loop (2026-06-13 note)](../../nano/wam/2026-06-13-fastwam-accelerate-training-loop.md)
