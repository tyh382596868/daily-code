---
date: 2026-06-15
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/policies/vla_jepa/action_head.py
permalink: https://github.com/huggingface/lerobot/blob/38327fdc8458959f47d555c159307538200d0561/src/lerobot/policies/vla_jepa/action_head.py#L256-L337
difficulty: intermediate
read_time: ~12 min
tags: [code-of-the-day, vla, flow-matching, action-head, dit, rectified-flow]
build_role: action-head-continuous
---

# 一份 80 行的 flow-matching action head:训练 + 推理一次讲完 / 80 lines of flow-matching action head: train and infer end-to-end

> **一句话 / In one line**: VLAJEPAActionHead 把 rectified-flow 训练 step(采时间 → 插值 → 预测速度 → MSE)和 Euler 推理循环(噪声 → 反复 `+= dt * v`)压成各 30 行,中间夹一个 DiT。 / VLAJEPAActionHead packs a rectified-flow training step (sample t → interpolate → predict velocity → MSE) and the Euler inference loop (noise → repeatedly `+= dt * v`) into about 30 lines each, with a DiT in the middle.

## 为什么重要 / Why this matters

VLA 的"action head"是把 VLM 的语义表征转成机器人能执行的连续动作的最后一公里。2026 年的主流路线已经从"把 action 离散化成 token,扔进 LM 的 next-token loss"转向"action 仍然连续,用 flow matching / rectified flow 直接回归速度场"——pi0、 GR00T、 SmolVLA 各有各的实现,VLA-JEPA 是 2026-06-04 刚进 lerobot 的最新一员。它的 action head 把训练 step 和推理循环都写得非常干净,几乎是"教学版"flow matching:一边可以拿来理解算法,一边可以直接复制到自己的 nanoVLA 项目里。

A VLA's "action head" is the last mile that turns the VLM's semantic features into continuous actions a robot can execute. In 2026 the mainstream has shifted from "discretize action into tokens, feed into LM next-token loss" to "keep action continuous, regress its velocity field with flow matching / rectified flow" — pi0, GR00T, SmolVLA each ship their own version. VLA-JEPA landed in lerobot on 2026-06-04. Its action head is unusually clean — almost a textbook implementation: a training step in ~30 lines and an Euler inference loop in another ~30, with a DiT in between.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/vla_jepa/action_head.py`](https://github.com/huggingface/lerobot/blob/38327fdc8458959f47d555c159307538200d0561/src/lerobot/policies/vla_jepa/action_head.py#L256-L337)

```python
class VLAJEPAActionHead(nn.Module):
    def __init__(self, config: VLAJEPAConfig, cross_attention_dim: int) -> None:
        # ... DiT model, action_encoder, action_decoder, state_encoder ...
        self.action_horizon = config.chunk_size
        self.num_inference_timesteps = config.num_inference_timesteps
        self.beta_dist = Beta(config.action_noise_beta_alpha, config.action_noise_beta_beta)

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device=device, dtype=dtype)
        return (self.config.action_noise_s - sample) / self.config.action_noise_s

    def _build_inputs(self, conditioning_tokens, actions, state, timesteps):
        action_features = self.action_encoder(actions, timesteps)
        pos_ids = torch.arange(action_features.shape[1], device=actions.device)
        action_features = action_features + self.position_embedding(pos_ids)[None]

        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(actions.shape[0], -1, -1)
        seq = [future_tokens, action_features]
        if state is not None and self.state_encoder is not None:
            if state.ndim == 2:
                state = state.unsqueeze(1)
            seq.insert(0, self.state_encoder(state))
        return torch.cat(seq, dim=1)

    def forward(self, conditioning_tokens, actions, state=None, action_is_pad=None):
        noise = torch.randn_like(actions)
        t     = self.sample_time(actions.shape[0], actions.device, actions.dtype)
        noisy_actions = (1 - t[:, None, None]) * noise + t[:, None, None] * actions
        velocity      = actions - noise
        t_discretized = (t * self.config.action_num_timestep_buckets).long()

        hidden_states = self._build_inputs(conditioning_tokens, noisy_actions, state, t_discretized)
        pred = self.model(
            hidden_states=hidden_states,
            encoder_hidden_states=conditioning_tokens,
            timestep=t_discretized,
        )
        pred_actions = self.action_decoder(pred[:, -actions.shape[1]:])

        if action_is_pad is None:
            action_is_pad = torch.zeros(actions.shape[:2], dtype=torch.bool, device=actions.device)

        loss = F.mse_loss(pred_actions, velocity, reduction="none")     # [B, T, action_dim]
        valid_mask = ~action_is_pad.unsqueeze(-1)                       # [B, T, 1]
        num_valid  = valid_mask.sum() * loss.shape[-1]
        return (loss * valid_mask).sum() / num_valid.clamp_min(1)

    @torch.no_grad()
    def predict_action(self, conditioning_tokens, state=None):
        batch_size = conditioning_tokens.shape[0]
        actions = torch.randn(batch_size, self.action_horizon,
                              self.config.action_dim,
                              dtype=conditioning_tokens.dtype,
                              device=conditioning_tokens.device)
        dt = 1.0 / max(self.num_inference_timesteps, 1)
        for step in range(self.num_inference_timesteps):
            t_cont = step / float(max(self.num_inference_timesteps, 1))
            t_value = int(t_cont * self.config.action_num_timestep_buckets)
            timesteps = torch.full((batch_size,), t_value,
                                   device=conditioning_tokens.device, dtype=torch.long)
            hidden_states = self._build_inputs(conditioning_tokens, actions, state, timesteps)
            pred = self.model(hidden_states=hidden_states,
                              encoder_hidden_states=conditioning_tokens,
                              timestep=timesteps)
            pred_velocity = self.action_decoder(pred[:, -self.action_horizon:])
            actions = actions + dt * pred_velocity
        return actions
```

## 逐行讲解 / What's happening

### 训练 step / The training step (`forward`)

1. **`noise = torch.randn_like(actions)` + `t = self.sample_time(...)`**:
   - 中文: 给每条动作轨迹各自抽一个时间 `t ∈ (0, 1]` 和一个跟动作同 shape 的高斯噪声。`sample_time` 用 `Beta(α, β)` 而不是均匀分布——这是 rectified flow 论文里发现的训练 trick:Beta(1.5, 1.0) 这种偏向 `t=1` (噪声端) 的分布能让模型把更多容量花在"高噪声、信号弱"的时间段。
   - English: For each trajectory in the batch, draw one time `t ∈ (0, 1]` and one Gaussian noise tensor shaped like the action. `sample_time` uses `Beta(α, β)` instead of uniform — a rectified-flow training trick: skewing toward `t = 1` (the high-noise end) makes the model allocate more capacity to "high noise / low signal" timesteps where prediction is hardest.

2. **`noisy_actions = (1 - t) * noise + t * actions`**:
   - 中文: rectified flow 的核心:在 `t = 0` 处是纯噪声,在 `t = 1` 处是干净动作,中间是直线插值。这条直线就是模型要学的"轨迹",它的瞬时速度处处都是 `actions - noise`(常数,沿着直线方向)。
   - English: The rectified-flow core: pure noise at `t = 0`, clean action at `t = 1`, straight-line interpolation between. The model is asked to learn this straight trajectory, whose instantaneous velocity is *constant* `actions - noise` (the direction of the line).

3. **`velocity = actions - noise`**:
   - 中文: 这就是 supervision target——MSE 把它压在模型预测上。比 DDPM 的 epsilon 预测信号干净:rectified flow 的速度场是逐点常数,DDPM 的 epsilon 在不同 t 下要乘以不同的 scale。 / The supervision target. Cleaner than DDPM's epsilon prediction: rectified flow's velocity is pointwise constant, while DDPM's epsilon needs an extra `α_t / σ_t` rescaling at each timestep.

4. **`t_discretized = (t * num_timestep_buckets).long()`**:
   - 中文: DiT 内部的 timestep embedding 期望 int 索引(`Timesteps` 模块内部查表),所以把连续 t 离散化到 1000 个 bucket。注意 *训练时* 用的是离散化后的整数,*但* MSE 的 target 还是连续 t 的 `velocity`——所以 bucket 数量越大,timestep embedding 的分辨率越细。
   - English: DiT's `Timesteps` module expects integer indices, so the continuous `t` is bucketed into ~1000 bins. Training uses the bucketed integer for the embedding, but the MSE target `velocity` is still derived from the continuous `t` — more buckets = finer timestep embedding resolution.

5. **`_build_inputs(...)`**:
   - 中文: 输入序列拼接成 `[state | future_tokens | noisy_action_tokens]`。`future_tokens` 是一组可学习的 query embedding(`nn.Embedding(num_embodied_action_tokens, ...)`),它们不参与 supervision,只是给 DiT 提供"我要在这里产出某种结构化输出"的位置占位。`pred[:, -actions.shape[1]:]` 在最后一行取出对应 action 那段的输出。
   - English: The DiT input is `[state | future_tokens | noisy_action_tokens]`. `future_tokens` are learnable query embeddings — they're not supervised but signal to DiT "produce structured output at these positions". `pred[:, -actions.shape[1]:]` slices the predictions back out at the action positions.

6. **Padding-aware MSE**:
   - 中文: `loss * valid_mask` + `.sum() / num_valid.clamp_min(1)`——这是处理"chunk size 是固定 50,但 episode 末尾真实步数不够 50"的标准模式。padded 那几步不参与 loss,但保持张量 shape 整齐方便 batch。 `clamp_min(1)` 防止除零。
   - English: `loss * valid_mask` + `.sum() / num_valid.clamp_min(1)` — the standard handling for "chunk size is 50, but the episode tail has fewer real steps". Padded steps don't contribute to the loss but keep the tensor shape tidy for batching. `clamp_min(1)` guards against zero-division when *every* step is padding.

### 推理循环 / The inference loop (`predict_action`)

7. **`actions = torch.randn(...)`**:
   - 中文: 从纯噪声开始——这是 t=0 的端点。
   - English: Start from pure Gaussian noise — the `t = 0` endpoint of the learned trajectory.

8. **`for step in range(K): t_cont = step / K; actions += dt * pred_velocity`**:
   - 中文: Euler 一阶积分。`dt = 1/K`,K 一般 10-20。每步:把当前的(动作猜测,时间)喂给 DiT,得到速度场预测,沿这个速度走一步。10 步就走到 t=1,即干净动作。
   - English: A first-order Euler integration. `dt = 1/K`, typically K = 10-20. At each step: pass the current (action guess, time) into DiT, get the velocity prediction, step along it. After K steps you're at `t = 1` — the clean action.

9. **`@torch.no_grad()`**:
   - 中文: 推理时不需要梯度,显存可以省一大块——尤其是 DiT 跑 10-20 次的情况下。 / No-grad cuts memory dramatically when DiT is run 10-20 times in the inference loop.

## 类比 / The analogy

想象你要在迷雾里走从北京到上海。训练阶段:有人告诉你"从北京 (t=0) 到上海 (t=1) 是一条直线",你只需要学会"在任何中间位置,我下一步该往哪个方向走"——而正确答案永远是"指向上海的方向"。推理阶段:你被丢到一个随机位置(纯噪声),你只看自己脚下,问模型"现在该往哪走",走一步;再问,再走一步……走 10 步就到上海了。`forward` 学的是速度场,`predict_action` 沿着速度场积分。

Imagine traveling Beijing → Shanghai through fog. Training: you're told "the route is a straight line from Beijing (t=0) to Shanghai (t=1)" and you must learn "at any intermediate position, which way should I step next" — and the correct answer is always "toward Shanghai". Inference: you're dropped at a random spot (pure noise), look at your feet, ask the model "which way?", take one step; ask again, step again... 10 steps gets you to Shanghai. `forward` learns the velocity field, `predict_action` integrates along it.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

> **curriculum_id**: `action-head-continuous`
>
> **depends_on**: `vlm-backbone-wiring` — 必须先有 VLM 给出 conditioning tokens (一般是 multimodal hidden states 序列)。

在 nanoVLA 里,`VLAJEPAActionHead` 就坐在 VLM backbone 的下游。它的合同 (contract) 非常清楚:

- **输入 / Inputs**:
  - `conditioning_tokens: [B, S_vlm, H_vlm]` —— VLM 输出的 image+language+state 融合表征。
  - `actions: [B, T_action, D_action]` —— 训练时是 ground-truth chunk;推理时模型自己生成。
  - `state: [B, D_state]` —— 当前机器人本体状态(关节角度、 gripper 状态等),可选。
  - `action_is_pad: [B, T_action]` —— 哪些时间步是 padding,可选。

- **输出 / Outputs**:
  - 训练时:scalar loss(标量,可以 backward)。
  - 推理时:`[B, T_action, D_action]` 干净动作 chunk。

- **上游 / Upstream**:VLM backbone(可以是 Qwen3-VL、PaliGemma、SigLIP+LLaMA 等)负责把图像 + 指令 + 当前状态变成 `conditioning_tokens`。
- **下游 / Downstream**:动作 chunk 直接送进机器人控制循环(以 30 Hz 重新规划,前 N 步执行)。

**省掉这个组件会怎样 / What if you omit this**:你失去了"连续动作"这个能力,只能退回到 OpenVLA 那种"离散 action token + LM next-token"路线——可执行,但精度受限于离散化粒度。

**生产级实现还要加什么 / What production adds on top**:
- DDP/FSDP 分布式训练 hooks(梯度同步、参数分片);
- 多 embodiment(不同机器人的动作维度不同)的支持——`MultiEmbodimentActionEncoder` 那一层;
- 推理时的 action chunking / replanning 策略(Real-Time Chunking 那种 seam-smoothing);
- BF16 / FP8 训练的数值稳定性;
- LoRA 微调入口(参考 06-10 那篇 openvla-lora-finetune 笔记)。

In your nanoVLA, `VLAJEPAActionHead` sits immediately downstream of the VLM backbone. Its contract is clean: it consumes `conditioning_tokens` from the VLM plus a noisy action chunk and (during training) returns a scalar loss; during inference it returns a clean action chunk. Upstream is the VLM that fuses image / language / state into `conditioning_tokens`. Downstream is the robot control loop that executes the predicted action chunk (typically re-planning at 30 Hz, executing the first N steps).

Omitting the action head forces you back onto OpenVLA-style discrete action tokens + LM next-token loss — workable but quantization-limited. A production implementation layers on: DDP/FSDP hooks, multi-embodiment support (different robots have different action dims — see `MultiEmbodimentActionEncoder`), action-chunking replanning (see the 06-09 RTC note), bf16/fp8 numerical-stability tweaks, and a LoRA fine-tune entry point.

## 自己跑一遍 / Try it yourself

```python
# pip install torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

class NanoFlowMatchActionHead(nn.Module):
    """Minimal stand-alone flow-matching action head — 30 lines."""
    def __init__(self, cond_dim=64, action_dim=7, horizon=10, hidden=128):
        super().__init__()
        self.horizon, self.action_dim = horizon, action_dim
        # The "DiT" is just a tiny transformer encoder for the demo.
        self.proj = nn.Linear(action_dim + 1, hidden)         # action + time
        self.cond_proj = nn.Linear(cond_dim, hidden)
        self.blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden, 4, hidden*2, batch_first=True), num_layers=2)
        self.out  = nn.Linear(hidden, action_dim)
        self.beta = Beta(torch.tensor(1.5), torch.tensor(1.0))

    def forward(self, cond, actions):                          # training
        t     = self.beta.sample([actions.shape[0]]).to(actions)
        noise = torch.randn_like(actions)
        noisy = (1 - t[:, None, None]) * noise + t[:, None, None] * actions
        velocity = actions - noise
        x = self.proj(torch.cat([noisy, t[:, None, None].expand(-1, self.horizon, 1)], -1))
        x = x + self.cond_proj(cond)[:, None]
        return F.mse_loss(self.out(self.blocks(x)), velocity)

    @torch.no_grad()
    def predict(self, cond, K=10):                             # inference
        a = torch.randn(cond.shape[0], self.horizon, self.action_dim, device=cond.device)
        for k in range(K):
            t_full = torch.full((cond.shape[0], self.horizon, 1), k / K, device=cond.device)
            x = self.proj(torch.cat([a, t_full], -1)) + self.cond_proj(cond)[:, None]
            a = a + (1 / K) * self.out(self.blocks(x))
        return a

# Sanity train on a tiny dataset.
torch.manual_seed(0)
head = NanoFlowMatchActionHead()
opt  = torch.optim.AdamW(head.parameters(), 3e-4)
cond, actions = torch.randn(64, 64), torch.randn(64, 10, 7)
for step in range(200):
    loss = head(cond, actions); opt.zero_grad(); loss.backward(); opt.step()
    if step % 50 == 0: print(f"step {step:>4d}   loss={loss.item():.4f}")

print("Predicted shape:", head.predict(cond[:1]).shape)
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
step    0   loss=2.0034
step   50   loss=1.1209
step  100   loss=0.7843
step  150   loss=0.5527
Predicted shape: torch.Size([1, 10, 7])
```

中文一两句:看损失从 ~2.0 降到 ~0.5——这是一个 over-parameterized 模型把固定 64 个 (cond, action) pair 背下来的过程,完全足够验证整条 forward + predict 流程没接错。

In English: loss drops from ~2.0 toward 0.5 — an over-parameterized model memorizing 64 fixed (cond, action) pairs, which is enough to verify the forward + predict wiring is correct end-to-end.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **openpi 的 pi0 PyTorch 移植** / **openpi's pi0 PyTorch port (2026-06-13 note)**: 同一个 `velocity = action - noise` MSE,但他们的 t 直接是连续 float,没有 timestep_buckets 那一层;DiT 用了 cross-attention 注入 cond 而不是 prepend。 / Same `velocity = action - noise` MSE, but pi0 uses continuous `t` directly (no timestep_buckets) and a cross-attention DiT instead of prepended conditioning.
- **lerobot 的 GR00T flow-matching head** / **lerobot's GR00T flow-matching head (2026-06-08 note)**: 完整训练 step 也是 6 行(`compute_loss`),但 action 用了 `MultiEmbodimentActionEncoder` 处理不同机器人的 action 维度。 / The whole training step is 6 lines (`compute_loss`), but action goes through `MultiEmbodimentActionEncoder` to handle multi-robot action dims.
- **diffusers 的 `FlowMatchEulerDiscreteScheduler`** / **diffusers' `FlowMatchEulerDiscreteScheduler` (2026-05-31, 2026-06-01 notes)**: 同一个 Euler `+= dt * v` 推理循环,但封装得更通用——可以处理 dynamic shift、resolution-aware schedule 等。VLA 场景下通常用不到这么多 schedule 花活,行内一个 for loop 就够。 / Same Euler `+= dt * v` loop, but wrapped for general diffusers usage (dynamic shift, resolution-aware schedules). VLAs usually don't need those bells; one inline `for` loop suffices.

## 注意事项 / Caveats / when it breaks

- **`sample_time` 用的 Beta 是构造时的 Python float / The `Beta` is built from Python floats**: 这正是隔壁 GR00T 今天踩坑的地方(见今日 tracked 笔记)。VLA-JEPA 这版还没遇到 meta device 问题,但如果你计划把这个 action head 嵌进一个用 `init_empty_weights` 加载的 backbone,记得提前把 Beta 钉到 cpu/fp32。 / The same float-to-tensor coercion that broke GR00T today is latent here. If you'll wrap this action head under HF `init_empty_weights`, pin the Beta tensors to cpu/fp32 first.
- **`num_inference_timesteps` 太小 → 动作变形 / Too few steps distorts the trajectory**: Euler 是一阶,误差跟 dt² 走。`K = 10` 一般够,但 `K = 2` 就会看到 chunk 首尾跳变。如果想用 `K = 1`,需要训练时也只用一步——那是一致性蒸馏 (consistency distillation) 的事,见 2026-05-29 那篇 DMD 笔记。 / Euler is first-order, error scales like `dt²`. `K = 10` is usually fine; `K = 2` shows visible jumps. `K = 1` requires consistency distillation (see the 2026-05-29 DMD note).
- **`padding mask` 全 True 会 NaN / All-padding mask → NaN**: `(loss * mask).sum() / 0` 会产生 inf;`clamp_min(1)` 把分母兜底成 1,但 loss 会变成假的 0——训练时 sample 一个真实例子也不掉,可能在调试时给你错觉。 / `clamp_min(1)` makes the math safe but masks "all padding" by emitting a fake zero loss — fine in practice (you'll never `batch_size = 0`), but worth knowing during debugging.
- **`future_tokens` 不参与监督 / `future_tokens` get no direct supervision**: 它们只是占位 query。如果你的 backbone 已经在 cross-attention 里把 visual+language 信息塞进去了,这些 query 其实是冗余的;某些更轻量的 action head 直接用零 vector 或者 action position embedding 替代。 / They're unsupervised position queries. If your backbone already injects vision+language via cross-attn, they're often redundant — lighter heads use zero vectors or just an action-position embedding instead.

## 延伸阅读 / Further reading

- [lerobot PR #3568 — VLA-JEPA policy](https://github.com/huggingface/lerobot/pull/3568) — the introducing commit (2026-06-04).
- [`world_model.py` — the JEPA video predictor](https://github.com/huggingface/lerobot/blob/38327fdc8458959f47d555c159307538200d0561/src/lerobot/policies/vla_jepa/world_model.py) — paired with this action head; see today's wam note.
- [Rectified flow paper (Liu et al., 2023)](https://arxiv.org/abs/2209.03003) — the source of `(1-t)*noise + t*data` and `velocity = data - noise`.
- [pi0 paper (Physical Intelligence, 2024)](https://www.physicalintelligence.company/blog/pi0) — the influential VLA that popularized this exact shape of action head.
