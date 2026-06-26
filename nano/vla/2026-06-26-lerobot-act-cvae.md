---
date: 2026-06-26
topic: vla
source: vla
repo: huggingface/lerobot
file: src/lerobot/policies/act/modeling_act.py
permalink: https://github.com/huggingface/lerobot/blob/6a788fbdb02cabfae60f7408636945df0b1eafa0/src/lerobot/policies/act/modeling_act.py#L322-L452
difficulty: advanced
read_time: ~14 min
tags: [code-of-the-day, vla, action-chunking, cvae, reparameterization, act, curriculum]
build_role: action-chunking — CVAE 变体 (cross-repo variant vs flow-matching in SmolVLA)
---

# ACT 的 CVAE 动作编码器：用重参数化技巧压缩动作序列，推理时 latent 直接置零 / ACT's CVAE Action Encoder: Compress Action Chunks via Reparameterization, Set Latent to Zero at Inference

> **一句话 / In one line**: ACT（Action Chunking Transformer）用 Conditional VAE 把整段动作序列压成 (μ, σ²) 参数的隐变量分布；训练时做重参数化采样学 KL 约束，推理时 latent 直接置零，整个 encoder 在推理时完全弃用。 / ACT uses a Conditional VAE to compress a full action chunk into a latent (μ, σ²); training samples the latent via reparameterization and learns KL regularization; inference sets latent to zeros and discards the encoder entirely.

## 为什么重要 / Why this matters

动作分块（action chunking）是现代 VLA 的核心设计之一：与其每步只预测一个动作，不如一次输出未来 T 步的完整轨迹，再滚动执行。怎么"压缩"这段 T 步轨迹的表示，是各流派 VLA 的核心分歧：

- **ACT（本篇）**: 用 CVAE —— 一个独立的 Transformer encoder 把动作序列编码成隐变量分布 (μ, σ²)，再用重参数化采样。
- **SmolVLA / OpenPi π₀**: 用 flow-matching —— 把动作预测建模成一个从噪声出发的 ODE，不需要显式的 encoder/decoder 分离。
- **OpenVLA / SeVa**: 把动作离散化成 token，直接用 next-token-prediction 的 cross-entropy loss，完全没有 latent space。

这三条路都能做 action chunking，但隐变量结构、推理效率、训练信号的形式完全不同。

Action chunking is one of the central design choices in modern VLAs: rather than predicting one action per step, predict a full T-step trajectory and execute it in a rolling window. How to "compress" this T-step trajectory representation is where VLA architectures diverge fundamentally:

- **ACT (this note)**: CVAE — a separate Transformer encoder compresses the action sequence into a latent distribution (μ, σ²), then reparameterizes.
- **SmolVLA / OpenPi π₀**: flow-matching — models action prediction as an ODE starting from noise; no explicit encoder/decoder separation.
- **OpenVLA / SeVa**: discretize actions into tokens; pure next-token-prediction with cross-entropy; no latent space at all.

All three approaches implement action chunking, but the latent structure, inference efficiency, and training signal differ entirely.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/act/modeling_act.py`](https://github.com/huggingface/lerobot/blob/6a788fbdb02cabfae60f7408636945df0b1eafa0/src/lerobot/policies/act/modeling_act.py#L322-L452)

```python
def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
    """A forward pass through the Action Chunking Transformer (with optional VAE encoder)."""
    batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]

    # === CVAE ENCODER PATH (training only) ===
    if self.config.use_vae and ACTION in batch and self.training:
        # Prepare VAE encoder input: [CLS token, robot_state, action_sequence]
        cls_embed = einops.repeat(
            self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
        )  # (B, 1, D)
        if self.config.robot_state_feature:
            robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE])
            robot_state_embed = robot_state_embed.unsqueeze(1)  # (B, 1, D)
        action_embed = self.vae_encoder_action_input_proj(batch[ACTION])  # (B, S, D)

        if self.config.robot_state_feature:
            vae_encoder_input = [cls_embed, robot_state_embed, action_embed]  # (B, S+2, D)
        else:
            vae_encoder_input = [cls_embed, action_embed]
        vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

        # Fixed positional embedding for the VAE encoder
        pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, S+2, D)

        # Key padding mask: CLS/state tokens are never padded, action tokens may be
        cls_joint_is_pad = torch.full(
            (batch_size, 2 if self.config.robot_state_feature else 1),
            False, device=batch[OBS_STATE].device,
        )
        key_padding_mask = torch.cat([cls_joint_is_pad, batch["action_is_pad"]], axis=1)

        # Forward: VAE encoder Transformer → CLS token output → project to (μ, log σ²)
        cls_token_out = self.vae_encoder(
            vae_encoder_input.permute(1, 0, 2),
            pos_embed=pos_embed.permute(1, 0, 2),
            key_padding_mask=key_padding_mask,
        )[0]  # select the class token → (B, D)
        latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
        mu = latent_pdf_params[:, : self.config.latent_dim]
        log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]  # 2 * log(sigma)

        # Reparameterization trick: latent = μ + exp(log_σ²/2) * ε,  ε ~ N(0,1)
        latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
    else:
        # === INFERENCE PATH: no encoder, latent = zeros ===
        mu = log_sigma_x2 = None
        latent_sample = torch.zeros(
            [batch_size, self.config.latent_dim], dtype=torch.float32
        ).to(batch[OBS_STATE].device)

    # === MAIN TRANSFORMER ENCODER ===
    # Input tokens: [latent_proj, robot_state_proj, env_state_proj, *image_features]
    encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
    encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
    if self.config.robot_state_feature:
        encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
    if self.config.env_state_feature:
        encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))
    if self.config.image_features:
        for img in batch[OBS_IMAGES]:
            cam_features = self.backbone(img)["feature_map"]
            cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
            cam_features = self.encoder_img_feat_input_proj(cam_features)
            cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
            cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")
            encoder_in_tokens.extend(list(cam_features))
            encoder_in_pos_embed.extend(list(cam_pos_embed))

    encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
    encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)
    encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

    # === DECODER: zero queries cross-attend encoder output → action predictions ===
    decoder_in = torch.zeros(
        (self.config.chunk_size, batch_size, self.config.dim_model),
        dtype=encoder_in_pos_embed.dtype,
        device=encoder_in_pos_embed.device,
    )
    decoder_out = self.decoder(
        decoder_in,
        encoder_out,
        encoder_pos_embed=encoder_in_pos_embed,
        decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
    )
    decoder_out = decoder_out.transpose(0, 1)   # (B, chunk_size, dim_model)
    actions = self.action_head(decoder_out)      # (B, chunk_size, action_dim)
    return actions, (mu, log_sigma_x2)
```

## 逐行讲解 / What's happening

1. **CVAE encoder 输入拼接：`[CLS, robot_state, action_sequence]`**
   - 中文: VAE encoder 接收的输入是 CLS token + 可选的机器人状态 + 整段 T 步动作序列拼接而成的序列。CLS token 是一个可学习的 embedding，类似 BERT 的 `[CLS]`，最终它的输出向量携带了对整段动作的"压缩摘要"。
   - English: The VAE encoder receives a sequence of: a learnable CLS token + optional robot state + the full T-step action sequence. The CLS token (like BERT's `[CLS]`) is the one whose final output is used as a "compressed summary" of the whole action chunk.

2. **`latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)`**
   - 中文: CLS token 输出经过一个线性投影得到 `(μ, log(σ²))` 拼接向量，维度 `2 * latent_dim`。前半是均值，后半是方差的对数（×2，即 `log(σ²)`，也可以写作 `2*log(σ)`）。
   - English: The CLS token output is projected linearly to a `2 * latent_dim` vector encoding both `μ` and `log(σ²)` (the log-variance). The ×2 log convention — called `log_sigma_x2` in the code — means `log(σ²) = 2·log(σ)`.

3. **重参数化技巧：`latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)`**
   - 中文: 这就是 VAE 的核心：不直接从 `N(μ, σ²)` 采样（不可微），而是先采 `ε ~ N(0,1)`，再用 `μ + σ·ε` 得到样本，梯度可以通过 `μ` 和 `σ` 回传。`log_sigma_x2.div(2).exp()` 就是 `exp(log(σ))` = σ。
   - English: This is the VAE's core trick: instead of sampling directly from `N(μ, σ²)` (non-differentiable), sample `ε ~ N(0,1)` and compute `μ + σ·ε`. Gradients flow through both `μ` and `σ`. `log_sigma_x2.div(2).exp()` computes `exp(log(σ))` = σ.

4. **推理时：`latent_sample = torch.zeros([batch_size, latent_dim])`**
   - 中文: 推理时没有 GT 动作序列，所以 encoder 完全不运行。latent 直接置零，相当于取隐变量分布的先验均值。这意味着 ACT 训练时隐变量携带的信息（多模态分布、执行意图）在推理时全部丢弃——CVAE 结构的主要作用是提供正则化，而非在推理期间提供真实隐变量。
   - English: At inference there's no ground-truth action sequence, so the encoder never runs. Latent is set to zeros — the prior mean. This means all the information the latent encoded during training (multi-modal distribution, execution intent) is discarded at inference time. The CVAE's role is primarily regularization rather than providing a meaningful latent at test time.

5. **主 encoder 输入：latent token 排在第一位**
   - 中文: 主 Transformer encoder 的第一个 token 是 `latent` 的线性投影。它后面跟着机器人状态、环境状态、然后是每个相机的 `(H*W)` 个图像特征 token。latent token 通过注意力机制影响所有其他 token，在训练时把动作意图"注入"进来。
   - English: The first token of the main Transformer encoder is the linear projection of `latent_sample`. It's followed by robot state, environment state, then `(H*W)` image feature tokens per camera. The latent token propagates action intent to all other tokens via attention — at training time; at inference this is just zeros, which the encoder's attention has learned to "ignore as prior."

6. **Decoder：零查询向量 cross-attend encoder 输出**
   - 中文: Decoder 的查询向量全是零（`chunk_size` 个零向量），只依靠位置嵌入区分不同的时间步。它们通过 cross-attention 从 encoder 输出中读取多模态条件信息，最终输出经 `action_head` 线性投影得到 `(B, chunk_size, action_dim)` 的动作序列。
   - English: Decoder queries are all zeros (chunk_size zero vectors), differentiated only by learned positional embeddings. They cross-attend encoder output to extract multimodal conditioning, then `action_head` projects each decoder output to an action vector, yielding `(B, chunk_size, action_dim)`.

## 类比 / The analogy

想象一位舞蹈动作编排师（CVAE encoder）看了一段 5 秒的完整舞蹈示范（action sequence），把它压缩成一张小纸条（latent），纸条上写了"风格：优雅、节奏：慢板"（μ 和 σ²）。训练时，表演者（主 Transformer）拿着这张纸条加上当前场景（图像 + 机器人状态），输出完整的动作轨迹。但在真正演出时（推理），没有示范视频，纸条上什么都不写（latent=zeros）——表演者仅凭当前场景和之前学到的经验来完成动作。编排师只是训练期的"辅导员"，演出时根本不出现。

A dance choreographer (CVAE encoder) watches a 5-second demonstration (action sequence) and distills it into a note card (latent) reading "style: graceful, tempo: adagio" (μ and σ²). During training, the main performer (Transformer) uses this note plus the current scene (images + robot state) to output the full movement trajectory. At performance time (inference), there's no demo video — the note card is left blank (latent=zeros). The performer relies entirely on the current scene and learned experience. The choreographer was only a training-time coach, absent at showtime.

## 在 nanoVLA 中的位置 / Where this lives in your nanoVLA

这是 nanoVLA 课程的 **action-chunking** 组件，依赖 **vision-encoder**（图像特征）和 **modality-projector**（视觉 token 投影）已在之前的课程中完成。

**上游 / Upstream**: 视觉编码器输出的 `(B, H*W, D)` 图像特征 token + 机器人状态向量 `(B, state_dim)` 是本组件的输入。
**本组件 / This component**: 训练时 CVAE encoder 把 `(B, T, action_dim)` 的动作序列压缩进 latent；推理时 latent=zeros，主 encoder 直接用视觉 + 状态 + 零latent 生成 encoder 输出，decoder 输出 `(B, T, action_dim)`。
**下游 / Downstream**: action chunk 送入执行引擎，按 `n_action_steps` 滚动执行，直到 chunk 耗尽再重新推理。
**如果省掉这个组件 / Without this component**: 退化为每步输出单个动作（`T=1`），失去"预见未来几步"的动作平滑效果，在高频率控制任务（机械臂 50Hz）上抖动显著增加。

This is the **action-chunking** component of the nanoVLA curriculum, depending on **vision-encoder** and **modality-projector** completed in earlier curriculum items.

**Upstream**: Vision encoder outputs `(B, H*W, D)` image feature tokens + robot state vector `(B, state_dim)`.
**This component**: During training, the CVAE encoder compresses `(B, T, action_dim)` action sequences into a latent. At inference, latent=zeros, the main encoder uses vision + state + zero-latent, and the decoder produces `(B, T, action_dim)`.
**Downstream**: The action chunk is rolled out for `n_action_steps` before re-querying the policy.
**Without this component**: Degrades to single-step prediction (`T=1`), losing the smoothing benefit of lookahead, increasing jitter in high-frequency control (robotic arm at 50 Hz).

**Production additions needed**: (1) temporal ensembling (running average of overlapping chunks, as `ACTTemporalEnsembler` does); (2) `action_is_pad` masking for variable-length demonstrations; (3) KL warmup schedule so the VAE doesn't collapse early in training.

## 自己跑一遍 / Try it yourself

```python
import torch
import torch.nn as nn

# Minimal standalone CVAE action encoder for nanoVLA
class MiniCVAEActionHead(nn.Module):
    def __init__(self, state_dim=9, action_dim=7, chunk_size=10, latent_dim=32, d_model=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.chunk_size = chunk_size
        # VAE encoder: project state + actions → latent
        self.vae_proj = nn.Linear(state_dim + action_dim, d_model)
        self.vae_out = nn.Linear(d_model, 2 * latent_dim)  # μ and log_σ²
        # Main decoder
        self.latent_proj = nn.Linear(latent_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)
        self.decoder = nn.Linear(d_model * 2, action_dim * chunk_size)

    def forward(self, state, actions=None):
        B = state.shape[0]
        if actions is not None and self.training:
            # VAE encode: pool over action dimension as a simple approximation
            combined = torch.cat([state, actions.mean(1)], dim=-1)
            h = self.vae_proj(combined).relu()
            params = self.vae_out(h)
            mu, log_sigma_x2 = params[:, :self.latent_dim], params[:, self.latent_dim:]
            latent = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu = log_sigma_x2 = None
            latent = torch.zeros(B, self.latent_dim, device=state.device)
        h = torch.cat([self.latent_proj(latent), self.state_proj(state)], dim=-1)
        actions_out = self.decoder(h.relu()).view(B, self.chunk_size, -1)
        return actions_out, (mu, log_sigma_x2)

model = MiniCVAEActionHead()
state = torch.randn(2, 9)
actions_gt = torch.randn(2, 10, 7)

model.train()
pred, (mu, log_sigma_x2) = model(state, actions_gt)
kl = (-0.5 * (1 + log_sigma_x2 - mu.pow(2) - log_sigma_x2.exp())).sum(-1).mean()
loss = ((pred - actions_gt) ** 2).mean() + 0.1 * kl
print(f"l1={loss.item():.4f}  kl={kl.item():.4f}")

model.eval()
with torch.no_grad():
    pred_infer, _ = model(state)   # latent=zeros at inference
print(f"inference chunk shape: {pred_infer.shape}")  # [2, 10, 7]
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
l1=X.XXXX  kl=X.XXXX
inference chunk shape: torch.Size([2, 10, 7])
```

中文：注意训练时 `kl` 不为零（VAE 正在压缩信息），推理时 `latent=zeros` 完全不依赖 encoder——这是 ACT CVAE 设计的核心属性。

English: Note that `kl` is nonzero during training (the VAE is compressing information), and at inference `latent=zeros` requires no encoder at all — this is the defining property of ACT's CVAE design.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **SmolVLA / OpenPi π₀（flow-matching 变体）** / **SmolVLA / OpenPi π₀ (flow-matching variant)**: 同样输出 action chunk `(B, T, action_dim)`，但不用 CVAE，而是用 flow matching——从随机噪声出发通过多步去噪生成动作序列，没有显式的 latent 压缩，推理需要多步 ODE 求解。 / Also outputs action chunks `(B, T, action_dim)` but uses flow matching instead of CVAE — starts from noise and denoises via multi-step ODE; no explicit latent compression; inference requires multiple denoising steps.
- **Isaac-GR00T ActionChunk（delta 归一化变体）** / **Isaac-GR00T ActionChunk (delta-normalization variant)**: 也用 flow-matching 做 action chunking，但在动作表示上加了 delta 归一化和 SLERP 旋转插值，是对 ACT 动作空间表示的改进。 / Also uses flow matching for chunking but adds delta-normalization and SLERP rotation interpolation on top, refining ACT's action space representation.
- **BET（Behavior Transformer）** / **BET (Behavior Transformer)**: 用 VQ-VAE 代替 CVAE 做动作序列量化，同样是"压缩-解压"框架，但 latent 是离散码本索引而非连续高斯分布。 / Uses VQ-VAE instead of CVAE for action sequence quantization — same compress-decompress framework, but the latent is a discrete codebook index rather than a continuous Gaussian.

## 注意事项 / Caveats / when it breaks

- **推理时 latent=zeros 假设 VAE 已收敛到合理先验** / **Inference latent=zeros assumes a converged VAE prior**: KL 项会推动 `μ → 0, σ → 1`，使得 latent=zeros 在推理时是一个合理的"平均动作意图"。如果训练早期 KL 权重太高，VAE 会过度压缩（posterior collapse），action diversity 下降。 / The KL term pushes `μ → 0, σ → 1` so that zeros is a sensible "average action intent" at inference. If KL weight is too high early in training, posterior collapse occurs and action diversity suffers.
- **`action_is_pad` mask 必须正确对齐** / **`action_is_pad` mask must be correctly aligned**: key_padding_mask 把填充的动作步标记为 `True`（忽略），如果对齐出错，VAE encoder 会把 padding 当作真实动作信息来压缩，导致 latent 被干扰。 / The key_padding_mask marks padded action steps as `True` (ignored). Misalignment causes the VAE encoder to compress padding as if it were real action data, corrupting the latent.
- **`log_sigma_x2` 命名** / **`log_sigma_x2` naming**: 变量名里的 `x2` 表示这是 `2*log(σ)` 而不是 `log(σ²)`（尽管两者数值相同）。KL 公式里用 `log_sigma_x2.exp()` 得到的是 σ²，不是 σ，这是常见的混淆来源。 / The `x2` in the name means `2·log(σ)` (numerically equal to `log(σ²)` but conceptually different). In the KL formula, `.exp()` gives σ², not σ — a common source of confusion.

## 延伸阅读 / Further reading

- [ACT paper: Learning Fine-Grained Bimanual Manipulation (arXiv:2304.13705)](https://huggingface.co/papers/2304.13705)
- [VAE paper: Auto-Encoding Variational Bayes (arXiv:1312.6114)](https://huggingface.co/papers/1312.6114) — Appendix B for the KL formula
- [LeRobot ACT policy implementation](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/modeling_act.py)
- [SmolVLA flow-matching step (2026-06-25 note)](../../nano/vla/2026-06-25-smolvla-flow-matching-step.md)
