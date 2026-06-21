---
date: 2026-06-21
topic: diffusion
source: trending
repo: Lightricks/LTX-2
file: packages/ltx-core/src/ltx_core/conditioning/types/reference_video_cond.py
permalink: https://github.com/Lightricks/LTX-2/blob/780984275fd47128b02bef9b5c085404276866ee/packages/ltx-core/src/ltx_core/conditioning/types/reference_video_cond.py#L12-L102
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, ic-lora, reference-conditioning, rope, denoise-mask, trending]
---

# IC-LoRA 40 行:把参考视频 token 原封不动地拼进去,让 DiT 自己学对齐 / IC-LoRA in 40 lines: append clean reference tokens and let the DiT learn alignment on its own

> **一句话 / In one line**: LTX-2 的 `VideoConditionByReferenceLatent` 把参考视频 token 直接拼在目标 latent 序列后面,给它们设 `denoise_mask = 1 - strength`(strength=1 时参考 token 全程保持干净),DiT 的 self-attention 自由 attend 两段——IC-LoRA 不需要任何新的 cross-attention 模块。/ LTX-2's `VideoConditionByReferenceLatent` appends reference video tokens directly after the target latent sequence, sets `denoise_mask = 1 - strength` (strength=1 keeps reference tokens clean throughout), and lets DiT self-attention attend freely across both — IC-LoRA needs no new cross-attention module.

## 为什么重要 / Why this matters

IC-LoRA (Image-Conditional LoRA) 解决了一个常见问题:怎样让 DiT 视频模型"参照某个视频/图像风格生成"——既不破坏原始模型权重,又不需要增加新的 attention 模块。

关键洞见:DiT 本来就有 self-attention,所以直接把参考 token 拼进输入序列,DiT 就能 attend 到它们。区别在于参考 token 的 `denoise_mask = 1 - strength`:当 `strength=1` 时,掩码为 0,参考 token 在扩散过程中**永远不被加噪**(始终保持干净);目标 token 的掩码为 1,正常扩散。DiT 学会了"attend 到那些干净的 token 来参考风格"。多分辨率支持靠 RoPE 位置缩放:如果参考是低分辨率的,乘以 `downscale_factor` 把它的 patch 坐标映射到目标的坐标系里。

IC-LoRA solves a common problem: how to make a DiT video model "generate conditioned on a reference video/image" — without altering the base weights or adding new attention modules.

The key insight: DiT already has self-attention, so appending reference tokens to the input sequence is enough — the DiT will naturally attend to them. The difference lies in `denoise_mask = 1 - strength`: at `strength=1`, the mask is 0, meaning reference tokens are **never noised** during diffusion (always clean); target tokens have mask=1 and follow normal diffusion. The DiT learns to "attend to those clean tokens for style/content reference". Multi-resolution support comes from RoPE position scaling: if the reference is lower-resolution, multiply its patch coordinates by `downscale_factor` to map them into the target's coordinate space.

## 代码 / The code

`Lightricks/LTX-2` — [`packages/ltx-core/src/ltx_core/conditioning/types/reference_video_cond.py`](https://github.com/Lightricks/LTX-2/blob/780984275fd47128b02bef9b5c085404276866ee/packages/ltx-core/src/ltx_core/conditioning/types/reference_video_cond.py#L12-L102)

```python
class VideoConditionByReferenceLatent(ConditioningItem):
    """
    IC-LoRA inference: append reference (control) tokens as clean latents.
    Trained by concatenating [noisy target | clean reference] and letting the
    DiT self-attention attend freely across both halves.
    """

    def __init__(
        self,
        latent: torch.Tensor,          # reference video latents [B, C, F, H, W]
        downscale_factor: int = 1,     # target/reference spatial ratio
        temporal_scale_factor: int = 1,# target/reference temporal ratio
        strength: float = 1.0,         # 1.0 = full conditioning, 0.0 = no conditioning
    ):
        self.latent = latent
        self.downscale_factor = downscale_factor
        self.temporal_scale_factor = temporal_scale_factor
        self.strength = strength

    def apply_to(
        self,
        latent_state: LatentState,
        latent_tools: VideoLatentTools,
    ) -> LatentState:
        """Append reference tokens with positions translated into the target frame."""

        # 1. Patchify the reference latent → tokens
        tokens = latent_tools.patchifier.patchify(self.latent)

        # 2. Compute reference patch positions in their own coordinate system
        latent_coords = latent_tools.patchifier.get_patch_grid_bounds(
            output_shape=VideoLatentShape.from_torch_shape(self.latent.shape),
            device=self.latent.device,
        )
        positions = get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=latent_tools.scale_factors,
            causal_fix=latent_tools.causal_fix,
        ).to(dtype=torch.float32)

        # 3. Translate temporal positions to match target FPS spacing
        positions[:, 0, ...] /= latent_tools.fps / self.temporal_scale_factor

        # 4. Align temporal positions so ref's last patch ends with target's last
        if self.temporal_scale_factor != 1:
            t_target = latent_state.positions[:, 0, 0:1, 1:2].to(dtype=torch.float32)
            positions[:, 0, ...] = torch.clamp(
                positions[:, 0, ...] - (self.temporal_scale_factor - 1) * t_target,
                min=0,
            )

        # 5. Scale spatial positions to match target coordinates
        if self.downscale_factor != 1:
            positions[:, 1, ...] *= self.downscale_factor  # height axis
            positions[:, 2, ...] *= self.downscale_factor  # width axis

        # 6. denoise_mask: 1 - strength
        #    strength=1.0 → mask=0.0 → reference stays clean throughout diffusion
        #    strength=0.0 → mask=1.0 → reference is denoised like target (no conditioning)
        denoise_mask = torch.full(
            size=(*tokens.shape[:2], 1),
            fill_value=1.0 - self.strength,
            device=self.latent.device, dtype=self.latent.dtype,
        )

        # 7. Build updated attention mask (ref tokens attend to everything)
        new_attention_mask = update_attention_mask(
            latent_state=latent_state,
            attention_mask=None,
            num_noisy_tokens=latent_tools.target_shape.token_count(),
            num_new_tokens=tokens.shape[1],
            batch_size=tokens.shape[0],
            device=self.latent.device, dtype=self.latent.dtype,
        )

        # 8. Append: [target | reference] for all sequence fields
        return LatentState(
            latent=torch.cat([latent_state.latent, torch.zeros_like(tokens)], dim=1),
            denoise_mask=torch.cat([latent_state.denoise_mask, denoise_mask], dim=1),
            positions=torch.cat([latent_state.positions, positions], dim=2),
            clean_latent=torch.cat([latent_state.clean_latent, tokens], dim=1),
            attention_mask=new_attention_mask,
        )
```

## 逐行讲解 / What's happening

1. **`tokens = latent_tools.patchifier.patchify(self.latent)` — 参考视频 token 化**
   - 中文: 把参考视频的 latent `[B, C, F, H, W]` 切成和目标视频相同规格的 patch token `[B, N_ref, D]`。patch size 和 temporal stride 由 patchifier 决定,与目标视频保持一致。
   - English: Cuts the reference video latent `[B, C, F, H, W]` into the same-spec patch tokens `[B, N_ref, D]` as the target. Patch size and temporal stride are set by the patchifier, matching the target video spec.

2. **`positions[:, 0, ...] /= latent_tools.fps / self.temporal_scale_factor` — 时间轴缩放**
   - 中文: RoPE 的时间位置以"秒"为单位。如果参考视频的帧率是目标的 1/4 (`temporal_scale_factor=4`),除以 `fps/4` 等于乘以 4/fps,把参考帧的时间坐标拉伸到和目标视频一样的间距。这样参考帧的时间位置和目标视频的时间位置在 RoPE 空间里对齐。
   - English: RoPE temporal positions are measured in seconds. If the reference FPS is 1/4 of the target (`temporal_scale_factor=4`), dividing by `fps/4` stretches the reference's time coordinates to match the target's spacing. Reference and target frames then occupy the same RoPE temporal space.

3. **`positions[:, 1, ...] *= self.downscale_factor` — 空间轴缩放**
   - 中文: 如果参考图像分辨率是目标的一半 (`downscale_factor=2`),参考 patch 的坐标乘以 2 后落在目标的坐标系里——参考图的左上角 patch 依然对应目标视频的左上角,右下角对应右下角。
   - English: If the reference is half the resolution of the target (`downscale_factor=2`), multiplying reference patch coordinates by 2 maps them into the target coordinate space — the reference's top-left patch still corresponds to the target's top-left, bottom-right to bottom-right.

4. **`denoise_mask = full(fill_value=1.0 - self.strength)` — IC-LoRA 的核心**
   - 中文: 这个 mask 控制每个 token 在 diffusion 中被加噪的程度。`mask=1` 表示"正常扩散",`mask=0` 表示"不加噪,保持干净"。当 `strength=1.0`,参考 token 的 mask=0——它们在每个扩散步骤都保持原样,DiT 始终能看到干净的参考视频 token。当 `strength=0.5`,参考 token 被半噪,条件信号减弱。
   - English: This mask controls how much each token is noised during diffusion. `mask=1` means "normal diffusion", `mask=0` means "no noise, stay clean". With `strength=1.0`, reference tokens have mask=0 — they remain clean at every diffusion step and the DiT always sees the unnoised reference. With `strength=0.5`, reference tokens are half-noised, weakening the conditioning signal.

5. **`latent=torch.cat([latent_state.latent, torch.zeros_like(tokens)], dim=1)` vs `clean_latent=torch.cat([..., tokens], dim=1)`**
   - 中文: `latent_state.latent` 是"加噪的"序列,`clean_latent` 是"干净的"序列。参考 token 在 `latent` 里是全零占位符(因为 mask=0,它们不参与加噪),但在 `clean_latent` 里是真实的参考 latent。DiT 前向里会根据 `denoise_mask` 混合这两条路,得到"有些位置加噪有些不加噪"的输入序列。
   - English: `latent_state.latent` is the "noised" sequence; `clean_latent` is the "clean" sequence. Reference tokens appear as zeros in `latent` (since mask=0 they don't participate in the noising process), but as actual reference latents in `clean_latent`. The DiT forward mixes both based on `denoise_mask`, producing an input where some positions are noised and others aren't.

## 类比 / The analogy

想象你在教一个画家临摹:你把"要画的空白画布"(加噪的目标 latent)和"参考画作"(干净的参考 token)一起放在桌上。画家可以自由看任何地方——他的眼睛就是 DiT 的 self-attention,没有物理隔板把画布和参考作品分开。关键是参考画作全程不会被泼墨——它一直保持干净。`strength` 控制"参考画用墨水盖掉多少":0 = 完全盖住(画家看不到参考),1 = 完全不盖(画家始终能清楚地看到参考)。

Imagine teaching a painter to copy from a reference: you place the "blank canvas" (noised target latent) and the "reference painting" (clean reference tokens) side by side on the same table. The painter's eyes — DiT self-attention — can look anywhere across both surfaces with no partition. The key: the reference painting never gets ink splashed on it — it stays clean throughout. `strength` controls "how much the reference is covered by ink before the painter sees it": 0 = fully covered (painter can't see reference), 1 = fully uncovered (painter always sees it clearly).

## 自己跑一遍 / Try it yourself

```python
import torch

B, N_tgt, N_ref, D = 2, 128, 32, 64

# 目标 token (noisy latent) + denoise_mask = 1.0
tgt_latent = torch.randn(B, N_tgt, D)
tgt_mask   = torch.ones(B, N_tgt, 1)

# 参考 token (clean) + denoise_mask = 1.0 - strength
strength   = 1.0
ref_tokens = torch.randn(B, N_ref, D)  # 干净的参考
ref_mask   = torch.full((B, N_ref, 1), fill_value=1.0 - strength)

# IC-LoRA: cat both
combined_latent = torch.cat([tgt_latent, torch.zeros_like(ref_tokens)], dim=1)
combined_clean  = torch.cat([tgt_latent, ref_tokens], dim=1)   # clean_latent
combined_mask   = torch.cat([tgt_mask,   ref_mask],   dim=1)

print("combined seq len:", combined_latent.shape[1])   # 160
print("ref portion mask (should be 0.0):", combined_mask[:, N_tgt:].unique())
print("tgt portion mask (should be 1.0):", combined_mask[:, :N_tgt].unique())
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
combined seq len: 160
ref portion mask (should be 0.0): tensor([0.])
tgt portion mask (should be 1.0): tensor([1.])
```

中文: 合并后的序列长度是目标 + 参考的总和。mask 精确区分了"要去噪的 token"和"保持干净的 token"。

English: The combined sequence length is target + reference tokens. The mask precisely partitions "tokens to denoise" from "tokens to keep clean".

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Inpainting mask in DDPM** / **DDPM 的 inpainting mask**: 最早的想法形式——已知区域的 mask=0(不加噪),待补全区域 mask=1(正常扩散)。IC-LoRA 本质是把这个 inpainting 思路推广到"参考视频作为已知区域"。/ The earliest form of this idea — known-region mask=0 (no noise), unknown-region mask=1 (normal diffusion). IC-LoRA generalizes this to "reference video as known region".
- **ControlNet** / **ControlNet**: 用 zero-conv 把控制信号注入 UNet 中间层——需要额外的网络参数。IC-LoRA 用的是 DiT 已有的 self-attention,只需在序列层面 append,无需新模块。/ Injects control signals into UNet middle layers via zero-convolutions — requires extra network parameters. IC-LoRA exploits DiT's existing self-attention; only sequence-level append, no new module.
- **图像生成里的 in-context learning** / **In-context learning for image generation**: 把"示例图像 + 目标 slot"拼在一起让模型处理——和 IC-LoRA 的序列拼接几乎完全相同的思路,只是 IC-LoRA 加了 denoise_mask 明确区分"参考"和"目标"。/ Concatenates example images with a target slot for the model to fill in — nearly identical to IC-LoRA's sequence concatenation, but IC-LoRA adds `denoise_mask` to explicitly partition "reference" from "target".

## 注意事项 / Caveats / when it breaks

- **序列长度线性增长** / **Sequence length grows linearly**: 参考视频越长,DiT 处理的 token 数越多,attention 的计算量二次增长。Lightricks 用低分辨率参考 (`downscale_factor=2`) 控制这个开销。/ Longer reference video → more tokens → quadratic attention cost. Lightricks controls this by using lower-resolution references (`downscale_factor=2`).
- **`downscale_factor` 必须匹配训练时的值** / **`downscale_factor` must match the training-time value**: LoRA 是在固定的 `downscale_factor` 下训练的。推理时用不同的值会导致 RoPE 位置错位,生成质量下降。数值存在 LoRA metadata 里。/ The LoRA was trained at a fixed `downscale_factor`. Using a different value at inference misaligns RoPE positions, degrading output quality. The value is stored in the LoRA metadata.
- **IC-LoRA ≠ 普通 LoRA** / **IC-LoRA ≠ vanilla LoRA**: 普通 LoRA 只微调 attention weight;IC-LoRA 的训练数据格式改变了(sequence 里多了参考 token),所以 IC-LoRA weight 不能直接套在 base model 的标准推理流程里——需要 `VideoConditionByReferenceLatent.apply_to` 先修改 latent 序列。/ Standard LoRA only fine-tunes attention weights; IC-LoRA changes the training data format (extra reference tokens in sequence), so IC-LoRA weights can't be used with the base model's standard inference pipeline — `apply_to` must first modify the latent sequence.

## 延伸阅读 / Further reading

- LTX-2 项目 (7772 stars): [GitHub Lightricks/LTX-2](https://github.com/Lightricks/LTX-2)
- IC-LoRA 思路相关: [In-Context LoRA for Diffusion Transformers (arXiv 2410.23775)](https://arxiv.org/abs/2410.23775)
- Inpainting 原始思路: [Repaint (arXiv 2201.09865)](https://arxiv.org/abs/2201.09865)
