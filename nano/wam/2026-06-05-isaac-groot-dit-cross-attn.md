---
date: 2026-06-05
topic: wam
source: wam
repo: NVIDIA/Isaac-GR00T
file: gr00t/model/modules/dit.py
permalink: https://github.com/NVIDIA/Isaac-GR00T/blob/626af89d3e914ec92eab5323e23b9ed44a7b26c8/gr00t/model/modules/dit.py#L222-L336
difficulty: advanced
read_time: ~12 min
tags: [code-of-the-day, wam, vla, dit, cross-attention, adaLN]
build_role: dit-block (cross-repo variant) — alternate DiT topology where conditioning enters via cross-attention to encoder features instead of in-block AdaLN-Zero. Bridges the WAM `dit-block` and VLA `action-head` slots.
---

# 同一个 DiT 骨架,两种条件注入方式:GR00T 的 cross-attn 变体 / Same DiT skeleton, two conditioning strategies: GR00T's cross-attention variant

> **一句话 / In one line**: 经典 DiT 把"timestep + 文本"通过 AdaLN-Zero 在**每个 block 内部**调制激活;GR00T 把它改成**每个 block 都 cross-attend 到 VLM features**,AdaLN 只在输出处出现一次 — 同一个骨架,完全不同的条件拓扑。 / Classic DiT modulates activations via AdaLN-Zero **inside every block** for timestep+text conditioning; GR00T instead **cross-attends to VLM features at every block** and uses AdaLN only once at the output — same skeleton, completely different conditioning topology.

## 为什么重要 / Why this matters

如果你只看过 facebookresearch/DiT(2026-05-25 那期讲过),你会以为"DiT block = self-attn + AdaLN-Zero",条件都是 timestep 一个向量经过 MLP 调出 6 个 modulation 参数注入到 LayerNorm。但 robotics 场景下条件不是一个 (B, D) 向量,而是 VLM 输出的整个 token 序列 (B, S, D)(图像 + 文本指令一起)— AdaLN-Zero 装不下这么多信息。NVIDIA 在 GR00T 里给出了一种很干净的替代:**每个 block 直接 cross-attention 到 VLM features**,timestep 单独走 AdaLN 但只在最后一层做。这个变体回答了一个核心问题:"我已经有一个强大的 VLM,怎么让 DiT 优雅地消费它的输出?"

If you've only read facebookresearch/DiT (covered 2026-05-25), you'll think "DiT block = self-attn + AdaLN-Zero", with the timestep flowing through an MLP that emits 6 modulation parameters injected into LayerNorm. But in robotics, conditioning isn't a single `(B, D)` vector — it's the *entire token sequence* from a VLM `(B, S, D)` (image + instruction together). AdaLN-Zero can't carry that much information. NVIDIA's GR00T offers a clean alternative: **cross-attention to VLM features at every block**, with timestep AdaLN appearing only at the output. This variant answers a key question: "I already have a powerful VLM — how does a DiT consume its output cleanly?"

## 代码 / The code

`NVIDIA/Isaac-GR00T` — [`gr00t/model/modules/dit.py`](https://github.com/NVIDIA/Isaac-GR00T/blob/626af89d3e914ec92eab5323e23b9ed44a7b26c8/gr00t/model/modules/dit.py#L222-L336)

```python
class DiT(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,          # robot action dimensions, not image patches
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        upcast_attention: bool = False,
        norm_type: str = "ada_norm",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        interleave_self_attention=False,
        cross_attention_dim: Optional[int] = None,
    ):
        super().__init__()
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.timestep_encoder = TimestepEncoder(embedding_dim=self.inner_dim, ...)

        all_blocks = []
        for idx in range(self.config.num_layers):
            use_self_attn = idx % 2 == 1 and interleave_self_attention
            curr_cross_attention_dim = cross_attention_dim if not use_self_attn else None
            all_blocks += [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    norm_type=norm_type,                 # "ada_norm" → AdaLN inside block
                    norm_eps=self.config.norm_eps,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=self.config.max_num_positional_embeddings,
                    final_dropout=final_dropout,
                    cross_attention_dim=curr_cross_attention_dim,   # ← key wiring
                )
            ]
        self.transformer_blocks = nn.ModuleList(all_blocks)

        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)   # → (shift, scale)
        self.proj_out_2 = nn.Linear(self.inner_dim, self.output_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,            # (B, T, D)   noisy action tokens
        encoder_hidden_states: torch.Tensor,    # (B, S, D)   VLM features
        timestep: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_all_hidden_states: bool = False,
    ):
        temb = self.timestep_encoder(timestep)                  # (B, D)

        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        all_hidden_states = [hidden_states]

        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1 and self.config.interleave_self_attention:
                # Self-attention only block (no encoder context)
                hidden_states = block(hidden_states, attention_mask=None,
                                      encoder_hidden_states=None,
                                      encoder_attention_mask=None, temb=temb)
            else:
                # Cross-attention to VLM features
                hidden_states = block(hidden_states, attention_mask=None,
                                      encoder_hidden_states=encoder_hidden_states,
                                      encoder_attention_mask=None, temb=temb)
            all_hidden_states.append(hidden_states)

        # Final AdaLN-Zero at the output, then project to action_dim
        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        if return_all_hidden_states:
            return self.proj_out_2(hidden_states), all_hidden_states
        return self.proj_out_2(hidden_states)
```

## 逐行讲解 / What's happening

1. **`output_dim: int = 26`**:
   - 中文: 一眼看出"这不是图像 DiT"。26 是 GR00T 的机器人 action 维数(双臂关节角 + gripper)。同一个 DiT 骨架,在 image WAM 里 `output_dim` 是 patch_size² × 3,在这里就是动作维度。
   - English: a dead giveaway that this isn't an image DiT. 26 is GR00T's robot action dim (bimanual joints + gripper). Same DiT skeleton: in an image WAM `output_dim` is patch_size² × 3; here it's an action vector.

2. **`cross_attention_dim`** 决定 block 的拓扑:
   - 中文: 在 `BasicTransformerBlock` 里,只要 `cross_attention_dim != None`,block 的 `self.attn1` 就变成 cross-attention(query 来自 hidden_states,K/V 来自 `encoder_hidden_states`)。GR00T 默认每个 block 都跟 VLM feature cross-attend,这是和经典 DiT 最大的拓扑差异。
   - English: in `BasicTransformerBlock`, if `cross_attention_dim != None`, `self.attn1` becomes cross-attention (query from hidden_states, K/V from `encoder_hidden_states`). GR00T defaults to cross-attending to VLM features in every block — the biggest topological difference from classic DiT.

3. **`interleave_self_attention`**:
   - 中文: 一个 flag,把奇数 block 的 cross-attention 改成 self-attention。等价于"双层结构":偶数层听 VLM 说话,奇数层让 action token 内部互相讨论。FastDiT 和 Stable Diffusion 3 都用这个模式。
   - English: flips odd-indexed blocks from cross- to self-attention. Equivalent to a "two-layer" structure: even layers listen to the VLM, odd layers let action tokens deliberate among themselves. FastDiT and Stable Diffusion 3 use this pattern.

4. **`temb = self.timestep_encoder(timestep)`** + 只在输出处 AdaLN:
   - 中文: 经典 DiT 在每个 block 内部都做 AdaLN-Zero(每层 6 个 modulation 参数),非常重量级。GR00T 把 timestep 信息延迟到最后一层 `norm_out * (1 + scale) + shift`。理由是 cross-attention 已经在每个 block 把"VLM 上下文"注入了,timestep 这种单向量只需要一个最终的微调即可。参数大幅减少。
   - English: classic DiT does AdaLN-Zero inside every block (6 modulation params per layer), heavy. GR00T defers timestep info to a single final `norm_out * (1 + scale) + shift`. The rationale: cross-attention has already injected VLM context at every block, so the timestep (a tiny per-step scalar context) only needs a single tail modulation. Drastically fewer parameters.

5. **`hidden_states.contiguous()`** 调用两次:
   - 中文: 工程细节。Cross-attention 的 K/V 来自 `encoder_hidden_states`,如果调用方传进来一个 view(比如 `transpose` 之后的 tensor),底层 SDPA kernel 在某些版本上会慢甚至报错。提前 `contiguous()` 是 defensive coding。
   - English: defensive engineering. Cross-attention's K/V comes from `encoder_hidden_states`; if the caller passes a view (e.g. post-`transpose`), some SDPA kernels slow down or crash. Forcing `contiguous()` up-front sidesteps it.

6. **`proj_out_1(SiLU(temb)).chunk(2, dim=1)`**:
   - 中文: 标准 AdaLN-Zero 的尾巴 — 一个 Linear 同时产出 shift 和 scale,SiLU + `chunk(2)`。`norm_out` 用 `elementwise_affine=False`(自己没有可学习 γ/β),因为 γ/β 全部由 timestep 动态生成。
   - English: standard AdaLN-Zero tail — one Linear emits both shift and scale, separated by `chunk(2)`. `norm_out` uses `elementwise_affine=False` (no learnable γ/β) because γ/β are dynamically produced from the timestep.

7. **`all_hidden_states.append(hidden_states)`**:
   - 中文: 把每个 block 的输出都收集起来。下游可以做"deep supervision"(每层都加 loss)或者 distillation。在生产 VLA 里这个 list 是 debug 神器。
   - English: collects every block's output. Downstream can do deep supervision (per-layer loss) or distillation; in a production VLA this list is a debugging gift.

## 类比 / The analogy

想象一个翻译团队在把英语小说翻成中文。**经典 DiT** 像每个翻译员都戴着一个耳机,耳机里反复播放"现在是第 t 步去噪,情绪稍微再激烈一点 / 平静一点"这种全局指令(AdaLN-Zero)。翻译员之间互相讨论用词(self-attention),但**没人能看原文**。**GR00T DiT** 把原文(VLM features)放在每个翻译员桌上,他们随时可以瞄一眼(cross-attention),全局情绪调节(timestep)只在最后一稿统一润色(只在 norm_out 做 AdaLN)。这种设计的代价是每个 block 都要做 cross-attention,慢一点;好处是条件信息容量大得多。

Picture a team translating an English novel into Chinese. **Classic DiT** is like every translator wearing an earpiece replaying global directions ("now we're at denoising step t, dial up the intensity a notch") — that's AdaLN-Zero injecting per-step modulation everywhere. Translators discuss word choice with each other (self-attention), but **nobody sees the source text**. **GR00T DiT** puts the source text (VLM features) on every translator's desk so they can glance whenever needed (cross-attention); global mood adjustments (timestep) only happen on the final polish (AdaLN at the output). The cost is per-block cross-attention overhead; the gain is far more conditioning capacity.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-WAM

**Curriculum item**: `dit-block` (already covered 2026-05-25 with the image DiT). Today is a **cross-repo variant** — same `dit-block` slot, alternate topology. Depends on `patchify-positional` (covered 2026-05-29).

中文:课程里 `dit-block` 这一格我们已经填过(图像 DiT 的 AdaLN-Zero block)。今天补充一种"VLM-conditioned"变体,在 nanoVLA 和 nanoWAM 之间架了一座桥:

- 在 **nanoWAM**(image / video world model)里,经典 DiT block(facebookresearch/DiT 那种)是默认选择 — 条件少(timestep + 文本 pooled embedding),AdaLN-Zero 够用。
- 在 **nanoVLA** 的 *action head* 里(课程下一步 `action-head-continuous`),你需要把 VLM 的 image+text token 序列当条件 — 此时 GR00T 这种 cross-attention DiT 才合适。它本质上是个"action-domain DiT",输出是动作而非像素。
- 如果你想做 **video WAM with action conditioning**(完整的 V-W-A-M),你会发现需要混合两种:对 video latent 做 patchify 后用 cross-attn 看 action token,然后用 AdaLN 处理 timestep。这种"混合 DiT"就是 GR00T 这段代码教你怎么写。

省掉这个变体,你的 nanoVLA action head 就只能用 MLP(把 VLM features pooled 成一个 D 维向量 + timestep + MLP)— 但这就丢掉了 image token 的空间信息,典型表现是抓取定位不准。

English: the curriculum's `dit-block` slot is already filled (image DiT with AdaLN-Zero). Today's note is a **cross-repo variant** — same slot, alternate topology — that bridges nanoVLA and nanoWAM:

- In **nanoWAM** (image/video world model), classic DiT (facebookresearch/DiT style) is the default — conditioning is small (timestep + pooled text), so AdaLN-Zero suffices.
- In **nanoVLA**'s *action head* (next curriculum item `action-head-continuous`), you must condition on the VLM's image+text token sequence — that's exactly where GR00T's cross-attention DiT fits. It's essentially an "action-domain DiT" emitting actions rather than pixels.
- For a **video WAM with action conditioning** (the full V-W-A-M), you'll find yourself hybridizing: patchify video latents, cross-attend to action tokens, AdaLN the timestep. That hybrid is exactly what GR00T's code teaches you how to wire.

Skip this variant and your nanoVLA action head can only use an MLP (pool VLM features → concat with timestep → MLP), which discards image-token spatial information — typical failure mode is poor grasp localization.

## 自己跑一遍 / Try it yourself

```python
# mini_groot_dit.py — pip install torch
import torch, torch.nn as nn, torch.nn.functional as F, math

class TimestepEmb(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(256, D), nn.SiLU(), nn.Linear(D, D))
    def forward(self, t):
        half = 128
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t[:, None].float() * freqs[None]
        emb = torch.cat([a.sin(), a.cos()], dim=-1)         # (B, 256)
        return self.mlp(emb)

class CrossAttnBlock(nn.Module):
    def __init__(self, D, H, S_dim):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.q  = nn.Linear(D, D)
        self.kv = nn.Linear(S_dim, 2 * D)
        self.o  = nn.Linear(D, D)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
        self.H = H
    def forward(self, x, enc):
        B, T, D = x.shape
        q = self.q(self.ln1(x)).view(B, T, self.H, D // self.H).transpose(1, 2)
        kv = self.kv(enc).view(B, enc.size(1), 2, self.H, D // self.H)
        k, v = kv.permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, T, D)
        x = x + self.o(a)
        return x + self.mlp(self.ln2(x))

class MiniGR00TDiT(nn.Module):
    def __init__(self, D=128, H=4, N=4, S_dim=192, out_dim=26):
        super().__init__()
        self.t_enc = TimestepEmb(D)
        self.blocks = nn.ModuleList([CrossAttnBlock(D, H, S_dim) for _ in range(N)])
        self.norm_out = nn.LayerNorm(D, elementwise_affine=False)
        self.proj1 = nn.Linear(D, 2 * D)
        self.proj2 = nn.Linear(D, out_dim)
    def forward(self, x, vlm_feats, t):
        temb = self.t_enc(t)                                # (B, D)
        for blk in self.blocks:
            x = blk(x, vlm_feats)
        shift, scale = self.proj1(F.silu(temb)).chunk(2, dim=-1)
        x = self.norm_out(x) * (1 + scale[:, None]) + shift[:, None]
        return self.proj2(x)

B, T_actions, S_tokens = 2, 8, 196
model = MiniGR00TDiT()
noisy_actions = torch.randn(B, T_actions, 128)              # noisy action tokens
vlm_feats     = torch.randn(B, S_tokens, 192)               # ViT output we saw in vla note
timestep      = torch.randint(0, 1000, (B,))
out = model(noisy_actions, vlm_feats, timestep)
print("action prediction shape:", out.shape)
```

运行 / Run with:
```bash
python mini_groot_dit.py
```

预期输出 / Expected output:
```
action prediction shape: torch.Size([2, 8, 26])
```

中文:输入 8 个带噪 action token 和 196 个 VLM feature token,输出每个 action token 的 26 维去噪结果。这就是一个 minimal 的 cross-attention DiT action head — 把它接到 nanoVLA 的 ViT(昨天的 vision-encoder)+ modality projector + LM backbone 后面,你就有一个完整的 VLA 推理路径。

English: input is 8 noisy action tokens and 196 VLM feature tokens; output is a 26-dim denoised action per token. This is a minimal cross-attention DiT action head — plug it after your nanoVLA's ViT (today's vision-encoder) + modality projector + LM backbone and you have a complete VLA inference path.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **facebookresearch/DiT** / **facebookresearch/DiT** (covered 2026-05-25): the AdaLN-Zero-everywhere variant. Pure self-attention, conditioning is just timestep + class. Same building block, opposite conditioning strategy.
- **Stable Diffusion 3 (MM-DiT)** / **SD3 MM-DiT**: concatenates image and text tokens into one sequence and uses full self-attention. Yet another resolution of the "how to inject text into DiT" question.
- **π₀ (Physical-Intelligence/openpi)** / **π₀**: action head uses an "action expert" transformer that cross-attends to PaliGemma's KV cache. Different parameter layout, same idea — actions cross-attend to a VLM.
- **Lumina-Next / Hunyuan-DiT** / **Lumina-Next, Hunyuan-DiT**: cross-attention to T5 text embeddings at every block (no AdaLN). Closer to GR00T's topology than to facebookresearch/DiT.

## 注意事项 / Caveats / when it breaks

- **Cross-attention 显存爆炸** / **cross-attention memory blowup**: 每个 block 都做 cross-attn,VLM features 序列长度 S 大时(像 SigLIP 14×14×N_camera = 通常 196–784),显存随 N_blocks × S 涨。建议用 FlashAttention 2/3 的 cross-attn 内核。 / Every block does cross-attn, so memory grows with `N_blocks × S` when the VLM feature length is large (typically 196–784 for SigLIP at multi-cam). Use FlashAttention 2/3's cross-attn kernel.
- **Timestep 信号容易被淹没** / **timestep signal can drown**: GR00T 把 timestep 留到最后才注入,有人发现训练初期 timestep 学得慢、容易 collapse 成"忽略 t"。补救:多复制几份 timestep token 也插进 encoder_hidden_states 一起 cross-attend。 / Deferring timestep to the tail means it can be drowned early in training; the model collapses to ignoring `t`. Fix: append a few timestep tokens to `encoder_hidden_states` so cross-attention also sees them.
- **`interleave_self_attention` 必须配 positional embed** / **`interleave_self_attention` needs pos embed**: 奇数 self-attn block 会丢失对 action token 顺序的感知,如果你没在 hidden_states 上加 positional embedding(action chunk index),输出会乱序。 / Odd-indexed self-attn blocks lose action-token ordering; without positional embeddings on `hidden_states` (action chunk index) the output is permutation-invariant.
- **`norm_elementwise_affine=False` 是 AdaLN 的硬性要求** / **`norm_elementwise_affine=False` is required**: 因为 γ/β 全部从 timestep 动态生成。如果你忘了关 `elementwise_affine`,会出现"两个 γ"(LN 自己的 + AdaLN 算的),训练直接发散。 / γ/β are entirely generated from the timestep; if `elementwise_affine=True` you have *two* γ's (LN's own + AdaLN's), and training diverges immediately.

## 延伸阅读 / Further reading

- [Scalable Diffusion Models with Transformers (DiT paper)](https://arxiv.org/abs/2212.09748) — the classic AdaLN-Zero variant
- [GR00T N1 — A Foundation Model for Generalist Humanoid Robots](https://research.nvidia.com/labs/gear/gr00t/) — the parent system this DiT is the action head of
- [π₀ paper (Physical Intelligence)](https://www.physicalintelligence.company/blog/pi0) — the closest sibling: cross-attention from action expert to VLM KV
- [Stable Diffusion 3 paper (MM-DiT)](https://arxiv.org/abs/2403.03206) — a third conditioning strategy
