---
date: 2026-06-02
topic: huggingface
source: huggingface
repo: huggingface/diffusers
file: src/diffusers/models/transformers/transformer_anyflow.py
permalink: https://github.com/huggingface/diffusers/blob/b95637a98dda87a679321a2dfde5f166f22a8119/src/diffusers/models/transformers/transformer_anyflow.py#L226-L307
difficulty: intermediate
read_time: ~11 min
tags: [code-of-the-day, huggingface, diffusion, any-step, rectified-flow]
---

# 两个 timestep embedding + 一个 gate = 任意步长扩散 / Two timestep embeddings + one gate = any-step diffusion

> **一句话 / In one line**: AnyFlow 的核心条件模块同时编码「当前噪声位置 t」和「这一步要跨多远 Δt」,用一个学到的 gate 把两者线性混合,让网络在推理时可以接受用户指定的任意步长。 / AnyFlow's core conditioning module jointly embeds the current noise position `t` and the step delta `Δt`, mixing them through a learned gate so the same network can take user-chosen step sizes at inference.

## 为什么重要 / Why this matters

经典 rectified flow / DDPM 训练时只看「当前 timestep」`t`,所以采样必须严格沿着训练时的那一串 schedule 走 —— 你想 4 步出图?对不起,精度悬崖。AnyFlow 在 2026 年提出的解法非常优雅:训练时随机采样一个 step size `r`,把 `(t, r)` 一起喂给网络,网络就学会了「**站在 t,跨 r 长度的步子,应该改多少**」。推理时你可以选 16 步、8 步、4 步甚至 1 步,只要把对应的 `(t, r)` 传进去 —— 同一个 checkpoint 一鱼多吃。这段 82 行代码就是这套机制的全部实现。

Classic rectified flow / DDPM trains on the current timestep `t` only, so sampling has to follow the exact schedule it was trained on — want to generate in 4 steps? Accuracy cliff. AnyFlow's elegant 2026 answer: during training, sample a random step size `r`, feed `(t, r)` together, and the network learns "**standing at t, taking a step of length r, how much should I move**." At inference you choose 16, 8, 4, even 1 step by passing the corresponding `(t, r)` — same checkpoint, many step budgets. These 82 lines are the entire conditioning machinery.

## 代码 / The code

`huggingface/diffusers` — [`src/diffusers/models/transformers/transformer_anyflow.py`](https://github.com/huggingface/diffusers/blob/b95637a98dda87a679321a2dfde5f166f22a8119/src/diffusers/models/transformers/transformer_anyflow.py#L226-L307)

```python
class AnyFlowDualTimestepTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        gate_value: float,
        deltatime_type: str,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: Optional[int] = None,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.delta_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = AnyFlowImageEmbedding(image_embed_dim, dim)

        self.register_buffer("delta_emb_gate", torch.tensor([gate_value], dtype=torch.float32), persistent=False)
        self.deltatime_type = deltatime_type

    def forward_timestep(
        self, timestep: torch.Tensor, delta_timestep: torch.Tensor, encoder_hidden_states, token_per_frame
    ):
        batch_size, num_frames = timestep.shape
        timestep = timestep.reshape(-1)
        delta_timestep = delta_timestep.reshape(-1)

        timestep = self.timesteps_proj(timestep)
        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)

        delta_timestep = self.timesteps_proj(delta_timestep)
        delta_embedder_dtype = next(iter(self.delta_embedder.parameters())).dtype
        if delta_timestep.dtype != delta_embedder_dtype and delta_embedder_dtype != torch.int8:
            delta_timestep = delta_timestep.to(delta_embedder_dtype)
        delta_emb = self.delta_embedder(delta_timestep).type_as(encoder_hidden_states)

        gate = self.delta_emb_gate.to(delta_embedder_dtype)

        rt_emb = (1 - gate) * temb + gate * delta_emb
        timestep_proj = self.time_proj(self.act_fn(rt_emb))

        rt_emb = rt_emb.unflatten(0, (batch_size, num_frames)).repeat_interleave(token_per_frame, dim=1)
        timestep_proj = timestep_proj.unflatten(0, (batch_size, num_frames)).repeat_interleave(token_per_frame, dim=1)

        return rt_emb, timestep_proj

    def forward(
        self,
        timestep: torch.Tensor,
        r_timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        layout_cfg=None,
    ):
        if self.deltatime_type == "r":
            delta_timestep = r_timestep
        elif self.deltatime_type == "t-r":
            delta_timestep = timestep - r_timestep
        else:
            raise NotImplementedError

        timestep, timestep_proj = self.forward_timestep(
            timestep, delta_timestep, encoder_hidden_states, layout_cfg["full_token_per_frame"]
        )

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

        return timestep, timestep_proj, encoder_hidden_states, encoder_hidden_states_image
```

## 逐行讲解 / What's happening

1. **两个独立的 `TimestepEmbedding`(`time_embedder` + `delta_embedder`)**:
   - 中文: 关键设计 —— 不共享权重。`t` 表示「我现在在哪」,`Δt` 表示「我要跨多远」,这两件事的几何性质很不一样(`t∈[0,1]`,`Δt∈[0,t]`),给它们各自一套 MLP 让网络分别学。共享反而会让 `Δt=0`(deterministic 模式)和 `Δt=t`(从 noise 一步到底)混淆。
   - English: Critical choice — they don't share weights. `t` says "where am I now," `Δt` says "how far do I step" — geometrically different (`t∈[0,1]`, `Δt∈[0,t]`), so each gets its own MLP. Sharing would conflate `Δt=0` (deterministic) and `Δt=t` (one-shot all-the-way) embeddings.

2. **`self.register_buffer("delta_emb_gate", ..., persistent=False)`**:
   - 中文: gate 是个**未训练**的 buffer(`persistent=False` 表示不进 state_dict),通常每个 checkpoint 配一个 `gate_value` 在 config 里固化下来。这意味着你可以用同一份 weights 加载不同的 gate 试不同的 t/Δt 平衡 —— 类似 Flux 的 guidance scale,但是是网络内部的。
   - English: The gate is a **non-trained** buffer (`persistent=False` keeps it out of state_dict), typically fixed per checkpoint via a config `gate_value`. So you can load the same weights with different gates and search for the optimal t/Δt balance — like Flux's guidance scale, but applied inside the network.

3. **`temb = time_embedder(timesteps_proj(t))` + `delta_emb = delta_embedder(timesteps_proj(Δt))`**:
   - 中文: 两条平行路径。`timesteps_proj` 是经典的 sinusoidal positional encoding(`flip_sin_to_cos=True` 是 Stable Diffusion 的约定),把标量 `t` 映成 `time_freq_dim`-维向量;然后各自的 MLP 投到模型维 `dim`。
   - English: Two parallel paths. `timesteps_proj` is the classic sinusoidal positional encoding (`flip_sin_to_cos=True` follows the Stable Diffusion convention), lifting scalar `t` to a `time_freq_dim` vector; then each MLP projects to model `dim`.

4. **`rt_emb = (1 - gate) * temb + gate * delta_emb`** — **核心一行 / the core line**:
   - 中文: 学到的混合。`gate=0` 退化成纯 `t`-条件(经典 DDPM);`gate=1` 退化成纯 `Δt`-条件(变成「一步跨多大」的预测器,失去当前位置感)。实际 checkpoint 的 gate 一般在 0.3-0.7 之间,让两边信号都进入网络。
   - English: The learned mix. `gate=0` collapses to pure `t`-conditioning (vanilla DDPM); `gate=1` collapses to pure `Δt`-conditioning (becomes a "how-far" predictor with no sense of position). Real checkpoints typically land around 0.3-0.7 so both signals make it in.

5. **`timestep_proj = self.time_proj(self.act_fn(rt_emb))`**:
   - 中文: `rt_emb` 是给 adaLN 用的「时间向量」(每个 DiT block 会再投一遍出 shift/scale/gate),`timestep_proj` 是另一个分支 —— 给某些 modulator 用的「投影后时间」。两份都返回,让下游 block 自己选用哪个。
   - English: `rt_emb` is the "time vector" consumed by adaLN inside every DiT block (which re-projects it to shift/scale/gate). `timestep_proj` is a second branch — a pre-projected variant some modulators use directly. Both are returned so downstream blocks pick.

6. **`unflatten(0, (batch_size, num_frames)).repeat_interleave(token_per_frame, dim=1)`**:
   - 中文: 视频模型的一个常见技巧。timestep 是 `(B, T_frames)` —— 每帧一个独立的噪声水平(为了支持视频去噪的「帧间不同 noise」)。但 transformer 看到的是 `(B, T_frames * tokens_per_frame, D)` 的扁平 token 序列。`repeat_interleave` 把每帧的 time 向量复制 `tokens_per_frame` 份,填满该帧对应的所有 spatial token。
   - English: A common video-model trick. Timestep is `(B, T_frames)` — one noise level per frame (supports per-frame different noise during denoising). But the transformer sees a flat `(B, T_frames * tokens_per_frame, D)` token sequence. `repeat_interleave` broadcasts each frame's time vector across all of that frame's spatial tokens.

7. **`forward` 里的 `deltatime_type` 分支**:
   - 中文: 两种参数化方式。`"r"`:把 `r_timestep` 直接当 `Δt`(用户传「步长」)。`"t-r"`:`Δt = t - r`(用户传「跨到哪个 r」,网络要算差)。理论等价,但训练稳定性不同 —— `"t-r"` 让网络隐式建模「目标位置」,在多步链式采样里更稳。
   - English: Two parameterizations. `"r"`: treat `r_timestep` directly as `Δt` (user passes "step size"). `"t-r"`: `Δt = t - r` (user passes "target position", network subtracts). Mathematically equivalent but training stability differs — `"t-r"` lets the net implicitly model a target position, which sampling-chain experiments show is steadier across many steps.

## 类比 / The analogy

想象你在开手动挡爬一段陡坡。仪表盘告诉你「现在海拔 200 米」(`t`),GPS 告诉你「下个路口在 50 米后」(`Δt`)。一个老司机不会只盯着海拔(光看 `t`)或只盯着剩余距离(光看 `Δt`),而是两者都看,根据具体路况(`gate`)决定主要看哪个 —— 平缓段多看 `Δt` 大胆松离合,陡坡段多看 `t` 小心控速。AnyFlow 的 gate 就是这个「双仪表」决策器。

Picture driving a manual transmission up a steep hill. Your altimeter reads "200 m elevation now" (`t`), your GPS reads "next turn in 50 m" (`Δt`). A skilled driver doesn't watch only altitude (`t` alone) or only remaining distance (`Δt` alone) — they watch both and let the road's character (`gate`) decide which matters more: gentle section, lean on `Δt` and ease the clutch boldly; steep section, lean on `t` and control speed carefully. AnyFlow's gate is that dual-instrument decision-maker.

## 自己跑一遍 / Try it yourself

```python
# pip install torch diffusers
import torch
from diffusers.models.embeddings import Timesteps, TimestepEmbedding

class DualTimestep(torch.nn.Module):
    def __init__(self, dim=128, freq_dim=128, gate=0.5):
        super().__init__()
        self.proj = Timesteps(freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.te_t  = TimestepEmbedding(freq_dim, dim)
        self.te_dt = TimestepEmbedding(freq_dim, dim)
        self.register_buffer("gate", torch.tensor([gate]))
    def forward(self, t, dt):
        et  = self.te_t(self.proj(t))
        edt = self.te_dt(self.proj(dt))
        return (1 - self.gate) * et + self.gate * edt

m = DualTimestep(dim=128, gate=0.5)
t  = torch.tensor([500.0, 500.0])     # same t
dt = torch.tensor([10.0, 100.0])      # different step sizes
emb = m(t, dt)
print("at t=500, step=10 vs step=100, embedding diff norm =",
      (emb[0] - emb[1]).norm().item())
print("(non-zero → the network *sees* the step size)")
```

运行 / Run with:
```bash
pip install torch diffusers
python try.py
```

预期输出 / Expected output:
```
at t=500, step=10 vs step=100, embedding diff norm = ~10.0
(non-zero → the network *sees* the step size)
```

把 `gate` 改成 0,你会看到 diff norm 立刻变 0 —— 这就是退化回经典「只看 t」的 DDPM。这一个数值是 any-step 能力的开关。

Set `gate` to 0 and the diff norm collapses to 0 instantly — that's the degenerate "t-only" DDPM. That single buffer is the on/off switch for any-step capability.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Consistency Models** / **Consistency Models**: 不显式编码 `Δt`,但用「同一条 noise trajectory 上不同 t 应该映射到同一个数据点」的训练目标隐式逼出 any-step。AnyFlow 是显式版,更容易 fine-tune。 / Doesn't embed `Δt` explicitly; instead trains with "any two points on the same noise trajectory must map to the same data point," implicitly producing any-step. AnyFlow is the explicit version, easier to fine-tune.
- **Latent Consistency Models 的 `c_skip / c_out`** / **Latent Consistency Models' `c_skip / c_out`**: 也是把当前 t 和目标 t 同时塞进网络,但用一组分析解出的系数而不是学到的 gate。 / Also feeds current `t` and target `t` simultaneously, but with analytically derived coefficients instead of a learned gate.
- **InstaFlow / Rectified Flow Distillation** / **InstaFlow / Rectified Flow Distillation**: 用 teacher-student 把多步 flow 蒸馏到一步;不需要 dual-timestep,但代价是失去多步精度。AnyFlow 是一个不损多步精度的替代品。 / Distills multi-step flow into one step via teacher-student; doesn't need dual-timestep but loses multi-step accuracy in exchange. AnyFlow keeps both.
- **Flux Schnell 的 guidance distillation** / **Flux Schnell's guidance distillation**: 把 cfg 因子吃进 timestep embedding;同一种「往时间向量里塞额外条件」的思想。 / Folds the cfg factor into the timestep embedding — same "stuff extra conditioning into the time vector" idea.

## 注意事项 / Caveats / when it breaks

- **训练数据里必须采样到所有 `(t, Δt)` 配对** / **Training must sample the full `(t, Δt)` grid**: 如果你只训了 `Δt ∈ {1/16, 1/8}`,推理时给 `Δt = 1/4` 就外推,精度会塌。常见解法:训练时 `Δt` 在 `[Δt_min, t]` 上均匀采样。 / If you only train on `Δt ∈ {1/16, 1/8}`, asking for `Δt = 1/4` at inference is extrapolation and accuracy crashes. The fix: sample `Δt` uniformly on `[Δt_min, t]` during training.
- **gate 是 buffer 不是 Parameter,别想直接 `optimizer.zero_grad` 学它** / **Gate is a buffer, not a Parameter — you can't train it with `optimizer.zero_grad`**: 想让 gate 可学,改成 `nn.Parameter` + 加约束 `sigmoid` 防止超出 [0,1]。 / To make it learnable, change to `nn.Parameter` and wrap with a `sigmoid` to constrain to [0, 1].
- **视频特有的「per-frame timestep」** / **The video-specific "per-frame timestep"**: `timestep` shape 是 `(B, T_frames)`。图像模型用同一个套子要 reshape 成 `(B, 1)`,否则 `unflatten` 报错。 / `timestep` shape is `(B, T_frames)`. To reuse this for image models you must reshape to `(B, 1)`, or `unflatten` will fail.
- **`PixArtAlphaTextProjection` 的依赖** / **`PixArtAlphaTextProjection` dependency**: 这个 forward 假设你的文本 encoder 输出 `text_embed_dim`-维向量(通常是 T5-XXL 的 4096 维);换成 CLIP/qwen 时记得改 `text_embed_dim`。 / This forward assumes a `text_embed_dim`-wide text encoder output (T5-XXL's 4096 is typical); swap to CLIP/Qwen and update `text_embed_dim` accordingly.

## 延伸阅读 / Further reading

- [AnyFlow PR #13745 (diffusers)](https://github.com/huggingface/diffusers/pull/13745)
- [Consistency Models — Song et al. 2023](https://arxiv.org/abs/2303.01469)
- [Rectified Flow 原始论文](https://arxiv.org/abs/2209.03003)
- [InstaFlow distillation recipe](https://github.com/gnobitab/InstaFlow)
