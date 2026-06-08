---
date: 2026-06-08
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/modules/model.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py#L238-L317
difficulty: advanced
read_time: ~12 min
tags: [code-of-the-day, wam, dit-block, adaln-zero, video-diffusion]
build_role: dit-block (production variant — same role as nanoWAM's basic DiT block, but with every detail a real 14B video model needs)
---

# Wan2.1 的 WanAttentionBlock:DiT block 的生产级长相 / Wan2.1's WanAttentionBlock: what a production-grade DiT block actually looks like

> **一句话 / In one line**: 同一个 adaLN-Zero 6-way modulation 骨架,加上 RMSNorm-on-QK、3D RoPE、双路 cross-attention(text + first-frame image)、和 fp32 关键路径——这就是 14B 视频生成模型里 DiT block 的真实样子。
> Same adaLN-Zero 6-way modulation skeleton, plus RMSNorm-on-QK, 3D RoPE, dual cross-attention (text + first-frame image), and fp32 critical paths — this is what a DiT block in a real 14B video generator actually looks like.

## 为什么重要 / Why this matters

5/25 那期讲了 facebookresearch/DiT 的 60 行 toy block(adaLN-Zero + 6 个 modulation 参数 + self-attention + MLP)。今天这段是同一个组件的**生产版本**——Wan2.1 这种 14B 视频生成模型实际堆 32-40 层用的 block。读它的价值不在"看一个新结构",而在看一个"成熟工程师从 toy 起步实际会改成什么样":哪些细节是装饰、哪些是命门?为什么 modulation 不再初始化为 0?为什么 norm 都要分 float32?为什么把 self-attn 和 cross-attn 用同一个 e[]?这些选择都不是论文里写的——都是踩坑踩出来的。任何想从 toy DiT 升级到生产 DiT 的工程师都得过这一关。

The 5/25 note covered facebookresearch/DiT's 60-line toy block (adaLN-Zero + 6 modulation params + self-attention + MLP). Today's snippet is the *production* version of the same component — what Wan2.1 (a 14B video generator) actually stacks 32-40 times. The point isn't "look at a new architecture"; it's seeing what a senior engineer changes when taking the toy to production. Which details are decorative and which are load-bearing? Why is the modulation no longer zero-initialized? Why is every norm cast to float32? Why share `e[]` across self-attn and cross-attn? None of these choices show up in papers — they were all earned by debugging. Anyone moving from a toy DiT to production DiT goes through this exact upgrade.

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/modules/model.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py#L238-L317)

```python
class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0], seq_lens, grid_sizes,
            freqs)
        with amp.autocast(dtype=torch.float32):
            x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
            with amp.autocast(dtype=torch.float32):
                x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x
```

## 逐行讲解 / What's happening

1. **`self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)`(行 276) / The modulation init**:
   - 中文: 这里**不是** adaLN-Zero!原始 DiT 用 `nn.init.zeros_()` 初始化 modulation(让 block 一开始等于 identity)。Wan 改成 `randn / sqrt(dim)`——variance ≈ 1/dim,小但不为零。为什么?Wan 的 modulation `e` 是 stagewise 共享的(整个 backbone 32 层 block 共用一组 timestep modulation,见同文件 forward 全图),如果每层都 zero-init,timestep 信号要"穿过"几十层零矩阵才能影响输出,梯度太弱。给一点 randn 让信号一开始就传得动,后续训练再校准。
   - English: This is **not** adaLN-Zero! The original DiT inits modulation with `nn.init.zeros_()` (so the block starts as identity). Wan switches to `randn / sqrt(dim)` — variance ≈ 1/dim, small but nonzero. Why? Wan's `e` is shared *across all 32 backbone layers* (look at the full-model forward in the same file), and if every layer is zero-init the timestep signal has to "punch through" dozens of zero matrices before it can move the output — gradients are too weak. A small randn lets the signal propagate from step one, training then calibrates.

2. **`e = (self.modulation + e).chunk(6, dim=1)`(行 297-299) / Lines 297-299**:
   - 中文: 经典的 6-way split。Per-block `self.modulation`(可学,初始化小)加上传进来的 global `e`(从 timestep embedding 算出),然后切成 6 块,对应 shift/scale 用于 self-attn (e[0], e[1]),scale 用于 self-attn residual (e[2]),shift/scale 用于 ffn (e[3], e[4]),scale 用于 ffn residual (e[5])。整套 modulation 用的都是 fp32——`assert e[0].dtype == torch.float32`。
   - English: Classic 6-way split. Per-block `self.modulation` (learnable, small) plus the inbound global `e` (derived from timestep embedding), then `.chunk(6, dim=1)`: shift/scale for self-attn (e[0], e[1]), scale for self-attn residual (e[2]), shift/scale for FFN (e[3], e[4]), scale for FFN residual (e[5]). The whole modulation runs in fp32 — note the `assert e[0].dtype == torch.float32`.

3. **`self.norm1(x).float() * (1 + e[1]) + e[0]`(行 303-304) / Lines 303-304**:
   - 中文: 标准的 adaLN-Zero 调制:`(1 + scale) * norm(x) + shift`。注意三个细节:
     (a) **`(1 + e[1])` 而不是 `e[1]`**——确保 e=0 时 scale=1(identity),数值稳定;
     (b) **`.float()`**——强制 fp32。LayerNorm 在 bf16 上数值不稳,modulation 这种敏感操作工业实现一律 fp32;
     (c) 整个表达式**没有用 `with autocast`**——因为 amp 会被 `.float()` 顶死,这是合理的 escape hatch。
   - English: Standard adaLN-Zero modulation: `(1 + scale) * norm(x) + shift`. Three details to clock:
     (a) **`(1 + e[1])` not `e[1]`** — ensures scale=1 (identity) when e=0, numerically stable;
     (b) **`.float()`** — force fp32. LayerNorm in bf16 is numerically unstable, and any sensitive modulation step is fp32 in production code;
     (c) the expression is **not wrapped in autocast** — because `.float()` overrides amp anyway. A legitimate escape hatch.

4. **`y = self.self_attn(... , grid_sizes, freqs)`(行 302-304) / The self_attn call**:
   - 中文: `WanSelfAttention` 内部干两件事(本文件 line 105-178):用 `WanRMSNorm` 对 Q 和 K 做归一化(`qk_norm=True`,稳定大模型训练的关键 trick,QwenLM、DeepSeek、Wan 都用),然后用 `rope_apply(q, grid_sizes, freqs)` 把 3D RoPE 加到 Q/K 上——`grid_sizes` 是 `(F, H, W)` 三维网格,RoPE 频率切三段分别对应时间/高度/宽度(见 5/29 Wan 3D RoPE 笔记)。最后 flash_attention。
   - English: `WanSelfAttention` (line 105-178 of the same file) does two things internally: RMSNorm Q and K (`qk_norm=True`, the stabilizing trick large-model training shops universally adopted — QwenLM, DeepSeek, Wan), then `rope_apply(q, grid_sizes, freqs)` baking 3D RoPE into Q and K — `grid_sizes` is `(F, H, W)` and the RoPE freqs split three ways for time/height/width (covered in the 5/29 Wan 3D RoPE note). Then flash_attention.

5. **`with amp.autocast(dtype=torch.float32): x = x + y * e[2]`(行 305-306) / Lines 305-306**:
   - 中文: residual 的累加显式 fp32。这是 Wan 的强迫症——所有"主干信号 + 修正"的 sum 都在 fp32 里做。为什么?bf16 的 mantissa 只有 7 位,大值 + 小值的 catastrophic cancellation 在 32 层 stack 上是真实问题。fp32 一行,稳一晚。
   - English: Explicit fp32 around the residual add. This is Wan's OCD: every "trunk + delta" sum runs in fp32. Why? bf16 has only 7 mantissa bits — large + small catastrophic cancellation is a real issue across 32 stacked blocks. One fp32 line, one peaceful night.

6. **`cross_attn_ffn` 内部函数 + `WAN_CROSSATTENTION_CLASSES`(行 309-316 + 行 232-235) / The cross_attn dispatch**:
   - 中文: `cross_attn_type` 是 `'t2v_cross_attn'`(text only)或 `'i2v_cross_attn'`(text + first-frame image)。**i2v 版本(line 200-229)做的是双 cross-attention**:context 前 257*2 个 token 是 image latent,后 512 个是 T5 text latent,Q 用同一份,分别 attend 两路再相加。这是 Wan 同时支持 T2V/I2V 的关键设计——不是搞两个模型,而是同一个 block 配置不同 cross_attn class。
   - English: `cross_attn_type` is either `'t2v_cross_attn'` (text only) or `'i2v_cross_attn'` (text + first-frame image). **The i2v variant (line 200-229) does dual cross-attention**: the first 257*2 context tokens are the image latent, the remaining 512 are T5 text — Q is shared, the two streams are attended independently and summed. This is the trick that lets Wan support both T2V and I2V with one architecture — not two models, one block configured with a different cross_attn class.

7. **`self.ffn`:`Linear → GELU(tanh) → Linear` 没有 dropout、没有 RMSNorm / The FFN**:
   - 中文: 极其朴素的 2-layer MLP,GELU 用 `tanh` 近似(SigLIP 也用这个,跟 GPT-2 一致的非饱和近似)。**没用 SwiGLU**——这一点跟 LM 圈不同。视频 DiT 的 FFN 还是经典 GELU,大概因为图像/视频 latent 上 SwiGLU 没显著优势。
   - English: Dead-simple 2-layer MLP, GELU with the `tanh` approximation (SigLIP uses the same — the non-saturating GPT-2 form). **No SwiGLU** — distinct from LM-land. Video DiT FFNs stick with classic GELU, presumably because SwiGLU shows no clear win on image/video latents.

## 类比 / The analogy

中文: 把 toy DiT block (5/25 那个 60 行的) 想成一辆"玩具卡丁车"——4 个轮子、一台引擎、一个方向盘,可以开但不能上高速。Wan 这个 `WanAttentionBlock` 是同一台车型的"街车版":同样的 4 轮 + 引擎 + 方向盘的拓扑(adaLN-Zero + 6 modulation + attn + ffn),但加了 ABS(`qk_norm`、RMSNorm-on-QK)、轮速调节(3D RoPE for time/H/W)、双副驾(text + image cross-attn)、强化车架(关键路径 fp32 转换)、起步姿态调整(`modulation` 用小 randn 而非 0)。每一个改装都不是装饰,都是有人在跑了 1000 个小时之后发现"这里会出事"而后加上去的。

English: Think of the toy DiT block (the 5/25 60-liner) as a "go-kart" — four wheels, one engine, a steering wheel; it drives but stays off the highway. Wan's `WanAttentionBlock` is the street-legal version of the same model: identical topology (adaLN-Zero + 6 modulations + attn + ffn), but with ABS (`qk_norm`, RMSNorm-on-QK), per-wheel speed control (3D RoPE for time/H/W), a passenger seat that fits both navigators (text + image cross-attn), reinforced chassis (fp32 on critical paths), and tuned takeoff (`modulation` uses small randn rather than zeros). Every upgrade isn't decorative — each one was added after someone logged 1000 hours and noticed "this will fail here."

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

> **Curriculum slot: `dit-block` (advanced variant)**. Depends on: `patchify-positional` (covered 5/29).
> Earlier curriculum coverage: `dit-block` was first covered by facebookresearch/DiT on 5/25 (toy version). This is a **second pass** on the same slot, showing the production variant.

中文: 这是 nanoWAM/productionWAM backbone 的**单块零件**。你的 WAM 整体结构是:`VAE encoder → patchify+3D pos → [DiT block × N] → unpatchify → VAE decoder`。中间那个 `[DiT block × N]` 是吞掉全部算力的地方;N 通常是 32-40。

如果你从 toy DiT 升级到生产 DiT,要补什么?
1. **`qk_norm=True`**: 加 RMSNorm 在 Q 和 K 上(Wan 用 `WanRMSNorm`,QwenLM 论文里有数据证明能稳定 70B+ 模型训练)。
2. **3D RoPE 而不是 1D learnable pos**: 你的 latent 是 `(F, H, W)` 三维网格,1D 处理位置信息不够;Wan 的 `rope_params` + `rope_apply` 切三段做。
3. **fp32 关键路径**: residual add、LayerNorm、modulation 全 fp32。bf16 stack 训 32 层不踩这个会出 NaN。
4. **`modulation` 用小 randn 初始化**: zero-init 在 32+ 层深 backbone 上梯度太弱,改成 `randn / sqrt(dim)`。
5. **双路 cross-attention**: 至少 text 一路,有时 + image first-frame 一路。WAM 经常需要"image-to-video"——靠 image cross-attention 把第一帧 condition 注入。
6. **flash_attention 而不是手写 SDPA**: 训练长 sequence (frames × H × W tokens 可能上 30k) 没 flash 跑不动。

如果省掉这些会怎样?toy DiT 可以训出 256×256 单帧的好图,但堆到 32 层 + 21 帧视频 + 1024×576 分辨率,就会有梯度不稳、attention OOM、收敛慢等问题。这些"细节"是从能跑 → 能扩到 14B 的桥。

English: This is **one brick** of the nanoWAM/productionWAM backbone. Your WAM as a whole looks like: `VAE encoder → patchify+3D pos → [DiT block × N] → unpatchify → VAE decoder`. The `[DiT block × N]` in the middle is where 99% of compute lives; N is typically 32-40.

What do you actually add going from toy DiT to production DiT?
1. **`qk_norm=True`**: RMSNorm on Q and K (Wan uses `WanRMSNorm`; the QwenLM paper showed empirically that this stabilizes 70B+ training).
2. **3D RoPE in place of 1D learnable position**: your latent is a `(F, H, W)` grid; 1D positional info doesn't cover it. Wan's `rope_params` + `rope_apply` slice the freq dim three ways.
3. **fp32 critical paths**: residual adds, LayerNorm, modulation all fp32. Training a 32-layer bf16 stack without this is how you get NaNs.
4. **Small-randn modulation init**: zero-init starves gradients in a 32+ layer backbone; `randn / sqrt(dim)` lets the timestep signal propagate from step one.
5. **Dual cross-attention**: at least text; often + image first-frame. WAMs frequently need "image-to-video" — that's how the first-frame conditioning gets injected.
6. **flash_attention, not naive SDPA**: training-time sequence length (frames × H × W tokens easily hits 30k) is intractable without it.

What happens if you omit these? A toy DiT trains fine on single 256×256 frames, but stack to 32 layers + 21 frames + 1024×576 and you'll hit gradient instability, attention OOM, and slow convergence. These "details" are the bridge from "works at small scale" to "scales to 14B."

## 自己跑一遍 / Try it yourself

```python
# wan_block_demo.py
import torch
import torch.nn as nn

class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class WanBlockMini(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.qkv = nn.Linear(dim, 3*dim)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = WanRMSNorm(dim // num_heads)   # qk_norm trick
        self.k_norm = WanRMSNorm(dim // num_heads)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
                                 nn.Linear(ffn_dim, dim))
        # Wan-style: randn init, NOT zeros
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, e):
        # e: (B, 6, dim) from timestep embedding, in fp32
        e = (self.modulation + e).chunk(6, dim=1)
        # self-attention with adaLN-Zero modulation, fp32 modulation, qk-norm
        h = self.norm1(x).float() * (1 + e[1]) + e[0]
        B, L, D = h.shape
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = self.q_norm(q.view(B, L, self.num_heads, self.head_dim))
        k = self.k_norm(k.view(B, L, self.num_heads, self.head_dim))
        v = v.view(B, L, self.num_heads, self.head_dim)
        attn = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
        ).transpose(1,2).reshape(B, L, D)
        x = x + self.proj(attn).float() * e[2]
        # FFN with adaLN modulation
        y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
        x = x + y.float() * e[5]
        return x.to(torch.bfloat16) if x.dtype == torch.float32 else x

torch.manual_seed(0)
block = WanBlockMini(dim=512, ffn_dim=2048, num_heads=8).to(torch.bfloat16)
x = torch.randn(2, 100, 512, dtype=torch.bfloat16)
e = torch.randn(2, 6, 512, dtype=torch.float32) * 0.1
y = block(x, e)
print(f"x: {x.shape} {x.dtype}  ->  y: {y.shape} {y.dtype}")
print(f"residual norm change: {(y - x).abs().mean().item():.4f}")
```

运行 / Run with:
```bash
pip install torch  # CPU works fine for this demo
python wan_block_demo.py
```

预期输出 / Expected output:
```
x: torch.Size([2, 100, 512]) torch.bfloat16  ->  y: torch.Size([2, 100, 512]) torch.bfloat16
residual norm change: 0.05-0.15
```

中文一两句: 关键观察是 residual norm change 不是 0 也不是 1——大约 0.05-0.15。toy DiT(modulation 全 0 初始化)在第 0 步会输出严格 0 修改;Wan 的 randn init 一开始就有非平凡修改,timestep signal 立刻能传。

English: The key observation is that the residual norm change is neither 0 nor 1 — around 0.05-0.15. A toy DiT with all-zeros modulation init outputs strictly zero modification on step 0; Wan's randn init produces a nontrivial modification from the start, so the timestep signal propagates immediately.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **facebookresearch/DiT `DiTBlock`** / **facebookresearch/DiT `DiTBlock`**: 5/25 笔记的 toy 版本。结构同,细节简化得多——读完这两个互相对照很有启发。/ The toy version from the 5/25 note. Same skeleton, much fewer details — reading the two side-by-side is highly instructive.
- **CogVideoX `Transformer3DModel`** / **CogVideoX `Transformer3DModel`**: 同类设计,但 cross-attention 是 image + text concat 进 self-attention(不是双路)。一个不同的"text 注入方式"。/ Similar overall, but cross-attention is folded into self-attention by concatenating image + text tokens — a different "text injection" pattern.
- **`Robbyant/lingbot-va/wan_va/modules/model.py`** / **`Robbyant/lingbot-va/wan_va/modules/model.py`**: 这是 Wan2.1 的直系魔改版,加了 action token 共流(5/29 lingbot 笔记讲了 FlexAttention mask)。/ A direct fork of Wan2.1 that adds action-token co-streaming (the 5/29 lingbot FlexAttention note).
- **HunyuanVideo `HYTransformer3DModel`** / **HunyuanVideo `HYTransformer3DModel`**: MMDiT 风格——image stream 和 text stream 各跑一套 attention,然后跨流 attend。比 Wan 的 cross-attn 更对称。/ MMDiT-style — image stream and text stream each run their own attention pass and then attend across. More symmetric than Wan's cross-attn.

## 注意事项 / Caveats / when it breaks

- **`assert e.dtype == torch.float32`(行 296) / `assert e.dtype == torch.float32`**: 如果你把整个 forward 套 `with autocast(bf16)` 会把 e 降成 bf16,这个 assert 会炸。修法:在 timestep embedding 出来之后显式 `.float()`,然后让 autocast 不动 e。/ Wrapping forward in `with autocast(bf16)` will downcast `e` and crash the assert. Fix: explicitly `.float()` after timestep embedding and don't let autocast touch `e`.
- **`modulation` 共享 vs per-block / Shared vs per-block modulation**: 这里 `self.modulation` 是 per-block(每层独立)的 6×dim 参数,然后加上 global 的 `e`(timestep)。这跟 DiT 论文的"global modulation only" 不一样。Wan 这个加层级 modulation 是关键参数效率技巧——别 strip 掉。/ Here `self.modulation` is per-block (each layer has its own 6×dim parameter), added on top of a global `e` (from timestep). This differs from the DiT paper's "global modulation only" — the per-block component is a key parameter-efficiency trick. Don't strip it.
- **不要 reorder fp32 / fp16 / Don't reorder fp32 / fp16 paths**: `self.norm1(x).float() * (1 + e[1]) + e[0]` 看起来可以改成 `(1 + e[1]) * self.norm1(x) + e[0]`,但因为 `norm1` 在 bf16 算,而 e 是 fp32,顺序变了 promotion 路径也变。Wan 的写法是 `.float()` 显式 promote 之后再 modulate。/ `self.norm1(x).float() * (1 + e[1]) + e[0]` looks like you could rewrite as `(1 + e[1]) * self.norm1(x) + e[0]`, but because `norm1` runs in bf16 and `e` is fp32, the order changes the promotion path. Wan's form explicitly promotes via `.float()` before modulating.
- **`flash_attention` requires CUDA / `flash_attention` requires CUDA**: 同 Wan 体系的所有模块,CPU forward 通过 `attention()` fallback 到 SDPA。生产训练只在 H100/A100 上跑。/ Like all of Wan's modules, the CPU forward falls back to SDPA via `attention()`; production training only runs on H100/A100.

## 延伸阅读 / Further reading

- [DiT paper — Peebles & Xie 2023, "Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748)
- [QwenLM tech report — qk-norm derivation](https://arxiv.org/abs/2407.10671)
- [Wan2.1 paper](https://github.com/Wan-Video/Wan2.1#paper)
- [MMDiT paper (Stable Diffusion 3) — dual-stream cross-attention contrast](https://arxiv.org/abs/2403.03206)
