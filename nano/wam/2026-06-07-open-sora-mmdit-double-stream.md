---
date: 2026-06-07
topic: wam
source: wam
repo: hpcaitech/Open-Sora
file: opensora/models/mmdit/layers.py
permalink: https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/mmdit/layers.py#L195-L253
difficulty: advanced
read_time: ~11 min
tags: [code-of-the-day, wam, dit-block, mm-dit, flux, sd3, dual-stream]
build_role: dit-block (cross-repo variant — MM-DiT dual-stream, building on the 2026-05-25 DiT block)
---

# Flux / SD3 的双流 DiT 块:图像和文本各自一套 QKV,只在 attention 那一步合体 / Flux / SD3's dual-stream DiT block: image and text get their own QKV, and they only meet at attention

> **一句话 / In one line**: 普通 DiT 把所有 token 当一类处理;Open-Sora 的 `DoubleStreamBlockProcessor` 给图像和文本各自一份 AdaLN modulation、QKV projection、MLP——只在 **一次 joint attention** 时把两路 token cat 起来算,算完再按边界切回去. / Plain DiT treats every token uniformly; Open-Sora's `DoubleStreamBlockProcessor` gives image and text streams their own AdaLN modulation, QKV projection, and MLP, and merges them only inside a **single joint attention** call — slicing the result back at the boundary afterwards.

## 为什么重要 / Why this matters

5 月 25 日学的 DiT 是 image-only:整张图过 LayerNorm、过一次 attention、过 MLP. 但 Flux、SD3、Open-Sora 这一代要做 text-conditioned video,文本和图像是**两种不同分布**——文本 token 短而稀疏,图像 latent 长而稠密. 如果硬塞一根流里,attention 主要被同分布的图像 token 吃掉,文本信号传不到. 经典解法是用 cross-attention,但成本是每块多一次 attention. MM-DiT 想出来个更优雅的办法:两路各算各的 norm/QKV/MLP,但把 q, k, v cat 到一起一次性算 attention——既保留每路的"个性",又让两路在 attention 里互相看见对方. 这就是 Flux、SD3、Open-Sora 共用的 backbone. 这 59 行就是它的全部.

The DiT block from 2026-05-25 is image-only — LayerNorm, attention, MLP, all over the same tokens. But Flux, SD3, and Open-Sora target text-conditioned video, and text vs. image latents are **two very different distributions**: text tokens are short and sparse, image latents are long and dense. Force them through a single stream and attention is dominated by the image-token mass — the text signal never gets through. The classic fix is cross-attention, but that costs an extra attention per block. MM-DiT pulls a more elegant trick: each modality gets its own LayerNorm / QKV / MLP, **but Q/K/V are concatenated for a single joint attention call** — preserving each stream's distributional character while letting them attend to each other in the shared attention. This is the backbone shared by Flux, SD3, and Open-Sora. These 59 lines are the whole thing.

## 代码 / The code

`hpcaitech/Open-Sora` — [`opensora/models/mmdit/layers.py#L195-L253`](https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/mmdit/layers.py#L195-L253)

```python
class DoubleStreamBlockProcessor:
    def __call__(self, attn: nn.Module, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor) -> tuple[Tensor, Tensor]:
        # attn is the DoubleStreamBlock;
        # process img and txt separately while both is influenced by text vec

        # vec will interact with image latent and text context
        img_mod1, img_mod2 = attn.img_mod(vec)  # get shift, scale, gate for each mod
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift

        if attn.img_attn.fused_qkv:
            img_qkv = attn.img_attn.qkv(img_modulated)
            img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        else:
            img_q = rearrange(attn.img_attn.q_proj(img_modulated), "B L (H D) -> B L H D", H=attn.num_heads)
            img_k = rearrange(attn.img_attn.k_proj(img_modulated), "B L (H D) -> B L H D", H=attn.num_heads)
            img_v = rearrange(attn.img_attn.v_proj(img_modulated), "B L (H D) -> B L H D", H=attn.num_heads)

        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)  # RMSNorm for QK Norm as in SD3 paper
        if not attn.img_attn.fused_qkv:
            img_q = rearrange(img_q, "B L H D -> B H L D")
            img_k = rearrange(img_k, "B L H D -> B H L D")
            img_v = rearrange(img_v, "B L H D -> B H L D")

        # prepare txt for attention
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        if attn.txt_attn.fused_qkv:
            txt_qkv = attn.txt_attn.qkv(txt_modulated)
            txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        else:
            txt_q = rearrange(attn.txt_attn.q_proj(txt_modulated), "B L (H D) -> B L H D", H=attn.num_heads)
            txt_k = rearrange(attn.txt_attn.k_proj(txt_modulated), "B L (H D) -> B L H D", H=attn.num_heads)
            txt_v = rearrange(attn.txt_attn.v_proj(txt_modulated), "B L (H D) -> B L H D", H=attn.num_heads)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)
        if not attn.txt_attn.fused_qkv:
            txt_q = rearrange(txt_q, "B L H D -> B H L D")
            txt_k = rearrange(txt_k, "B L H D -> B H L D")
            txt_v = rearrange(txt_v, "B L H D -> B H L D")

        # run actual attention, image and text attention are calculated together by concat different attn heads
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn1[:, : txt_q.shape[2]], attn1[:, txt_q.shape[2] :]

        # calculate the img bloks
        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        return img, txt
```

## 逐行讲解 / What's happening

1. **`img_mod1, img_mod2 = attn.img_mod(vec)` 和 `txt_mod1, txt_mod2 = attn.txt_mod(vec)` / Two `Modulation` heads on the shared `vec`**:
   - 中文: `vec` 是把 timestep embedding 和 pooled text embedding 加一起的全局条件向量(`y + t`). 同一个 `vec` 喂给两个不同的 `Modulation` 头(`img_mod` / `txt_mod`),各自产出 6 个调制量 `(shift, scale, gate) × 2`. 两路 stream 共享条件信号,但调制的方式独立——这正是"个性化"的来源.
   - English: `vec` is the global conditioning vector (`pooled_text + timestep`), shared across modalities. The same `vec` is fed to two independent `Modulation` heads, producing `(shift, scale, gate) × 2` for each stream. Same condition signal, independent modulation — that's where the "personality" of each stream comes from.

2. **`(1 + img_mod1.scale) * img_modulated + img_mod1.shift` 是 AdaLN-Zero / `(1 + img_mod1.scale) * img_modulated + img_mod1.shift` is AdaLN-Zero**:
   - 中文: `img_norm1` 是 `LayerNorm(elementwise_affine=False)`——不带 γ、β. 紧接着用条件量做 affine. `+1` 让训练初期 scale ≈ 1(`scale` 头初始化为 0). 5 月 25 日 DiT 教过同样的式子,这里只是每路一份.
   - English: `img_norm1` is `LayerNorm(elementwise_affine=False)` (no affine), then the conditioning vector supplies the shift/scale. The `+1` keeps the scale near 1 at init (the `scale` head starts at 0). Same equation as the DiT block from 2026-05-25, just instantiated per stream.

3. **`fused_qkv` 分支 / The `fused_qkv` branch**:
   - 中文: 训练阶段通常 `fused_qkv=True`——一根 `Linear(dim, 3*dim)` 一步算出 Q/K/V,效率最高. 推理(尤其需要量化时)可能要分成三根 `Linear`. 两条分支都做同样的 `rearrange`,最后把 head 维提到第 2 维,符合 `flash_attn` 的 `(B, H, L, D)` 约定.
   - English: training usually has `fused_qkv=True` — one `Linear(dim, 3*dim)` produces Q/K/V together, the fastest option. Inference (especially quantized) may need three separate `Linear`s. Both branches `rearrange` to the `(B, H, L, D)` layout `flash_attn` expects.

4. **`attn.img_attn.norm(img_q, img_k, img_v)` 是 QK-Norm / `attn.img_attn.norm(...)` is QK-Norm**:
   - 中文: 这是 SD3 论文里加的稳定 trick——对 Q 和 K 做 RMSNorm,V 不动. 解决高维 attention 训练不稳定的问题,被 Flux/SD3/Open-Sora 全套继承. `QKNorm` 只接收 `v` 是因为它要 `q.to(v)` / `k.to(v)` 把 dtype 对齐.
   - English: a stability trick from the SD3 paper — RMSNorm on Q and K, V untouched. Solves the high-dim attention instability issue. Inherited by Flux/SD3/Open-Sora. `QKNorm` takes `v` only to call `q.to(v)` / `k.to(v)` and align dtypes.

5. **`q = torch.cat((txt_q, img_q), dim=2)` 是 MM-DiT 的灵魂 / `q = torch.cat((txt_q, img_q), dim=2)` is the heart of MM-DiT**:
   - 中文: 沿 token 维(`dim=2`)把 txt 和 img 的 Q 拼起来,K、V 同操作. 这 *一根* attention call 里,每个 token 都能 attend 到所有 txt + 所有 img token——本质上是 joint self-attention. 注意 *txt 放在前面*——后面切回去要用 `txt_q.shape[2]` 当边界.
   - English: concat Q (and K, V) along the token dim. Inside this *single* attention call, every token sees every other — full joint self-attention. Critically, *txt comes first* — the boundary `txt_q.shape[2]` is the slice index used to split the result back.

6. **`attn1 = attention(q, k, v, pe=pe)` 的 `pe` 是位置编码 / `attn1 = attention(q, k, v, pe=pe)` and the `pe` argument**:
   - 中文: `pe` 是 3D RoPE(`EmbedND` 算出来的,axes_dim 比如 `[16, 56, 56]` 三个轴 cat). 关键是:txt token 在 RoPE 计算时被分配到一个"特殊"的轴位置(通常是 `(0, 0, 0)`),不参与 spatial 位置——避免它干扰 image latent 的几何先验. Open-Sora 的 RoPE 代码在 5 月 29 日 wan21-3d-rope 已经讲过.
   - English: `pe` is 3D RoPE (axes_dim e.g. `[16, 56, 56]` concatenated; see the 2026-05-29 wan21-3d-rope note). Crucially, txt tokens get assigned a "neutral" axis position (often `(0, 0, 0)`) so they don't bias the spatial geometry of image latents — the RoPE is applied only meaningfully to image positions.

7. **`txt_attn, img_attn = attn1[:, : txt_q.shape[2]], attn1[:, txt_q.shape[2] :]` 切回两路 / `txt_attn, img_attn = attn1[:, : txt_q.shape[2]], attn1[:, txt_q.shape[2] :]` to split back**:
   - 中文: attention 结果用 *txt 的序列长度* 切两段,前一段给 txt、后一段给 img. 因为前面 cat 时 txt 放在前面,所以边界就是 `txt_q.shape[2]`. 注意这里是 `:` 切 token 维,不是 head 维.
   - English: the attention output is sliced by `txt_q.shape[2]` since txt was concatenated first. Slicing on the *token* axis, not the head axis. After this, each stream is back in its own subspace.

8. **`img = img + img_mod1.gate * proj(img_attn)` 双残差 / Double residual: `img = img + img_mod1.gate * proj(img_attn)` then again for MLP**:
   - 中文: gate 是 AdaLN-Zero 论文里那个 `α` 参数——初始化为 0,所以训练起步时这一块**完全是恒等映射**,梯度只从 residual 走. 这是 DiT 系列训练稳定的关键. txt 流做同样的事,但用自己那一套 `txt_mod` 和 `txt_norm2 / txt_mlp`.
   - English: the `gate` is the `α` parameter from the AdaLN-Zero paper — initialized to 0, so at training start the whole block is essentially an identity and gradients flow through the residual only. This is *the* training-stability trick of the DiT family. The txt stream does the same thing with its own `txt_mod` / `txt_norm2` / `txt_mlp`.

## 类比 / The analogy

把 image latent 和 text token 想成两个**翻译公司**:一家翻译图像(image stream),一家翻译文字(text stream). 它们各自有自己的字典(QKV 投影)、自己的笔记本(MLP)、自己的工作风格(AdaLN 调制). 平时分开办公,但每天有一次**联合会议**——所有员工(text + image tokens)坐到一个大会议室里(joint attention),互相听对方的发言. 会议结束,各自带着新听到的信息回自己公司继续干活(残差 + MLP). 5 月 25 日学的 DiT 块只有一家公司——所有员工都在一家,一起上一起下;Flux/SD3/Open-Sora 的 MM-DiT 拆成两家,会议时合班,日常分开. 这样文字员工就不会被图像员工的"嗓门大"压制,但需要时还是能开会沟通.

Imagine image latents and text tokens as two **translation companies**. One handles images (image stream), the other handles text. Each has its own dictionaries (QKV projections), its own notebooks (MLPs), its own working style (AdaLN modulation). They sit in separate offices most of the time, but once a day there's a **joint meeting** — all staff (text + image tokens) gather in one big conference room (joint attention) and hear each other out. After the meeting, each goes back to their own office with the new info (residual + MLP). The DiT block from 2026-05-25 is one company — everyone in the same room all the time. MM-DiT runs two companies — they meet briefly, but daily work stays separate. The text staff isn't drowned out by the image majority, yet they still get full context when it matters.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

**Curriculum item**: `dit-block` (cross-repo variant of the item covered 2026-05-25 — 依赖项 / depends on: `patchify-positional` covered 2026-05-29)

中文: 在你的 nanoWAM 路线图里,5 月 25 日学的标准 DiT 块是 **入门级**:做无条件或单图条件的 latent diffusion 已经够用. 一旦要做 text-conditioned video(典型 WAM 任务:"生成一个机器人抓起红色方块"),text token 进来后就必须有个机制把语义信号注入到 image latent 序列里. 你有两条路:(a) cross-attention 块——image self-attn + image-to-text cross-attn,每块两次 attention. (b) MM-DiT 的 dual-stream——每块只一次 attention,但每路都要复制一份 norm/QKV/MLP. (b) 计算量低、内存高;(a) 反过来. 现代视频扩散基本统一到 (b),因为 attention 是开销大头,而 norm/MLP 参数翻倍可以靠 SP/FSDP 拆掉. **从 nanoWAM 扩到 production WAM 的关键扩展**是加 *第三条流——action stream*,做 image / text / action 三流共用一次 attention. 5 月 29 日 lingbot 的 FlexAttention 已经在做这件事,只不过它把 image 和 action token cat 进同一根流——你也可以照 Flux 这套思路,给 action 流一套独立的 Modulation + QKV + MLP,只在 joint attention 时合体,这样三流互不干扰但能互相听见.

In your nanoWAM roadmap, the plain DiT block from 2026-05-25 is the **entry level** — fine for unconditional or single-image-conditioned latent diffusion. The moment you want text-conditioned video (the classic WAM task: "generate a robot picking up the red block"), you need a mechanism to inject the text semantics into the image-latent sequence. Two options: (a) cross-attention blocks — image self-attn plus image-to-text cross-attn, two attentions per block; (b) MM-DiT dual-stream — one attention per block, but every stream duplicates its norm/QKV/MLP. Option (b) is compute-cheaper and memory-heavier; (a) the reverse. Modern video diffusion is converging on (b) because attention dominates the runtime budget, while the duplicated norm/MLP params can be sharded with SP/FSDP. **The key extension to production WAM** is adding a *third stream* — an action stream — so image / text / action all share a single joint attention. The FlexAttention block from 2026-05-29 (lingbot) already does this, but it concatenates image and action tokens into one stream. You could also follow the Flux template more strictly: give the action stream its own Modulation + QKV + MLP and only merge at joint attention. Three streams that ignore each other day-to-day but always meet briefly.

## 自己跑一遍 / Try it yourself

```python
# pip install torch
import torch, torch.nn as nn

class DualStreamBlock(nn.Module):
    def __init__(self, dim=64, heads=4):
        super().__init__()
        self.h, self.d = heads, dim // heads
        # one set of weights per stream
        for s in ("img", "txt"):
            setattr(self, f"{s}_ln", nn.LayerNorm(dim, elementwise_affine=False))
            setattr(self, f"{s}_qkv", nn.Linear(dim, 3 * dim, bias=False))
            setattr(self, f"{s}_proj", nn.Linear(dim, dim, bias=False))
            setattr(self, f"{s}_mlp", nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, dim)))

    def split_qkv(self, x):
        B, L, _ = x.shape
        q, k, v = x.reshape(B, L, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        return q, k, v

    def forward(self, img, txt):
        iq, ik, iv = self.split_qkv(self.img_qkv(self.img_ln(img)))
        tq, tk, tv = self.split_qkv(self.txt_qkv(self.txt_ln(txt)))
        q = torch.cat([tq, iq], dim=2); k = torch.cat([tk, ik], dim=2); v = torch.cat([tv, iv], dim=2)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)  # (B, H, T+I, D)
        out = out.transpose(1, 2).reshape(out.shape[0], -1, self.h * self.d)
        txt_out, img_out = out[:, :txt.shape[1]], out[:, txt.shape[1]:]
        img = img + self.img_proj(img_out); img = img + self.img_mlp(img)
        txt = txt + self.txt_proj(txt_out); txt = txt + self.txt_mlp(txt)
        return img, txt

B, dim = 2, 64
img = torch.randn(B, 100, dim)  # 100 image-latent tokens
txt = torch.randn(B, 16, dim)   # 16 text tokens
img_out, txt_out = DualStreamBlock(dim)(img, txt)
print("img:", img.shape, "→", img_out.shape, "  txt:", txt.shape, "→", txt_out.shape)
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
img: torch.Size([2, 100, 64]) → torch.Size([2, 100, 64])   txt: torch.Size([2, 16, 64]) → torch.Size([2, 16, 64])
```

中文: 两路 token 各自的序列长度 (100 vs 16) 进出不变,但 attention 是在 116 个 token 上一起算的——`txt_out` 已经能"看到"图像 latent,`img_out` 也能"看到"文字. 这就是 MM-DiT 的本质.

The two streams keep their sequence lengths (100 vs 16) end-to-end, but attention runs on all 116 tokens at once — so `txt_out` has already attended to image latents and vice versa. That's the entirety of MM-DiT.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Flux original code (Black Forest Labs)**: 中文: Open-Sora 这份代码注释里就写 "Modified from Flux"——Flux 是这个设计的源头. SD3 几乎同款,只在 RoPE 实现细节上有差. / English: the Open-Sora file header explicitly says "Modified from Flux" — Flux is the original. SD3 is nearly identical apart from RoPE details.
- **HiDream-I1 (Wan / Hunyuan tech reports)**: 中文: 大型视频扩散基本都抄这个 dual-stream,把第二路从 text 扩到 "text + image-condition" 双任务. / English: most large video-diffusion stacks copy the dual-stream and extend the second stream to "text + image-condition" multi-tasks.
- **lerobot SmolVLA 的 VLM + action expert**: 中文: 同样的思想——VLM 跑 image+text,action expert 跑 action,两路都到 LM 的 attention 里 cat 一起算. 2026-05-29 教过. 一个搬到 LM,一个留在 diffusion,但底层是同一种"两路分行、joint attention 合班"的设计.
- **lerobot SmolVLA's VLM + action expert**: English: same idea on the LM side — VLM runs image+text, action expert runs actions, both feed into the LM's attention concatenated. Covered 2026-05-29. Same "two pipelines, one joint attention" pattern, on a different substrate.
- **Stable Diffusion 3.5 Large**: 中文: 用 MM-DiT 但堆了 38 层 dual-stream + 6 层 single-stream(后面 token 已经融合了,就不用 dual 了). 是 Flux 之外的另一个公开 baseline.
- **Stable Diffusion 3.5 Large**: English: uses MM-DiT with 38 dual-stream layers + 6 single-stream layers (by then the tokens are sufficiently fused). Public baseline alongside Flux.

## 注意事项 / Caveats / when it breaks

- **txt 必须放在 cat 前面 (或者你的边界 slice 必须配套)**: 中文: 代码里硬编码 `txt_q.shape[2]` 当 split 边界. 如果哪天有人把顺序反过来(`cat((img, txt))`)忘改 slice,attention 输出会被错误地切两段,模型 silent 跑错——但 loss 看起来还能降.
- **txt must come first in the cat (or your slice index must match)**: English: the slice boundary `txt_q.shape[2]` hard-codes the ordering. Swapping to `cat((img, txt))` without updating the slice produces a silently broken model — loss still decreases, but the streams are swapped.
- **每路 Modulation 参数翻一倍**: 中文: `img_mod` 和 `txt_mod` 各占一个 `Linear(dim, 6*dim)`. 38 层 dual-stream 的总参数能比 image-only DiT 多 20-30%. 对 nanoWAM 来说要么用 (1) `dual_stream_layers=4`,要么直接共享 `Modulation` 头.
- **Modulation params double per stream**: English: `img_mod` and `txt_mod` are each `Linear(dim, 6*dim)`. Across 38 dual-stream layers this can add 20-30% to the total parameter count vs. image-only DiT. For nanoWAM, either keep `dual_stream_layers=4` or share the Modulation head between streams.
- **QK-Norm 不能省**: 中文: 高维 cat 后做 attention,如果不归一化 Q/K,training loss 会出现 NaN/Inf. SD3 论文里专门讨论过这个. 去掉 `QKNorm` 你大概率训不动.
- **QK-Norm is not optional**: English: high-dim Q/K after concat without RMSNorm produces NaN/Inf during training — the SD3 paper specifically discusses this. Drop `QKNorm` and your run will diverge.
- **RoPE 对 txt 的处理是关键**: 中文: 如果给 txt token 也算 spatial RoPE,会把 txt 投影到一个不该有的几何空间——破坏 cross-modal alignment. 必须给 txt 一个 "neutral" RoPE 坐标(通常 (0,0,0)).
- **RoPE handling for txt is critical**: English: applying spatial RoPE to txt tokens projects them onto a geometry they shouldn't have, breaking cross-modal alignment. Assign txt a "neutral" RoPE coordinate (typically `(0, 0, 0)`).

## 延伸阅读 / Further reading

- [SD3 paper (Stability AI, 2024)](https://arxiv.org/abs/2403.03206) — section 3 introduces MM-DiT and the QK-Norm trick
- [Flux source (BFL)](https://github.com/black-forest-labs/flux) — the upstream this Open-Sora code derives from
- [DiT note from 2026-05-25](../../2026/05/2026-05-25-dit-adaln-zero-block.md) — the single-stream ancestor and the `+1` AdaLN-Zero trick
- [3D RoPE note from 2026-05-29](2026-05-29-wan21-3d-rope.md) — how the `pe` argument is built
- [SmolVLA dual-expert note (2026-05-29)](../vla/2026-05-29-smolvla-vlm-with-expert.md) — the same dual-stream idea applied on the LM side
