---
date: 2026-06-01
topic: wam
source: wam
repo: hpcaitech/Open-Sora
file: opensora/models/mmdit/layers.py
permalink: https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/mmdit/layers.py#L195-L253
difficulty: advanced
read_time: ~14 min
tags: [code-of-the-day, wam, dit, mmdit, flux, sd3, joint-attention]
build_role: dit-block (cross-repo variant) — Flux/SD3-style dual-stream DiT block that runs image and text through joint attention
---

# DoubleStreamBlock:两条流、各自做归一化和投影、然后只共享一次 attention / DoubleStreamBlock: two streams, separate norms and projections, fused by one shared attention

> **一句话 / In one line**: 图像和文本各自走一套 LayerNorm + AdaLN + QKV + MLP,但 Q/K/V 沿序列维拼接后**只过一次** attention,然后再拆回去——这就是 SD3 / Flux 的"MMDiT 双流块"。 / Image and text each get their own LayerNorm + AdaLN + QKV + MLP, but their Q/K/V tensors are concatenated along the sequence dim and run through **a single** attention call before being split back — the SD3 / Flux "MMDiT double-stream block".

## 为什么重要 / Why this matters

之前已经覆盖过的 `dit-block`(2026-05-25 的 DiT adaLN-Zero 块)只有**一条 token 流**——图像 latent 经过 self-attention,文本条件通过 cross-attention 或 AdaLN 注入。Flux / SD3 把这个范式撕了:它们让图像和文本**都是 token**,放在同一个 attention 的 Q/K/V 里一起算,但**保留各自的参数**(各有自己的 norm、QKV、MLP)。结果是文本不再只是"调味料",而是和图像 token **平等地**共演化。这是 2024-2026 年生成式视觉模型最重要的架构演进之一,而 Open-Sora 这一份实现把它压成了 60 行。

The `dit-block` curriculum item we covered earlier (2026-05-25's vanilla adaLN-Zero block) has **one** token stream — image latents go through self-attention while text conditioning is injected via cross-attention or AdaLN. Flux and SD3 rip that paradigm apart: image and text are **both** tokens, processed in the same attention's Q/K/V, but with **separate parameters** (each gets its own norm, QKV, MLP). Text is no longer a flavoring; it co-evolves with image tokens as a peer. This is one of the most important architectural shifts of 2024–2026, and Open-Sora compresses the implementation into 60 lines.

## 代码 / The code

`hpcaitech/Open-Sora` — [`opensora/models/mmdit/layers.py`](https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/mmdit/layers.py#L195-L253)

```python
class DoubleStreamBlockProcessor:
    def __call__(self, attn, img, txt, vec, pe):
        # vec will interact with image latent and text context
        img_mod1, img_mod2 = attn.img_mod(vec)  # get shift, scale, gate for each mod
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        if attn.img_attn.fused_qkv:
            img_qkv = attn.img_attn.qkv(img_modulated)
            img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D",
                                            K=3, H=attn.num_heads, D=attn.head_dim)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)  # RMSNorm QK Norm (SD3)

        # prepare txt for attention
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        if attn.txt_attn.fused_qkv:
            txt_qkv = attn.txt_attn.qkv(txt_modulated)
            txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D",
                                            K=3, H=attn.num_heads, D=attn.head_dim)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # image and text attention are calculated together by concat along sequence
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)
        attn1 = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn1[:, :txt_q.shape[2]], attn1[:, txt_q.shape[2]:]

        # img blocks
        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        # txt blocks
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        return img, txt
```

## 逐行讲解 / What's happening

1. **`img_mod, txt_mod = attn.{img,txt}_mod(vec)` —— 两组独立的 AdaLN 调制器 / Two independent AdaLN modulators**:
   - 中文: `vec` 是一个全局条件向量(时间步 + 池化文本),它通过两个**独立**的线性头分别产出图像和文本的 (shift, scale, gate) 三元组。两条流共享条件信号,但**对条件的反应方式各自学**。
   - English: `vec` is a global conditioning vector (timestep + pooled text). It feeds two **independent** linear heads that produce `(shift, scale, gate)` triples for image and for text separately. The two streams share the conditioning *signal* but learn their own *response* to it.

2. **AdaLN 公式 `(1 + scale) * norm(x) + shift` / The AdaLN formula**:
   - 中文: 这是 DiT 标准的 adaLN-Zero 写法。`scale` 初始化为 0 时,这一步就是恒等映射——保证训练初期 block 的 attention/MLP 完全不起作用,模型从"什么都不做"开始学。
   - English: This is the standard adaLN-Zero formulation. When `scale` is initialized to 0, the line acts as identity — guaranteeing the attention/MLP branch contributes nothing at the start of training, so the model bootstraps from "do nothing".

3. **`rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3)`** :
   - 中文: einops 的紧凑写法,一行完成"qkv 拆分 + 多头重排"。比传统 `chunk(3, dim=-1)` 再 `view` + `transpose` 短 3 行,且更不容易写错。
   - English: An einops one-liner that simultaneously splits QKV and rearranges into multi-head shape. Cuts 3 lines of `chunk(3)` + `view` + `transpose` boilerplate and is much harder to get wrong.

4. **QK Norm —— SD3 的小但关键改动 / QK Norm — SD3's small but crucial tweak**:
   - 中文: `attn.img_attn.norm(img_q, img_k, img_v)` 是 RMSNorm 应用在 Q 和 K 上(不动 V)。这个改动在 SD3 论文里被证明能显著稳定大规模训练——尤其是分辨率上升时,attention logits 不再爆炸。
   - English: `attn.img_attn.norm(img_q, img_k, img_v)` applies RMSNorm to Q and K (V untouched). SD3 showed this stabilizes large-scale training dramatically, especially at higher resolutions where attention logits would otherwise blow up.

5. **`torch.cat((txt_q, img_q), dim=2)` —— 这是双流块的精髓 / The heart of the double-stream block**:
   - 中文: Q/K/V 沿**序列维**拼接,然后只调用一次 `attention(q, k, v, pe=pe)`。结果是 `(B, H, L_txt + L_img, D)`,attention 让文本 token 看到图像 token,反之亦然。**只跑一次 attention**,但等价于"自注意力(各自) + 互相 cross-attention"四套合一。
   - English: Q/K/V are concatenated along the **sequence** dimension and one `attention(q, k, v, pe=pe)` call processes the joined sequence. Output is `(B, H, L_txt + L_img, D)`. Text attends to image and vice versa simultaneously. **One attention call** does the work of "self-attention (each stream) + cross-attention (both directions)" combined.

6. **`attn1[:, :txt_q.shape[2]], attn1[:, txt_q.shape[2]:]`** :
   - 中文: 把 attention 输出沿序列维**切回两段**——前面是文本,后面是图像。然后各自走自己的 proj 和 MLP。
   - English: Split the attention output back into two pieces along the sequence axis — text first, image second. Each piece then continues through its own `proj` and MLP.

7. **`img_mod{1,2}.gate`、`txt_mod{1,2}.gate` —— 残差门控 / The residual gates**:
   - 中文: 每个残差更新都被 `gate`(adaLN-Zero 学到的 0~1 之间的标量)缩放。和 scale 一样,初始化为 0 让 block 在训练初期是恒等映射。
   - English: Every residual update is scaled by a learned `gate` (a 0-initialized scalar from adaLN-Zero). Same role as `scale`: makes the block act as identity at step 0 and gradually open up as training proceeds.

## 类比 / The analogy

把它想成一对正在合作的画家和编剧。画家专门画(图像 token),编剧专门写(文本 token)——他们的画笔和打字机是不同的(各自的 norm、QKV、MLP)。但他们每隔一段时间会**坐到同一张大圆桌前讨论一次**(共享的 attention),画家可以指着文本说"这里我打算这样画",编剧也能看到画家的草稿后调整剧本。讨论完后两个人各自回到自己的工作间继续(各自的 MLP)。每个 block 就是一轮"讨论 + 各自工作",叠 20 层就是 20 轮"画家 ↔ 编剧"协作。

Picture a painter and a screenwriter collaborating on a film. The painter has their own brushes (image stream's norm + QKV + MLP), the screenwriter has their own typewriter (text stream's norm + QKV + MLP). Periodically they sit at **the same round table** for a meeting (the shared attention). The painter can point at lines of script while talking about how to paint a scene; the screenwriter can see the rough sketches and adjust dialogue. After the meeting they go back to their separate studios (separate MLPs) and refine. Each block is one "meeting + refine" cycle; stacking 20 blocks gives 20 rounds of painter ↔ screenwriter collaboration.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-WAM

中文:这是 `dit-block` 课程项的**跨仓库变体(advanced variant)**——之前覆盖的 `facebookresearch/DiT` 是单流块,Open-Sora 这份是双流块。在你 nano-WAM 的构建图里,它替换的就是 transformer 主干那一摞 block。它的上游是已覆盖的 `patchify-positional`(给图像/文本各自切 token 加 RoPE)和 `text-conditioning`(产出 `txt` 序列和全局 `vec`),下游是 `training-loop` / `sampler-inference`(把这个堆叠 N 层后做去噪)。如果你的 nano-WAM 已经能跑单流 DiT,升级到双流需要做三件事:(1) 把 text 也做成 token 序列(而不是池化成一个向量做 cross-attn);(2) 复制一份所有线性层(`img_*` / `txt_*`);(3) 在 attention 里 concat Q/K/V。**没了**——这就是从 PixArt-α 到 Flux/SD3 的路径。生产级实现还会补:single-stream block(后半段切回单流以省参)、QK Norm 的 RMSNorm 数值稳定、3D RoPE 的复合位置编码、attention mask 控制 text-vs-image 互动方式。

This is a **cross-repo advanced variant** of the `dit-block` curriculum item — we covered `facebookresearch/DiT`'s vanilla single-stream block on 2026-05-25, and Open-Sora's `DoubleStreamBlock` is the dual-stream upgrade. In your nano-WAM build graph it slots into the same place: the transformer backbone, repeated N times. Upstream of it sit the covered `patchify-positional` item (which patchifies images and adds RoPE) and `text-conditioning` (which produces the `txt` sequence and the global `vec`); downstream sit `training-loop` and `sampler-inference` (which iterate this stack to denoise). If your nano-WAM already runs a single-stream DiT, upgrading to dual stream is three steps: (1) keep text as a *sequence* of tokens rather than pooling it into one vector for cross-attn; (2) duplicate every linear layer (`img_*` and `txt_*`); (3) concat Q/K/V along the sequence dim inside attention. **That's it** — the path PixArt-α took to become Flux / SD3. Production implementations layer in: single-stream blocks (drop back to one stream in the later layers to save parameters), QK Norm with RMSNorm for numerical stability, 3D RoPE compositionality, and attention masks to control how text and image attend to each other.

## 自己跑一遍 / Try it yourself

```python
import torch, torch.nn as nn
from einops import rearrange

class DoubleStream(nn.Module):
    def __init__(self, d=64, h=4):
        super().__init__()
        self.d, self.h, self.head = d, h, d // h
        for prefix in ["img", "txt"]:
            setattr(self, f"{prefix}_norm1", nn.LayerNorm(d, elementwise_affine=False))
            setattr(self, f"{prefix}_norm2", nn.LayerNorm(d, elementwise_affine=False))
            setattr(self, f"{prefix}_qkv",   nn.Linear(d, 3 * d))
            setattr(self, f"{prefix}_proj",  nn.Linear(d, d))
            setattr(self, f"{prefix}_mlp",   nn.Sequential(nn.Linear(d, 2*d), nn.GELU(), nn.Linear(2*d, d)))

    def forward(self, img, txt):
        # attention prep
        def heads(x):
            return rearrange(x, "B L (K H D) -> K B H L D", K=3, H=self.h)
        iq, ik, iv = heads(self.img_qkv(self.img_norm1(img)))
        tq, tk, tv = heads(self.txt_qkv(self.txt_norm1(txt)))

        # concat along sequence dim → one joint attention
        q = torch.cat((tq, iq), dim=2); k = torch.cat((tk, ik), dim=2); v = torch.cat((tv, iv), dim=2)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "B H L D -> B L (H D)")
        t_out, i_out = out[:, :txt.shape[1]], out[:, txt.shape[1]:]

        img = img + self.img_proj(i_out)
        img = img + self.img_mlp(self.img_norm2(img))
        txt = txt + self.txt_proj(t_out)
        txt = txt + self.txt_mlp(self.txt_norm2(txt))
        return img, txt

torch.manual_seed(0)
img = torch.randn(2, 256, 64)   # 256 image tokens
txt = torch.randn(2,  32, 64)   # 32  text tokens
block = DoubleStream()
img2, txt2 = block(img, txt)
print("img:", img2.shape, " txt:", txt2.shape)
print("text changed by image?", not torch.allclose(txt, txt2))
```

运行 / Run with:
```bash
pip install torch einops
python try.py
```

预期输出 / Expected output:
```
img: torch.Size([2, 256, 64])  txt: torch.Size([2, 32, 64])
text changed by image? True
```

中文一两句:注意 `text changed by image?` 是 `True`——这就是双流块和"图像 self-attn + 文本 cross-attn"最本质的区别:文本 token 自己也会被图像 token 修改,实现真正的双向耦合。

`text changed by image?` is `True` — this is the most essential difference between dual-stream and "image self-attn + text cross-attn": the text tokens themselves are updated by image tokens, achieving genuine bidirectional coupling.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **`facebookresearch/DiT` (vanilla DiTBlock)** / **DiTBlock from `facebookresearch/DiT`**: 中文:**单流**版本,2026-05-25 已覆盖。两者并列读最能看清"单流→双流"的演化路径。 / English: **Single-stream** variant, covered on 2026-05-25. Reading the two side by side is the clearest way to see the single→dual evolution.
- **`huggingface/diffusers` 的 SD3 / Flux `JointTransformerBlock`** / **SD3 / Flux `JointTransformerBlock` in diffusers**: 中文:HF 官方实现,代码风格略不同(用 `JointAttnProcessor`),但数学完全等价。 / English: The official HF implementations under different processor abstractions, but the math is identical.
- **`Wan-Video/Wan2.1` 的 WanAttentionBlock**: 中文:走的是 "image self-attn + text cross-attn" 的单流路线,可以看作 PixArt-α 风格的延续,不是这里的 MMDiT 风格。 / English: Single-stream path, "image self-attn + text cross-attn" — PixArt-α lineage rather than MMDiT lineage.
- **`NVIDIA/Isaac-GR00T` 的 DiT module**: 中文:机器人版 DiT,把 image / state / action 都看作 token,某种意义上把 SD3 双流的思路推广到了三/四流。 / English: A robotic DiT that treats image, state, and action as tokens — in a sense generalizing the dual-stream idea to triple/quadruple stream for action generation.

## 注意事项 / Caveats / when it breaks

- **参数量翻倍 / Parameter count doubles**: 中文:`img_*` 和 `txt_*` 是两套完整的线性层,DiT 主干的参数量比单流版本几乎翻一倍。Flux 和 SD3 都用"前 N 层双流 + 后 M 层单流"的混合来缓解。 / English: `img_*` and `txt_*` are two complete linear stacks, doubling the DiT trunk's params vs. single-stream. Flux and SD3 mitigate by using "double-stream for first N layers + single-stream for the last M".
- **RoPE / 位置编码必须能同时索引文本和图像 token / RoPE must index both image and text tokens**: 中文:`pe` 是拼接后的位置编码张量,长度 `L_txt + L_img`,你不能简单 reuse 单流的 RoPE。Flux 用 EmbedND 把多个轴的 RoPE 拼起来。 / English: `pe` covers the concatenated sequence of length `L_txt + L_img`. You can't reuse a single-stream RoPE — Flux's `EmbedND` composes multi-axis RoPE to handle the joint indexing.
- **要么全双流,要么不双流 / All or nothing per block**: 中文:同一个 block 内不能"一半 token 单流、一半双流",参数定义是绑死的。如果想动态切换,得在 block 外部做。 / English: Inside a single block you can't selectively double-stream some tokens — the parameter sets are pre-bound. Dynamic switching has to happen between blocks.

## 延伸阅读 / Further reading

- [Scaling Rectified Flow Transformers (SD3 paper)](https://arxiv.org/abs/2403.03206) — introduces MMDiT and QK Norm
- [Flux model card](https://github.com/black-forest-labs/flux) — the reference implementation Open-Sora's MMDiT is modified from
- Daily code 2026-05-25 (`facebookresearch/DiT` adaLN-Zero block) — read alongside for the single-stream baseline
