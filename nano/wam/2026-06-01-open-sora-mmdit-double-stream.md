---
date: 2026-06-01
topic: wam
source: wam
repo: hpcaitech/Open-Sora
file: opensora/models/mmdit/layers.py
permalink: https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/mmdit/layers.py#L173-L308
difficulty: advanced
read_time: ~13 min
tags: [code-of-the-day, wam, dit, mmdit, double-stream]
build_role: dit-block (cross-repo variant)
---

# 把 DiT 一分为二:Open-Sora 的 MMDiT DoubleStreamBlock / Splitting DiT in two: Open-Sora's MMDiT DoubleStreamBlock

> **一句话 / In one line**: 图像和文本各自走一套 norm/QKV/MLP 和 adaLN 调制,只在 attention 那一步把 q/k/v concat 在一起做 joint softmax——这就是 SD3 / FLUX / Open-Sora / Hunyuan-Video 的 backbone block / Image and text each get their own norm / QKV / MLP and adaLN modulation, but at attention time their q/k/v concat for a joint softmax — that's the backbone block of SD3 / FLUX / Open-Sora / Hunyuan-Video.

## 为什么重要 / Why this matters

你之前已经从 facebookresearch/DiT 学过单流 `DiTBlock`(adaLN-Zero modulation 把 timestep + condition 一起注入)。MMDiT 是同一个想法的"双子座"版本:**两种 modality 各持有完整的一套参数,只在 attention 中相遇**。这把"text → image 的 cross-attention"换成了"image + text 在 attention 内部共享 softmax 视野",训练更稳、条件遵循更强,是 SD3/FLUX 起的新一代默认 backbone。把单流和双流 block 对照读完,你就掌握了所有现代图像/视频 DiT 的两种主要骨架。

You already learned the single-stream `DiTBlock` (adaLN-Zero modulation injecting timestep + condition) from facebookresearch/DiT. MMDiT is the same idea's twin: **each modality holds a full set of weights and the two only meet inside attention**. This replaces "text → image cross-attention" with "image + text sharing one softmax window", which is more stable to train and follows conditions more tightly — that's why SD3 / FLUX / Open-Sora / Hunyuan-Video all default to it. Reading single-stream and double-stream blocks side-by-side gives you a complete grip on the two main backbones used in modern image / video DiTs.

## 代码 / The code

`hpcaitech/Open-Sora` — [`opensora/models/mmdit/layers.py`](https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/mmdit/layers.py#L173-L308)

```python
@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)
        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlockProcessor:
    def __call__(self, attn, img, txt, vec, pe):
        # vec will interact with image latent and text context
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B H L D",
            K=3, H=attn.num_heads, D=attn.head_dim,
        )
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)  # RMSNorm for QK Norm as in SD3 paper

        # prepare txt for attention
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B H L D",
            K=3, H=attn.num_heads, D=attn.head_dim,
        )
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention — image and text concat on the seq dim
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn1[:, : txt_q.shape[2]], attn1[:, txt_q.shape[2]:]

        # image block
        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        img = img + img_mod2.gate * attn.img_mlp(
            (1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift
        )

        # text block
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp(
            (1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift
        )
        return img, txt


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio, qkv_bias=False, fused_qkv=True):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # image stream
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, fused_qkv=fused_qkv)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        # text stream  (identical shape, separate params)
        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, fused_qkv=fused_qkv)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.processor = DoubleStreamBlockProcessor()

    def forward(self, img, txt, vec, pe):
        return self.processor(self, img, txt, vec, pe)
```

## 逐行讲解 / What's happening

1. **`ModulationOut` 和 `Modulation`**:
   - 中文: adaLN-Zero 原始版本是输出 `shift, scale, gate` 三个张量。`Modulation(double=True)` 输出 **两组**(attention 前一组,MLP 前一组),所以 `multiplier=6`。一个 `Linear` 输出 `6*D`,然后 `chunk(6)` 切六份分到两个 `ModulationOut`。`silu` 前置非线性是 FLUX/SD3 沿用 DiT 的设计。
   - English: vanilla adaLN-Zero emits `shift, scale, gate`. `Modulation(double=True)` emits **two** such triples (one before attention, one before MLP), hence `multiplier=6`. One `Linear` outputs `6*D` then `chunk(6)` splits it across two `ModulationOut` objects. The pre-`silu` is the DiT convention SD3 / FLUX kept.

2. **两路独立的 `img_mod` / `txt_mod`**:
   - 中文: 图像和文本各有**自己的** Modulation Linear——这就是 MMDiT 比单流 DiT 多一份参数的根源。同一个 `vec`(timestep + 条件 embedding)分别进入两路,产出两套 shift/scale/gate。这让模型可以学到"text 流和 image 流在同一时间步该不同地处理"。
   - English: each modality has **its own** Modulation Linear — that's the extra parameter cost of MMDiT vs. single-stream DiT. The same `vec` (timestep + condition embedding) goes into both, producing two distinct shift/scale/gates. This lets the model learn "text and image streams should be processed differently at the same timestep".

3. **`img_modulated = (1 + scale) * norm(img) + shift`**:
   - 中文: adaLN-Zero 的核心公式。`+ 1` 的作用是 zero-init 时让 modulation 等价于 "identity"——训练初期 `scale` 接近 0,这一项变 `1 * norm(img) + 0`,完全等价于普通 LayerNorm。这是 DiT 训练稳定的关键 trick。
   - English: the adaLN-Zero formula. `+ 1` makes a zero-initialized `scale` collapse to identity: `1 * norm(img) + 0` is a plain LayerNorm. That's why DiT-family training is so stable — the residual path starts as a no-op.

4. **`img_attn.qkv(img_modulated)` + `rearrange "B L (K H D) -> K B H L D"`**:
   - 中文: 这是 einops 的"fused qkv 一行拆 head"魔法——一个 `Linear(D, 3*D)` 输出 `[B, L, 3*H*D_h]`,einops 一步拆成 `[3, B, H, L, D_h]`,赋值给 q, k, v。比手动 reshape + transpose 简洁得多。
   - English: einops one-liner for "fused-qkv + split-heads" — one `Linear(D, 3*D)` outputs `[B, L, 3*H*D_h]`, then einops reshapes to `[3, B, H, L, D_h]` and tuple-unpacks into q, k, v. Much cleaner than manual reshape + transpose.

5. **`img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)` (QK-Norm)**:
   - 中文: SD3 论文引入的 trick——对 q 和 k 各做一次 RMSNorm,稳定 attention 数值。FLUX 和 Open-Sora 都继承。
   - English: a trick SD3 introduced — RMSNorm on q and k separately, stabilizing attention logits. Both FLUX and Open-Sora inherit it.

6. **`q = torch.cat((txt_q, img_q), dim=2)` (joint attention)**:
   - 中文: **整段代码最重要的一行**。在 head/seq 维上把 txt 和 img 的 q/k/v 拼起来,然后跑一次 softmax。这意味着 attention map 里 image patch 可以同时看 image patch 和 text token,反之亦然——这就是"joint attention"。比 cross-attention(只能 image → text 单向看)对称、强、训练快。
   - English: the **most important line**. Concat txt and img q/k/v along the sequence dim, run one softmax. The attention map lets image patches attend to both image patches and text tokens, and vice versa — true "joint attention". More symmetric and easier to train than asymmetric image→text cross-attention.

7. **`txt_attn, img_attn = attn1[:, :txt_q.shape[2]], attn1[:, txt_q.shape[2]:]`**:
   - 中文: 共享 softmax 的输出再按"前 N_txt 个是 txt 的,后 N_img 个是 img 的"切开,分别进各自的 proj 和 MLP。
   - English: slice the joint-attention output back into txt-half and img-half along the seq dim, route each through its own proj and MLP.

8. **`img = img + img_mod2.gate * img_mlp((1+scale)*norm2(img)+shift)`**:
   - 中文: MLP 之前的第二组 modulation。两组 `gate` 都是 zero-init,所以 block 起步是个 identity residual——和 DiT 一脉相承。
   - English: the second modulation, before the MLP. Both `gate`s are zero-init, so each block starts as an identity residual — straight from the DiT playbook.

## 类比 / The analogy

想象一个图书馆里有两个独立的会议室:一个室里坐的全是画家(image stream),另一室坐的全是评论家(text stream)。两个房间各自做笔记、用各自的语言。但每隔一段时间,所有人会一起走进一个公共大会议室开"联席讨论会"(joint attention)——大家在同一张白板上发言,听对方一切发言,讨论完各自带新见解回自己房间继续修改。MMDiT 就是这个"两个独立房间 + 一个共享会议"的结构;单流 DiT 等价于让画家和评论家自始至终挤在同一间房用同一支笔。

Picture a library with two separate meeting rooms: one full of painters (image stream), one full of critics (text stream). Each room takes its own notes in its own language. Periodically, everyone walks into a shared conference hall to brainstorm together on one whiteboard (joint attention), then carries new insights back to their own room. MMDiT is exactly that "two rooms + one shared meeting" structure; single-stream DiT is everyone crammed into one room with one pen.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

中文: 这是 `nano_wam` 课程里 **dit-block** 的**跨仓库变体**(原版从 facebookresearch/DiT 学过)。依赖项 `patchify-positional` 已经在 Wan2.1 RoPE 那节覆盖。它的输入:`img` 是 patchify+RoPE 后的视觉 latent token `[B, L_img, D]`,`txt` 是 T5/CLIP 文本编码 token `[B, L_txt, D]`,`vec` 是 timestep + 条件向量 `[B, D]`(通过 `WanTimeTextImageEmbedding` 类的产物),`pe` 是 RoPE。输出:更新后的 `(img, txt)` 双流。**下游谁**:N 个 DoubleStreamBlock 之后,Open-Sora 会把两路 cat 进 N 个 SingleStreamBlock(FLUX 也是这套混合,前 N 个 double + 后 M 个 single),最后 LastLayer 输出 noise prediction。**上游谁**:`text-conditioning`(T5 编码)+ `patchify-positional` + timestep embedding。**省掉会怎样**:在 nanoWAM 里你完全可以只用单流 DiTBlock,条件用 cross-attention 注入就够;upgrade 到 double-stream 是"想要条件遵循更紧、训练更稳"时的下一步。**生产 WAM 要补什么**:(a) 视频时空 attention(把 `L_img` 拆成 `T*H*W`,attention 要 spatial-only 或者 spatial-temporal 混合),(b) RoPE 替代 `pe` 的 sinusoidal,(c) sequence parallelism + ring attention 把超长 token 序列切到多卡,(d) FP8 路径(可以直接套你今天学的 dinov3 fp8 Linear),(e) action token 第三路 stream(为 WAM 的 action conditioning 留位置)。

English: this is the **cross-repo variant** of `dit-block` in the `nano_wam` curriculum — you've already covered the single-stream original from facebookresearch/DiT. Dependency `patchify-positional` was covered via Wan2.1 RoPE. Inputs: `img` is patchified+RoPE-encoded visual latent tokens `[B, L_img, D]`, `txt` is T5/CLIP text tokens `[B, L_txt, D]`, `vec` is the timestep+condition vector `[B, D]` (output of `WanTimeTextImageEmbedding`), and `pe` is RoPE. Output: updated `(img, txt)` pair. **Downstream**: after N DoubleStreamBlocks, Open-Sora / FLUX concat the streams and pass them through M SingleStreamBlocks, then a LastLayer predicts noise. **Upstream**: `text-conditioning` (T5 encoding) + `patchify-positional` + timestep embedding. **What breaks if you skip it**: nothing breaks — a single-stream DiTBlock + cross-attention is perfectly viable in nanoWAM. MMDiT is the upgrade you do when you want tighter condition adherence and more stable training. **What production WAM needs on top**: (a) spatio-temporal attention for video (split `L_img` into `T*H*W`, use spatial-only or block-sparse spatio-temporal patterns), (b) RoPE instead of sinusoidal `pe`, (c) sequence parallelism + ring attention to shard ultra-long token sequences across GPUs, (d) an FP8 path (drop in the dinov3 FP8 Linear you learned today), (e) a third action stream for WAM's action conditioning.

## 自己跑一遍 / Try it yourself

```python
# mmdit_block_demo.py — minimal double-stream block, no extra deps
import torch, torch.nn as nn
from dataclasses import dataclass

@dataclass
class Mod: shift: torch.Tensor; scale: torch.Tensor; gate: torch.Tensor

class Modulation(nn.Module):
    def __init__(self, D): super().__init__(); self.lin = nn.Linear(D, 6 * D)
    def forward(self, vec):
        s = self.lin(torch.nn.functional.silu(vec))[:, None].chunk(6, -1)
        return Mod(*s[:3]), Mod(*s[3:])

class DoubleStream(nn.Module):
    def __init__(self, D=64, H=4):
        super().__init__(); self.H, self.Dh = H, D // H
        self.img_mod = Modulation(D); self.txt_mod = Modulation(D)
        self.img_n1 = nn.LayerNorm(D, elementwise_affine=False)
        self.img_n2 = nn.LayerNorm(D, elementwise_affine=False)
        self.txt_n1 = nn.LayerNorm(D, elementwise_affine=False)
        self.txt_n2 = nn.LayerNorm(D, elementwise_affine=False)
        self.img_qkv = nn.Linear(D, 3 * D); self.img_proj = nn.Linear(D, D)
        self.txt_qkv = nn.Linear(D, 3 * D); self.txt_proj = nn.Linear(D, D)
        self.img_mlp = nn.Sequential(nn.Linear(D, 4*D), nn.GELU(), nn.Linear(4*D, D))
        self.txt_mlp = nn.Sequential(nn.Linear(D, 4*D), nn.GELU(), nn.Linear(4*D, D))
        # zero-init gates so first forward = identity residual
        for m in (self.img_mod, self.txt_mod): nn.init.zeros_(m.lin.weight); nn.init.zeros_(m.lin.bias)

    def _qkv(self, x, qkv):
        B, L, _ = x.shape
        q, k, v = qkv(x).reshape(B, L, 3, self.H, self.Dh).permute(2, 0, 3, 1, 4)
        return q, k, v

    def forward(self, img, txt, vec):
        im1, im2 = self.img_mod(vec); tm1, tm2 = self.txt_mod(vec)
        i = (1 + im1.scale) * self.img_n1(img) + im1.shift
        t = (1 + tm1.scale) * self.txt_n1(txt) + tm1.shift
        iq, ik, iv = self._qkv(i, self.img_qkv)
        tq, tk, tv = self._qkv(t, self.txt_qkv)
        # joint attention
        q = torch.cat([tq, iq], 2); k = torch.cat([tk, ik], 2); v = torch.cat([tv, iv], 2)
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        Lt = tq.shape[2]
        to, io = o[:, :, :Lt], o[:, :, Lt:]
        io = io.transpose(1, 2).reshape(img.shape); to = to.transpose(1, 2).reshape(txt.shape)
        img = img + im1.gate * self.img_proj(io)
        img = img + im2.gate * self.img_mlp((1 + im2.scale) * self.img_n2(img) + im2.shift)
        txt = txt + tm1.gate * self.txt_proj(to)
        txt = txt + tm2.gate * self.txt_mlp((1 + tm2.scale) * self.txt_n2(txt) + tm2.shift)
        return img, txt

B, D = 2, 64
img = torch.randn(B, 16*16, D); txt = torch.randn(B, 8, D); vec = torch.randn(B, D)
block = DoubleStream(D)
img2, txt2 = block(img, txt, vec)
print("first-forward identity check:")
print("  img diff:", (img2 - img).abs().max().item())  # should be ~0
print("  txt diff:", (txt2 - txt).abs().max().item())  # should be ~0
```

运行 / Run with:
```bash
pip install torch
python mmdit_block_demo.py
```

预期输出 / Expected output:
```
first-forward identity check:
  img diff: 0.0
  txt diff: 0.0
```

中文: zero-init gate 让第一次 forward 完全等价于 residual identity——你能 observed 到 `diff=0`,这就是 adaLN-Zero 起 "zero" 的来源。把 modulation 的初始化注释掉,你会看到 `diff` 立刻变大,训练前期 loss 也会更晃。

English: the zero-init gate makes the very first forward a pure identity residual — you'll see `diff=0`. That's literally what "Zero" in adaLN-Zero means. Comment out the modulation init and the `diff` jumps immediately; loss will jitter more during the first thousand steps.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **SD3 (`MMDiTBlock`)** / **Stable Diffusion 3**: 论文里画的就是这个图;Open-Sora 是它的开源忠实复制。
- **FLUX** / **Black Forest Labs FLUX**: 前 19 个 block 是 DoubleStream,后 38 个是 SingleStream,合计 57 个。这种"前 N 双流 + 后 M 单流"是 2024-2026 的工程默认选择。
- **Hunyuan-Video / CogVideoX-5B(后期)/ Hunyuan-Video, late CogVideoX**: 视频 DiT 同样用这套 double + single 拼装,只是 attention 加了时空轴。
- **AuraFlow / Lumina-Next**: 文本流额外加 register tokens,但 block 主结构一致。
- **`nano_wam` 课程里你的单流 DiTBlock**(2026-05-25 那篇 facebookresearch/DiT): 同结构的**单流**版本——参数量是 MMDiT 的一半,但条件遵循略弱。两者是 nanoWAM 主干的两种合理选择。
- **VLA 里的 SmolVLA + action expert**(你 2026-05-29 那篇): 思路非常像——VLM 和 action expert 各持参数,只在 attention 处共享 KV。这是**同一个组合原则在不同 modality 上的应用**。

## 注意事项 / Caveats / when it breaks

- **参数量翻倍 / parameter count doubles**: 中文: 每个 block 有完整两套 norm + qkv + proj + MLP,所以 MMDiT 同 hidden_size 比单流 DiT 参数大约 1.8×(LayerNorm 无参数所以不是严格 2×)。在显存紧张的 nanoWAM 里要权衡。
- **`txt` 序列长度往往远小于 `img` / `L_txt` ≪ `L_img`**: 中文: T5 编码一般 77-256 token,而 video latent 序列可达 16k+。joint attention 是 `(L_txt + L_img)^2` 的,主导项还是 `L_img`,所以 MMDiT 比 cross-attention 多出的开销有限。
- **first-forward identity 依赖 zero-init / first-forward identity needs zero-init**: 中文: 默认 `nn.Linear` 用 Xavier,不会给你 zero gate。务必显式 `nn.init.zeros_(img_mod.lin.weight & bias)`,否则 adaLN-Zero 的好处直接消失。
- **QK-Norm 不是可选 / QK-Norm is not optional**: 中文: 没它训 high-resolution 视频很容易在 attention logit 上炸。SD3 论文专门强调过。
- **共享 `vec` 但权重不同 / shared `vec` but separate weights**: 中文: 不要把 `img_mod` 和 `txt_mod` 合并成一个共享 Linear——这样会失去 MMDiT 的核心好处(modality-specific 条件路径)。
- **batch 维度 `vec` broadcast / `vec` shape gotcha**: 中文: 代码里 `out = lin(...)[:, None, :]` 加了一个 seq 维,让 modulation 能 broadcast 到 token 序列上。自己写时漏掉这个 `[:, None, :]` 是常见 bug。

## 延伸阅读 / Further reading

- [Stable Diffusion 3 paper §3.2 MM-DiT](https://arxiv.org/abs/2403.03206) — MMDiT 的论文出处
- [FLUX 模型卡 / blog](https://blackforestlabs.ai/announcing-black-forest-labs/) — 工程化版本
- [DiT 原始论文 (Peebles & Xie 2022)](https://arxiv.org/abs/2212.09748) — 单流前身
- [adaLN-Zero discussion(DiT 论文 §3.3)](https://arxiv.org/abs/2212.09748)
- 你之前的 daily code:[DiT adaLN-Zero block](../../2026/05/2026-05-25-dit-adaln-zero-block.md) ← 单流原版,和今天对照读
