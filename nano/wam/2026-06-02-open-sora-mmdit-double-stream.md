---
date: 2026-06-02
topic: wam
source: wam
repo: hpcaitech/Open-Sora
file: opensora/models/mmdit/layers.py
permalink: https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/mmdit/layers.py#L195-L253
difficulty: advanced
read_time: ~12 min
tags: [code-of-the-day, wam, mmdit, double-stream, flux, dit-block-variant]
build_role: nanoWAM / dit-block (variant) — replaces vanilla cross-attention DiT with parallel image/text streams that share one joint attention, the Flux/SD3 design.
---

# 一次 attention 调用,两条互不共享权重的流 / One attention call, two streams that never share weights

> **一句话 / In one line**: MMDiT 双流块给图像和文本各自一套 Q/K/V、norm、MLP 和 adaLN 调制,然后把 Q/K/V 沿 sequence 维拼起来过**一次**联合 attention —— 没有任何 cross-attention 层,文本和图像在同一锅汤里互相影响。 / The MMDiT double-stream block gives image and text *separate* Q/K/V, norms, MLP and adaLN modulations, then concatenates Q/K/V along the sequence dim for ONE joint attention call — no cross-attention anywhere, image and text just meet in the same attention soup.

## 为什么重要 / Why this matters

到 5 月底为止 nanoWAM 课程已经覆盖了原始 DiT 的 adaLN-Zero 块(`facebookresearch/DiT` 的 `DiTBlock`)。但 2024-2025 年发布的几乎所有最强生图/生视频模型 —— Flux、Stable Diffusion 3、Open-Sora、HunyuanVideo —— 都改成了 **MMDiT(Multi-Modal DiT)双流**架构。它对 vanilla DiT 是一个关键升级:文本不再通过单独的 cross-attention 注入,而是和图像 token 一起,在同一个 self-attention 里互相看见。代价是参数翻倍(每个块两套权重),收益是文本-图像对齐显著提升 —— 在 nanoWAM 里你可以把「文本」换成「动作」,用同样的双流结构让 action tokens 和 video latents 真正互相 attend,这正是 StarVLA / lingbot-va 一类工作的核心。

By late May the nanoWAM curriculum had already covered the original DiT adaLN-Zero block (`facebookresearch/DiT`'s `DiTBlock`). But nearly every top-of-the-leaderboard 2024-2025 image/video model — Flux, Stable Diffusion 3, Open-Sora, HunyuanVideo — switched to the **MMDiT (Multi-Modal DiT) double-stream** design. It's a critical upgrade over vanilla DiT: text no longer enters through a separate cross-attention layer, it joins the image tokens inside the SAME self-attention. The cost is doubled parameters per block (two sets of weights). The benefit is dramatically better text-image alignment — and in nanoWAM you can replace "text" with "actions" and use the same double-stream pattern to let action tokens and video latents truly attend to each other, which is exactly what StarVLA / lingbot-va are doing.

## 代码 / The code

`hpcaitech/Open-Sora` — [`opensora/models/mmdit/layers.py`](https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/mmdit/layers.py#L195-L253)

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

1. **`img_mod1, img_mod2 = attn.img_mod(vec)` 和 `txt_mod1, txt_mod2 = attn.txt_mod(vec)`**:
   - 中文: 两套独立的 adaLN-Zero 调制器,从同一个全局条件向量 `vec`(timestep + 池化文本)各自产生 6 个调制值(shift, scale, gate)× 2 sub-block。即每个块里图像分支用一组,文本分支用另一组。同一个时间步,两条流被「调到不同节奏」。
   - English: Two independent adaLN-Zero modulators take the same global conditioning vector `vec` (timestep + pooled text) and emit 6 modulation values (shift, scale, gate) × 2 sub-blocks each. So each block tunes the image branch with one set and the text branch with another. Same timestep, two streams are "set to different tempos."

2. **`img_modulated = (1 + img_mod1.scale) * img_norm1(img) + img_mod1.shift`**:
   - 中文: 经典 adaLN-Zero 公式 —— normalize 后乘 `(1+scale)` 再加 `shift`。`scale` 初始化为 0,所以训练开始时这个 block 是 identity(不改 img),网络可以慢慢学到怎么调制。这是 DiT 之所以稳的原因,MMDiT 完整继承。
   - English: Classic adaLN-Zero formula — normalize, multiply by `(1+scale)`, add `shift`. `scale` zero-inits so the block is an identity at the start of training (doesn't touch `img`), and the network can learn its modulation gradually. This is *the* reason DiT trains stably, and MMDiT inherits it verbatim.

3. **`img_qkv = attn.img_attn.qkv(img_modulated)` 和 `txt_qkv = attn.txt_attn.qkv(txt_modulated)`**:
   - 中文: **两套独立的 QKV linear**,这是 MMDiT vs vanilla DiT 最大的结构差异。vanilla DiT 里文本通过单独的 cross-attention 层注入,只有 image 有 QKV;MMDiT 里文本和图像各有一套完整的 QKV 投影,网络学到的是「两套独立的 token 嵌入」。
   - English: **Two independent QKV linears** — the single biggest structural difference vs vanilla DiT. Vanilla DiT injects text through a separate cross-attention layer (only image has QKV); MMDiT gives text its own full QKV projection, so the network learns "two independent token embeddings."

4. **`attn.img_attn.norm(img_q, img_k, img_v)` (QK-Norm)**:
   - 中文: SD3 paper 提出的稳定 trick:对 q 和 k 各自做 RMSNorm,再去算 attention 分数。理由:大模型训到深层时 q·k 数值会爆,QK-Norm 把它们限制在单位球面上,attention 分数变成「方向相似度」而不是「方向+幅度」。MMDiT 标配。
   - English: Stability trick introduced in the SD3 paper: apply RMSNorm to q and k separately *before* the dot product. Reason: in deep models q·k explodes in magnitude; QK-Norm constrains them to the unit sphere, turning attention scores into "directional similarity" rather than "direction × magnitude." Standard in MMDiT.

5. **`q = torch.cat((txt_q, img_q), dim=2)` (and same for k, v)** — **核心一行 / the key line**:
   - 中文: 这一句是 MMDiT 的精髓。把文本和图像的 Q/K/V 沿 **sequence 维**(dim=2,因为 layout 是 `(B, H, L, D)`)拼起来,然后跑**一次** attention。结果是:每个 image token 都能看到所有 text token,反之亦然,完全 bidirectional,没有任何 mask。
   - English: This single line is the soul of MMDiT. Concatenate text and image Q/K/V along the **sequence axis** (dim=2, since layout is `(B, H, L, D)`), then run ONE attention call. Result: every image token can see every text token and vice versa, fully bidirectional, no mask anywhere.

6. **`txt_attn, img_attn = attn1[:, :txt_q.shape[2]], attn1[:, txt_q.shape[2]:]`**:
   - 中文: attention 出来后按前 `L_txt` 个切回给文本流,后 `L_img` 个切回给图像流。两条流物理上分开,但内容已经互相浸染过 —— 关键就是「分→合→分」这个结构。
   - English: After attention, split the first `L_txt` tokens back to the text stream and the rest to the image stream. The two streams are physically separated again, but their contents have already been cross-pollinated — the magic is exactly this "split → merge → split" structure.

7. **`img = img + img_mod1.gate * attn.img_attn.proj(img_attn)` (and the MLP sub-block)**:
   - 中文: 两条流各自的输出 projection、残差和 MLP。**注意 `img_mod1.gate` 和 `txt_mod1.gate` 是不同的 scalar**,所以网络可以学到「这一层对图像信号开大,对文本信号关小」的差异化策略。比如靠近输出的层往往把文本 gate 关掉,让文本「冻结」、图像继续 refine。
   - English: Each stream's own output projection, residual, and MLP. **Note `img_mod1.gate` and `txt_mod1.gate` are different scalars**, so the network can learn differential strategies like "this layer opens the image gate wide and clamps the text gate." Layers near the output typically clamp the text gate to "freeze" text and let image refine further.

8. **整段是 `Processor`,不是 `nn.Module`**:
   - 中文: 这是 diffusers/Flux 的设计 —— attention 模块本身只持有权重(在 `DoubleStreamBlock` 里),计算逻辑放在一个无状态的 `Processor` 类里。这样可以热替换 attention backend(FlashAttention、xformers、Flex)而不动权重,通过 `set_processor` 切换。
   - English: This is the diffusers/Flux design — the attention module only holds weights (in `DoubleStreamBlock`); the compute lives in a stateless `Processor` class. That lets you hot-swap attention backends (FlashAttention, xformers, Flex) without touching weights, via `set_processor`.

## 类比 / The analogy

想象两支乐队同台演出,一支演奏图像(古典弦乐),一支演奏文本(爵士)。两支乐队各有自己的乐器、自己的指挥(`img_mod` / `txt_mod`,根据全局节拍 `vec` 调速)、自己的乐谱(QKV 权重)。但他们站在同一个舞台上、面对同一群观众 —— 当 attention 这一刻到来,所有乐手不分队伍地一起聆听台下任何人的发声(`torch.cat`),决定下一拍该奏什么。听完之后,弦乐手回弦乐区,爵士手回爵士区(split 回去),各自继续按自己的乐谱演奏(MLP)—— 但他们已经互相影响过了。

Picture two bands sharing a stage — a string quartet plays the "image" part, a jazz combo plays the "text" part. Each band has its own instruments, its own conductor (`img_mod` / `txt_mod`, taking tempo cues from the global `vec`), and its own sheet music (QKV weights). But they're on the same stage facing the same audience — when the attention moment comes, every musician listens to every other musician regardless of band (`torch.cat`) and decides the next beat. After listening, strings return to the string section, jazz returns to the combo (split back), and each continues from its own sheet music (MLP) — but they've already influenced each other.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-WAM

**Curriculum slot**: cross-repo variant of `dit-block` (originally covered 2026-05-25 via vanilla DiT's `DiTBlock`). Dependency: `patchify-positional` (covered 2026-05-29). This is the **upgrade path** when your nanoWAM has the basic DiT working and you want better text-image alignment, or you want to swap "text" for "actions."

在你的 nanoWAM 里这是 `dit-block` 课程项的**进阶变体**。最小集成步骤:(1)把单流 DiTBlock 拆成 `DoubleStreamBlock` —— 每个 block 持有 `img_*` 和 `txt_*` 两套 LayerNorm + Linear。(2)forward 收两个输入 `(img_latent, txt_emb)`,各自调制和 patchify。(3)关键改动:在 attention 前 cat Q/K/V,attention 后切回去。(4)对 nanoWAM 的 action 版本,把 `txt` 改名 `act`,文本 encoder 改成 action MLP(把 7 维 EE pose 投到 dim 维即可)。(5)上层 `Transformer` 把 N 个 DoubleStreamBlock 串起来,最后再用几个 SingleStreamBlock(把两条流合并)收尾 —— Flux/Open-Sora 都是这个「双流 14 层 + 单流 38 层」的混合架构。

In your nanoWAM this is the **advanced variant** of the `dit-block` slot. Minimal integration: (1) split single-stream DiTBlock into `DoubleStreamBlock` — each block holds `img_*` and `txt_*` versions of LayerNorm + Linear. (2) forward takes two inputs `(img_latent, txt_emb)`, each modulated and projected independently. (3) The critical change: cat Q/K/V before attention, split after. (4) For the action-conditioned nanoWAM, rename `txt` → `act` and replace the text encoder with an action MLP (a Linear from 7-dim EE pose to `dim` is enough). (5) The outer `Transformer` stacks N DoubleStreamBlocks then a few SingleStreamBlocks (merge to one stream) at the end — Flux and Open-Sora both use this hybrid "double-stream 14 + single-stream 38" architecture. Skipping this and staying with cross-attention DiT is fine but you'll cap your text-image alignment around SD1.5/2 levels.

## 自己跑一遍 / Try it yourself

```python
# pip install torch einops
import torch, torch.nn as nn
from einops import rearrange

class MiniDoubleStream(nn.Module):
    def __init__(self, dim=64, heads=4):
        super().__init__()
        self.h, self.d = heads, dim // heads
        for stream in ("img", "txt"):
            self.add_module(f"{stream}_ln", nn.LayerNorm(dim, elementwise_affine=False))
            self.add_module(f"{stream}_qkv", nn.Linear(dim, dim * 3))
            self.add_module(f"{stream}_proj", nn.Linear(dim, dim))

    def forward(self, img, txt):
        def split_qkv(stream_name, x):
            qkv = getattr(self, f"{stream_name}_qkv")(getattr(self, f"{stream_name}_ln")(x))
            return rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.h, D=self.d)
        iq, ik, iv = split_qkv("img", img)
        tq, tk, tv = split_qkv("txt", txt)
        q = torch.cat((tq, iq), dim=2)
        k = torch.cat((tk, ik), dim=2)
        v = torch.cat((tv, iv), dim=2)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "B H L D -> B L (H D)")
        t_out, i_out = out[:, :txt.size(1)], out[:, txt.size(1):]
        return img + self.img_proj(i_out), txt + self.txt_proj(t_out)

m = MiniDoubleStream()
img = torch.randn(1, 256, 64)  # 16x16 latent
txt = torch.randn(1, 77, 64)   # 77 text tokens
img_out, txt_out = m(img, txt)
print(f"img: {img.shape} -> {img_out.shape}")
print(f"txt: {txt.shape} -> {txt_out.shape}")
print(f"img and txt have NO shared QKV weights:",
      not (m.img_qkv.weight is m.txt_qkv.weight))
```

运行 / Run with:
```bash
pip install torch einops
python try.py
```

预期输出 / Expected output:
```
img: torch.Size([1, 256, 64]) -> torch.Size([1, 256, 64])
txt: torch.Size([1, 77, 64]) -> torch.Size([1, 77, 64])
img and txt have NO shared QKV weights: True
```

把 `iq, ik, iv` 那一支删掉(让两条流不 cat,各自走 attention),你会看到 img 和 txt 互相完全看不见 —— 那就退化成了「两个独立的 transformer 跑在一起」。`torch.cat` 那两行是让两条流变成一锅汤的唯一原因。

If you delete the image branch entirely (let the two streams attend independently, no cat), img and txt become invisible to each other — degenerating into "two independent transformers running side by side." Those `torch.cat` lines are the only reason the two streams become one soup.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Stable Diffusion 3 (MMDiT 原始论文)** / **Stable Diffusion 3 (the MMDiT paper)**: 一模一样的双流结构,SD3 是这套架构的标准参考实现。 / Same double-stream structure exactly; SD3 is the canonical reference implementation.
- **Flux.1 [dev]** / **Flux.1 [dev]**: 用 19 个 DoubleStreamBlock + 38 个 SingleStreamBlock,Open-Sora 的代码直接基于 Flux 改。Black Forest Labs 把这套架构变成了开源生图新基线。 / 19 DoubleStreamBlocks + 38 SingleStreamBlocks; Open-Sora's code is a direct port. Black Forest Labs made this the open-source baseline.
- **HunyuanVideo (Tencent)** / **HunyuanVideo (Tencent)**: 视频版双流 + 3D RoPE + 时空 attention,本质上是同一个模板换成 3D。 / Video variant: double stream + 3D RoPE + spatio-temporal attention — same template in 3D.
- **StarVLA / Robbyant-VA 的 action-as-text trick** / **StarVLA / Robbyant-VA's action-as-text trick**: 用双流块,但 `txt` 流换成 `act` 流;curriculum 里这正是 nanoWAM 从单流升级到带动作条件的双流的标准路径。 / Use the double-stream block but rename `txt` → `act`; in our curriculum this is exactly nanoWAM's upgrade path from a single-stream image model to an action-conditioned dual stream.
- **vanilla DiT(已覆盖) + cross-attention** / **Vanilla DiT (covered) + cross-attention**: 对比组 —— 一个单流块 + 一层独立的 cross-attention 注入文本。MMDiT 是把这一额外层「内化」到 self-attention 里的等价改写。 / The control group — single-stream block + a separate cross-attention layer. MMDiT is the equivalent rewrite that "internalizes" that extra layer into self-attention.

## 注意事项 / Caveats / when it breaks

- **参数翻倍** / **2× parameters**: 每个块两套 QKV/proj/MLP/norm,参数量大约是 vanilla DiT 的 1.6-1.8 倍(MLP 也翻倍但 norm 没参数)。Flux-dev 12B 里大约 7B 都在这些 double-stream 块上。 / Each block doubles QKV/proj/MLP/norm, so total params are roughly 1.6-1.8× vanilla DiT (MLP also doubles, norms don't have params). About 7B of Flux-dev's 12B lives in these double-stream blocks.
- **`L_txt + L_img` 决定 attention 复杂度** / **`L_txt + L_img` drives attention cost**: attention 是 O((L_txt + L_img)²),所以文本特别长会显著拖慢。SD3 的做法是用 T5 + CLIP 的 77 + 77 = 154 个 text token,Flux 是 256,Open-Sora 视频版可能 512+。计算 budget 必须按 `L_total²` 算。 / Attention is O((L_txt + L_img)²), so very long text really hurts. SD3 uses T5 + CLIP for 77 + 77 = 154 text tokens, Flux uses 256, Open-Sora video can hit 512+. Budget compute by `L_total²`.
- **`pe` 位置编码必须包含两条流** / **`pe` positional encoding must span both streams**: 单流 RoPE 表只有 `L_img` 项,双流要 `L_txt + L_img` 项 —— 通常给文本用一段不同的 RoPE 频率,或者干脆给文本不加 pos(让顺序由 cross-stream attention 自己学)。 / Single-stream RoPE only covers `L_img`; double-stream needs `L_txt + L_img`. Usual fix: give text a different RoPE frequency band, or skip positions for text entirely and let the cross-stream attention learn ordering.
- **训练初期可能崩** / **Early training instability**: 因为两条流参数翻倍,梯度方差天然就大。建议照 SD3 用 QK-Norm(本代码已有)+ scale-down 初始化 `q_proj/k_proj` 的 weight 标准差。 / Doubled parameters means doubled gradient variance early on. Follow SD3: enable QK-Norm (already present) and scale down `q_proj/k_proj` init weight std.
- **Processor 设计的陷阱** / **Processor design gotcha**: `Processor` 拿到的 `attn` 是一个引用,任何对 `attn.xxx` 的赋值会写回 `DoubleStreamBlock`。一般只读,但有些自定义 backend 会偷偷给 attn 加 cache,要小心多 process 共享同一个 attn 时的状态污染。 / The `Processor` gets `attn` by reference, so any assignment to `attn.xxx` mutates the underlying `DoubleStreamBlock`. Typically read-only, but custom backends sometimes sneak a cache onto `attn`, causing state pollution when multiple processors share one block.

## 延伸阅读 / Further reading

- [Stable Diffusion 3 paper (MMDiT)](https://arxiv.org/abs/2403.03206)
- [Flux.1 model card](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [Open-Sora MMDiT model code](https://github.com/hpcaitech/Open-Sora/blob/main/opensora/models/mmdit/model.py)
- [HunyuanVideo paper — video MMDiT](https://arxiv.org/abs/2412.03603)
- [Parallel-layers trick (single-stream block)](https://arxiv.org/abs/2302.05442)
