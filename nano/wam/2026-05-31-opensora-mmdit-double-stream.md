---
date: 2026-05-31
topic: wam
source: wam
repo: hpcaitech/Open-Sora
file: opensora/models/mmdit/layers.py
permalink: https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/mmdit/layers.py#L195-L253
difficulty: advanced
read_time: ~14 min
tags: [code-of-the-day, wam, dit, mmdit, sd3, flux, double-stream, adaln-zero]
build_role: DiT-block variant — joint image+text attention with separate streams (SD3 / Flux pattern)
---

# MM-DiT 把"文字"和"图像"做成两条对等的 stream / MM-DiT treats text and image as two peer streams in one attention

> **一句话 / In one line**: 图像和文字各自有自己的 adaLN 调制 + 自己的 Q/K/V 投影,然后把两边的 QKV *拼起来* 做一次联合 attention,出口再切回两边 —— SD3 / Flux 的核心 trick。 / Image and text each have their own adaLN modulation + their own Q/K/V projection; the QKV pairs are then *concatenated* into one joint attention, and the output is split back — the core SD3 / Flux trick.

## 为什么重要 / Why this matters

5-25 教过的 *vanilla* DiT block(adaLN-Zero)只关心一种 token:image latent。文本怎么进来?早期做法是 cross-attention —— 加一条额外的 attention 层,query 是 image、KV 是文字。问题是 cross-attention 让文本永远处于"被读取"的位置,反过来 image 没法影响文本的表示,文字侧信息利用得不充分。Stable Diffusion 3 / Flux 的革命就是 MM-DiT:把文本 token *直接拼到* image token 旁边,一起做 self-attention。这样每一步 image 既能看文字、文字也能看 image、文字之间也能互相 refine。Open-Sora 把这个 pattern 写得格外干净 —— 59 行 `DoubleStreamBlockProcessor` 把 SD3 论文里的所有关键 trick(adaLN-Zero × 2 streams、QK-RMSNorm、concat-attention、split-back)都摆在台面上。

The *vanilla* DiT block (adaLN-Zero) covered on 2026-05-25 only cares about one token type: image latents. Text is usually injected via cross-attention — a separate attention layer where image is the query and text is KV. The problem: text is always "read from," never "written to," so image cannot refine text representations and a lot of text-side signal is wasted. SD3 / Flux's revolution is MM-DiT: text tokens are *concatenated* alongside image tokens and a single self-attention runs over both. Every step lets image see text, text see image, and text-to-text refine. Open-Sora's 59-line `DoubleStreamBlockProcessor` is the cleanest exposition I know — all the SD3 tricks (twin adaLN-Zeros, QK-RMSNorm, concat-attention, split-back) on one screen.

## 代码 / The code

`hpcaitech/Open-Sora` — [`opensora/models/mmdit/layers.py`](https://github.com/hpcaitech/Open-Sora/blob/7ad6a96a135feb81f755c84fb391818718f6beb2/opensora/models/mmdit/layers.py#L195-L253)

```python
class DoubleStreamBlockProcessor:
    def __call__(self, attn, img, txt, vec, pe):
        # attn is the DoubleStreamBlock;
        # process img and txt separately while both is influenced by text vec

        # vec will interact with image latent and text context
        img_mod1, img_mod2 = attn.img_mod(vec)  # shift, scale, gate for each mod
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        if attn.img_attn.fused_qkv:
            img_qkv = attn.img_attn.qkv(img_modulated)
            img_q, img_k, img_v = rearrange(
                img_qkv, "B L (K H D) -> K B H L D",
                K=3, H=attn.num_heads, D=attn.head_dim)
        else:
            img_q = rearrange(attn.img_attn.q_proj(img_modulated), "B L (H D) -> B H L D", H=attn.num_heads)
            img_k = rearrange(attn.img_attn.k_proj(img_modulated), "B L (H D) -> B H L D", H=attn.num_heads)
            img_v = rearrange(attn.img_attn.v_proj(img_modulated), "B L (H D) -> B H L D", H=attn.num_heads)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)  # RMSNorm for QK Norm (SD3)

        # prepare txt for attention
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        if attn.txt_attn.fused_qkv:
            txt_qkv = attn.txt_attn.qkv(txt_modulated)
            txt_q, txt_k, txt_v = rearrange(
                txt_qkv, "B L (K H D) -> K B H L D",
                K=3, H=attn.num_heads, D=attn.head_dim)
        else:
            txt_q = rearrange(attn.txt_attn.q_proj(txt_modulated), "B L (H D) -> B H L D", H=attn.num_heads)
            txt_k = rearrange(attn.txt_attn.k_proj(txt_modulated), "B L (H D) -> B H L D", H=attn.num_heads)
            txt_v = rearrange(attn.txt_attn.v_proj(txt_modulated), "B L (H D) -> B H L D", H=attn.num_heads)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention — image and text together via concat
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn1[:, : txt_q.shape[2]], attn1[:, txt_q.shape[2] :]

        # calculate the img blocks
        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        # calculate the txt blocks
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        return img, txt
```

## 逐行讲解 / What's happening

1. **`vec` 同时驱动 image 和 text 的调制 / `vec` modulates both streams (lines 201-202)**:
   - 中文: `vec` 通常是 `time_emb + pooled_text_emb`(全局上下文)。它被两份独立的 `Modulation` 模块各自映射成 6 个调制参数 `(shift, scale, gate) × 2`,因为每条 stream 各有两个残差子层(attn + mlp)。注意 `img_mod` 和 `txt_mod` 是 *不同* 的模块 —— 各自学怎么从全局 vec 提取调制信号。
   - English: `vec` is typically `time_emb + pooled_text_emb` (the global context). Two independent `Modulation` modules turn it into 6 params each — `(shift, scale, gate) × 2` — because each stream has two residual sub-layers (attn + mlp). Critically, `img_mod` and `txt_mod` are *different* modules, each learning how to extract its modulation from the same global vec.

2. **adaLN-Zero 调制 / adaLN-Zero modulation (lines 205-206, 223-224)**:
   - 中文: `(1 + scale) * norm(x) + shift` 是 adaLN-Zero 的标准式子(5-25 笔记里讲过)。注意是 `(1 + scale)` 不是 `scale` —— 初始化 `scale=0` 时它退化成 identity,模型才能稳定从"什么也不调制"开始训。
   - English: `(1 + scale) * norm(x) + shift` is the adaLN-Zero formula (covered on 2026-05-25). The `1 +` is the "Zero" part — at init `scale=0` makes the modulation an identity, letting the network start from "no modulation" and learn from there.

3. **`fused_qkv` 的两条路径 / The `fused_qkv` two-path branch (lines 208-220, 225-236)**:
   - 中文: 性能优化分支。融合路径用一个 `nn.Linear(d, 3d)` 同时算 Q/K/V,然后 `rearrange("B L (K H D) -> K B H L D", K=3)` 一次切开三块;非融合路径有 3 个独立 Linear。融合通常快 20-30% 因为 GEMM 更大;但加载预训练权重不方便,所以两套都留。
   - English: a perf optimization branch. Fused: one `nn.Linear(d, 3d)` computes Q/K/V at once, then `rearrange("B L (K H D) -> K B H L D", K=3)` splits them in one shot. Unfused: 3 separate `Linear`s. Fused is ~20-30% faster because the GEMM is bigger; unfused is friendlier to loading checkpoints from non-fused models — hence both paths.

4. **QK-RMSNorm (lines 216, 232)**:
   - 中文: `attn.img_attn.norm(img_q, img_k, img_v)` 对 Q 和 K 各做一次 RMSNorm。这是 SD3 论文专门强调的稳定性修复 —— 不加的话在 fp16/bf16 下 attention logit `q·k^T` 会数值爆炸,训练 loss 突然飞到 NaN。这一行短小但救命。
   - English: `attn.img_attn.norm(img_q, img_k, img_v)` applies RMSNorm separately to Q and K. SD3's paper calls this out as a stability fix — without it, the `q·k^T` logits explode in fp16/bf16 and loss flips to NaN mid-training. Short line, very critical.

5. **核心:concat 做联合 attention / The core: concat for joint attention (lines 239-244)**:
   - 中文: `torch.cat((txt_q, img_q), dim=2)` 把文字 token 拼到 image token 的左边(`dim=2` 是序列长度维,因为 `B H L D` 布局)。一次 `attention(q, k, v, pe=pe)` 同时让两个流互相 attend。出口 `attn1[:, : txt_q.shape[2]]` 截前一段是文字 attention 结果,`attn1[:, txt_q.shape[2] :]` 是 image attention 结果 —— 单纯的索引切片就完成了双向信息流。
   - English: `torch.cat((txt_q, img_q), dim=2)` prepends text tokens to image tokens along the sequence axis (which is `dim=2` in the `B H L D` layout). One `attention(q, k, v, pe=pe)` call lets the two streams attend to each other and themselves. The output is split back by simple indexing — `attn1[:, : txt_q.shape[2]]` is text, `attn1[:, txt_q.shape[2] :]` is image. That's it. Bidirectional info flow in two lines.

6. **门控残差 / Gated residual (lines 247-252)**:
   - 中文: `img = img + img_mod1.gate * proj(img_attn)` —— `gate` 也是从 `vec` 里学的(adaLN-Zero 初始化是 0),所以训练初期 attention 子层"几乎没贡献",网络从恒等映射开始。同理 MLP 子层。文字流走同款双残差。
   - English: `img = img + img_mod1.gate * proj(img_attn)` — the `gate` is also learned from `vec` (adaLN-Zero init to 0), so the attention sub-layer "contributes almost nothing" early in training and the network starts from identity. Same for the MLP. The text stream has the symmetric double residual.

## 类比 / The analogy

想象一个法庭。Vanilla DiT 块像 "image 是被告,text 是法官念给陪审团的剧本"(cross-attention 单向)。MM-DiT 块像 "image 和 text 都是 *证人*,被请到同一张证人席上 —— image 的每一句话 text 都听得到,反过来也成立,而且证人之间还会相互打断、补充"。`adaLN(vec=time)` 像审判长在每一轮发言前给两位证人各自一张"今天的语气提示卡"(怎么强调、怎么淡化)。`gate=0` 初始化像第一天他们俩都安静坐着、只到必要时才开口 —— 训练就是慢慢让他们学会"什么时候说话比较有用"。出口的索引切片只是法警把两位证人各自带回自己的座位。

Picture a courtroom. A vanilla DiT block is "image is the defendant, text is the script the judge reads to the jury" (one-way cross-attention). An MM-DiT block is "both image *and* text are witnesses sitting on the *same* witness stand — image hears every text utterance, vice versa, and witnesses interrupt and complement each other." `adaLN(vec=time)` is the bailiff handing each witness a tone card before each round ("emphasize this, downplay that"). The `gate=0` init means on day one both witnesses sit silently and only speak when necessary — training is the slow process of learning when speaking helps. The output index-split is just the bailiff returning each witness to their seat.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-WAM

**Curriculum item**: `dit-block` — *advanced cross-repo variant* of the vanilla DiT block already covered on 2026-05-25 (`facebookresearch/DiT`).

中文: nanoWAM 的 `dit-block` slot 在 5-25 已经被 vanilla adaLN-Zero 块占住了 —— 那是文本无关、纯图像 latent 用的最小完备 DiT 块。今天教的 MM-DiT 是它的 *升级路线*:当你要做 text-to-video / text-to-image 生成时,把 vanilla `DiTBlock` 换成 `DoubleStreamBlock`,文本不再走 cross-attention 而是变成对等的第二条流。在你的 from-scratch 实现里,这个组件位于 patchify-positional 之后(它接受已经 RoPE 过的 q/k)、temporal-compression 之前(它处理的是 2D-patch 的 token,3D 时序 caching 由外层 VAE 解决),输入是 `(img, txt, vec, pe)` 四件套,输出是更新后的 `(img, txt)` 两件套。如果你 *要构造一个 action-conditioned 世界模型*,把这里的 `txt` 换成 `action_tokens`(同样的形状,但 embedder 不同),你就得到了一条干净的 "image stream + action stream" 双流架构 —— 这正是 5-29 教过的 lingbot-va FlexAttention 路径的另一种实现方式,区别在于:lingbot 用 *单流 + mask* 来区分 image vs action token,MM-DiT 用 *双流 + 各自 modulation*。生产级实现还需要补:(1) 一些层做 single-stream(SD3/Flux 都是混着用的,前 N 层 double、后 M 层 single),(2) sequence-parallel 切分,(3) FlexAttention 把 RoPE 嵌入 attention 内部。

English: nanoWAM's `dit-block` slot has been filled by the vanilla adaLN-Zero block from 2026-05-25 (`facebookresearch/DiT`) — the text-agnostic minimal complete DiT block. Today's MM-DiT is its *upgrade path*: when you build text-to-video / text-to-image, swap the vanilla `DiTBlock` for `DoubleStreamBlock` and text becomes a peer stream instead of cross-attention bait. In a from-scratch implementation this component sits between patchify-positional (which produces RoPE'd q/k) and temporal-compression (3D timing is handled by the surrounding VAE). Inputs: `(img, txt, vec, pe)`. Outputs: updated `(img, txt)`. For an **action-conditioned** world model, swap `txt` for `action_tokens` (same shape, different embedder) and you get a clean "image stream + action stream" dual-stream architecture — an alternative to the lingbot-va FlexAttention path from 2026-05-29. The two differ in spirit: lingbot uses *single-stream + mask* to distinguish image vs action tokens, MM-DiT uses *dual-stream + separate modulation*. Production additions: (1) some layers as single-stream (SD3/Flux interleave them — first N layers double, last M layers single), (2) sequence-parallel sharding, (3) FlexAttention embedding RoPE inside the kernel.

## 自己跑一遍 / Try it yourself

```python
# pip install torch
# Minimal MM-DiT block — no Open-Sora dependency.
import torch
import torch.nn as nn
import torch.nn.functional as F

class StreamModulation(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, 6 * d, bias=True)
        nn.init.zeros_(self.lin.weight); nn.init.zeros_(self.lin.bias)   # adaLN-Zero init
    def forward(self, vec):
        s1, sc1, g1, s2, sc2, g2 = self.lin(F.silu(vec))[:, None].chunk(6, dim=-1)
        return (s1, sc1, g1), (s2, sc2, g2)

class MMDiTBlock(nn.Module):
    def __init__(self, d=64, nh=4):
        super().__init__()
        self.nh, self.d = nh, d
        self.img_mod, self.txt_mod = StreamModulation(d), StreamModulation(d)
        self.img_n1 = nn.LayerNorm(d, elementwise_affine=False)
        self.txt_n1 = nn.LayerNorm(d, elementwise_affine=False)
        self.img_qkv, self.txt_qkv = nn.Linear(d, 3*d), nn.Linear(d, 3*d)
        self.img_proj, self.txt_proj = nn.Linear(d, d), nn.Linear(d, d)
        self.img_n2 = nn.LayerNorm(d, elementwise_affine=False)
        self.txt_n2 = nn.LayerNorm(d, elementwise_affine=False)
        self.img_mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.txt_mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))

    def _qkv(self, qkv_proj, x):
        B, L, _ = x.shape
        qkv = qkv_proj(x).reshape(B, L, 3, self.nh, self.d // self.nh).permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]                        # (B, nh, L, d/nh) each

    def forward(self, img, txt, vec):
        (s1, sc1, g1), (s2, sc2, g2) = self.img_mod(vec)
        (ts1, tsc1, tg1), (ts2, tsc2, tg2) = self.txt_mod(vec)
        iq, ik, iv = self._qkv(self.img_qkv, (1 + sc1)*self.img_n1(img) + s1)
        tq, tk, tv = self._qkv(self.txt_qkv, (1 + tsc1)*self.txt_n1(txt) + ts1)
        q = torch.cat([tq, iq], dim=2); k = torch.cat([tk, ik], dim=2); v = torch.cat([tv, iv], dim=2)
        a = F.scaled_dot_product_attention(q, k, v)
        tA, iA = a[:, :, :tq.shape[2]], a[:, :, tq.shape[2]:]
        tA = tA.transpose(1, 2).reshape(*txt.shape); iA = iA.transpose(1, 2).reshape(*img.shape)
        img = img + g1 * self.img_proj(iA)
        img = img + g2 * self.img_mlp((1 + sc2)*self.img_n2(img) + s2)
        txt = txt + tg1 * self.txt_proj(tA)
        txt = txt + tg2 * self.txt_mlp((1 + tsc2)*self.txt_n2(txt) + ts2)
        return img, txt

torch.manual_seed(0)
blk = MMDiTBlock(d=64, nh=4)
img = torch.randn(2, 16, 64)   # 2 batch, 16 image tokens
txt = torch.randn(2,  8, 64)   # 8 text tokens
vec = torch.randn(2, 64)
img2, txt2 = blk(img, txt, vec)
print("delta image norm:", (img2 - img).norm().item())   # ≈ 0 at init (adaLN-Zero!)
print("delta text  norm:", (txt2 - txt).norm().item())   # ≈ 0 at init
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
delta image norm: 0.0
delta text  norm: 0.0
```

中文: 因为 adaLN-Zero 初始化全 0(`StreamModulation.__init__` 里的 `zeros_`),两条 stream 的输出在第一个前向严格等于输入 —— 这就是 SD3 训练能稳的关键。把 `zeros_` 改成默认初始化重新跑,你会看到 delta 立刻变成数值 1+,梯度从此一片混乱。

English: because adaLN-Zero is initialized to zero (`StreamModulation.__init__`'s `zeros_`), both stream outputs are exactly equal to the inputs on the first forward — that's what makes SD3 training stable. Remove the `zeros_` and rerun: deltas jump to >1 immediately and gradients become a mess.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Stable Diffusion 3 原始 MMDiTBlock** / **Original SD3 MMDiTBlock**: 中文: 同款,只是没有 `fused_qkv` 切换。Open-Sora 加的融合路径是为了大模型推理加速。 / English: the same recipe minus the fused-qkv toggle. Open-Sora added fusion for inference speed.
- **Flux 的 DoubleStreamBlock + SingleStreamBlock 混合** / **Flux's DoubleStreamBlock + SingleStreamBlock mix**: 中文: Flux 前 19 层用 double-stream(text+img 各自一份),后 38 层换成 single-stream(text+img concat 成一个流,只一份 modulation)。前者学双向交互,后者节省 KV 存储 + 加速。 / English: Flux runs 19 double-stream layers then 38 single-stream layers (text+img share one stream and one modulation). Double learns bidirectional interaction; single saves KV memory.
- **lingbot-va FlexAttention BlockMask** / **lingbot-va FlexAttention BlockMask**: 中文: 5-29 教过 `action-conditioning` 的另一种解法 —— 单流 + 7 个 predicate 组合出一个 mask 来 *模拟* 双流。MM-DiT 是双流的"硬件"实现,lingbot 是 mask 的"软件"实现。 / English: 2026-05-29 covered the alternative for `action-conditioning` — single stream + 7 mask predicates composed into a BlockMask. MM-DiT is the "hardware" dual-stream, lingbot is the "software" masked-single-stream.
- **NVIDIA GR00T `dit.py` 的 spatial-temporal split** / **NVIDIA GR00T `dit.py` spatial-temporal split**: 中文: 同样的 "把不同模态拼起来跑一次 attention" 的思路,只是 token 类型从 (image, text) 变成 (vision_tokens, language_tokens, action_tokens)。 / English: same "concat heterogeneous modalities into one attention" idea, but the token types become (vision, language, action) instead of (image, text).

## 注意事项 / Caveats / when it breaks

- **位置编码必须能跨越文本边界** / **Positional encoding must cross the text boundary**: 中文: `pe` 同时作用于文字+图像 token,所以你的 RoPE 不能让文字 token 拿到"超大的位置 id"(否则 attention logits 偏向不平衡)。SD3 给文本 token 用全 0 位置 id,Flux 给它一个独立的小 id 区间。 / English: `pe` is applied to *both* text and image tokens after concat. Don't give text tokens huge position IDs — SD3 zeros them out, Flux assigns text a small dedicated range.
- **`txt_q.shape[2]` 是 head 之后的 sequence 维** / **`txt_q.shape[2]` is the post-head sequence dim**: 中文: 因为 `B L (K H D) -> K B H L D`,经过 rearrange 之后维度变成 `(B, H, L, D)`,所以 `txt_q.shape[2] = L_txt`。split 错了会把 head 维当作序列维,完全乱套。 / English: after the `K B H L D` rearrange, dim 2 is the sequence axis. Slicing the wrong dimension after attention scrambles heads and sequence — silent disaster.
- **`Modulation` 必须 zero-init** / **`Modulation` must be zero-initialized**: 中文: 不 zero-init,网络一开始就被两套调制大幅扭曲,gradient 在 LayerNorm 之外的 element-wise 乘上失控。 / English: without zero init, the random modulation distorts both streams heavily on step 0; gradients through the element-wise multiply explode.
- **两个 stream 共享 `num_heads` / `head_dim`** / **Streams share `num_heads` / `head_dim`**: 中文: 因为最后要 concat,Q/K/V 的最后两维必须一致。如果 image 和 text 想用不同 hidden_dim,得在 modulation 前各自 project 一次。 / English: streams must share `num_heads` and `head_dim` because we concat. If image and text need different hidden dims, project them to a common one before modulation.

## 延伸阅读 / Further reading

- SD3 paper: <https://arxiv.org/abs/2403.03206>
- Flux model card: <https://huggingface.co/black-forest-labs/FLUX.1-schnell>
- Past entry on vanilla DiT adaLN-Zero: `2026/05/2026-05-25-dit-adaln-zero-block.md`
- Past entry on lingbot-va FlexAttention (the masked-single-stream alternative): `nano/wam/2026-05-29-lingbot-flex-mask-compose.md`
