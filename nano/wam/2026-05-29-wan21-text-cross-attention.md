---
date: 2026-05-29
topic: wam
source: wam
repo: Wan-Video/Wan2.1
file: wan/modules/model.py
permalink: https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py#L162-L184
difficulty: beginner
read_time: ~7 min
tags: [code-of-the-day, wam, text-conditioning, cross-attention]
build_role: Text/image conditioning — how text/CLIP embeddings inject into the DiT via cross-attention
---

# 文本条件就是 25 行 cross-attention / Text conditioning is 25 lines of cross-attention

> **一句话 / In one line**: Wan2.1 把"如何把文本提示喂进视频 DiT"压缩到这 25 行 —— q 来自 video latent,k/v 来自 T5 编码,经过 flash-attention 就完事,KV-mask 用 `context_lens` 处理变长。 / Wan2.1 expresses "how a text prompt enters a video DiT" in 25 lines — query comes from video latents, key/value from T5 embeddings, flash-attention does the rest, and `context_lens` handles variable-length text.

## 为什么重要 / Why this matters

WAM 要根据 prompt "robot puts the red cube into the bowl" 生成视频(或动作 latent),那么 text encoder 的输出要在 DiT 内部跟视频 token 交互。常规思路有三种:(1) prefix concat —— 把 text token 拼在 video token 前面做 self-attention;(2) cross-attention —— 在 DiT 每层之间插一个 query=video, key=text 的 attention;(3) FiLM / adaLN —— 把 text 压成一个全局向量做 affine 调制。Wan2.1 选了 (2),而且把它做得极其干净:一个 `WanT2VCrossAttention` 类,只覆盖 forward,继承自 self-attention 复用 q/k/v 投影。今天这一段是"text → video"这件事的最小可工作版本,nanoWAM 直接抄过去就能跑。

A WAM that takes "robot puts the red cube into the bowl" and emits a clip needs the text encoder output to interact with video tokens *inside* the DiT. The three common designs are (1) prefix concat into self-attention, (2) cross-attention with query=video, key/value=text, or (3) FiLM/adaLN where text is squashed into a global affine modulator. Wan2.1 chose (2) and the implementation is unbelievably small: one `WanT2VCrossAttention` class that overrides only `forward` and reuses q/k/v projections from the self-attention parent. These 25 lines are the minimum viable text-conditioning module; copy them straight into nanoWAM and you have a working text-to-video conditioning path.

## 代码 / The code

`Wan-Video/Wan2.1` — [`wan/modules/model.py`](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py#L162-L184)

```python
class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]            # video latent tokens
            context(Tensor): Shape [B, L2, C]      # T5/CLIP text tokens
            context_lens(Tensor): Shape [B]        # true text length per sample
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x
```

## 逐行讲解 / What's happening

1. **继承自 `WanSelfAttention` / Subclassing `WanSelfAttention`**:
   - 中文:复用父类构造好的 `self.q / self.k / self.v / self.o / self.norm_q / self.norm_k`(都是 Linear + RMSNorm),所以这里只写 forward。改成 cross-attention 只意味着 q 用 video,k/v 用 text —— 同样的投影矩阵也能干 cross 的活,关键是输入接哪里。
   - English: the parent class already owns `self.q / self.k / self.v / self.o / self.norm_q / self.norm_k` (Linears + RMSNorm). The subclass only overrides `forward`. Cross-attention reuses the *same* projection layers — what changes is just whether q reads from video or text.

2. **q 来自 video / Query from video latents**:
   - 中文:`q = norm_q(self.q(x))`,`x` 是 video latent tokens 形状 `[B, L1, C]`。每个 video token "提问":我应该看 prompt 里的哪些词?
   - English: `q = norm_q(self.q(x))` with `x` shaped `[B, L1, C]` (video latent tokens). Each video token asks "which words in the prompt should I attend to?"

3. **k/v 来自 text / Key and value from text embeddings**:
   - 中文:`k = norm_k(self.k(context))`、`v = self.v(context)`,`context` 形状 `[B, L2, C]` 是 T5/CLIP 的输出。注意 v 上没有 norm —— 这是 Wan2.1 的设计,只 norm q/k 来稳定 attention score,v 保持原始幅度。
   - English: `k = norm_k(self.k(context))`, `v = self.v(context)` with `context` shaped `[B, L2, C]` from a T5/CLIP encoder. Note v is *not* normed — Wan2.1 normalises only q and k to stabilise attention scores; v keeps its original magnitude.

4. **`flash_attention(q, k, v, k_lens=context_lens)` / Variable-length mask**:
   - 中文:`context_lens` 是每个 sample 实际有效的 text token 数(因为 batch 里 prompts 长度不同,要 pad)。flash-attention 用它做 key-side masking,等价于在 softmax 前把 padding 位置加 `-inf`,但 flash 内部跳过这些 block,根本不算。
   - English: `context_lens` is the true text length per sample (prompts vary in length and are padded). flash-attention treats it as key-side masking; the kernel skips padded blocks entirely instead of subtracting `-inf` after the matmul.

5. **`flatten(2) + self.o(x)` / Heads merge + output projection**:
   - 中文:multi-head 输出 `[B, L1, n, d]` 用 `flatten(2)` 合并成 `[B, L1, n*d]`,再过一个 Linear `self.o` 把它投回 hidden_size。这是 transformer 最标准的"merge heads + output linear"。
   - English: merge heads `[B, L1, n, d] → [B, L1, n*d]`, then `self.o` projects back to `hidden_size`. The boilerplate every multi-head attention has.

## 类比 / The analogy

像看电影时旁边有人念字幕。video token 是观众的眼睛("我现在看到画面 X,该往哪边看?"),text token 是字幕("剧本里这一刻应该是机器人抓杯子")。观众根据画面提问,字幕给提示;每个观众只听自己 prompt 里真正写到的词(context_lens 排除空白行),不去听别人的 prompt。

Picture watching a film with someone reading the script aloud beside you. Video tokens are the viewers' eyes ("I'm looking at frame X, what should I focus on?"), text tokens are the narrator ("the script says the robot grabs the cup here"). Each viewer queries based on what they see; the narrator answers only with the words actually in *their* prompt (`context_lens` blocks out the padding) — viewers can't eavesdrop on other people's scripts.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-WAM

中文:在 nanoWAM 里这就是 `nano/wam/blocks/cross_attn_text.py`,挂在每个 DiT block 里 self-attn 之后、FFN 之前。上游是 T5(或 CLIP / Qwen)的 text encoder + 一个 Linear adapter(把 text hidden_dim 投到 DiT 的 hidden_dim);下游是 DiT 的 FFN。如果你完全省掉 cross-attention,只用 self-attention 把 text 拼前缀,**理论可以**,但会让 video 的 self-attention 序列变长(text token 也参与 video self-attention),flash kernel 上明显变慢。换成 cross-attention 后,video self-attn 序列长度恒定,text 长度变化只影响 cross-attn 一段,效率高很多。生产实现要补:**dropout text 做 CFG**(详见明天的 classifier-free-guidance 笔记)、**multi-prompt fusion**(同时编码多个 sub-prompt 再拼)、以及 **adaptive layer-wise scale**(早期 DiT 层强化 text 影响,后期减弱)。

English: in nanoWAM this is `nano/wam/blocks/cross_attn_text.py`, sitting inside each DiT block after self-attention and before the FFN. Upstream is a T5/CLIP/Qwen text encoder plus a Linear adapter (project text hidden dim → DiT hidden dim). Downstream is the DiT FFN. Skipping cross-attention and prefix-concat into self-attention also works *in theory*, but it inflates the video self-attention sequence length on every layer — slow under flash. Cross-attention keeps video self-attn length constant and isolates text variability to one short side. Production essentials: **drop text to enable CFG** (see tomorrow's note), **multi-prompt fusion** for compound prompts, and **layer-wise scaling** that emphasises text in early DiT layers and fades it out later.

## 自己跑一遍 / Try it yourself

```python
# pip install torch
import torch, torch.nn as nn, torch.nn.functional as F

class CrossAttn(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h, self.d = heads, dim // heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
    def forward(self, x, ctx, ctx_lens):
        B, L1, _ = x.shape; _, L2, _ = ctx.shape
        q = self.q(x).view(B, L1, self.h, self.d).transpose(1, 2)
        k = self.k(ctx).view(B, L2, self.h, self.d).transpose(1, 2)
        v = self.v(ctx).view(B, L2, self.h, self.d).transpose(1, 2)
        # build key-side mask from ctx_lens
        kv_mask = torch.arange(L2)[None, :] < ctx_lens[:, None]   # [B, L2]
        kv_mask = kv_mask[:, None, None, :]                        # broadcast over (heads, L1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=kv_mask)
        out = out.transpose(1, 2).flatten(2)
        return self.o(out)

attn = CrossAttn(dim=64, heads=4)
video  = torch.randn(2, 16, 64)        # B=2, 16 video tokens
text   = torch.randn(2, 8,  64)        # B=2, padded to 8 text tokens
ctx_ln = torch.tensor([5, 8])          # sample 0 has 5 real tokens, sample 1 has 8

y = attn(video, text, ctx_ln)
print("out shape:", y.shape)            # [2, 16, 64]
print("sample-0 only attended to 5 text tokens (rest masked).")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
out shape: torch.Size([2, 16, 64])
sample-0 only attended to 5 text tokens (rest masked).
```

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **PixArt-α / Stable Diffusion** / **PixArt-α / Stable Diffusion**: 中文 — 几乎一模一样的 cross-attention 注入,差别仅在 text encoder(SD 用 CLIP,Wan 用 T5,PixArt 用 T5)。 / English — near-identical cross-attention plumbing; only the text encoder differs (SD/CLIP, Wan/T5, PixArt/T5).
- **WanI2VCrossAttention(同文件)** / **WanI2VCrossAttention (same file)**: 中文 — 在 text cross-attn 旁边再挂一份 image cross-attn,实现 image-to-video。结构一模一样,只是 context 来源换成了 image latent。 / English — alongside the text version, the file adds an image-context cross-attention for image-to-video. Same shape, different context source.
- **SmolVLA expert(昨天讲过)** / **SmolVLA expert (yesterday's vla note)**: 中文 — VLA 里"expert cross-attend VLM 的 KV"是同一个范式,只是 q/k/v 来源换成 expert/VLM。 / English — yesterday's VLA note about SmolVLA's expert cross-attending into the VLM uses the exact same construction with different q/k/v sources.

## 注意事项 / Caveats / when it breaks

- **不 norm v 是有意的** / **Skipping v normalisation is intentional**: 中文 — RMSNorm q 和 k 是为了 logits 数值稳定,norm v 会破坏值域,生成质量会掉。别多加。 / English — RMSNorm on q/k stabilises logits; norming v destroys the value range and degrades generation. Do not "fix" this.
- **context 必须 padded** / **`context` must be padded**: 中文 — 同 batch 不同 prompts 长度不同,要 pad 到 max_len 并传 context_lens。忘传 context_lens 就会去 attend padding,等价于学到了"BLANK"这个 token 的含义。 / English — different samples have different prompt lengths; you must pad to `max_len` and pass `context_lens`. Forgetting `context_lens` makes the model attend to padding and silently corrupts conditioning.
- **flash_attention 的 `k_lens` 接口因版本而异** / **`flash_attention`'s `k_lens` API varies**: 中文 — Wan 这里包了一层自家 flash_attention,真正部署时要看版本是支持 cu_seq_k 还是 mask;PyTorch 原生 SDPA 用 attn_mask 接口(见上面 try.py)。 / English — Wan's `flash_attention` wraps the actual kernel. Direct flash-attn API expects `cu_seq_k`, while PyTorch SDPA accepts an `attn_mask`. Pick the right entry point per backend.

## 延伸阅读 / Further reading

- [Wan2.1 model.py — full block including this cross-attn](https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py)
- [PixArt-α: cross-attention text injection](https://arxiv.org/abs/2310.00426)
- [Flash Attention 2 paper](https://arxiv.org/abs/2307.08691)
- [Attention Is All You Need — the original cross-attention spec](https://arxiv.org/abs/1706.03762)
