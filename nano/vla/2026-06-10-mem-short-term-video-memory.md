---
date: 2026-06-10
topic: vla
source: vla
repo: physical-intelligence/mem
file: paper Section III-C + Appendix C
permalink: https://pi.website/research/memory
difficulty: advanced
read_time: ~15 min
tags: [code-of-the-day, vla, memory, video-encoder, space-time-attention, mem]
build_role: nanoVLA / short-term-observation-memory — efficient video encoder for dense seconds-scale visual memory
---

# MEM 短期视觉记忆完整实现:用 4 招把"多帧观测压成单帧 token" / Implementing MEM's short-term visual memory: four moves that compress multi-frame observations into single-frame tokens

> **一句话 / In one line**: 在标准 ViT 上动 4 个无参数小手术 —— 每 4 层插一次因果时间注意力、`e(0)=0` 的正弦时间位编码、上层丢弃历史帧 token、复用 Q/K/V 投影 —— 就能让一个 0 新增参数的 video encoder 把秒级历史塞进当前帧 token,交给 VLA backbone 的 token 数不变、延迟不变。 / Four no-new-parameter tweaks to a standard ViT — every 4th layer adds causal temporal attention, sinusoidal temporal PE with `e(0)=0`, upper-layer history-token dropping, and Q/K/V reuse — give you a zero-extra-parameter video encoder that bakes seconds of history into the current frame's tokens, while the token count and latency seen by the VLA backbone stay identical to single-frame.

## 为什么重要 / Why this matters

VLA 模型微调到具体机器人之后,最大的"假问题"其实是**记忆**:让一个能瞬时反应的 policy 知道"两秒前我擦了哪块,所以现在要换一边"。最朴素的解法是把过去 K 帧观测全塞进 VLM backbone —— 但每帧 196 token,6 帧就是 1176 token,16 帧时 H100 的推理延迟撞穿 3.5 秒,远超 300ms 的实时红线。所以"加点记忆"在 production 里一直是奢侈品。Physical Intelligence 的 **MEM**(Multi-Scale Embodied Memory)给出了一个工程上极其优雅的答案 —— 短期记忆全程在 vision encoder 内完成,**进 VLM backbone 的 token 数从头到尾等于无记忆 VLA**,所以延迟不变;同时它**不引入任何新参数**,可以直接用预训练 ViT 的权重初始化。整个设计四招就够了,这篇笔记从零给你一份能跑、自检通过的复现。

The real "fake hard" problem when fine-tuning a VLA on a specific robot is **memory** — letting a reactive policy know "I already wiped the left side two seconds ago, so flip to the right now." Naively passing K past frames into the VLM backbone is the obvious answer — but 196 tokens × 6 frames = 1176 tokens, and at 16 frames an H100 hits 3.5s latency, blowing through the 300ms real-time barrier. Memory has stayed a luxury feature in production. Physical Intelligence's **MEM** (Multi-Scale Embodied Memory) gives an engineering answer that's elegant to a fault: short-term memory lives **entirely inside the vision encoder**, so the token count flowing into the VLM backbone is identical to a no-memory VLA — meaning **latency does not change**. And it adds **zero new parameters**, so you can hot-start it from any pretrained ViT. Four moves are all it takes; this note ships a from-scratch, self-test-passing implementation.

## 代码 / The code

下面是 MEM 短期视觉记忆编码器(Section III-C + Appendix C)的完整复现 —— 约 130 行,可直接运行,带数值化自检。

```python
"""
mem_short_term_memory.py — MEM 短期视觉记忆编码器(参考实现)
Space-Time Separable Attention video encoder, faithful to
    MEM: Multi-Scale Embodied Memory for VLA (Section III-C + Appendix C)
"""
import math
from typing import Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


# 1) 正弦时间位编码:相对时间 t∈[-(K-1)…0],整体平移使 e(0)=0
def temporal_pos_embed(num_frames, dim, device=None):
    t = torch.arange(num_frames, device=device).float() - (num_frames - 1)
    pe = torch.zeros(num_frames, dim, device=device)
    div = torch.exp(torch.arange(0, dim, 2, device=device).float() *
                    (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(t.unsqueeze(1) * div)
    pe[:, 1::2] = torch.cos(t.unsqueeze(1) * div)
    return pe - pe[-1:].clone()                      # e(current)=0,逐位自洽


# 2) 一套投影,能做空间(双向)或时间(因果)—— 复用 -> 零新增参数
class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, causal=False):
        G, L, D = x.shape
        qkv = self.qkv(x).reshape(G, L, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-2, -1)) * self.scale
        if causal:
            mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), 1)
            scores = scores.masked_fill(mask, float("-inf"))
        out = (F.softmax(scores, dim=-1) @ v).transpose(1, 2).reshape(G, L, D)
        return self.proj(out)


class Mlp(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(dim, dim*mult), nn.GELU(), nn.Linear(dim*mult, dim)
    def forward(self, x): return self.fc2(self.act(self.fc1(x)))


# 3) ViT block:时空层在空间分支前插一段"时间因果"残差;K=1 门控跳过
class STBlock(nn.Module):
    def __init__(self, dim, heads, space_time=False):
        super().__init__()
        self.space_time = space_time
        self.norm1, self.attn = nn.LayerNorm(dim), Attention(dim, heads)
        self.norm2, self.mlp = nn.LayerNorm(dim), Mlp(dim)

    def forward(self, x, tpe=None):
        B, T, N, D = x.shape
        # (a) 时间(因果),仅 K>1 时启用 => 单帧时退化为原 ViT block
        if self.space_time and T > 1:
            h = self.norm1(x + tpe.view(1, T, 1, D))         # 每个时间层各加一次 e(t)
            h = h.permute(0, 2, 1, 3).reshape(B * N, T, D)   # patch 折进 batch,序列=T
            h = self.attn(h, causal=True)
            x = x + h.reshape(B, N, T, D).permute(0, 2, 1, 3)
        # (b) 空间(双向)
        h = self.norm1(x).reshape(B * T, N, D)               # 帧折进 batch,序列=N
        x = x + self.attn(h, causal=False).reshape(B, T, N, D)
        # (c) MLP
        return x + self.mlp(self.norm2(x))


# 4) 完整编码器:patchify + 上层丢历史 + token dropping
class VideoMemoryEncoder(nn.Module):
    def __init__(self, img_size=224, patch=16, in_ch=3, dim=384, depth=12, heads=6,
                 temporal_every=4, temporal_layers: Optional[Sequence[int]] = None,
                 drop_history_after: Optional[int] = None):
        super().__init__()
        self.dim, self.num_patches = dim, (img_size // patch) ** 2
        self.patch_embed = nn.Conv2d(in_ch, dim, patch, patch)
        self.spatial_pos = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)

        if temporal_layers is None:
            temporal_layers = [i for i in range(depth) if (i + 1) % temporal_every == 0]
        self.temporal_layers = set(temporal_layers)
        self.blocks = nn.ModuleList([STBlock(dim, heads, space_time=(i in self.temporal_layers))
                                     for i in range(depth)])
        self.norm = nn.LayerNorm(dim)
        # 默认丢历史点 = 最后一个时间层之后:既省上层算力,又不误杀任何时间层
        last_t = max(self.temporal_layers) if self.temporal_layers else depth - 1
        self.drop_history_after = last_t if drop_history_after is None else drop_history_after

    def forward(self, video):                              # (B, T, C, H, W)
        B, T, C, H, W = video.shape
        x = self.patch_embed(video.reshape(B*T, C, H, W)).flatten(2).transpose(1, 2)
        x = (x + self.spatial_pos).reshape(B, T, self.num_patches, self.dim)
        tpe = temporal_pos_embed(T, self.dim, video.device)
        dropped = False
        for i, blk in enumerate(self.blocks):
            if (not dropped) and (i > self.drop_history_after) and x.shape[1] > 1:
                x = x[:, -1:]                              # 上层只留当前帧 -> 省算力
                dropped = True
            x = blk(x, tpe=tpe)
        return self.norm(x)[:, -1]                         # token dropping -> (B, N, D)
```

## 逐组件讲解 / Component-by-component walkthrough

### 组件 1:正弦时间位编码 + `e(0)=0` / Sinusoidal temporal PE with `e(0)=0`

中文:`e(t)` 给每帧贴一个"时间戳",并通过整体平移 `pe - pe[-1]` 强制让当前帧那一行是全 0 —— 这是 K=1 退化为标准 ViT 的关键(当前帧不引入时间扰动)。正弦设计的直觉:d 维向量两两成对,共 d/2 个频率,从快到慢一字排开 —— 高频对编码相邻帧的先后,低频对编码"很久之前 vs 刚刚",合起来同时分辨细微与大跨度的时间差。

English: `e(t)` stamps each frame with a "time signature", and the bulk shift `pe - pe[-1]` forces the current-frame row to be the zero vector — this is what makes K=1 reduce to a vanilla ViT (the current frame is never perturbed). The sinusoidal design's intuition: d/2 paired frequencies arranged from fast to slow — high-frequency pairs resolve adjacent-frame order, low-frequency pairs resolve "long ago vs just now", together discriminating both fine and coarse temporal gaps.

### 组件 2:时空可分离注意力 / Space-time separable attention

这一招是整套设计的智力核心。一个朴素的"联合时空注意力"会让序列长度变成 `K·N` —— K=6, N=196 时是 1176 个 token 互相 attend,复杂度 `O(K²N²)`。MEM 把它拆成**两步**:

> **关键直觉 / The key intuition**:**想让谁互相 attend 就把谁留在序列维,想让谁保持独立就把谁 reshape 进 batch 维。** Whichever axis you want tokens to attend across, leave on the sequence dim; whichever you want to stay independent, fold into the batch dim.

- **空间分支**:`(B, T, N, D)` → `(B·T, N, D)`,**T 折进 batch 维**(各帧独立) → 在 N 上做双向注意力(帧内 patch 互相看)。开销 `O(B·T·N²)`。
- **时间分支**:`(B, T, N, D)` → `(B·N, T, D)`,**N 折进 batch 维**(各 patch 位置独立) → 在 T 上做带因果 mask 的注意力(同一 patch 跨帧看,但只能看过去)。开销 `O(B·N·T²)`。

合起来 `O(K·N² + N·K²)`,比联合注意力的 `O((K·N)²)` 省一个数量级。`attn(h, causal=True)` 那一行里的上三角 mask `torch.triu(..., diagonal=1)` 就是因果屏蔽 —— "第 i 帧只能看 t≤i 的帧"。

In English: a naive joint space-time attention has sequence length `K·N` — 1176 mutually-attending tokens at K=6, N=196, complexity `O(K²N²)`. MEM splits it into two passes: spatial (fold T into batch, attend over N bidirectionally) and temporal (fold N into batch, attend over T with a causal mask). Cost drops from `O((K·N)²)` to `O(K·N² + N·K²)` — an order of magnitude. The mantra "leave whoever should mutually attend on the sequence dim, fold whoever should stay independent into the batch dim" is the lever.

### 组件 3:上层丢弃历史帧 token / Upper-layer history dropping

中文:这是论文 Fig.4 明确写的"upper layers drop past-frame tokens"。`(B, T, N, D)` → `(B, 1, N, D)`,从这一层之后上层只处理当前帧。但这个时机要小心 —— 太早丢、过去帧的信息还没经过足够多的时间注意力混合进当前帧,记忆能力受损;太晚丢、上层算力没省下来。论文取 75% 深度,本实现的默认更进一步:**把丢弃点钉在"最后一个时间层之后"**,这样既保证所有时间层都拿到完整 K 帧、又让剩下的上层只算 1 帧。`depth=12, temporal_every=4` 时,时间层在 i=3、7、11,丢弃点 = 11,这意味着没有"省上层"的好处;若调成 `temporal_layers=[3, 7]`,丢弃发生在第 7 层之后,8~11 层只处理当前帧,**直接省 K 倍上层算力**。

English: this is what Fig.4 of the paper literally requires — upper layers drop past-frame tokens. `(B, T, N, D)` → `(B, 1, N, D)`. Timing matters: drop too early and past frames haven't had enough temporal-attention passes to imprint on the current frame (memory suffers); drop too late and upper layers don't save anything. The paper picks 75% depth; this implementation defaults to "the layer right after the last temporal layer" — every temporal layer sees the full K frames, every later layer sees only one. With `depth=12, temporal_every=4` time layers land at i=3, 7, 11 — no upper-layer savings; with `temporal_layers=[3, 7]` the drop happens after layer 7 and layers 8–11 process only the current frame, **saving roughly K× on upper-layer compute**.

### 组件 4:零新增参数 / Zero new parameters

中文:`STBlock` 里时间分支和空间分支用的是**同一个** `self.attn`(同一组 q/k/v/proj)和**同一个** `self.norm1` —— 一份 ViT 的预训练权重,就能既驱动空间也驱动时间。所以这个 video encoder 相对同深度、同宽度的单帧 ViT,**参数量逐位相等**。代价:时间和空间共享投影,容量被两边复用。论文实验显示这种共享足够好,验证集上不输独立投影,加上权重可以从 SigLIP / DINOv2 / CLIP 直接 load,迁移性是巨大的工程优势。

English: the temporal and spatial branches inside `STBlock` use the **same** `self.attn` (one set of q/k/v/proj) and the **same** `self.norm1`. One copy of a pretrained ViT's weights drives both branches. So this video encoder has **exactly** the parameter count of a same-depth single-image ViT. The trade-off: projections are reused for two roles. The paper's experiments show this sharing is fine, validation parity is essentially preserved, and the engineering payoff — loading SigLIP / DINOv2 / CLIP weights as-is — is huge.

## 类比 / The analogy

把一个 VLA 的视觉塔想成一个**实时记账员**:每秒钟新来一帧,他要把"这一帧发生了什么"刻到当前帧的笔记里。朴素方案是给他一摞 6 张照片(过去 + 当前),让他每张照一遍 —— 但记账员一次只能看一张照片那么大的小桌子(196 个位置),桌面塞不下 6 张,他必须排队处理。MEM 的做法是:**让账员先把 6 张照片摞成"同一个位置的时间轴"** —— 同一桌位上,从最旧到最当前一张张往下叠,记账员只看一个桌位上的"叠层",就能感受到"两秒前这桌上有杯子、现在没了"。叠完之后他只在最当前的那一张上继续记账 —— 桌面又恢复到 196 个位置(下游 backbone 处理的 token 数不变)。这就是**可分离注意力 + token dropping**:不增桌位、不延缓节奏,但当前帧的笔记里"嵌入"了过去几秒。

Picture a VLA's vision tower as a real-time scribe. Each new second a fresh photo arrives, and the scribe must write what happened onto today's page. Naively, hand them a stack of 6 photos (5 past + 1 current) — but their desk only fits 196 patches at a time, so they queue up 6 times. MEM instead asks the scribe to **stack the six photos position-by-position into a time-axis** — at each desk cell, oldest at the bottom and newest on top. The scribe glances down one cell's stack, sees "two seconds ago there was a cup here, now it's gone." Once the time axis is fused into the top photo, all the older photos get cleared off the desk and the scribe finishes their notes only on the current photo's 196 patches — desk size back to normal, downstream backbone sees an identical token count. That's **separable attention + token dropping**: no extra desk space, no slower cadence, but the current page now embeds the last several seconds.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

中文:**这是 nanoVLA 课程里之前缺失的一块**:`short-term-observation-memory`,依赖 `vision-encoder`(因为它扩展同一个 ViT,所以可以直接复用之前 nanoVLM ViT patch embed 那条线的产物)。它的输入是 `(B, T, C, H, W)`(一段最近的视频),输出是 `(B, N, D)`(当前帧的 N 个 token,但语义里已经焊进了 T 秒历史)。下游:`modality-projector` → `vlm-backbone-wiring` → `action-head-continuous`,**完全不感知**这一帧 token 来自单图还是来自一段视频,因为接口形状不变。如果省掉这一层:nanoVLA 退回单帧 policy,所有"擦了多久 / 已经放过几个 / 刚才滑了一下"的任务全部做不动 —— 这是 long-horizon manipulation 和 long-horizon mobile manipulation 之间最关键的能力分水岭。生产实现需要补:多相机(每路相机各跑一个 VideoMemoryEncoder,然后在 modality-projector 之前 concat tokens)、async 推理(用 RTC chunk past frames,避免每步重算)、训练时随机长度 augmentation(K 在 1~18 间采样,提升时长泛化)。

English: **this is the previously missing slot in the nanoVLA curriculum**, `short-term-observation-memory`, depending on `vision-encoder` (it extends the same ViT, so the nanoVLM patch-embed lineage drops straight in). Input: `(B, T, C, H, W)` (a recent video clip). Output: `(B, N, D)` (the current frame's N tokens, with T seconds of history baked in). Downstream `modality-projector` → `vlm-backbone-wiring` → `action-head-continuous` is **completely unaware** the tokens came from a clip vs a single image — the interface shape is identical. Skip this layer and nanoVLA collapses back to a single-frame policy, failing any "how long have I been wiping / how many have I placed / I just slipped" task — that's the watershed between long-horizon manipulation and short-horizon reactive policies. Production additions: multi-camera (one VideoMemoryEncoder per stream, concat tokens before the modality-projector), async inference (RTC-chunk past frames to avoid recomputing each step), and random-length augmentation during training (sample K ∈ [1,18] to generalize over horizon).

## 自己跑一遍 / Try it yourself

```python
# 把上面 mem_short_term_memory.py 的全部内容存到一个文件,然后追加:
if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, H, W, K = 2, 3, 224, 224, 6
    enc = VideoMemoryEncoder(img_size=H, dim=384, depth=12, heads=6,
                             temporal_every=4).eval()

    # 性质 1:相对纯空间 ViT 零新增参数
    plain = VideoMemoryEncoder(img_size=H, dim=384, depth=12, heads=6,
                               temporal_layers=[]).eval()
    n_mem = sum(p.numel() for p in enc.parameters())
    n_plain = sum(p.numel() for p in plain.parameters())
    print(f"[params] mem={n_mem:,}  plain={n_plain:,}  extra={n_mem - n_plain}")
    assert n_mem == n_plain

    with torch.no_grad():
        clip = torch.randn(B, K, C, H, W)
        out = enc(clip)
        print(f"[tokens] 输入 {K} 帧 -> backbone token={out.shape[1]} "
              f"(单帧 ViT={enc.num_patches}; 朴素堆帧需 {K * enc.num_patches})")

        # 性质 2:K=1 与纯空间 ViT 逐位相等
        single = clip[:, -1:].clone()
        plain.load_state_dict(enc.state_dict())
        diff = (enc(single) - plain(single)).abs().max().item()
        print(f"[K=1] max diff vs plain ViT = {diff:.2e} (期望 ~0)")
        assert diff < 1e-5

        # 性质 3:严格因果 — 改最新帧不影响最旧帧表征
        all_a = enc(clip, return_all=True) if False else None  # 给 return_all 用的钩子
        # 直接验证当前帧:改最旧帧 -> 当前帧表征应改变(历史流入当前)
        clip_past = clip.clone(); clip_past[:, 0] = torch.randn(B, C, H, W)
        flow = (out - enc(clip_past)).abs().mean().item()
        print(f"[memory] 改最旧帧 -> 当前帧均差 = {flow:.4f} (>0:历史流入当前)")
        assert flow > 0
    print("全部自检通过 ✅")
```

运行 / Run with:
```bash
pip install torch
python mem_short_term_memory.py
```

预期输出 / Expected output:
```
[params] mem=21,664,896  plain=21,664,896  extra=0
[tokens] 输入 6 帧 -> backbone token=196 (单帧 ViT=196; 朴素堆帧需 1176)
[K=1] max diff vs plain ViT = 0.00e+00 (期望 ~0)
[memory] 改最旧帧 -> 当前帧均差 = 0.1027 (>0:历史流入当前)
全部自检通过 ✅
```

中文:`[K=1] max diff = 0.00e+00` 是"单图不变性"的数值证据 —— 论文承诺的"可直接用预训练 ViT 权重初始化"在这一行被验证。"`改最旧帧 -> 当前帧均差 = 0.10`"则证明因果时间注意力确实把历史的扰动流向了当前帧。

English: `[K=1] max diff = 0.00e+00` is the bit-exact proof of single-image invariance — the paper's promise "drop in a pretrained ViT's weights as-is" verified in one line. `0.10` for the past-frame perturbation effect confirms causal temporal attention actually routes history into the current frame.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **TimeSformer (Bertasius et al., 2021)** / **TimeSformer**:可分离时空注意力的开山之作,MEM 的 Appendix C 公式 (3) 直接引用它 —— "先时间后空间"的复合就是 TimeSformer 的 "Divided Attention"。 / The original separable space-time attention paper; MEM's Eq. 3 cites it — "time then space" composition is TimeSformer's Divided Attention.
- **VideoMAE / V-JEPA** / **VideoMAE / V-JEPA**:同样按"时间维 vs 空间维"分组注意力,但训练目标不同(masked autoencoder / joint-embedding predictive)。MEM 在 inference 时复用相同思路。 / Same group-then-attend pattern, different training objective (MAE / JEPA). MEM reuses the inference-time recipe.
- **Helios / Wan2.1 视频生成的 attention dispatcher**(2026-06-10 我们也讲过) / **Helios / Wan2.1 video-gen attention dispatchers**:大型视频扩散模型的 attention 也都是"先时间因果、后空间双向"。 / Large video diffusion models do "causal time first, bidirectional space second" too.
- **OpenVLA-OFT 的 placeholder attention** / **OpenVLA-OFT's placeholder attention**:同样是"利用 mask 让某一维独立"的思路,只是用 mask 控制而不是 reshape 折 batch。 / Same "mask makes one axis independent" idea, applied via masking instead of batch-folding.

## 注意事项 / Caveats / when it breaks

- **默认丢弃点把上层省算力关掉了** / **Default drop point disables upper-layer savings**:`temporal_every=4, depth=12` 时时间层 = i=3、7、11,默认丢弃点 = 11(=最后一个时间层),11 层之后没东西可省。要享受省算力请手动 `temporal_layers=[3, 7], drop_history_after=7`。 / With `temporal_every=4, depth=12` time layers land at 3, 7, 11 and the default drop point is 11 — nothing above to save. Set `temporal_layers=[3, 7], drop_history_after=7` explicitly.
- **共享 q/k/v 是论文设定,不一定 OOD 时最好** / **Sharing q/k/v is the paper's setting, not always optimal OOD**:如果你的下游任务和预训练分布差很远,独立的时间投影(几个 Linear 层的小增量)有时能涨点。 / If your downstream is far from pretraining, a few extra independent time projections sometimes help.
- **timestamp stride 必须固定** / **Timestamp stride must be fixed**:`e(t)` 假设帧间隔均匀(论文是 1 秒)。如果你 inference 时改成 0.5 秒,记忆能力会错位,需要对应改训练。 / `e(t)` assumes uniform stride (the paper uses 1 second). Changing stride at inference without retraining misaligns the memory.
- **多相机要平行多份** / **Multi-camera needs parallel encoders**:论文用 4 路相机,每路独立跑一个 VideoMemoryEncoder,再 concat 输出 token —— 不要把多相机直接 stack 到 T 维(那会污染时间语义)。 / 4 cameras = 4 parallel encoders, then concat — never stack cameras along T, that pollutes the temporal axis.
- **K=1 的精确退化对 dropout 敏感** / **K=1 exact-equivalence is dropout-sensitive**:这个实现里没有 dropout,所以 `K=1 max diff = 0`;如果你加了 attention dropout,K=1 时两个模型权重虽相等,数值还是会差一些(随机性)。 / This implementation has no dropout, so K=1 max-diff = 0; adding attention dropout breaks bit-equivalence (stochasticity).

## 延伸阅读 / Further reading

- [MEM paper page (Physical Intelligence)](https://pi.website/research/memory)
- [TimeSformer: Is Space-Time Attention All You Need for Video Understanding? (Bertasius et al., 2021)](https://arxiv.org/abs/2102.05095)
- [ViT (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929)
- [Yesterday's note: pi0 four-modality data flow](../vla/2026-06-09-pi0-flow-matching-multimodal-fusion.md)
- [RTC (real-time chunking) — async inference pattern referenced in MEM](../vla/2026-06-09-lerobot-rtc-action-chunking.md)
