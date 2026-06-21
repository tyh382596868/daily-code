---
date: 2026-06-21
topic: wam
source: wam
repo: yuantianyuan01/FastWAM
file: src/fastwam/models/wan22/mot.py
permalink: https://github.com/yuantianyuan01/FastWAM/blob/45d8e1458921d83f8ad6cf9ce993d371208dabd0/src/fastwam/models/wan22/mot.py#L447-L538
difficulty: advanced
read_time: ~14 min
tags: [code-of-the-day, wam, action-conditioning, mixture-of-transformers, joint-attention, dit]
build_role: action-conditioning (Mixture-of-Transformers variant — action expert and video expert pool Q/K/V for joint attention at every layer)
---

# Mixture of Transformers：动作 expert 和视频 expert 在每一层共享注意力池 / Mixture of Transformers: Action Expert and Video Expert Share One Attention Pool at Every Layer

> **一句话 / In one line**: FastWAM 的 MoT 不是简单地把动作 token 拼到视频 token 上，而是让每个 expert（视频 DiT、动作 DiT）各自计算 Q/K/V，然后把所有 expert 的 Q、K、V 拼起来做一次联合注意力，再把结果切分回各个 expert——动作 token 在每一层都能看到全部视频 token，视频也能看到动作。 / FastWAM's MoT doesn't simply concatenate action tokens onto video tokens; instead each expert (video DiT, action DiT) computes its own Q/K/V, then all experts' Q, K, V are concatenated for one joint attention, and the output is split back to each expert — action tokens see all video tokens at every layer, and vice versa.

## 为什么重要 / Why this matters

在 WAM 课程里，我们已经见过两种动作条件化方式：
- **Cross-attention**：动作 token 作为 query，视频 token 作为 key/value——动作能看到视频，但视频看不到动作。
- **Token concatenation**（token 拼接）：把动作 token 直接拼到视频 token 序列里做自注意力。简单，但动作 token 和视频 token 共享完全相同的权重，没有专属表示能力。

MoT（Mixture of Transformers）是第三种方案：每个 expert 保留自己的一套权重（独立的 Q/K/V 投影和 MLP），但在注意力步骤上所有 expert 共同参与。结果是：
1. 动作 token 和视频 token **互相看到对方**（双向），比 cross-attention 更强。
2. 每个 expert **有自己的独立权重**（独立语义空间），比 token concat 更强。
3. 只做一次注意力计算，不额外增加 forward 步数。

In the WAM curriculum, we've seen two action-conditioning strategies:
- **Cross-attention**: action tokens as queries, video tokens as keys/values — actions see video, but video doesn't see actions.
- **Token concatenation**: directly append action tokens to the video token sequence for self-attention. Simple, but action and video tokens share identical weights, lacking specialized representations.

MoT (Mixture of Transformers) is a third approach: each expert keeps its own weight set (independent Q/K/V projections and MLP), but all experts participate jointly in the attention step. The result:
1. Action tokens and video tokens **see each other bidirectionally** — stronger than cross-attention.
2. Each expert **has independent weights** (separate semantic space) — stronger than token concat.
3. Only one attention computation — no extra forward passes.

## 代码 / The code

`yuantianyuan01/FastWAM` — [`src/fastwam/models/wan22/mot.py`](https://github.com/yuantianyuan01/FastWAM/blob/45d8e1458921d83f8ad6cf9ce993d371208dabd0/src/fastwam/models/wan22/mot.py#L447-L538)

```python
def forward(
    self,
    embeds_all: Dict[str, torch.Tensor],
    attention_mask: torch.Tensor,
    freqs_all: Dict[str, torch.Tensor],
    context_all: Dict[str, Optional[dict]],
    t_mod_all: Dict[str, torch.Tensor],
):
    tokens_all = {k: v for k, v in embeds_all.items()}

    for layer_idx in range(self.num_layers):
        q_chunks = []
        k_chunks = []
        v_chunks = []
        cached = {}
        seq_lens = []

        for name in self.expert_order:
            expert = self.mixtures[name]
            block = expert.blocks[layer_idx]
            x = tokens_all[name]
            freqs = freqs_all[name]
            t_mod = t_mod_all[name]

            (
                q, k, v,
                residual_x,
                gate_msa, shift_mlp, scale_mlp, gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert, block=block, x=x, freqs=freqs, t_mod=t_mod,
            )

            q_chunks.append(q)
            k_chunks.append(k)
            v_chunks.append(v)
            seq_lens.append(x.shape[1])
            cached[name] = {
                "block": block,
                "residual_x": residual_x,
                "gate_msa": gate_msa,
                "shift_mlp": shift_mlp,
                "scale_mlp": scale_mlp,
                "gate_mlp": gate_mlp,
                "use_gradient_checkpointing": use_gradient_checkpointing,
            }

        # concat all tokens for mixed attention
        q_cat = torch.cat(q_chunks, dim=1)
        k_cat = torch.cat(k_chunks, dim=1)
        v_cat = torch.cat(v_chunks, dim=1)

        mixed = self._mixed_attention(
            q_cat=q_cat, k_cat=k_cat, v_cat=v_cat,
            attention_mask=attention_mask
        )

        start = 0
        for name, seq_len in zip(self.expert_order, seq_lens):
            end = start + seq_len
            mixed_slice = mixed[:, start:end, :]
            cached_expert = cached[name]
            block = cached_expert["block"]
            tokens_all[name] = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=cached_expert["residual_x"],
                mixed_slice=mixed_slice,
                gate_msa=cached_expert["gate_msa"],
                shift_mlp=cached_expert["shift_mlp"],
                scale_mlp=cached_expert["scale_mlp"],
                gate_mlp=cached_expert["gate_mlp"],
                use_gradient_checkpointing=cached_expert["use_gradient_checkpointing"],
            )
            start = end

    return tokens_all
```

## 逐行讲解 / What's happening

1. **`tokens_all = {k: v for k, v in embeds_all.items()}` — 可变状态字典**:
   - 中文: 以 dict 形式持有每个 expert 的 token 状态（通常 key 为 `"video"` 和 `"action"`），在层循环中逐层更新。这是 MoT forward pass 的主状态。
   - English: Holds each expert's token state (typically keys `"video"` and `"action"`) as a dict, updated layer by layer in the loop. This is the main state of the MoT forward pass.

2. **内循环：每个 expert 独立计算 Q/K/V**:
   - 中文: `_build_expert_attention_io` 用该 expert 自己的权重（独立的 QKV 投影矩阵）把 `x` 变换成 Q、K、V，同时计算 DiT 的 adaLN 调制参数（gate_msa、shift_mlp、scale_mlp、gate_mlp）和残差连接的 residual_x。关键点：这一步**不执行**注意力，只准备 Q/K/V。
   - English: `_build_expert_attention_io` transforms `x` into Q, K, V using **this expert's own weights** (independent QKV projection matrices), while computing DiT adaLN modulation params (gate_msa, shift_mlp, scale_mlp, gate_mlp) and the residual_x for the skip connection. Critical: this step does **not** run attention — it only prepares Q/K/V.

3. **`q_cat = torch.cat(q_chunks, dim=1)` — 联合注意力池**:
   - 中文: 把所有 expert 的 Q 沿序列维度（dim=1）拼接：若视频有 T·H·W 个 token，动作有 A 个 token，则 q_cat 的序列长度是 T·H·W + A。K 和 V 同理。这是 MoT 的核心操作——所有 expert 共同进入同一次 attention。
   - English: Concatenate all experts' Q along the sequence dimension (dim=1): if video has T·H·W tokens and action has A tokens, q_cat has sequence length T·H·W + A. Same for K and V. This is the core MoT operation — all experts enter the same joint attention together.

4. **`mixed = self._mixed_attention(q_cat, k_cat, v_cat, attention_mask)`**:
   - 中文: 一次标准的多头注意力（flash attention / sdpa），输入是拼好的全序列 Q/K/V，输出形状与 q_cat 相同（即 [B, T_video+T_action, D]）。`attention_mask` 控制哪些 token 对可以互相看到（如因果遮罩或自定义结构遮罩）。
   - English: One standard multi-head attention (flash attention / sdpa), taking the full concatenated Q/K/V as input, output shape identical to q_cat (i.e. [B, T_video+T_action, D]). `attention_mask` controls which token pairs can attend to each other (e.g. causal mask or custom structure mask).

5. **切分 + `_apply_post_with_optional_checkpoint` — 每个 expert 独立后处理**:
   - 中文: 用记录的 `seq_lens` 把联合注意力输出切回各 expert 的切片。每个 expert 再用自己的 gate_msa（注意力门控）、MLP（独立权重）完成这一层的残差更新。这是 "混合注意力，独立 MLP" 的结构。
   - English: Use the recorded `seq_lens` to slice the joint attention output back into each expert's portion. Each expert then applies its own gate_msa (attention gate), MLP (independent weights), and skip connection to complete the layer's residual update. This is the "mixed attention, independent MLP" structure.

## 类比 / The analogy

想象一次会议圆桌讨论。视频 expert 代表视觉团队（T·H·W 成员），动作 expert 代表动作规划团队（A 成员）。在每个讨论层次（transformer 层），所有人**同时提问、倾听、回答**（联合 Q/K/V 注意力）——视觉团队和动作团队都能听到对方说的一切。讨论结束后，每个人回到自己的小组，用自己的笔记本（独立 MLP 权重）整理出本组的下一步行动（MLP 更新）。整个会议重复多轮（多 transformer 层），直到双方都收敛到一个共同理解的状态。

Imagine a round-table meeting. The video expert represents the visual team (T·H·W members), the action expert represents the action planning team (A members). At each discussion round (transformer layer), everyone **asks, listens, and answers simultaneously** (joint Q/K/V attention) — the visual and action teams both hear everything the other says. After each discussion, everyone returns to their own subgroup and uses their own notes (independent MLP weights) to formulate the next steps (MLP update). The full meeting repeats for many rounds (multiple transformer layers) until both teams converge on a shared understanding.

## 在 nanoWAM 中的位置 / Where this lives in your nanoWAM

这个 MoT 是 nanoWAM 中 `action-conditioning` 课程槽位的第三种实现变体，与之前的 cross-attention 和 token concatenation 并列。

中文：在你自己从头搭的 nanoWAM 里，`action-conditioning` 组件紧跟在 VAE 编码（上游：把视频帧压缩成 latent token）和 timestep embedding（上游：扩散时间步编码）之后，输出送入后续 transformer 层直至 VAE 解码（下游：还原成视频像素）。MoT 的输入是 `{video: [B, T*H*W, D], action: [B, A, D]}`，输出是经过全层更新的同格式字典。如果省掉 action expert（退化成纯视频 DiT），模型就无法进行动作条件化视频生成，变成无条件或文本条件的纯世界模型。

实现 nanoWAM 版 MoT 的最小化路径：保留一个完整的视频 DiT backbone，额外添加一个只有 2-4 层的 action expert DiT（参数量远小于视频 DiT），在每层用 `torch.cat` 拼接两个 expert 的 Q/K/V 做联合 FlashAttention，再切分回来各自做 MLP。微调时只更新 action expert 权重，视频 DiT 保持冻结，计算效率极高。生产级实现还需要：结构化 attention mask（限制视频 token 之间的时序因果）、RoPE 频率对两个 expert 的坐标空间分别设计，以及梯度检查点（gradient checkpointing）管理显存。

English: In your from-scratch nanoWAM, the `action-conditioning` component sits after VAE encoding (upstream: compress video frames into latent tokens) and timestep embedding (upstream: encode diffusion timestep), with output flowing into subsequent transformer layers until VAE decoding (downstream: reconstruct video pixels). MoT takes `{video: [B, T*H*W, D], action: [B, A, D]}` as input and produces the same-format dict updated through all layers. If you drop the action expert (degenerating to a pure video DiT), the model loses action-conditioned generation, becoming an unconditional or text-only world model.

The minimal nanoWAM MoT implementation: keep a full video DiT backbone, add a small action-expert DiT with only 2-4 layers (far fewer parameters than the video DiT), concatenate each layer's Q/K/V from both experts for joint FlashAttention, then split and apply independent MLPs. Fine-tuning only updates action expert weights, keeping the video DiT frozen — extremely compute-efficient. A production implementation additionally needs: structured attention masks (enforcing temporal causality among video tokens), RoPE frequency schedules designed separately for each expert's coordinate space, and gradient checkpointing for memory management.

## 自己跑一遍 / Try it yourself

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyMoT(nn.Module):
    """Minimal Mixture-of-Transformers: 2 experts, joint attention, independent MLPs."""
    def __init__(self, d=64, heads=4, n_layers=2):
        super().__init__()
        self.n_layers = n_layers
        self.video_qkv = nn.ModuleList([nn.Linear(d, 3*d) for _ in range(n_layers)])
        self.action_qkv = nn.ModuleList([nn.Linear(d, 3*d) for _ in range(n_layers)])
        self.video_mlp  = nn.ModuleList([nn.Sequential(nn.Linear(d,d*4), nn.GELU(), nn.Linear(d*4,d)) for _ in range(n_layers)])
        self.action_mlp = nn.ModuleList([nn.Sequential(nn.Linear(d,d*4), nn.GELU(), nn.Linear(d*4,d)) for _ in range(n_layers)])
        self.heads, self.d = heads, d

    def forward(self, video, action):  # video: (B,T,D), action: (B,A,D)
        for i in range(self.n_layers):
            Tv, Ta = video.shape[1], action.shape[1]
            # each expert computes its own Q, K, V
            qv, kv, vv = self.video_qkv[i](video).chunk(3, -1)
            qa, ka, va = self.action_qkv[i](action).chunk(3, -1)
            # joint attention pool
            Q = torch.cat([qv, qa], 1); K = torch.cat([kv, ka], 1); V = torch.cat([vv, va], 1)
            attn = F.scaled_dot_product_attention(Q, K, V)
            video  = video  + attn[:, :Tv] ; video  = video  + self.video_mlp[i](video)
            action = action + attn[:, Tv:] ; action = action + self.action_mlp[i](action)
        return video, action

B, T, A, D = 2, 16, 4, 64
video  = torch.randn(B, T, D)
action = torch.randn(B, A, D)
mot = TinyMoT(d=D)
v_out, a_out = mot(video, action)
print("video out :", v_out.shape)   # (2, 16, 64)
print("action out:", a_out.shape)   # (2, 4, 64)
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
video out : torch.Size([2, 16, 64])
action out: torch.Size([2, 4, 64])
```

中文：这个 30 行实现捕捉了 MoT 的核心：两个 expert 各自用独立权重计算 Q/K/V（不同的 `video_qkv` 和 `action_qkv`），然后拼接做一次 `scaled_dot_product_attention`，再切分。注意到动作 token 的输出 `a_out` 会受到视频 token `kv`、`vv` 的影响，反之亦然——这就是"混合"的含义。

English: This 30-line implementation captures the MoT core: two experts compute Q/K/V with independent weights (`video_qkv` vs `action_qkv`), concatenate for one `scaled_dot_product_attention`, then split back. Note that action token output `a_out` is influenced by video tokens' `kv`, `vv`, and vice versa — that's what "mixed" means.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Wan2.1 DiT (cross-attention action conditioning)** / **Wan2.1 DiT（cross-attention 动作条件化）**: 动作作为 cross-attention 的 key/value，视频 query 看动作，但动作不看视频——MoT 是对这个限制的直接改进。 / Action as cross-attention key/value; video query sees action but action doesn't see video — MoT directly addresses this limitation.
- **Unified Sequence Modeling (e.g. Gato, AnyMAL)** / **统一序列建模**: 把视频 token 和动作 token 直接拼接成一条序列做 LM，等价于 token concat，但没有独立 expert 权重。 / Concat video and action tokens into one sequence for LM — equivalent to token concat without independent expert weights.
- **Joint Video-Language Attention in Video LLMs (e.g. Video-LLaVA)** / **视频 LLM 联合注意力**: 视频 patch token 和文本 token 在同一 attention 层交互，但两种 token 通常共享 LM 权重，MoT 更进一步为每种 token 保留独立权重。 / Video patch tokens and text tokens interact in the same attention layer, but both share LM weights; MoT goes further by keeping independent weights per token type.

## 注意事项 / Caveats / when it breaks

- **attention_mask 必须覆盖全序列** / **attention_mask must cover the full sequence**: 联合注意力的序列长度是两个 expert 的总和，如果 mask 大小只设为视频序列长度，会触发形状不匹配错误（代码里有显式检查）。 / The joint attention sequence length is the sum of both experts; if the mask is sized only to the video sequence, a shape mismatch error occurs (the code has explicit checks for this).
- **显存随 expert 数量线性增长** / **Memory scales linearly with expert count**: 每个 expert 都有自己的 Q/K/V，联合序列长度 = Σ(所有 expert 序列长)，注意力计算的复杂度是 O(总序列长²)。超过 4-5 个 expert 时需要稀疏注意力。 / Each expert contributes its own Q/K/V; joint sequence length = Σ(all expert lengths); attention complexity is O(total_seq²). With more than 4-5 experts, sparse attention is needed.
- **RoPE 频率需要为每个 expert 单独设计** / **RoPE frequencies need per-expert design**: 视频 token 的 2D/3D 空间坐标和动作 token 的关节坐标含义不同，不能共享同一套 RoPE 频率，否则位置编码失去意义。 / Video tokens have 2D/3D spatial coordinates while action tokens have joint-space coordinates — sharing RoPE frequencies across experts makes positional encoding meaningless.

## 延伸阅读 / Further reading

- [FastWAM repository](https://github.com/yuantianyuan01/FastWAM)
- [Mixture of Experts (MoE) survey](https://arxiv.org/abs/2407.06204) — MoT 与 MoE 的关系
- [Wan2.1 technical report](https://arxiv.org/abs/2503.20314)
- [Joint Video-Action Modeling for World Models](https://arxiv.org/abs/2502.09465)
