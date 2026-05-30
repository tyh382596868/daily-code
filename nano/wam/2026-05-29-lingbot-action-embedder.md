---
date: 2026-05-29
topic: wam
source: wam
repo: Robbyant/lingbot-va
file: wan_va/modules/model.py
permalink: https://github.com/Robbyant/lingbot-va/blob/58c2ae5bac46bd8114065bea9d7d256eb67c16c3/wan_va/modules/model.py#L624-L649
difficulty: beginner
read_time: ~8 min
tags: [code-of-the-day, wam, action, embedder, projector, single-stream]
build_role: Action encoder + head (single-stream design) — actions enter and leave the *same* DiT as video latents
---

# lingbot-va 的 action 接线方案:两个 Linear 加一个 deepcopy / lingbot-va's action stack: two Linears and a deepcopy

> **一句话 / In one line**: lingbot-va 不给 action 单独搭一套网络,而是用 `nn.Linear(action_dim → hidden)` 把动作直接编成一种 token,塞进**同一个** DiT 与 video latents 共流(用昨天的 FlexAttention mask 隔离),最后 `nn.Linear(hidden → action_dim)` 把模型输出投回动作维度。 / lingbot-va doesn't build a second model for actions — it projects actions into a token via `nn.Linear(action_dim → hidden)`, feeds them into the *same* DiT alongside video latents (separated by yesterday's FlexAttention mask), and projects back with `nn.Linear(hidden → action_dim)`.

## 为什么重要 / Why this matters

WAM 怎么处理 action 有几种流派,lingbot-va 是"最小变更派"—— action 只是另一种长度的 token。这么做的好处:(1) 不用维护两个 DiT;(2) action 自动获得 cross-attention 看 text、获得 self-attention 看历史 video,无需特殊连线;(3) 训练 / 推理代码改动最小。代价是:action token 数 vs video token 数严重失衡(几十 vs 几万),attention 里 action 那一段贡献容易被淹没,需要靠 mask + loss 加权救场。这一节看的就是这种"单流派"最小可工作实现 —— 只有两个 nn.Linear + 一个 deepcopy。

There are several schools for handling actions in WAM. lingbot-va belongs to the minimal-changes school: actions are just tokens of a different length. Benefits: (1) no second DiT to maintain, (2) actions automatically get cross-attention to text and self-attention to video history without extra wiring, (3) training / inference paths are basically unchanged. The cost: action tokens (dozens) are dwarfed by video tokens (tens of thousands) within attention, so loss weighting and FlexAttention masking must compensate. These few lines are the minimal viable implementation — two Linears and a deepcopy.

## 代码 / The code

`Robbyant/lingbot-va` — [`wan_va/modules/model.py`](https://github.com/Robbyant/lingbot-va/blob/58c2ae5bac46bd8114065bea9d7d256eb67c16c3/wan_va/modules/model.py#L624-L649)

```python
# === Inside the DiT __init__ ===

# ---- Encoders (token-shaping) ----
self.patch_embedding_mlp = nn.Linear(
    in_channels * patch_size[0] * patch_size[1] * patch_size[2],
    inner_dim)                                              # video patch → token
self.action_embedder = nn.Linear(action_dim, inner_dim)     # action vector → token

# ---- Conditioning ----
self.condition_embedder = WanTimeTextImageEmbedding(
    dim=inner_dim, time_freq_dim=freq_dim,
    time_proj_dim=inner_dim * 6, text_embed_dim=text_dim,
    pos_embed_seq_len=pos_embed_seq_len)
self.condition_embedder_action = deepcopy(self.condition_embedder)   # ← independent timestep embedder

# ---- The shared transformer trunk ----
self.blocks = nn.ModuleList([
    WanTransformerBlock(inner_dim, ffn_dim, num_attention_heads,
                        cross_attn_norm, eps, attn_mode=attn_mode)
    for _ in range(num_layers)])

# ---- Heads (token → output) ----
self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
self.proj_out = nn.Linear(inner_dim,
                          out_channels * math.prod(patch_size))      # token → video patch
self.action_proj_out = nn.Linear(inner_dim, action_dim)              # token → action vector
```

## 逐行讲解 / What's happening

1. **两条对称的"编码 → 解码"管线 / Two symmetric encode → decode pipelines**:
   - 中文:`patch_embedding_mlp` 把一个 video patch(`patch_size_product * channels` 维的向量)投到 DiT 的 `inner_dim`;`action_embedder` 把一个长度 `action_dim` 的动作向量同样投到 `inner_dim`。注意 video 那边比 action 多了一步 patchify,但**最终都到同一个 hidden 空间**,可以共流。
   - English: `patch_embedding_mlp` projects one video patch (`patch_size_product * channels`) into DiT `inner_dim`; `action_embedder` projects one action vector into the same `inner_dim`. Video needs a patchify upstream, but both arrive in the same hidden space and can flow side by side.

2. **`condition_embedder_action = deepcopy(condition_embedder)` 是关键一行 / The key one-liner**:
   - 中文:为什么不直接共享 timestep/text embedder?**因为 action 和 video 用不同的 noise schedule**。训练时(看昨天的 `_add_noise` 笔记)`train_scheduler_action` 和 `train_scheduler_latent` 是两套独立的调度器;同一个物理时间 `t=0.5` 在 video 端代表"中等噪声"、在 action 端可能代表"几乎无噪声"。两个 embedder 学到不同的 timestep → token 映射。`deepcopy` 是结构上同款、参数上独立。
   - English: why not share the timestep / text embedder? **Because actions and video have independent noise schedules.** Yesterday's `_add_noise` note showed `train_scheduler_action` and `train_scheduler_latent` are two separate schedulers; the same physical `t=0.5` means "medium noise" on the video side but possibly "almost clean" on the action side. The two embedders learn different timestep → token mappings. `deepcopy` keeps the architecture identical but the parameters independent.

3. **`blocks` 是同一个 ModuleList,两条流共享 / Shared transformer blocks**:
   - 中文:`self.blocks` 没有"action 专属"的副本 —— video token 和 action token 在每个 block 里走同样的 attention 和 FFN。差别全在**输入序列怎么拼**和**FlexAttention mask 怎么写**(见昨天的 mask 笔记)。
   - English: `self.blocks` has no action-only twin. Video tokens and action tokens go through the same attention and FFN. The difference is purely in (a) how the input sequence is concatenated and (b) the FlexAttention mask (yesterday's note).

4. **两个独立 head / Two output heads**:
   - 中文:`proj_out` 形状 `inner_dim → out_channels * patch_size_product`,因为 video token 解码回来需要乘开 patch;`action_proj_out` 直接 `inner_dim → action_dim`,因为 action 本来就是一维向量,没有 patchify 这一层。训练 step 里 `latent_pred, action_pred = pred` 就是这两个 head 各自产出。
   - English: `proj_out` is `inner_dim → out_channels * patch_size_product` because video tokens need to be unpatchified; `action_proj_out` is `inner_dim → action_dim` directly because actions were never patchified. The `latent_pred, action_pred = pred` split in the training step pulls these two outputs apart.

5. **没有 ActionEncoder MLP / No pi0-style action MLP**:
   - 中文:dreamzero / openpi 那种"action + timestep concat → swish → MLP"的写法在这里不存在 —— lingbot-va 把 timestep 信息走 `condition_embedder_action` 通过 adaLN-Zero 注入到每个 block,所以 action token 本身只需要一个 Linear。架构更扁。
   - English: dreamzero / openpi style "concat action with timestep, MLP, swish" is *absent* here. lingbot-va injects timestep via `condition_embedder_action` and adaLN-Zero into every block, so the action token itself only needs a single Linear at the input. The architecture is flatter.

## 类比 / The analogy

像编辑一份大文档时,把"图片"和"标注文字"都用同一台打字机打出来 —— 区别在于图片是巨型的(占很多行)、标注是小的(只占一行)。两种内容用同一个排版引擎(DiT)和同样的版式规则(blocks),但你给图片和标注各发了一支独立的笔(`condition_embedder` vs `condition_embedder_action`)来标注它们各自的"时间戳"(改动版本)。最后输出时图片走"图像导出"(`proj_out`),标注走"纯文本导出"(`action_proj_out`)。

Picture editing a long document where you'd typeset both photos and captions on the same typewriter — photos take many lines, captions take one. The same layout engine (DiT) and same typesetting rules (blocks) handle both, but you carry two independent pens (`condition_embedder` vs `condition_embedder_action`) for marking each item's own revision timestamp. At export time, photos go through the photo printer (`proj_out`) and captions through plain-text export (`action_proj_out`).

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-WAM

中文:在 nanoWAM 里这是 `nano/wam/model/single_stream_action.py` —— 整个"action 接入"层。上游:dataloader 输出 `{video_latents, actions, text_emb}` 三件套,patchify 给 video,直接通过 `action_embedder` 给 action,然后**沿序列维度拼起来**(video tokens 在前、action register tokens 在后),通过同一个 `blocks` ModuleList,出口处再切回两段过 `proj_out` 和 `action_proj_out`。如果省掉这条 action 流(只做纯 video diffusion):你只能预测未来帧,没法预测未来动作 —— WAM 退化成普通 video diffusion。生产实现的关键扩展:(1) **action loss 权重**(action token 数少,loss 容易被 video 淹没,要 5-10× 加权);(2) **noisy action conditioning**(训练时一定概率给 condition action 加噪声,防止推理 drift);(3) **VAE for action**(如果 action 维度很高,可以再加一个小 VAE 压缩 —— lingbot-va 没做);(4) **multi-embodiment**(切换不同机器人时把 `action_embedder` 改成 `CategorySpecificLinear`,见 2026-05-29 Isaac-GR00T 笔记)。

English: in nanoWAM this is `nano/wam/model/single_stream_action.py` — the whole "action plug-in" layer. Upstream: the dataloader hands `{video_latents, actions, text_emb}`; patchify the video, run `action_embedder` on the action, **concatenate along the sequence axis** (video tokens first, action register tokens after), pass through the shared `blocks`, then split the output and run `proj_out` for video and `action_proj_out` for action. Skip this and nanoWAM degenerates to pure video diffusion — no future-action prediction. Production extensions: (1) **upweight action loss** (action tokens are few, video loss otherwise dominates) by 5-10×, (2) **noisy action conditioning** during training to prevent drift at autoregressive rollout, (3) **action VAE** if action dim is large (lingbot-va doesn't bother), (4) **multi-embodiment** by swapping `action_embedder` for `CategorySpecificLinear` (see the Isaac-GR00T note).

## 自己跑一遍 / Try it yourself

```python
# pip install torch
# Minimal "video + action share one transformer" forward pass.
import torch, torch.nn as nn

torch.manual_seed(0)
inner_dim, action_dim, video_patch_dim = 64, 7, 16
n_video_tokens, n_action_tokens = 12, 4
B = 2

action_embedder    = nn.Linear(action_dim, inner_dim)
patch_embedding    = nn.Linear(video_patch_dim, inner_dim)
trunk              = nn.TransformerEncoderLayer(inner_dim, nhead=4, batch_first=True)
proj_out           = nn.Linear(inner_dim, video_patch_dim)
action_proj_out    = nn.Linear(inner_dim, action_dim)

video_patches  = torch.randn(B, n_video_tokens, video_patch_dim)
action_vectors = torch.randn(B, n_action_tokens, action_dim)

# 1) Encode both into the shared hidden space
video_tokens  = patch_embedding(video_patches)            # [B, 12, 64]
action_tokens = action_embedder(action_vectors)            # [B,  4, 64]

# 2) Concatenate and pass through the SAME trunk
seq = torch.cat([video_tokens, action_tokens], dim=1)     # [B, 16, 64]
out = trunk(seq)

# 3) Split back and decode independently
video_out, action_out = out[:, :n_video_tokens], out[:, n_video_tokens:]
video_pred  = proj_out(video_out)                         # [B, 12, 16]
action_pred = action_proj_out(action_out)                 # [B,  4,  7]
print("video_pred  :", video_pred.shape)
print("action_pred :", action_pred.shape)
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
video_pred  : torch.Size([2, 12, 16])
action_pred : torch.Size([2, 4, 7])
```

中文:整个"接入 action"用了 6 行代码,核心就是"两个 Linear 在前、两个 Linear 在后,中间共享 trunk"。

English: bolting actions onto a video transformer is six lines — two Linears in, two Linears out, one shared trunk.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **NaVid / RT-2(prefix-concat 推理)** / **NaVid / RT-2 (prefix-concat decoding)**: 中文 — LLM 里"在 video token 序列后面接 action token"的思路,跟 lingbot-va 数学上等价。 / English — the "append action tokens after video tokens" idea in LLM-style decoders; mathematically the same.
- **昨天的 FlexAttention mask** / **Yesterday's FlexAttention mask**: 中文 — 这条 action 流要工作,必须配合 mask 把"action 不能看自己 clean 当前帧"约束住。两段笔记一起看。 / English — this single-stream action design only works when paired with yesterday's mask that prevents "action attending its own clean current frame". Read them together.
- **FastWAM 的 ActionDiT(今天另一篇)** / **FastWAM's ActionDiT (today's other note)**: 中文 — 对照实验:那边不共享 trunk,而是单独一份 ActionDiT。 / English — the contrastive design: FastWAM does not share the trunk; it spins up a parallel ActionDiT.
- **dreamzero register tokens(今天第三篇)** / **dreamzero register tokens (today's third note)**: 中文 — 把 action token 当成 transformer 的"register tokens"塞进序列,但每个 register 用独立的 1D RoPE。lingbot-va 没用 RoPE 区分 action token,纯靠 mask。 / English — pushes the design further: action becomes "register tokens" appended to video, each with its own 1-D RoPE. lingbot-va doesn't use RoPE to distinguish action — it relies solely on the mask.

## 注意事项 / Caveats / when it breaks

- **`deepcopy` 不能省** / **Don't skip `deepcopy`**: 中文 — 如果 video 和 action 共享同一个 `condition_embedder`,两边 timestep 分布不同会拉扯参数,训练曲线发散。 / English — sharing one `condition_embedder` couples the two timestep distributions and destabilises training.
- **action token 数太少会被 attention 淹没** / **Action tokens get drowned in attention**: 中文 — 一段视频 30k token,action 一帧 4 个 token,attention 里 action 的 logits 几乎不动。要么加 loss 权重,要么用 register token + 强 RoPE 强制让它"可见"。 / English — with 30 k video tokens and 4 action tokens per frame, attention barely budges for actions. Compensate via loss weighting or distinct positional encoding (the dreamzero approach).
- **action_dim 必须知道,且固定** / **`action_dim` must be known and fixed**: 中文 — `nn.Linear(action_dim, inner_dim)` 的输入维度是写死的。换机器人就要换 head,或者用 CategorySpecificLinear。 / English — `nn.Linear(action_dim, inner_dim)` hard-codes the input width. Switching robots means replacing the head or upgrading to `CategorySpecificLinear`.

## 延伸阅读 / Further reading

- [LingBot-VA paper](https://github.com/Robbyant/lingbot-va/blob/main/LingBot_VA_paper.pdf)
- [Diffusion Policy — single-stream action diffusion ancestor](https://arxiv.org/abs/2303.04137)
- [Today's FastWAM ActionDiT note](./2026-05-29-fastwam-action-dit.md)
- [Today's dreamzero action register note](./2026-05-29-dreamzero-action-registers.md)
