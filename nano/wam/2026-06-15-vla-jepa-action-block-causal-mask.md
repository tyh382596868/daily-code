---
date: 2026-06-15
topic: wam
source: wam
repo: huggingface/lerobot
file: src/lerobot/policies/vla_jepa/world_model.py
permalink: https://github.com/huggingface/lerobot/blob/38327fdc8458959f47d555c159307538200d0561/src/lerobot/policies/vla_jepa/world_model.py#L313-L418
difficulty: intermediate
read_time: ~12 min
tags: [code-of-the-day, wam, world-model, action-conditioning, frame-causal-mask, jepa]
build_role: action-conditioning
---

# Action 不在 cross-attn 里,直接和帧 token 并排坐:VLA-JEPA 的 ActionConditionedVideoPredictor / Actions don't go into cross-attn — they sit alongside frame tokens: VLA-JEPA's `ActionConditionedVideoPredictor`

> **一句话 / In one line**: 把每一帧的 H·W 个 patch token 和当时执行的 action token 拼成一个 block,然后用 16 行就能构造的 frame-causal mask 让每个 block 只能看到自己和过去所有 block——一次 attention 把"动作 → 下一帧"的世界模型学完。 / Each frame's `H·W` patch tokens are concatenated with that frame's action tokens into one block; a 16-line frame-causal mask makes every block attend to itself and all earlier blocks. One attention call learns "action → next frame".

## 为什么重要 / Why this matters

World action model (WAM) 里"action 怎么注入"是个有不止一种正确答案的设计题。 GR00T 用 cross-attention,FastWAM 用 action prefill cache,lingbot-va 用 FlexAttention 拼接七张 boolean mask。VLA-JEPA 这个 2026-06-04 新加进 lerobot 的实现采取了第四种路线:**把 action token 和 video patch token 当成同等公民,在一个 self-attention 流里通过 mask 控制谁能看谁**。这种"内联交错"的设计非常适合训练阶段——一次 forward 就把所有 frame 都过完,不需要 KV cache,也不需要单独写一个 action encoder 接 cross-attn。 对从头搭 nanoWAM 的人来说,这是除了 cross-attn 之外的另一个值得理解的范式。

In a world action model (WAM), "how do actions condition the video stream" has multiple correct answers. GR00T uses cross-attention; FastWAM uses an action-prefill cache; lingbot-va composes seven boolean masks via FlexAttention. VLA-JEPA, freshly landed in lerobot on 2026-06-04, picks a fourth route: **treat action tokens and video patch tokens as equal citizens in a single self-attention stream, and let a mask govern who attends to whom**. This "inline interleave" is exceptionally clean during training — one forward processes every frame, no KV cache, no separate action-encoder-into-cross-attn plumbing. Worth understanding if you're building nanoWAM from scratch.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/policies/vla_jepa/world_model.py`](https://github.com/huggingface/lerobot/blob/38327fdc8458959f47d555c159307538200d0561/src/lerobot/policies/vla_jepa/world_model.py#L313-L418)

```python
def build_action_block_causal_attention_mask(
    num_frames: int, grid_height: int, grid_width: int, add_tokens: int = 1
) -> torch.Tensor:
    tokens_per_frame = add_tokens + grid_height * grid_width
    num_tokens = num_frames * tokens_per_frame
    mask = torch.zeros(num_tokens, num_tokens, dtype=torch.bool)
    mask_block = torch.ones(tokens_per_frame, tokens_per_frame, dtype=torch.bool)
    local_window_time = num_frames

    for current_frame in range(num_frames):
        first_context_frame = max(0, current_frame - local_window_time + 1)
        for context_frame in range(first_context_frame, current_frame + 1):
            row = slice(current_frame * tokens_per_frame, (current_frame + 1) * tokens_per_frame)
            col = slice(context_frame * tokens_per_frame, (context_frame + 1) * tokens_per_frame)
            mask[row, col] = mask_block
    return mask


class ActionConditionedVideoPredictor(nn.Module):
    """JEVLA1-compatible action-conditioned V-JEPA predictor."""

    def __init__(self, num_frames, img_size, patch_size, tubelet_size,
                 embed_dim, action_embed_dim, predictor_embed_dim,
                 depth, num_heads, mlp_ratio, num_action_tokens_per_step,
                 use_extrinsics=False):
        super().__init__()
        self.is_frame_causal = True
        self.predictor_embed = nn.Linear(embed_dim,        predictor_embed_dim, bias=True)
        self.action_encoder  = nn.Linear(action_embed_dim, predictor_embed_dim, bias=True)
        # ... grid_height, grid_width, predictor_blocks (RoPE attention DiTs) ...
        self.predictor_norm  = nn.LayerNorm(predictor_embed_dim, eps=1e-6)
        self.predictor_proj  = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

    def forward(self, frame_tokens, action_tokens, extrinsics=None):
        # starVLA input convention: frame_tokens [B, T*H*W, D], actions [B, T*A, D].
        x = self.predictor_embed(frame_tokens)
        batch_size, num_context_tokens, hidden_dim = x.size()
        num_frames = num_context_tokens // (self.grid_height * self.grid_width)

        actions = self.action_encoder(action_tokens)
        actions = actions.view(batch_size, num_frames, -1, hidden_dim)
        cond_tokens = actions.shape[2]

        x = x.view(batch_size, num_frames, self.grid_height * self.grid_width, hidden_dim)
        x = torch.cat([actions, x], dim=2).flatten(1, 2)   # interleave per-frame

        attn_mask = build_action_block_causal_attention_mask(
            num_frames, self.grid_height, self.grid_width, add_tokens=cond_tokens
        )
        attn_mask = attn_mask[: x.size(1), : x.size(1)].to(x.device, non_blocking=True)

        for block in self.predictor_blocks:
            x = block(x,
                      attn_mask=attn_mask,
                      num_frames=num_frames,
                      grid_height=self.grid_height,
                      grid_width=self.grid_width,
                      action_tokens=cond_tokens)

        x = x.view(batch_size, num_frames, cond_tokens + self.grid_height * self.grid_width, hidden_dim)
        x = x[:, :, cond_tokens:, :].flatten(1, 2)         # discard action positions
        x = self.predictor_norm(x)
        return self.predictor_proj(x)
```

## 逐行讲解 / What's happening

### Mask 构造 / Building the mask

1. **`tokens_per_frame = add_tokens + H * W`**:
   - 中文: 每个"逻辑帧"现在包含 `H·W` 个视觉 patch token 加上 `add_tokens` 个 action token(`add_tokens` 在 forward 里就是 `cond_tokens`,即一帧对应的 action token 数量)。这是这套设计最关键的一步抽象:**把动作 token 当作"帧的一部分"**。
   - English: A "logical frame" now contains `H·W` visual patch tokens plus `add_tokens` action tokens (`add_tokens = cond_tokens` from forward — the per-frame action token count). The crucial abstraction: **action tokens are part of the frame.**

2. **`mask = torch.zeros(N, N, dtype=torch.bool)` + `mask_block = torch.ones(tokens_per_frame, tokens_per_frame, dtype=torch.bool)`**:
   - 中文: 整个 N×N 的 mask 默认全 False(不允许 attention),要打开的部分用 `mask_block`(全 True 的方块)填进去。
   - English: The whole N×N mask defaults to False (no attention); the parts to open up are stamped with `mask_block` (an all-True square the size of one frame).

3. **嵌套 for 循环 / Nested `for current_frame` / `for context_frame`**:
   - 中文: 对每个 current frame,context_frame 从 `max(0, current - window + 1)` 到 `current`——也就是覆盖当前帧的过去 `window` 帧 + 当前帧本身。这里 `local_window_time = num_frames`,等价于"无限往回看",即标准的 frame-causal:第 t 帧可以看到 0..t 所有帧的所有 token。
   - English: For each current frame, context_frame ranges from `max(0, current - window + 1)` through `current`. Here `local_window_time = num_frames` means "see all past frames" — standard frame-causal: frame t attends to all tokens in frames `0..t`.

4. **`mask[row, col] = mask_block` / The stamping step**:
   - 中文: 把一个 `tokens_per_frame × tokens_per_frame` 的全 True 方块"盖章"在 `(current_frame, context_frame)` 这个 block 位置。同帧内 (intra-frame) 也允许全连接 — 这意味着 *帧内* 没有 causal mask,视觉 patch 之间可以双向 attention,这是 ViT 的标准设定。
   - English: An all-True `tokens_per_frame × tokens_per_frame` square is *stamped* at the `(current_frame, context_frame)` block position. Within a frame, attention is fully bidirectional (standard ViT) — only the *between-frame* axis is causal.

### Forward 流程 / The `forward` pipeline

5. **`x = self.predictor_embed(frame_tokens)`**:
   - 中文: 把外部 (一般是 V-JEPA encoder 产出) 的 `embed_dim` 投影到 predictor 内部的 `predictor_embed_dim`。这是 V-JEPA 经典做法——encoder 和 predictor 用 *不同* 隐维度,encoder 通常更大,predictor 更小更窄。
   - English: Project the externally produced (typically V-JEPA encoder) `embed_dim` to the predictor's internal `predictor_embed_dim`. Classic V-JEPA — encoder is deep & wide, predictor is shallower & narrower.

6. **`actions = self.action_encoder(action_tokens).view(B, num_frames, -1, D)`**:
   - 中文: action token(可以是 VLM 输出的某些 hidden state)同样投到 predictor 空间,然后 reshape 成 `[B, T_frames, A, D]`——每帧有 A 个 action token。
   - English: Action tokens (often VLM hidden states corresponding to action slots) are projected into predictor space, then reshaped to `[B, T_frames, A, D]` — A action tokens per frame.

7. **`x.view(B, num_frames, H*W, D)` + `torch.cat([actions, x], dim=2).flatten(1, 2)`**:
   - 中文: 这是 "inline interleave"的核心两行:先把帧 token reshape 成 `[B, T, H·W, D]`,然后在第 2 维(逐帧维度)拼上 action token,变成 `[B, T, A + H·W, D]`,最后 `flatten(1, 2)` 把 (T, A+H·W) 压扁成一条 `[B, T·(A+H·W), D]` 的序列。 *顺序很关键*——action 在前,frame token 在后,后面 `x[:, :, cond_tokens:, :]` 才能精确切回纯 video 部分。
   - English: The "inline interleave" — reshape frames to `[B, T, H·W, D]`, concat action tokens along the per-frame axis to get `[B, T, A + H·W, D]`, then `flatten(1, 2)` collapses `(T, A+H·W)` into one sequence `[B, T·(A+H·W), D]`. *Order matters*: actions first within each frame, so the later `x[:, :, cond_tokens:, :]` slice recovers exactly the video positions.

8. **`attn_mask = build_action_block_causal_attention_mask(...)` + `[: x.size(1), : x.size(1)]`**:
   - 中文: 用上面那个工具函数生成完整 mask,然后把它截到当前 batch 的真实长度(因为 `extrinsics` 那条分支可能加额外 token,这里 `[:N, :N]` 兜底)。Mask 是 `bool`,SDPA 接受 bool mask:True = 允许 attend。
   - English: Build the full mask via the helper, then crop to the actual sequence length (the `use_extrinsics` branch can add one extra per-frame token; the slice guards against off-by-one). The mask is `bool`; SDPA accepts bool masks with True = allowed.

9. **`for block in self.predictor_blocks: x = block(...)`**:
   - 中文: 堆叠的 ACBlock(AC = Action-Conditioned)各自带 RoPE 注意力。注意每个 block 都被告知 `(num_frames, grid_height, grid_width, action_tokens)`——RoPE 需要知道每个 token 的 (frame, h, w) 坐标。
   - English: A stack of ACBlock (Action-Conditioned) layers, each with RoPE attention. Every block receives `(num_frames, grid_height, grid_width, action_tokens)` — RoPE needs each token's `(frame, h, w)` coordinates to apply the right rotation.

10. **`x.view(B, num_frames, A + H·W, D); x = x[:, :, cond_tokens:, :].flatten(1, 2)`**:
    - 中文: 跑完 N 层 attention 之后,把序列还原成 `[B, T, A+H·W, D]`,*丢掉* 前 A 个 action 位置,只保留视频 patch 那段。这就是 JEPA 风格的 "predict future patch embedding" 输出。
    - English: After N attention layers, reshape back to `[B, T, A+H·W, D]`, *discard* the first A action positions, keep only the video patch part. That's the JEPA-style "predict future patch embedding" output.

## 类比 / The analogy

想象一个会议室的桌子:每张桌子坐 H·W 个评委(视觉 patch)和 A 个发言人(action)。会议从早上 9 点开到晚上 6 点(num_frames 个时间步),每小时换一桌人——但 *新的一桌人能看到所有过去几桌的讨论记录*,反之不行。同一桌内,所有评委和发言人是 *完全自由交流* 的(同帧内无 causal mask)。跨桌之间,只能"后辈看前辈"(frame-causal)。`build_action_block_causal_attention_mask` 就是会议的"谁能听谁说话"规则手册;forward 则是把每张桌子的人(action + video patches)拼成一条长队走进会议室。

Picture a meeting where each table seats `H·W` reviewers (video patches) + `A` speakers (actions). The meeting spans `num_frames` time-slots, one table per slot. A new table sees the transcripts of *all earlier* tables but no future ones (frame-causal across tables). Within a table, reviewers and speakers chat freely (bidirectional intra-frame attention). `build_action_block_causal_attention_mask` is the "who can listen to whom" rulebook; `forward` lines all tables (actions + video patches) into one long queue and walks them into the meeting room.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

> **curriculum_id**: `action-conditioning`
>
> **depends_on**: `dit-block`(我们已经在 06-08 wan21、06-07 open-sora、06-05 isaac-groot 笔记里覆盖过)。

`action-conditioning` 是 WAM 里所有"动作如何影响下一帧视频"的设计入口。 nano-WAM 课程已经看过三种实现:
- **GR00T (cross-attn)**:action 走独立 cross-attention 流,2026-06-04 笔记。
- **lingbot-va (FlexAttention 拼接 7 张 bool mask)**:同流但用 FlexAttention 的 BlockMask,2026-05-29 笔记。
- **VLA-JEPA (今天)**:同流 + 简单的 frame-causal mask,直接 cat。

这三种代表了三档抽象:GR00T 用一个标准 cross-attn 模块(最高耦合,但可移植性最好);lingbot-va 用 FlexAttention 把 mask 写成谓词(灵活但难调试);VLA-JEPA 用纯 cat + 一个 16 行的 mask builder(最简单也最直观,但 mask 必须按 `(A + H·W)` 块对齐,改 chunk size 时要重新构造)。

**输入 / Inputs**:
- `frame_tokens: [B, T·H·W, embed_dim]` — V-JEPA encoder 输出的视频 patch token。
- `action_tokens: [B, T·A, action_embed_dim]` — VLM 输出对应的 action position 的 hidden states。

**输出 / Output**:`[B, T·H·W, embed_dim]` — 预测的"下一帧"patch embedding(JEPA 风格)。

**省掉这个组件会怎样 / What if you omit this**:动作和视频独立,你的 WAM 退化成"无控视频生成"——它能预测下一帧但不知道是因为机器人做了什么动作,失去了 imitation learning / planning 用的因果信号。

**生产级实现还要加什么 / What production adds**:
- 大 batch / 长序列下的 sparse attention(FlexAttention 的 BlockMask 比 dense bool mask 显存便宜 100×);
- Classifier-free guidance:训练时随机 drop action token,推理时混合 cond / uncond(参考 2026-06-10 那篇 DiT LabelEmbedder 笔记);
- Action chunking(每帧 1 个 action 太粗,实际 robotics 是 30 Hz, 一帧对应多个 action subtoken);
- 多视角融合(2-3 个相机)。

**Where this fits in nano-WAM**: `action-conditioning` is the slot governing how actions influence the next predicted frame. We've now covered three implementations of the same slot — GR00T's cross-attention, lingbot-va's FlexAttention mask composition, and today's VLA-JEPA inline-cat-plus-frame-causal-mask. The three sit on different points of an abstraction tradeoff: cross-attn is the most portable, FlexAttention is the most flexible, the inline-cat is the simplest and most direct.

Inputs: `frame_tokens: [B, T·H·W, embed_dim]`, `action_tokens: [B, T·A, action_embed_dim]`. Output: `[B, T·H·W, embed_dim]` next-frame patch embeddings (JEPA style). Omit this component and your WAM degrades to unconditional video generation — it predicts the next frame but doesn't know which action caused the transition, killing the imitation-learning / planning signal. Production layers on top: sparse FlexAttention for memory, classifier-free guidance (random action dropout at training), action chunking (multiple sub-tokens per frame), multi-camera fusion.

## 自己跑一遍 / Try it yourself

```python
# pip install torch
import torch

def build_action_block_causal_mask(num_frames, H, W, A):
    """A = action tokens per frame. Returns a [(A+H*W)*T, (A+H*W)*T] bool mask."""
    tokens_per_frame = A + H * W
    N = num_frames * tokens_per_frame
    mask = torch.zeros(N, N, dtype=torch.bool)
    block = torch.ones(tokens_per_frame, tokens_per_frame, dtype=torch.bool)
    for cur in range(num_frames):
        for ctx in range(cur + 1):                        # 0..cur inclusive
            r = slice(cur * tokens_per_frame, (cur + 1) * tokens_per_frame)
            c = slice(ctx * tokens_per_frame, (ctx + 1) * tokens_per_frame)
            mask[r, c] = block
    return mask

# Tiny demo: 3 frames, 2×2 grid, 1 action token per frame.
T, H, W, A = 3, 2, 2, 1
m = build_action_block_causal_mask(T, H, W, A)
print(m.int())
print("shape:", m.shape, "  density:", m.float().mean().item())
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        ...
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
shape: torch.Size([15, 15])   density: 0.666...
```

中文一两句:看左下三角:每隔 5 行(`A + H·W = 1 + 4 = 5`)往右扩 5 列——这就是 block-causal 在矩阵上的样子。 *帧内* 是全 1 方块,*帧间* 是阶梯状下三角。

In English: the bottom-left triangle is block-staircase — every 5 rows (`A + H·W = 1 + 4`) the row-range expands by 5 columns. *Within* a frame: solid 1-block. *Across* frames: staircase lower triangle.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **NVIDIA Isaac-GR00T (cross-attention 变体)** / **NVIDIA Isaac-GR00T (cross-attn variant) — 2026-06-05 note**: 同一个 action-conditioning 任务,GR00T 把 action 走独立 cross-attn 流。 / Same task, GR00T routes action through a separate cross-attn stream.
- **Robbyant/lingbot-va (FlexAttention 多 mask 组合)** / **Robbyant/lingbot-va (FlexAttention multi-mask compose) — 2026-05-29 note**: 同一个 inline interleave 思想,但用 7 个 boolean 谓词组合出一个 BlockMask。 / Same inline-interleave thought, but with 7 boolean predicates composed into a BlockMask.
- **Wan2.1 的 WanAttentionBlock** / **Wan2.1's WanAttentionBlock — 2026-06-08 note**: text + image cross-attn 接进 self-attn 块——和 VLA-JEPA 这版结构上更像 wan2.1 的 *镜像*(他们做的是 text→video,VLA-JEPA 做的是 action→video)。 / Mirror image of VLA-JEPA: Wan2.1 routes text into self-attn for text→video; VLA-JEPA routes action into self-attn for action→video.
- **decision transformer 把 (state, action, return) 拼成一长串** / **Decision Transformer interleaves `(state, action, return)` in one stream**: 同一种"打破模态边界,让 self-attention 自由组合不同模态"的设计哲学。 / Same philosophy of "break modality boundaries, let one self-attention mix everything".

## 注意事项 / Caveats / when it breaks

- **`mask = torch.zeros(N, N, dtype=bool)` 显存 O(N²) / Memory is O(N²)**: 16 帧、 16×16 grid、 4 action token/帧 → N = 16·(4 + 256) ≈ 4160,N² ≈ 17M bool ≈ 17 MB,可以接受;但如果你做 64 帧 32×32,N² ≈ 4.7 G bool ≈ 4.7 GB,直接 OOM。生产级要用 FlexAttention 的 `BlockMask`(把 mask 表达成谓词,内部按 block 稀疏存储)。 / 16 frames × 16×16 grid × 4 actions ≈ 17 MB, OK. 64 frames × 32×32 grid ≈ 4.7 GB bool — instant OOM. Production must move to `FlexAttention.BlockMask`.
- **`add_tokens` 必须等于 `cond_tokens` / `add_tokens` must match `cond_tokens`**: forward 里 `cond_tokens = actions.shape[2]`(每帧 action 数量),mask 构造时传的 `add_tokens` 必须一致,否则切片错位。当 `use_extrinsics=True` 时多了一个 token,`cond_tokens += 1`——这是这段代码里最容易踩的 bug。 / If you set `use_extrinsics=True`, `cond_tokens` is +1 and so is `add_tokens`. Forget to bump one and the slices misalign.
- **同帧内允许双向 attention,可能泄露未来 / Intra-frame attention is bidirectional and can leak**: 这套设计假设"action token 在帧开始时已经被执行/知晓",所以 patch 看到 action 是合理的。但如果你在做"先有图像再决定 action"的场景(标准 imitation learning),需要在 intra-frame 也加 causal——action 必须 *在* patch token 后面、不能让 patch 看到 action。改 mask 的方法:把 `mask_block` 改成"action 后面的 patch 看不到 action 前面的 patch"那种细粒度。 / The intra-frame mask is fully bidirectional. If your training data is "image arrives, then action is predicted" (standard imitation), you may need a finer mask where patches don't peek at future actions in the same frame.
- **`predictor_proj` 投回 encoder embed_dim 之后再做 loss / The final projection back to `embed_dim`**: 这里跟 V-JEPA 的"target encoder"配合——loss 是在 `embed_dim` 空间(encoder 输出空间)做 L1 / L2,不在 predictor 内部空间。如果你两个空间维度填错,loss 算不通。 / The output is projected back to encoder space; the JEPA loss lives in that space, not in predictor space. Mismatched dims = silent shape error.

## 延伸阅读 / Further reading

- [lerobot PR #3568 — VLA-JEPA introduction](https://github.com/huggingface/lerobot/pull/3568)
- [Today's VLA note — the paired flow-matching action head](../vla/2026-06-15-vla-jepa-flow-matching-action-head.md)
- [V-JEPA paper (Meta AI, 2024)](https://arxiv.org/abs/2404.08471) — the target-encoder + predictor decomposition behind this WAM design.
- [PyTorch FlexAttention docs](https://pytorch.org/blog/flexattention/) — the production-friendly replacement for dense bool masks at large N.
