---
date: 2026-06-23
topic: wam
source: wam
repo: yuantianyuan01/FastWAM
file: src/fastwam/models/wan22/wan_video_dit.py
permalink: https://github.com/yuantianyuan01/FastWAM/blob/45d8e14589219c0cdd1e1f7bbebe34f2e2f41879/src/fastwam/models/wan22/wan_video_dit.py#L64-L147
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, wam, attention-mask, causal, video-dit, broadcasting, fastwam]
build_role: dit-block — the attention mask factory for video-frame-grouped causal or diagonal attention in the DiT backbone
---

# 8 行广播构建视频帧分组因果注意力掩码 / Build Video-Frame-Grouped Causal Attention Masks in 8 Lines of Broadcasting

> **一句话 / In one line**: `create_group_causal_attn_mask` 用 `repeat_interleave` 给每个 token 赋时间帧索引，再用外积比较 `>=` 或 `==` 来生成因果或帧内对角注意力掩码，整个逻辑只有 8 行。 / `create_group_causal_attn_mask` assigns a frame index to every token via `repeat_interleave`, then computes an outer-product comparison (`>=` or `==`) to produce causal or within-frame diagonal attention masks — the entire logic is 8 lines.

## 为什么重要 / Why this matters

在世界动作模型（WAM）里，视频序列被切成一帧帧的 latent token 组。注意力掩码决定了帧间的信息流：因果掩码让每帧只能看到自身及所有过去帧（用于 autoregressive 生成），对角掩码让每帧只能看到自身（用于帧内独立去噪）。FastWAM 把两种模式统一到一个函数里，用 PyTorch 广播的外积比较来生成，零循环、零 if-else 嵌套。这个函数在 WAM 的 DiT backbone 里被复用于视频 latent token 和动作 token，也支持通过逻辑 AND 组合成混合掩码（视频因果 + 动作帧内）——这是一个理解"组结构注意力掩码"设计的极佳案例。

In a World Action Model (WAM), a video sequence is split into groups of latent tokens per frame. The attention mask controls inter-frame information flow: causal masks let each frame attend to itself and all past frames (for autoregressive generation), diagonal masks restrict attention to within the same frame (for per-frame independent denoising). FastWAM unifies both modes in one function using PyTorch broadcasting outer-product comparisons — no loops, no nested if-else. This function is reused in the WAM DiT backbone for both video latent tokens and action tokens, and can be combined via logical AND to build mixed masks (video-causal + action-diagonal). It is a near-perfect example of grouped-structure attention mask design.

## 代码 / The code

`yuantianyuan01/FastWAM` — [`src/fastwam/models/wan22/wan_video_dit.py`](https://github.com/yuantianyuan01/FastWAM/blob/45d8e14589219c0cdd1e1f7bbebe34f2e2f41879/src/fastwam/models/wan22/wan_video_dit.py#L64-L147)

```python
def create_group_causal_attn_mask(
    num_temporal_groups: int,
    num_query_per_group: int,
    num_key_per_group: int,
    mode: str = "causal",
) -> torch.Tensor:
    """
    Build a (L, S) boolean attention mask for video-frame-grouped sequences.

    Args:
        num_temporal_groups:  number of video frames (time steps)
        num_query_per_group:  tokens per frame on the query side
        num_key_per_group:    tokens per frame on the key side
        mode:  "causal"         — attend to current frame and all past frames
               "group_diagonal" — attend only within the same frame

    Returns:
        attn_mask: BoolTensor of shape (L, S), True = allowed to attend
                   L = num_temporal_groups * num_query_per_group
                   S = num_temporal_groups * num_key_per_group
    """
    # assign a time-step index to every query token
    query_time_indices = torch.arange(num_temporal_groups).repeat_interleave(
        num_query_per_group
    )  # shape: (L,)

    # assign a time-step index to every key token
    key_time_indices = torch.arange(num_temporal_groups).repeat_interleave(
        num_key_per_group
    )  # shape: (S,)

    # broadcasting outer product comparison
    query_time_indices = query_time_indices.unsqueeze(1)   # (L, 1)
    key_time_indices   = key_time_indices.unsqueeze(0)     # (1, S)

    if mode == "causal":
        attn_mask = query_time_indices >= key_time_indices  # (L, S)
    elif mode == "group_diagonal":
        attn_mask = query_time_indices == key_time_indices  # (L, S)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return attn_mask   # BoolTensor (L, S)
```

## 逐行讲解 / What's happening

1. **`torch.arange(num_temporal_groups).repeat_interleave(num_query_per_group)` — 分配时间索引 / assign time indices**:
   - 中文: `arange(T)` 生成 `[0, 1, 2, ..., T-1]`，`repeat_interleave(Q)` 把每个元素重复 Q 次得到 `[0,0,...,0, 1,1,...,1, ..., T-1,...,T-1]`。这就是序列中每个 token 对应的"帧号"。对 key 侧做同样操作（可以有不同的每帧 token 数 `num_key_per_group`）。
   - English: `arange(T)` gives `[0,1,...,T-1]`, `repeat_interleave(Q)` repeats each element Q times to produce `[0,0,...,0, 1,1,...,1, ..., T-1,...,T-1]` — the frame index of every token in the sequence. The same is done for the key side (potentially with a different per-frame count `num_key_per_group`).

2. **`unsqueeze(1)` / `unsqueeze(0)` — 为广播做准备 / preparing for broadcasting**:
   - 中文: `(L, 1)` 和 `(1, S)` 在比较时通过广播自动扩展为 `(L, S)`，等价于外积。这是一个避免两层 for 循环的经典技巧。
   - English: Shapes `(L, 1)` and `(1, S)` broadcast to `(L, S)` under comparison — this is the outer-product trick that avoids two nested for-loops and keeps the function a pure tensor operation.

3. **`mode == "causal"` — `>=` 比较 / causal mask**:
   - 中文: `query_frame >= key_frame` 表示 query token 可以 attend 到同帧及所有过去帧的 key token。这是标准的因果掩码，但以帧为粒度而非以 token 为粒度——同一帧内的所有 query token 都能看到同一帧内的所有 key token（而非只能看到自身之前的 token）。
   - English: `query_frame >= key_frame` means a query token can attend to all key tokens in the same frame and all past frames. This is causal at the **frame** granularity, not the token granularity — within a frame, all query tokens can see all key tokens in that frame.

4. **`mode == "group_diagonal"` — `==` 比较 / diagonal mask**:
   - 中文: `query_frame == key_frame` 表示每个 token 只能 attend 到同一帧内的 token，完全阻断跨帧信息流。这适用于"帧独立去噪"的场景——比如在 world model rollout 的每帧内部做 DiT 推理时，不需要（也不应该）看其他帧。
   - English: `query_frame == key_frame` allows attention only within the same frame, blocking all cross-frame information. This is appropriate for per-frame independent denoising — e.g. during world-model rollout within a single frame, you don't want (and shouldn't have) cross-frame attention.

5. **复用与组合 / reuse and composition**:
   - 中文: 同一函数可以用不同参数生成两个掩码，然后用 `mask_video & mask_action` 组合，构建"视频 latent 用因果 + 动作 token 用帧内对角"的混合掩码——FastWAM 训练时正是这样使用的。
   - English: Call the function twice with different arguments to get two masks, then combine them via `mask_video & mask_action` to build a "video-causal + action-diagonal" mixed mask — exactly how FastWAM uses it during training.

## 类比 / The analogy

想象一个会议室排座图，行（query）和列（key）都是与会者。"因果模式"相当于"你可以交谈的规则是：只能找座位编号不超过自己的人说话（即当前帧及过去帧）"。"对角模式"相当于"只能找坐在同一桌的人说话（同帧）"。用时间帧号做外积比较，就是把这两条规则翻译成一个可以直接贴到 attention 上的布尔座位图。

Picture a seating chart in a conference room, with rows as speakers (queries) and columns as listeners (keys). "Causal mode" is the rule "you may only talk to people seated in rows numbered no higher than yours" (current and past frames). "Diagonal mode" is "you may only talk to people at the same table as you" (same frame). The frame-index outer-product comparison is just this rule table translated into a boolean seating chart you can paste directly onto the attention layer.

## 在 nanoWAM 中的位置 / Where this lives in your nanoWAM

> **curriculum item**: `dit-block` — the DiT backbone block's attention mask factory. Previous coverage: `nano/wam/2026-06-05-isaac-groot-dit-cross-attn.md` (cross-attention conditioning in a DiT block) and `nano/wam/2026-06-03-fastwam-action-prefill-cache.md` (prefill KV cache in the same FastWAM codebase). This note focuses on the attention mask factory specifically.

中文：在你自己的 nanoWAM 里，这个函数是 DiT backbone 的**注意力掩码工厂**。在模型初始化阶段调用一次，生成训练期间不变的固定掩码（因果模式），存为 buffer；在 world model rollout 推理时，如果每步帧数固定，也可以预先生成并缓存。上游依赖：视频 VAE 编码器（`vae-encoder-decoder`）决定每帧的 latent token 数（即 `num_key_per_group`）；下游：DiT block 的自注意力层接收这个掩码，控制跨帧信息流，进而影响生成的时间一致性。如果省掉这个掩码（改用全注意力），模型在训练时会看到未来帧，导致测试时的因果错误。

English: In your nanoWAM, this function is the **attention mask factory** for the DiT backbone. Call it once at model init to produce the fixed training mask (causal mode) and register it as a buffer. At world-model rollout inference, if the frame count is fixed, pre-compute and cache the mask. Upstream dependency: the video VAE encoder (`vae-encoder-decoder`) determines the number of latent tokens per frame (`num_key_per_group`). Downstream: the DiT block's self-attention layer receives this mask and uses it to control cross-frame information flow, which directly affects temporal coherence of generated video. Omitting the mask (using full attention) would let the model attend to future frames during training, causing causal errors at test time.

## 自己跑一遍 / Try it yourself

```python
import torch

def create_group_causal_attn_mask(T, Q, K, mode="causal"):
    qi = torch.arange(T).repeat_interleave(Q).unsqueeze(1)  # (T*Q, 1)
    ki = torch.arange(T).repeat_interleave(K).unsqueeze(0)  # (1, T*K)
    if mode == "causal":         return qi >= ki
    if mode == "group_diagonal": return qi == ki
    raise ValueError(mode)

T, Q, K = 4, 3, 3  # 4 frames, 3 latent tokens per frame
m_causal = create_group_causal_attn_mask(T, Q, K, "causal")
m_diag   = create_group_causal_attn_mask(T, Q, K, "group_diagonal")

print("causal mask (T=4, Q=K=3):")
print(m_causal.int())
print("\ndiagonal mask:")
print(m_diag.int())

# combined: video causal + action diagonal
# (imagine first T*Q rows are video tokens, last Ta*Qa rows are action tokens)
Ta, Qa, Ka = 4, 2, 2
m_action = create_group_causal_attn_mask(Ta, Qa, Ka, "group_diagonal")
print("\naction diagonal mask (T=4, Q=K=2):")
print(m_action.int())
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
causal mask (T=4, Q=K=3):
tensor([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ...
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
diagonal mask:
tensor([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ...])
```

中文：仔细看因果掩码的最后三行——它们全为 1，因为最后一帧的 query token 可以看到所有 key token（T-1 >= 所有帧索引）。对角掩码里每个 3×3 的对角块是全 1，其余全 0。

English: Observe that the last three rows of the causal mask are all-ones — the last frame's query tokens can attend to every key token (T-1 ≥ all frame indices). In the diagonal mask, each 3×3 block on the diagonal is all-ones; everything else is zero.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Wan2.1 video attention** / **Wan2.1 视频注意力**: Wan2.1 的视频 DiT 也用类似的分组因果掩码，但它是在 3D RoPE 坐标里隐式编码的，而非显式 boolean mask。 / Wan2.1's video DiT also uses grouped causal attention, but encodes it implicitly in 3D RoPE coordinates rather than as an explicit boolean mask.
- **Open-Sora causal 3D attention** / **Open-Sora 因果 3D 注意力**: Open-Sora 的 causal 3D VAE (`nano/wam/2026-06-11`) 用了帧维度的因果卷积——和这里的注意力掩码解决同一个"时间因果性"问题，但在不同的模块层次。 / Open-Sora's causal 3D VAE (`nano/wam/2026-06-11`) uses causal convolutions along the frame dimension — solving the same temporal causality problem as this attention mask, but at a different module level.
- **`torch.tril` 因果掩码** / **`torch.tril` causal mask**: 经典 LLM 因果掩码是 `torch.tril(torch.ones(L, L))` — 这是 token-level 因果性，是本文 frame-level 因果性的特例（每帧 Q=K=1）。 / Classic LLM causal mask is `torch.tril(torch.ones(L, L))` — that is token-level causality, a special case of this frame-level mask with Q=K=1 per frame.

## 注意事项 / Caveats / when it breaks

- **内存随帧数二次增长** / **memory scales quadratically with frame count**: The mask is `(T×Q) × (T×K)` booleans. For T=64 frames, Q=K=256 tokens/frame, that's 64×256=16384 — a 16384×16384 boolean mask ≈ 256 MB. Consider registering it as a buffer and using `torch.backends.cuda.enable_flash_sdp` with explicit `attn_mask` only when the sequence length is manageable.
- **查询和键的每帧 token 数可以不同** / **Q and K per group can differ**: The function correctly supports `num_query_per_group ≠ num_key_per_group` (e.g. query side is downsampled). Ensure the caller passes the right values for each side.
- **组合掩码时维度对齐** / **dimension alignment when combining masks**: When combining video and action masks into a joint mask, you need to pad with zeros/ones appropriately for cross-video-to-action and cross-action-to-video attention — this bookkeeping is the caller's responsibility.

## 延伸阅读 / Further reading

- [FastWAM `wan_video_dit.py`](https://github.com/yuantianyuan01/FastWAM/blob/45d8e14589219c0cdd1e1f7bbebe34f2e2f41879/src/fastwam/models/wan22/wan_video_dit.py)
- [Wan2.1 technical report](https://github.com/Wan-Video/Wan2.1) — grouped video attention in production
- [Flash Attention masked attention](https://github.com/Dao-AILab/flash-attention) — efficient implementation of masked self-attention for long sequences
