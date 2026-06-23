---
date: 2026-06-23
topic: robotics
source: trending
repo: shihao1895/MemoryVLA
file: vla/memory_vla.py
permalink: https://github.com/shihao1895/MemoryVLA/blob/d732ea9072bc4d9fce95e0c6060db50caca89ae3/vla/memory_vla.py#L158-L335
difficulty: advanced
read_time: ~15 min
tags: [code-of-the-day, robotics, memory, token-merging, cross-attention, vla, iclr2026]
---

# MemoryVLA CogMemBank：ICLR 2026 认知记忆库 — 跨时间 Transformer 检索 + Gate 融合 + ToMe 整合 / MemoryVLA CogMemBank: ICLR 2026 Cognitive Memory Bank — Cross-Transformer Retrieval + GateFusion + ToMe Consolidation

> **一句话 / In one line**: CogMemBank 实现了 VLA 的持久化情节记忆：每个时间步做四步循环——情节追踪清空旧记忆、跨时间 Transformer 检索历史、GateFusion 门控融合、Token Merging 整合压缩——让机器人能"记住" 30 步前看到的场景。 / CogMemBank implements persistent episodic memory for VLAs: each timestep runs a four-step loop — episode tracking to clear stale memory, cross-time Transformer retrieval, GateFusion blending, and Token Merging consolidation — letting the robot "recall" observations from 30 steps ago.

## 为什么重要 / Why this matters

绝大多数 VLA（包括 OpenVLA、pi0、GR00T）的感知窗口只有当前帧或极短的历史，面对"先拿红盒子再拿蓝盒子"这类需要长时记忆的任务束手无策。MemoryVLA（ICLR 2026，286 stars）在 VLA 主干旁挂了一个持久的情节记忆库，每个时间步存入当前帧特征，检索时通过 CrossTransformerBlock 做跨时间注意力，用 GateFusion 的 sigmoid 门控融合检索结果和当前表示。当记忆库满时，不是简单的 FIFO，而是用 Token Merging（ToMe）找到余弦相似度最高的相邻条目并平均，保留语义丰富的帧、丢弃冗余帧。整套流程每步只有 ~10-15 行，逻辑非常清晰。

Most VLAs (including OpenVLA, pi0, GR00T) have a perceptual window of only the current frame or a very short history, making tasks that require long-range memory (e.g. "pick up the red box, then the blue box") impossible. MemoryVLA (ICLR 2026, 286 stars) adds a persistent episodic memory bank alongside the VLA backbone. Each timestep stores the current frame features; retrieval runs CrossTransformerBlock cross-attention across stored history; GateFusion's sigmoid gate blends retrieved memory with the current representation. When the bank is full, instead of simple FIFO eviction, Token Merging (ToMe) finds the two most cosine-similar consecutive entries and averages them, preserving semantically rich frames and discarding redundant ones. The full pipeline is ~10–15 lines per step.

## 代码 / The code

`shihao1895/MemoryVLA` — [`vla/memory_vla.py`](https://github.com/shihao1895/MemoryVLA/blob/d732ea9072bc4d9fce95e0c6060db50caca89ae3/vla/memory_vla.py#L158-L335)

```python
class GateFusion(nn.Module):
    """Sigmoid-gated blend of two feature tensors."""
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim * 2, dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # x1, x2: (N, D)
        scale = torch.sigmoid(self.proj(torch.cat([x1, x2], dim=-1)))  # (N, D)
        return scale * x1 + (1.0 - scale) * x2


class CogMemBank(nn.Module):
    """Cognitive memory bank with cross-transformer retrieval and ToMe consolidation.

    Stores per-episode history as a list of (timestep, feature) tuples.
    bank: dict[episode_id -> list[(timestep, Tensor[N, D])]]
    """
    def __init__(self, dim: int, num_heads: int, max_bank_size: int = 16,
                 use_tome: bool = True, use_timestep_pe: bool = True):
        super().__init__()
        self.max_bank_size    = max_bank_size
        self.use_tome         = use_tome
        self.cross_attn_block = CrossTransformerBlock(dim, num_heads)
        self.gate_fusion      = GateFusion(dim)
        self.timestep_pe      = nn.Embedding(1024, dim) if use_timestep_pe else None
        self.bank: dict       = {}

    # ------------------------------------------------------------------ #
    # Step 4 helper: Token Merging consolidation                          #
    # ------------------------------------------------------------------ #
    def _consolidate_with_token_merge(self, episode_id: int) -> None:
        entries = self.bank[episode_id]       # list of (timestep, feat)
        T = len(entries)
        if T < 2:
            return
        # compute cosine similarity between consecutive entries
        feats = [e[1] for e in entries]       # list of Tensor[N, D]
        sims  = []
        for i in range(T - 1):
            f1, f2 = feats[i].float(), feats[i + 1].float()
            cos_sim = F.cosine_similarity(f1, f2, dim=-1).mean().item()
            sims.append(cos_sim)
        # merge the most similar consecutive pair
        idx = int(torch.tensor(sims).argmax().item())
        t_merged = entries[idx][0]            # keep earlier timestep
        f_merged = (feats[idx] + feats[idx + 1]) * 0.5
        self.bank[episode_id] = entries[:idx] + [(t_merged, f_merged)] + entries[idx + 2:]

    # ------------------------------------------------------------------ #
    # Main per-timestep loop                                              #
    # ------------------------------------------------------------------ #
    def process_batch(
        self,
        tokens: torch.Tensor,        # (B, N, D) — current-step working memory
        episode_ids: list[int],      # (B,) — which episode each sample belongs to
        timesteps: list[int],        # (B,) — current timestep per sample
    ) -> torch.Tensor:
        B, N, D = tokens.shape
        outputs = []

        for i in range(B):
            ep_id     = episode_ids[i]
            t_step    = timesteps[i]
            working   = tokens[i]    # (N, D)

            # Step 1: episode tracking — clear bank on new episode
            if ep_id not in self.bank or (
                len(self.bank.get(ep_id, [])) > 0
                and t_step <= self.bank[ep_id][-1][0]   # timestep reset = new episode
            ):
                self.bank[ep_id] = []

            # Step 2: memory retrieval
            if len(self.bank[ep_id]) == 0:
                # no history yet — skip retrieval, use working memory as-is
                fused = working
            else:
                # build key/value from all stored history tokens + timestep PE
                mem_feats = []
                for (mem_t, mem_feat) in self.bank[ep_id]:
                    if self.timestep_pe is not None:
                        pe = self.timestep_pe(
                            torch.tensor(mem_t, device=working.device).clamp(max=1023)
                        ).unsqueeze(0)                  # (1, D)
                        mem_feats.append(mem_feat + pe) # (N, D) broadcast
                    else:
                        mem_feats.append(mem_feat)
                memory_kv = torch.cat(mem_feats, dim=0).unsqueeze(0)  # (1, T*N, D)
                query      = working.unsqueeze(0)                      # (1, N, D)

                # cross-attention: working memory queries over episodic history
                retrieved = self.cross_attn_block(query, memory_kv).squeeze(0)  # (N, D)

                # Step 3: adaptive fusion — gate-blend retrieved with working
                fused = self.gate_fusion(working, retrieved)   # (N, D)

            outputs.append(fused)

            # Step 4: memory consolidation — store current frame, then compress
            self.bank[ep_id].append((t_step, working.detach()))
            if len(self.bank[ep_id]) > self.max_bank_size:
                if self.use_tome:
                    self._consolidate_with_token_merge(ep_id)
                else:
                    self.bank[ep_id].pop(0)   # FIFO fallback

        return torch.stack(outputs, dim=0)    # (B, N, D)
```

## 逐行讲解 / What's happening

1. **`GateFusion` — sigmoid 门控融合 / sigmoid gate fusion**:
   - 中文: 把 `x1`（当前表示）和 `x2`（检索到的历史）在最后一维拼接，过一个线性层再 sigmoid，得到每个维度的混合权重 `scale`。最终输出是 `scale * x1 + (1-scale) * x2`——当 `scale≈1` 时完全信任当前帧，当 `scale≈0` 时完全使用历史记忆。
   - English: Concatenates `x1` (current) and `x2` (retrieved history) on the last axis, passes through a linear layer and sigmoid to get per-dimension mixing weights `scale`. Output is `scale * x1 + (1-scale) * x2` — when `scale≈1`, the model trusts current observations; when `scale≈0`, it relies on historical memory.

2. **Step 1 — 情节追踪 / episode tracking**:
   - 中文: 记忆库以 `episode_id` 为 key 存储历史。当检测到时间步重置（`t_step <= last_stored_timestep`，说明开始了新的情节）或遇到未知 episode 时，清空该情节的记忆。这保证了不同情节间的隔离。
   - English: The bank is keyed by `episode_id`. When a timestep reset is detected (`t_step ≤ last_stored_timestep`, indicating a new episode) or an unknown episode appears, the bank for that episode is cleared. This ensures clean isolation between episodes.

3. **Step 2 — 记忆检索 / memory retrieval**:
   - 中文: 历史中每条记录的特征加上 timestep PE（让模型知道该帧是多少步之前的），然后把所有历史特征在序列维度拼接成 `memory_kv`，当前 `working` 作为 query，执行 `CrossTransformerBlock` 的跨时间注意力。这让当前帧能"问"历史帧"你们有没有见过相关的场景？"
   - English: Each stored feature is summed with its timestep positional encoding (so the model knows how far back each memory is), then all history features are concatenated along the sequence axis to form `memory_kv`. The current `working` tokens serve as queries in `CrossTransformerBlock` cross-attention — letting the current frame "ask" historical frames "have you seen something relevant?"

4. **Step 3 — GateFusion / adaptive fusion**:
   - 中文: 用 `GateFusion` 融合当前表示和检索结果。注意这里不是简单相加，而是有学习参数的门控——当历史帮助不大（比如情节开始时记忆为空），门自然偏向当前帧；当历史高度相关，门偏向历史。
   - English: `GateFusion` blends the current working memory with the retrieved result. Unlike simple addition, the gate has learned parameters — when history is unhelpful (e.g. at episode start with an empty bank), the gate naturally weights the current frame higher; when history is highly relevant, it weights the retrieved memory higher.

5. **Step 4 — Token Merging 整合 / ToMe consolidation**:
   - 中文: 存入当前帧后，若记忆库超过 `max_bank_size`，不是简单弹出最老的条目（FIFO），而是找相邻条目中余弦相似度最高的一对，用平均特征替换它们。这相当于保留了语义丰富的关键帧，压缩了冗余的静止帧（比如机器人等待时连续多帧几乎相同）。
   - English: After storing the current frame, if the bank exceeds `max_bank_size`, instead of FIFO eviction it finds the most cosine-similar consecutive pair and replaces them with their average. This effectively retains semantically rich keyframes and compresses redundant near-identical frames (e.g. consecutive still frames while the robot waits).

## 类比 / The analogy

把 CogMemBank 想象成一个摄影师的工作日记本，每天（时间步）贴一张照片（特征）。日记本只有 16 页（`max_bank_size`），但摄影师不会简单撕掉最早的一页——他会找两张看起来最像的相邻照片合成一张，腾出空间给新照片。每次查阅日记时（检索），他先翻阅所有照片，通过"联想门"（GateFusion）决定现在的状况该多信任当天的感受（当前帧）还是日记里记录的历史。

Think of CogMemBank as a photographer's work diary, where a new photo (feature) is pasted every day (timestep). The diary has only 16 pages (`max_bank_size`), but instead of simply tearing out the oldest page, the photographer finds the two most similar adjacent photos and composites them into one, freeing space for the new photo. When reviewing the diary (retrieval), he browses all photos and uses an "association gate" (GateFusion) to decide how much to trust today's observations versus what the diary records from the past.

## 自己跑一遍 / Try it yourself

```python
import torch, torch.nn as nn, torch.nn.functional as F

class GateFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim*2, dim)
    def forward(self, x1, x2):
        s = torch.sigmoid(self.proj(torch.cat([x1, x2], -1)))
        return s*x1 + (1-s)*x2

class TinyMemBank:
    def __init__(self, max_size=4):
        self.max_size = max_size
        self.bank = {}
        self.gate = GateFusion(8)

    def _tome(self, ep):
        feats = [e[1] for e in self.bank[ep]]
        sims  = [F.cosine_similarity(feats[i].float(), feats[i+1].float(), dim=-1).mean() for i in range(len(feats)-1)]
        idx   = int(torch.stack(sims).argmax())
        merged = (feats[idx] + feats[idx+1]) * 0.5
        self.bank[ep] = self.bank[ep][:idx] + [(self.bank[ep][idx][0], merged)] + self.bank[ep][idx+2:]

    def step(self, feat, ep_id, t_step):
        if ep_id not in self.bank: self.bank[ep_id] = []
        if self.bank[ep_id] and t_step <= self.bank[ep_id][-1][0]:
            self.bank[ep_id] = []  # new episode
        history = self.bank[ep_id]
        if history:
            mem = torch.stack([e[1] for e in history]).mean(0)  # simple mean retrieval
            fused = self.gate(feat, mem)
        else:
            fused = feat
        self.bank[ep_id].append((t_step, feat.detach()))
        if len(self.bank[ep_id]) > self.max_size:
            self._tome(ep_id)
        return fused

bank = TinyMemBank(max_size=4)
for t in range(8):
    feat = torch.randn(8)
    out  = bank.step(feat, ep_id=0, t_step=t)
    print(f"t={t}: bank_size={len(bank.bank[0])}, out_norm={out.norm().item():.3f}")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
t=0: bank_size=1, out_norm=...
t=1: bank_size=2, ...
t=2: bank_size=3, ...
t=3: bank_size=4, ...
t=4: bank_size=4, ...   ← ToMe kicks in, size stays ≤ 4 from here
t=5: bank_size=4, ...
t=6: bank_size=4, ...
t=7: bank_size=4, ...
```

中文：注意从 t=4 开始记忆库大小稳定在 4，每次新存入后都触发一次 ToMe 整合。如果把 `_tome` 换成简单的 `pop(0)`（FIFO），输出的 `out_norm` 会有细微差异——因为 FIFO 丢弃最旧的帧，而 ToMe 保留的是"最不相似"的帧组合，保留了更多信息。

English: Note that bank size stabilizes at 4 from t=4 onward — each new insertion triggers a ToMe consolidation. Replace `_tome` with `pop(0)` (FIFO) and the `out_norm` values shift slightly — FIFO drops the oldest frame, while ToMe retains the "least-similar" frame combination, preserving more information.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Token Merging (ToMe, Bolya et al. 2022)** / **Token Merging 原论文**: ToMe 原本用于 ViT 推理加速，通过合并相似 token 减少序列长度。MemoryVLA 把它挪用于记忆整合，在时间维度上做"最相似帧合并"。 / ToMe was originally proposed for ViT inference speed-up by merging similar tokens. MemoryVLA repurposes it for memory consolidation, merging most-similar frames in the time dimension.
- **DexBotics MemVLA** / **DexBotics MemVLA** (`2026/06/2026-06-09-dexbotic-memvla-token-merge-memory.md`): 这个 repo 在 2026-06-09 的 daily-code 里也用了 ToMe 做记忆整合，但结合了 short-term video memory，设计细节不同。 / This repo also used ToMe for memory consolidation in the 2026-06-09 daily-code, but combined with short-term video memory with different design details.
- **Perceiver AR** / **Perceiver AR**: Perceiver AR 用一个固定大小的 latent array 作为长时序记忆，通过 cross-attention 与当前输入交互——和 CogMemBank 的"固定大小 bank + 跨注意力检索"在结构上高度相似。 / Perceiver AR uses a fixed-size latent array as long-range memory and interacts with current input via cross-attention — structurally very similar to CogMemBank's "fixed-size bank + cross-attention retrieval."

## 注意事项 / Caveats / when it breaks

- **`.detach()` 阻断了梯度** / **`.detach()` cuts gradient flow**: `self.bank[ep_id].append((t_step, working.detach()))` stores detached tensors — memory bank entries do NOT propagate gradients back to the encoder. This is intentional to avoid backprop through unbounded history, but means the encoder is not trained to produce "memorable" features via the memory path.
- **字典记忆库不适合大 batch** / **dict bank does not scale to large batches**: The per-episode dict is a Python-level structure. For large parallel environments or vectorized simulators, consider a tensor-based circular buffer instead.
- **ToMe 整合是近似的** / **ToMe consolidation is approximate**: Averaging two feature vectors discards information, especially if the two frames have meaningful differences the cosine similarity misses (e.g. two frames with similar global statistics but different local keypoints). For high-precision tasks this can degrade recall quality.

## 延伸阅读 / Further reading

- [MemoryVLA paper (ICLR 2026)](https://github.com/shihao1895/MemoryVLA)
- [Token Merging: Your ViT But Faster (Bolya et al. 2022)](https://arxiv.org/abs/2210.09461)
- [Perceiver AR](https://arxiv.org/abs/2202.07765) — fixed-size latent memory with cross-attention
