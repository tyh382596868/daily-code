---
date: 2026-05-29
topic: wam
source: wam
repo: Robbyant/lingbot-va
file: wan_va/modules/model.py
permalink: https://github.com/Robbyant/lingbot-va/blob/58c2ae5bac46bd8114065bea9d7d256eb67c16c3/wan_va/modules/model.py#L152-L201
difficulty: advanced
read_time: ~15 min
tags: [code-of-the-day, wam, flex-attention, block-mask, action-frame-fusion]
build_role: attention masking — how video latents and action latents share one attention stream
---

# lingbot-va 怎么用 7 个小函数拼出"视频+动作"的注意力 mask / lingbot-va composes seven mask predicates into one FlexAttention block-mask for video + action

> **一句话 / In one line**: lingbot-va 不写整张 `[N, N]` 的 attention mask，而是写 7 个返回 bool 的小 lambda（causal、self、window、clean-vs-noise……），用 `and_masks`/`or_masks` 拼成一个 BlockMask，再让 FlexAttention 在 flash-attn 速度下执行复杂的"视频+动作"联合注意力。 / Instead of materializing a `[N, N]` mask, lingbot-va writes seven tiny boolean predicates (causal, self, window, clean-vs-noise, …) and composes them with `and_masks`/`or_masks` into one FlexAttention BlockMask — flash-attention-fast even when the policy is "video latents + action latents in one diffusion target".

## 为什么重要 / Why this matters

WAM（World Action Model）训练里有个绕不开的问题：你想用同一个 transformer 同时处理视频 latents 和动作 latents，但两种 token 之间的"谁能看谁"远比 LLM 复杂 —— 干净的过去帧能被任何东西看，当前 noisy 帧只能看过去的干净帧但不能看当前的 noisy 帧自己，跨样本不能串扰，时间上还要加窗口约束。如果你用传统的 `additive_mask` 写法，一张 `[N, N]` 的 bool 矩阵 (N 通常 30 k+) 是上百 MB，根本塞不进 attention kernel。FlexAttention 给了一个数学家友好的写法：你只写"对 query q 和 key k，能不能看？"这样的小函数，PyTorch 帮你编译成 block-mask（block-sparse 的稀疏掩码），再让 flash kernel 跑。这段 50 行代码是这个组合范式最干净的工业实例。

WAMs run into a hairy attention question: you want one transformer to consume both video latents and action latents, but the "who can see whom" rules are much trickier than in an LLM. Clean past frames can be attended by anything, current noisy tokens can attend clean tokens up to (but not including) themselves, samples within the same batch must not bleed, and time-windowing is part of the recipe. A traditional `[N, N]` additive mask is hundreds of MB at typical WAM token counts and breaks the flash kernel. FlexAttention's API flips that: you write tiny predicates "for query position `q_idx` and key position `kv_idx`, can `q` attend `kv`?" and PyTorch compiles them into a block-sparse BlockMask that the flash kernel can consume. These 50 lines are the cleanest industrial example of that compose pattern.

## 代码 / The code

`Robbyant/lingbot-va` — [`wan_va/modules/model.py`](https://github.com/Robbyant/lingbot-va/blob/58c2ae5bac46bd8114065bea9d7d256eb67c16c3/wan_va/modules/model.py#L152-L201)

```python
@staticmethod
@torch.no_grad()
def _get_mask_mod(seq_ids, frame_ids, noise_ids, window_size):
    def seq_mask(b, h, q_idx, kv_idx):
        return (seq_ids[q_idx] == seq_ids[kv_idx]) & (seq_ids[q_idx] >= 0) & (seq_ids[kv_idx] >= 0)

    def block_causal_mask(b, h, q_idx, kv_idx):
        return frame_ids[kv_idx] <= frame_ids[q_idx]

    def block_causal_mask_exclude_self(b, h, q_idx, kv_idx):
        return frame_ids[kv_idx] < frame_ids[q_idx]

    def block_self_mask(b, h, q_idx, kv_idx):
        return frame_ids[kv_idx] == frame_ids[q_idx]

    def clean2clean_mask(b, h, q_idx, kv_idx):
        return (noise_ids[q_idx] == 1) & (noise_ids[kv_idx] == 1)

    def noise2clean_mask(b, h, q_idx, kv_idx):
        return (noise_ids[q_idx] == 0) & (noise_ids[kv_idx] == 1)

    def noise2noise_mask(b, h, q_idx, kv_idx):
        return (noise_ids[q_idx] == 0) & (noise_ids[kv_idx] == 0)

    def block_window_mask(b, h, q_idx, kv_idx, window_size):
        return (frame_ids[q_idx] - frame_ids[kv_idx]).abs() <= window_size

    mask_list = []
    mask_list.append(and_masks(clean2clean_mask, block_causal_mask))                 # clean→clean: causal across frames
    mask_list.append(and_masks(noise2clean_mask, block_causal_mask_exclude_self))    # noise→clean: only strictly past
    mask_list.append(and_masks(noise2noise_mask, block_self_mask))                   # noise→noise: only within the same frame
    mask = or_masks(*mask_list)
    mask = and_masks(mask, seq_mask)                                                 # no cross-sample leakage
    mask = and_masks(mask, partial(block_window_mask, window_size=window_size))      # temporal window
    return mask
```

## 逐行讲解 / What's happening

1. **三个"身份"张量 / Three identity tensors**:
   - 中文：`seq_ids`、`frame_ids`、`noise_ids` 是 `init_mask` 阶段构造的，长度都等于序列总长。`seq_ids[i]` 是 token i 属于 batch 里第几个样本；`frame_ids[i]` 是它属于第几帧；`noise_ids[i] = 0/1` 是它"干净"还是"加噪"。这三条向量决定了所有规则。
   - English: three tensors are precomputed by `init_mask` and indexed inside the predicates. `seq_ids[i]` says which batch element token `i` belongs to; `frame_ids[i]` says which time-step; `noise_ids[i] ∈ {0,1}` says whether the token is the clean half or the noisy half. The seven predicates are all `seq_ids`/`frame_ids`/`noise_ids` arithmetic.

2. **预测函数的签名 / Predicate signature `(b, h, q_idx, kv_idx)`**:
   - 中文：FlexAttention 要求每个 mask 函数接受 `(batch, head, q_idx, kv_idx)` 四个标量索引，返回一个 bool。框架会用 `torch.compile` 把这些函数编译成 GPU 内核，再调用 `create_block_mask` 把它们投射成 block-sparse mask。
   - English: FlexAttention's contract is that every predicate takes `(batch, head, q_idx, kv_idx)` and returns a `bool`. `torch.compile` turns them into GPU code, and `create_block_mask` evaluates them at block-corner positions to build a BlockMask — so the kernel can skip entire 128×128 blocks at runtime.

3. **三种 (clean, noise) 关系 / The three (clean, noise) interactions**:
   - 中文：
     - `clean2clean`：两个干净 token，规则是 **causal** —— 早帧能被晚帧看。
     - `noise2clean`：query 是 noisy，key 是 clean，规则是 **strictly causal** —— noisy 帧只能看严格早于它的 clean 帧，看自己同帧的 clean 都不行（因为 clean 是 ground truth，会泄漏目标）。
     - `noise2noise`：两个 noisy token，只允许在 **同一帧内** 互看 —— 因为同一帧内的 noisy token 是同一个噪声水平的同步去噪，需要互相看；跨帧的 noisy 互看会让 self-forcing 跑偏。
   - English:
     - `clean2clean` is **causal** — earlier clean frames can be attended by later clean frames.
     - `noise2clean` is **strictly causal** — a noisy query can attend clean keys only at strictly earlier frames; same-frame clean is forbidden because the clean half is the ground truth and would leak the target.
     - `noise2noise` is **self-only** — noisy tokens may attend other noisy tokens only within the same frame; cross-frame noisy-to-noisy bleeds noise across timesteps and destroys self-forcing.

4. **`or_masks` + `and_masks` 组合 / Compose with `or_masks` and `and_masks`**:
   - 中文：三条规则用 `or_masks` 拼起来 —— 任何一条满足就放行。然后整体再 `and_masks(seq_mask, ...)`：不管你前面怎么 or，跨样本永远禁；最后 `and_masks(block_window_mask, ...)`：再加一层时间窗约束。**逻辑组合就是命题逻辑** —— `or_masks` 是 ∨，`and_masks` 是 ∧。
   - English: the three predicates are unioned via `or_masks` — any one being true lets the attention through. The whole thing is then `and_masks`-ed with `seq_mask` (no cross-sample leakage, ever) and finally with the temporal `block_window_mask`. It really is just propositional logic: `or_masks` ≡ ∨, `and_masks` ≡ ∧.

5. **`seq_ids >= 0` 处理 padding / `seq_ids >= 0` handles padding**:
   - 中文：序列尾部的 padding 位置 `seq_ids = -1`、`frame_ids = -1`，因此 `seq_mask` 自动把它们 mask 掉。这就是为什么 `init_mask` 里用 `F.pad(..., value=-1)`。
   - English: padding positions are set to `-1` in all three identity tensors, so `seq_mask` drops them automatically. That is why `init_mask` pads with `value=-1` instead of `value=0`.

## 类比 / The analogy

像是一个剧组拍片的"谁能跟谁聊天"规则表：演员（clean token）可以跟所有更早场次的演员对台词，临时演员（noise token）只能听**已经拍好**的早场演员的台词，但**不许跟同场次的正式演员对**（剧透）；同场临时演员之间可以即兴对话（一起被去噪），但跨场不行。规则一多，与其每天发一张几百兆的"谁能跟谁聊"大表给现场，不如直接发"判断三件事"的小卡片 —— 这正是 FlexAttention 的范式。

Think of a film set's "who is allowed to talk to whom" policy. Lead actors (clean tokens) can rehearse with every earlier-scene lead. Extras (noise tokens) may listen to earlier-scene leads (already-shot footage) but never to a lead in the same scene (no spoilers). Same-scene extras can improv together (joint denoising at the same noise level), but never with extras from another scene. Instead of printing a 500 MB spreadsheet of allowed pairings every morning, the AD hands out a wallet card with three short rules — that wallet card is exactly the FlexAttention predicate-and-compose pattern.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-WAM

中文：在 nanoWAM 里这就是 `attention/mask.py` —— 整个 DiT 骨干每一层的 attention 在调 `flex_attention` 之前，都必须知道这张 BlockMask 长什么样。上游是数据加载：你把视频 latent 和 action latent 在通道上拼起来形成一个 `[B, F_total, C, H, W]` 的张量，patchify 后得到一长串 token；中间这一步就是 `init_mask` + `_get_mask_mod` 在 batch 开始时跑一次，产出 BlockMask 缓存；下游是每层 `WanAttention.forward(q, k, v, block_mask)` 直接用。如果你省掉这一层、改用 dense bool mask 或 `is_causal=True`，会出现：(1) clean 和 noise 没区分，noise 看到自己当帧的 clean（target leak），训练直接崩；(2) 跨样本可能漏到对方的 token 上，BatchNorm-like 行为；(3) 时间窗丢失，长视频内存爆炸。生产级实现需要补：基于 token 数自适应选择 `BLOCK_M`/`BLOCK_N`、跨 device mesh 同步 frame_ids（FSDP 下 token 顺序可能被打乱）、把 mask 缓存到 GPU 常量内存避免每个 forward 都重建。

English: in nanoWAM this is `attention/mask.py` — every DiT layer needs to know what BlockMask to hand `flex_attention`. Upstream is the dataloader: video latents and action latents are concatenated along the channel-time axis to form a `[B, F_total, C, H, W]` tensor and patchified into one flat token sequence. `init_mask` + `_get_mask_mod` runs once per batch to produce the cached BlockMask. Downstream, every `WanAttention.forward(q, k, v, block_mask)` simply consumes it. If you skipped this and used a dense bool mask or `is_causal=True`, three things break: (1) clean and noise tokens are not distinguished and the noisy query sees its own clean target — silent training collapse; (2) cross-sample bleed turns the model into a batch-correlated mess; (3) without the temporal window, long videos blow out the kernel's shared memory. A production WAM adds: token-count-aware `BLOCK_M`/`BLOCK_N` autotuning, frame-id synchronization across the device mesh (FSDP rearranges token order), and caching the BlockMask in GPU constant memory to amortise across forwards.

## 自己跑一遍 / Try it yourself

```python
# pip install torch
# requires PyTorch >= 2.5 for stable FlexAttention API
import torch
from functools import partial
from torch.nn.attention.flex_attention import create_block_mask, flex_attention, and_masks, or_masks

# Toy: B=1, F=4 frames, 2 tokens per frame. First 8 are clean past, next 8 are noisy current.
seq_ids   = torch.tensor([0]*16, dtype=torch.long, device="cuda")
frame_ids = torch.tensor([0,0, 1,1, 2,2, 3,3, 0,0, 1,1, 2,2, 3,3], device="cuda")
noise_ids = torch.tensor([1,1, 1,1, 1,1, 1,1, 0,0, 0,0, 0,0, 0,0], device="cuda")  # 1=clean, 0=noise

def seq_mask(b, h, q, k):           return (seq_ids[q] == seq_ids[k])
def block_causal(b, h, q, k):       return frame_ids[k] <= frame_ids[q]
def block_causal_excl(b, h, q, k):  return frame_ids[k] <  frame_ids[q]
def block_self(b, h, q, k):         return frame_ids[k] == frame_ids[q]
def c2c(b, h, q, k):                return (noise_ids[q] == 1) & (noise_ids[k] == 1)
def n2c(b, h, q, k):                return (noise_ids[q] == 0) & (noise_ids[k] == 1)
def n2n(b, h, q, k):                return (noise_ids[q] == 0) & (noise_ids[k] == 0)

mask = or_masks(and_masks(c2c, block_causal),
                and_masks(n2c, block_causal_excl),
                and_masks(n2n, block_self))
mask = and_masks(mask, seq_mask)
bm = create_block_mask(mask, 1, 1, 16, 16, device="cuda")
print(bm)                                    # textual block-sparse summary
print("density:", bm.to_dense().float().mean().item())
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
BlockMask(...)
density: 0.43...
```

中文：注意整张 16×16 mask 只有约 43% 是 True —— 这正是 FlexAttention 在大序列上能加速的来源：它会跳过整块全是 False 的 block，根本不算。

English: the dense mask is only ~43% True. That sparsity is exactly what FlexAttention exploits at scale — entirely-False blocks are skipped without entering the flash kernel at all.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **PyTorch FlexAttention 教程 / PyTorch FlexAttention tutorial**: 中文 — 官方教程里 sliding-window + causal 的组合也是 `and_masks`，但只用两条规则。 / English — the official tutorial composes `sliding_window` ∧ `causal`, the same idiom at minimal scale.
- **dreamzero 的 rectified-flow 调度器 / dreamzero's rectified-flow scheduler**: 中文 — 昨天的 wam 笔记里 dreamzero 的 scheduler 也假定有 "clean past + noisy current" 的输入划分，因为它就是为这类 attention mask 量身设计的。 / English — yesterday's dreamzero scheduler note shows the same "clean past + noisy current" split, because it is designed to feed exactly this kind of mask.
- **CausalForcing (今日 trending)** / **CausalForcing (today's trending)**: 中文 — 今日 trending 的 Causal-Forcing 也是 self-forcing 训练，KV cache 设计跟这里的 mask 严格对应：clean 半边是 KV，noisy 半边只查询。 / English — today's trending Causal-Forcing also runs self-forcing training, and its KV-cache split mirrors this mask exactly: clean half stored as KV, noisy half only as queries.
- **Wan2.1 (上游基座) / Wan2.1 (upstream foundation)**: 中文 — lingbot-va 继承自 Wan2.1，原始 Wan 的 attention 只做 spatial-temporal causal，lingbot-va 加上 noise-id 维度让它支持 action-frame fusion。 / English — lingbot-va is built on Wan2.1, which only handled spatial-temporal causal masking. lingbot-va extended the predicate set to handle action+frame fusion under self-forcing.

## 注意事项 / Caveats / when it breaks

- **mask 函数必须 traceable** / **Predicates must be torch-traceable**: 中文 — 函数内不能有 Python `if seq_ids[q] == 0:` 这种依赖具体值的分支，否则 `torch.compile` 会图断裂。所有逻辑必须用张量布尔运算表达（`&`、`|`、`==`、`<=`）。 / English — `torch.compile` traces these predicates, so any Python-value-dependent branching breaks the compile. Everything must be expressed in tensor boolean ops.
- **frame_ids 设计直接决定语义** / **`frame_ids` semantics matter**: 中文 — `init_mask` 里有一行 `frame_ids = [latent_frame_id // chunk_size * 2] * 2 + [action_frame_id // chunk_size * 2 + 1] * 2`，意思是 action 帧的"虚拟时间戳"是视频帧的两倍 + 1，让 action 总是夹在两个视频帧之间。改 chunk_size、改加 1 全会改语义。 / English — `init_mask` packs `frame_ids = [video_frame // chunk * 2] * 2 + [action_frame // chunk * 2 + 1] * 2`, encoding the convention that an action frame is sandwiched between two video frames at offset 2. Changing the `* 2 + 1` rule silently changes the model's temporal semantics.
- **window_size 太小会丢长程信号** / **Small `window_size` drops long-range context**: 中文 — 默认 window 是几帧，对短任务足够，但长 horizon manipulation 需要更大窗，相应地 mask 密度变高、速度变慢。 / English — the default window covers only a few frames; long-horizon manipulation needs a larger window, which increases mask density and slows the kernel proportionally.

## 延伸阅读 / Further reading

- [PyTorch FlexAttention blog post](https://pytorch.org/blog/flexattention/)
- [Wan2.1 — the video diffusion base model lingbot-va extends](https://github.com/Wan-Video/Wan2.1)
- [Self-Forcing for autoregressive video diffusion](https://arxiv.org/abs/2503.20451)
- [LingBot-VA paper (PDF in repo root)](https://github.com/Robbyant/lingbot-va/blob/main/LingBot_VA_paper.pdf)
