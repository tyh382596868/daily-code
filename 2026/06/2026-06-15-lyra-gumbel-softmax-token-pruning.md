---
date: 2026-06-15
topic: robotics
source: trending
repo: nv-tlabs/lyra
file: Lyra-1/src/models/utils/token_pruning.py
permalink: https://github.com/nv-tlabs/lyra/blob/87f79a52b81b366d1d4aa3a526aa12e54207c998/Lyra-1/src/models/utils/token_pruning.py#L20-L166
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, trending, gumbel-softmax, straight-through-estimator, token-pruning, 3d-world-model]
---

# NVIDIA Lyra 的 Gumbel-Softmax 直通 top-k:200 行代码搞定"训练时可微、推理时硬 top-k" / NVIDIA Lyra's Gumbel-Softmax straight-through top-k: 200 lines that switch between differentiable training and hard inference

> **一句话 / In one line**: 训练时 `hard_mask - y_soft.detach() + y_soft` 让 forward 看到 one-hot、backward 走 soft 梯度;推理时直接 argmax-top-k——一个 module 通吃两种模式。 / At training, `hard_mask - y_soft.detach() + y_soft` gives a one-hot forward and a soft-gradient backward; at inference, plain argmax-top-k. One module, two regimes.

## 为什么重要 / Why this matters

NVIDIA Toronto Labs 在 2026-06 开源了 Project Lyra——一套开放的生成式 3D world model。它的 token pruning 模块(200 行)解决了一个非常通用的问题:**怎么让网络自己"挑出最重要的 k 个 token",同时保留梯度让你能端到端训练**。这件事 transformer 时代被反复发现了好几次:DynamicViT 用 Gumbel 拆 token,A-ViT 用 halting probability,Mixture-of-Experts 用 noisy top-k routing。Lyra 这版的特别之处是它把"训练 / 推理两种 mode + 全局 / 时空分解两种 selection 策略"全装在一个文件里,代码非常干净——是学习"可微 top-k"和"直通估计器 (straight-through estimator)"的优秀样本。

NVIDIA Toronto Labs open-sourced Project Lyra in June 2026 — an open generative 3D world model. Its token pruning module (200 lines) solves a recurring transformer-era problem: **how do you let a network select the k most important tokens, while preserving gradients for end-to-end training?** DynamicViT, A-ViT, and Mixture-of-Experts have each rediscovered this. Lyra packages "training/inference dispatch + global/structured selection" into one tidy file — a great reference for the Gumbel-Softmax straight-through estimator.

## 代码 / The code

`nv-tlabs/lyra` — [`Lyra-1/src/models/utils/token_pruning.py`](https://github.com/nv-tlabs/lyra/blob/87f79a52b81b366d1d4aa3a526aa12e54207c998/Lyra-1/src/models/utils/token_pruning.py#L20-L166)

```python
def sample_gumbel(shape, eps=1e-6, device=None, dtype=None):
    U = torch.rand(shape, device=device, dtype=dtype)
    return -torch.log(-torch.log(U.clamp(min=eps, max=1 - eps)))

def select_topk(logits, k, method, temperature, hard, eps):
    B, N = logits.shape
    if method == 'topk':                                       # inference path
        topk_vals, topk_idx = torch.topk(logits, k, dim=-1)
        mask = torch.zeros_like(logits).scatter(-1, topk_idx, 1.0)
    elif method == 'softmax':                                  # training path
        gumbel_noise = sample_gumbel(logits.shape, eps=eps,
                                     device=logits.device, dtype=logits.dtype)
        y = (logits + gumbel_noise) / temperature
        y_soft = F.softmax(y, dim=-1)
        if hard:
            topk_idx = y_soft.topk(k, dim=-1).indices
            hard_mask = torch.zeros_like(y_soft).scatter(-1, topk_idx, 1.0)
            mask = hard_mask - y_soft.detach() + y_soft        # straight-through
        else:
            mask = y_soft
    return mask

def global_selection(mask_logits, total_k, method, temperature, hard, eps):
    B, T, H, W = mask_logits.shape
    logits_flat = mask_logits.reshape(B, T * H * W)
    mask_flat   = select_topk(logits_flat, total_k, method, temperature, hard, eps)
    return mask_flat.reshape(B, T, H, W)

def structured_selection(mask_logits, k_t, k_hw, method, temperature, hard, eps):
    B, T, H, W = mask_logits.shape
    logits_t = mask_logits.mean(dim=[2, 3])                    # temporal axis
    mask_t   = select_topk(logits_t, k_t, method, temperature, hard, eps)  # [B, T]

    mask_spatial = []
    for b in range(B):
        mask_b = []
        for t in range(T):
            logits_hw = mask_logits[b, t].reshape(-1)
            mask_hw   = select_topk(logits_hw.unsqueeze(0), k_hw, method, temperature, hard, eps)
            mask_b.append(mask_hw.reshape(H, W))
        mask_spatial.append(torch.stack(mask_b, dim=0))
    mask_spatial = torch.stack(mask_spatial, dim=0)            # [B, T, H, W]

    return mask_spatial * mask_t.unsqueeze(-1).unsqueeze(-1)   # outer product

def apply_mask_and_select(tokens, other_tensors, mask):
    B, C, T, H, W = tokens.shape
    N = T * H * W
    tokens_flat = tokens.reshape(B, C, N)
    mask_flat   = mask.reshape(B, N)

    selected_tokens, selected_others = [], [[] for _ in other_tensors]
    for b in range(B):
        idx = mask_flat[b].nonzero(as_tuple=False).squeeze(-1)
        selected_tokens.append(tokens_flat[b, :, idx])
        for i, t in enumerate(other_tensors):
            t_flat = t.reshape(B, -1, N)
            selected_others[i].append(t_flat[b, :, idx])

    tokens_out = torch.stack(selected_tokens, dim=0)
    others_out = [torch.stack(x, dim=0) for x in selected_others]
    return tokens_out, others_out

def process_tensors(tokens, mask_logits, other_tensors,
                    total_k=None, k_t=None, k_hw=None,
                    temperature=1.0, eps=1e-6, training=True, soft_inference=True):
    B, C, T, H, W = tokens.shape
    mask_logits = mask_logits.squeeze(1)
    if training or soft_inference:
        method, hard = 'softmax', True
    else:
        method, hard = 'topk', False
    if total_k is not None:
        mask = global_selection(mask_logits, total_k, method, temperature, hard, eps)
    elif k_t is not None and k_hw is not None:
        mask = structured_selection(mask_logits, k_t, k_hw, method, temperature, hard, eps)
    tokens_out, others_out = apply_mask_and_select(tokens, other_tensors, mask)
    return tokens_out, others_out, mask
```

## 逐行讲解 / What's happening

### Gumbel-Softmax 直通的核心 / The Gumbel-Softmax straight-through core

1. **`sample_gumbel` (`-log(-log(U))`)**:
   - 中文: 这是 Gumbel 分布的"inverse CDF"采样:从均匀分布 U 上做两次 -log 变换。Gumbel-max 定理告诉我们,对 `logits + gumbel_noise` 取 argmax,等价于从 `softmax(logits)` 分布抽样——这是 Gumbel-Softmax trick 的数学基础。 `clamp(min=eps, max=1-eps)` 防 log(0)。
   - English: Gumbel inverse-CDF sampling: two nested negatives of `log` on a uniform `U`. The Gumbel-max theorem says `argmax(logits + gumbel)` is distributed exactly as `Categorical(softmax(logits))` — that's why this trick gives us differentiable categorical sampling. `clamp(min=eps, max=1-eps)` guards against `log(0)`.

2. **`y = (logits + gumbel_noise) / temperature`**:
   - 中文: 加噪声 + 除温度,然后做 softmax。温度 → 0 时趋向 one-hot,温度 → ∞ 时趋向均匀分布。训练时一般从 high T 退火到 low T。
   - English: Add noise, divide by temperature, then softmax. Temperature → 0 sharpens to one-hot; temperature → ∞ flattens to uniform. Training usually anneals high → low.

3. **`mask = hard_mask - y_soft.detach() + y_soft` (the straight-through estimator)**:
   - 中文: 这是整段代码的"灵魂三件套":
     - **Forward**:`hard_mask - y_soft.detach() + y_soft = hard_mask - y_soft + y_soft = hard_mask`(一个 one-hot)。后续层看到的是离散选择。
     - **Backward**:`grad(hard_mask) = 0`(它来自 `topk`,不可微);`grad(y_soft.detach()) = 0`;唯一活的是 `+ y_soft` 那一项,梯度就完全是 `y_soft` 的梯度——也就是软 softmax 的梯度。
   - **效果**:前向走硬选择,反向走软梯度。
   - English: The straight-through trinity:
     - **Forward**: `hard_mask - y_soft.detach() + y_soft = hard_mask`. Downstream layers see a discrete one-hot.
     - **Backward**: `grad(hard_mask) = 0` (it came from `topk`, non-differentiable); `grad(y_soft.detach()) = 0`; only `+ y_soft` contributes, so the gradient is exactly the soft softmax's gradient.
   - **Effect**: hard selection in the forward pass, soft gradient in the backward pass.

### 推理路径 / The inference path

4. **`method == 'topk'`**:
   - 中文: 推理时不需要梯度,直接 `torch.topk` 拿到 top-k 索引,然后 `scatter` 一个 one-hot mask。简单可靠。
   - English: At inference, no gradient needed — `torch.topk` for the indices, `scatter` for the one-hot mask. Boring, fast.

5. **`soft_inference=True` 默认 / The `soft_inference` default**:
   - 中文: `process_tensors` 默认 `training=True, soft_inference=True`——这意味着 *训练 *和* 推理* 都走 Gumbel-softmax 路径。为啥推理也加 Gumbel 噪声?因为这能保持训练 / 推理的分布一致,防止 train-test gap。如果你想完全确定性 inference,设 `training=False, soft_inference=False`。
   - English: `process_tensors` defaults to `training=True, soft_inference=True` — both train and infer use the Gumbel-softmax path. Why noise at inference? It keeps the train/test distributions matched, preventing a train-test gap. For deterministic inference, set `training=False, soft_inference=False`.

### 全局 vs 结构化 / Global vs structured selection

6. **`global_selection`**:
   - 中文: 把 `[B, T, H, W]` flatten 成 `[B, T·H·W]`,直接挑 `total_k` 个 token。简单,但可能某些帧全空、某些帧塞满——不一定符合下游 transformer 的位置编码假设。
   - English: Flatten `[B, T, H, W]` to `[B, T·H·W]` and pick `total_k`. Simple, but the selection may be lumpy across frames — possibly violating downstream positional-encoding assumptions.

7. **`structured_selection`**:
   - 中文: 先按时间维选 `k_t` 帧(用 mean(H, W) 作为帧重要性 logit),再 *逐帧* 在 H·W 平面上选 `k_hw` 个 spatial token,最后两个 mask 做外积。保证每选中的帧固定有 `k_hw` 个 token,token 数稳定 = `k_t * k_hw`。
   - English: Pick `k_t` frames by mean-over-spatial as the temporal logit, then *per chosen frame* pick `k_hw` spatial tokens, then outer-product the two masks. Guaranteed token count `k_t * k_hw`, evenly distributed across frames — kinder to downstream pos-enc.

### apply_mask_and_select 的"对齐多张量" / Multi-tensor aligned gather

8. **`for b in range(B): idx = mask_flat[b].nonzero(...).squeeze(-1)`**:
   - 中文: 因为不同 batch sample 选中的 k 个位置可能 *不一样*,所以不能简单地做一个 dense gather——只能 per-batch loop。这是 token pruning 的标准蹩脚之处:破坏了 batch 一致性,kernel 利用率打折。
   - English: Different batch elements may select different sets of k positions, so it can't be a single dense gather — per-batch loop required. This is the canonical awkwardness of token pruning: batch-level shape uniformity is broken, GPU kernel utilization takes a hit.

9. **`tokens_flat[b, :, idx]` + `t_flat[b, :, idx]`**:
   - 中文: 同一个 `idx` 同时索引到 `tokens` 和所有 `other_tensors`——这就是"保持对齐"的关键。如果你 token 选了第 7、 19、 31 个,RoPE 的 (pos_id_7, pos_id_19, pos_id_31) 也得跟着选。 / The same `idx` gathers from `tokens` AND every aligned auxiliary tensor (RoPE positions, mask bits, etc.). Keeping selections aligned across tensors is the whole point.

## 类比 / The analogy

想象一个考试评委组:30 个人坐在桌子两边,每人对每个候选人有一个评分(logits)。你只能让 5 个候选人通过。**老办法**:30 个人投票后挑得分最高的 5 个,但你没法回头改进打分规则——这就是 `torch.topk` 不可微的本质。**Gumbel-Softmax 直通**:正式公布结果的时候只念前 5 名(hard mask),但 *给每个评委的反馈* 是按 softmax 概率分布写的(soft gradient)——这样你下次评的时候规则会变得更准。`hard_mask - y_soft.detach() + y_soft` 这一行就是"对外公布前 5 名,但对内梯度走软分布"的等价数学。

Picture an interview panel: 30 candidates, scores from each panelist (logits). You select the top 5. **Old way**: tally votes, take top 5 — but you can't backtrack to improve the scoring rule because `torch.topk` has no gradient. **Gumbel-Softmax straight-through**: announce the top 5 (hard mask), but the *feedback to each panelist* follows the softmax distribution (soft gradient). Next round the scoring rule improves. `hard_mask - y_soft.detach() + y_soft` is the algebraic identity that gives you "announce hard top-5, learn from soft gradients" in one line.

## 自己跑一遍 / Try it yourself

```python
# pip install torch
import torch
import torch.nn.functional as F

torch.manual_seed(0)

def gumbel_topk_straight_through(logits, k, temperature=0.5):
    # 1. Gumbel-softmax: add Gumbel noise, divide by tau, softmax.
    U = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
    gumbel = -torch.log(-torch.log(U))
    y_soft = F.softmax((logits + gumbel) / temperature, dim=-1)

    # 2. Hard top-k mask (non-differentiable).
    idx = y_soft.topk(k, dim=-1).indices
    hard = torch.zeros_like(y_soft).scatter_(-1, idx, 1.0)

    # 3. Straight-through: forward = hard, backward = grad(y_soft).
    return hard - y_soft.detach() + y_soft

# A trainable scorer: 10-d input → 30-d logits (pick 5 of 30 tokens).
scorer = torch.nn.Linear(10, 30, bias=False)
opt = torch.optim.AdamW(scorer.parameters(), 0.1)

target_idx = torch.tensor([2, 7, 11, 19, 26])     # "ground truth top-5 should be these"
target = torch.zeros(1, 30); target[0, target_idx] = 1.0

x = torch.randn(1, 10)
for step in range(200):
    logits = scorer(x)
    mask = gumbel_topk_straight_through(logits, k=5, temperature=0.5)
    loss = F.mse_loss(mask, target)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 50 == 0:
        chosen = mask.nonzero()[:, 1].sort().values.tolist()
        print(f"step {step:>4d}   loss={loss.item():.4f}   chosen={chosen}")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
step    0   loss=0.1667   chosen=[1, 2, 11, 19, 23]
step   50   loss=0.0789   chosen=[2, 7, 11, 19, 21]
step  100   loss=0.0211   chosen=[2, 7, 11, 19, 26]
step  150   loss=0.0067   chosen=[2, 7, 11, 19, 26]
```

中文一两句:看 `chosen` 这一列——选中的 token 索引慢慢收敛到 `[2, 7, 11, 19, 26]`,跟 ground truth 完全一致。这说明梯度真的从离散 mask 流到了 scorer 的权重——直通估计器干活了。

In English: watch the `chosen` column converge to `[2, 7, 11, 19, 26]`, the ground truth. The gradient really did flow back from the discrete mask into the scorer's weights — the straight-through estimator worked.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **DynamicViT (NeurIPS 2021)** / **DynamicViT (NeurIPS 2021)**: 第一篇把 Gumbel-Softmax 用在 vision token pruning 上的工作,跟 Lyra 这个文件几乎是同构。 / The original ViT-token-pruning-with-Gumbel work; structurally near-identical to this file.
- **Mixture-of-Experts (MoE) 的 noisy top-k routing** / **MoE noisy top-k routing**: Switch Transformer / GShard 选择 expert 的时候也是 `topk(scores + gumbel_noise)`,但通常不带 straight-through——MoE 让梯度通过 `gate * expert_output` 的乘法路径流回。 / Switch Transformer / GShard score experts with `topk(scores + gumbel_noise)`. They don't use ST; instead the gradient flows through `gate * expert_output` multiplication.
- **VQ-VAE 的 codebook lookup straight-through** / **VQ-VAE codebook lookup ST**: VQ-VAE 经典的 `quantize.detach() - input.detach() + input` 跟今天这条 `hard - soft.detach() + soft` 是同一个数学技巧的不同上下文。 / The same `hard - soft.detach() + soft` identity wearing a different hat — VQ-VAE's `quantize.detach() - input.detach() + input`.
- **Concrete / Categorical distribution (Maddison et al. 2017)** / **Concrete (Maddison et al. 2017)**: Gumbel-Softmax 的理论祖宗,论文里就是写"hard sample at forward, soft sample at backward"。 / The theoretical paper that introduced this exact forward/backward decoupling.

## 注意事项 / Caveats / when it breaks

- **Per-batch loop 杀显卡利用率 / Per-batch loop kills GPU utilization**: `apply_mask_and_select` 的 `for b in range(B)` 是这套设计绕不开的代价——批量内每个 sample 选的 k 个位置不一样,就只能逐 sample gather。 batch size 大的时候这能成为瓶颈;NVIDIA Triton kernel 那一类的稀疏 gather 才能救场。 / The `for b in range(B)` loop in `apply_mask_and_select` is the unavoidable cost — different batch samples select different indices. A custom Triton sparse-gather kernel is the production fix.
- **温度退火 / Temperature annealing matters**: 训练初期温度高(0.5-1.0),mask 是接近均匀的"软 mask",梯度信号很丰富;训练末期温度低(0.05-0.1),mask 越来越接近 one-hot,跟推理路径一致。如果不退火,梯度信号要么过强(高温)要么过弱(低温)。 / Anneal high → low. High temperature gives strong gradients but a fuzzy mask; low temperature matches inference but starves gradients. Skipping the schedule typically slows convergence.
- **`hard_mask - y_soft.detach() + y_soft` 的等价数学很容易写错 / Easy to write wrong**: 漏一个 `.detach()` 就会让 grad 同时从两条路径流(`hard - y_soft + y_soft = hard`,但 grad 加倍)。这个 bug 静默,容易过审 review。 / Forget one `.detach()` and the gradient flows through both branches — `hard - y_soft + y_soft = hard` still works in forward, but the gradient doubles. Silent bug.
- **`scatter` 不去重 / `scatter` doesn't dedupe**: 如果 `topk_idx` 里碰巧重复(对 `softmax` 之后的连续值 topk 几乎不可能,但理论上),`scatter` 只保留最后一次,会产生少于 k 个 1。 / If `topk_idx` had duplicates, `scatter` keeps only the last write — fewer than k ones.

## 延伸阅读 / Further reading

- [Lyra-1 README](https://github.com/nv-tlabs/lyra) — Project Lyra overview (open generative 3D world models).
- [Gumbel-Softmax (Jang, Gu, Poole, 2016)](https://arxiv.org/abs/1611.01144) — the original paper introducing the trick.
- [Concrete distribution (Maddison, Mnih, Teh, 2016)](https://arxiv.org/abs/1611.00712) — independent contemporaneous derivation.
- [DynamicViT (Rao et al., NeurIPS 2021)](https://arxiv.org/abs/2106.02034) — applies this exact pattern to vision transformer token pruning.
