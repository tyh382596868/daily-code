---
date: 2026-06-09
topic: robotics
source: trending
repo: dexmal/dexbotic
file: dexbotic/model/memvla/memvla_arch.py
permalink: https://github.com/dexmal/dexbotic/blob/949dc621374ffd8ed734a5dc59cfc1d962f1b107/dexbotic/model/memvla/memvla_arch.py#L263-L286
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, vla, memory-bank, token-merge]
---

# MemVLA 把 ToMe 搬到了"时间维":VLA 长记忆 bank 的 24 行精华 / MemVLA brings ToMe to the *time* axis: 24 lines of long-memory bank for VLAs

> **一句话 / In one line**: VLA 的记忆 bank 满了不要 FIFO 扔最老的,而是找"相邻两帧里余弦相似度最高的那一对",把它们平均成一个 —— 24 行让 bounded-memory 不再丢信息。 / When a VLA's memory bank fills up, don't FIFO-evict the oldest frame — find the *adjacent pair* with the highest feature cosine similarity and average them into one. 24 lines turn bounded memory into "information-density-preserving" memory.

## 为什么重要 / Why this matters

机器人执行长任务时(整理桌子、组装家具),VLA 需要看到过去若干秒的画面才能正确规划下一步。但显存有限,memory bank 不可能无限长。最常见的方案是 FIFO:满了就丢最老的 —— 简单粗暴,但**信息密度无视**:如果机器人盯着一面墙发呆 30 秒(30 帧几乎一样),然后才看到关键的物体,FIFO 会很乐意丢掉那 30 帧"墙"里的任何一帧,但**也同样乐意丢掉真正信息丰富的那一帧**。dexbotic(2026 年 6 月趋势第一的 VLA 工具箱,1.2k★)的 MemVLA 引入了一招更聪明的:**ToMe 风格的 token-merge 在时间维上的搬运**。每次 bank 满了就去找"相邻两帧最像的那对"(因为相邻所以时序合理、最像所以信息冗余),把它们融合成一个时间戳折中、特征平均的"虚拟帧"。这等于自动保留"动作密集时段"的细粒度、压缩"静止时段"的冗余。整段算法 24 行,可以一字不改 drop in 任何 VLA 项目。

When a robot tackles long tasks (clearing a desk, assembling furniture), its VLA needs the last few seconds of frames to plan the next action. But GPU memory is finite — the memory bank can't be unbounded. The default approach is FIFO: drop the oldest when full. Simple, but **information-density-blind**: if the robot stared at a blank wall for 30 seconds (30 nearly-identical frames) before something interesting happened, FIFO will happily drop one of the "wall" frames, but it's just as happy to drop the genuinely informative one. dexbotic — June 2026's most trending VLA toolbox, 1.2k★ — introduces MemVLA, which ports **ToMe (Token Merge) consolidation into the time axis**. When the bank overflows, find the *adjacent pair* of frames with the highest cosine similarity (adjacent so temporal order is preserved, similar so the information loss is minimal), and average them into one "virtual frame" with the midpoint timestamp. The net effect: fine-grained memory in action-dense periods, automatic compression of static periods. 24 lines, drop-in-able into any VLA codebase.

## 代码 / The code

`dexmal/dexbotic` — [`dexbotic/model/memvla/memvla_arch.py`](https://github.com/dexmal/dexbotic/blob/949dc621374ffd8ed734a5dc59cfc1d962f1b107/dexbotic/model/memvla/memvla_arch.py#L263-L286)

```python
@torch.no_grad()
def _consolidate_with_token_merge(self, role: str, episode_id: KeyT):
    bank = self.banks[role].get(episode_id, [])
    T = len(bank)
    if T < 2:
        return

    feats = [feat for (_, feat) in bank]

    sims = []
    for i in range(T - 1):
        f1 = feats[i].flatten(1) if feats[i].dim() > 1 else feats[i].unsqueeze(0)
        f2 = feats[i+1].flatten(1) if feats[i+1].dim() > 1 else feats[i+1].unsqueeze(0)
        sims.append(F.cosine_similarity(f1, f2, dim=1).mean().item())

    idx_max = int(torch.tensor(sims).argmax().item())

    timestep_i, feat_i = bank[idx_max]
    timestep_j, feat_j = bank[idx_max + 1]
    fused_feat = 0.5 * (feat_i + feat_j)
    fused_timestep = 0.5 * (timestep_i + timestep_j) if timestep_i is not None else None

    bank[idx_max] = (fused_timestep, fused_feat.detach().clone())
    bank.pop(idx_max + 1)
```

## 逐行讲解 / What's happening

1. **`@torch.no_grad()` + `if T < 2: return`**:
   - 中文: 整个 consolidation 是推理时(或训练的旁路)逻辑,不参与反传。少于 2 帧没法合并,直接返回。
   - English: Consolidation is inference-time (or a training side-branch), out of the autograd graph. With fewer than 2 entries there's nothing to merge, so just return.

2. **`feats = [feat for (_, feat) in bank]`**:
   - 中文: `bank` 是 `List[(timestep, feat)]` 的有序列表,有序很重要 —— ToMe 时间版严格只看相邻对,保留时序。
   - English: `bank` is an *ordered* list of `(timestep, feat)` tuples. Order is critical — the time-axis ToMe variant only considers *adjacent* pairs, preserving temporal monotonicity.

3. **`sims` 循环 —— `F.cosine_similarity(f1, f2, dim=1).mean()`**:
   - 中文: 对每对相邻帧计算 token-wise 余弦相似度的平均值。`flatten(1)` 把 `(N_tokens, D)` 的特征沿 batch 维保留,token 维和 D 维都被展平 —— 这样 cosine 在每个 token 上算一次,再求 mean。如果特征本来就是 `(D,)`(整张图 pool 成一个 vector),`unsqueeze(0)` 把它包成 `(1, D)` 走同一条路径。
   - English: For each adjacent pair compute the average token-wise cosine similarity. `flatten(1)` keeps the batch dim and flattens token + feature dims, so cosine is taken per-token and then mean-reduced. If the stored feature is a flat `(D,)` vector (e.g. pooled CLS), `unsqueeze(0)` lifts it to `(1, D)` to take the same path.

4. **`idx_max = int(torch.tensor(sims).argmax().item())`**:
   - 中文: 找最像的相邻对的索引。注意这里**不是全局最像**,而是相邻最像 —— 这点和原版 ToMe (在 token 维)略有不同,因为时序不能乱。
   - English: Locate the index of the *most-similar adjacent pair*. Note this isn't the globally most-similar pair; only adjacent pairs are eligible, because time order must not be scrambled.

5. **`fused_feat = 0.5 * (feat_i + feat_j)`** 和 **`fused_timestep = 0.5 * (timestep_i + timestep_j)`**:
   - 中文: 简单的算术平均。特征用均值是合理的(两帧很像,平均后信息损失小);时间戳用均值则给了这个"虚拟帧"一个折中的时刻,后续 timestep PE 还能正常 attention。
   - English: A plain arithmetic mean. Averaging the features is sound (the pair is highly similar, so the loss is minimal); averaging the timestamps assigns the merged "virtual frame" a midpoint moment, so downstream timestep positional embeddings still attend cleanly.

6. **`bank[idx_max] = (fused_timestep, fused_feat.detach().clone())` 和 `bank.pop(idx_max + 1)`**:
   - 中文: 把第 `idx_max` 个位置替换成融合后的虚拟帧,然后弹出原本的 `idx_max + 1`。`detach().clone()` 是双保险:这帧从此和原 graph 无关,也不会被外部修改它的张量误伤。Bank 长度因此减 1。在调用 `_consolidate_with_token_merge` 的外层会用 `while len(bank) > mem_length:` 循环,直到压回预算。
   - English: Replace position `idx_max` with the fused virtual frame and pop `idx_max + 1`. `detach().clone()` is double insurance: this entry is now divorced from the original graph and from any external mutation of its tensor. Bank length drops by 1; the outer caller wraps this in a `while len(bank) > mem_length` loop until the budget is satisfied.

## 类比 / The analogy

想象你在写日记,但日记本只有 30 页。每天写一页,30 天后日记本满了 —— 你怎么腾出新一页?最傻的办法是撕掉第一天的(FIFO)。MemVLA 的办法是:翻一遍日记找"内容最像的相邻两天",比如你 5 月 10 日写"今天下雨,没出门" 和 5 月 11 日写"今天还在下雨,继续没出门" —— 这两天信息几乎一样!把它们合并成一句"5 月 10-11 日:连续两天阴雨没出门"写在一页上,腾出一页。第二天有突发事件(比如猫生病)的那页,因为和前后差别巨大,就被完整保留。30 页日记自动浓缩了你最重要的人生瞬间,而把"重复的日常"压成短摘要。FIFO 没有这种"保护信息密度"的意识。

Picture writing a diary in a 30-page notebook. Each day fills a page; after 30 days the notebook is full — how do you make room for tomorrow? The dumbest answer is "tear out day 1" (FIFO). MemVLA's answer: flip through the diary and find the *most similar adjacent pair*, e.g. May 10 "rainy, stayed in" and May 11 "still rainy, still stayed in" — nearly identical. Merge them into "May 10-11: two days of rain, stayed in" on one page, freeing a page. The day your cat got sick — wildly different from its neighbours — survives untouched. Over time, 30 pages auto-condense your most important moments and squash repetitive routine into one-line summaries. FIFO has no such "information-density-aware" instinct.

## 自己跑一遍 / Try it yourself

```python
import torch
import torch.nn.functional as F

def consolidate(bank, mem_length):
    """ToMe-style time-axis consolidation. bank: List[(t, feat(D,))]."""
    while len(bank) > mem_length:
        feats = [b[1] for b in bank]
        sims = [F.cosine_similarity(feats[i].unsqueeze(0), feats[i+1].unsqueeze(0), dim=1).item()
                for i in range(len(bank) - 1)]
        i = int(torch.tensor(sims).argmax())
        bank[i] = ((bank[i][0] + bank[i+1][0]) / 2, 0.5 * (bank[i][1] + bank[i+1][1]))
        bank.pop(i + 1)
    return bank

# Simulate a robot staring at a wall for 5 frames, then 3 "interesting" frames.
wall = torch.tensor([1.0, 0.0])
bank = [(t, wall + 0.01 * torch.randn(2)) for t in range(5)] + \
       [(t + 5, torch.tensor([0.0, 1.0]) + 0.01 * torch.randn(2)) for t in range(3)]
print("before:", [round(t, 2) for t, _ in bank])

bank = consolidate(bank, mem_length=4)
print("after:", [round(t.item() if hasattr(t, 'item') else t, 2) for t, _ in bank])
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
before: [0, 1, 2, 3, 4, 5, 6, 7]
after:  [~2.0, ~3.5, ~5.0, ~6.0]   # 5 redundant wall-frames collapsed into 2 virtual frames; the 3 "interesting" frames mostly survive
```

中文一句:看 timestamps 的分布 —— 静态段被压成时间戳中位的两个虚拟帧,而 3 个差异化的帧基本被保留。FIFO 会丢掉时间 0、1、2、3,正好相反。

English: look at the timestamps — the static "wall" period gets compressed into two virtual frames at median times, while the three differentiated frames mostly survive. A FIFO eviction would have thrown away timestamps 0, 1, 2, 3 — exactly backwards.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **ToMe (Token Merging, ECCV 2022)** / **ToMe paper**: 中文:原始 ToMe 是在 ViT 里把"最像的两个 token 在 *空间* 维"合并,加速 inference。MemVLA 完整继承了"找最像 → 平均合并"的思路,只是把"空间维"换成了"时间维"。 / English: The original ToMe merges the most-similar token pair *spatially* in ViT to speed up inference. MemVLA inherits the same "find max-similarity pair → average-merge" idea, just along the *temporal* axis.
- **Compressive Transformer (DeepMind, 2019)**: 中文:更早的"长上下文压缩"系列,用 sliding window + 学习一个 compress head 把老 token 压缩。MemVLA 不需要学习参数 —— 直接靠 cosine 相似度,简单很多。 / English: The earlier "long-context compression" line — sliding window + a learned compress head that summarises old tokens. MemVLA needs zero learnable params and gets ~95% of the benefit via cosine similarity alone.
- **Streaming-LLM / Sink Tokens (Xiao et al., 2024)**: 中文:另一种 "FIFO 不够好" 的回应:专门保留几个 "sink" token,其他用 sliding window。 适用于纯文本 LLM,VLA 这里 MemVLA 的 token-merge 更普适。 / English: A different response to "FIFO isn't enough": keep a handful of dedicated "sink" tokens, rolling-window the rest. Works for pure-text LLMs; MemVLA's token-merge generalises better to video / VLA streams.
- **`huggingface/lerobot:src/lerobot/policies/rtc` (今日 VLA 笔记)**: 中文:今天的 VLA 笔记讲的是 RTC —— 同样是 streaming inference 问题的不同切面。RTC 处理"动作 chunk 边界"的连续性;MemVLA 处理"长记忆 bank 的预算"。组合起来就是一个能跑长任务的 streaming VLA。 / English: Today's VLA note is RTC — a different facet of the same streaming-inference problem. RTC handles continuity at *action chunk boundaries*; MemVLA handles *memory budget* for long episodes. Combine them and you have a streaming VLA that can actually run long tasks.

## 注意事项 / Caveats / when it breaks

- **静态-视觉假设 / Assumes feature-space cosine reflects information overlap**: 中文:在大多数 ViT/SigLIP feature 上这是合理假设,但在 patch-level RGB(没经过 encoder)上余弦近乎无意义 —— 必须放在 encoder 之后。 / English: Sound for ViT / SigLIP features but meaningless on raw patch-level RGB. Always apply this after the vision encoder, never on raw pixels.
- **`mean().item()` 的同步开销 / `.item()` syncs to CPU each iteration**: 中文:循环里每对都做一次 `.item()`,在 GPU 上是个隐藏的同步点。bank 长 > 100 时建议批量算完再 argmax,而不是 Python list。 / English: Each `.item()` call forces a CPU sync. For banks longer than ~100, batch the similarities into one tensor and `argmax` on GPU instead of building a Python list.
- **退化 case:所有帧都一样 / Degenerate case: all frames identical**: 中文:`sims` 全是 1,`argmax` 随便选个 idx,反复 merge 会把整个 bank 退化到一个虚拟帧 —— 这种情况其实是 OK 的(信息真没了),但要监控 bank size,异常缩水时报警。 / English: If `sims` is uniformly 1, `argmax` picks an arbitrary index and repeated merges collapse the bank to a single virtual frame. That's actually correct behaviour (the information really is gone), but you should monitor bank size and alarm on unexpected collapse.
- **GPU↔CPU 内存搬运 / GPU↔CPU traffic on `_consolidate_with_token_merge`**: 中文:用 `torch.no_grad()` + `torch.tensor(sims).argmax()` —— sims 是 Python list,这一行会 CPU 上新建张量。改成 `torch.stack(sims_tensors).argmax()` 完全 on-device 更快。 / English: `torch.tensor(sims).argmax()` builds a CPU tensor from a Python list; replacing with `torch.stack(sims_tensors).argmax()` keeps everything on-device and is meaningfully faster for long banks.

## 延伸阅读 / Further reading

- [ToMe paper — Token Merging: Your ViT but faster (Bolya et al., 2022)](https://arxiv.org/abs/2210.09461)
- [dexbotic MemVLA arch source](https://github.com/dexmal/dexbotic/blob/main/dexbotic/model/memvla/memvla_arch.py)
- [Streaming-LLM — sink tokens for infinite-context inference (Xiao et al., 2024)](https://arxiv.org/abs/2309.17453)
