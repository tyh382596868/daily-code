---
date: 2026-06-09
topic: robotics
source: tracked
repo: huggingface/lerobot
file: src/lerobot/rewards/robometer/modeling_robometer.py
permalink: https://github.com/huggingface/lerobot/blob/49755a3d9e7d43ae93092de8324e75348955afab/src/lerobot/rewards/robometer/modeling_robometer.py#L92-L102
difficulty: beginner
read_time: ~8 min
tags: [code-of-the-day, robotics, reward-model, categorical-distribution]
---

# ROBOMETER 用 11 行就把"离散桶 logits"变成连续进度信号 / ROBOMETER turns "discrete-bin logits" into a continuous progress signal in 11 lines

> **一句话 / In one line**: 让模型输出 `N` 个均匀分布在 `[0, 1]` 的奖励桶概率,然后用 *桶中心的 softmax 加权均值* 把它压回一个连续数 —— 这就是 C51 类分布式 RL 的核心,被搬到机器人奖励模型上。 / Let the model output `N` softmax probabilities over evenly spaced reward bins in `[0, 1]`, then collapse them into a single continuous value as the *softmax-weighted mean of the bin centers* — the core trick from C51-style distributional RL, repurposed as a robot reward head.

## 为什么重要 / Why this matters

机器人 RL 里最让人头疼的事之一就是"奖励长什么样"。直接回归一个标量进度(0 → 1)看似简单,但训练时损失曲面很尖、对噪声极其敏感;改成分类问题(`success` / `fail`)又太粗糙,丢掉了"完成了 70%"这种细粒度信息。ROBOMETER —— LeRobot 在 2026 年 5 月底刚 merge 的通用机器人奖励模型 —— 给出的妥协方案是"分类的训练,回归的推理":训练时把进度切成 `num_bins` 个桶用 cross-entropy 学,推理时再把 softmax 还原成一个 `[0, 1]` 的实数。整段还原代码就 11 行,但里面藏的是 C51(分布式 RL)那一套完整思路。

One of the most annoying questions in robot RL is "what shape should the reward take?" Regressing a scalar progress signal (0 → 1) is intuitive but produces a sharp, noise-sensitive loss landscape; turning it into a binary `success`/`fail` classifier throws away the fine-grained "70% there" information that's actually most useful for policy improvement. ROBOMETER — the general-purpose robot reward model LeRobot merged in late May 2026 — splits the difference: train as a classifier over `num_bins` progress bins, but at inference time collapse the softmax back into a continuous `[0, 1]` number. The whole inference-time decoder is 11 lines, and it carries the entire idea behind C51-style distributional RL.

## 代码 / The code

`huggingface/lerobot` — [`src/lerobot/rewards/robometer/modeling_robometer.py`](https://github.com/huggingface/lerobot/blob/49755a3d9e7d43ae93092de8324e75348955afab/src/lerobot/rewards/robometer/modeling_robometer.py#L92-L102)

```python
def convert_bins_to_continuous(bin_logits: Tensor) -> Tensor:
    """Collapse per-bin logits into a single value in ``[0, 1]``.

    The discrete progress head outputs ``num_bins`` logits per frame. Bins are
    evenly spaced centers in ``[0, 1]``; the continuous prediction is the
    softmax-weighted mean of those centers.
    """
    bin_probs = torch.softmax(bin_logits, dim=-1)
    num_bins = bin_logits.shape[-1]
    bin_centers = torch.linspace(0.0, 1.0, num_bins, device=bin_logits.device, dtype=bin_logits.dtype)
    return (bin_probs * bin_centers).sum(dim=-1)
```

## 逐行讲解 / What's happening

1. **第 99 行 / Line 99 (`torch.softmax(bin_logits, dim=-1)`)**:
   - 中文: 把模型最后一层吐出来的 `num_bins` 维 logit 沿最后一维做 softmax,得到一个离散概率分布 —— "我有多大把握进度在第 k 个桶里"。
   - English: Soft-max the model's final-layer `num_bins`-dim logits along the last axis, producing a discrete probability distribution — "how confident am I that progress falls in bin k".

2. **第 101 行 / Line 101 (`torch.linspace(0.0, 1.0, num_bins, ...)`)**:
   - 中文: 在 `[0, 1]` 上均匀分 `num_bins` 个等距点,这些就是"桶心"。注意 `device`/`dtype` 直接从 logits 借,避免 CPU↔GPU 来回搬。
   - English: Lay down `num_bins` equally spaced anchors in `[0, 1]` — these are the bin *centers*. Note how the call inherits `device` / `dtype` directly from the logits so no implicit CPU↔GPU copies sneak in.

3. **第 102 行 / Line 102 (`(bin_probs * bin_centers).sum(dim=-1)`)**:
   - 中文: 算"概率 × 桶心"的期望 —— 概率分布的均值就是连续预测。如果模型对第 7 个桶最有信心、对邻居略有信心,那期望就会落在第 7 个桶心附近、但略微被拉向邻居。比 `argmax` 平滑得多。
   - English: Compute the expectation `E[bin_center]` under the predicted distribution — the mean of the categorical distribution *is* the continuous prediction. If the model puts most mass on bin 7 with some leakage to neighbors, the expectation lands near bin 7's center but is gently pulled toward the neighbors. Much smoother than `argmax`.

## 类比 / The analogy

想象一台老式厨房秤,只有 10 个粗刻度:0、100、200、…、900 克。你想称一个 250 克的橘子,但指针卡在 200 和 300 之间不停抖。如果你只看"指针离哪个刻度最近"就报 200,误差有 50 克。但如果你看到"指针 60% 时间停在 200、40% 停在 300",算个加权:`0.6 × 200 + 0.4 × 300 = 240` 克 —— 突然就接近真值了。 `convert_bins_to_continuous` 干的就是这件事:把分类器的"在每个刻度上的把握"翻译成秤盘上的精确读数。

Think of an old kitchen scale with only ten coarse tick marks: 0, 100, 200, …, 900 grams. You want to weigh a 250-gram orange, but the needle keeps shaking between the 200 and 300 ticks. If you only report "the tick the needle is closest to," you'd say 200 g — off by 50 g. But if you notice "the needle spends 60% of its time on 200 and 40% on 300," a weighted average `0.6 × 200 + 0.4 × 300 = 240 g` snaps you back near the truth. `convert_bins_to_continuous` does exactly this: it translates a classifier's "confidence on each tick" into a precise read-out on the dial.

## 自己跑一遍 / Try it yourself

```python
import torch

def convert_bins_to_continuous(bin_logits):
    bin_probs = torch.softmax(bin_logits, dim=-1)
    num_bins = bin_logits.shape[-1]
    centers = torch.linspace(0.0, 1.0, num_bins, dtype=bin_logits.dtype)
    return (bin_probs * centers).sum(dim=-1)

# A 21-bin head: 0.00, 0.05, ..., 1.00.
# Frame 1: very confident at bin 14 (= 0.70).
# Frame 2: split between bins 4 (0.20) and 5 (0.25) — a 50/50 tie.
logits = torch.full((2, 21), -10.0)
logits[0, 14] = 5.0
logits[1, 4] = 5.0
logits[1, 5] = 5.0

print(convert_bins_to_continuous(logits))
# tensor([0.7000, 0.2250])  -> exactly the midpoint between 0.20 and 0.25
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
tensor([0.7000, 0.2250])
```

中文一句:第二个 frame 的输出 `0.2250` 恰好是两个均分桶 (0.20 和 0.25) 的几何中点 —— 这种"软插值"行为是 `argmax` 永远给不了你的,但只要桶够密、softmax 不熔化,它几乎免费。

English: the second frame returns `0.2250` — exactly the midpoint between the two tied bins (0.20 and 0.25). This kind of soft interpolation between bins is something `argmax` can never give you, and it costs essentially nothing as long as your bins are dense enough and your softmax isn't saturated.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **C51 / Distributional Q-learning (DeepMind, 2017)** / **C51 / Distributional Q-learning**: 中文:这套"桶 logits → 期望"的还原步骤是 C51 的回声 —— C51 让 Q 网络回归一个 51 桶的回报分布而不是一个标量,推理时再用 `(p · v).sum()` 取期望,被证明能让训练稳定不少。 / English: The "bin-logits → expectation" decoding step echoes C51, which has the Q-network emit a 51-bin return distribution instead of a scalar; inference takes `(p · v).sum()` exactly like here, and the swap demonstrably stabilises training.
- **Dreamer / DreamerV3 — symlog two-hot encoder**: 中文:DreamerV3 用"双热编码"把回报编进一个对称的对数桶网格,再用同样的 softmax-加权-均值还原回标量。机器人 world-model 圈现在几乎人手一份。 / English: DreamerV3's symlog two-hot encoder packs returns into a symmetric log-spaced bin grid, then decodes with exactly the same softmax-weighted-mean trick. Half the robotics world-model papers from 2024-2026 import this verbatim.
- **MuZero / Muesli value head**: 中文:同样把价值切成桶学,推理时再聚合 —— 关键好处是 cross-entropy 训练比 MSE 更稳。 / English: MuZero and Muesli both train value heads as categorical bins and re-aggregate at inference. The win is the same — cross-entropy is far more forgiving to train than MSE on long-tailed targets.

## 注意事项 / Caveats / when it breaks

- **桶要够密 / Bins must be dense enough**: 中文:如果只有 5 个桶,你的连续预测最多就 5 个不同的"准均匀"值,失去意义。论文常用 21、51、101 个桶。 / English: With only 5 bins your continuous prediction collapses into at most 5 distinct "quasi-uniform" values and the smoothing wins disappear. Standard choices are 21, 51, or 101 bins.
- **多峰分布会被坑 / Multi-modal distributions get averaged away**: 中文:如果模型 50% 信进度是 0.2、50% 信是 0.8,期望就是 0.5 —— 一个完全错误的答案。这是 C51 家族共有的失败模式。需要先 `argmax` 检测分布尖锐度再决定要不要这么聚合。 / English: If the model is 50% confident on 0.2 and 50% on 0.8, the expectation is 0.5 — a value the model never assigned any belief to. This is a well-known C51 failure mode; production code usually first checks distribution sharpness (e.g. via the top-2 mass ratio) before trusting the expectation.
- **桶心要和训练时对齐 / Bin centers must match training**: 中文:这里用 `linspace(0, 1)` 默认从 0 开始,如果训练时桶是 `(0.5/N, 1 - 0.5/N)` 这种"半步偏移"的中心,推理就会系统性偏移。永远从 config 里读 `num_bins`,绝不要硬编码。 / English: This snippet uses `linspace(0, 1)` from 0 to 1. If training used half-step-offset centers (`(0.5/N, 1 - 0.5/N)`), inference will be systematically biased. Always pipe `num_bins` from config; never hard-code it in two places.

## 延伸阅读 / Further reading

- [ROBOMETER paper (arXiv 2603.02115)](https://arxiv.org/abs/2603.02115)
- [C51: A Distributional Perspective on Reinforcement Learning (Bellemare et al., 2017)](https://arxiv.org/abs/1707.06887)
- [DreamerV3 paper — symlog two-hot encoder appendix](https://arxiv.org/abs/2301.04104)
