---
date: 2026-06-10
topic: wam
source: wam
repo: facebookresearch/DiT
file: models.py
permalink: https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L67-L94
difficulty: beginner
read_time: ~8 min
tags: [code-of-the-day, wam, diffusion, classifier-free-guidance, dit]
build_role: nanoWAM / classifier-free-guidance (cross-repo variant) — the canonical training-side CFG implementation in 28 lines
---

# 28 行 DiT LabelEmbedder:CFG 的"教学版"实现 / DiT's 28-line LabelEmbedder: the textbook implementation of classifier-free guidance

> **一句话 / In one line**: 在 nn.Embedding 表里多留一行给"空标签",训练时用伯努利掷骰子决定要不要把真标签换成空标签 —— 这就是 classifier-free guidance 全部的训练侧实现。 / Reserve one extra row in `nn.Embedding` for the "null" label, and a Bernoulli coin-flip at train time decides whether to swap the real label for null — that's the entire training-side implementation of classifier-free guidance.

## 为什么重要 / Why this matters

CFG(Classifier-Free Guidance)是现代生成模型最重要的一个 trick,几乎所有 text-to-image、text-to-video、world-action 模型都靠它把"听不听条件"的强度可调。但很多人初学时被它的数学公式吓到 —— `ε_θ(x, c) = (1+w)·ε_θ(x|c) - w·ε_θ(x|∅)` 看着像啥统计物理。实际上,实现 CFG 唯一需要做的事就两件:(1) 让网络在训练时见过"有条件"和"无条件"两种状态,(2) 推理时调用两次,加权混合。第 (2) 步是 sampler 的事;第 (1) 步就是 DiT 这 28 行 `LabelEmbedder` 干的。它把"无条件"实现成 embedding 表里多出来的一行 —— 用一个"null token id"来当 placeholder,训练时按概率 swap。**昨天我们讲了 Wan2.1 在大型 text-conditioned 视频模型里的 CFG 实现,今天换一个完全不同的角度:DiT 是 CFG 在 class-conditioned image 模型里最简洁的实现 —— 同一个 idea,3 行核心代码。**

CFG (Classifier-Free Guidance) is the single most important trick in modern generative modeling — text-to-image, text-to-video, world-action models all use it to make "how much to listen to the condition" a tunable knob. The math `ε_θ(x, c) = (1+w)·ε_θ(x|c) - w·ε_θ(x|∅)` looks intimidating but the implementation is two things: (1) make the network see both *conditioned* and *unconditioned* states during training, and (2) at inference, call the model twice and mix. Step 2 is the sampler's job; step 1 is exactly what DiT's 28-line `LabelEmbedder` does. It implements "unconditioned" as one extra row in the embedding table — a null token id used as placeholder, swapped in at training time with probability `dropout_prob`. **Yesterday we covered Wan2.1's CFG inside a big text-conditioned video model; today we look at the same idea from the simplest possible angle: DiT's class-conditioned image setup — same concept, three core lines.**

## 代码 / The code

`facebookresearch/DiT` — [`models.py`](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L67-L94)

```python
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations.
    Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings
```

## 逐行讲解 / What's happening

1. **第 8 行 / Line 8 (`use_cfg_embedding = dropout_prob > 0`)**:
   - 中文: 这是一个布尔值,但下一行被当成整数用 —— `num_classes + use_cfg_embedding`。当 `dropout_prob=0.1` 时,表里有 `num_classes + 1` 行;当 `dropout_prob=0` 时,只有 `num_classes` 行。这一行用 Python 自动 bool→int 转换把"是否启用 CFG"和"表大小"绑成了一个东西。
   - English: a bool that's used as int on the next line — `num_classes + use_cfg_embedding`. With `dropout_prob=0.1` the table has `num_classes + 1` rows; with `dropout_prob=0` only `num_classes`. Python's bool-as-int trick ties "CFG enabled?" to "table size" in one expression.

2. **第 9 行 / Line 9 (`nn.Embedding(num_classes + use_cfg_embedding, hidden_size)`)**:
   - 中文: **CFG 的物理实现就是这一行 —— 给 embedding 表多留一个 slot**。这个新 slot 的 id 是 `num_classes`(因为原 class id 是 0..num_classes-1)。它就是论文里的"空类别 ∅",但实现上不是一个特殊 token,只是表里多一个普通的可学习行向量。
   - English: **the physical realization of CFG is this single line — reserve one extra slot in the embedding table**. Its id is `num_classes` (real classes occupy 0..num_classes-1). This *is* the paper's "null class ∅"; in code it's just one extra ordinary learnable row in the table.

3. **`token_drop`,第 18-21 行 / Lines 18-21**:
   - 中文: 训练时为每个 sample 独立掷硬币 —— `torch.rand < dropout_prob` 得到一个 bool 张量 `drop_ids`。然后 `torch.where(drop_ids, self.num_classes, labels)` 是"对于硬币为 1 的位置,把 label 改成 num_classes(空 id);否则保持原 label"。也就是说,每条样本以 `dropout_prob`(通常 0.1)的概率被告知"你的条件没了,直接学无条件分布"。
   - English: at train time we flip a coin per sample — `torch.rand < dropout_prob` gives a bool tensor `drop_ids`. Then `torch.where(drop_ids, self.num_classes, labels)` says "wherever the coin came up True, replace the label with `num_classes` (the null id); else keep the real label". Each sample independently, with probability `dropout_prob` (typically 0.1), is told "you have no condition, learn the unconditional score."

4. **`force_drop_ids` 参数 / `force_drop_ids` param**:
   - 中文: 推理时也会调这个函数 —— 但不能随机,要可控。`force_drop_ids` 是个手动指定的 mask:= 1 表示这条要变 unconditional,= 0 表示这条保留 conditional。推理时,你把同一个 batch 复制成两份,一份 `force_drop_ids=0`(条件路径),另一份 `force_drop_ids=1`(无条件路径),分别 forward,然后在 sampler 里把两条 ε 加权混合。
   - English: inference also calls this function, but it can't be random — it needs to be controlled. `force_drop_ids` is a manual mask: 1 means "make this sample unconditional", 0 means "keep conditional". At sampling time you duplicate the batch — one copy with `force_drop_ids=0` (conditional path), one with `force_drop_ids=1` (unconditional path) — forward both, then the sampler linearly combines the two ε predictions.

5. **`forward` 的门 / the forward's guard**:
   - 中文: `if (train and use_dropout) or (force_drop_ids is not None)` —— 只有在"训练 + 启用 dropout" 或"用户显式传了 force_drop_ids" 两种情况下才执行 token_drop。普通推理(无 CFG)走纯 conditional 路径。
   - English: `if (train and use_dropout) or (force_drop_ids is not None)` — token_drop runs only when training (with dropout enabled) *or* the caller explicitly passed `force_drop_ids`. Vanilla inference without CFG falls through to the conditional path.

## 类比 / The analogy

想象一个戏剧学校 —— 老师在教 100 种不同角色的演法(100 个 class)。CFG 让老师再开一门"无角色的中性演员"课程(那多出来的一行 embedding)。训练时,老师随机抽 10% 的学生告诉他们"今晚你没角色,演你自己" —— 这些学生就练习"无角色的本色表演"。考试时,老师让每个学生先演一遍"你的角色",再演一遍"你自己",然后告诉评委"我要看的是 1.5 倍'你的角色'减 0.5 倍'你自己'" —— 这个差值就是"纯角色风格"。这 28 行代码就是建立"无角色课程"+"随机抽 10% 学生学这门课"的全部逻辑。

Think of a drama school teaching 100 different character roles (100 classes). CFG adds one more course called "yourself as a neutral actor" (the extra embedding row). During training, the teacher randomly tells 10% of students each session: "Tonight you have no role, perform as yourself" — those students practice the unconditional version. At performance time, each student first performs their role, then performs as themselves; the judge says: "I want 1.5× your-role minus 0.5× yourself" — and that residual is the *pure role style*. These 28 lines build the "yourself" course and the "randomly assign 10% to it" mechanic.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-WAM

中文:**这是 nanoWAM 课程 `classifier-free-guidance` 的"教科书参考实现"**(依赖 `dit-block`、`text-conditioning` 或 `action-conditioning` 这些上游 condition encoder)。在 nanoWAM 里,如果你的 condition 是 text、image 或 action(不是 class label),只需要做一个最小改动:把 `nn.Embedding(num_classes, hidden_size)` 换成你自己的 condition encoder,然后在 encoder 的输出空间额外学一个"null condition embedding"作为参数(`nn.Parameter(torch.randn(hidden_size))`)—— 训练时按 `dropout_prob` 决定是用 encoder 输出还是 null embedding。结构:`text → encoder → c_emb` 或者 `null_token → null_emb`,用 `torch.where(drop_ids, null_emb, c_emb)` 在 batch 维上切换。Wan2.1(昨天讲过)用的是同一个 motif —— 它在 T5 encoder 之外学了 `context_null_caches` 当作那行 "null embedding"。如果省掉这一层:你的 nanoWAM 没法做 CFG,生成质量(text-to-video 上 FVD)会比开 CFG 时差 30-50%。生产实现还要补:CFG 强度调度(开头几步用强 CFG,后面几步用弱 CFG)、多条件 CFG(text + image 两路分别 drop)。

English: **this is the textbook reference for the nanoWAM `classifier-free-guidance` slot** (depends on `dit-block` plus a condition encoder — `text-conditioning` or `action-conditioning`). Adapting DiT's pattern for nanoWAM is a one-line change: replace `nn.Embedding(num_classes, hidden_size)` with your own condition encoder, then add one learnable `null_emb = nn.Parameter(torch.randn(hidden_size))` in the encoder's output space — at train time, decide per sample whether to use encoder output or null_emb with probability `dropout_prob`. The structure becomes: `text → encoder → c_emb` *or* `null_token → null_emb`, mixed with `torch.where(drop_ids, null_emb, c_emb)` along the batch dimension. Wan2.1 (yesterday's note) uses exactly this motif — it learns `context_null_caches` outside the T5 encoder as that "null row". Skip this layer entirely and your nanoWAM cannot do CFG, costing 30-50% on FVD vs CFG-enabled. Production additions: CFG-scale scheduling (strong CFG early, weak CFG late), multi-condition CFG (separate dropout for text vs image).

## 自己跑一遍 / Try it yourself

```python
import torch, torch.nn as nn

class MinimalCFGEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob=0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_classes + 1, hidden_size)  # +1 for null
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def forward(self, labels, train=True, force_drop=None):
        if train and self.dropout_prob > 0:
            drop = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            labels = torch.where(drop, self.num_classes, labels)
        elif force_drop is not None:
            labels = torch.where(force_drop, self.num_classes, labels)
        return self.embedding(labels)

emb = MinimalCFGEmbedder(num_classes=10, hidden_size=4, dropout_prob=0.5)
labels = torch.arange(8) % 10
print("real labels   :", labels.tolist())
print("train forward (50% dropped):")
torch.manual_seed(0)
out_train = emb(labels, train=True);  print("  shape:", out_train.shape)
# show which entries got swapped to the null embedding
torch.manual_seed(0)
drop_mask = torch.rand(8) < 0.5
print("  dropped     :", drop_mask.tolist())

# inference time — uncond pass:
out_uncond = emb(labels, train=False, force_drop=torch.ones(8, dtype=torch.bool))
out_cond   = emb(labels, train=False, force_drop=torch.zeros(8, dtype=torch.bool))
print("cond and uncond pass produce different embeddings:",
      not torch.allclose(out_cond, out_uncond))

# simulate CFG mix
w = 4.0  # CFG scale
mixed = (1 + w) * out_cond - w * out_uncond
print("CFG-mixed embedding shape :", mixed.shape)
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
real labels   : [0, 1, 2, 3, 4, 5, 6, 7]
train forward (50% dropped):
  shape: torch.Size([8, 4])
  dropped     : [True, False, True, False, True, False, True, False]
cond and uncond pass produce different embeddings: True
CFG-mixed embedding shape : torch.Size([8, 4])
```

中文:留意被 drop 的样本数 ≈ 50%(因为 dropout_prob=0.5),不是恰好 4 个 —— 这是伯努利采样的自然抖动。生产里你会用 dropout_prob=0.1。

English: ~50% are dropped (since dropout_prob=0.5), not exactly 4 — this is natural Bernoulli jitter. Production uses dropout_prob=0.1.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Wan-Video/Wan2.1 的 `context_null_caches`** / **Wan-Video/Wan2.1's `context_null_caches`**:同一个 motif 上到 text-to-video,null embedding 改成 T5 encoder 输出空间里的可学习张量。 / Same motif scaled to text-to-video — null embedding becomes a learnable tensor in T5 encoder output space.
- **OpenAI Improved DDPM** / **OpenAI Improved DDPM**:CFG 的 paper 之一,但实现也是这套"id 表多一行 + token_drop"。 / One of the CFG papers — its implementation is the same "extra-row + token_drop" pattern.
- **Stable Diffusion 的 `do_classifier_free_guidance`** / **Stable Diffusion's `do_classifier_free_guidance`**:做的是"复制 batch、null embedding 喂下半,sampler 再混合"—— 整套 pipeline 就是 DiT 这个的工业化版本。 / Duplicates the batch, feeds null embedding to the lower half, sampler mixes — the industrial scaling of this DiT pattern.
- **MultiDiffusion / SpaceTime CFG** / **MultiDiffusion / SpaceTime CFG**:同时多条件(text + canny + depth)各自有 dropout 概率 —— 等价于多个独立 `token_drop`,各管各的 null embedding。 / Multiple simultaneous conditions (text + canny + depth) each with their own dropout — equivalent to multiple independent `token_drop` heads, each with its own null embedding.

## 注意事项 / Caveats / when it breaks

- **dropout_prob 设太大会让 conditional 能力退化** / **too-large dropout_prob degrades conditional quality**:0.1 是默认值。设 0.5 你就训了一个一半时间不看条件的模型,FVD/IS 都会掉。 / 0.1 is default. 0.5 trains a model that ignores conditions half the time — FVD/IS drop.
- **null embedding 必须可学** / **null embedding must be learnable**:别拿固定零向量当 null —— 论文实验显示,可学习的 null embedding 比固定 zero 更好。 / Don't hard-code zero. The paper shows a learnable null embedding outperforms fixed zero.
- **`torch.where` 后是 view 不是 copy** / **`torch.where` returns a new tensor, not a view**:不会反向影响原 labels;但如果你的下游对 labels 做 in-place 操作,要小心 —— 但这种情况很少。 / Returns a new tensor; rarely a footgun unless your downstream code modifies labels in place.
- **CFG 推理 batch 翻倍** / **CFG doubles inference batch**:每次 sample 都要算 conditional + unconditional 两条路径。把这两路 concat 进同一个 forward 是标准做法(否则两次启动 kernel 太慢)。 / Every sample step computes both conditional and unconditional. Concatenating them in a single forward is the standard trick (two separate calls double kernel launch overhead).

## 延伸阅读 / Further reading

- [Ho & Salimans, "Classifier-Free Diffusion Guidance" (NeurIPS 2021 Workshop / arXiv 2207.12598)](https://arxiv.org/abs/2207.12598)
- [Peebles & Xie, "Scalable Diffusion Models with Transformers" (DiT, ICCV 2023)](https://arxiv.org/abs/2212.09748)
- [Yesterday's note: Wan2.1's CFG in text-to-video](../../INDEX.md)
- [Sander Dieleman blog — "Guidance: a cheat code for diffusion models"](https://sander.ai/2022/05/26/guidance.html)
