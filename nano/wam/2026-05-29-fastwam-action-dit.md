---
date: 2026-05-29
topic: wam
source: wam
repo: yuantianyuan01/FastWAM
file: src/fastwam/models/wan22/action_dit.py
permalink: https://github.com/yuantianyuan01/FastWAM/blob/45d8e1458921d83f8ad6cf9ce993d371208dabd0/src/fastwam/models/wan22/action_dit.py#L18-L98
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, wam, action-dit, parallel-architecture, decoupled]
build_role: Action head (decoupled-DiT design) — a full parallel transformer dedicated to actions
---

# FastWAM 给 action 单独搭了一个完整的 DiT / FastWAM spins up a full second DiT just for actions

> **一句话 / In one line**: FastWAM 不让 action 蹭 video 的 transformer,而是 import 同一份 `DiTBlock` 单独搭一个小型 `ActionDiT`,前面 `nn.Linear(action_dim → hidden)` 编码、中间 N 个 DiTBlock、末尾用带 adaLN-Zero 调制的 `ActionHead` 解码 —— 视频 DiT 和 action DiT 是**两个独立模型**,通过共享的 text/time embedding 协同。 / FastWAM doesn't piggyback actions on the video transformer — it imports the same `DiTBlock` and builds a separate `ActionDiT`. Input: `Linear(action_dim → hidden)`; trunk: N stacked DiTBlocks; output: `ActionHead` with adaLN-Zero modulation. Video DiT and action DiT are two distinct models that coordinate only through shared text/time embeddings.

## 为什么重要 / Why this matters

跟 lingbot-va 把 action 当 token 塞进同一个 DiT 不同,FastWAM 选了"双 DiT 并行"路线:一个 DiT 负责 video,一个 DiT 负责 action,两者参数完全分离。这种做法的优点是 (1) action 的训练/推理速度**与 video 序列长度脱钩** —— 你可以 video DiT 50 步、action DiT 4 步;(2) action 模型可以小得多(几百 M 参数 vs 14B),部署时单独 serve 给机器人本体;(3) 两边可以用不同的 noise schedule、不同的 lr;(4) 同一个 action DiT 可以挂在多种 video DiT 后面("backbone-pluggable")。代价是失去 video → action 的细粒度联合建模 —— 两者只通过 text 间接耦合。这一节看 FastWAM 怎么把它实现成 ~100 行。

Unlike lingbot-va's "actions are just tokens in the video DiT", FastWAM goes parallel: one DiT for video, a separate DiT for actions, fully independent parameters. The wins are (1) action inference cost is **decoupled from video sequence length** — you can run video DiT for 50 steps and action DiT for 4; (2) the action model can be tiny (hundreds of M params vs 14 B for video); (3) different noise schedules and learning rates per side; (4) the same `ActionDiT` plugs into many video backbones ("backbone-pluggable"). The trade-off is losing fine-grained joint video↔action modelling — coupling happens only through the shared text encoder. These 100 lines are the implementation.

## 代码 / The code

`yuantianyuan01/FastWAM` — [`src/fastwam/models/wan22/action_dit.py`](https://github.com/yuantianyuan01/FastWAM/blob/45d8e1458921d83f8ad6cf9ce993d371208dabd0/src/fastwam/models/wan22/action_dit.py#L18-L98)

```python
class ActionHead(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.proj = nn.Linear(hidden_dim, out_dim)
        self.modulation = nn.Parameter(
            torch.randn(1, 2, hidden_dim) / hidden_dim**0.5)

    def forward(self, x, t):
        # adaLN-Zero style modulation, scaled by the timestep embedding
        shift, scale = (self.modulation.to(t.dtype).to(t.device)
                        + t.unsqueeze(1)).chunk(2, dim=1)
        shift = shift.squeeze(1)
        scale = scale.squeeze(1)
        return self.proj(self.norm(x) * (1 + scale.unsqueeze(1))
                                       + shift.unsqueeze(1))


class ActionDiT(nn.Module):
    ACTION_BACKBONE_SKIP_PREFIXES = ("action_encoder.", "head.")

    def __init__(self, hidden_dim, action_dim, ffn_dim, text_dim, freq_dim,
                 eps, num_heads, attn_head_dim, num_layers,
                 use_gradient_checkpointing=False):
        super().__init__()
        self.action_encoder  = nn.Linear(action_dim, hidden_dim)        # in
        self.text_embedding  = nn.Sequential(
            nn.Linear(text_dim,   hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim))
        self.time_embedding  = nn.Sequential(
            nn.Linear(freq_dim,   hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.time_projection = nn.Sequential(nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 6))                     # 6 adaLN-Zero coefs
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim=hidden_dim, attn_head_dim=attn_head_dim,
                     num_heads=num_heads, ffn_dim=ffn_dim, eps=eps)
            for _ in range(num_layers)])
        self.head  = nn.Linear(hidden_dim, action_dim)                 # out (or use ActionHead)
        self.freqs = precompute_freqs_cis(attn_head_dim, end=1024)     # 1-D RoPE for actions
        self.use_gradient_checkpointing = use_gradient_checkpointing

    def pre_dit(self, action_tokens, timestep, context, context_mask=None):
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))

        tokens      = self.action_encoder(action_tokens)
        context_emb = self.text_embedding(context)
        freqs       = self.freqs[:action_tokens.shape[1]].view(-1, 1, 1, self.attn_head_dim // 2)

        return {"tokens": tokens, "freqs": freqs, "t": t, "t_mod": t_mod,
                "context": context_emb, "context_mask": context_mask}
```

## 逐行讲解 / What's happening

1. **`ACTION_BACKBONE_SKIP_PREFIXES` 暴露设计哲学 / The skip-prefix tells you the philosophy**:
   - 中文:`("action_encoder.", "head.")` 这两个前缀在加载 video DiT 预训练时被**跳过**—— 也就是说 FastWAM 的工作流是"先训 video DiT,然后 freeze 大部分参数,只新加 `action_encoder` 和 `head` 两个 Linear 训练 action"。这就是 "backbone-pluggable" 的具体含义。
   - English: `("action_encoder.", "head.")` are skipped when loading the video DiT pretrain — i.e. the workflow is "train the video DiT, freeze most of it, only the new `action_encoder` and `head` Linears get fine-tuned for action". That's what "backbone-pluggable" means concretely.

2. **`action_encoder = nn.Linear(action_dim, hidden_dim)` 跟 lingbot-va 一样 / Same input projector as lingbot-va**:
   - 中文:输入侧两家其实一致,**差异从这里之后开始** —— lingbot 把 token 拼进 video 序列共享 trunk,FastWAM 立刻喂进自己的 `self.blocks`。
   - English: the input projector is identical to lingbot-va. **The split happens next**: lingbot concatenates into the video sequence and shares the trunk; FastWAM hands the tokens to its own private `self.blocks`.

3. **完整的 `text_embedding` / `time_embedding` / `time_projection` 链条 / The full embedding chain**:
   - 中文:`text_embedding` 把 T5 输出投到 action DiT 的 hidden;`time_embedding` 把 timestep 投到 hidden;`time_projection` 再扩展到 `6 * hidden_dim`,这 6 段对应每个 DiTBlock 里的 adaLN-Zero 调制(`shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp`)。action DiT 有自己的一整套,**不复用 video DiT 的**。
   - English: `text_embedding` lifts T5 output into action-DiT hidden; `time_embedding` lifts timestep; `time_projection` blows it to `6 * hidden_dim`, the six adaLN-Zero coefficients per DiTBlock (`shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp`). Action DiT carries its own — **the video DiT's embedding is not reused**.

4. **`DiTBlock` 是 import 的,**不是复制粘贴**** / `DiTBlock` is imported, not copy-pasted**:
   - 中文:`from .wan_video_dit import DiTBlock` —— action DiT 和 video DiT 用**同一份代码** DiTBlock,只是堆出来的 stack 不一样。这给课程式构建带来便利:你建好了 video DiT 的 block,action DiT 完全免费复用,代码 0 行新增。
   - English: `from .wan_video_dit import DiTBlock` — action DiT and video DiT share the *same class*; only the stack length differs. Once you've built the video DiT block, action DiT reuses it for free, no new lines required.

5. **`ActionHead` 用 adaLN-Zero 风格的调制 / `ActionHead` uses an adaLN-Zero-style modulation**:
   - 中文:跟 DiT 文章里的 head 几乎一模一样 —— `LayerNorm(x) * (1 + scale) + shift`,然后 `proj`。`modulation` 是 learnable 偏置,加在 timestep embedding 上;`scale = 0, shift = 0` 时输出就是普通 LayerNorm + Linear,所以 zero-init 时这个 head 是恒等映射,训练初期稳定。
   - English: nearly identical to the DiT paper's head — `LayerNorm(x) * (1 + scale) + shift` then `proj`. `modulation` is a learnable bias added to the timestep embedding; at zero init `(scale, shift) = 0` makes the head act as a plain LayerNorm + Linear, giving stable early training (the adaLN-Zero recipe).

6. **`pre_dit` 把所有准备工作打包成一个 dict / `pre_dit` packages all preprocessing**:
   - 中文:这是 FastWAM 的工程美学 —— `pre_dit` 输入 token + context + timestep,输出一个包含 `tokens / freqs / t / t_mod / context / context_mask` 的字典;forward 时把 dict 解开依次过 blocks。这种"前置 → 主体 → 后置"切片让 LoRA / gradient checkpoint / TRT 编译都很容易切入。
   - English: a stylistic touch — `pre_dit` takes raw `(tokens, context, timestep)` and returns a dict with `tokens / freqs / t / t_mod / context / context_mask`; the forward unpacks it and walks the blocks. Splitting "pre → trunk → post" makes LoRA / gradient checkpointing / TRT compilation easy to slot in between stages.

7. **`precompute_freqs_cis(attn_head_dim, end=1024)` 是 1-D RoPE / 1-D RoPE for action tokens**:
   - 中文:action 是时间序列(没有空间维),所以用 1-D RoPE 就够了(对比 video 的 3-D RoPE,前两天的笔记)。`end=1024` 是支持的最长 action 序列长度,1024 帧 robot 动作通常绰绰有余。
   - English: action tokens form a 1-D time series (no spatial dims), so 1-D RoPE suffices (contrast with video's 3-D RoPE from earlier notes). `end=1024` caps the action sequence length — plenty for robot trajectories.

## 类比 / The analogy

像一家公司里有两条生产线:主线生产电视机(video DiT),旁线生产遥控器(action DiT)。两条线用一样的"装配工位"(`DiTBlock` 复用),但人员、工序、节拍各自独立。两条线唯一的交流是订单系统(`text_embedding` 由市场部统一下发),不互相督查;这样大电视生产慢时,小遥控器还能照样下线。

Imagine a factory with two production lines: TVs (video DiT) on one floor, remote controls (action DiT) on another. They share the same workstation design (`DiTBlock` reused) but have independent crews, schedules, and rhythms. Their only interface is the central order system (`text_embedding` from marketing). When TVs slow down, remotes still ship on time.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-WAM

中文:在 nanoWAM 里这是 `nano/wam/model/action_dit.py`,跟 `video_dit.py` 平级。上游:从同一份 dataloader 拿 `actions`、`text_emb`、`action_timestep`(注意 timestep 跟 video 独立);下游:输出 `action_pred` 直接送 loss,推理时直接给机器人。它的依赖图很干净 —— 只依赖 `DiTBlock`(2026-05-25 dit-adaln-zero 笔记)、`sinusoidal_embedding_1d`、`precompute_freqs_cis`,**完全不依赖 video VAE / video 3D RoPE / video sampler**。这就是 FastWAM 的核心卖点:可以独立部署 action DiT 到机器人,video DiT 留在云端。生产实现要补:(1) **action VAE / 归一化**(joint angles 维度跨关节差异大,要按关节归一化);(2) **multi-modal 输入**(把当前 image embedding 也塞进 cross-attention,让 action DiT 见到当前观测);(3) **few-step distillation**(把 action DiT 蒸到 1-2 步以达到 50Hz 控制频率)。

English: in nanoWAM this lives at `nano/wam/model/action_dit.py`, peer to `video_dit.py`. Upstream: pull `actions`, `text_emb`, and an *independent* `action_timestep` from the same dataloader. Downstream: feed `action_pred` to the loss in training, send it straight to the robot at inference. The dependency graph is wonderfully clean — only `DiTBlock`, `sinusoidal_embedding_1d`, and `precompute_freqs_cis`. **It does not depend on the video VAE, video 3-D RoPE, or video sampler.** That's FastWAM's headline benefit: ship action DiT to the robot, keep video DiT in the cloud. Production additions: (1) **action VAE / per-joint normalisation** (joint-angle ranges vary across joints), (2) **multi-modal conditioning** (cross-attend to current image embedding too, so action DiT sees current observation), (3) **few-step distillation** (compress action DiT down to 1-2 steps for 50 Hz control).

## 自己跑一遍 / Try it yourself

```python
# pip install torch
# Minimal ActionDiT: input -> N DiTBlocks -> ActionHead
import torch, torch.nn as nn, math

torch.manual_seed(0)
B, T_action, action_dim, hidden, heads, n_layers = 2, 8, 7, 64, 4, 3

class TinyDiTBlock(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d, h, batch_first=True)
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        self.ffn   = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(self, x, t_mod):
        # adaLN-Zero stub: scale/shift from timestep modulation
        shift, scale, gate = t_mod[:, 0], t_mod[:, 1], t_mod[:, 2]
        h = self.norm1(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = x + gate.unsqueeze(1) * self.attn(h, h, h)[0]
        return x + self.ffn(self.norm2(x))

class ActionHead(nn.Module):
    def __init__(self, d, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(d, elementwise_affine=False)
        self.proj = nn.Linear(d, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, d) / d ** 0.5)
    def forward(self, x, t):
        shift, scale = (self.modulation + t.unsqueeze(1)).chunk(2, dim=1)
        return self.proj(self.norm(x) * (1 + scale.squeeze(1).unsqueeze(1))
                                       + shift.squeeze(1).unsqueeze(1))

action_in   = nn.Linear(action_dim, hidden)
t_emb       = nn.Linear(1, hidden)
t_mod_proj  = nn.Linear(hidden, hidden * 3)              # 3 instead of 6 for the toy block
blocks      = nn.ModuleList([TinyDiTBlock(hidden, heads) for _ in range(n_layers)])
head        = ActionHead(hidden, action_dim)

actions  = torch.randn(B, T_action, action_dim)
timestep = torch.rand(B, 1)
tokens   = action_in(actions)
t        = t_emb(timestep)                                # [B, hidden]
t_mod    = t_mod_proj(t).unflatten(1, (3, hidden))        # [B, 3, hidden]
for blk in blocks:
    tokens = blk(tokens, t_mod)
out = head(tokens, t)
print("action prediction shape:", out.shape)              # [B, T_action, action_dim]
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
action prediction shape: torch.Size([2, 8, 7])
```

中文:30 行代码就是一个独立的 mini-ActionDiT。FastWAM 真实代码加上了 LoRA、TRT 编译、KV cache 等,核心骨架就是这样。

English: 30 lines for a working mini-ActionDiT. The production FastWAM adds LoRA, TRT compilation, KV caching, but the skeleton is exactly this.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **SmolVLA 的 expert / SmolVLA's expert**: 中文 — 同样思路:大 VLM + 小专家。SmolVLA 用 cross-attention 让 expert 读 VLM 的 KV,FastWAM 让 ActionDiT 完全独立。 / English — same big-model-plus-small-expert idea. SmolVLA's expert cross-attends into the VLM; FastWAM's ActionDiT is fully independent.
- **OpenVLA / RT-2 / OpenVLA / RT-2**: 中文 — 反向:不分两个 DiT,整个 VLM 同时做视觉理解和动作预测。资源消耗大但联合建模更强。 / English — opposite extreme: one VLM does both vision and action. Heavier but tighter coupling.
- **π₀ 的 flow-matching head / π₀'s flow-matching head**: 中文 — π₀ 是混合:用 PaliGemma 当 backbone,然后挂一个 flow-matching action head(几层独立的 transformer)。介于 FastWAM 和 lingbot-va 之间。 / English — hybrid: PaliGemma backbone + flow-matching action head (a few independent transformer layers). Sits between FastWAM and lingbot-va.

## 注意事项 / Caveats / when it breaks

- **`text_embedding` 不能复用 video DiT 的** / **Do not share `text_embedding` with the video DiT**: 中文 — 同样的 T5 输入,action DiT 需要把它压到自己更小的 hidden_dim,所以参数必须独立。否则 action DiT 表达能力上限被 video DiT 的 hidden_dim 卡死。 / English — even though the T5 input is shared, action DiT needs to compress it into its (smaller) hidden_dim. Sharing the parameter caps action DiT's capacity to the video DiT's hidden_dim.
- **`precompute_freqs_cis(end=1024)` 是硬上限** / **`end=1024` is a hard ceiling**: 中文 — 超长 trajectory(> 1024 步)会越界。生产里要么调大,要么截断。 / English — long trajectories beyond 1024 frames hit the precomputed RoPE limit. Bump `end` or truncate.
- **action DiT 的 noise schedule 通常更轻** / **Action DiT usually wants a lighter schedule**: 中文 — video diffusion 训练时 sigma 范围大(0.003 - 1),action 通常 0 - 0.5 就够(因为机器人动作流形比图像窄)。盲目复用 video 的 scheduler 会让 action 训练浪费在高噪声端。 / English — video diffusion runs sigma up to ~1; actions usually need only 0-0.5 (the action manifold is much narrower). Reusing video's scheduler wastes capacity on noise levels that never matter for actions.
- **CFG scale 也要分别调** / **Tune CFG scale separately**: 中文 — video 的 CFG=5-9 不一定适合 action;过强 CFG 会让动作过冲,失去平滑性。 / English — video's CFG=5-9 doesn't transfer to actions. High CFG on actions overshoots and breaks motion smoothness.

## 延伸阅读 / Further reading

- [FastWAM repo](https://github.com/yuantianyuan01/FastWAM)
- [Today's lingbot-va action note (single-stream design)](./2026-05-29-lingbot-action-embedder.md)
- [Today's dreamzero action register note (integrated registers)](./2026-05-29-dreamzero-action-registers.md)
- [DiT paper — adaLN-Zero head design](https://arxiv.org/abs/2212.09748)
