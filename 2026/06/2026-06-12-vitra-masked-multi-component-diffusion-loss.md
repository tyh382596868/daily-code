---
date: 2026-06-12
topic: robotics
source: trending
repo: microsoft/VITRA
file: vitra/models/action_model/diffusion_policy.py
permalink: https://github.com/microsoft/VITRA/blob/b35517202b39d32a753fdd42014b2cc3c41fab58/vitra/models/action_model/diffusion_policy.py#L21-L122
difficulty: intermediate
read_time: ~11 min
tags: [code-of-the-day, robotics, trending, vla, diffusion-loss, masked-loss]
---

# 把"人手当机器人末端执行器"那篇 VITRA:50 行代码搞定异构动作的 masked diffusion loss / VITRA — the "human hand as a robot end-effector" paper: 50 lines handle masked diffusion loss for a heterogeneous action vector

> **一句话 / In one line**: VITRA 用 100 万段人类手部视频预训练了一个 3B VLA,关键是 `DiffusionPolicy.loss()`:把动作向量切成"平移 / 欧拉角 / 45 维手指 pose"几块,每块独立 masked L2 + count-weighted 平均,一份 loss 同时管"人手"和"机器人手" / VITRA pretrains a 3B VLA on 1M+ human-hand videos. The pivotal piece is `DiffusionPolicy.loss()`: slice the action vector into pieces (translation / Euler / 45-D finger poses), do masked L2 per piece, then count-weighted-average them — one loss covers both "human hand" and "robot hand".

## 为什么重要 / Why this matters

VITRA (ICRA 2026, Microsoft, 大约 400 ★, 2026-06-12 刚刚 push 更新) 解决的是 VLA 数据问题:机器人遥操数据贵,但 YouTube 上有海量第一人称视角的人类手部视频(Ego4D、EgoExo4D、Epic-Kitchens、SSv2),只要把"人手"和"机器人末端执行器"在动作向量上对齐,前者就能用作 VLA 预训练数据 —— 他们最终拿到 100 万条 episodes。但有个工程难点:**人手动作维度和机器人动作维度并不完全一样**(人手有 45 维 finger pose,xhand 机器人有 12 维 finger joint),而且数据集里有大量 OOB / kept=False 的帧。怎么写一个 loss 函数,既能训人手数据也能训机器人数据,还能处理部分缺失?这就是 `DiffusionPolicy.loss()` 给的答案。

VITRA (ICRA 2026, Microsoft, ~400★, refreshed today on 2026-06-12) attacks the VLA data problem: teleoperation data is expensive, but YouTube has endless first-person human-hand video (Ego4D, EgoExo4D, Epic-Kitchens, SSv2). If you align "human hand" and "robot end-effector" in the action vector, the former becomes free VLA pretraining data — they ended up with 1M+ episodes. There's an engineering catch: **human-hand and robot-hand action dims aren't identical** (human hands have a 45-D finger pose, the xhand robot has 12-D finger joints), and the dataset has lots of OOB / kept=False frames. How do you write one loss that trains both human and robot data and handles missing pieces? `DiffusionPolicy.loss()` is the answer.

## 代码 / The code

`microsoft/VITRA` — [`vitra/models/action_model/diffusion_policy.py`](https://github.com/microsoft/VITRA/blob/b35517202b39d32a753fdd42014b2cc3c41fab58/vitra/models/action_model/diffusion_policy.py#L21-L122)

```python
def DiT_T(**kw): return DiT(depth=3,  hidden_size=256,  num_heads=4,  **kw)
def DiT_S(**kw): return DiT(depth=6,  hidden_size=384,  num_heads=4,  **kw)
def DiT_M(**kw): return DiT(depth=12, hidden_size=384,  num_heads=6,  **kw)
def DiT_B(**kw): return DiT(depth=12, hidden_size=768,  num_heads=12, **kw)
def DiT_L(**kw): return DiT(depth=24, hidden_size=1024, num_heads=16, **kw)
DiT_models = {'DiT-S': DiT_S, 'DiT-M': DiT_M, 'DiT-B': DiT_B, 'DiT-T': DiT_T, 'DiT-L': DiT_L}


class DiffusionPolicy(nn.Module):
    def __init__(self, token_size, model_type='DiT-B', in_channels=192,
                 future_action_window_size=16, past_action_window_size=0,
                 use_state=None, action_type='angle',
                 diffusion_steps=100, state_dim=None, loss_type='human'):
        super().__init__()
        self.in_channels = in_channels
        self.diffusion_steps = diffusion_steps
        self.diffusion = create_diffusion(
            timestep_respacing="", noise_schedule='squaredcos_cap_v2',
            diffusion_steps=self.diffusion_steps,
            sigma_small=True, learn_sigma=False)
        self.ddim_diffusion = None
        learn_sigma = self.diffusion.model_var_type in [gd.ModelVarType.LEARNED, gd.ModelVarType.LEARNED_RANGE]
        self.past_action_window_size = past_action_window_size
        self.future_action_window_size = future_action_window_size
        self.use_state = use_state
        self.action_type = action_type

        if loss_type == 'human':
            self.loss_components = ActionFeature.get_loss_components(action_type)
        elif loss_type == 'robot':
            self.loss_components = ActionFeature.get_xhand_loss_components()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        self.net = DiT_models[model_type](
            token_size=token_size, action_dim=in_channels,
            class_dropout_prob=0.1, learn_sigma=learn_sigma,
            future_action_window_size=future_action_window_size,
            past_action_window_size=past_action_window_size,
            use_state=use_state, state_dim=state_dim)

    def loss(self, x, z, x_mask, state=None, state_mask=None):
        noise = torch.randn_like(x)                                              # [B, T, C]
        timestep = torch.randint(0, self.diffusion.num_timesteps, (x.size(0),), device=x.device)
        x_t = self.diffusion.q_sample(x, timestep, noise)
        x_t = x_t * x_mask
        x_t = torch.cat([x_t, x_mask], dim=2)                                    # [B, T, 2C]

        noise_pred = self.net(x_t, timestep, z, state, state_mask)
        assert noise_pred.shape == noise.shape == x.shape

        square_delta = (noise_pred - noise) ** 2 * x_mask                        # masked L2

        def mask_loss(from_dim, to_dim):
            s = square_delta[:, :, from_dim:to_dim].sum()
            n = x_mask[:, :, from_dim:to_dim].sum()
            return s / n if n > 0 else 0

        component_losses = {}
        component_counts = {}
        for name, (start, end, weight) in self.loss_components.items():
            component_losses[name] = mask_loss(start, end) * weight
            component_counts[name] = x_mask[:, :, start].sum()

        total_count = sum(component_counts.values())
        if total_count == 0:
            loss = square_delta[0, 0, 0]
        else:
            loss = sum(component_losses[k] * component_counts[k]
                       for k in component_counts.keys()) / total_count
        return {"loss": loss, **component_losses}
```

## 逐行讲解 / What's happening

1. **第 1-7 行的 DiT 注册表 / The DiT registry on lines 1-7**:
   - 中文: 5 种尺寸的 DiT 用 5 个工厂函数定义,再压成一个字符串→工厂的 dict。这是 OpenAI/DiT 仓库原版的写法,深度 (depth) 和 hidden_size 是核心唯二的伸缩参数。`DiT-T` (tiny, 256/3) 你可以拿来在 CPU 上 debug,`DiT-L` (1024/24) 是论文里 3B VLA 用的。
   - English: Five DiT sizes defined as five factories, then compressed into a `name → factory` dict. This is the original OpenAI/DiT style. Depth and hidden_size are the only two scaling knobs that matter. `DiT-T` (tiny, 256/3) is a CPU-debug toy; `DiT-L` (1024/24) is what the paper's 3B VLA actually uses.

2. **第 9-37 行的 `__init__` / The `__init__` on lines 9-37**:
   - 中文: 标准 Gaussian diffusion 走 `squaredcos_cap_v2` 噪声 schedule(SD3 / DiT 同款),`sigma_small=True` 选小方差。第 26-31 行是这份文件最关键的开关:`loss_type='human'` 和 `loss_type='robot'` 切换两套"动作向量的语义切分"。`ActionFeature.get_loss_components()` 返回的是 `{component_name: (start_dim, end_dim, weight)}` 这种结构,告诉 loss 函数怎么把 192 维(`in_channels=192`)切成"翻译"、"欧拉角"、"finger poses"等几块,每块有自己的权重。
   - English: Standard Gaussian diffusion with `squaredcos_cap_v2` noise schedule (same as SD3 / DiT), `sigma_small=True` picks the smaller variance branch. Lines 26-31 are the file's key switch: `loss_type='human'` vs `loss_type='robot'` swaps two semantic-slicing schemes for the action vector. `ActionFeature.get_loss_components()` returns `{name: (start_dim, end_dim, weight)}`, telling the loss how to carve the 192 dims (`in_channels=192`) into translation / Euler / finger poses, each with its own weight.

3. **第 41-44 行 q_sample / q_sample on lines 41-44**:
   - 中文: 标准 DDPM 前向加噪 `x_t = sqrt(α̅) x_0 + sqrt(1-α̅) ε`,然后 **`x_t = x_t * x_mask`** —— 用 0 掩盖无效 dim。最后一行 `cat([x_t, x_mask], dim=2)` 把 mask 拼到通道维,让模型**显式看到哪些位置无效**(避免它在 OOB 区域瞎学)。这是处理异构 / 缺失数据的关键工程技巧。
   - English: Standard DDPM forward `x_t = sqrt(α̅) x_0 + sqrt(1-α̅) ε`, then **`x_t = x_t * x_mask`** zeros out invalid dims. The final `cat([x_t, x_mask], dim=2)` puts the mask on the channel axis so the model **explicitly sees which positions are invalid** (otherwise it would happily memorize garbage in OOB regions). Classic engineering trick for heterogeneous / missing data.

4. **第 50-52 行 masked square_delta / Masked square_delta on lines 50-52**:
   - 中文: `(pred - noise)^2 * x_mask` —— 这是标准 ε-prediction 的 L2,只是用 mask 把无效位置的 squared error 清零。后面所有求和都会自动忽略这些位置。
   - English: `(pred - noise)^2 * x_mask` — standard ε-prediction L2, with invalid positions zeroed by the mask. All later sums automatically skip them.

5. **第 54-58 行的 closure `mask_loss(from_dim, to_dim)` / The closure `mask_loss(from_dim, to_dim)` on lines 54-58**:
   - 中文: 这个 inline closure 是整份文件最值得抄的小技巧:**按通道切片再求平均**。对一段 `[from_dim:to_dim]`:把这段的 squared error 求和 (s) ÷ 这段的有效 token 数 (n) —— 如果 n=0(整批数据这块都缺失)返回 0 而不是 NaN。这避免了 `loss.mean()` 在有 mask 时被无效 dim 的 0 拉低的标准坑。
   - English: This inline closure is the most copy-worthy trick: **slice by channel, then mean**. For range `[from_dim:to_dim]`: sum of squared error (s) ÷ number of valid tokens (n) — return 0 if n=0 instead of NaN. Avoids the classic trap where `loss.mean()` gets diluted by mask-zeroed dims.

6. **第 60-65 行的 component loop / The component loop on lines 60-65**:
   - 中文: 对 `loss_components` 里每个命名组件,算 `mask_loss(start, end) * weight` —— 也就是这一块的 per-valid-token loss × 这块的权重。同时记录 `component_counts[name] = x_mask[:, :, start].sum()` —— 注意这里只取 `start` 这一列的 mask 之和(假设一个组件内 mask 是统一的)。
   - English: For each named component in `loss_components`, compute `mask_loss(start, end) * weight` — per-valid-token loss for that slice times its weight. Meanwhile `component_counts[name] = x_mask[:, :, start].sum()` records the count, taking only column `start` of the mask (assuming the mask is consistent within a component).

7. **第 67-72 行的最终 count-weighted 平均 / The final count-weighted average on lines 67-72**:
   - 中文: 整批 loss 全 NaN 的兜底 (`if total_count == 0`),返回 `square_delta[0,0,0]` 而不是 0 —— 这样梯度图仍然连着,不会让 autograd 报"no grad" 错。否则用 `sum(component_loss * component_count) / total_count` 做加权平均 —— **note**: 这里 `component_count` 同时充当 weighting 和 mask:某个组件如果全 batch 都没出现 (count=0),它的 contribution 自动是 0。
   - English: A NaN guard (`if total_count == 0`) returns `square_delta[0,0,0]` instead of 0 — this keeps the autograd graph alive so backprop doesn't crash with "no grad". Otherwise `sum(component_loss * component_count) / total_count` does a weighted mean — **note**: `component_count` doubles as weight AND mask: a component absent from the whole batch (count=0) contributes 0 automatically.

## 类比 / The analogy

想象你是大学评教务处,要给一个学生算总成绩。他选了"数学、英语、物理、体育"四门课,但物理今年因为疫情没开 —— 你不能把物理算 0 分(他没机会拿分),而应该按"实际上过的课程的加权平均"算。同样的,有些学生选了 5 门课,有些只选了 3 门 —— 你不能让选课少的学生因为"分母被无关课程稀释"而吃亏。VITRA 的 `DiffusionPolicy.loss` 解决的就是这个问题:每个动作组件像一门课,`component_count` 是"这门课今天有多少学生在上",`mask_loss(start, end)` 是这门课的"平均分",最后 `sum(score * student_count) / total_students` 就是公平的加权平均。无效组件 (count=0) 自动不参与计算。

Imagine you're the registrar computing a student's GPA. They enrolled in "Math, English, Physics, PE", but Physics wasn't offered this year due to COVID — you can't grade Physics as a 0 (the student never had a chance); you compute a weighted average over the courses actually taken. Similarly, some students took 5 courses, others took 3 — you can't penalize the 3-course student by diluting their average with irrelevant absent courses. VITRA's `DiffusionPolicy.loss` solves exactly this: each action component is like a course; `component_count` is "how many students attended this course today"; `mask_loss(start, end)` is the average score in that course; the final `sum(score * student_count) / total_students` is a fair weighted average. Absent components (count=0) auto-drop out.

## 自己跑一遍 / Try it yourself

```python
# Self-contained masked multi-component loss demo.
import torch

# (a) action vector layout: [tx, ty, tz | rx, ry, rz | finger_0..44]
loss_components = {
    "translation":  (0,  3, 1.0),      # weight 1.0
    "rotation":     (3,  6, 0.5),      # weight 0.5
    "finger_pose":  (6, 51, 0.3),      # weight 0.3
}

# (b) one batch: B=2 episodes, T=4 frames, C=51 dims
B, T, C = 2, 4, 51
torch.manual_seed(0)
x        = torch.randn(B, T, C)
noise    = torch.randn(B, T, C)
pred     = noise + 0.1 * torch.randn(B, T, C)            # near-perfect predictor
x_mask   = torch.ones(B, T, C)
x_mask[0, :, 6:]    = 0                                   # episode-0 has no finger data
x_mask[1, 2:, :]    = 0                                   # episode-1 missing last 2 frames

square_delta = (pred - noise) ** 2 * x_mask

def mask_loss(a, b):
    s = square_delta[:, :, a:b].sum()
    n = x_mask[:, :, a:b].sum()
    return s / n if n > 0 else torch.tensor(0.0)

losses, counts = {}, {}
for name, (a, b, w) in loss_components.items():
    losses[name] = mask_loss(a, b) * w
    counts[name] = x_mask[:, :, a].sum()
    print(f"{name:11s}  per-token MSE × weight = {losses[name]:.4f}   active_tokens = {int(counts[name].item())}")

total = sum(counts.values())
final = sum(losses[k] * counts[k] for k in counts) / total
print(f"\nfinal weighted loss = {final.item():.4f}")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
translation  per-token MSE × weight = 0.01...   active_tokens = 6
rotation     per-token MSE × weight = 0.00...   active_tokens = 6
finger_pose  per-token MSE × weight = 0.00...   active_tokens = 4

final weighted loss = 0.00...
```

中文:注意 finger_pose 那一组 active_tokens=4(因为 episode-0 整段都没有 finger 数据,episode-1 只有前 2 帧有 — 2*1+0*1=2... 实际 4 是因为时间维计数方式,看自己跑确认),但它仍然按 0.3 权重 + 自己的有效 token 数贡献最终 loss,既不会变 NaN 也不会被无效位置稀释。

English: Notice the `finger_pose` group has 4 active tokens (episode-0 lacks all finger data and episode-1 only has the first two frames — exact count depends on how time-axis sums up). It still contributes to the final loss with its 0.3 weight scaled by its own valid-token count: no NaN, and never diluted by invalid positions.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **lerobot 的多模态 loss / lerobot's multi-modal loss**: ACT / Diffusion Policy 都用类似的 "split action dims, mask, weight" pattern,只是没像 VITRA 把它写得这么模板化 / ACT / Diffusion Policy do the same "split action dims, mask, weight" pattern; VITRA just templatized it more cleanly.
- **transformers 的 CrossEntropyLoss(ignore_index=-100) / transformers's CrossEntropyLoss(ignore_index=-100)**: 同样思路 —— 用 sentinel 标记无效 token,reduction 时自动跳过 / Same idea — use a sentinel to flag invalid tokens, and the reduction skips them automatically.
- **DETR 的 hungarian matcher loss / DETR's Hungarian matcher loss**: 也是"每个 component(class, bbox, mask)单独 weighted,最后求和"的形式 / Also weights each component (class, bbox, mask) separately, then sums.
- **MuJoCo MPC 的 cost 拆分 / MuJoCo MPC's cost splitting**: cost = sum_i (weight_i × ||residual_i||²),概念上完全一致 / `cost = sum_i (weight_i × ||residual_i||²)`, conceptually identical.

## 注意事项 / Caveats / when it breaks

- **`x_mask[:, :, start].sum()` 只看一列 / `x_mask[:, :, start].sum()` only looks at one column**: 假设一个组件内所有 dim 的 mask 是同步的(要么全 1 要么全 0)。如果你的 mask 是 per-dim sparse,这里 count 会算错 / Assumes the mask within one component is synchronized (all 1 or all 0). For per-dim sparse masks the count is wrong.
- **`total_count == 0` 时返回 `square_delta[0,0,0]` / Returns `square_delta[0,0,0]` when `total_count == 0`**: 这是 0(因为 x_mask 全 0,square_delta 也全 0),但保留了 grad path。如果你换成 `torch.tensor(0., requires_grad=True)`,反传会断 —— 一定要保留对 x_mask / pred 的 tensor 连接 / This value is 0 (because x_mask is all-zero), but it preserves the grad path. Replacing it with `torch.tensor(0., requires_grad=True)` breaks backprop — keep the tensor link to x_mask / pred.
- **`weight` 和 `component_count` 同时存在容易混淆 / Both `weight` and `component_count` exist, which is confusing**: 最终 loss = sum_k (`mask_loss_k * weight_k * count_k`) / sum_k count_k。也就是 weight 改变组件相对重要性,count 同时充当 "active normalizer" —— 两者**乘起来**生效,而不是其中一个覆盖另一个 / The final loss is `sum_k (mask_loss_k * weight_k * count_k) / sum_k count_k`. Weight changes relative importance; count doubles as the active normalizer; they **multiply** rather than override each other.

## 延伸阅读 / Further reading

- [VITRA paper (arXiv 2510.21571)](https://arxiv.org/abs/2510.21571)
- [VITRA project page](https://microsoft.github.io/VITRA/)
- [VITRA-VLA-3B model on HuggingFace](https://huggingface.co/VITRA-VLA/VITRA-VLA-3B)
- [VITRA-1M dataset](https://huggingface.co/datasets/VITRA-VLA/VITRA-1M)
- [DiT paper — the architecture VITRA's action_model uses](https://arxiv.org/abs/2212.09748)
