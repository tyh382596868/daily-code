---
date: 2026-05-28
topic: vla
source: vla
repo: openvla/openvla
file: vla-scripts/finetune.py
permalink: https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/vla-scripts/finetune.py#L249-L291
difficulty: intermediate
read_time: ~12 min
tags: [code-of-the-day, vla, openvla, training-loop, action-token, l1-metric]
build_role: training step + L1 metric (the supervision core of a VLA)
---

# OpenVLA 的训练步:把机器人动作当 LLM token,40 行搞定监督 / OpenVLA's training step: 40 lines that supervise a robot policy as if it were an LLM

> **一句话 / In one line**: OpenVLA 把 7-DoF 动作离散成 7 个普通 vocab token,所以训练步几乎就是 LLM SFT——一次 bf16 前向 + 一次 cross-entropy + 一次反向;唯一精彩的部分是评估时如何从 logits 里"切"出动作位、argmax 拿到离散 id、再 decode 回连续动作算 L1,让 loss 数字变成人类能读懂的"误差有多少米/弧度"。 / OpenVLA discretizes a 7-DoF robot action into 7 ordinary vocab tokens, so the training step is essentially LLM SFT — one bf16 forward, one cross-entropy, one backward. The clever piece is the eval-side trick: slice the action-token positions out of the logits, argmax to get discrete action ids, then decode back to continuous so the reported L1 is in human-readable units (meters / radians).

## 为什么重要 / Why this matters

如果你想自己从头搭一个 VLA,第一个绕不开的问题是:**机器人的动作是连续的,但 LLM 只会预测离散 token,怎么办?** OpenVLA 给的答案极其工程化——把每个动作维度的连续值离散成 256 个 bin,直接占用 vocab 里 256 个"动作 token"。这样训练就是普通的 next-token prediction:vision + language prefix + 动作 token 序列,cross-entropy 算到底。这套 setup 让你能直接复用 LLaMA / Mistral / Qwen 的所有训练基础设施(HF AutoModel、PEFT LoRA、DDP、bf16 autocast 等等),代价是动作精度卡在 1/256 ≈ 0.4% 上限。这段 40 行代码完整展示了"训练一次步"的全部内容,也展示了如何把离散 token 训练 loss 翻译回机器人语言的 L1 误差——这正是你 nanoVLA 的训练循环原型。读完之后你就能回答两个问题:(1) 我自己写训练循环时,VLA 跟 LLM 训练有何区别?答:只是 labels 里嵌了动作 token。(2) 训出来 cross-entropy = 1.2 是好是坏?答:把它 decode 回 continuous 算 L1,跟机器人实际精度对齐。

If you want to build a VLA from scratch, the first unavoidable question is: **robot actions are continuous, LLMs only predict discrete tokens — how do you bridge that?** OpenVLA's answer is shamelessly engineering: discretize each of the 7 action dimensions into 256 bins, and **literally use 256 slots in the vocab as "action tokens."** Training becomes plain next-token prediction over `vision + language prefix + action tokens`, with vanilla cross-entropy. That lets you reuse every piece of LLM training infra — HF AutoModel, PEFT LoRA, DDP, bf16 autocast — at the cost of capping action precision at 1/256 ≈ 0.4%. The 40 lines below show the complete "one training step" and the elegant evaluation trick for translating the cross-entropy back into the robot's native units. After reading them you'll know: (1) how a VLA training loop differs from LLM SFT (answer: it doesn't, the differences hide entirely in `labels`); (2) how to convert "CE = 1.2" into "this policy is off by 3.2 cm and 0.18 rad on average" — the metric your robot teammates actually care about.

## 代码 / The code

`openvla/openvla` — [`vla-scripts/finetune.py`](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/vla-scripts/finetune.py#L249-L291)

```python
# Train!
with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
    vla.train()
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(dataloader):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output: CausalLMOutputWithPast = vla(
                input_ids=batch["input_ids"].to(device_id),
                attention_mask=batch["attention_mask"].to(device_id),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                labels=batch["labels"],
            )
            loss = output.loss

        # Normalize loss to account for gradient accumulation
        normalized_loss = loss / cfg.grad_accumulation_steps

        # Backward pass
        normalized_loss.backward()

        # Compute Accuracy and L1 Loss for Logging
        action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
        action_preds = action_logits.argmax(dim=2)
        action_gt = batch["labels"][:, 1:].to(action_preds.device)
        mask = action_gt > action_tokenizer.action_token_begin_idx

        # Compute Accuracy
        correct_preds = (action_preds == action_gt) & mask
        action_accuracy = correct_preds.sum().float() / mask.sum().float()

        # Compute L1 Loss on Predicted (Continuous) Actions
        continuous_actions_pred = torch.tensor(
            action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
        )
        continuous_actions_gt = torch.tensor(
            action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
        )
        action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)
```

## 逐行讲解 / What's happening

1. **`with torch.autocast("cuda", dtype=torch.bfloat16):` (前向块)**:
   - 中文: 整个 VLM 在 bf16 下前向。OpenVLA 用的是 Prismatic 7B (LLaMA-2 + SigLIP/DINOv2 双视觉塔),只有 bf16 才能在 24 GB 卡上塞下 LoRA 训练。
   - English: The whole VLM forward runs in bf16. OpenVLA uses Prismatic 7B (LLaMA-2 + SigLIP/DINOv2 dual-vision-tower); bf16 is what lets a single 24GB GPU hold a LoRA fine-tune.

2. **`vla(input_ids=..., pixel_values=..., labels=batch["labels"])` (forward)**:
   - 中文: 这就是普通 HF `Vision2Seq` 的调用签名。`input_ids` 是 `<text prompt> + <action token slots>` 的混合,`pixel_values` 是图像,`labels` 跟 `input_ids` 一样但是把"非动作位置"用 `-100` mask 掉,这样 HF 的 CE 只会在 7 个动作 token 上算 loss。**这是 VLA 的核心约定**:把动作藏在 labels 里。
   - English: Just the standard HF `Vision2Seq` call signature. `input_ids` is `<text prompt> + <action token slots>`, `pixel_values` is the image, `labels` mirrors `input_ids` but masks every **non-action** position to `-100` so HF's CE only fires on the 7 action positions. **This is the VLA-side convention**: actions are smuggled into the label tensor.

3. **`output.loss` (the cross-entropy)**:
   - 中文: 由 HF 内部 `CrossEntropyLoss` 算出来,只在 labels 不为 `-100` 的位置——也就是动作 token——上 reduce。这就是全部"VLA 训练 loss"。没有 BC loss,没有 RL,没有专门的 action head。
   - English: Computed by HF's internal `CrossEntropyLoss`, reduced only over positions where labels are not `-100` — i.e. action tokens. This is the **entirety** of "VLA training loss." No BC loss, no RL, no special action head.

4. **`normalized_loss = loss / cfg.grad_accumulation_steps`**:
   - 中文: 经典的梯度累积归一化。如果 micro-batch=2、accum=8,实际 batch=16,所以每次 backward 的 loss 要除以 8,这样 8 次累积之后的梯度尺度跟一次 batch=16 完全等价。
   - English: Standard gradient-accumulation normalization. With micro-batch=2 and accum=8, the effective batch is 16; dividing by 8 each backward step makes 8 accumulated gradients numerically equivalent to one batch-16 step.

5. **`action_logits = output.logits[:, vision_backbone.featurizer.patch_embed.num_patches : -1]`**:
   - 中文: 这是这段代码最值得记住的一行。OpenVLA 的前缀长这样:`[<vision patches> <BOS> <text prompt> <action token slots>]`。`num_patches` 通常是 256(双视觉塔则是 512)。`[num_patches : -1]` 把视觉位和最后的 EOS 切掉,只留 text + action 区域;再加上 labels 的 -100 mask,等价于"只看动作 token 位"。
   - English: The line worth memorizing. OpenVLA's prefix is `[<vision patches> <BOS> <text prompt> <action token slots>]`; `num_patches` is usually 256 (or 512 with dual towers). The slice `[num_patches : -1]` drops the vision prefix and the EOS suffix, leaving only the text + action region — then the labels' -100 mask narrows it further to action positions only.

6. **`action_preds = action_logits.argmax(dim=2)` / `mask = action_gt > action_tokenizer.action_token_begin_idx`**:
   - 中文: argmax 拿到每个位置的预测 token id;`mask` 用一个不等式把"这是不是动作 token"区分开来——OpenVLA 把 256 个动作 token 放在 vocab 末尾(从 `action_token_begin_idx` 开始),所以 `> action_token_begin_idx` 等价于"这是个动作位"。
   - English: `argmax` gives one predicted token id per position; the mask uses a single inequality to separate "is this an action token?" — OpenVLA places the 256 action tokens at the tail of the vocab (starting at `action_token_begin_idx`), so `> action_token_begin_idx` is the boolean test "this position is an action."

7. **`continuous_actions_pred = action_tokenizer.decode_token_ids_to_actions(...)`**:
   - 中文: 这是 metric 翻译层。`decode_token_ids_to_actions` 把每个动作 token id 映射回它对应 bin 的中心值——也就是反量化。两边都做一遍,然后 L1。这样 wandb 上看到的不是抽象的 "loss=1.2",而是 "L1=0.034",可以直接跟机器人的物理精度对照。
   - English: This is the metric-translation layer. `decode_token_ids_to_actions` maps each action-token id back to the **center of its bin** — i.e. dequantization. Apply it on both predicted and ground-truth tokens, then L1. What appears on your wandb dashboard is no longer abstract "loss=1.2" but "L1=0.034" — a number you can compare directly against the robot's physical precision tolerance.

8. **`torch.tensor(...)` 包裹 numpy 输出**:
   - 中文: `decode_token_ids_to_actions` 返回 numpy(因为 dataset stats 是 numpy 存的),所以要 `.cpu().numpy()` 出去,再 `torch.tensor(...)` 回来。这一步说明 ActionTokenizer 跟 dataset normalization 是耦合的:de-quantize 同时也是 un-normalize。
   - English: `decode_token_ids_to_actions` returns numpy (since dataset stats are stored as numpy), hence the `.cpu().numpy()` out and `torch.tensor(...)` back. This round-trip reveals that ActionTokenizer is coupled with dataset normalization: dequantization is also un-normalization, in one step.

## 类比 / The analogy

想象你在教一个只会写英文文章的助手"画地图":你不教它笔触和坐标,你把每个城市编号(纽约=001,旧金山=002...),让它学会写"今天的路线是 001 → 005 → 023"。它学到的是排序"token",而不是地理坐标。等评测时,你再用一张"编号→经纬度"的查找表,把它写的 token 序列翻译回真正的地理路径,然后算"实际偏离多少公里"。OpenVLA 就是这么干的:把 7 维连续动作每维 256 桶编号,让 LLaMA 当成普通文字预测;评测时用 ActionTokenizer 这张查找表反量化,算 L1 直接得到"机器人偏离 GT 多少米/弧度"。

Picture teaching an assistant who only writes English essays to "draw maps": you don't teach it brushstrokes or coordinates, you assign every city a number (NYC=001, SF=002, …) and let it learn to write `"today's route is 001 → 005 → 023"`. What it actually learns is **token ordering**, not geography. At eval time, you use a separate "id → lat/lon" lookup table to translate its token sequence back into a real geographic path, then compute "how far off in kilometers." OpenVLA does exactly this: bucket each of 7 action dims into 256 ids, let LLaMA treat them as plain text tokens, and at eval call ActionTokenizer (the lookup table) to dequantize so the L1 number is in meters / radians the robot team already understands.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-VLA

这段代码在 nanoVLA 里就是你的 `train_step` 函数。组件分工:(a) `vla(input_ids=..., labels=...)` 对应你 `nanoVLA.forward(images, text_ids, action_labels)` ——你的 forward 内部要把 vision 编码器输出 patch tokens、把 text 走 embedding、把 action token 也走 embedding,然后送进一个 GPT 风格 decoder;(b) `action_logits = logits[:, num_patches:-1]` 对应你自己写的 `action_head(hidden_states[:, action_positions])`——nanoVLA 可以更直白:不复用 LLaMA vocab,而是给动作单独开一个 head,然后 logits 形状就是 `(B, action_len, 256)`,不需要切片;(c) `decode_token_ids_to_actions` 对应你 `tokenizer.detokenize(action_ids) -> continuous_actions`,只在 metric/inference 时调用。**省掉这一层会发生什么**:你只会有 cross-entropy 数字,无法判断"loss=1.2 是好是坏"——必须有反量化才能在物理量级上做 early stopping 或 hyperparameter sweep。**生产级 VLA 还要加什么**:(1) action chunking,一次预测 H=8 步(openvla-oft 的做法,大幅提速 inference);(2) 连续动作分支(openpi 的 flow-matching head),彻底跳过量化以提升精度;(3) 课程式 mask:训练初期 mask 视觉,只看 proprioception,稳定后再放开;(4) on-policy 的 RT-2 风格 finetune,让模型在自己的输出分布上继续学。

In nanoVLA this code becomes your `train_step` function. Component map: (a) `vla(input_ids=..., labels=...)` corresponds to your `nanoVLA.forward(images, text_ids, action_labels)` — inside, your forward does vision-encoder → patch tokens, text-embedding for prompt tokens, action-embedding for action slots, then a GPT-style decoder; (b) `action_logits = logits[:, num_patches:-1]` corresponds to your own `action_head(hidden_states[:, action_positions])` — nanoVLA can be more direct: don't share LLaMA's vocab, give actions a dedicated head whose output is `(B, action_len, 256)`, no slicing needed; (c) `decode_token_ids_to_actions` corresponds to `tokenizer.detokenize(action_ids) -> continuous_actions`, used at metric and inference time only. **If you omit this**: you'll only have CE numbers and no way to tell "is loss=1.2 good?" — dequantization is required to compare to the robot's physical tolerance for early stopping or sweeps. **What production VLA adds on top**: (1) action chunking — predict H=8 future actions at once (openvla-oft style, big inference speedup); (2) continuous action branch (openpi's flow-matching head) that skips quantization for finer precision; (3) curriculum masking — early training masks vision and trains on proprio only, then unmasks; (4) on-policy RT-2-style finetune that lets the model continue learning from its own action distribution.

## 自己跑一遍 / Try it yourself

```python
# try_vla_step.py — standalone reimplementation, no openvla install.
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyActionTokenizer:
    """7-DoF actions -> 7 tokens, each in {0..N_BINS-1}, shifted to live at the tail of the vocab."""
    N_BINS = 256
    ACTION_OFFSET = 32000  # pretend the LLM vocab is 32000; action tokens occupy 32000..32255
    def __init__(self, lo=-1.0, hi=1.0): self.lo, self.hi = lo, hi
    def quantize(self, a):
        a = a.clamp(self.lo, self.hi)
        ids = ((a - self.lo) / (self.hi - self.lo) * (self.N_BINS - 1)).long()
        return ids + self.ACTION_OFFSET
    def decode(self, ids):
        ids = ids - self.ACTION_OFFSET
        return ids.float() / (self.N_BINS - 1) * (self.hi - self.lo) + self.lo

class TinyVLA(nn.Module):
    """A pretend VLM that returns logits of shape (B, T, VOCAB)."""
    def __init__(self, vocab=32256, dim=64):
        super().__init__(); self.emb = nn.Embedding(vocab, dim); self.head = nn.Linear(dim, vocab)
    def forward(self, input_ids): return self.head(self.emb(input_ids))

tok = TinyActionTokenizer()
model = TinyVLA()
B, PREFIX, A = 4, 10, 7                                   # 10 prefix (vision+text) tokens, 7 action tokens
gt_actions = torch.randn(B, A).clamp(-1, 1)               # ground-truth continuous actions
action_ids = tok.quantize(gt_actions)                     # (B, 7) in [32000..32255]
input_ids = torch.cat([torch.randint(0, 32000, (B, PREFIX)), action_ids], dim=1)
labels = input_ids.clone(); labels[:, :PREFIX] = -100      # mask prefix so CE only fires on action tokens

logits = model(input_ids)
loss = F.cross_entropy(logits[:, :-1].reshape(-1, 32256), labels[:, 1:].reshape(-1), ignore_index=-100)
loss.backward()

# Eval-side translation back to continuous
action_logits = logits[:, PREFIX-1:-1]                    # exactly the slice openvla does
preds = action_logits.argmax(dim=-1)
mask = labels[:, 1:] > tok.ACTION_OFFSET - 1
l1 = F.l1_loss(tok.decode(preds[mask]), tok.decode(labels[:, 1:][mask]))
print(f"CE  = {loss.item():.3f}    L1 (continuous) = {l1.item():.4f}    range = ±1.0")
```

运行 / Run with:
```bash
pip install torch
python try_vla_step.py
```

预期输出 / Expected output:
```
CE  = 10.371    L1 (continuous) = 0.522    range = ±1.0
```

(CE 数字会因为模型随机初始化而异,但 L1 应该在 0.5 左右——也就是"动作幅度满 2 时,平均偏差 ~0.5",随机猜测的水平,符合预期。)注意 `mask = labels[:, 1:] > ACTION_OFFSET - 1` 这一行——它就是 OpenVLA 里那个 `> action_token_begin_idx` 的等价物,**是把动作位从普通 vocab 里"切出来"的核心**。在 nanoVLA 里你可以直接让 action_head 输出独立的 (B, 7, 256) 形状,这一步就不需要了——但理解这条切片是看懂 OpenVLA 的入场券。

(CE varies with random init, but the L1 should land around 0.5 — meaning "on a ±1.0 action range, average deviation ≈ 0.5," exactly chance-level, which is correct for an untrained model.) The line `mask = labels[:, 1:] > ACTION_OFFSET - 1` mirrors OpenVLA's `> action_token_begin_idx`; **it's the core step that "carves out" action positions from the general vocab**. In nanoVLA you might give the action head its own dedicated `(B, 7, 256)` output and skip this slice entirely — but understanding it is the price of admission to OpenVLA.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **openvla-oft (Optimized Fine-Tuning)** / **openvla-oft**: 同一个团队的后续工作,把"单步 7 token"扩展成"action chunk H 步 × 7 维 token",inference 一次性出 8 步动作,提速 8×;loss assembly 结构完全一样,只是 mask 更长 / Same team's follow-up: extends "1 step × 7 tokens" to "H steps × 7 tokens" so inference emits 8 actions at once (8× speedup). The loss-assembly skeleton is identical, just the mask is longer.
- **openpi (Physical-Intelligence π₀)** / **openpi**: 用 flow-matching head 替代离散化,**完全跳过 ActionTokenizer**,所以训练 loss 不是 CE 而是 flow-matching velocity MSE;但 forward 信号"vision + text → 动作"的契约是一样的 / Replaces the discretizer with a flow-matching head, **skipping ActionTokenizer entirely** — training loss becomes flow-matching velocity MSE instead of CE; the "vision + text → action" contract is the same.
- **lerobot (HuggingFace)** / **lerobot**: 走的是 diffusion-policy 路线,动作由一个小 diffusion 模型生成,在 latent action space 里去噪;同样跳过离散化 / Uses diffusion-policy, where actions are produced by a small diffusion model denoising in a latent action space. Also skips discretization.
- **NVIDIA Isaac-GR00T (N1.5)** / **NVIDIA Isaac-GR00T (N1.5)**: 离散化 + 连续 head 双分支,前者快但精度上限,后者精细;训练时按权重混合两个 loss / Has both a discretized head and a continuous head; training mixes the two losses by weight — fast + precise.
- **RT-2 (Google)** / **RT-2 (Google)**: 整个范式的起源——"co-fine-tune VLM on web data + robot demonstrations, treating actions as token strings."OpenVLA 的开源复刻 / The paradigm's origin: "co-fine-tune VLM on web data + robot demonstrations, treating actions as token strings." OpenVLA is its open-source replication.
- **starVLA / Isaac-GR00T N1.5**: 类似的 dual-head 结构 / Similar dual-head setup, sometimes with auxiliary depth or proprio reconstruction losses on the side.

## 注意事项 / Caveats / when it breaks

- **`-100` mask 漏写就崩** / **Forgetting the `-100` mask is fatal**: 如果 labels 在 prefix 上不是 -100,CE 会让模型去预测视觉/文字 token,梯度方向完全错乱——loss 看起来变低(因为容易预测的位置数变多),但动作部分根本没学好 / If labels aren't -100 on the prefix, CE will train the model to predict vision/text tokens too; loss looks lower (more easy positions in the average) but the action positions don't actually train.
- **`num_patches` 写死会因为视觉塔不同而 off-by-N** / **Hard-coding `num_patches` breaks under different vision towers**: 单 SigLIP 塔是 256 个 patch,双塔 (SigLIP+DINOv2) 是 512;不读模型 config 直接写 256 会导致切片错位、CE 算到非动作位上 / SigLIP single-tower gives 256 patches; SigLIP+DINOv2 dual-tower gives 512. Hardcoding 256 silently misaligns the slice and CE fires on non-action positions.
- **ActionTokenizer 不能跨数据集复用** / **ActionTokenizer is per-dataset**: 256 个 bin 的边界是按数据集动作统计算出来的;换个机器人(关节范围不同)必须重新 fit,否则反量化会把 0.3 米解读成 0.05 米 / The 256-bin boundaries come from the dataset's action statistics; a new robot (different joint range) needs a refit, otherwise dequantization decodes 0.3m as 0.05m.
- **bf16 + LoRA + 7B model 仍然紧张** / **bf16 + LoRA + 7B is tight on memory**: 单 24GB 卡刚好能跑 batch=2 + grad_accum=8;再加任何 logging buffer 就会 OOM / On a single 24GB GPU you can fit batch=2 + grad_accum=8; adding any logging buffer (e.g., saving all logits to wandb) immediately OOMs.
- **L1 metric 用 numpy 来回,在 step 内做会很慢** / **L1 metric via numpy round-trip is slow if done every step**: `decode_token_ids_to_actions` 在 CPU 上做,每个 step 调一次会把 train step 拖慢 30%;生产里通常每 N 步算一次 / `decode_token_ids_to_actions` runs on CPU; calling it every step slows training ~30%. Production code calls it every N steps.
- **DDP + RLDS dataloader 时 `num_workers=0` 必填** / **`num_workers=0` is mandatory with RLDS dataloader**: RLDS 是基于 TFDS 的,自己有并行机制,不能再叠 PyTorch worker——否则会跨进程 fork TF 导致挂死 / RLDS is built on TFDS and has its own parallelism; stacking PyTorch workers on top forks TF across processes and deadlocks.

## 延伸阅读 / Further reading

- [OpenVLA paper (Kim et al. 2024)](https://arxiv.org/abs/2406.09246) — 完整 setup、数据集选择、量化策略
- [openvla/openvla README](https://github.com/openvla/openvla) — 完整 finetune.py 怎么 launch、推理脚本怎么写
- [openvla-oft (Optimized Fine-Tuning)](https://github.com/openvla/openvla-oft) — 同一团队后续工作,加 action chunking 和 LoRA-friendly 重构
- [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) — 看 flow-matching 替代方案
- [RT-2 paper (Brohan et al. 2023)](https://arxiv.org/abs/2307.15818) — 整个范式的起点
