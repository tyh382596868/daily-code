---
date: 2026-05-29
topic: vla
source: vla
repo: openvla/openvla
file: vla-scripts/finetune.py
permalink: https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/vla-scripts/finetune.py#L172-L191
difficulty: intermediate
read_time: ~9 min
tags: [code-of-the-day, vla, lora, finetune, peft]
build_role: Fine-tune / LoRA entry point — adapting a pretrained 7B VLA to a new robot on a single GPU
---

# 7B VLA 微调到新机器人,核心就是 PEFT 的几行 / Fine-tuning a 7B VLA to a new robot is a handful of PEFT lines

> **一句话 / In one line**: OpenVLA 把 7B 模型适配到新机器人,核心是 `LoraConfig(target_modules="all-linear")` + `get_peft_model` —— 冻结整个 7B,只在每个 Linear 旁挂一对低秩矩阵训练,可训参数降到 ~1%,单卡就能 fine-tune。 / OpenVLA adapts a 7B model to a new robot with `LoraConfig(target_modules="all-linear")` + `get_peft_model` — freeze the whole 7B, attach a low-rank pair beside every Linear, drop trainable params to ~1%, and fine-tune on a single GPU.

## 为什么重要 / Why this matters

VLA 的预训练模型动辄 7B 参数,在 Open X-Embodiment 这种百万级数据上训。但你拿到手要适配的是**你自己的机器人**(不同相机、不同夹爪、几百条示教)。全参数微调 7B 需要多卡 + 几百 GB 显存,而且小数据上容易过拟合 / 灾难性遗忘。LoRA 是标准答案:冻结预训练权重,只训练注入的低秩增量。这段代码是 VLA 微调的实际入口 —— 它把 LoRA、量化、DDP 三件事串起来,是你 nanoVLA "适配新机器人" 流程的模板。注意它跟前几天讲过的 PEFT LoRA forward(数学)互补:那篇讲"LoRA 怎么算",这篇讲"LoRA 怎么挂到一个真实 VLA 上训练"。

VLA pretrained models run 7B params, trained on million-scale Open X-Embodiment. But what you actually adapt to is **your own robot** (different cameras, different gripper, a few hundred demos). Full fine-tuning of 7B needs multi-GPU and hundreds of GB, and overfits / catastrophically forgets on small data. LoRA is the standard answer: freeze pretrained weights, train only injected low-rank deltas. This is the practical fine-tuning entry point for a VLA — it wires LoRA, quantization, and DDP together, the template for your nanoVLA "adapt to a new robot" flow. It complements the earlier PEFT LoRA forward note (the maths): that one was "how LoRA computes", this is "how LoRA bolts onto a real VLA for training".

## 代码 / The code

`openvla/openvla` — [`vla-scripts/finetune.py`](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/vla-scripts/finetune.py#L172-L191)

```python
# [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
if cfg.use_lora:
    lora_config = LoraConfig(
        r=cfg.lora_rank,                      # rank of the low-rank update (e.g. 32)
        lora_alpha=min(cfg.lora_rank, 16),    # scaling factor alpha
        lora_dropout=cfg.lora_dropout,
        target_modules="all-linear",          # inject into every Linear in the model
        init_lora_weights="gaussian",
    )
    vla = get_peft_model(vla, lora_config)
    vla.print_trainable_parameters()          # e.g. "trainable: 1.2% of 7.5B"

# Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True,
          gradient_as_bucket_view=True)

# Create Optimizer =>> only the LoRA params have requires_grad=True
trainable_params = [param for param in vla.parameters() if param.requires_grad]
optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

# Create Action Tokenizer (discrete-action route: actions become token ids)
action_tokenizer = ActionTokenizer(processor.tokenizer)
```

For quantized (QLoRA) training, earlier in the file:

```python
if cfg.use_quantization:
    assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
    # ... load model in 4-bit, then prepare_model_for_kbit_training(vla)
```

## 逐行讲解 / What's happening

1. **`LoraConfig(r=..., lora_alpha=..., target_modules="all-linear")` / The LoRA spec**:
   - 中文:`r` 是低秩维度(32 表示每个 Linear 旁挂 `[d, 32]` 和 `[32, d]` 两个小矩阵);`lora_alpha` 是缩放(实际增量是 `alpha/r * BA`);`target_modules="all-linear"` 是关键 —— 不用手写"注入哪些层",PEFT 自动找到所有 `nn.Linear` 注入。VLA 里这意味着 vision、language、action head 的所有线性层都挂上 LoRA。
   - English: `r` is the low-rank dim (32 means a `[d, 32]` and `[32, d]` pair beside each Linear); `lora_alpha` is the scaling (the effective delta is `alpha/r * BA`); `target_modules="all-linear"` is the key — no manual layer list, PEFT finds every `nn.Linear` and injects. For a VLA this means vision, language, and action-head Linears all get LoRA.

2. **`lora_alpha=min(lora_rank, 16)` / Capped alpha**:
   - 中文:OpenVLA 把 alpha 上限设到 16。alpha/r 决定 LoRA 增量的尺度,太大不稳。这个 `min` 是个经验保护。
   - English: OpenVLA caps alpha at 16. Since alpha/r sets the LoRA delta scale, too large is unstable. The `min` is an empirical guardrail.

3. **`get_peft_model(vla, lora_config)` / Inject and freeze**:
   - 中文:这一行做两件事:(a) 把所有原始权重 `requires_grad=False`;(b) 给每个目标 Linear 旁挂上可训练的 LoRA 矩阵对。返回的 `vla` 表面上还是同一个模型,但只有 LoRA 参数能训。`print_trainable_parameters()` 会打出类似 "trainable 1.2%"。
   - English: this line does two things: (a) sets all original weights to `requires_grad=False`; (b) attaches a trainable LoRA matrix pair beside each target Linear. The returned `vla` looks like the same model but only LoRA params train. `print_trainable_parameters()` prints something like "trainable 1.2%".

4. **`init_lora_weights="gaussian"` / Initialization**:
   - 中文:LoRA 的 B 矩阵通常初始化为 0(保证训练开始时增量为 0,模型 = 预训练),A 矩阵高斯初始化。"gaussian" 指 A 的初始化方式。开始时 `BA = 0`,所以微调从预训练模型平滑出发。
   - English: LoRA's B matrix is typically zero-init (so the delta is 0 at start, model = pretrained), A is Gaussian. "gaussian" refers to A's init. At start `BA = 0`, so fine-tuning departs smoothly from the pretrained model.

5. **`DDP(..., find_unused_parameters=True)` / Multi-GPU wrapper**:
   - 中文:`find_unused_parameters=True` 是因为 LoRA 模型里大量参数(冻结的原始权重)不产生梯度,DDP 默认会报错,这个 flag 让它容忍。`gradient_as_bucket_view=True` 省显存。
   - English: `find_unused_parameters=True` is needed because most params (the frozen originals) produce no gradient, which DDP would otherwise error on. `gradient_as_bucket_view=True` saves memory.

6. **`trainable_params = [p for p in vla.parameters() if p.requires_grad]` / Optimizer sees only LoRA**:
   - 中文:optimizer 只拿 `requires_grad=True` 的参数 —— 也就是 LoRA 矩阵。AdamW 的状态(动量、二阶矩)只为这 ~1% 参数分配,所以优化器显存也省了 99%。
   - English: the optimizer takes only `requires_grad=True` params — the LoRA matrices. AdamW's state (momentum, second moment) is allocated only for that ~1%, so optimizer memory drops 99% too.

7. **`ActionTokenizer` / Discrete-action route reminder**:
   - 中文:OpenVLA 走的是**离散动作**路线 —— 动作被 `ActionTokenizer` 分箱成 token id,微调时和语言一样用交叉熵。这跟今天另一篇 GR00T flow matching 的连续路线是对照。微调 LoRA 对两条路线都通用。
   - English: OpenVLA uses the **discrete-action** route — actions are binned into token ids by `ActionTokenizer` and fine-tuned with cross-entropy like language. This contrasts with today's GR00T flow-matching continuous route. LoRA fine-tuning applies to both routes.

## 类比 / The analogy

像给一位资深翻译(预训练 7B)适配一个新领域(你的机器人)。你不会让他重新学整个语言(全参微调,太贵),而是给他一本薄薄的"领域术语小抄"(LoRA 低秩矩阵)夹在原书里。原书一个字不改(冻结),他查术语时顺手翻一下小抄做微调。小抄只占原书的 1% 厚度,放进口袋(单卡显存)就能带走。QLoRA 更进一步:把原书印成缩印版(4-bit 量化)再夹小抄,口袋更省地方。

Like adapting a veteran translator (pretrained 7B) to a new domain (your robot). You don't make them relearn the whole language (full fine-tune, too costly) — you slip a thin domain-glossary (LoRA low-rank matrices) into the original book. The book is untouched (frozen); they consult the glossary for adjustments. The glossary is 1% of the book's thickness, fits in a pocket (single-GPU memory). QLoRA goes further: shrink-print the book (4-bit quantization) before adding the glossary, saving even more pocket space.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-VLA

中文:在 nanoVLA 里这是 `nano/vla/scripts/finetune.py` —— "适配新机器人"的入口脚本,是用户实际跑的命令行。上游:一个预训练好的 nanoVLA(VLM backbone + action head,前面几篇笔记搭好的)+ 几百条新机器人示教数据;下游:训练循环 + checkpoint 保存(注意要存 dataset statistics 用于推理时反归一化动作)。如果省掉 LoRA、做全参微调:小数据上会灾难性遗忘(把预训练学到的视觉语言能力训坏),而且单卡放不下 7B 的优化器状态。LoRA 是"小数据适配大模型"的标配。生产实现要补:(1) **选择性 target_modules**(`all-linear` 简单但有时只 LoRA attention 的 q/v 更省更稳);(2) **QLoRA**(4-bit 量化基座 + LoRA,把 7B 塞进 24GB 消费卡);(3) **merge LoRA**(部署时 `merge_and_unload` 把 LoRA 增量合并回权重,推理零开销);(4) **dataset statistics**(动作归一化的均值方差必须存下,否则推理时输出尺度全错)。

English: in nanoVLA this is `nano/vla/scripts/finetune.py` — the "adapt to a new robot" entry script users actually run. Upstream: a pretrained nanoVLA (VLM backbone + action head from earlier notes) + a few hundred new-robot demos. Downstream: the training loop + checkpoint saving (crucially, save dataset statistics to de-normalize actions at inference). Skip LoRA and full-fine-tune: catastrophic forgetting on small data (ruining the pretrained vision-language ability) and the 7B optimizer state won't fit one GPU. LoRA is the standard for adapting a big model to small data. Production additions: (1) **selective target_modules** (`all-linear` is simple, but sometimes LoRA-ing only attention q/v is cheaper and more stable), (2) **QLoRA** (4-bit base + LoRA to fit 7B on a 24 GB consumer card), (3) **merge LoRA** (`merge_and_unload` folds deltas back into weights for zero-overhead inference), (4) **dataset statistics** (action-normalization mean/std must be saved or inference outputs are at the wrong scale).

## 自己跑一遍 / Try it yourself

```python
# pip install torch peft
import torch, torch.nn as nn
from peft import LoraConfig, get_peft_model

# A toy "VLA": a stack of Linears standing in for a big pretrained model
model = nn.Sequential(
    nn.Linear(512, 512), nn.GELU(),
    nn.Linear(512, 512), nn.GELU(),
    nn.Linear(512, 7),                      # action head: 7-DoF
)
total = sum(p.numel() for p in model.parameters())

lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.0,
    target_modules="all-linear",
    init_lora_weights="gaussian",
)
lora_model = get_peft_model(model, lora_config)

trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
print(f"total params     : {total:,}")
print(f"trainable (LoRA) : {trainable:,}")
print(f"trainable %      : {100*trainable/total:.2f}%")

# verify the base weights are frozen
base_frozen = all(not p.requires_grad for n, p in lora_model.named_parameters() if "lora" not in n)
print("base weights frozen:", base_frozen)
```

运行 / Run with:
```bash
pip install torch peft
python try.py
```

预期输出 / Expected output:
```
total params     : ~530,000
trainable (LoRA) : ~16,000
trainable %      : ~3%
base weights frozen: True
```

中文:可训练参数只占百分之几,基座全冻结。在真实 7B VLA 上这个比例是 ~1%,意味着优化器显存省 99%,单卡就能微调。

English: trainable params are a few percent, base fully frozen. On a real 7B VLA this is ~1%, meaning 99% less optimizer memory and single-GPU fine-tuning.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **前几天的 PEFT LoRA forward 笔记** / **The earlier PEFT LoRA forward note**: 中文 — 那篇讲 `h = Wx + (alpha/r)·B(Ax)` 的数学,这篇讲怎么把它挂到 VLA 上训练。配对看。 / English — that one covered the `h = Wx + (alpha/r)·B(Ax)` maths; this one covers bolting it onto a VLA. Read them paired.
- **π₀ / GR00T fine-tune scripts** / **π₀ / GR00T fine-tune scripts**: 中文 — 同样用 PEFT LoRA 适配,只是动作是连续 flow matching 而非离散 token。 / English — also use PEFT LoRA for adaptation, just with continuous flow-matching actions instead of discrete tokens.
- **LLM instruction tuning** / **LLM instruction tuning**: 中文 — LoRA 的发源地,VLA 直接借用 LLM 社区的 PEFT 生态。 / English — LoRA's birthplace; VLAs reuse the LLM community's PEFT ecosystem wholesale.

## 注意事项 / Caveats / when it breaks

- **`find_unused_parameters=True` 必开** / **`find_unused_parameters=True` is mandatory**: 中文 — LoRA 下大部分参数无梯度,不开这个 DDP 会直接报错。但它有性能开销,纯 LoRA 场景可以考虑用 FSDP 替代。 / English — most params have no gradient under LoRA; without this flag DDP errors out. It has overhead, so consider FSDP for pure-LoRA setups.
- **dataset statistics 必须存** / **Must save dataset statistics**: 中文 — 动作训练时归一化了,推理时要用同样的均值方差反归一化。忘了存,机器人动作尺度全错。 / English — actions are normalized in training; inference must de-normalize with the same mean/std. Forget to save them and the robot's action scale is wrong.
- **alpha/r 比值决定稳定性** / **The alpha/r ratio sets stability**: 中文 — 增量尺度是 `alpha/r`。盲目调大 r 但不调 alpha 会让有效 LR 变化,训练不稳。 / English — the delta scale is `alpha/r`. Bumping r without adjusting alpha changes the effective LR and destabilises training.
- **QLoRA 要先 `prepare_model_for_kbit_training`** / **QLoRA needs `prepare_model_for_kbit_training`**: 中文 — 4-bit 量化基座要先调这个函数(开启梯度检查点、上采 LayerNorm 到 fp32),否则训练发散。 / English — a 4-bit base needs `prepare_model_for_kbit_training` first (enables gradient checkpointing, upcasts LayerNorm to fp32) or training diverges.

## 延伸阅读 / Further reading

- [LoRA: Low-Rank Adaptation (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
- [QLoRA (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314)
- [OpenVLA paper](https://arxiv.org/abs/2406.09246)
- [Today's VLA action survey doc](./README-action-survey.md)
