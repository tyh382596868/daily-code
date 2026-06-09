---
date: 2026-06-08
topic: vla
source: vla
repo: openvla/openvla
file: prismatic/vla/datasets/datasets.py
permalink: https://github.com/openvla/openvla/blob/main/prismatic/vla/datasets/datasets.py#L62-L65
difficulty: intermediate
read_time: ~12 min
tags: [code-of-the-day, vla, openvla, training-target, ignore-index, teacher-forcing, exposure-bias, nano-vla]
build_role: training-step (deep-dive variant) — OpenVLA's training target is plain next-token prediction; one IGNORE_INDEX line restricts loss to the 7 action positions
---

# OpenVLA 的训练目标就是标准 LM 的 next-token prediction,只是 labels 多了一行 mask / OpenVLA's training target is just standard LM next-token prediction — only one line of label masking restricts the loss to the 7 action positions

> **一句话 / In one line**: OpenVLA 和标准 transformer 语言模型用的是**完全同一个** CE loss、softmax、argmax;它特别在哪?**只在 `labels[: -(len(action)+1)] = IGNORE_INDEX` 这一行 — 把所有非动作位置的 loss 关掉,让 7 个 CE 同时算 7 次 next-token prediction**。 / OpenVLA uses the **exact same** CE loss, softmax, and argmax as any standard transformer LM. What's special about it? Just one line: `labels[: -(len(action)+1)] = IGNORE_INDEX` — kills loss at every non-action position, so 7 CEs simultaneously do 7 next-token predictions.

## 为什么重要 / Why this matters

很多人初看 OpenVLA 都觉得它"和 LLM 不一样,毕竟它输出动作不是文字"。**这是误解**。从训练目标看,OpenVLA 跟训练 GPT 的代码**逐字相同** — 同一个 32000 维 logit、同一个 softmax over vocab、同一个 cross-entropy、同一个 argmax decode。它的"特殊性"全藏在 dataloader 里**一行 label mask**:把序列前面 ~531 个位置(BOS + vision + language)的 label 全填成 `IGNORE_INDEX=-100`,只留下最后 7 个 action token 位置当真值。`F.cross_entropy(..., ignore_index=-100)` 自动跳过 -100 → loss 只在 7 个位置算。这种"借标准 LM 训练栈"的设计让 OpenVLA 直接复用 HuggingFace 现成的 trainer、generation、KV cache,**这才是 7B 大模型能在普通 PyTorch 训练栈上跑起来的根本原因**。

When people first see OpenVLA they think "it must be very different from an LLM since it outputs actions, not words". **That's a misreading**. From the training-target standpoint, OpenVLA's code is **byte-for-byte identical** to training GPT — same 32000-dim logit, same softmax over vocab, same cross-entropy, same argmax decode. Its specialness lives in one line of label masking in the dataloader: fill the first ~531 label positions (BOS + vision + language) with `IGNORE_INDEX=-100`, keep only the last 7 action positions as real targets. `F.cross_entropy(..., ignore_index=-100)` auto-skips the -100s, so loss is computed only at 7 positions. This "ride the standard LM training stack" design is **the actual reason a 7B VLA can run on vanilla PyTorch trainer / HF generate / KV cache** without rewriting anything.

## 代码 / The code

The OpenVLA-special bit, [`prismatic/vla/datasets/datasets.py:54-65`](https://github.com/openvla/openvla/blob/main/prismatic/vla/datasets/datasets.py#L54-L65):

```python
# Tokenize (w/ `base_tokenizer`)
input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
labels    = list(input_ids)

# Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
#   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
pixel_values = self.image_transform(img)

# [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
labels[: -(len(action) + 1)] = IGNORE_INDEX            # ← this single line is the entire OpenVLA-special bit
if not self.predict_stop_token:
    labels[-1] = IGNORE_INDEX
```

The actual loss computation, [`transformers/loss/loss_utils.py:48-70`](https://github.com/huggingface/transformers/blob/main/src/transformers/loss/loss_utils.py#L48-L70) — **standard HuggingFace `ForCausalLMLoss`, OpenVLA uses it unmodified**:

```python
def ForCausalLMLoss(logits, labels, vocab_size, ignore_index=-100, ...):
    logits = logits.float()
    # Shift so that tokens < n predict n
    labels       = nn.functional.pad(labels, (0, 1), value=ignore_index)
    shift_labels = labels[..., 1:].contiguous()
    # Flatten
    logits       = logits.view(-1, vocab_size)             # (B*L, 32000)
    shift_labels = shift_labels.view(-1)                   # (B*L,)
    loss = F.cross_entropy(logits, shift_labels,
                           ignore_index=ignore_index)      # ← ignore_index=-100 跳过非动作位置
    return loss
```

And `F.cross_entropy(..., ignore_index=-100)` is PyTorch built-in:

```python
# Conceptually
mask = shift_labels != -100              # 531 个位置是 False, 7 个位置是 True
loss = -log_softmax(logits)[arange, shift_labels][mask].mean()
```

## 逐行讲解 / What's happening

1. **`input_ids` 和 `labels` 同源** / **input_ids and labels start identical**:
   - 中文: 先把 `input_ids` 完整 copy 一份给 `labels`,这一刻它们一样。后面只对 `labels` 动手脚,`input_ids` 保持完整(因为模型 forward 时位置 i 需要看到位置 i 的真实 token)。
   - English: copy `input_ids` into `labels` so both start the same. Only `labels` gets modified afterward; `input_ids` must stay intact because position i in the forward pass needs to see the real token at position i.

2. **`labels[: -(len(action) + 1)] = IGNORE_INDEX`** — **整篇论文的灵魂** / **the soul of the whole paper**:
   - 中文: `len(action) = 7`,所以这一行把 labels 倒数第 8 个位置之前的所有位置都设成 -100。倒数 8 个位置正好是 `[T1, T2, T3, T4, T5, T6, T7, EOS]`。
   - English: `len(action) = 7`, so this fills all positions up to (but not including) the last 8 with -100. The last 8 are exactly `[T1, T2, T3, T4, T5, T6, T7, EOS]`.

3. **HF 的 shift-by-1** / **HF's shift-by-1**:
   - 中文: `pad(labels, (0,1), -100)` 在右边补一个 -100,然后 `labels[..., 1:]` 把整个 labels 向左挪 1 位。这样位置 i 的 logits **用来预测位置 i+1 的 token**。这是标准 LM 的做法。
   - English: pad a `-100` on the right then slice `[..., 1:]` — labels shift left by one. Now logits at position i are used to predict the token at position i+1. Standard LM mechanics.

4. **`F.cross_entropy(..., ignore_index=-100)`** — **8 个真位置贡献 loss** / **8 effective positions contribute to loss**:
   - 中文: shift 之后,只有以下 8 个位置的 shift_label ≠ -100:`ASSISTANT:` 位预测 T1、T1 位预测 T2、... 、T6 位预测 T7、T7 位预测 EOS。每个位置算一个标准 CE,平均后回传。其余 530+ 个位置被 PyTorch 直接跳过。
   - English: after the shift, exactly 8 positions have shift_label ≠ -100: `ASSISTANT:` predicts T1, T1 predicts T2, …, T6 predicts T7, T7 predicts EOS. Each contributes one standard CE; average and backprop. The other 530+ positions are silently skipped by PyTorch.

5. **梯度怎么流** / **How the gradient flows**:
   - 中文: 虽然 loss 只在 8 个位置算,梯度通过 attention 反向**穿过整个序列** — vision 的 hidden state 给 action token 的 hidden state 贡献了信息,所以 vision projector 也收到梯度。这就是为什么训练时整个 LM + projector 都要训,不能只训 action 部分。
   - English: loss only fires at 8 positions, but gradients flow back through attention across the **entire** sequence — vision's hidden state contributed to action's, so the vision projector receives gradient too. That's why the whole LM + projector must be trained, not just the action portion.

6. **测试时同一个 argmax** / **Inference is the same argmax**:
   - 中文: 推理时不带 action token 进 prompt,greedy decode 7 步,每步 `next_token = argmax(softmax(logits[-1]))`,**完全是标准 LM 的 generate**。token id 落点涌现地在 vocab 末尾 256 slot 里,因为训练分布是这么压的。
   - English: at inference, no action tokens in the prompt; greedy decode 7 steps; each step `next_token = argmax(softmax(logits[-1]))`. **Pure standard LM generate**. Token ids land in the last 256 vocab slots emergently — the training distribution shaped them to.

## 类比 / The analogy

**OpenVLA = 一个被极端"偏科"训练的语文老师**。教材里有 540 页内容,但**只在最后 7 页**改作业打分。前面 533 页(BOS、图像描述、对话开场)随便学生写什么都不扣分,只关心最后那 7 个 action token。久而久之,这个学生的语言能力的"焦点"完全压到了最后 7 个位置 — 输入 ABC 全 → 输出最后 7 个总是 action token。它本质还是个语文老师,但你只考最后 7 题。

OpenVLA = **a Chinese-class teacher who only grades the last 7 questions**. The textbook has 540 pages but feedback only applies to the last 7. The first 533 pages (BOS, image description, conversation opener) — write whatever, no grade. Eventually the student's language ability laser-focuses on those last 7 positions: feed anything → the last 7 outputs always turn into action tokens. Still fundamentally a language teacher, just tested narrowly.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-VLA

这是 nanoVLA 课程里 `training-step` 槽位的**深度变体**(5/28 笔记用 OpenVLA finetune.py 讲过基础版,今天专门拆"训练目标本身的数学结构")。

理解这个 mask 模式有几个直接的工程价值:

1. **可以混训 multi-task**: 同一个 batch 里有的样本是"机器人任务"(`labels` 的 action 区有真值),有的样本是"VQA / captioning"(`labels` 的 text 输出区有真值,action 区被 mask)。**只需要改 labels 的 mask 位置**,模型代码、loss 函数完全不动。这就是为什么 OpenVLA 能继承 Prismatic VLM 的 image-text pretrain,且能在机器人任务上再加新的 vision-text 任务 co-train。

2. **可以教模型生成"思维链 + 动作"**: 把 action token 改成 `<thought> reasoning here </thought><action> T1..T7 </action>`,只在 `<action>` 段保留 loss,**模型自然学会先想再动**。RT-2 论文里的 "chain-of-thought robot" 就是这个 trick。

3. **可以 mask 掉特定动作维度**: 比如训练时把 `T7`(gripper)的 label 设成 -100,模型就只学 6-DoF pose,完全不输出 gripper 信号 — 适合"末端执行器由控制器单独管"的场景。

4. **debug 时可以 ablation**: 临时把 vision 段的 label 改回 input_ids 的真值,模型同时被监督"复述"图像内容 → 看 vision projector 学得对不对(类似 mae-style 自我检查)。

production 实现要补:
- **dataset balancing**: 当多 task 混训时,action loss 和 text loss 数量级不同,要按 task 加权。OpenVLA 没做这事(只有 action task),OpenVLA-OFT 也是。RT-X 系列用 hand-tuned weights。
- **gradient clipping per task**: 7 个 action CE 求平均得到的 scalar 量级和 GPT 训练时(几千 token 平均)非常不同,grad clip 阈值要重调。
- **per-position learning rate**: 偶尔有人会给 action token 一个更高的 lr(梯度信号稀疏 → 学得慢),但 OpenVLA 没这么做。

This is the **deep-dive variant** of the `training-step` slot in the nanoVLA curriculum (the 5/28 note covered the basic version via OpenVLA's `finetune.py`; today specifically dissects the math structure of the training target).

Understanding this mask pattern unlocks several engineering moves:

1. **Multi-task co-training**: same batch can contain robot-action samples (real targets in the action region) and VQA / captioning samples (real targets in the text-output region, action region masked). **Only the `labels` mask shape changes** — model code and loss function untouched. That's how OpenVLA inherits Prismatic VLM's image-text pretrain and adds new vision-text co-tasks on top.

2. **Chain-of-thought + action generation**: replace action tokens with `<thought> reasoning here </thought><action> T1..T7 </action>` and only keep loss in `<action>` region — model naturally learns to "think before acting". This is the RT-2 "chain-of-thought robot" trick.

3. **Mask specific action dimensions**: set `T7` (gripper) label to -100 during training; model learns to output only 6-DoF pose, ignoring gripper — useful when the end-effector is managed by a separate controller.

4. **Debug-time ablation**: temporarily restore vision-region labels to real input_ids and model simultaneously gets supervised on "reconstructing" image content — useful sanity check for the vision projector (MAE-style).

Production needs:
- **Dataset balancing**: under multi-task, action loss and text loss have different magnitudes; weight per task. OpenVLA doesn't (action-only), OFT doesn't either. RT-X uses hand-tuned weights.
- **Gradient clipping per task**: averaging 7 action CEs gives a scalar with very different magnitude than GPT pretraining (thousands of tokens averaged); re-tune grad-clip threshold.
- **Per-position learning rate**: occasionally folks give action tokens a higher LR (gradient signal is sparse → learns slowly), but OpenVLA didn't.

## 自己跑一遍 / Try it yourself

```python
# try.py — show that "OpenVLA training target == standard LM CE with masked labels"
import torch, torch.nn as nn, torch.nn.functional as F

V = 32000        # vocab size
L = 20           # short sequence for demo
N_ACT = 4        # 假装 4 个 action token

# 1. fake LM forward
logits = torch.randn(1, L, V, requires_grad=True)
labels = torch.randint(0, V, (1, L))

# 2. OpenVLA-style mask: 把前 L-N_ACT-1 个位置设 -100
labels_openvla = labels.clone()
labels_openvla[:, : -(N_ACT + 1)] = -100

# 3. HuggingFace 标准 ForCausalLMLoss (shift-by-1 + ignore_index=-100)
def for_causal_lm_loss(logits, labels, ignore_index=-100):
    labels = F.pad(labels, (0, 1), value=ignore_index)
    shift_labels = labels[..., 1:].contiguous().view(-1)
    return F.cross_entropy(logits.view(-1, logits.size(-1)),
                           shift_labels, ignore_index=ignore_index)

# 4. 对比 vanilla LM vs OpenVLA
loss_vanilla = for_causal_lm_loss(logits, labels)
loss_openvla = for_causal_lm_loss(logits, labels_openvla)

# 5. 检查 OpenVLA loss 只在最后几个位置算
labels_shift = F.pad(labels_openvla, (0, 1), value=-100)[..., 1:]
n_effective = (labels_shift != -100).sum().item()
print(f"vanilla LM loss (all {L-1} positions):                {loss_vanilla.item():.4f}")
print(f"OpenVLA-style loss (only {n_effective} effective positions): {loss_openvla.item():.4f}")
print(f"non-masked positions in OpenVLA labels: {n_effective}  ← should be {N_ACT+1}")

# 6. 关键检查:vanilla loss 改 labels 前面任何位置都变,OpenVLA loss 完全不变
labels_changed = labels.clone()
labels_changed[:, 0] = (labels[:, 0] + 1) % V                                    # 改第 0 位
labels_openvla_changed = labels_changed.clone()
labels_openvla_changed[:, : -(N_ACT + 1)] = -100
loss_v2 = for_causal_lm_loss(logits, labels_changed)
loss_ov2 = for_causal_lm_loss(logits, labels_openvla_changed)
print(f"\nchanged labels[0]:")
print(f"  vanilla loss diff: {abs(loss_v2 - loss_vanilla).item():.4f}  (changes!)")
print(f"  OpenVLA loss diff: {abs(loss_ov2 - loss_openvla).item():.6f}  (zero — mask kills it)")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
vanilla LM loss (all 19 positions):                 10.4xxx
OpenVLA-style loss (only 5 effective positions):     10.2xxx
non-masked positions in OpenVLA labels: 5  ← should be 5

changed labels[0]:
  vanilla loss diff: 0.xxxx  (changes!)
  OpenVLA loss diff: 0.000000  (zero — mask kills it)
```

**vanilla loss 改前面任何 label 都受影响,OpenVLA loss 完全无感**。这就是"OpenVLA = 标准 LM + label mask"的数值证据。
**Vanilla loss is sensitive to changes in any early label; OpenVLA loss is completely insensitive**. That's the numerical proof of "OpenVLA = standard LM + label mask".

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **指令微调 (SFT) 的 prompt-completion mask** / **Instruction-tuning's prompt-completion masking**: 训练 ChatGPT 时也只在 assistant 回答的 token 上算 loss,prompt 部分全 -100。**OpenVLA 是这个 trick 在 robotics 上的直接应用** / Same trick in chat finetuning — loss only on assistant tokens, prompt is all -100. **OpenVLA is the robotics application of this**.
- **Span-corruption / T5**: T5 在被腐蚀的 span 位置算 loss,其他位置 -100 / T5 computes loss only at corrupted spans, others are -100.
- **Constrained decoding / structured output**: VAL / DeepSeek-VL 系列让模型只在特定 JSON 字段位置算 loss / Models like VAL / DeepSeek-VL compute loss only at specific JSON field positions.
- **RT-2 (Brohan et al., 2023)**: OpenVLA 的直系前身,完全同样的训练目标 / OpenVLA's direct predecessor; identical training target.

## 注意事项 / Caveats / when it breaks

- **`IGNORE_INDEX` 写错了静默失败** / **Silently broken if `IGNORE_INDEX` is wrong**: 默认 -100 是 HF 约定,如果你换了个值(比如 0)忘记同步 `F.cross_entropy(ignore_index=...)`,loss 还能算出来但**完全错** — 0 是合法 token id,模型会被监督"前面所有位置都输出 token 0" / Default -100 is HF convention; if you change it (e.g. to 0) and forget to sync `F.cross_entropy(ignore_index=...)`, the loss still computes but is **silently wrong** — 0 is a legal token id, model gets supervised to "output token 0 everywhere".
- **`len(action) + 1` 这个 +1 是 BOS 还是 EOS?** / **What's the `+1` for?**: 保留最后 8 个位置 = 7 action + 1 EOS。如果 `predict_stop_token=False`,EOS 那一位也被 mask 掉,实际只有 7 个有效 loss 位置 / Keeps the last 8 positions = 7 actions + 1 EOS. If `predict_stop_token=False`, EOS is also masked, leaving only 7 effective loss positions.
- **shift-by-1 是 HF 内部做的,别手动 shift** / **HF shifts internally — don't double-shift**: 注释里写得很清楚 `IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!`。如果用自己写的 trainer 又手动 shift 一次,会偏移 2 位 / The comment says it: HF shifts inside `LLM.forward(..., labels=labels)`. If you write a custom trainer and shift again, you'll be off by 2.
- **vocab 末尾 256 slot 必须是低频 token** / **Last 256 vocab slots must be low-frequency tokens**: ActionTokenizer 假设 BPE vocab 末尾是几乎没用的 token(LLaMA-2 的情况确实如此),如果你换 backbone(比如 Mistral / Qwen),要先验证一下;否则 action token "占用"了 BPE 真用得到的字 / ActionTokenizer assumes the last 256 vocab slots are essentially unused (true for LLaMA-2). If you swap backbones (Mistral, Qwen), verify first; otherwise actions will overwrite real BPE tokens.

## 延伸阅读 / Further reading

- OpenVLA paper § 3.3 — the action tokenization scheme and training objective
- HuggingFace `LlamaForCausalLM` source — the standard CE + shift you're "borrowing" verbatim
- "Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks" (Bengio et al., 2015) — the canonical paper on exposure bias, the phenomenon OpenVLA inherits but doesn't fix
- OpenVLA-OFT paper — fixes exposure bias by replacing the autoregressive decode with parallel L1 regression head
