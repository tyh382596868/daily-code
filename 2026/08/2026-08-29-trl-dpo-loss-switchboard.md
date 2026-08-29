---
date: 2026-08-29
topic: huggingface
source: huggingface
repo: huggingface/trl
file: trl/trainer/dpo_trainer.py
permalink: https://github.com/huggingface/trl/blob/6630e17a42976fb1db84034e94e0cf058e8baa21/trl/trainer/dpo_trainer.py#L1323-L1365
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, huggingface, trl, dpo]
---

# TRL DPO loss：偏好学习先变成 log-ratio / TRL DPO Loss: Preference Learning Starts as a Log Ratio

> **一句话 / In one line**: TRL 的 DPO loss 先比较 chosen/rejected 在 policy 和 reference 下的 logprob 差，再按不同 loss 类型把这个 margin 变成训练信号。 / TRL's DPO loss first compares chosen/rejected logprob gaps under policy and reference models, then turns that margin into a training signal with the selected loss type.

## 为什么重要 / Why this matters

DPO 的核心不是“chosen 分数高就行”，而是 policy 相对 reference 是否更偏向 chosen。TRL 把这件事拆成几个 log-ratio，再支持 sigmoid、hinge、robust 等不同目标，让同一个 trainer 能覆盖多种偏好优化变体。

DPO is not simply "score the chosen answer higher"; it asks whether the policy prefers the chosen answer more than the reference does. TRL decomposes that into log ratios and supports sigmoid, hinge, robust, and other objectives in the same trainer.

## 代码 / The code

`huggingface/trl` — [`trl/trainer/dpo_trainer.py`](https://github.com/huggingface/trl/blob/6630e17a42976fb1db84034e94e0cf058e8baa21/trl/trainer/dpo_trainer.py#L1323-L1365)

```python
def dpo_loss(
    self,
    chosen_logps: torch.FloatTensor,
    rejected_logps: torch.FloatTensor,
    ref_chosen_logps: torch.FloatTensor,
    ref_rejected_logps: torch.FloatTensor,
):
    chosen_logratios = chosen_logps.to(self.accelerator.device) - (
        not self.reference_free
    ) * ref_chosen_logps.to(self.accelerator.device)
    rejected_logratios = rejected_logps.to(self.accelerator.device) - (
        not self.reference_free
    ) * ref_rejected_logps.to(self.accelerator.device)

    if self.f_divergence_type == FDivergenceType.ALPHA_DIVERGENCE.value:
        alpha_coef = FDivergenceConstants.ALPHA_DIVERGENCE_COEF_DEFAULT
        if self.f_divergence_params and FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY in self.f_divergence_params:
            alpha_coef = float(self.f_divergence_params[FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY])
        logits = (cap_exp(rejected_logratios * -alpha_coef) - cap_exp(chosen_logratios * -alpha_coef)) / alpha_coef
    else:
        logratios = chosen_logps - rejected_logps
        if self.reference_free:
            ref_logratios = torch.tensor([0], dtype=logratios.dtype, device=logratios.device)
        else:
            ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits = logratios - ref_logratios

    if self.loss_type == "sigmoid":
        losses = (
            -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
            - F.logsigmoid(-self.beta * logits) * self.label_smoothing
        )
    elif self.loss_type == "hinge":
        losses = torch.relu(1 - self.beta * logits)
    return losses
```

## 逐行讲解 / What's happening

1. **第 1331-1336 行 / Lines 1331-1336**:
   - 中文: 先算 policy logprob 减 reference logprob；`reference_free` 时 reference 项被乘成 0。
   - English: It starts with policy logprob minus reference logprob; under `reference_free`, the reference term is multiplied away.
2. **第 1338-1343 行 / Lines 1338-1343**:
   - 中文: alpha divergence 分支不是简单 margin，而是用 capped exponential 改写 chosen/rejected 的差。
   - English: The alpha-divergence branch uses capped exponentials instead of a plain margin.
3. **第 1345-1351 行 / Lines 1345-1351**:
   - 中文: 标准 DPO 分支先得到 chosen-rejected 的 policy margin，再减去 reference margin。
   - English: The standard DPO branch computes the policy chosen-minus-rejected margin, then subtracts the reference margin.
4. **第 1353-1361 行 / Lines 1353-1361**:
   - 中文: `sigmoid` 是软目标，`hinge` 是间隔目标；两者共享同一个 `logits`。
   - English: `sigmoid` is a soft objective, while `hinge` is a margin objective; both consume the same `logits`.

## 类比 / The analogy

这像比较两位评委的偏好变化。不是看新评委给 A 打了多少分，而是看他比旧评委更偏向 A 还是更偏向 B。

It is like comparing how two judges changed their preference. You do not only ask how highly the new judge scored A; you ask whether the new judge favors A over B more than the old judge did.

## 自己跑一遍 / Try it yourself

```python
import math

chosen, rejected = -1.0, -2.0
ref_chosen, ref_rejected = -1.2, -1.8
beta = 0.5

policy_margin = chosen - rejected
ref_margin = ref_chosen - ref_rejected
logit = policy_margin - ref_margin
sigmoid_loss = -math.log(1 / (1 + math.exp(-beta * logit)))
hinge_loss = max(0, 1 - beta * logit)

print(round(logit, 3))
print(round(sigmoid_loss, 3))
print(round(hinge_loss, 3))
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
0.4
0.598
0.8
```

这里的 `0.4` 是 policy 相对 reference 多出来的偏好 margin。

The `0.4` is the extra preference margin the policy has relative to the reference.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **RLHF KL penalty** / **RLHF KL penalty**: 训练目标经常比较 policy 和 reference 的偏离。 / Objectives often compare policy behavior against a reference model.
- **ORPO / CPO variants** / **ORPO / CPO variants**: 也把 chosen/rejected 差值变成可优化 margin。 / They also turn chosen/rejected gaps into an optimizable margin.
- **contrastive losses** / **contrastive losses**: 核心都是把“好样本比坏样本强多少”写成 logit。 / The common idea is expressing how much a good sample beats a bad one as a logit.

## 注意事项 / Caveats / when it breaks

- **logprob 形状要对齐** / **Logprob shapes must align**: chosen/rejected 和 reference tensor 必须逐样本对应。 / Chosen/rejected and reference tensors must match sample by sample.
- **reference-free 改变语义** / **Reference-free changes meaning**: 此时目标不再是“相对 reference 的改进”。 / The objective is no longer an improvement over a reference model.
- **`beta` 控制强度** / **`beta` controls sharpness**: 太大可能让 loss 对 margin 过度敏感。 / Too large a value can make the loss overly sensitive to the margin.

## 延伸阅读 / Further reading

- TRL DPO trainer: https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py
- TRL docs: https://huggingface.co/docs/trl/
