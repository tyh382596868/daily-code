---
date: 2026-06-26
topic: robotics
source: tracked
repo: starVLA/starVLA
file: starVLA/training/train_starvla_cotrain.py
permalink: https://github.com/starVLA/starVLA/blob/6dc01d0781a817c007f74927a75bf63d89d521e2/starVLA/training/train_starvla_cotrain.py#L357-L397
difficulty: advanced
read_time: ~12 min
tags: [code-of-the-day, robotics, cotrain, deepspeed, accelerate, vla, distributed-training]
---

# StarVLA 双流协同训练：一个 step 里两条 backward / StarVLA Dual-Stream Cotrain: Two Backward Passes in One Step

> **一句话 / In one line**: `VLAMTrainer._train_step` 在同一个优化器步里先对动作 DiT loss backward、再对 VLM 语言 loss backward，同时兼容 DeepSpeed ZeRO 和 Hugging Face Accelerate 两种分布式框架。 / `VLAMTrainer._train_step` calls backward twice in one optimizer step — first for the action DiT loss, then for the VLM language loss — and handles both DeepSpeed ZeRO and HuggingFace Accelerate backends transparently.

## 为什么重要 / Why this matters

VLA 模型通常面临一个两难困境：要让动作头（往往是 DiT 或 flow-matching 模块）产生高质量的轨迹，同时又不能让骨干 VLM 的语言能力退化。StarVLA 的解法是**协同训练（cotrain）**：每个 step 同时计算两条 loss，一条来自动作 DiT（`action_loss`），另一条来自 VLM 语言模型目标（`vlm_loss`），梯度在骨干网络上自然叠加。

难点不在算法层面，而在工程层面：DeepSpeed ZeRO-2/3 和 Hugging Face Accelerate 对"梯度累积边界"和"backward"有完全不同的 API。这段代码用一个 `hasattr` 检测分支优雅地把两套系统统一在同一个 `_train_step` 里，让上层调用者完全不用关心底层框架。

Most VLA training setups face a dilemma: the action head (typically a DiT or flow-matching module) needs high-quality training signal, but the backbone VLM can quietly lose its language capabilities. StarVLA's solution is **cotraining**: every step computes two losses — `action_loss` from the action DiT and `vlm_loss` from the VLM language objective — letting their gradients accumulate naturally on the shared backbone.

The hard part isn't the algorithm; it's the engineering: DeepSpeed ZeRO-2/3 and HuggingFace Accelerate have fundamentally different APIs for gradient accumulation boundaries and backward passes. This code uses a single `hasattr` branch to unify both systems under one `_train_step`, so the caller never needs to care which backend is running.

## 代码 / The code

`starVLA/starVLA` — [`starVLA/training/train_starvla_cotrain.py`](https://github.com/starVLA/starVLA/blob/6dc01d0781a817c007f74927a75bf63d89d521e2/starVLA/training/train_starvla_cotrain.py#L357-L397)

```python
def _train_step(self, batch_vla, batch_vlm):
    """Execute single training step."""
    log_dict = {}
    # DeepSpeed path (ZeRO stage 2/3):
    if hasattr(self.model, "is_gradient_accumulation_boundary") and hasattr(self.model, "backward"):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output_dict = self.model.forward(batch_vla)
            action_loss = output_dict["action_loss"]
        self.model.backward(action_loss)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            unwrapped = self.accelerator.unwrap_model(self.model)
            vlm_output = unwrapped.qwen_vl_interface(**batch_vlm)
            vlm_loss = vlm_output.loss * self.config.trainer.loss_scale.vlm
        self.model.backward(vlm_loss)
        optimizer_stepped = bool(self.model.is_gradient_accumulation_boundary())
        self.model.step()
        if optimizer_stepped:
            self.lr_scheduler.step()
        log_dict.update({
            "action_dit_loss": action_loss.item(),
            "vlm_loss": vlm_loss.item(),
            "_optimizer_step": optimizer_stepped,
        })
        return log_dict
    # Accelerate path:
    with self.accelerator.accumulate(self.model):
        self.optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output_dict = self.model.forward(batch_vla)
            action_loss = output_dict["action_loss"]
            total_loss = action_loss
        self.accelerator.backward(total_loss)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            unwrapped = self.accelerator.unwrap_model(self.model)
            vlm_output = unwrapped.qwen_vl_interface(**batch_vlm)
            vlm_loss = vlm_output.loss * self.config.trainer.loss_scale.vlm
        self.accelerator.backward(vlm_loss)
        if self.config.trainer.gradient_clipping is not None:
            self.accelerator.clip_grad_norm_(
                self.model.parameters(), self.config.trainer.gradient_clipping
            )
        self.optimizer.step()
        if self.accelerator.sync_gradients:
            self.lr_scheduler.step()
        log_dict.update({
            "action_dit_loss": action_loss.item(),
            "vlm_loss": vlm_loss.item(),
            "_optimizer_step": self.accelerator.sync_gradients,
        })
    return log_dict
```

## 逐行讲解 / What's happening

1. **`hasattr(self.model, "is_gradient_accumulation_boundary") and hasattr(self.model, "backward")`**
   - 中文: DeepSpeed 把模型包成 `engine` 对象，它自带 `.backward()` 和 `.is_gradient_accumulation_boundary()`；Accelerate 不提供这些属性。这一行 duck-typing 检测决定走哪条路径。
   - English: DeepSpeed wraps the model into an engine object that owns `.backward()` and `.is_gradient_accumulation_boundary()`; Accelerate doesn't expose these. This duck-typing check picks the right path without importing DeepSpeed directly.

2. **DeepSpeed 路径：两次 `self.model.backward()`**
   - 中文: DeepSpeed 的 engine 会在内部做梯度累积计数。两次 `backward()` 的梯度在 engine 内部叠加，直到 `is_gradient_accumulation_boundary()` 返回 `True` 时，`model.step()` 才触发一次真正的参数更新和 ZeRO 通信。
   - English: DeepSpeed's engine tracks gradient accumulation internally. The two `backward()` calls accumulate gradients inside the engine. Only when `is_gradient_accumulation_boundary()` returns `True` does `model.step()` trigger an actual parameter update and ZeRO communication.

3. **`unwrapped = self.accelerator.unwrap_model(self.model)` + `unwrapped.qwen_vl_interface(**batch_vlm)`**
   - 中文: VLM 语言损失需要直接调用 `qwen_vl_interface` —— 这是被 DDP/FSDP/DeepSpeed 包裹前的原始模块上的方法，所以两条路径都需要先 unwrap 才能调用。包裹层不会破坏原始属性，只是访问时需要一层间接。
   - English: The VLM language loss calls `qwen_vl_interface`, a method on the original (unwrapped) module. Both paths need `unwrap_model` because DDP/FSDP/DeepSpeed wrappers don't forward unknown attribute names to the inner module.

4. **Accelerate 路径：`with self.accelerator.accumulate(self.model):`**
   - 中文: 这个上下文管理器让 Accelerate 跟踪梯度累积状态。在非边界 step 里，`self.accelerator.sync_gradients` 为 `False`，因此不会触发 all-reduce；在边界 step 里，两次 `backward` 之后会做一次同步，然后执行 `optimizer.step()`。
   - English: This context manager lets Accelerate track gradient accumulation state. On non-boundary steps `sync_gradients` is `False` so no all-reduce fires; on boundary steps a single sync happens after both `backward` calls before `optimizer.step()`.

5. **`vlm_loss = vlm_output.loss * self.config.trainer.loss_scale.vlm`**
   - 中文: VLM 损失按比例缩放，防止语言目标压过动作目标（或反之）。这个 `loss_scale.vlm` 超参数决定了骨干网络的语言能力保持多少——调小会让动作损失主导，调大会更保留语言能力。
   - English: Scaling the VLM loss prevents one objective from dominating the other. `loss_scale.vlm` is the key hyperparameter controlling how much the backbone retains its language capability — lower values let the action loss dominate, higher values preserve more language fidelity.

## 类比 / The analogy

想象一位大厨同时学两门课：刀工课（动作技能）和食材知识课（语言理解）。每次练习（optimizer step），他先练一遍切法、记住手感（backward action_loss），再复习一遍食材搭配（backward vlm_loss），然后把两次的心得统一写进学习笔记（optimizer.step）。DeepSpeed 相当于一个严格记录"今天练了几次"的助教，只有攒够 N 次才让他更新笔记；Accelerate 则是一个稍微宽松的管理方式，但核心节奏完全相同。

Think of a chef learning two skills simultaneously: knife technique (action skill) and ingredient knowledge (language understanding). Each practice session, they first drill a cutting technique and absorb the muscle memory (`backward action_loss`), then review flavor combinations (`backward vlm_loss`), and finally consolidate both into their notes (`optimizer.step`). DeepSpeed is like a strict teaching assistant who only lets them update their notes after exactly N repetitions; Accelerate is a slightly more flexible style, but the rhythm is identical.

## 自己跑一遍 / Try it yourself

```python
import torch
import torch.nn as nn

# Minimal mock of the cotrain step (Accelerate-style path)
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(64, 64)
        self.action_head = nn.Linear(64, 7)
        self.lm_head = nn.Linear(64, 32000)

    def forward_vla(self, x):
        feat = self.backbone(x)
        action = self.action_head(feat)
        return {"action_loss": ((action - torch.randn_like(action)) ** 2).mean()}

    def forward_vlm(self, x, labels):
        logits = self.lm_head(self.backbone(x))
        return torch.nn.functional.cross_entropy(logits, labels)

model = ToyModel()
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

x_vla = torch.randn(4, 64)
x_vlm = torch.randn(4, 64)
labels = torch.randint(0, 32000, (4,))

opt.zero_grad()
out = model.forward_vla(x_vla)
out["action_loss"].backward()  # first backward

vlm_loss = model.forward_vlm(x_vlm, labels) * 0.1
vlm_loss.backward()  # second backward — gradients accumulate on backbone

opt.step()
print(f"action_loss={out['action_loss'].item():.4f}  vlm_loss={vlm_loss.item():.4f}")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
action_loss=X.XXXX  vlm_loss=X.XXXX
```

中文：注意这里调用了两次 `.backward()` 但只调用了一次 `opt.zero_grad()` 和 `opt.step()`。骨干网络（backbone）的梯度会被两次 backward 叠加，而两个 head 各自只得到一次梯度。这正是协同训练梯度叠加的核心机制。

English: Notice two `.backward()` calls but only one `opt.zero_grad()` and one `opt.step()`. The backbone accumulates gradients from both passes; each head only sees one. This is exactly the cotrain gradient-accumulation mechanism in action.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **OpenPi π₀** / **OpenPi π₀**: 也用双流：一条 flow-matching 动作损失 + 一条语言模型 loss，但通过 JAX `pmap` 实现，两条 loss 合并成一个 `total_loss` 再做一次 backward，而不是两次。 / Also uses dual streams: flow-matching action loss + LM loss, but in JAX `pmap` they merge into a single `total_loss` before a single backward rather than two separate passes.
- **OpenVLA** / **OpenVLA**: 只有一条 next-token-prediction loss（动作 token 被离散化进词表），不需要第二条 backward。StarVLA 的 cotrain 是为了保留连续动作 DiT 头与语言头同时训练的梯度叠加。 / Only one next-token-prediction loss (actions are discretized into the vocabulary), so no second backward is needed — StarVLA's cotrain is specifically for keeping a continuous action DiT head co-trained with the language head.
- **Gemini / Gato 风格 multi-task** / **Gemini / Gato-style multi-task**: 把多个任务的 loss 简单相加再 backward 一次；StarVLA 的两次 backward 方式更接近 Meta 的 Joint Embedding Predictive Architecture（JEPA）风格的解耦训练。 / Simply sum losses from multiple tasks and call backward once; StarVLA's two-backward approach is closer to decoupled training as in Meta's JEPA-style joint embedding predictive architectures.

## 注意事项 / Caveats / when it breaks

- **`_optimizer_step` 日志标志** / **`_optimizer_step` logging flag**: 在梯度累积步里，`_optimizer_step=False`，这时 loss 日志里的值不代表真正更新过的参数产生的损失。如果用这个标志做 checkpoint 触发，需要额外注意。 / During accumulation steps `_optimizer_step=False`, so logged loss values don't correspond to freshly updated parameters. Be careful if using this flag to trigger checkpoints.
- **DeepSpeed `unwrap_model` + `qwen_vl_interface`** / **DeepSpeed `unwrap_model` + `qwen_vl_interface`**: 两条路径都通过 `accelerator.unwrap_model` 取底层模块。如果 VLM 接口方法在 ZeRO-3 的分片参数下被调用，需要确保参数已被聚合（`gather_16bit_weights_on_model_save` 或 `GatheredParameters` 上下文）。 / Both paths call `unwrap_model` to reach the inner module. Under ZeRO-3 parameter sharding, calling methods that need materialized parameters requires explicitly gathering them first.
- **学习率调度器位置** / **LR scheduler placement**: 只有当 `optimizer_stepped` 或 `sync_gradients` 为 `True` 时才调用 `lr_scheduler.step()`。如果用线性 warmup scheduler 但记数逻辑出错，会导致 LR 调度提前或延迟。 / `lr_scheduler.step()` is guarded by the optimizer-stepped flag. If the scheduler counts actual optimizer steps rather than micro-steps, an off-by-one in this guard will silently shift the entire LR schedule.

## 延伸阅读 / Further reading

- [StarVLA paper (arXiv)](https://arxiv.org/abs/2506.09578) — 详细描述双流协同训练的实验设置和消融
- [DeepSpeed ZeRO paper](https://arxiv.org/abs/1910.02054) — ZeRO stage 2/3 的参数分片与梯度聚合原理
- [HuggingFace Accelerate gradient accumulation docs](https://huggingface.co/docs/accelerate/en/usage_guides/gradient_accumulation)
