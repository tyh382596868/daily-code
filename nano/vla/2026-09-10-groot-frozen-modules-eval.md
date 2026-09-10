---
date: 2026-09-10
topic: vla
source: vla
repo: NVIDIA/Isaac-GR00T
file: gr00t/model/gr00t_n1d7/gr00t_n1d7.py
permalink: https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/model/gr00t_n1d7/gr00t_n1d7.py#L140-L157
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, fine-tune-lora, frozen-modules, training]
build_role: fine-tune-lora advanced variant
---

# GR00T frozen eval：冻结参数还要冻结行为 / GR00T Frozen Eval: Freeze Behavior, Not Just Parameters

> **一句话 / In one line**: GR00T 的 action head 在训练时会把被冻结的 projector、DiT、VLLN 模块切回 eval，避免 Trainer 的全局 `train()` 改变 dropout/batchnorm 行为。 / GR00T's action head switches frozen projector, DiT, and VLLN modules back to eval during training so the Trainer's global `train()` call does not change dropout or batchnorm behavior.

## 为什么重要 / Why this matters

中文：微调 VLA 时，经常只训练少数 adapter 或 action head 层。把参数 `requires_grad=False` 只是阻止梯度更新，不能阻止 dropout、batchnorm 或其他 training-mode 分支改变输出。GR00T 在 forward 开始前主动恢复冻结模块的 eval mode，把“不可训练”和“行为稳定”绑在一起。

English: VLA fine-tuning often trains only adapters or a subset of the action head. Setting `requires_grad=False` prevents parameter updates, but it does not stop dropout, batchnorm, or training-mode branches from changing outputs. GR00T restores eval mode for frozen modules at the start of forward, pairing non-trainability with stable behavior.

## 代码 / The code

`NVIDIA/Isaac-GR00T` — [`gr00t/model/gr00t_n1d7/gr00t_n1d7.py`](https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/model/gr00t_n1d7/gr00t_n1d7.py#L140-L157)

```python
def set_frozen_modules_to_eval_mode(self):
    """
    Huggingface will call model.train() at each training_step. To ensure
    the expected behaviors for modules like dropout, batchnorm, etc., we
    need to call model.eval() for the frozen modules.
    """
    if self.training:
        if not self.tune_projector:
            self.state_encoder.eval()
            self.action_encoder.eval()
            self.action_decoder.eval()
            if self.config.add_pos_embed:
                self.position_embedding.eval()
        if not self.tune_diffusion_model:
            self.model.eval()
        if not self.tune_vlln:
            self.vlln.eval()
            self.vl_self_attention.eval()
```

The forward path calls it before encoding features:

```python
# Set frozen modules to eval
self.set_frozen_modules_to_eval_mode()
backbone_output = self.process_backbone_output(backbone_output)
```

## 逐行讲解 / What's happening

1. **第 140-145 行 / Lines 140-145**:
   - 中文: 注释点明触发源：Hugging Face Trainer 会在每个 training step 调用全模型 `train()`。
   - English: The comment names the trigger: Hugging Face Trainer calls `train()` on the whole model at each training step.
2. **第 146-152 行 / Lines 146-152**:
   - 中文: projector 系列不调时，state/action encoder、decoder 和可选 position embedding 都回到 eval。
   - English: When the projector stack is not tuned, the state/action encoders, decoder, and optional position embedding return to eval.
3. **第 153-154 行 / Lines 153-154**:
   - 中文: action diffusion/DiT 主体不调时，也保持 eval mode。
   - English: If the action diffusion/DiT body is frozen, it also stays in eval mode.
4. **第 155-157 行 / Lines 155-157**:
   - 中文: VLM feature normalization 和额外 self-attention 冻结时同样处理，避免上下文特征漂移。
   - English: VLM feature normalization and extra self-attention receive the same handling when frozen, avoiding context feature drift.

## 类比 / The analogy

中文：像工厂里某条生产线被封存。只拔掉工资卡不够，还要把机器开关拨到待机；否则传送带可能还会按训练节奏乱动。

English: It is like mothballing one line in a factory. Stopping payroll is not enough; the machines also need to be put in standby, or the conveyor can still move with the training schedule.

## 在 nanoVLA 中的位置 / Where this lives in your nanoVLA

中文：这是 `fine-tune-lora` 的 advanced variant，位于模型 forward 的最前面、特征编码之前。上游是训练框架统一切到 training mode 后的模型；下游是只让目标 adapter/head 更新的稳定 forward。生产版 nanoVLA 应把 freeze plan 设计成显式配置：哪些模块训练、哪些模块 eval、哪些 normalization 统计冻结，并在每个 step 前校验它们没有被框架回写。

English: This is an advanced `fine-tune-lora` variant at the front of model forward, before feature encoding. The upstream state is a model globally switched to training mode by the trainer; downstream is a stable forward where only selected adapters or heads update. A production nanoVLA should make the freeze plan explicit: which modules train, which stay eval, which normalization statistics are frozen, and verify them before every step.

## 自己跑一遍 / Try it yourself

```python
class Module:
    def __init__(self, name):
        self.name = name
        self.training = True
    def eval(self):
        self.training = False

modules = {"encoder": Module("encoder"), "adapter": Module("adapter")}
tune_encoder = False

if not tune_encoder:
    modules["encoder"].eval()

print({name: module.training for name, module in modules.items()})
```

运行 / Run with:

```bash
python try.py
```

预期输出 / Expected output:

```text
{'encoder': False, 'adapter': True}
```

中文：冻结参数和切 eval 是两件事；微调代码要同时表达。

English: Freezing parameters and switching to eval are separate actions; fine-tuning code should express both.

## 注意事项 / Caveats / when it breaks

- **eval 不等于 no grad** / **Eval is not no-grad**: eval mode 不会自动关闭梯度，仍需设置 `requires_grad`。
- **训练框架可能反复调用 `train()`** / **The trainer may repeatedly call `train()`**: 一次性设置 eval 不够，最好在 forward 或 callback 中恢复。
- **adapter 不能被误冻结** / **Adapters must not be frozen accidentally**: freeze plan 要和 optimizer parameter groups 一起测试。

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **PEFT/LoRA fine-tuning** / **PEFT/LoRA fine-tuning**: base model 常冻结，但 adapter 保持训练。
- **vision backbone freezing** / **vision backbone freezing**: 图像 encoder 的 dropout/batchnorm 行为通常要固定。
- **teacher-student training** / **teacher-student training**: teacher 不训练时也应保持 eval。

## 延伸阅读 / Further reading

- [Isaac-GR00T gr00t_n1d7.py](https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/model/gr00t_n1d7/gr00t_n1d7.py)
- [Hugging Face Trainer](https://huggingface.co/docs/transformers/trainer)
