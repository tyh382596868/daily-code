---
date: 2026-07-01
topic: huggingface
source: huggingface
repo: huggingface/trl
file: trl/experimental/sdft/teacher_sync.py
permalink: https://github.com/huggingface/trl/blob/40f670fc06007daa3d4824198ed47e477cb1a2cd/trl/experimental/sdft/teacher_sync.py#L68-L179
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, huggingface, peft, ema-teacher]
---

# TRL 的 PEFT adapter EMA teacher：不用复制整模型的自蒸馏 / TRL PEFT Adapter EMA Teacher: Self-Distillation Without Copying the Whole Model

> **一句话 / In one line**: 把 teacher 做成第二个 LoRA adapter，只 EMA adapter 权重，而不是复制一份 70B base model。 / Make the teacher a second LoRA adapter and EMA only adapter weights instead of duplicating a 70B base model.

## 为什么重要 / Why this matters

自蒸馏通常需要一个 teacher 模型，但大模型复制一份太贵。TRL 这段 callback 只保存 PEFT adapter 的 shadow weights：初始化一个全零 teacher adapter，训练中对 student adapter 做 EMA，再临时切到 teacher adapter 写回权重。base model 始终共享。

Self-distillation usually needs a teacher model, but duplicating a large model is expensive. This TRL callback keeps only PEFT adapter shadow weights: initialize a zero teacher adapter, EMA the student adapter during training, then temporarily switch adapters to write teacher weights back. The base model stays shared.

## 代码 / The code

`huggingface/trl` — [`trl/experimental/sdft/teacher_sync.py`](https://github.com/huggingface/trl/blob/40f670fc06007daa3d4824198ed47e477cb1a2cd/trl/experimental/sdft/teacher_sync.py#L68-L179)

````python
class PEFTAdapterEMACallback(TrainerCallback):
    """
    Callback that maintains an EMA copy of PEFT adapter weights for use as a teacher model in self-distillation.

    The callback creates a secondary adapter ("teacher") with zero-initialized weights and maintains shadow weights
    that are updated via exponential moving average: `teacher_weight = (1-α) * teacher_weight + α * student_weight`

    Usage:
        ```python
        >>> trainer.add_callback(
        ...     PEFTAdapterEMACallback(
        ...         model=model,
        ...         teacher_adapter_name="teacher",
        ...         update_rate=0.05,
        ...     )
        ... )
        ```
    """

    def __init__(
        self,
        model,
        teacher_adapter_name: str = "teacher",
        update_rate: float = 0.05,
        sync_steps: int = 1,
        accelerator=None,
    ):
        self.model = model
        self.teacher_adapter_name = teacher_adapter_name
        self.update_rate = update_rate
        self.sync_steps = sync_steps
        self.accelerator = accelerator
        self.shadow_weights: dict[str, torch.Tensor] | None = None
        self.teacher_adapter_config = None
        self._initialized = False

    def _get_student_state_dict(self):
        """Get student adapter state dict using PEFT keys (without adapter name)."""
        from peft import get_peft_model_state_dict

        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(self.model)
        else:
            model = self.model
        return get_peft_model_state_dict(model)

    def _initialize_teacher_adapter(self):
        """Create teacher adapter with zero weights initialized from student adapter."""
        from peft import get_peft_model_state_dict, set_peft_model_state_dict

        if self._initialized:
            return

        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(self.model)
        else:
            model = self.model

        adapter_name = model.active_adapter
        if adapter_name is None:
            adapter_name = "default"

        self.teacher_adapter_config = model.peft_config.get(adapter_name)

        student_state = get_peft_model_state_dict(model)

        teacher_state = {k: torch.zeros_like(v) for k, v in student_state.items()}

        model.add_adapter(self.teacher_adapter_name, self.teacher_adapter_config)

        model.set_adapter(self.teacher_adapter_name)
        set_peft_model_state_dict(model, teacher_state, adapter_name=self.teacher_adapter_name)

        model.set_adapter(adapter_name)

        self.shadow_weights = {k: v.clone().zero_() for k, v in teacher_state.items()}

        self._initialized = True
        logger.info(f"Initialized PEFT adapter EMA teacher with adapter name: {self.teacher_adapter_name}")

    @torch.no_grad()
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % self.sync_steps != 0:
            return

        if not self._initialized:
            self._initialize_teacher_adapter()

        if self.shadow_weights is None:
            return

        if self.accelerator is None and "accelerator" in kwargs:
            self.accelerator = kwargs["accelerator"]

        student_state = self._get_student_state_dict()

        for key, student_param in student_state.items():
            if key in self.shadow_weights:
                shadow = self.shadow_weights[key]
                shadow.data = (1 - self.update_rate) * shadow.data + self.update_rate * student_param.data

        from peft import set_peft_model_state_dict

        if self.accelerator is not None:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
        else:
            unwrapped_model = self.model

        original_adapter = unwrapped_model.active_adapter
        unwrapped_model.set_adapter(self.teacher_adapter_name)
        set_peft_model_state_dict(unwrapped_model, self.shadow_weights, adapter_name=self.teacher_adapter_name)
        unwrapped_model.set_adapter(original_adapter)
````

## 逐行讲解 / What's happening

1. **第 87-103 行 / Lines 87-103**: 中文: callback 保存模型、teacher adapter 名称、更新率、同步步数和 shadow 权重容器。 / English: The callback stores the model, teacher adapter name, update rate, sync interval, and shadow-weight container.
2. **第 114-146 行 / Lines 114-146**: 中文: 初始化时复制当前 adapter 配置，但把权重置零，新增为 `teacher` adapter，再切回原 adapter。 / English: Initialization copies the active adapter config, zeroes its weights, adds it as the `teacher` adapter, then switches back.
3. **第 148-179 行 / Lines 148-179**: 中文: 每隔 `sync_steps` 读取 student adapter 权重，执行 `shadow = (1-alpha) shadow + alpha student`，再写入 teacher adapter。 / English: Every `sync_steps`, it reads student adapter weights, applies `shadow = (1-alpha) shadow + alpha student`, and writes the result into the teacher adapter.

## 类比 / The analogy

像同一本教材夹了两套便利贴：学生便利贴每天改，老师便利贴只按滑动平均慢慢更新，书本本身不用复印。

It is like one textbook with two sets of sticky notes: the student notes change every day, the teacher notes move by a slow average, and the textbook itself is never copied.


## 自己跑一遍 / Try it yourself

```python
import torch
student = {'lora_A': torch.tensor([1.0, 3.0])}
shadow = {'lora_A': torch.zeros(2)}
alpha = 0.25
for step in range(3):
    shadow['lora_A'] = (1-alpha)*shadow['lora_A'] + alpha*student['lora_A']
    print(step, shadow['lora_A'].tolist())
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```text
0 [0.25, 0.75]
1 [0.4375, 1.3125]
2 [0.578125, 1.734375]
```

中文: 这个小例子保留了源码里的关键控制流，但把依赖压到最低，便于你直接观察形状、索引或状态变化。

English: The miniature keeps the original control-flow idea while stripping dependencies down so the shape, index, or state change is visible.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **EMA model weights** / **EMA model weights**: 中文: diffusion 和 RL 里常用 EMA teacher 稳定目标。 / English: Diffusion and RL often use EMA teachers to stabilize targets.
- **Multiple adapters** / **Multiple adapters**: 中文: PEFT 的多 adapter 机制让同一个 base model 承载多个行为版本。 / English: PEFT's multi-adapter mechanism lets one base model host multiple behavior versions.

## 注意事项 / Caveats / when it breaks

- **只适合纯 LoRA / Best for pure LoRA**: 中文: 如果还有非 adapter 参数在训练，teacher 只跟踪 adapter 会不完整。 / English: If non-adapter parameters are also trained, tracking only adapters is incomplete.
- **同步频率影响滞后 / Sync cadence controls lag**: 中文: `sync_steps` 太大时 teacher 会明显落后。 / English: A large `sync_steps` makes the teacher lag noticeably.

## 延伸阅读 / Further reading

- Source permalink above.
- Project repository linked from the frontmatter.
