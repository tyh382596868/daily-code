---
date: 2026-07-03
topic: vla
source: vla
repo: openvla/openvla
file: vla-scripts/finetune.py
permalink: https://github.com/openvla/openvla/blob/main/vla-scripts/finetune.py#L97-L180
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, fine-tune-lora, peft]
build_role: fine-tune-lora variant for adapting a pretrained VLA
---

# OpenVLA LoRA 微调栈：先注册 AutoClass，再包 PEFT / OpenVLA LoRA Fine-Tune Stack: Register AutoClasses, Then Wrap with PEFT

> **一句话 / In one line**: 这段脚本把 OpenVLA 接进 Hugging Face AutoClasses，再用 `LoraConfig(target_modules="all-linear")` 只训练低秩 adapter。 / This script wires OpenVLA into Hugging Face AutoClasses, then trains only low-rank adapters with `LoraConfig(target_modules="all-linear")`.

## 为什么重要 / Why this matters

生产 VLA 很少从零训练。更常见的路径是拿一个视觉-语言-动作底座，用少量机器人数据做参数高效微调。OpenVLA 的脚本展示了完整入口：量化可选、AutoClass 注册、processor/model 加载、LoRA 包装、DDP 和 optimizer。

Production VLAs are rarely trained from scratch. A more common path is to start from a vision-language-action base model and adapt it with a small robot dataset. OpenVLA's script shows the full entry point: optional quantization, AutoClass registration, processor/model loading, LoRA wrapping, DDP, and optimizer setup.

## 代码 / The code

`openvla/openvla` — [`vla-scripts/finetune.py`](https://github.com/openvla/openvla/blob/main/vla-scripts/finetune.py#L97-L180)

```python
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
```

## 逐行讲解 / What's happening

1. **量化只服务 LoRA / Quantization is only for LoRA**: 中文: 4-bit 训练被限制在 LoRA 路径，避免全量量化训练的复杂性。 / English: 4-bit training is restricted to the LoRA path, avoiding the complexity of fully quantized training.
2. **注册 AutoClass / Register AutoClasses**: 中文: 这几行让 `AutoProcessor` 和 `AutoModelForVision2Seq` 能识别 OpenVLA 自定义类型。 / English: These lines teach `AutoProcessor` and `AutoModelForVision2Seq` how to instantiate OpenVLA's custom classes.
3. **PEFT 包装 / PEFT wrapping**: 中文: `target_modules="all-linear"` 把低秩 adapter 放到所有 Linear 上，覆盖语言模型和投影器里的主要可训练矩阵。 / English: `target_modules="all-linear"` places adapters on all Linear layers, covering the main trainable matrices in the language model and projector.

## 类比 / The analogy

像给一台成品机械臂换一套可拆的夹具：本体不重新铸造，只在关键连接处加可训练的小部件，让它学会新任务。

It is like adding removable tooling to a finished robot arm. You do not rebuild the whole arm; you attach small trainable parts at key joints so it can learn a new task.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

这是 `fine-tune-lora` 组件的生产版变体。上游是已经能做 `predict_action` 的 VLA 底座和 RLDS 数据管道，下游是训练循环和 checkpoint 保存。nanoVLA 里可以先只支持 projector/LM 的 LoRA；生产版再补量化、DDP、统计保存和恢复训练。

This is a production-style variant of the `fine-tune-lora` component. Upstream are a VLA base model that can already run `predict_action` and an RLDS data pipeline; downstream are the training loop and checkpoint writer. In nanoVLA, start with LoRA on the projector/LM only; a production version adds quantization, DDP, statistics saving, and resume support.

## 自己跑一遍 / Try it yourself

```python
import torch
import torch.nn as nn

base = nn.Linear(4, 3, bias=False)
rank = 2
lora_a = nn.Linear(4, rank, bias=False)
lora_b = nn.Linear(rank, 3, bias=False)
for p in base.parameters():
    p.requires_grad = False
x = torch.randn(5, 4)
y = base(x) + lora_b(lora_a(x))
loss = y.pow(2).mean()
loss.backward()
print(base.weight.grad, lora_a.weight.grad.shape, lora_b.weight.grad.shape)
```

运行 / Run with:

```bash
pip install torch
python try.py
```

预期输出 / Expected output:

```text
None torch.Size([2, 4]) torch.Size([3, 2])
```

底座权重没有梯度，LoRA 两个小矩阵有梯度；这就是参数高效微调的最小形态。

The base weight has no gradient, while the two LoRA matrices do. That is the smallest useful shape of parameter-efficient fine-tuning.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **PEFT LoRA trainer** / **PEFT LoRA trainer**: Hugging Face PEFT 也是先冻结底座，再注入 adapter。
- **QLoRA** / **QLoRA**: 把底座量化到低 bit，仍然只训练 LoRA adapter。

## 注意事项 / Caveats / when it breaks

- **`all-linear` 不一定总合适** / **`all-linear` is not always right**: 小数据集上可能过拟合，生产训练要比较 target module 范围。
- **动作统计必须同步保存** / **Action statistics must be saved with the adapter**: VLA 输出需要反归一化，adapter 单独保存不够。

## 延伸阅读 / Further reading

- [OpenVLA finetune script](https://github.com/openvla/openvla/blob/main/vla-scripts/finetune.py)
- [PEFT LoRA documentation](https://huggingface.co/docs/peft/main/en/package_reference/lora)
