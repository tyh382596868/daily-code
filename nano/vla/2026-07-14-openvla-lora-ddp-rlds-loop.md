---
date: 2026-07-14
topic: vla
source: vla
repo: openvla/openvla
file: vla-scripts/finetune.py
permalink: https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/vla-scripts/finetune.py#L139-L277
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, vla, fine-tune-lora, rlds, ddp]
build_role: fine-tune-lora advanced training entry for a from-scratch nanoVLA
---

# OpenVLA fine-tune：AutoClass、LoRA、DDP、RLDS 接成一条训练线 / OpenVLA fine-tune: AutoClass, LoRA, DDP, and RLDS as One Training Line

> **一句话 / In one line**: OpenVLA 的 fine-tune 入口先注册 HF AutoClass，再按需包 LoRA/量化，最后用 RLDS batch transform 直接喂动作监督。 / OpenVLA's fine-tune entry registers HF AutoClasses, optionally wraps LoRA/quantization, then feeds action supervision through an RLDS batch transform.

## 为什么重要 / Why this matters

VLA 微调不是“加载模型然后 backward”这么简单。图像 processor、action tokenizer、RLDS 数据、LoRA adapter、DDP wrapper 和动作指标必须共享同一个 tokenizer/model contract。这段代码把这些合约按顺序接起来，是生产式微调脚本的骨架。

VLA fine-tuning is not just “load model and call backward.” The image processor, action tokenizer, RLDS data, LoRA adapter, DDP wrapper, and action metrics must share one tokenizer/model contract. This code wires those contracts in order and forms the skeleton of a production fine-tuning script.

## 代码 / The code

`openvla/openvla` — [`vla-scripts/finetune.py`](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/vla-scripts/finetune.py#L139-L277)

```python
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
optimizer = AdamW([param for param in vla.parameters() if param.requires_grad], lr=cfg.learning_rate)
action_tokenizer = ActionTokenizer(processor.tokenizer)

batch_transform = RLDSBatchTransform(
    action_tokenizer,
    processor.tokenizer,
    image_transform=processor.image_processor.apply_transform,
    prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
)
vla_dataset = RLDSDataset(..., batch_transform, resize_resolution=tuple(vla.module.config.image_sizes), ...)
```

## 逐行讲解 / What's happening

1. **AutoClass 注册 / AutoClass registration**:
   - 中文: 自定义 OpenVLA config、processor、model 注册进 Transformers 的通用加载入口。
   - English: Custom OpenVLA config, processor, and model are registered into Transformers' generic loading path.
2. **量化和 LoRA 顺序 / Quantization and LoRA order**:
   - 中文: 量化训练先 `prepare_model_for_kbit_training`，再用 PEFT 把可训练低秩层挂进去。
   - English: Quantized training first calls `prepare_model_for_kbit_training`, then PEFT attaches trainable low-rank layers.
3. **DDP 包装后只优化可训练参数 / Optimize trainable params after DDP wrapping**:
   - 中文: optimizer 只拿 `requires_grad=True` 的参数，LoRA 模式不会误更新整模型。
   - English: The optimizer receives only `requires_grad=True` parameters, so LoRA mode does not update the whole model.
4. **RLDS transform 统一动作和文本 / RLDS transform unifies actions and text**:
   - 中文: `ActionTokenizer` 和 `processor.tokenizer` 一起进入 batch transform，保证 action label 和语言 token 对齐。
   - English: `ActionTokenizer` and `processor.tokenizer` enter the batch transform together, keeping action labels aligned with text tokens.

## 类比 / The analogy

这像改装一辆车上赛道：先确认底盘和接口标准，再换轻量化套件，最后接上计时器和赛道数据。顺序错了，零件能装上也跑不稳。

It is like preparing a car for a track: first confirm chassis and interface standards, then add lightweight parts, then connect timing and track data. If the order is wrong, parts may fit but the car will not run reliably.

## 在 nanoVLA 中的位置 / Where this lives in your nano-VLA

这是 `fine-tune-lora` 的高级版本，依赖 `vision-encoder`、`vlm-backbone-wiring`、`action-tokenizer` 和 `training-step` 都已经存在。nanoVLA 最小版可以先全参训练一个小模型；一旦要复用大 VLM，就需要这类入口把 processor、adapter、数据 transform 和 action metric 固定成一个合约。

This is an advanced `fine-tune-lora` variant, assuming `vision-encoder`, `vlm-backbone-wiring`, `action-tokenizer`, and `training-step` already exist. A minimal nanoVLA can start with full fine-tuning of a small model; once it reuses a large VLM, this entry point freezes the processor, adapter, data transform, and action metrics into one contract.

## 自己跑一遍 / Try it yourself

```python
params = {"backbone": False, "lora_A": True, "lora_B": True}
trainable = [name for name, requires_grad in params.items() if requires_grad]
batch = {"text": "pick cup", "actions": [3, 4, 5]}
labels = [tok for tok in batch["actions"] if tok > 2]
print(trainable)
print(labels)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
['lora_A', 'lora_B']
[3, 4, 5]
```

LoRA 微调的核心是只让 adapter 参数进 optimizer，同时 action label 仍然走同一套 tokenizer 合约。

The core of LoRA fine-tuning is that only adapter parameters enter the optimizer while action labels still use the same tokenizer contract.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **PEFT LoRA training** / **PEFT LoRA training**: 大语言模型微调也先冻结底座，再只训练 adapter。 / LLM fine-tuning also freezes the base model and trains only adapters.
- **LeRobot policy training** / **LeRobot policy training**: 也把 dataset transform、policy forward 和 action metrics 放进同一个训练 step。 / It also keeps dataset transforms, policy forward, and action metrics in one training step.

## 注意事项 / Caveats / when it breaks

- **AutoClass 注册要早 / Register AutoClasses early**: 在 `from_pretrained` 之后注册已经来不及。 / Registering after `from_pretrained` is too late.
- **DDP 包装改变访问路径 / DDP changes access paths**: 包装后配置要通过 `vla.module.config` 访问。 / After wrapping, config access goes through `vla.module.config`.

## 延伸阅读 / Further reading

- [OpenVLA fine-tune script](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/vla-scripts/finetune.py#L139-L277)
