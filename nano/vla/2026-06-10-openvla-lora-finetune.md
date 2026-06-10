---
date: 2026-06-10
topic: vla
source: vla
repo: openvla/openvla
file: vla-scripts/finetune.py
permalink: https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/vla-scripts/finetune.py#L113-L220
difficulty: intermediate
read_time: ~11 min
tags: [code-of-the-day, vla, lora, fine-tuning, peft, ddp]
build_role: nanoVLA / fine-tune-lora — the LoRA entry-point script for adapting a pretrained VLA to a new robot dataset
---

# OpenVLA 的 LoRA 微调入口:把"教 VLA 干新活"压成了 8 行 PEFT 调用 / OpenVLA's LoRA fine-tune entry point: teaching a VLA a new skill in 8 lines of PEFT

> **一句话 / In one line**: 从冻结的 7B OpenVLA 起步,8 行 `LoraConfig + get_peft_model` 就给所有 Linear 层各塞一个 rank-32 adapter,DDP 一包,RLDS 数据 loader 一接,你就有了一个能用单卡 A6000 训练的、能学新机器人任务的 VLA 微调脚本。 / Starting from a frozen 7B OpenVLA, 8 lines of `LoraConfig + get_peft_model` attach a rank-32 adapter to every Linear, then DDP wraps it and an RLDS dataloader feeds it — and you've got a VLA fine-tune that fits on a single A6000.

## 为什么重要 / Why this matters

VLA(Vision-Language-Action)模型微调的现实痛点是 GPU 内存 —— 把 OpenVLA 这种 7B 模型的所有参数都打开梯度,需要 80GB 显存训练才能呼吸。LoRA 是这个领域的标配:冻结主体,只学每个 Linear 的两个低秩矩阵 A、B,参数量从 7B 降到 30M,内存从 80GB 降到 24GB。但是把 LoRA 套到 VLA 上有几个魔鬼细节:`target_modules` 怎么选?LoRA rank、alpha 怎么定?要不要叠 4-bit 量化?和 DDP/FSDP 怎么协作?OpenVLA 的官方 `finetune.py` 给出了一份"工程上 trade-off 都调好了的标准答案":`target_modules="all-linear"`、`r=32, alpha=min(r, 16), dropout=0`、可选 4-bit、DDP 包装、`find_unused_parameters=True` —— 这些选择背后都是踩过坑的,直接拿来当 nanoVLA 的 fine-tune 模板就行。

VLA fine-tuning's real pain is GPU memory — turning on gradients for all 7B parameters of OpenVLA needs ≥80GB. LoRA is the field standard: freeze the backbone, learn two low-rank matrices A, B per Linear; the trainable count drops from 7B to ~30M and memory from 80GB to 24GB. But wiring LoRA to a VLA has devils in the details: what does `target_modules` choose? What rank, alpha, dropout? Stack 4-bit quant on top? How to play with DDP/FSDP? OpenVLA's official `finetune.py` gives you the production answer: `target_modules="all-linear"`, `r=32, alpha=min(r, 16), dropout=0`, optional 4-bit, DDP wrap, `find_unused_parameters=True` — each choice has a war story behind it. It's the recipe to copy verbatim into your nanoVLA.

## 代码 / The code

`openvla/openvla` — [`vla-scripts/finetune.py`](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/vla-scripts/finetune.py#L113-L220)

```python
@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
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

    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Fine-tuning Dataset
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )
    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        ...
    )
```

## 逐行讲解 / What's happening

1. **`@draccus.wrap()` + `FinetuneConfig`**:
   - 中文: draccus 是个超轻量的 dataclass-CLI 工具,让你能 `python finetune.py --lora_rank=32 --use_quantization=True` 直接覆盖 dataclass 字段。整个脚本的 "配置"只有一个 dataclass,对照之下不用读 50 个 argparse 调用。
   - English: draccus is an ultra-lightweight dataclass-to-CLI bridge — `python finetune.py --lora_rank=32 --use_quantization=True` overrides the dataclass fields directly. The whole script has one config object, no scattered argparse.

2. **`PartialState()` + `distributed_state.local_process_index`**:
   - 中文: 这是 `accelerate` 库提供的"轻 DDP 初始化"。它替你处理"现在我是第几号进程"、"该用哪张卡",一行就把 DDP 启动准备好。
   - English: `accelerate`'s lightweight distributed-state init. Resolves rank and device in one call so you don't have to call `dist.init_process_group` and `torch.cuda.set_device` separately.

3. **`BitsAndBytesConfig(load_in_4bit=True, ..., bnb_4bit_quant_type="nf4")`**:
   - 中文: QLoRA 三件套之一。`nf4` 是 4-bit NormalFloat 量化(权重以 4-bit 存,在 forward 时反量化回 bf16 算),让 7B 模型的常驻显存从 14GB 降到 4GB。但 LoRA adapter 仍然是 bf16 全精度 —— 这是 QLoRA "训练精度不掉、显存爆降"的秘密。
   - English: one third of QLoRA. `nf4` is 4-bit NormalFloat — weights stored at 4 bits, dequantized to bf16 on the fly in forward. 7B resident memory drops from 14GB to ~4GB. The LoRA adapters themselves stay full bf16 precision — that's QLoRA's "memory crashes but accuracy holds" trick.

4. **`AutoConfig.register("openvla", OpenVLAConfig)` 四连**:
   - 中文: 这四行是因为 OpenVLA 是个第三方模型,HF 的 Auto 系列不内置识别。注册之后 `AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b")` 才会把它路由到 `OpenVLAForActionPrediction`。如果你 fine-tune 自己写的 nanoVLA,这一段就是你的"把 nanoVLA 接进 HF 生态"的入口。
   - English: four lines that exist because OpenVLA isn't a built-in HF model. Registering routes `AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b")` to the custom `OpenVLAForActionPrediction`. If you ever wire your own nanoVLA into the HF ecosystem, this is the boilerplate.

5. **`prepare_model_for_kbit_training(vla)` (QLoRA path only)**:
   - 中文: 这个 PEFT 工具函数做四件事:把 LayerNorm 强制升回 fp32(防止 quant 误差累积)、关掉量化层的 grad、把 input embedding 调整为可梯度(因为 4-bit 反量化阻断了梯度路径)、`use_cache=False`。不调它,QLoRA 训练会出现 NaN 或者根本不学。
   - English: this PEFT utility does four things: upcasts LayerNorm to fp32 (otherwise quant error compounds), disables grads on quantized layers, makes input embeddings differentiable (4-bit dequant breaks the gradient path), and sets `use_cache=False`. Skip it and QLoRA training NaNs or stalls.

6. **`target_modules="all-linear"`**:
   - 中文: 这是 OpenVLA 团队踩坑后的结论。早期 LoRA 论文只对 attention 的 q、v 投影插 adapter,但 VLA 的视觉、语言、动作三块跨模态,把所有 Linear(包括 MLP up/gate/down、视觉的 patch embed)都接 adapter 才能学好新机器人任务。代价:可训练参数多一倍。收益:任务成功率高几个 pt。`"all-linear"` 是 PEFT 提供的语义化字符串,会自动遍历所有 `nn.Linear`(但不包含 head)。
   - English: hard-won lesson. The original LoRA paper inserts adapters only on attention q, v. For VLAs — where vision / language / action span three modalities — only inserting adapters everywhere (all MLP up/gate/down, vision patch embed too) gives enough capacity for a new robot task. Cost: ~2× trainable params. Benefit: several points of task-success rate. `"all-linear"` is PEFT's semantic shorthand that walks every `nn.Linear` (excluding heads).

7. **`lora_alpha=min(cfg.lora_rank, 16)`**:
   - 中文: 这是个有意思的调度。LoRA 公式是 `ΔW = (alpha/r) * B @ A`,常规做法是 `alpha = 2r`。OpenVLA 反其道,alpha 上限设到 16:`r=32` 时 alpha=16,scaling=0.5;`r=8` 时 alpha=8,scaling=1.0。原因:VLA 训练比 LLM 微调更容易"被 LoRA 覆盖原能力",小 scaling 让 ΔW 一开始非常温柔,慢慢 ramp up,效果上是显式正则化。
   - English: an unusual choice. The LoRA formula is `ΔW = (alpha/r) · B @ A`; the textbook setting is `alpha = 2r`. OpenVLA caps alpha at 16: at r=32 alpha=16 → scaling 0.5; at r=8 alpha=8 → scaling 1.0. Reason: VLA training is more prone to "LoRA stomping on base capabilities" than language-only fine-tunes; a smaller scaling makes ΔW gentle initially and ramps gradually — implicit regularization.

8. **`init_lora_weights="gaussian"`**:
   - 中文: PEFT 默认用 Kaiming 初始化 A、B=0,保证训练开始时 ΔW=0 不影响 base 模型。"gaussian" 是论文推荐的 A ~ N(0, σ²)、B=0 —— 对 VLA 这种"已经会做大部分动作"的模型,gaussian 收敛更快。
   - English: PEFT defaults to Kaiming for A and zero-init for B, ensuring ΔW=0 at t=0 so the base model is untouched. "gaussian" is the paper recipe: A ~ N(0, σ²), B=0 — converges faster for VLAs that already know most actions and only need refinement.

9. **`DDP(vla, ..., find_unused_parameters=True, gradient_as_bucket_view=True)`**:
   - 中文: 两个关键 flag。`find_unused_parameters=True` 必须打开,因为 LoRA 让大量 base 参数不参与梯度,DDP 默认会因为找不到 grad 而报错。`gradient_as_bucket_view=True` 是 PyTorch 2.x 引入的"用 view 而非拷贝构造 grad bucket",节约一份梯度内存 —— 对 7B 模型来说省 14GB。
   - English: two critical flags. `find_unused_parameters=True` is mandatory — LoRA leaves most base params without grads and DDP otherwise errors on missing grads. `gradient_as_bucket_view=True` (PyTorch 2.x+) constructs grad buckets via views instead of copies, saving 14GB on a 7B model.

10. **`trainable_params = [p for p in vla.parameters() if p.requires_grad]`**:
    - 中文: 关键的一行。`vla.parameters()` 仍然包括 7B 个 base 参数,但 LoRA wrap 之后只有 adapter 参数 `requires_grad=True`。这里过滤一遍,AdamW 只为 30M 参数维护 moment / variance,优化器 state 内存也跟着省。
    - English: critical filter. `vla.parameters()` still yields all 7B params, but only adapter params have `requires_grad=True` after LoRA wrapping. Filtering means AdamW only maintains moments/variances for ~30M params — optimizer state memory drops in lockstep.

## 类比 / The analogy

想象你是一家工厂的总工程师,你有一台造小型机器人的成熟生产线(预训练好的 OpenVLA)。客户来了说"我想要一种新功能 —— 能拧瓶盖的小机器人"。两种方案:(A)全厂停工,把整条产线拆了重建 —— 这是全量微调,贵、慢、还可能把原来会造的所有功能都拆坏了;(B)在每个工位旁边各加一个"可调参的小工具盘"(LoRA adapter),工人想拧瓶盖时就额外用一下小工具盘,造其他东西时小工具盘干脆不动 —— 这是 LoRA。OpenVLA 的脚本就是这套"在每个工位旁加一个小工具盘"的标准蓝图:加多大、加哪些工位、工人和工具盘怎么协同(DDP + grad bucket)、工具盘什么时候开多大力(alpha)—— 都有定好的参数。

You're the chief engineer of a factory with a mature small-robot production line (pretrained OpenVLA). A client wants a new feature: a bottle-cap-twisting variant. Two options. (A) Shut down the whole factory, tear up the line, rebuild — this is full fine-tuning. Expensive, slow, and risky to all the other things the line already made. (B) Add a tunable tool tray next to every workstation (LoRA adapter). Workers use the tool tray when twisting caps; they ignore it for other jobs. That's LoRA. OpenVLA's script is the blueprint for option B — what size tool tray, which workstations get one, how the workers coordinate with the tray (DDP + grad bucket), how much force the tray applies initially (alpha) — every choice has been tuned.

## 在 nanoVLA / nanoWAM 中的位置 / Where this lives in your nano-VLA

中文:**这就是 nanoVLA 课程里的 `fine-tune-lora` 组件**(curriculum graph 里依赖 `vlm-backbone-wiring`)。在你自己从头搭的 nanoVLA 里,这个脚本对应"用户拿到你训练好的 base VLA 之后,要在新机器人数据上让它 work"的入口点。输入:一个已经会基本 vision-language-action 推理的 nanoVLA(已经把 vision-encoder + modality-projector + vlm-backbone + action-head 接好的产物)+ 一份 RLDS 格式的新机器人数据。输出:一个 LoRA adapter 检查点。下游:`inference-loop` 加载这个 adapter,实时跑你的新机器人。如果省掉这一层,你的 VLA 就是个"只会原始训练任务的死模型"。生产实现还需要补:eval callback、checkpoint resume、cosine LR schedule、gradient clipping、wandb metrics(代码片段后面就是)、多机 NCCL 配置 —— 但骨架就是上面这 8 行 PEFT 调用。

English: **this is the nanoVLA `fine-tune-lora` slot in the curriculum graph** (depends on `vlm-backbone-wiring`). In your from-scratch nanoVLA, this script corresponds to the user's entry point after you've shipped a base VLA: "I have your pretrained nanoVLA + a new robot dataset; make it work." Inputs: a base nanoVLA (which already has vision-encoder + modality-projector + vlm-backbone + action-head wired) and an RLDS dataset. Output: a LoRA adapter checkpoint. Downstream: the `inference-loop` loads this adapter and runs your robot. Omit this layer and your VLA is a dead model that only does its pretraining task. Production needs to add: eval callbacks, checkpoint resume, cosine LR schedule, gradient clipping, wandb metrics (the script continues with these), multi-node NCCL setup — but the skeleton is those 8 lines of PEFT.

## 自己跑一遍 / Try it yourself

```python
# Minimal nanoVLA-style LoRA fine-tune scaffold.
# Replace `tiny_vla` with your own from-scratch VLA module.
import torch, torch.nn as nn
from peft import LoraConfig, get_peft_model

class TinyVLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = nn.Linear(1024, 768)          # pretend vision encoder
        self.lm     = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768))
        self.head   = nn.Linear(768, 7)              # 7-dim action
    def forward(self, img, label):
        z = self.lm(self.vision(img))
        a = self.head(z)
        return ((a - label) ** 2).mean()

base = TinyVLA()
lora_cfg = LoraConfig(
    r=8, lora_alpha=8, lora_dropout=0.0,
    target_modules="all-linear", init_lora_weights="gaussian",
)
peft_model = get_peft_model(base, lora_cfg)
peft_model.print_trainable_parameters()

opt = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=5e-4)
img, lab = torch.randn(4, 1024), torch.randn(4, 7)
for step in range(20):
    loss = peft_model(img, lab)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 5 == 0: print(f"step {step:2d}  loss {loss.item():.4f}")
```

运行 / Run with:
```bash
pip install torch peft
python try.py
```

预期输出 / Expected output:
```
trainable params: ~8,000  ||  all params: ~1.7M  ||  trainable%: ~0.47
step  0  loss 1.42
step  5  loss 0.95
step 10  loss 0.61
step 15  loss 0.41
```

中文:注意 trainable% < 1% —— 这就是 LoRA 的优势。loss 仍然在掉,因为 adapter 学到了 task 信号。把 `r=8` 换成 `r=2` 试试,你会看到 trainable% 还会再降一半,loss 收敛会慢些 —— 这就是 rank 选择的直观影响。

English: trainable% < 1% — that's LoRA's advantage. Loss still drops because the adapters carry the task signal. Try r=2: trainable% halves again and loss converges slower — that's the rank-selection trade-off made concrete.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **lerobot 的 SmolVLA fine-tune** / **lerobot's SmolVLA fine-tune**:同样的 `get_peft_model` + `target_modules="all-linear"` 套路,但用 LeRobot 的 hub-style 数据集 API。 / Same `get_peft_model + target_modules="all-linear"` pattern, but loads via LeRobot's hub-style dataset API.
- **OpenVLA-OFT** / **OpenVLA-OFT**:同 repo 家族的"非自回归"动作头 fine-tune;LoRA 配置基本继承自这个脚本,只是 action head 换了。 / Same family, but with a non-autoregressive action head; LoRA config inherited almost verbatim, only the action head differs.
- **Physical-Intelligence/openpi 的 pi0 fine-tune** / **openpi's pi0 fine-tune (Physical-Intelligence)**:JAX 实现,但 LoRA 设置 `r=32, alpha=16, target_modules="all-linear"` 完全相同 —— 收敛后的成功率甚至和 PyTorch 版可对齐。 / JAX implementation but identical LoRA settings; final task-success aligns with the PyTorch version.
- **NVIDIA Isaac-GR00T 的 LoRA fine-tune** / **NVIDIA Isaac-GR00T LoRA fine-tune**:DiT action head 上做 LoRA 时,他们只对 cross-attention 投影插 adapter(rank=64),而对 MLP 全冻 —— 跟 OpenVLA 的 "all-linear" 是两种哲学。 / On the DiT action head, GR00T LoRAs only the cross-attention projections (rank=64) and freezes MLPs entirely — opposite philosophy from OpenVLA's "all-linear".

## 注意事项 / Caveats / when it breaks

- **`find_unused_parameters=True` 性能税** / **`find_unused_parameters=True` perf tax**:这个 flag 让 DDP 在每个 step 多扫一遍参数表确定哪些有 grad —— 大模型上单 step 多 5-10%。改用 FSDP 可以省掉这点,但要重写 LoRA + FSDP 兼容代码。 / This flag makes DDP scan params each step to find missing grads — costs 5-10% per step on large models. FSDP avoids it but requires rewriting LoRA + FSDP wrapper.
- **`all-linear` 包含 LayerNorm 的 affine?** / **does `all-linear` include LayerNorm affines?**:不,`all-linear` 只匹配 `nn.Linear`。LayerNorm 的 weight / bias 不会被 LoRA 化(它们也不该被 —— 量很小,效果差)。 / No, `all-linear` matches only `nn.Linear`. LayerNorm affines are not LoRAed (and shouldn't be — they're tiny and rarely help).
- **QLoRA + DDP 在多卡上吃不消大 batch** / **QLoRA + DDP doesn't love big batches across many GPUs**:bnb 的 dequant 路径在多卡上有 host-side bottleneck。推荐:8 卡以内用 DDP,8 卡以上换 FSDP + bitsandbytes 4-bit 量化的 future API。 / bnb dequant has a host-side bottleneck that bites past ~8 GPUs. Stick to DDP for ≤8 GPUs; use FSDP + the upcoming bnb 4-bit API beyond.
- **alpha 调小一定降效果?** / **does smaller alpha always reduce effect?**:不必然 —— alpha 越小,等价于 ΔW 步长越小,需要训练步数越多。OpenVLA 在 200k step 上 alpha=16 收敛;短训练(< 20k)需要把 alpha 调到 32。 / Not necessarily — smaller alpha means smaller ΔW step, requiring more train steps. OpenVLA converges at alpha=16 in 200k steps; short runs (<20k) want alpha=32.

## 延伸阅读 / Further reading

- [OpenVLA paper — Section "Fine-tuning OpenVLA"](https://arxiv.org/abs/2406.09246)
- [QLoRA paper — full recipe for nf4 + LoRA](https://arxiv.org/abs/2305.14314)
- [PEFT docs — LoraConfig reference](https://huggingface.co/docs/peft/main/en/package_reference/lora)
- [Sebastian Raschka — "Practical Tips for Finetuning LLMs Using LoRA"](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)
