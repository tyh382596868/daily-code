---
date: 2026-07-22
topic: huggingface
source: huggingface
repo: huggingface/trl
file: trl/trainer/grpo_trainer.py
permalink: https://github.com/huggingface/trl/blob/8db673e40d81a2851b5f57931ce81c68a3849aa3/trl/trainer/grpo_trainer.py#L1034-L1109
difficulty: intermediate
read_time: ~8 min
tags: [code-of-the-day, huggingface, trl, grpo, vllm]
---

# TRL GRPO：训练器里把生成后端也变成可插拔模块 / TRL GRPO: Make the Generation Backend Pluggable Inside the Trainer

> **一句话 / In one line**: GRPO 训练既要算 loss，也要不断生成 completion；TRL 把 vLLM 后端封装成 `VLLMGeneration`，把采样参数和分布式约束集中在初始化阶段。 / GRPO training computes loss but also keeps generating completions; TRL wraps the vLLM backend as `VLLMGeneration`, centralizing sampling parameters and distributed constraints during initialization.

## 为什么重要 / Why this matters

RLHF/RLAIF 训练器不只是一个普通 supervised trainer。它每隔几步要调用当前策略生成文本，再用 reward 和 advantage 更新模型。如果生成路径和训练路径互相散落，batch size、tensor parallel、logprob、sleep mode 很容易不一致。这里的设计是：把生成后端当成 trainer 的一个明确子系统。

An RLHF/RLAIF trainer is not just a supervised trainer. It repeatedly asks the current policy to generate text, then updates from rewards and advantages. If generation and training paths are scattered, batch size, tensor parallelism, logprobs, and sleep mode can drift apart. The design here treats generation as an explicit trainer subsystem.

## 代码 / The code

`huggingface/trl` — [`trl/trainer/grpo_trainer.py`](https://github.com/huggingface/trl/blob/8db673e40d81a2851b5f57931ce81c68a3849aa3/trl/trainer/grpo_trainer.py#L1034-L1109)

```python
# Simplified teaching slice, preserving the initialization shape.
if self.use_vllm:
    self.vllm_generation = VLLMGeneration(
        model=self.model,
        accelerator=self.accelerator,
        processing_class=self.processing_class,
        mode=args.vllm_mode,
        server_base_url=args.vllm_server_base_url,
        tensor_parallel_size=args.vllm_tensor_parallel_size,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_length=args.vllm_max_model_length,
        max_num_seqs=(
            args.per_device_train_batch_size
            * args.vllm_tensor_parallel_size
            * args.steps_per_generation
        ),
        temperature=self.temperature,
        top_p=self.top_p,
        top_k=self.top_k,
        min_p=self.min_p,
        max_completion_length=self.max_completion_length,
        logprobs=0,
        generation_kwargs=args.generation_kwargs,
    )
    self._last_loaded_step = -1
else:
    generation_kwargs = {
        "max_new_tokens": self.max_completion_length,
        "do_sample": True,
        "temperature": self.temperature,
        "top_p": self.top_p,
    }
```

## 逐行讲解 / What's happening

1. **`use_vllm` 分出生成后端 / `use_vllm` selects the generation backend**: 中文: 同一个 trainer 可以走 vLLM，也可以走 Transformers 原生 `generate`。 English: the same trainer can use vLLM or fall back to native Transformers generation.
2. **模型和 accelerator 一起传入 / Model and accelerator travel together**: 中文: 生成后端需要知道当前训练模型和分布式环境。 English: the generation backend needs both the current training model and the distributed environment.
3. **`max_num_seqs` 从训练形状推出来 / `max_num_seqs` is derived from training shape**: 中文: per-device batch、tensor parallel、每次生成步数一起决定 vLLM 并发上限。 English: per-device batch size, tensor parallelism, and generation cadence jointly determine vLLM concurrency.
4. **采样参数集中传递 / Sampling parameters are centralized**: 中文: temperature、top-p、top-k、min-p 不在调用点到处重复。 English: temperature, top-p, top-k, and min-p are not repeated across call sites.
5. **`_last_loaded_step` 避免重复加载 / `_last_loaded_step` avoids useless reloads**: 中文: 梯度累积期间没必要每个 microstep 都把权重塞进生成引擎。 English: during gradient accumulation, weights do not need to be reloaded into the generation engine at every microstep.

## 类比 / The analogy

像餐厅把外卖窗口独立出来。厨房仍然做菜，但外卖窗口有自己的排队上限、打包规则和出餐节奏；经理只在菜单变化时同步一次。

It is like a restaurant splitting out a takeout counter. The kitchen still cooks, but the counter has its own queue limit, packaging rules, and dispatch rhythm; the manager syncs it only when the menu changes.

## 自己跑一遍 / Try it yourself

```python
def max_num_seqs(per_device, tensor_parallel, steps_per_generation):
    return per_device * tensor_parallel * steps_per_generation

cfg = dict(per_device=4, tensor_parallel=2, steps_per_generation=3)
print(max_num_seqs(**cfg))
print({"temperature": 0.8, "top_p": 0.95, "logprobs": 0})
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
24
{'temperature': 0.8, 'top_p': 0.95, 'logprobs': 0}
```

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **TRL online trainers** / **TRL online trainers**: 中文: 在线 DPO、GRPO、RLOO 都需要把“采样”和“训练”放进同一控制循环。 / English: online DPO, GRPO, and RLOO all need sampling and training inside one control loop.
- **vLLM serving** / **vLLM serving**: 中文: serving 侧也把并发序列数、最大长度和采样参数集中配置。 / English: serving also centralizes concurrent sequence count, maximum length, and sampling parameters.
- **Accelerate/FSDP** / **Accelerate/FSDP**: 中文: 分布式训练器常把设备策略在构造阶段固定，减少 step 内分支。 / English: distributed trainers often fix device strategy at construction time to reduce per-step branching.

## 注意事项 / Caveats / when it breaks

- **生成权重要同步 / Generation weights must be synced**: 训练模型更新后，生成后端如果没加载新权重，reward 会来自旧策略。 / If the generation backend is not refreshed after model updates, rewards come from an old policy.
- **并发不是越大越好 / More concurrency is not always better**: `max_num_seqs` 太大可能抢走训练显存。 / Too large a `max_num_seqs` can steal memory from training.
- **logprobs 配置要和 loss 匹配 / Logprob settings must match the loss**: GRPO 只需要生成 token 的 logprob 修正，别无脑打开大日志。 / GRPO needs generated-token logprobs for correction; do not enable heavy logging blindly.

## 延伸阅读 / Further reading

- [TRL GRPO trainer](https://github.com/huggingface/trl/blob/8db673e40d81a2851b5f57931ce81c68a3849aa3/trl/trainer/grpo_trainer.py#L1034-L1109)

