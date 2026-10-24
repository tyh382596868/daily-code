---
date: 2026-10-24
topic: robotics
source: tracked
repo: huggingface/lerobot
file: src/lerobot/policies/pi0/processor_pi0.py
permalink: https://github.com/huggingface/lerobot/blob/d3793b7e36522321c1c455f0dc4826db3793dceb/src/lerobot/policies/pi0/processor_pi0.py#L95-L157
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, robotics, policy-processor]
---

# LeRobot PI0 处理顺序：先相对化，再归一化 / LeRobot PI0 Processor Order: Make Actions Relative Before Normalizing

> **一句话 / In one line**: PI0 的输入、输出处理链把几何变换放在归一化的两侧，保持模型内外的动作语义一致。 / PI0 places geometric conversion on opposite sides of normalization so action semantics remain consistent across the model boundary.

## 为什么重要 / Why this matters

机器人 policy 的失败常常不是网络结构问题，而是训练和部署时动作坐标系不同。这里把输入固定为 `raw → relative → normalize`，并严格反向还原输出。

Many robot-policy failures are coordinate-contract failures, not architecture failures. This pipeline fixes the input as `raw → relative → normalize` and reverses that contract on output.

## 代码 / The code

`huggingface/lerobot` — [`processor_pi0.py`](https://github.com/huggingface/lerobot/blob/d3793b7e36522321c1c455f0dc4826db3793dceb/src/lerobot/policies/pi0/processor_pi0.py#L127-L157)

```python
relative_step = RelativeActionsProcessorStep(
    enabled=config.use_relative_actions,
    exclude_joints=getattr(config, "relative_exclude_joints", []),
    action_names=getattr(config, "action_feature_names", None),
)

input_steps = [
    steps.rename_observations, steps.add_batch_dim, Pi0NewLineProcessor(),
    TokenizerProcessorStep(tokenizer_name=config.text_tokenizer_name,
                           max_length=config.tokenizer_max_length,
                           padding_side="right", padding="max_length"),
    steps.to_device, relative_step, steps.normalize,
]
output_steps = [steps.unnormalize,
                AbsoluteActionsProcessorStep(enabled=config.use_relative_actions,
                                             relative_step=relative_step),
                steps.to_cpu]
```

## 逐行讲解 / What's happening

`relative_step` 保留为同一个对象，因此输出端能够使用与输入端相同的关节排除规则。输入先整理 observation、tokenize 任务文字，再把相对动作送进统计归一化；输出按完全相反的顺序回到可执行的绝对动作。

`relative_step` is shared, so the output side uses the same joint-exclusion rule as the input side. Observations and language are prepared first, relative actions are normalized, and predictions are inverted back into executable absolute actions.

## 类比 / The analogy

像国际包裹：先把本地地址改写成统一格式，经过物流网络后再翻回目的地的本地地址。中间系统不该猜测原始坐标系。

It is like international shipping: convert a local address into a common format, then convert it back at the destination. The transport network should not guess the original convention.

## 自己跑一遍 / Try it yourself

```python
import torch
base = torch.tensor([10., 20.]); absolute = torch.tensor([11., 18.])
relative = absolute - base
mean, std = relative.mean(), relative.std().clamp_min(1e-6)
predicted_absolute = ((relative - mean) / std * std + mean) + base
print(predicted_absolute)
```

运行 / Run with: `pip install torch && python try.py`

预期输出 / Expected output: `tensor([11., 18.])`。关键不是标准化公式，而是它必须夹在可逆的坐标变换之间。

Expected output: `tensor([11., 18.])`. The important part is not the normalization formula; it is placing it inside an invertible coordinate conversion.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **openpi transforms** / **openpi transforms**: 同样把动作变换和数据统计封装为可组合步骤。 / It likewise makes action conversion and data statistics composable steps.

## 注意事项 / Caveats / when it breaks

- **训练统计必须匹配坐标系** / **Statistics must match the coordinate frame**: 绝对动作统计不能拿来归一化相对动作。 / Absolute-action statistics cannot normalize relative actions.

## 延伸阅读 / Further reading

- [LeRobot PI0 processor source](https://github.com/huggingface/lerobot/blob/d3793b7e36522321c1c455f0dc4826db3793dceb/src/lerobot/policies/pi0/processor_pi0.py#L95-L157)
