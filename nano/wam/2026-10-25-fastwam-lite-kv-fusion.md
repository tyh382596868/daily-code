---
date: 2026-10-25
topic: wam
source: wam
repo: huggingface/lerobot
file: src/lerobot/policies/fastwam/wan/modular.py
permalink: https://github.com/huggingface/lerobot/blob/d3793b7e36522321c1c455f0dc4826db3793dceb/src/lerobot/policies/fastwam/wan/modular.py#L628-L706
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, attention, kv-cache]
build_role: sampler-inference (advanced variant, per-head cross-layer KV fusion)
---

# FastWAM Lite KV 融合：每个动作头选择视频层 / FastWAM Lite KV Fusion: Let Each Action Head Choose Video Layers

> **一句话 / In one line**: LiteKVFusion 为每个动作层和 attention head 学一个视频层软选择，再融合缓存的 K/V。 / LiteKVFusion learns a soft selection over video layers for every action layer and attention head, then fuses cached K/V.

## 为什么重要 / Why this matters

世界模型的视频 backbone 很深，动作 expert 若逐层重算或只固定读取最后一层，都会浪费已有的视频表示。这个模块只增加 `action_layers × heads × video_layers` 个 logits，却允许动作查询从不同深度取到合适的缓存。

A world model's video backbone is deep. An action expert that recomputes every layer, or always reads only the last layer, wastes available video representations. This module adds only `action_layers × heads × video_layers` logits while allowing action queries to use cached context from the right depth.

## 代码 / The code

`huggingface/lerobot` — [`wan/modular.py`](https://github.com/huggingface/lerobot/blob/d3793b7e36522321c1c455f0dc4826db3793dceb/src/lerobot/policies/fastwam/wan/modular.py#L628-L706)

```python
self.logits = nn.Parameter(torch.zeros(num_action_layers, num_heads, num_video_layers))

def mixing_weights(self, action_layer_idx):
    return functional.softmax(self.logits[action_layer_idx], dim=-1, dtype=torch.float32)

beta = self.mixing_weights(action_layer_idx)  # [H, L_v]
k_stack = torch.stack([entry["k"] for entry in video_kv_cache], dim=0)
v_stack = torch.stack([entry["v"] for entry in video_kv_cache], dim=0)
beta = beta.to(dtype=k_stack.dtype).permute(1, 0).view(self.num_video_layers, 1, 1, num_heads, 1)
k_fused = (k_stack.view(self.num_video_layers, batch_size, seq_len, num_heads, head_dim) * beta).sum(dim=0)
v_fused = (v_stack.view(self.num_video_layers, batch_size, seq_len, num_heads, head_dim) * beta).sum(dim=0)
return k_fused.flatten(2), v_fused.flatten(2)
```

## 逐行讲解 / What's happening

1. **第 659 行 / Line 659**：零初始化使 softmax 起点是均匀分布，新增融合不会一开始偏向任意视频层。 / Zero initialization makes the initial softmax uniform, so new fusion does not initially favor a video layer.
2. **第 669 行 / Line 669**：归一化在 float32 中进行，避免低精度参数储存损坏概率分布。 / Normalization runs in float32 so low-precision parameter storage does not damage the distribution.
3. **第 691-706 行 / Lines 691-706**：先把层维堆叠，再按 head 广播权重并求和；K 和 V 共享同一权重，保证它们是配对的记忆。 / Layer caches are stacked, head weights are broadcast and summed, and K/V share weights so they remain paired memory.

## 类比 / The analogy

像每位厨师在不同菜谱版本中挑选自己需要的步骤，而不是整支厨房只能读最后一版。 / It is like each chef selecting useful steps from different revisions of a recipe rather than forcing the whole kitchen to read only the latest revision.

## 在 nanoWAM 中的位置 / Where this lives in your nanoWAM

这是 `sampler-inference` 的跨层缓存进阶实现。上游视频 DiT 先预填每层 K/V；动作 DiT 的每个 layer/head 输入这些缓存，输出一个融合后的 `(K, V)` 给 attention。最小 nanoWAM 可以先固定用最后一层；生产版再加入 cache 生命周期、分布式并行与数值监控。

This is an advanced cross-layer-cache variant of `sampler-inference`. The video DiT prefills K/V at every layer; each action-DiT layer/head consumes those caches and emits one fused `(K, V)` for attention. A minimal nanoWAM can start with the final layer only, while production needs cache lifecycle management, distributed parallelism, and numerical monitoring.

## 自己跑一遍 / Try it yourself

```python
import torch

layers = torch.tensor([[1., 10.], [3., 30.]])  # [layer, head]
weights = torch.softmax(torch.zeros(2, 2), dim=-1)
print((layers * weights.T).sum(0))
```

运行 / Run with: `pip install torch && python try.py`

预期输出 / Expected output: `tensor([ 2., 20.])`。

均匀起点就是逐层平均；训练会把每个 head 的重心推向有用层。 / The uniform start is a layer average; training moves each head toward useful layers.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **ELMo 标量混合 / ELMo scalar mix**：学习组合不同 Transformer 层。 / It learns a combination of Transformer layers.
- **跨层 feature pyramid / Feature pyramids**：任务头从多个视觉尺度取信息。 / Task heads draw information from multiple visual scales.

## 注意事项 / Caveats / when it breaks

- **层数必须一致 / Layer count must agree**：缓存条目数和配置不同会改变语义，代码会拒绝它。 / A different cache entry count changes semantics, so the code rejects it.
- **显存不是免费 / Memory is not free**：保留所有层 K/V 换来了选择能力，也增加了显存。 / Retaining every layer's K/V buys flexibility at a memory cost.

## 延伸阅读 / Further reading

- [LiteKVFusion source](https://github.com/huggingface/lerobot/blob/d3793b7e36522321c1c455f0dc4826db3793dceb/src/lerobot/policies/fastwam/wan/modular.py)
