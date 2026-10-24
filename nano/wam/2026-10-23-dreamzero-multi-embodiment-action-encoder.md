---
date: 2026-10-23
topic: wam
source: wam
repo: dreamzero0/dreamzero
file: groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py
permalink: https://github.com/dreamzero0/dreamzero/blob/ab790c198fbce33503358efbbd4187ce9a89adf3/groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py#L31-L90
difficulty: advanced
read_time: ~10 min
tags: [code-of-the-day, wam, action-conditioning]
build_role: action-conditioning
---

# DreamZero 多 embodiment 动作编码：同一模型，按机器人选权重 / DreamZero Multi-Embodiment Action Encoding: Pick Weights Per Robot

> **一句话 / In one line**: DreamZero 让每个 embodiment 选择一套线性层，再把动作与时间嵌入融合为条件 token。 / DreamZero selects linear layers per embodiment, then fuses action and time embeddings into conditioning tokens.

## 为什么重要 / Why this matters

不同机器人动作维度和语义不同；共享世界模型时，条件编码器必须知道动作来自哪种身体。

Robot actions differ in width and meaning. A shared world model needs its conditioning encoder to know which body produced the action.

## 代码 / The code

`dreamzero0/dreamzero` — [`wan_video_dit_action_casual_chunk.py`](https://github.com/dreamzero0/dreamzero/blob/ab790c198fbce33503358efbbd4187ce9a89adf3/groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py#L31-L90)

```python
class CategorySpecificLinear(nn.Module):
    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)

def forward(self, actions, timesteps, cat_ids):
    a_emb = self.W1(actions, cat_ids)
    tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)
    x = torch.cat([a_emb, tau_emb], dim=-1)
    return self.W3(swish(self.W2(x, cat_ids)), cat_ids)
```

## 逐行讲解 / What's happening

索引 `self.W[cat_ids]` 取出 batch 中每个样本对应的矩阵；`bmm` 同时完成每个样本自己的投影。动作和 diffusion timestep 拼接后再经 MLP，成为视频 DiT 可消费的 token。

`self.W[cat_ids]` gathers a per-sample matrix; `bmm` applies those individual projections together. After concatenating actions with the diffusion timestep, an MLP creates tokens the video DiT can consume.

## 类比 / The analogy

像多语种翻译台：内容相同，但先按说话者语言选字典，才能翻成统一的会议语言。

It is a multilingual interpreter: choose the speaker's dictionary first, then translate into the common meeting language.

## 在 nanoWAM 中的位置 / Where this lives in your nano-WAM

这是已覆盖 `action-conditioning` 的另一种实现：输入 `(action, timestep, embodiment_id)`，输出 action tokens；下游为 DiT cross-attention 或拼接序列。最小版可只用共享 MLP，生产版需处理不同 action schema、mask 和归一化。

This is another implementation of covered `action-conditioning`: `(action, timestep, embodiment_id)` in, action tokens out, consumed by DiT cross-attention or a concatenated sequence. A minimal build can use one shared MLP; production needs schemas, masks, and normalization.

## 自己跑一遍 / Try it yourself

```python
import torch
W = torch.randn(2, 3, 4); x = torch.randn(2, 5, 3); ids = torch.tensor([0, 1])
print(torch.bmm(x, W[ids]).shape)
```

运行 / Run with: `pip install torch && python try.py`

预期输出 / Expected output: `torch.Size([2, 5, 4])`。

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **DreamZero action encoder** / **DreamZero action encoder**: 用 category-specific 投影处理多 embodiment。 / It uses category-specific projections for multiple embodiments.

## 注意事项 / Caveats / when it breaks

- 每类独立矩阵会随 embodiment 数线性增大；数据不足时易过拟合。 / Per-category matrices grow linearly with embodiment count and can overfit sparse data.

## 延伸阅读 / Further reading

- [DreamZero source](https://github.com/dreamzero0/dreamzero/blob/ab790c198fbce33503358efbbd4187ce9a89adf3/groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py)
