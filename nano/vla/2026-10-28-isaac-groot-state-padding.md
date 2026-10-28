---
date: 2026-10-28
topic: vla
source: vla
repo: NVIDIA/Isaac-GR00T
file: gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py
permalink: https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py#L2726-L2778
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, vla, embodiment]
build_role: vision-encoder
---

# Isaac GR00T state padding：多 embodiment 先归一，再补到同一宽度 / Isaac GR00T State Padding: Normalize Then Pad Every Embodiment

> **一句话 / In one line**: GR00T 先按机器人类型归一化 state，再在尾部补零至共享的 `max_state_dim`。 / GR00T normalizes state per embodiment, then right-pads it to a shared `max_state_dim`.

## 为什么重要 / Why this matters

不同机器人 state 维度不同；主干网络需要固定张量形状，预处理层则必须保留每种 embodiment 的统计语义。

Robot states have different widths. The backbone needs a fixed tensor shape, while preprocessing must preserve each embodiment’s statistics.

## 代码 / The code

`NVIDIA/Isaac-GR00T` — [`processing_gr00t_n1d7.py`](https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py#L2726-L2778)

```python
norm_state_dict = self.state_action_processor.apply_state(
    state=state_data, embodiment_tag=embodiment_tag.value
)
normalized_states = torch.cat(
    [torch.from_numpy(norm_state_dict[key]) for key in state_keys], dim=-1
)
padding_shape = (*normalized_states.shape[:-1],
                 self.max_state_dim - normalized_states.shape[-1])
normalized_states = torch.cat([normalized_states, torch.zeros(padding_shape)], dim=-1)
```

## 逐行讲解 / What's happening

归一化器按 embodiment 选统计量；各 state 字段拼接后，只在最后一维补零。固定宽度令 batch 与模型简单，真实维度信息应由 mask 或配置持有。

The normalizer selects per-embodiment statistics. After concatenation, only the final dimension is padded. Fixed width simplifies batches and the real width must remain available via mask or config.

## 在 nanoVLA 中的位置 / Where this lives in your nanoVLA

这是 observation encoder 的入口契约：`state → normalize → pad + mask → state token`，让一套 VLA 能接多种机器人。

This is the observation-encoder contract: `state → normalize → pad + mask → state token`, enabling one VLA across robots.

## 自己跑一遍 / Try it yourself

```python
import torch
state = torch.tensor([[0.2, -0.1]])
print(torch.cat([state, torch.zeros(1, 2)], dim=-1))
```

运行 / Run with `pip install torch && python try.py`。输出 / Expected output: `tensor([[ 0.2000, -0.1000, 0., 0.]])`。

## 注意事项 / Caveats / when it breaks

- **mask 不可省 / Keep the mask**：零可能是真实值；没有有效维度 mask，模型无法区分 padding。 / Zero can be valid data, so retain a validity mask.
