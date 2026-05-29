---
date: 2026-05-29
topic: robotics
source: tracked
repo: NVIDIA/Isaac-GR00T
file: gr00t/model/modules/embodiment_conditioned_mlp.py
permalink: https://github.com/NVIDIA/Isaac-GR00T/blob/626af89d3e914ec92eab5323e23b9ed44a7b26c8/gr00t/model/modules/embodiment_conditioned_mlp.py#L59-L79
difficulty: beginner
read_time: ~9 min
tags: [code-of-the-day, robotics, multi-embodiment, bmm]
---

# 一个 Linear 层装下所有机器人 / One Linear layer for every robot body

> **一句话 / In one line**: GR00T 用一个 `(num_embodiments, in_dim, out_dim)` 的权重张量 + 一次 `bmm` 就让同一个网络服务 N 种本体，每个 batch 元素自己挑自己那张权重。 / GR00T fits an arbitrary number of robot embodiments into a single Linear layer by stacking weights into a 3-D tensor and dispatching each batch element through `torch.bmm` to its own slice.

## 为什么重要 / Why this matters

GR00T 想让同一个基座模型同时学会人形机器人、双臂机器人、单臂夹爪 —— 它们的关节数、动作维度、运动学全都不一样。最朴素的做法要么每种本体训练一个 head，要么把 MoE 之类的大件搬出来。这段 20 行代码给了第三种选择：把"per-embodiment weight"直接做成一个三维参数，前向时用 `cat_ids` 当索引，最后用 `bmm` 一次性算完。这是多本体学习里最小可工作的单元，也是后面 `CategorySpecificMLP` 和 `MultiEmbodimentActionEncoder` 的积木。

NVIDIA's GR00T trains one foundation policy across humanoids, bimanual arms, single-arm grippers — bodies with totally different joint counts and dynamics. The naive options are either one head per embodiment or some heavy MoE machinery. These 20 lines pick a third path: stack the per-embodiment weights into a single 3-D parameter, use the embodiment id as an index at forward time, and let `torch.bmm` do the dispatch in a single fused matmul. It is the smallest unit of multi-embodiment learning, and the rest of the multi-embodiment encoder is just two of these stacked.

## 代码 / The code

`NVIDIA/Isaac-GR00T` — [`gr00t/model/modules/embodiment_conditioned_mlp.py`](https://github.com/NVIDIA/Isaac-GR00T/blob/626af89d3e914ec92eab5323e23b9ed44a7b26c8/gr00t/model/modules/embodiment_conditioned_mlp.py#L59-L79)

```python
class CategorySpecificLinear(nn.Module):
    """Linear layer with category-specific weights and biases for multi-embodiment support."""

    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        """
        Args:
            x: [B, T, input_dim] input tensor
            cat_ids: [B] category/embodiment IDs
        Returns:
            [B, T, hidden_dim] output tensor
        """
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)
```

## 逐行讲解 / What's happening

1. **第 66 行 / Line 66 (`self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))`)**:
   - 中文：权重不再是 2D 的 `[in, out]`，而是直接堆成 3D `[num_embodiments, in, out]`。每一片 `W[k]` 就是第 k 种本体的专属 Linear。`0.02` 是标准的小方差初始化。
   - English: instead of a 2-D `[in, out]` matrix, the parameter is a 3-D tensor with one slice per embodiment. `W[k]` is the entire Linear layer for embodiment `k`. `0.02` is the small-variance init you would use for any GPT-style Linear.
2. **第 67 行 / Line 67 (`self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))`)**:
   - 中文：偏置同样按本体分组，`b[k]` 是第 k 种本体的偏置向量。
   - English: bias is sharded the same way — `b[k]` is embodiment `k`'s bias.
3. **第 77 行 / Line 77 (`selected_W = self.W[cat_ids]`)**:
   - 中文：fancy indexing。`cat_ids` 是 `[B]`，`self.W` 是 `[K, in, out]`，索引一次得到 `[B, in, out]` —— 每个 batch 元素都拿到了"属于自己"的那张权重。
   - English: fancy indexing. `cat_ids` has shape `[B]`, `self.W` has shape `[K, in, out]`, the result is `[B, in, out]` — every batch element now carries its own private weight matrix.
4. **第 79 行 / Line 79 (`return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)`)**:
   - 中文：`bmm` 是 batched matmul：`[B, T, in] @ [B, in, out] -> [B, T, out]`。一句话把 B 个不同的 Linear 全跑完，不用任何 Python for 循环。偏置 `[B, out]` 用 `unsqueeze(1)` 变成 `[B, 1, out]` 后广播到 T 维。
   - English: `bmm` is batched matmul: `[B, T, in] @ [B, in, out] -> [B, T, out]`. It runs B different Linear layers in one fused kernel, no Python loop. The bias `[B, out]` is unsqueezed to `[B, 1, out]` and broadcast across the time dim.

## 类比 / The analogy

像是一个工具腰带：你不需要为每种家务买一个全新的工具箱，腰带上挂着 N 个工具袋，做哪种家务就抽哪个袋子。`cat_ids` 是工种标签，`bmm` 是同时让 B 个工人各自抽出袋子干活。

Think of a tool belt. You do not buy a new toolbox for each chore — your belt carries N pouches, and based on the chore label you reach into the right one. `cat_ids` are the chore labels for the whole crew, and `bmm` is the entire crew reaching into the right pouch and starting work in lockstep.

## 自己跑一遍 / Try it yourself

```python
# pip install torch
import torch
import torch.nn as nn

class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, in_dim, out_dim):
        super().__init__()
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, in_dim, out_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, out_dim))
    def forward(self, x, cat_ids):
        return torch.bmm(x, self.W[cat_ids]) + self.b[cat_ids].unsqueeze(1)

layer = CategorySpecificLinear(num_categories=3, in_dim=4, out_dim=2)
# pretend each row uses a different "embodiment"
x = torch.randn(3, 5, 4)              # B=3 episodes, T=5 steps, in_dim=4
cat = torch.tensor([0, 2, 1])         # robot A, robot C, robot B
out = layer(x, cat)
print(out.shape)                      # torch.Size([3, 5, 2])

# Sanity check: row 0 of bmm must equal a plain Linear on W[0]
ref = x[0] @ layer.W[0] + layer.b[0]
print(torch.allclose(out[0], ref))    # True
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
torch.Size([3, 5, 2])
True
```

中文：每一行用了不同的权重，但它们是在同一次 `bmm` 里一起算完的 —— 没有 Python for 循环，GPU 满载。

English: each batch row uses a different weight, yet all three are computed in one `bmm` call — no Python loop, GPU stays saturated.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **MoE expert dispatch** / **MoE expert dispatch**: 中文 — Mixture-of-Experts 里"每个 token 选一个专家"本质上就是 `expert_id` 索引到 `[E, in, out]` 的权重栈，再 `bmm`/`einsum`。 / English — selecting one expert per token is the same trick: index a `[num_experts, in, out]` weight stack and run a batched matmul.
- **Conditional computation in Diffusion Transformers** / **Conditional computation in DiT**: 中文 — class-conditional DiT 里有时候也会出现 per-class affine 参数，做法一致。 / English — class-conditional DiTs sometimes use per-class affine params built the same way.
- **HyperNetworks** / **HyperNetworks**: 中文 — 极端情况下"那张权重"是另一个网络现场生成的，但消费端的接口仍然是 fancy index + bmm。 / English — at the extreme, "that weight" is generated by another network on the fly, but the consumer interface stays `index then bmm`.

## 注意事项 / Caveats / when it breaks

- **参数量随本体线性增长** / **Parameter count grows linearly with embodiments**: 中文 — `K * in * out`。100 种本体 + 4096×4096 Linear 就是 1.6B 参数 —— 一层！所以通常只在 head/encoder 上用 `CategorySpecificLinear`，骨干仍然共享。 / English — `K * in * out`. With 100 embodiments and a 4096×4096 Linear that is 1.6 B parameters in one layer. You almost always limit this trick to encoders/heads and keep the trunk shared.
- **冷启动不稳** / **Cold start can be noisy**: 中文 — 没见过的本体（数据少）训出来的那一片 `W[k]` 几乎是随机的。生产里要么用低秩 `W[k] = W_shared + U[k] V[k].T`，要么把新本体做 LoRA。 / English — an embodiment with few episodes ends up with a near-random `W[k]`. Production systems either factorize `W[k] = W_shared + U[k] V[k].T` or apply a LoRA-style delta for new bodies.
- **动作维度还得另外解决** / **Action dim still varies separately**: 中文 — `CategorySpecificLinear` 只解决"权重不一样"，不解决"输出维度不一样"。后者用文件后面的 `expand_action_dimension` 处理，把已有 `W` 沿输出维拼接。 / English — this layer makes the *weights* per-embodiment but not the *shape*. The shape mismatch is handled later by `expand_action_dimension`, which tiles existing `W` along the output dim.

## 延伸阅读 / Further reading

- [Isaac-GR00T tech report — multi-embodiment foundation policy](https://github.com/NVIDIA/Isaac-GR00T)
- [Switch Transformer — the dispatching idea at LLM scale](https://arxiv.org/abs/2101.03961)
- [HyperNetworks (Ha et al., 2016)](https://arxiv.org/abs/1609.09106)
