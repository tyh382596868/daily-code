---
date: 2026-08-21
topic: diffusion
source: tracked
repo: facebookresearch/DiT
file: models.py
permalink: https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L185-L216
difficulty: intermediate
read_time: ~10 min
tags: [code-of-the-day, diffusion, dit, initialization, adaln-zero]
---

# DiT 初始化：先让模型安静下来 / DiT Initialization: Start the Model Quietly

> **一句话 / In one line**: DiT 用 Xavier 初始化普通投影，用固定 sin/cos 位置表，然后把 adaLN 调制层和最终输出层置零，让条件分支从近似无影响开始学习。 / DiT uses Xavier for ordinary projections, fixed sin/cos position tables, then zeroes adaLN modulation and the final output head so conditioning starts from a near-silent state.

## 为什么重要 / Why this matters

扩散 Transformer 很深，条件分支一开始如果就强烈改写每个 block，训练容易不稳。DiT 的初始化把“能表达复杂条件”的结构搭好，但让调制和输出先从 0 开始，等梯度逐步学会该放大哪些通道、该输出哪些 residual。

A diffusion Transformer is deep. If conditioning strongly rewrites every block on step one, training can become unstable. DiT builds the conditional machinery up front, but starts modulation and output at zero so gradients can gradually learn which channels to open.

## 代码 / The code

`facebookresearch/DiT` — [`models.py`](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L185-L216)

```python
def _basic_init(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
self.apply(_basic_init)

# Initialize (and freeze) pos_embed by sin-cos embedding:
pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

# Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
w = self.x_embedder.proj.weight.data
nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
nn.init.constant_(self.x_embedder.proj.bias, 0)

# Initialize label embedding table:
nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

# Initialize timestep embedding MLP:
nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

# Zero-out adaLN modulation layers in DiT blocks:
for block in self.blocks:
    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

# Zero-out output layers:
nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
nn.init.constant_(self.final_layer.linear.weight, 0)
nn.init.constant_(self.final_layer.linear.bias, 0)
```

## 逐行讲解 / What's happening

1. **第 185-189 行 / Lines 185-189 (basic init)**:
   - 中文: 所有 `Linear` 用 Xavier，bias 清零，给普通 MLP/attention 投影一个稳定起点。
   - English: Every `Linear` gets Xavier weights and zero bias, giving ordinary projections a stable start.
2. **第 191-198 行 / Lines 191-198 (position and patch init)**:
   - 中文: 位置表直接拷贝固定 sin/cos；patch embedding 的卷积权重摊平成线性层再初始化。
   - English: The position table is copied from fixed sin/cos features; patch embedding conv weights are flattened and initialized like a linear layer.
3. **第 200-205 行 / Lines 200-205 (conditioning embeddings)**:
   - 中文: label embedding 和 timestep MLP 用小方差正态分布，让条件信号存在但不过猛。
   - English: Label and timestep embeddings use small-normal weights, so conditioning exists without dominating.
4. **第 207-216 行 / Lines 207-216 (zero gates and head)**:
   - 中文: adaLN 最后一层和输出 head 都清零，模型一开始不会凭随机条件分支乱改 latent。
   - English: The last adaLN layers and output head are zeroed, so random conditioning cannot immediately rewrite latents.

## 类比 / The analogy

像调音台开机。线路都已经接好，但每个推子先归零；工程师再慢慢推起真正需要的声道，而不是一上来让所有声道满音量。

It is like powering up a mixing console. The cables are connected, but every fader starts at zero; the engineer raises only the channels that matter.

## 自己跑一遍 / Try it yourself

```python
blocks = [
    {"hidden": 3.0, "gate": 0.0, "residual": 10.0},
    {"hidden": 3.0, "gate": 0.2, "residual": 10.0},
]

for block in blocks:
    out = block["hidden"] + block["gate"] * block["residual"]
    print(out)
```

运行 / Run with:
```bash
python try.py
```

预期输出 / Expected output:
```text
3.0
5.0
```

中文: `gate=0` 时 residual 分支已经存在，但不会改变主干；训练后 gate 变大才开始起作用。

English: With `gate=0`, the residual branch exists but does not affect the trunk; it matters only after training opens the gate.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **ResNet zero-gamma** / **ResNet zero-gamma**: residual block 的最后归一化缩放从 0 开始，让深层网络先近似恒等映射。 / The final normalization scale can start at zero so a deep residual net begins near identity.
- **ControlNet zero conv** / **ControlNet zero conv**: 控制分支接入主模型时先零初始化，避免一开始破坏预训练生成能力。 / Control branches often use zero-initialized projections so they do not disrupt a pretrained generator at startup.

## 注意事项 / Caveats / when it breaks

- **不是所有层都清零** / **Not every layer is zeroed**: 只有调制出口和最终输出被清零；普通投影仍要有可学习的随机基底。
- **恢复 checkpoint 时别重跑初始化** / **Do not reinitialize after loading**: 如果加载 checkpoint 后再次调用初始化，会抹掉已经学到的调制和输出。

## 延伸阅读 / Further reading

- [DiT `initialize_weights`](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L185-L216)
- [DiT repository](https://github.com/facebookresearch/DiT)
