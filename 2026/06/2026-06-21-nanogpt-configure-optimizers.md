---
date: 2026-06-21
topic: infrastructure
source: tracked
repo: karpathy/nanoGPT
file: model.py
permalink: https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/model.py#L263-L287
difficulty: intermediate
read_time: ~8 min
tags: [code-of-the-day, infrastructure, optimizer, weight-decay, fused-adamw]
---

# nanoGPT 的优化器配置：一行判断分出"要衰减"和"不衰减" / nanoGPT's Optimizer Setup: One-Line Rule to Split Decay vs. No-Decay

> **一句话 / In one line**: 只需 `p.dim() >= 2` 一个条件，就把所有参数分成两组——矩阵权重衰减、偏置和 LayerNorm 不衰减——再探测并启用融合 AdamW 内核。 / A single `p.dim() >= 2` condition splits all parameters into two groups — matrix weights decay, biases and LayerNorm params do not — then probes for and enables the fused AdamW CUDA kernel.

## 为什么重要 / Why this matters

Weight decay 在 GPT 训练中是双刃剑：对矩阵权重（embedding、注意力、FFN）施加正则化能防止过拟合；但对偏置、LayerNorm 的 scale/shift 参数施加衰减则是**错误的**——这些 1D 参数没有多余的自由度可以被惩罚。大多数教程忽略了这个细节，导致训练不稳定或最终质量下降。

nanoGPT 用一个极简判据解决了这个问题：**维度 ≥ 2 的张量衰减，维度 < 2 的不衰减**。所有矩阵（2D）和高维张量天然满足"要衰减"，所有偏置和 LayerNorm 参数天然满足"不衰减"，逻辑清晰、没有硬编码名字列表。

此外，这个函数还在运行时自动探测 PyTorch 是否支持 fused AdamW（`torch.optim.AdamW` 是否接受 `fused` 参数），在 CUDA 上自动升级到融合实现，把优化器步骤从多个内核调用压缩成一个，带来显著的训练速度提升。

Weight decay is double-edged in GPT training: regularizing matrix weights (embedding, attention, FFN) prevents overfitting, but applying decay to biases and LayerNorm scale/shift is **wrong** — those 1D parameters have no redundant degrees of freedom to penalize. Most tutorials gloss over this, causing subtle instability or quality loss.

nanoGPT solves it with one minimal rule: **tensors with dim ≥ 2 decay, everything else doesn't**. All matrices and higher-dimensional tensors satisfy "decay," all biases and LayerNorm parameters satisfy "no decay" — no hardcoded name lists needed. The function also runtime-detects fused AdamW availability and enables it on CUDA automatically, collapsing the optimizer step from many kernel launches into one.

## 代码 / The code

`karpathy/nanoGPT` — [`model.py`](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/model.py#L263-L287)

```python
def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in self.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    print(f"using fused AdamW: {use_fused}")

    return optimizer
```

## 逐行讲解 / What's happening

1. **第 263-267 行 / Lines 263-267 (参数收集 / collect parameters)**:
   - 中文: `named_parameters()` 返回全部参数，包括冻结的；再过滤一次 `requires_grad` 只保留需要训练的参数。
   - English: `named_parameters()` returns all parameters including frozen ones; the second filter keeps only trainable ones.

2. **第 268-270 行 / Lines 268-270 (`p.dim() >= 2` 分组)**:
   - 中文: 这是全文最核心的一行。`dim() >= 2` 意味着"矩阵或更高维张量"，自然囊括 embedding 表（2D）、线性层权重（2D）、注意力投影（2D）。`dim() < 2` 则是 1D 的偏置和 LayerNorm 参数。
   - English: This is the key line. `dim() >= 2` means "matrix or higher-dimensional tensor," naturally capturing embedding tables (2D), linear weights (2D), and attention projections (2D). `dim() < 2` catches 1D biases and LayerNorm params.

3. **第 271-274 行 / Lines 271-274 (两组 optim_groups)**:
   - 中文: PyTorch AdamW 原生支持按参数组设置不同超参数。这里仅覆盖 `weight_decay`；lr 和 betas 在第 284 行全局设置，两组共享。
   - English: PyTorch AdamW natively supports per-group hyperparameters. Only `weight_decay` differs here; `lr` and `betas` are set globally at line 284 and shared across both groups.

4. **第 279-281 行 / Lines 279-281 (`inspect.signature` 探测 fused)**:
   - 中文: 用 `inspect.signature` 检查 `torch.optim.AdamW` 是否接受 `fused` 参数，而不是硬编码 PyTorch 版本号。这是一种"能力探测"（capability probe）而非"版本探测"（version probe），更稳健。
   - English: Uses `inspect.signature` to check whether `torch.optim.AdamW` accepts a `fused` argument, rather than hardcoding a PyTorch version number. This is capability probing rather than version probing — more robust.

5. **第 282-284 行 / Lines 282-284 (条件启用 fused)**:
   - 中文: `fused=True` 只在 CUDA 上启用（CPU 不支持）。`**extra_args` 是一个优雅的技巧：如果 `use_fused` 为 False，`extra_args` 是空字典，等价于什么都不传。
   - English: `fused=True` is only enabled on CUDA (not supported on CPU). `**extra_args` is a clean trick: if `use_fused` is False, `extra_args` is an empty dict, equivalent to passing nothing.

## 类比 / The analogy

想象你在给一个机械钟上紧发条（weight decay 相当于给弹簧加阻力）。发条本身（矩阵权重）加阻力是合理的——过紧的弹簧会被这个阻力修正。但如果你对指针（偏置）也加同样的阻力，指针就会被错误地拉离正确位置。`p.dim() >= 2` 就像一条规则：只对"有弹性的零件"施加阻力，对"定位零件"不动。

Imagine tightening a mechanical clock's mainspring (weight decay is like adding resistance to the spring). Adding resistance to the mainspring itself (matrix weights) makes sense — an over-wound spring gets corrected. But adding that same resistance to the clock hands (biases) would wrongly pull them away from correct positions. `p.dim() >= 2` is the rule: apply resistance only to "elastic parts," leave "positioning parts" alone.

## 自己跑一遍 / Try it yourself

```python
import inspect
import torch
import torch.nn as nn

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 64)          # 2D → decay
        self.proj  = nn.Linear(64, 64)              # weight 2D → decay, bias 1D → no-decay
        self.norm  = nn.LayerNorm(64)               # weight+bias 1D → no-decay

    def configure_optimizers(self, weight_decay=0.1, lr=3e-4):
        param_dict = {n: p for n, p in self.named_parameters() if p.requires_grad}
        decay_params   = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        groups = [
            {'params': decay_params,   'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        fused_ok = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        extra = dict(fused=True) if (fused_ok and torch.cuda.is_available()) else {}
        return torch.optim.AdamW(groups, lr=lr, **extra)

model = TinyGPT()
opt = model.configure_optimizers()
for g in opt.param_groups:
    print(f"wd={g['weight_decay']}, params={[p.shape for p in g['params']]}")
```

运行 / Run with:
```bash
pip install torch
python try.py
```

预期输出 / Expected output:
```
wd=0.1, params=[torch.Size([100, 64]), torch.Size([64, 64])]
wd=0.0, params=[torch.Size([64]), torch.Size([64]), torch.Size([64])]
```

中文：注意第一组只有矩阵（embedding 和 Linear.weight），第二组是三个 1D 向量（Linear.bias、LayerNorm.weight、LayerNorm.bias）——`p.dim()` 判据完美地把它们分开了。

English: The first group contains only matrices (embedding and Linear.weight), the second group holds three 1D vectors (Linear.bias, LayerNorm.weight, LayerNorm.bias) — the `p.dim()` criterion separates them perfectly without any name-matching logic.

## 在别处也能看到这个模式 / Where this pattern shows up elsewhere

- **Llama / transformers `Trainer`** / **transformers Trainer**: `get_parameter_names()` 过滤名字里含 `"layernorm"` 或 `"bias"` 的参数来避免衰减 / filters param names containing `"layernorm"` or `"bias"` — name-based instead of shape-based, more brittle.
- **timm optimizer factories** / **timm**: 用 `no_weight_decay()` 方法让模型显式注册不衰减的参数名，需要每个模型自己维护白名单 / models explicitly register no-decay param names via `no_weight_decay()`, requiring each model to maintain its own whitelist.
- **nanoVLM** / **nanoVLM**: 直接复用了 nanoGPT 的这一函数，说明此模式已成为小型 GPT 系项目的事实标准 / directly reuses this exact function, showing it has become the de-facto standard in GPT-family repos.

## 注意事项 / Caveats / when it breaks

- **自定义高维参数** / **Custom high-dim params**: 若你有一个 3D 以上的参数但语义上类似偏置（例如某些位置编码），`dim() >= 2` 会错误地把它放进 decay 组。此时需要按名字额外过滤。 / If you have a 3D+ parameter that semantically acts like a bias (e.g. certain positional encodings), `dim() >= 2` wrongly puts it in the decay group. Requires extra name-based filtering.
- **fused AdamW 需要 CUDA contiguous 张量** / **fused AdamW needs contiguous tensors**: 梯度累积步骤中如果用了 `.grad = None` 手动清零，偶尔可能触发 fused 内核的连续性检查。 / Gradient accumulation steps using `.grad = None` can occasionally trigger contiguity checks in the fused kernel.
- **PyTorch < 2.0 没有 fused AdamW** / **PyTorch < 2.0 lacks fused AdamW**: `inspect.signature` 会探测到 `fused` 参数不存在，自动回退，不会崩溃。 / `inspect.signature` detects the missing `fused` arg and falls back gracefully — no crash.

## 延伸阅读 / Further reading

- [Decoupled Weight Decay Regularization (Loshchilov & Hutter, 2017)](https://arxiv.org/abs/1711.05101)
- [PyTorch AdamW fused kernel docs](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
- [nanoGPT model.py full source](https://github.com/karpathy/nanoGPT/blob/master/model.py)
